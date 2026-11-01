"""学習済みの方策で Go2 を動かし、画面なしで mp4 に書き出す。

go2_eval.py はビューア（X11）を開くので、Wayland だけのデスクトップや SSH 越しでは動かない。
こちらはカメラで撮って videos/ に保存する。

    uv run python go2_record.py -e go2-walking --ckpt 100 --seconds 10
"""

import argparse
import math
import os
import pickle

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env

# カメラはロボットの右後ろから追いかける。ロボットの胴体からの相対位置。
CAM_OFFSET = (-1.5, -2.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out", type=str, default=None, help="既定は videos/<exp>_ckpt<N>.mp4")
    # 押し方と倒れる判定を、学習時の設定から変えて撮るための引数。
    # 押す処理を書く前に学習した結果（go2-walking）には push_* の設定がないので、ここで足す。
    parser.add_argument("--push-interval", type=float, default=None, help="押す間隔（秒）。未指定なら学習時の設定、それも無ければ押さない")
    parser.add_argument("--push-vel", type=float, default=None, help="押す速さの上限（m/s）。未指定なら学習時の設定")
    parser.add_argument("--roll-limit", type=float, default=None, help="roll がこの角度（度）を超えたらリセット。倒れる様子を撮るなら 180")
    parser.add_argument("--pitch-limit", type=float, default=None, help="pitch がこの角度（度）を超えたらリセット。倒れる様子を撮るなら 180")
    parser.add_argument("--log-euler", action="store_true", help="押す前後の roll・pitch・yaw と速度を表示する")
    # 起き上がりを見せるための引数。宙で立て直してしまうと、倒れている場面が映らないため。
    parser.add_argument("--start-roll", type=float, default=None, help="最初に前後の軸のまわりに転がす角度（度）。180 で仰向け")
    parser.add_argument("--hold", type=float, default=0.0, help="最初のこの秒数は方策を止め、脚を初期の角度のまま保つ")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}

    if args.push_interval is not None:
        env_cfg["push_interval_s"] = args.push_interval
    if args.push_vel is not None:
        env_cfg["max_push_vel"] = args.push_vel
    # 押さない録画でも step が push_* を読むので、無ければ「録画中は一度も押さない」値にしておく。
    env_cfg.setdefault("push_interval_s", 1e9)
    env_cfg.setdefault("max_push_vel", 0.0)
    if args.roll_limit is not None:
        env_cfg["termination_if_roll_greater_than"] = args.roll_limit
    if args.pitch_limit is not None:
        env_cfg["termination_if_pitch_greater_than"] = args.pitch_limit

    # Go2Env は __init__ の中で scene.build() まで済ませるので、build の直前にカメラを差し込む。
    cams = []
    original_build = gs.Scene.build

    def build_with_camera(scene, *a, **k):
        cams.append(scene.add_camera(res=(1280, 720), pos=(2.0, -2.0, 1.0), lookat=(0, 0, 0.3), fov=40, GUI=False))
        return original_build(scene, *a, **k)

    gs.Scene.build = build_with_camera
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )
    gs.Scene.build = original_build
    cam = cams[0]

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    # 押して撮った動画で、押さずに撮った動画を上書きしないよう、名前を分ける。
    pushed = env_cfg["max_push_vel"] > 0 and env_cfg["push_interval_s"] < 1e9
    out = args.out or f"videos/{args.exp_name}_ckpt{args.ckpt}{'_push' if pushed else ''}.mp4"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    obs_dict = env.reset()
    if args.start_roll is not None:
        half = math.radians(args.start_roll) / 2
        quat = torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]], dtype=gs.tc_float, device=gs.device)
        env.robot.set_quat(quat, zero_velocity=True)
    hold_steps = int(args.hold / env.dt)
    cam.start_recording(save_to_filename=out, fps=int(round(1 / env.dt)))
    if args.log_euler:
        print("  時刻 | roll  pitch  yaw（度） | 世界の速度 x  y | 胴体から見た速度 x  y（m/s）")
    interval = env_cfg["push_interval_s"]
    with torch.no_grad():
        for i in range(int(args.seconds / env.dt)):
            actions = policy(obs_dict)
            if i < hold_steps:
                # 行動 0 は「脚を初期の角度に保つ」指令。方策を止めて、転がった姿勢のまま寝かせる。
                actions = torch.zeros_like(actions)
            obs_dict, _, _, _ = env.step(actions)
            t = (i + 1) * env.dt
            # 押す時刻の少し前から1.5秒後まで、0.1秒ごとに姿勢と速度を表示する。
            since_push = t % interval
            if args.log_euler and (since_push < 1.5 or since_push > interval - 0.1) and (i + 1) % 5 == 0:
                r, p, y = env.base_euler[0].tolist()
                vw = env.robot.get_vel()[0].tolist()
                vb = env.base_lin_vel[0].tolist()
                print(f"{t:6.2f} | {r:6.1f} {p:6.1f} {y:6.1f} | {vw[0]:6.2f} {vw[1]:6.2f} | {vb[0]:6.2f} {vb[1]:6.2f}")
            base = env.base_pos[0].tolist()
            cam.set_pose(pos=[b + o for b, o in zip(base, CAM_OFFSET)], lookat=base)
            cam.render()
    cam.stop_recording()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
