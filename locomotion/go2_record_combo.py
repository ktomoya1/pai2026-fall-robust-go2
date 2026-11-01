"""押されて踏ん張る方策と、起き上がる方策を、倒れたかどうかで切り替えて撮る。

筋書き: 立つ → 横から中くらいに押す（踏ん張る）→ 後ろから強く押す（倒れる）
→ 倒れたら起き上がり用の方策に切り替える → 立ったら元の方策に戻す。

    uv run python go2_record_combo.py --stand go2-stand-push-circle:999 --getup go2-getup:1499
"""

import argparse
import math
import os
import pickle

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env

CAM_OFFSET = (-1.8, -2.2, 1.1)
FALLEN_TILT_DEG = 45.0  # roll か pitch がこれを超えたら、起き上がり用に切り替える
UPRIGHT_GRAVITY_Z = -0.95  # 胴体から見た重力の z がこれより下（ほぼ真下）なら、立っているとみなす
UPRIGHT_HOLD_S = 0.5  # 立った状態がこの時間続いたら、元の方策に戻す


def load(spec):
    exp, ckpt = spec.split(":")
    with open(f"logs/{exp}/cfgs.pkl", "rb") as f:
        cfgs = pickle.load(f)
    return exp, int(ckpt), cfgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stand", default="go2-stand-push-circle:999", help="押されて踏ん張る方策（実験名:回数）")
    parser.add_argument("--getup", default="go2-getup:1499", help="起き上がる方策（実験名:回数）")
    parser.add_argument("--pushes", default="3:90:2.0,7:0:4.0", help="押す時刻（秒）:向き（度、0=前へ 90=左へ）:速さ（m/s）をカンマ区切り")
    parser.add_argument("--seconds", type=float, default=14.0)
    parser.add_argument("--out", default="videos/combo.mp4")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")

    stand_exp, stand_ckpt, (env_cfg, obs_cfg, reward_cfg, command_cfg, stand_train_cfg) = load(args.stand)
    getup_exp, getup_ckpt, (_, _, _, _, getup_train_cfg) = load(args.getup)
    reward_cfg["reward_scales"] = {}
    # 環境の定期的な押しは止める（押すのはこのスクリプト）。最初は立った姿勢、倒れてもリセットしない。
    env_cfg["push_interval_s"] = 1e9
    env_cfg["max_push_vel"] = 0.0
    env_cfg["getup"] = False
    env_cfg["termination_if_roll_greater_than"] = 180
    env_cfg["termination_if_pitch_greater_than"] = 180
    env_cfg["episode_length_s"] = max(env_cfg["episode_length_s"], args.seconds + 1)

    cams = []
    original_build = gs.Scene.build

    def build_with_camera(scene, *a, **k):
        cams.append(scene.add_camera(res=(1280, 720), pos=(2.0, -2.0, 1.0), lookat=(0, 0, 0.3), fov=40, GUI=False))
        return original_build(scene, *a, **k)

    gs.Scene.build = build_with_camera
    env = Go2Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)
    gs.Scene.build = original_build
    cam = cams[0]

    def policy_of(exp, ckpt, train_cfg):
        runner = OnPolicyRunner(env, train_cfg, f"logs/{exp}", device=gs.device)
        runner.load(os.path.join(f"logs/{exp}", f"model_{ckpt}.pt"))
        return runner.get_inference_policy(device=gs.device)

    stand_policy = policy_of(stand_exp, stand_ckpt, stand_train_cfg)
    getup_policy = policy_of(getup_exp, getup_ckpt, getup_train_cfg)

    pushes = []
    for item in args.pushes.split(","):
        t, deg, vel = (float(x) for x in item.split(":"))
        pushes.append((int(round(t / env.dt)), math.radians(deg), vel))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    obs_dict = env.reset()
    mode = "stand"
    upright_steps = 0
    cam.start_recording(save_to_filename=args.out, fps=int(round(1 / env.dt)))
    with torch.no_grad():
        for i in range(int(args.seconds / env.dt)):
            for step, angle, vel in pushes:
                if i == step:
                    # 犬から見た向きで押す（立っている間の yaw のずれを打ち消す）
                    heading = angle + math.radians(env.base_euler[0, 2].item())
                    push = torch.tensor([[vel * math.cos(heading), vel * math.sin(heading)]], dtype=gs.tc_float, device=gs.device)
                    env.robot.set_dofs_velocity(push, dofs_idx_local=(0, 1))
                    print(f"{i * env.dt:5.2f}s 押す: 向き {math.degrees(angle):.0f}度, {vel} m/s")

            policy = stand_policy if mode == "stand" else getup_policy
            obs_dict, _, _, _ = env.step(policy(obs_dict))

            tilt = max(abs(env.base_euler[0, 0].item()), abs(env.base_euler[0, 1].item()))
            if mode == "stand" and tilt > FALLEN_TILT_DEG:
                mode = "getup"
                upright_steps = 0
                print(f"{(i + 1) * env.dt:5.2f}s 倒れた（傾き {tilt:.0f}度）→ 起き上がり用に切り替え")
            elif mode == "getup":
                upright_steps = upright_steps + 1 if env.projected_gravity[0, 2].item() < UPRIGHT_GRAVITY_Z else 0
                if upright_steps * env.dt >= UPRIGHT_HOLD_S:
                    mode = "stand"
                    print(f"{(i + 1) * env.dt:5.2f}s 立ち上がった → 元の方策に戻す")

            base = env.base_pos[0].tolist()
            cam.set_pose(pos=[b + o for b, o in zip(base, CAM_OFFSET)], lookat=base)
            cam.render()
    cam.stop_recording()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
