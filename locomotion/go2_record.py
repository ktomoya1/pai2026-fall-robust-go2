"""学習済みの方策で Go2 を動かし、画面なしで mp4 に書き出す。

go2_eval.py はビューア（X11）を開くので、Wayland だけのデスクトップや SSH 越しでは動かない。
こちらはカメラで撮って videos/ に保存する。

    uv run python go2_record.py -e go2-walking --ckpt 100 --seconds 10
"""

import argparse
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
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}

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

    out = args.out or f"videos/{args.exp_name}_ckpt{args.ckpt}.mp4"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    obs_dict = env.reset()
    cam.start_recording(save_to_filename=out, fps=int(round(1 / env.dt)))
    with torch.no_grad():
        for _ in range(int(args.seconds / env.dt)):
            obs_dict, _, _, _ = env.step(policy(obs_dict))
            base = env.base_pos[0].tolist()
            cam.set_pose(pos=[b + o for b, o in zip(base, CAM_OFFSET)], lookat=base)
            cam.render()
    cam.stop_recording()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
