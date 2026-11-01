"""押す向きと強さごとに、学習済みの犬が倒れた割合を数える。

犬を「向き × 強さ × reps 匹」並べ、2秒立たせてから全員を一斉に1回だけ押し、
その後3秒のあいだに倒れたかを数える。押す向きは犬から見た向き（yaw のずれを打ち消す）。

    uv run python go2_push_eval.py -e go2-stand-push-xy --ckpt 999
"""

import argparse
import math
import os
import pickle

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env

# 犬から見て、どちらへ押し出すか（度）。0 は前へ（＝後ろから押す）、90 は左へ（＝右から押す）。
DIRS = {
    "前へ": 0,
    "左前へ": 45,
    "左へ": 90,
    "左後ろへ": 135,
    "後ろへ": 180,
    "右後ろへ": 225,
    "右へ": 270,
    "右前へ": 315,
}
SETTLE_S = 2.0  # 押す前に立たせておく時間
AFTER_S = 3.0  # 押したあと、倒れたかを見る時間
FALL_TILT_DEG = 45.0  # roll か pitch がこれを超えたら倒れたとみなす
FALL_HEIGHT_M = 0.15  # 胴体の中心がこれより低くなったら倒れたとみなす


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--vels", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0], help="押す速さ（m/s）")
    parser.add_argument("--reps", type=int, default=25, help="向きと強さの組ごとの犬の数")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    # 押すのはこのスクリプトが1回だけ。環境の定期的な押しは止め、倒れてもリセットさせない。
    env_cfg["push_interval_s"] = 1e9
    env_cfg["max_push_vel"] = 0.0
    env_cfg["termination_if_roll_greater_than"] = 180
    env_cfg["termination_if_pitch_greater_than"] = 180
    env_cfg["episode_length_s"] = 100.0
    # 起き上がりの練習をした方策でも、評価は立った姿勢から始める。
    env_cfg["getup"] = False

    combos = [(d, v) for d in DIRS for v in args.vels]
    n = len(combos) * args.reps
    env = Go2Env(
        num_envs=n,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    # i 番目の犬を押す向き（ラジアン）と速さ
    angle = torch.tensor([DIRS[d] for d, _ in combos for _ in range(args.reps)], dtype=gs.tc_float, device=gs.device)
    angle = angle * math.pi / 180
    speed = torch.tensor([v for _, v in combos for _ in range(args.reps)], dtype=gs.tc_float, device=gs.device)

    obs_dict = env.reset()
    fallen = torch.zeros(n, dtype=torch.bool, device=gs.device)
    with torch.no_grad():
        for _ in range(int(SETTLE_S / env.dt)):
            obs_dict, _, _, _ = env.step(policy(obs_dict))

        # 犬から見た向きを、世界の向きに直す。立っている間に yaw が少しずれているため。
        heading = angle + env.base_euler[:, 2] * math.pi / 180
        push_vel = torch.stack([speed * torch.cos(heading), speed * torch.sin(heading)], dim=1)
        env.robot.set_dofs_velocity(push_vel, dofs_idx_local=(0, 1))

        for _ in range(int(AFTER_S / env.dt)):
            obs_dict, _, _, _ = env.step(policy(obs_dict))
            tilt = torch.maximum(env.base_euler[:, 0].abs(), env.base_euler[:, 1].abs())
            fallen |= (tilt > FALL_TILT_DEG) | (env.base_pos[:, 2] < FALL_HEIGHT_M)

    # 押してから AFTER_S 秒後に、胴体がほぼ真上を向いて立っているか（倒れても起き上がれたかを含む）
    standing = (env.projected_gravity[:, 2] < -0.9) & (env.base_pos[:, 2] > FALL_HEIGHT_M)

    def table(title, flags):
        rate = flags.float().view(len(DIRS), len(args.vels), args.reps).mean(dim=2) * 100
        print(f"\n{args.exp_name} / model_{args.ckpt}.pt: {title}（%、各 {args.reps} 匹）")
        print("| 押す向き | " + " | ".join(f"{v:.1f} m/s" for v in args.vels) + " |")
        print("|---|" + "---|" * len(args.vels))
        for i, d in enumerate(DIRS):
            print(f"| {d} | " + " | ".join(f"{r:.0f}" for r in rate[i].tolist()) + " |")
        print("| **平均** | " + " | ".join(f"{r:.0f}" for r in rate.mean(dim=0).tolist()) + " |")

    table("倒れた割合", fallen)
    table(f"{AFTER_S:.0f}秒後に立っていた割合", standing)


if __name__ == "__main__":
    main()
