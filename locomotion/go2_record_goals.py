"""旗を順に回らせ、途中で押して撮る。

目標の向き（env.target_yaw）を毎ステップ「次の旗の方向」に差し替えるだけで、方策はそのまま使う。
旗に近づいたら次の旗へ。最後の旗の次は最初の旗に戻る。

    uv run python go2_record_goals.py -e go2-run --ckpt 1999
"""

import argparse
import math
import os
import pickle

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env
from vlm_locate import COLORS, FLAG_RADIUS, FLAG_TOP, START_HEIGHT, Locator, overhead_camera, pixel_to_floor, random_flags

import numpy as np
from PIL import Image

REACH_M = 0.4  # 旗にこの距離まで近づいたら、次の旗へ
# 床に立てる旗（位置と色）。llm_route.py の旗の番号と同じ並び: 0 白（スタート、中央）、1 赤（前）、2 青（左）、3 緑（後ろ）、4 黄（右）
FLAGS = [((0.0, 0.0), (0.95, 0.95, 0.95)), ((3.0, 0.0), (0.9, 0.15, 0.15)), ((0.0, 3.0), (0.15, 0.35, 0.95)), ((-3.0, 0.0), (0.15, 0.75, 0.25)), ((0.0, -3.0), (0.95, 0.85, 0.1))]
TURN_IN_PLACE_DEG = 30.0  # 旗の方向とのずれがこれより大きい間は、ゆっくり歩きながら向きを変える
# 完全に止める（0）と、方策はその場で大きく回れずに固まった（止まったまま回る練習をほとんどしていないため）
TURN_SPEED = 0.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", default="go2-run")
    parser.add_argument("--ckpt", type=int, default=1999)
    parser.add_argument("--goals", default="3:0,3:3,0:3,0:0", help="旗の位置 x:y（m）をカンマ区切り")
    parser.add_argument("--pushes", default="6:90:3.0,15:0:6.0", help="押す時刻（秒）:向き（度、犬から見て。0=前へ 90=左へ）:速さ（m/s）")
    parser.add_argument("--speed", type=float, default=1.0, help="前に進む速さの命令（m/s）")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", default="videos/goals.mp4")
    # VLM 版: 旗をランダムな位置に置き、回る順番（旗の番号）だけを受け取って、位置は VLM に画像から見つけさせる
    parser.add_argument("--flags-seed", type=int, default=None, help="旗をランダムな位置に置く（乱数の種）。未指定なら既定の位置")
    parser.add_argument("--route", default=None, help="回る旗の番号の順番（例 3,2,0）。指定すると --goals の代わりに使う")
    parser.add_argument("--vlm", action="store_true", help="--route の旗の位置を、真上の画像から VLM で見つける")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    env_cfg["push_interval_s"] = 1e9
    env_cfg["max_push_vel"] = 0.0
    env_cfg["getup"] = False
    env_cfg["heading_command"] = True
    env_cfg["termination_if_roll_greater_than"] = 180
    env_cfg["termination_if_pitch_greater_than"] = 180
    env_cfg["episode_length_s"] = args.seconds + 1
    command_cfg["lin_vel_x_range"] = [args.speed, args.speed]

    flags = FLAGS
    if args.flags_seed is not None:
        positions = random_flags(args.flags_seed)
        flags = [(positions[i], FLAGS[i][1]) for i in range(len(FLAGS))]
    goals = [tuple(float(v) for v in g.split(":")) for g in args.goals.split(",")]
    cx = sum(p[0] for p, _ in flags) / len(flags)
    cy = sum(p[1] for p, _ in flags) / len(flags)

    # 旗（当たり判定なしの円柱）とカメラは、Go2Env の build の直前に差し込む
    cams = []
    original_build = gs.Scene.build

    def build_with_extras(scene, *a, **k):
        for i, ((fx, fy), color) in enumerate(flags):
            h = START_HEIGHT if i == 0 else FLAG_TOP
            scene.add_entity(
                gs.morphs.Cylinder(pos=(fx, fy, h / 2), radius=FLAG_RADIUS, height=h, fixed=True, collision=False),
                surface=gs.surfaces.Default(color=color),
            )
        cams.append(overhead_camera(scene, (0.0, 0.0)))
        cams.append(
            scene.add_camera(res=(1280, 720), pos=(cx - 4.0, cy - 5.0, 5.0), lookat=(cx, cy, 0.0), fov=50, GUI=False)
        )
        return original_build(scene, *a, **k)

    gs.Scene.build = build_with_extras
    env = Go2Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)
    gs.Scene.build = original_build
    overhead, cam = cams
    if not hasattr(env, "target_yaw"):
        raise SystemExit("go2_env.py に self.target_yaw がありません（向きの命令の目標を差し替えられない）")

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    pushes = []
    for item in args.pushes.split(","):
        t, deg, vel = (float(x) for x in item.split(":"))
        pushes.append((int(round(t / env.dt)), math.radians(deg), vel))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # 命令はこのスクリプトが毎ステップ決める。4秒ごとの選び直しで上書きされないよう止める。
    env._resample_commands = lambda envs_idx: None
    obs_dict = env.reset()
    env.commands[:] = 0.0

    if args.route:
        route = [int(r) for r in args.route.split(",")]
        if args.vlm:
            image = Image.fromarray(np.asarray(overhead.render()[0]))
            image.save(os.path.splitext(args.out)[0] + "_overhead.png")
            locator = Locator()
            goals = []
            for idx in route:
                name = COLORS[idx][0]
                uv, answer = locator.find(image, name)
                if uv is None:
                    raise SystemExit(f"VLM が {name} の旗を見つけられなかった: {answer}")
                gx, gy = pixel_to_floor(*uv, (0.0, 0.0), z=START_HEIGHT if idx == 0 else FLAG_TOP)
                tx, ty = flags[idx][0]
                print(f"VLM: {name:5s} の旗 → ({gx:+.2f}, {gy:+.2f})  本当 ({tx:+.2f}, {ty:+.2f})  ずれ {math.dist((gx, gy), (tx, ty)):.2f} m")
                goals.append((gx, gy))
            del locator
            torch.cuda.empty_cache()
        else:
            goals = [flags[idx][0] for idx in route]
    goal_i = 0
    finished_at = None
    cam.start_recording(save_to_filename=args.out, fps=int(round(1 / env.dt)))
    with torch.no_grad():
        for i in range(int(args.seconds / env.dt)):
            x, y = env.base_pos[0, 0].item(), env.base_pos[0, 1].item()
            gx, gy = goals[goal_i]
            if finished_at is None and math.hypot(gx - x, gy - y) < REACH_M:
                print(f"{i * env.dt:5.2f}s 目的地 {goal_i + 1}/{len(goals)} に到着 ({gx}, {gy})")
                if goal_i == len(goals) - 1:
                    finished_at = i
                goal_i = min(goal_i + 1, len(goals) - 1)
                gx, gy = goals[goal_i]
            target = math.atan2(gy - y, gx - x)
            env.target_yaw[:] = target
            # 旗の方向とのずれが大きい間は、ゆっくり歩きながら向きを変える
            err = (target - math.radians(env.base_euler[0, 2].item()) + math.pi) % (2 * math.pi) - math.pi
            env.commands[:, 0] = TURN_SPEED if abs(math.degrees(err)) > TURN_IN_PLACE_DEG else args.speed
            if i % int(0.5 / env.dt) == 0:
                tilt = max(abs(env.base_euler[0, 0].item()), abs(env.base_euler[0, 1].item()))
                print(
                    f"{i * env.dt:5.1f}s 位置 ({x:+.2f},{y:+.2f}) 目的地 {goal_i + 1}: ({gx:+.2f},{gy:+.2f}) 距離 {math.hypot(gx - x, gy - y):.2f}m "
                    f"向きのずれ {math.degrees(err):+5.0f}度 前進の命令 {env.commands[0, 0].item():.1f} 回転の命令 {env.commands[0, 2].item():+.2f} 傾き {tilt:.0f}度"
                )

            for step, angle, vel in pushes:
                if i == step:
                    heading = angle + math.radians(env.base_euler[0, 2].item())
                    push = torch.tensor([[vel * math.cos(heading), vel * math.sin(heading)]], dtype=gs.tc_float, device=gs.device)
                    env.robot.set_dofs_velocity(push, dofs_idx_local=(0, 1))
                    print(f"{i * env.dt:5.2f}s 押す: 向き {math.degrees(angle):.0f}度, {vel} m/s")

            if finished_at is not None:
                # 最後の目的地に着いたら、その場で止まる（前に進む命令を 0 にする）
                env.commands[:, 0] = 0.0
                if i - finished_at > int(2.0 / env.dt):
                    break
            obs_dict, _, _, _ = env.step(policy(obs_dict))
            cam.render()
    cam.stop_recording()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
