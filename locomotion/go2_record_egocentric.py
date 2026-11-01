"""犬の頭のカメラの画像だけで、VLM に旗を探させて回る（天井カメラも、旗の座標も使わない）。

1秒ごとに頭のカメラで前を撮り、VLM に目的の色の旗の枠を聞く。
- 見えたら: 枠の中心が画像の真ん中からどれだけ左右にずれているかを角度に直し、その方向を目標の向きにする
- 見えなければ: 左に回って探す
- 旗の根元が十分近く（床までの角度から見積もった距離が REACH_M 以下）に写ったら、到着として次の旗へ

    uv run python go2_record_egocentric.py -e go2-run2 --ckpt 1999 --flags-seed 0 --route 3,4,0
"""

import argparse
import math
import os
import pickle

import numpy as np
import torch
from PIL import Image
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env
from vlm_locate import COLORS, FLAG_RADIUS, FLAG_TOP, START_HEIGHT, Locator, random_flags, refine

HEAD_W, HEAD_H = 1176, 672  # 頭のカメラの解像度（28 の倍数）。遠くの小さな旗も写るよう高めにする
HEAD_FOV = 60.0  # 頭のカメラの縦の画角（度）
HEAD_FORWARD, HEAD_UP = 0.30, 0.15  # 胴体の中心から見たカメラの位置（前に 0.30m、上に 0.15m）
HEAD_PITCH_DEG = 10.0  # カメラを少し下に向ける（度）
QUERY_S = 1.0  # VLM に聞く間隔（秒）
REACH_M = 0.8  # 旗の根元までの見積もり距離がこれ以下になったら、到着
TURN_DEG = 30.0  # 旗の方向とのずれがこれより大きい間は、ゆっくり歩きながら向きを変える
TURN_SPEED = 0.3
SEARCH_DEG = 60.0  # 旗が見えないときは、今の向きから左にこれだけ回った方向を目標にする


def head_pose(base, yaw):
    """胴体の位置と向き（yaw、ラジアン）から、頭のカメラの位置と注視点を作る。傾き（roll・pitch）は無視して水平に保つ。"""
    c, s = math.cos(yaw), math.sin(yaw)
    pos = (base[0] + HEAD_FORWARD * c, base[1] + HEAD_FORWARD * s, base[2] + HEAD_UP)
    down = math.tan(math.radians(HEAD_PITCH_DEG))
    lookat = (pos[0] + c, pos[1] + s, pos[2] - down)
    return pos, lookat


def bearing_of(u):
    """画像の横の位置 u を、カメラの正面からの左右の角度（ラジアン、左が正）に直す。"""
    half_h = math.tan(math.radians(HEAD_FOV) / 2)
    half_w = half_h * HEAD_W / HEAD_H
    return math.atan(-(u - HEAD_W / 2) / (HEAD_W / 2) * half_w)


def ground_distance(v, cam_height):
    """旗の根元の画像の縦の位置 v から、床までの距離を見積もる（床が平らだと仮定）。"""
    half_h = math.tan(math.radians(HEAD_FOV) / 2)
    below = math.radians(HEAD_PITCH_DEG) + math.atan((v - HEAD_H / 2) / (HEAD_H / 2) * half_h)
    if below <= 0:
        return float("inf")
    return cam_height / math.tan(below)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", default="go2-run2")
    parser.add_argument("--ckpt", type=int, default=1999)
    parser.add_argument("--flags-seed", type=int, default=0)
    parser.add_argument("--route", default="3,4,0", help="回る旗の番号の順番（llm_route.py の --route）")
    parser.add_argument("--pushes", default="", help="押す時刻（秒）:向き（度）:速さ（m/s）。空なら押さない")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out", default="videos/egocentric.mp4")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    env_cfg.update(push_interval_s=1e9, max_push_vel=0.0, getup=False, heading_command=True)
    env_cfg["termination_if_roll_greater_than"] = 180
    env_cfg["termination_if_pitch_greater_than"] = 180
    env_cfg["episode_length_s"] = args.seconds + 1

    flags = random_flags(args.flags_seed)
    route = [int(r) for r in args.route.split(",")]

    cams = []
    original_build = gs.Scene.build

    def build_with_extras(scene, *a, **k):
        for i, (fx, fy) in flags.items():
            h = START_HEIGHT if i == 0 else FLAG_TOP
            scene.add_entity(
                gs.morphs.Cylinder(pos=(fx, fy, h / 2), radius=FLAG_RADIUS, height=h, fixed=True, collision=False),
                surface=gs.surfaces.Default(color=COLORS[i][1]),
            )
        cams.append(scene.add_camera(res=(HEAD_W, HEAD_H), pos=(0.3, 0, 0.45), lookat=(1.3, 0, 0.3), fov=HEAD_FOV, GUI=False))
        cams.append(scene.add_camera(res=(1280, 720), pos=(-5.0, -6.0, 6.0), lookat=(0.0, 0.0, 0.0), fov=50, GUI=False))
        return original_build(scene, *a, **k)

    gs.Scene.build = build_with_extras
    env = Go2Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)
    gs.Scene.build = original_build
    head, cam = cams

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)
    locator = Locator()

    pushes = []
    for item in filter(None, args.pushes.split(",")):
        t, deg, vel = (float(x) for x in item.split(":"))
        pushes.append((int(round(t / env.dt)), math.radians(deg), vel))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frames_dir = os.path.splitext(args.out)[0] + "_head"
    os.makedirs(frames_dir, exist_ok=True)
    env._resample_commands = lambda envs_idx: None
    obs_dict = env.reset()
    env.commands[:] = 0.0

    goal_i, finished_at = 0, None
    query_every = int(QUERY_S / env.dt)
    target, seen = 0.0, False
    cam.start_recording(save_to_filename=args.out, fps=int(round(1 / env.dt)))
    with torch.no_grad():
        for i in range(int(args.seconds / env.dt)):
            base = env.base_pos[0].tolist()
            yaw = math.radians(env.base_euler[0, 2].item())
            pos, lookat = head_pose(base, yaw)
            head.set_pose(pos=pos, lookat=lookat, up=(0.0, 0.0, 1.0))  # up を毎回渡さないと、向きを変えるたびに傾きがたまる

            if finished_at is None and i % query_every == 0:
                idx = route[goal_i]
                name = COLORS[idx][0]
                image = Image.fromarray(np.asarray(head.render()[0]))
                box, answer = locator.ask_box(image, name)
                detail = refine(image, box, name, with_bottom=True) if box else None
                gx, gy = flags[idx]
                true_dist = math.hypot(gx - base[0], gy - base[1])
                if detail is None:
                    seen = False
                    target = yaw + math.radians(SEARCH_DEG)
                    print(f"{i * env.dt:5.1f}s {name:6s} 見えない → 左に回って探す（本当の距離 {true_dist:.1f}m）")
                else:
                    seen = True
                    u, _, v_bottom, npix = detail
                    b = bearing_of(u)
                    dist = ground_distance(v_bottom, pos[2])
                    target = yaw + b
                    print(f"{i * env.dt:5.1f}s {name:6s} 見えた: 左右 {math.degrees(b):+5.0f}度, 見積もり距離 {dist:.1f}m（本当 {true_dist:.1f}m）")
                    image.save(os.path.join(frames_dir, f"{i * env.dt:05.1f}s_{name}.png"))
                    if dist < REACH_M:
                        print(f"{i * env.dt:5.1f}s {name} の旗に到着 ({goal_i + 1}/{len(route)})")
                        if goal_i == len(route) - 1:
                            finished_at = i
                        else:
                            goal_i += 1
                            seen = False

            env.target_yaw[:] = target
            err = (target - yaw + math.pi) % (2 * math.pi) - math.pi
            if finished_at is not None:
                env.commands[:, 0] = 0.0
                if i - finished_at > int(2.0 / env.dt):
                    break
            else:
                env.commands[:, 0] = args.speed if (seen and abs(math.degrees(err)) <= TURN_DEG) else TURN_SPEED

            for step, angle, vel in pushes:
                if i == step:
                    heading = angle + yaw
                    push = torch.tensor([[vel * math.cos(heading), vel * math.sin(heading)]], dtype=gs.tc_float, device=gs.device)
                    env.robot.set_dofs_velocity(push, dofs_idx_local=(0, 1))
                    print(f"{i * env.dt:5.2f}s 押す: 向き {math.degrees(angle):.0f}度, {vel} m/s")

            obs_dict, _, _, _ = env.step(policy(obs_dict))
            cam.render()
    cam.stop_recording()
    print(f"saved {args.out}（頭のカメラの画像は {frames_dir}/）")


if __name__ == "__main__":
    main()
