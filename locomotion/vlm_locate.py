"""真上から見下ろした画像から、VLM（Qwen2.5-VL-3B）で色つきの旗の位置を見つける。

VLM は画像の中の旗を枠（ピクセルの座標）で答える。カメラは真上から真下を向いているので、
枠の中心のピクセルを、比例の計算で床の座標 (x, y) に直せる。

単体で試す（旗をランダムに置き、本当の位置と VLM の答えを比べる）:
    uv run python vlm_locate.py --seed 0
"""

import argparse
import json
import math
import random
import re

import numpy as np
import torch
from PIL import Image

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
# 旗の色。llm_route.py の旗の番号と同じ並び: 0 白（スタート、中央）、1 赤（前）、2 青（左）、3 緑（後ろ）、4 黄（右）
COLORS = {
    0: ("white", (0.95, 0.95, 0.95)),
    1: ("red", (0.9, 0.15, 0.15)),
    2: ("blue", (0.15, 0.35, 0.95)),
    3: ("green", (0.15, 0.75, 0.25)),
    4: ("yellow", (0.95, 0.85, 0.1)),
}
# 色つきの旗を置く向き（度、0 が前 +x、90 が左 +y）
DIRECTIONS = {1: 0.0, 2: 90.0, 3: 180.0, 4: 270.0}
RES = 672  # 28 の倍数にしておくと、Qwen2.5-VL が画像を縮めず、答えの座標がそのままピクセルになる
CAM_HEIGHT = 9.0  # カメラの高さ（m）。低いほど旗が大きく写るが、写る範囲は狭くなる
FOV = 50.0  # 縦の画角（度）。高さ 9m で床の ±3.9m が写る
FLAG_TOP = 0.6  # 旗の上面の高さ（m）。真上からは上面が見えるので、この高さで視差を直す
FLAG_RADIUS = 0.4  # 旗の半径（m）。小さいと VLM が位置を外しやすい
START_HEIGHT = 0.02  # スタート（白）は床に敷いた薄い円盤。犬が中に埋もれて見えなくならないように


def overhead_camera(scene, center):
    """真上から真下を向くカメラ。画像の上が +y、右が +x になるよう up を +y にする。"""
    cx, cy = center
    return scene.add_camera(res=(RES, RES), pos=(cx, cy, CAM_HEIGHT), lookat=(cx, cy, 0.0), up=(0.0, 1.0, 0.0), fov=FOV, GUI=False)


def pixel_to_floor(u, v, center, z=FLAG_TOP):
    """画像のピクセル (u, v) を、高さ z の面の上の床の座標 (x, y) に直す。"""
    cx, cy = center
    # 画像の中心から端までが、高さ (CAM_HEIGHT - z) の所で何 m にあたるか
    half = (CAM_HEIGHT - z) * math.tan(math.radians(FOV) / 2)
    x = cx + (u - RES / 2) / (RES / 2) * half
    y = cy - (v - RES / 2) / (RES / 2) * half
    return x, y


# 枠の中で旗の色の画素を拾う条件（RGB、0〜255）
COLOR_MASKS = {
    "white": lambda r, g, b: (r > 215) & (g > 215) & (b > 215),
    "red": lambda r, g, b: (r > 170) & (g < 100) & (b < 100),
    "blue": lambda r, g, b: (b > 170) & (r < 100) & (g < 140),
    "green": lambda r, g, b: (g > 150) & (r < 100) & (b < 130),
    "yellow": lambda r, g, b: (r > 190) & (g > 170) & (b < 90),
}


def refine(image, box, color_name, margin=20):
    """VLM の枠（少し広げる）の中で、その色の画素の中心を取り直す。画素が無ければ None。"""
    rgb = np.asarray(image).astype(int)
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(int(x1) - margin, 0), max(int(y1) - margin, 0)
    x2, y2 = min(int(x2) + margin, w), min(int(y2) + margin, h)
    crop = rgb[y1:y2, x1:x2]
    mask = COLOR_MASKS[color_name](crop[..., 0], crop[..., 1], crop[..., 2])
    vs, us = np.nonzero(mask)
    if len(us) < 20:
        return None
    return x1 + us.mean(), y1 + vs.mean()


class Locator:
    def __init__(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(MODEL)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")

    def find(self, image, color_name):
        """画像から color_name の旗を探し、枠の中心のピクセル (u, v) を返す。見つからなければ None。"""
        prompt = (
            f"This is a top-down view of a floor with several colored cylinders. "
            f"Locate the {color_name} cylinder. Output its bbox coordinates in JSON format "
            f'like [{{"bbox_2d": [x1, y1, x2, y2], "label": "{color_name} cylinder"}}].'
        )
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        answer = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        m = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", answer)
        if not m:
            return None, answer
        x1, y1, x2, y2 = (float(g) for g in m.groups())
        # VLM が「どの色の旗がどのあたりか」を決め、正確な位置は枠の中の色の画素で取り直す
        refined = refine(image, (x1, y1, x2, y2), color_name)
        if refined is not None:
            return refined, answer
        return ((x1 + x2) / 2, (y1 + y2) / 2), answer


def random_flags(seed, radius=(2.5, 3.5), jitter_deg=20.0):
    """白（スタート）は原点。色つきの旗は前・左・後ろ・右の四方に1本ずつ、距離と向きを少しずらして置く。"""
    rng = random.Random(seed)
    pos = {0: (0.0, 0.0)}
    for i, deg in DIRECTIONS.items():
        a = math.radians(deg + rng.uniform(-jitter_deg, jitter_deg))
        r = rng.uniform(*radius)
        pos[i] = (r * math.cos(a), r * math.sin(a))
    return pos


def main():
    import genesis as gs

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="videos/vlm_overhead.png")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")
    flags = random_flags(args.seed)
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    for i, (fx, fy) in flags.items():
        h = START_HEIGHT if i == 0 else FLAG_TOP
        scene.add_entity(
            gs.morphs.Cylinder(pos=(fx, fy, h / 2), radius=FLAG_RADIUS, height=h, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=COLORS[i][1]),
        )
    cam = overhead_camera(scene, (0.0, 0.0))
    scene.build()
    rgb = cam.render()[0]
    image = Image.fromarray(np.asarray(rgb))
    image.save(args.save)
    print(f"saved {args.save}")

    locator = Locator()
    for i, (name, _) in COLORS.items():
        uv, answer = locator.find(image, name)
        tx, ty = flags[i]
        if uv is None:
            print(f"{name:5s}: 見つからない（答え: {answer}）")
            continue
        x, y = pixel_to_floor(*uv, (0.0, 0.0), z=START_HEIGHT if i == 0 else FLAG_TOP)
        print(f"{name:5s}: 本当 ({tx:+.2f}, {ty:+.2f})  VLM ({x:+.2f}, {y:+.2f})  ずれ {math.dist((x, y), (tx, ty)):.2f} m")


if __name__ == "__main__":
    main()
