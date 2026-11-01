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


# 枠の中で旗の色の画素を拾う条件。明るさ（影）に左右されないよう、色相（H、0〜255）と彩度（S）で判定する
# PIL の HSV: 赤 ≈ 0 / 255、黄 ≈ 40、緑 ≈ 90、青 ≈ 160
COLOR_MASKS = {
    "white": lambda h, s, v: (s < 40) & (v > 170),
    "red": lambda h, s, v: ((h < 12) | (h > 240)) & (s > 110) & (v > 50),
    "yellow": lambda h, s, v: (h > 25) & (h < 55) & (s > 110) & (v > 60),
    "green": lambda h, s, v: (h > 70) & (h < 110) & (s > 100) & (v > 40),
    "blue": lambda h, s, v: (h > 145) & (h < 180) & (s > 100) & (v > 110),  # 床と空も青っぽい（明るさ 90 以下）ので、明るい画素だけ
}


def refine(image, box, color_name, margin=20, with_bottom=False):
    """VLM の枠（少し広げる）の中で、その色の画素の中心を取り直す。画素が無ければ None。
    with_bottom=True なら、色の画素の一番下の行（旗の根元）と画素数も返す。"""
    hsv = np.asarray(image.convert("HSV")).astype(int)
    h, w = hsv.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(int(x1) - margin, 0), max(int(y1) - margin, 0)
    x2, y2 = min(int(x2) + margin, w), min(int(y2) + margin, h)
    crop = hsv[y1:y2, x1:x2]
    mask = COLOR_MASKS[color_name](crop[..., 0], crop[..., 1], crop[..., 2])
    vs, us = np.nonzero(mask)
    if len(us) < 20:
        return None
    if with_bottom:
        return x1 + us.mean(), y1 + vs.mean(), y1 + vs.max(), len(us)
    return x1 + us.mean(), y1 + vs.mean()


class Locator:
    def __init__(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(MODEL)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")

    def ask_box(self, image, color_name):
        """
        image: カメラで撮った画像。頭のカメラ(1176, 672)、天井カメラ(672, 672)。
        color_name: 探す旗の色。"white", "red", "blue", "green", "yellow"

        return:
            box: 長さ4のタプル(x1, y1, x2, y2)(float)
            answer: 文字列
        """

        # 1. 質問文を作る
        prompt = (
            f"The image may be a top-down view, where cylinders look like circles. "
            f"Locate the {color_name} cylinder in the image. "
            f"If there is no {color_name} cylinder, answer none. "
            f'Output its bbox coordinates in JSON format like [{{"bbox_2d": [x1, y1, x2, y2], "label": "{color_name} cylinder"}}].'
        )
        # 2. メッセージを作る
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]},
        ]
        # 3. メッセージを、モデルが読める１つの文字列に直す
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # 関数の結果を文字列で受け取る
        # 4. 文字列と画像をまとめて入力し、GPUに送る、返り値はPytorchの配列
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        # 5. VLMの返答を受け取り、答えの部分だけを文字にする。学習しないのでtorch.no_grad()で勾配の記録をしない
        # **辞書は辞書をkey=valueでばらす
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=64, do_sample=False) # 答えの長さの上限は64, いつも一番確からしいトークンを選ぶ
        # 6. モデルは数の列を返すのでデコードする
        # [0]は１件ぶんの答えの文字列が入ったリストを返す
        answer = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0] # <|im_start|>などの区切りトークンをスキップする
        # 7. 答えを文章から探す
        match = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", answer)
        if not match:
            return None, answer
        # 7. 答えから４つの数を取り出す
        # match.groups()は一致した文字列を順番にタプルで返す
        box = tuple(float(g) for g in match.groups())
        return box, answer

    def find(self, image, color_name):
        """画像から color_name の旗を探し、旗の中心のピクセル (u, v) を返す。見つからなければ None。
        VLM が「どの色の旗がどのあたりか」を枠で答え（ask_box）、正確な位置は枠の中の色の画素で取り直す。"""
        box, answer = self.ask_box(image, color_name)
        if box is None:
            return None, answer
        refined = refine(image, box, color_name)
        if refined is not None:
            return refined, answer
        x1, y1, x2, y2 = box
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
