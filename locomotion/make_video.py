"""撮った動画をつなぎ、字幕をのせて、提出用の1本にする。

    uv run python make_video.py            # videos/final.mp4
    uv run python make_video.py --dry-run  # 素材がそろっているかと、全体の長さだけ表示

各場面は (素材, 開始秒, 終了秒, 再生速度, 字幕)。素材が画像なら、その画像を指定の秒数だけ出す。
"""

import argparse
import os

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
BG = (20, 24, 38)

# (素材, 開始秒, 終了秒, 速さ, 字幕)。素材が "title:" で始まるときは文字だけの画面
SCENES = [
    ("title:転倒に強い犬型ロボット\n押されても踏ん張り、倒れても起き上がって巡回を続ける\nGenesis × Unitree Go2 × 強化学習（PPO）", 0, 5, 1.0, ""),
    ("videos/stand_push3.0.mp4", 0, 9, 1.0, "弱い押し（〜0.6 m/s）で練習しただけの犬は、強く押されると倒れる"),
    ("videos/stand-push_push3.0_nolimit.mp4", 0, 9, 1.0, "学習中に強く押す（〜3.0 m/s）と、踏ん張れるようになる"),
    ("videos/getup_back.mp4", 0, 8, 1.0, "仰向けに転がした姿勢から始めて「立っているほど加点」で学習 → 自分で起き上がる"),
    ("videos/unified_strong.mp4", 2, 16, 1.0, "1つの方策で両方: 押されて踏ん張る → 6.0 m/s で倒れても起き上がる"),
    ("videos/run.mp4", 2, 16, 1.0, "走りながら（1.5 m/s）押されても、起き上がって元の向きに戻り、また走り出す"),
    ("videos/vlm_patrol_overhead.png", 0, 3, 1.0, "指示「緑に寄ってから黄色に行って、スタートに戻って」→ LLM が旗の順番に、VLM が天井カメラの画像から旗の位置を見つける"),
    ("videos/vlm_patrol.mp4", 0, 18, 1.0, "旗を順に巡回。途中で押されても立て直して続ける"),
    ("videos/egocentric_push.mp4", 0, 60, 2.0, "【2倍速】犬の頭のカメラだけで旗を探す。押されて見失うと、回って探し直す"),
    ("title:結果\n・3.0 m/s の押し: どの向きでも倒れなくなった（押されるだけの練習では後ろから押されると倒れた）\n・4.0 m/s で倒れても、3秒以内に全員起き上がった\n・課題: 止まったまま大きく回れない、後ろからの押しに弱い", 0, 8, 1.0, ""),
    ("title:今後\n段差や坂、つまずきやすい地面でも、倒れにくく、倒れても起き上がれるようにする", 0, 4, 1.0, ""),
]


def font(size):
    return ImageFont.truetype(FONT, size)


def wrap(text, f, width):
    """文字の幅を測って折り返す（日本語は空白で区切れないため、1文字ずつ詰める）。"""
    lines, line = [], ""
    for ch in text:
        if ImageFont.ImageFont and f.getlength(line + ch) > width and line:
            lines.append(line)
            line = ch
        else:
            line += ch
    if line:
        lines.append(line)
    return lines


def caption(img, text):
    if not text:
        return img
    draw = ImageDraw.Draw(img, "RGBA")
    f = font(32)
    lines = wrap(text, f, W - 80)
    pad, line_h = 14, 44
    box_h = pad * 2 + line_h * len(lines)
    draw.rectangle([0, H - box_h, W, H], fill=(10, 12, 20, 200))
    for k, line in enumerate(lines):
        draw.text((40, H - box_h + pad + k * line_h), line, font=f, fill=(240, 242, 246))
    return img


def title_frame(text):
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    lines = text.split("\n")
    y = H // 2 - 60 * len(lines) // 2
    for k, line in enumerate(lines):
        f = font(52 if k == 0 else 30)
        for sub in wrap(line, f, W - 120) or [""]:
            w = draw.textlength(sub, font=f)
            draw.text(((W - w) / 2, y), sub, font=f, fill=(240, 138, 75) if k == 0 else (225, 228, 235))
            y += 70 if k == 0 else 46
    return img


def fit(img):
    """縦横比を保って 1280x720 に収め、余白は背景色で埋める。"""
    img.thumbnail((W, H))
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
    return canvas


def frames_of(src, start, end, speed):
    if src.startswith("title:"):
        img = title_frame(src[len("title:"):])
        for _ in range(int((end - start) * FPS)):
            yield img
        return
    if src.lower().endswith(".png"):
        img = fit(Image.open(src).convert("RGB"))
        for _ in range(int((end - start) * FPS)):
            yield img
        return
    with av.open(src) as c:
        stream = c.streams.video[0]
        rate = float(stream.average_rate)
        frames = [f.to_image() for f in c.decode(stream)]
    first, last = int(start * rate), min(int(end * rate), len(frames))
    n_out = int((last - first) / rate / speed * FPS)
    for k in range(n_out):
        yield fit(frames[first + int(k * speed * rate / FPS)].copy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="videos/final.mp4")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total = 0.0
    for src, start, end, speed, _ in SCENES:
        ok = src.startswith("title:") or os.path.exists(src)
        total += (end - start) / speed
        print(f"{'OK ' if ok else '無い'} {src.splitlines()[0][:50]:50s} {start:>4}–{end:<4} ×{speed}")
    print(f"全体の長さ: {total:.0f} 秒（1〜3分の条件: {'OK' if 60 <= total <= 180 else '外れている'}）")
    if args.dry_run:
        return

    with av.open(args.out, "w") as out:
        stream = out.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        for src, start, end, speed, text in SCENES:
            for img in frames_of(src, start, end, speed):
                frame = av.VideoFrame.from_ndarray(np.asarray(caption(img.copy(), text)), format="rgb24")
                for packet in stream.encode(frame):
                    out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
