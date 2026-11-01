"""日本語の指示を、LLM（Qwen3-1.7B）で旗を回る順番に直す。

    uv run python llm_route.py "1番と3番の旗を回ってから、スタートに戻って"
    → 1,3,0

旗の番号と位置は go2_record_goals.py の既定（0:(0,0) スタート、1:(3,0)、2:(3,3)、3:(0,3)）に合わせる。
LLM の答えは形を確かめ、旗の番号の並び以外が返ってきたら、言い直させる。
"""

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"
FLAGS = {0: (0.0, 0.0), 1: (3.0, 0.0), 2: (0.0, 3.0), 3: (-3.0, 0.0), 4: (0.0, -3.0)}

SYSTEM = """あなたは、四足歩行ロボットの巡回ルートを決める係です。
床には旗が5本あります。ロボットは真ん中の白い旗の所にいて、前を向いています。
- 0番: 白い旗。真ん中のスタート地点
- 1番: 赤い旗。前
- 2番: 青い旗。左
- 3番: 緑の旗。後ろ
- 4番: 黄色い旗。右
ロボットは今、0番（スタート）にいます。
ユーザーの日本語の指示を読み、ロボットが向かう旗の番号を、向かう順にカンマ区切りで答えてください。
答えは番号の並びだけを1行で書き、説明は書かないでください。
例: 指示「2番の旗に行って戻ってきて」→ 2,0
例: 指示「赤の旗に行って」→ 1
例: 指示「緑に寄ってから青に行って」→ 3,2
例: 指示「右の旗を見てから戻って」→ 4,0
例: 指示「全部の旗を順番に回って」→ 1,2,3,4,0"""


def ask(tokenizer, model, instruction, retry_note=""):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instruction + retry_note},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def parse(answer):
    line = answer.splitlines()[0] if answer else ""
    if not re.fullmatch(r"\s*[0-4](\s*,\s*[0-4])*\s*", line):
        return None
    return [int(x) for x in line.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instruction", help="日本語の指示")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")

    answer = ask(tokenizer, model, args.instruction)
    route = parse(answer)
    for _ in range(2):
        if route:
            break
        print(f"LLM の答え「{answer}」は番号の並びではないので、言い直させる")
        answer = ask(tokenizer, model, args.instruction, "\n（旗の番号 0〜4 をカンマ区切りで、番号だけ答えてください）")
        route = parse(answer)
    if not route:
        raise SystemExit(f"LLM が番号の並びを返さなかった: {answer}")

    print(f"指示: {args.instruction}")
    print(f"LLM の答え: {answer}")
    # go2_record_goals.py の --route に渡す形（旗の番号のカンマ区切り）
    print("--route " + ",".join(str(i) for i in route))
    # go2_record_goals.py の --goals に渡す形（旗の位置 x:y のカンマ区切り）
    print("--goals " + ",".join(f"{FLAGS[i][0]:g}:{FLAGS[i][1]:g}" for i in route))


if __name__ == "__main__":
    main()
