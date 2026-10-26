# locomotion

Unitree Go2 の歩行を強化学習（PPO）で学ぶ。

`go2_env.py`・`go2_train.py`・`go2_eval.py`・`go2_backflip.py`・`backflip/` は
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) v1.4.5 の `examples/locomotion` をそのままコピーしたもの（Apache-2.0、`LICENSE`）。
これを土台に、階段を登れるように手を加えていく。

`go2_record.py` は追加したもの。ビューアを開かずに、学習済みの方策の動きを mp4 に書き出す。

```bash
uv run python go2_train.py                 # 平地の歩行を学習（logs/go2-walking/）
uv run python go2_record.py --ckpt 100     # videos/go2-walking_ckpt100.mp4
```
