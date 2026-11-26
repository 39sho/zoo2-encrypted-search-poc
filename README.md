# zoo2-encrypted-search-poc

E2EE が有効なチャンネルも検索タブから全文検索したい、という課題について、
「暗号文上で全文検索すると実際どうなるのか」を方式比較するために作った実験コード。

比較の記録は [Zli-UoA/zoo2#723](https://github.com/Zli-UoA/zoo2/issues/723) と [RESULT.md](RESULT.md)。

**これは実験用の参照実装で、本番コードではありません。** 設計は未確定。

## 内容

| ファイル | 内容 |
|---|---|
| `emvp.py` | EMVP（Encrypted Matrix-Vector Product）の小規模実装。Lexicure §3.3.1 に沿う |
| `pir.py` | Paillier 加法準同型による単一サーバー PIR |
| `main.py` | BM25 → CountSketch → EMVP → 復号・ランキング → PIR 取得の end-to-end デモ |
| `scale.py` | 文書数を変えた規模別ベンチマーク |
| `RESULT.md` | 計測結果と制約 |

おまけで、最初に試した汎用 FHE（CKKS / OpenFHE）の PoC も比較用に計測している。数値は `RESULT.md` に載っている（コードは未同梱）。

## セットアップ

Python 3.12 と NumPy 2.x を使う。Nix 環境なら例えば:

```bash
nix --extra-experimental-features 'nix-command flakes' shell \
  nixpkgs#python312 nixpkgs#gcc --command bash -c '
    python3 -m venv .venv &&
    .venv/bin/pip install -r requirements.txt
  '
```

## 実行

```bash
# Nix を使っている場合のみ必要
export LD_LIBRARY_PATH=$(dirname $(find /nix/store -path "*/lib/libstdc++.so.6" | head -1))

.venv/bin/python main.py    # end-to-end デモ
.venv/bin/python scale.py   # 規模別ベンチマーク
```

`main.py` はクエリ「会議 資料」で日本語メッセージ 16 件を検索し、
EMVP で復号したスコアのランキングと、PIR で取得した本文を表示する。

## 制約（重要）

- EMVP の肝であるトラップドア付き擬似乱数行列を省略し、乱数行列を直接掛けている。
  クライアント側の軽さは未検証。
- PIR は O(N) 個の暗号文を送る素朴な形。通信削減は未実装。
- 静的索引のみ。追加・削除・更新は未対応。
- 安全性は semi-honest のみ。
- 語彙 155・スケッチ次元 256 と小規模で、実規模のレイテンシは未検証。

詳細は [RESULT.md](RESULT.md)。
