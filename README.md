# zoo2-encrypted-search-poc

E2EE が有効なチャンネルも検索タブから全文検索したい、という課題について、
「暗号文上で全文検索すると実際どうなるのか」を方式比較するために作った実験コード。

比較の記録: [Zli-UoA/zoo2#723](https://github.com/Zli-UoA/zoo2/issues/723)
- [RESULT.md](RESULT.md) — 最初の小規模計測（16 文書）
- [BENCHMARK.md](BENCHMARK.md) — 規模を上げた両方式の比較（1,000〜100,000 文書）

**実験用の参照実装で、本番コードではない。設計は未確定。**

## ディレクトリ

```
common.py            日本語 bigram / BM25 / CountSketch / 合成コーパス
emvp_poc/
  emvp.py            EMVP プロトコル（行列・クエリ暗号化、評価、復号）
  pir.py             Paillier 加法準同型による単一サーバー PIR
  search.py          EMVP + PIR の end-to-end デモ
ckks_poc/
  core.py            比較用の CKKS 内積（複数文書をパッキング）
benchmark.py         両方式を同じ条件で規模別に計測
```

## セットアップ

Python 3.12 を使う（OpenFHE の wheel が 3.12 用）。Nix なら:

```bash
nix --extra-experimental-features 'nix-command flakes' shell \
  nixpkgs#python312 nixpkgs#gcc --command bash -c '
    python3 -m venv .venv &&
    .venv/bin/pip install -r requirements.txt &&
    .venv/bin/pip install -r requirements-ckks.txt
  '
```

EMVP だけなら `requirements-ckks.txt` は不要。

## 実行

```bash
# Nix を使っている場合のみ必要
export LD_LIBRARY_PATH=$(dirname $(find /nix/store -path "*/lib/libstdc++.so.6" | head -1))

.venv/bin/python -m emvp_poc.search      # 日本語 16 文書の end-to-end デモ
.venv/bin/python benchmark.py --method emvp --sizes 1000 10000 100000
.venv/bin/python benchmark.py --method ckks --sizes 1000 10000
```

`search.py` はクエリ「会議 資料」で検索し、EMVP の復号スコア順位と PIR で取得した本文を表示する。

## 制約（重要）

- EMVP の肝であるトラップドア付き擬似乱数行列を省略し、乱数行列を直接掛けている。
  クライアント側の軽さは未検証。
- 体は `F_65521`、特徴量 64 次元と小さく、论文のパラメータ（p≈2^30, d≈10^4）ではない。
  絶対値は PoC パラメータでの比較。
- PIR は O(N) 個の暗号文を送る素朴な形。通信削減は未実装。
- 静的索引のみ。追加・削除・更新は未対応。
- 安全性は semi-honest のみ。
- CKKS は比較用で、10 万件は未計測（外挿では約 17 分）。

詳細は [RESULT.md](RESULT.md) と [BENCHMARK.md](BENCHMARK.md)。
