"""Lexicure 方式（EMVP + PIR）の leakage-free 検索 PoC。

フロー:
  1. クライアント: 日本語メッセージから bigram 語彙を作り BM25 ベクトルを計算
  2. クライアント: CountSketch で d=64 に圧縮し、EMVP で行列を暗号化してサーバーへ
  3. クライアント: クエリも CountSketch + EMVP で暗号化して送る
  4. サーバー: 暗号文のまま行列ベクトル積を計算し、暗号化スコアを返す
  5. クライアント: 復号して top-K を選ぶ（サーバーは結果を見ない）
  6. クライアント: PIR で top-1 の本文を取得（サーバーは位置を見ない）
"""
import math
import random
import time
from collections import Counter

import numpy as np

from emvp import EMVPClient, P, evaluate, matvec_mod
from pir import PaillierPIR, encode_text, decode_text

CORPUS = [
    "明日の会議の資料を共有します",
    "暗号化チャンネルの検索を実験する",
    "全文検索は準同型暗号でできるか",
    "鍵のローテーションを忘れないで",
    "会議は午後3時からです",
    "検索タブから暗号化チャンネルも探したい",
    "漏えいを許容しない方式を選ぶ",
    "インデックスは暗号文のまま保存する",
    "資料のPDFをアップロードしました",
    "復号した平文は保存しない",
    "新しい端末を検証しました",
    "アクセスパターンも見せない",
    "会議の資料を検索する",
    "暗号化された検索インデックス",
    "検索結果は暗号文のまま返す",
    "サーバーは平文を見ない",
]
QUERY = "会議 資料"


def bigrams(text: str):
    chars = [c for c in text if not c.isspace()]
    return ["".join(chars[i : i + 2]) for i in range(len(chars) - 1)]


def bm25_matrix(corpus, vocab, scale=4096):
    n = len(corpus)
    term_id = {t: i for i, t in enumerate(vocab)}
    tfs = [Counter(bigrams(doc)) for doc in corpus]
    df = Counter()
    for tf in tfs:
        df.update(tf.keys())
    avgdl = sum(sum(tf.values()) for tf in tfs) / n
    k1, b = 1.5, 0.75
    M = np.zeros((n, len(vocab)), dtype=np.int64)
    for d, tf in enumerate(tfs):
        dl = sum(tf.values())
        for term, cnt in tf.items():
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = cnt + k1 * (1 - b + b * dl / avgdl)
            w = idf * cnt * (k1 + 1) / denom
            M[d, term_id[term]] = int(round(w * scale)) % P
    return M


class CountSketch:
    def __init__(self, vocab_size, dim, seed):
        rng = random.Random(seed)
        self.dim = dim
        self.a = [rng.randrange(1, P) for _ in range(2)]
        self.g = [rng.randrange(1, P) for _ in range(2)]

    def _h(self, i):
        return ((self.a[0] * i + self.a[1]) % P) % self.dim

    def _g(self, i):
        return 1 if ((self.g[0] * i + self.g[1]) % 2) == 0 else -1

    def sketch(self, vec):
        out = np.zeros(self.dim, dtype=np.int64)
        for i in np.nonzero(vec)[0]:
            j = self._h(int(i))
            out[j] = (out[j] + self._g(int(i)) * int(vec[i])) % P
        return out


def main():
    n = len(CORPUS)
    vocab = sorted({g for doc in CORPUS for g in bigrams(doc)})
    d, k, block = 256, 128, 32
    sketch = CountSketch(len(vocab), d, seed=11)

    print(f"docs={n}, vocab={len(vocab)}, sketch_dim={d}, EMVP n={d + k}, blocks={(d + k) // block}")
    print(f"query: {QUERY}\n")

    # --- クライアント: セットアップ ---
    t0 = time.perf_counter()
    M_plain = bm25_matrix(CORPUS, vocab)
    M_sketch = np.array([sketch.sketch(row) for row in M_plain])
    client = EMVPClient(n, d, k, block, seed=23)
    M_hat = client.encrypt_matrix(M_sketch)
    print(f"[client] BM25 + sketch + EMVP encrypt: {time.perf_counter() - t0:.3f}s")
    print(f"         encrypted index: {M_hat.shape} (server holds only this)\n")

    # --- クライアント: クエリ暗号化 ---
    t_online = time.perf_counter()
    q_plain = np.zeros(len(vocab), dtype=np.int64)
    for term in bigrams(QUERY):
        if term in {t: i for i, t in enumerate(vocab)}:
            q_plain[vocab.index(term)] = 1
    q_sketch = sketch.sketch(q_plain)
    t0 = time.perf_counter()
    q_hat, state = client.encrypt_query(q_sketch)
    q_hat2, _ = client.encrypt_query(q_sketch)
    print(f"[client] query encrypt: {time.perf_counter() - t0:.3f}s")
    same_bytes = all(np.array_equal(a, b) for a, b in zip(q_hat, q_hat2))
    print(f"         same query -> identical ciphertext: {same_bytes} (False = no search-pattern leak)\n")

    # --- サーバー: 暗号文のまま行列ベクトル積 ---
    t0 = time.perf_counter()
    A = evaluate(M_hat, q_hat, block)
    server_time = time.perf_counter() - t0
    print(f"[server] EMVP evaluate: {server_time:.3f}s")
    print(f"         server sees only M_hat, q_hat and returns A={A.shape}\n")

    # --- クライアント: 復号とランキング ---
    t0 = time.perf_counter()
    scores = client.decrypt(A, state)
    decrypt_time = time.perf_counter() - t0
    expected = matvec_mod(M_sketch, q_sketch)
    err = int(np.max(np.abs(scores.astype(object) - expected.astype(object))))
    ranked = sorted(range(n), key=lambda i: (-int(scores[i]), i))
    print(f"[client] decrypt+rank: {decrypt_time:.3f}s")
    print(f"         max error vs sketch inner product: {err}")
    print("         top results:")
    for i in ranked[:5]:
        print(f"           score={int(scores[i]):7d}  {CORPUS[i]}\n")

    # --- PIR で本文を取得 ---
    t0 = time.perf_counter()
    pir = PaillierPIR(bits=512, seed=7)
    pir_keygen = time.perf_counter() - t0
    target = ranked[0]
    items = [encode_text(doc, pir.n) for doc in CORPUS]
    t0 = time.perf_counter()
    encrypted_query = pir.query(target, n)
    pir_query = time.perf_counter() - t0
    t0 = time.perf_counter()
    answer = PaillierPIR.answer_with_mod(items, encrypted_query, pir.n2)
    pir_server = time.perf_counter() - t0
    recovered = decode_text(pir.decrypt(answer))
    print(f"[PIR] keygen: {pir_keygen:.3f}s, query: {pir_query:.3f}s, server: {pir_server:.3f}s")
    print(f"      requested index={target} (hidden from server), recovered='{recovered}'")
    print(f"      match: {recovered == CORPUS[target]}\n")

    print("=== summary ===")
    print(f"online query latency (query enc + EMVP + decrypt + PIR): "
          f"{time.perf_counter() - t_online:.3f}s")
    print(f"EMVP server time: {server_time:.3f}s for {n} docs")
    print("leakage-free properties demonstrated:")
    print("  - search pattern: same query encrypts differently")
    print("  - access pattern: server computes one uniform matvec over all docs")
    print("  - result pattern: server never sees decrypted scores or top-K")
    print("  - retrieval: server sees one PIR ciphertext, not the requested index")


if __name__ == "__main__":
    main()
