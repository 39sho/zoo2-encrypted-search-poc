"""EMVP + PIR による leakage-free 検索の end-to-end デモ。"""
import time

import numpy as np

from common import CountSketch, bigrams, bm25_matrix, query_vector
from emvp_poc.emvp import EMVPClient, evaluate, matvec_mod
from emvp_poc.pir import PaillierPIR, decode_text, encode_text

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


def main():
    docs = [bigrams(s) for s in CORPUS]
    vocab = sorted({t for doc in docs for t in doc})
    d, k, block = 256, 128, 32
    sketch = CountSketch(len(vocab), d, seed=11)
    print(f"docs={len(docs)}, vocab={len(vocab)}, sketch={d}, EMVP n={d+k}, blocks={(d+k)//block}")
    print(f"query: {QUERY}\n")

    t0 = time.perf_counter()
    M = np.array([sketch.sketch(row) for row in bm25_matrix(docs, vocab)])
    client = EMVPClient(len(docs), d, k, block, seed=23)
    M_hat = client.encrypt_matrix(M)
    print(f"[client] BM25 + sketch + EMVP encrypt: {time.perf_counter() - t0:.3f}s")
    print(f"         encrypted index: {M_hat.shape} (server holds only this)\n")

    t_online = time.perf_counter()
    q = sketch.sketch(query_vector(bigrams(QUERY), vocab))
    t0 = time.perf_counter()
    qhat, state = client.encrypt_query(q)
    qhat2, _ = client.encrypt_query(q)
    q_enc = time.perf_counter() - t0
    same = all(np.array_equal(a, b) for a, b in zip(qhat, qhat2))
    print(f"[client] query encrypt: {q_enc:.3f}s")
    print(f"         same query -> identical ciphertext: {same} (False = no search-pattern leak)\n")

    t0 = time.perf_counter()
    A = evaluate(M_hat, qhat, block)
    server = time.perf_counter() - t0
    print(f"[server] EMVP evaluate: {server:.3f}s, returns A={A.shape}\n")

    t0 = time.perf_counter()
    scores = client.decrypt(A, state)
    dec = time.perf_counter() - t0
    expected = matvec_mod(M, q)
    err = int(np.max(np.abs(scores.astype(object) - expected.astype(object))))
    ranked = sorted(range(len(docs)), key=lambda i: (-int(scores[i]), i))
    print(f"[client] decrypt+rank: {dec:.3f}s, max error vs sketch inner product: {err}")
    for i in ranked[:5]:
        print(f"  score={int(scores[i]):7d}  {CORPUS[i]}")
    print()

    pir = PaillierPIR(bits=512, seed=7)
    target = ranked[0]
    items = [encode_text(s, pir.n) for s in CORPUS]
    t0 = time.perf_counter()
    eq = pir.query(target, len(items))
    pq = time.perf_counter() - t0
    t0 = time.perf_counter()
    ans = PaillierPIR.answer_with_mod(items, eq, pir.n2)
    ps = time.perf_counter() - t0
    recovered = decode_text(pir.decrypt(ans))
    print(f"[PIR] query={pq:.3f}s, server={ps:.3f}s, requested index={target} (hidden)")
    print(f"      recovered='{recovered}', match={recovered == CORPUS[target]}\n")

    print("=== summary ===")
    print(f"online (query enc + EMVP + decrypt + PIR): "
          f"{time.perf_counter() - t_online:.3f}s")
    print("leakage-free: randomized queries / uniform full-corpus matvec / "
          "server never sees scores or retrieved index")


if __name__ == "__main__":
    main()
