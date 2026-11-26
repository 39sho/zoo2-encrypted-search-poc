"""規模別ベンチマーク（issue 用の数値を出す）。"""
import time
from collections import Counter

import numpy as np

from emvp import EMVPClient, P, evaluate, matvec_mod
from main import CORPUS, QUERY, CountSketch, bigrams, bm25_matrix


def run(n_docs: int):
    corpus = (CORPUS * ((n_docs + len(CORPUS) - 1) // len(CORPUS)))[:n_docs]
    vocab = sorted({g for doc in corpus for g in bigrams(doc)})
    d, k, block = 256, 128, 32
    sketch = CountSketch(len(vocab), d, seed=11)

    t = time.perf_counter()
    M = np.array([sketch.sketch(row) for row in bm25_matrix(corpus, vocab)])
    client = EMVPClient(n_docs, d, k, block, seed=23)
    M_hat = client.encrypt_matrix(M)
    setup = time.perf_counter() - t

    q = sketch.sketch(np.array([1 if g in bigrams(QUERY) else 0 for g in vocab], dtype=np.int64))
    t = time.perf_counter()
    qhat, state = client.encrypt_query(q)
    q_enc = time.perf_counter() - t

    t = time.perf_counter()
    A = evaluate(M_hat, qhat, block)
    server = time.perf_counter() - t

    t = time.perf_counter()
    scores = client.decrypt(A, state)
    dec = time.perf_counter() - t

    expected = matvec_mod(M, q)
    assert np.array_equal(scores, expected)
    print(f"docs={n_docs:5d} | setup={setup:6.2f}s | query_enc={q_enc:.3f}s | "
          f"EMVP server={server:6.3f}s | decrypt={dec:.3f}s | "
          f"online={q_enc + server + dec:.3f}s")


if __name__ == "__main__":
    for n in (16, 64, 256):
        run(n)
