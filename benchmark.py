"""両方式を同じ条件で比較するベンチマーク。

使い方:
  python benchmark.py --method emvp  --sizes 1000 10000 100000
  python benchmark.py --method ckks  --sizes 1000 10000

両方式とも語彙 64・特徴量 64 次元・同じ BM25 ベクトルを使う。
CKKS は 1 暗号文に 8 文書をパッキングする。
"""
import argparse
import random
import time

import numpy as np

from common import CountSketch, bm25_matrix, generate_corpus
from emvp_poc.emvp import EMVPClient, evaluate, matvec_mod

VOCAB = [f"語{i:02}" for i in range(64)]
SCALE = 256
D = 64
K = 128
B = 32
SEED = 0


def make_data(n_docs, query_terms=4):
    docs = generate_corpus(n_docs, VOCAB, terms_per_doc=20, seed=SEED)
    M = bm25_matrix(docs, VOCAB, scale=SCALE)
    q = np.zeros(D, dtype=np.int64)
    for i in random.Random(SEED).sample(range(D), query_terms):
        q[i] = 1
    return docs, M, q


def run_emvp(n_docs):
    _, M, q = make_data(n_docs)
    sketch = CountSketch(D, D, seed=1, identity=True)
    M = np.array([sketch.sketch(row) for row in M])
    q = sketch.sketch(q)

    t = time.perf_counter()
    client = EMVPClient(n_docs, D, K, B, seed=23)
    M_hat = client.encrypt_matrix(M)
    setup = time.perf_counter() - t

    t = time.perf_counter()
    qhat, state = client.encrypt_query(q)
    q_enc = time.perf_counter() - t

    t = time.perf_counter()
    A = evaluate(M_hat, qhat, B)
    server = time.perf_counter() - t

    t = time.perf_counter()
    scores = client.decrypt(A, state)
    dec = time.perf_counter() - t

    expected = matvec_mod(M, q)
    assert np.array_equal(scores, expected), "EMVP score mismatch"
    return {
        "setup": setup,
        "query_enc": q_enc,
        "server": server,
        "decrypt": dec,
        "online": q_enc + server + dec,
        "ct": M_hat.shape[1] * n_docs,
    }


def run_ckks(n_docs):
    from ckks_poc.core import run

    _, M, q = make_data(n_docs)
    doc_vectors = [[float(x) for x in row] for row in M]
    query = [float(x) for x in q]
    scores, timings, n_ct = run(doc_vectors, query, V=D, batch=512, scale_bits=50)
    expected = [sum(a * b for a, b in zip(row, query)) for row in doc_vectors]
    err = max(abs(s - e) for s, e in zip(scores, expected))
    assert err < 1e-3, f"CKKS error too large: {err}"
    return {**timings, "online": timings["query_encrypt"] + timings["server"] + timings["decrypt"],
            "ct": n_ct, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["emvp", "ckks"], required=True)
    ap.add_argument("--sizes", nargs="+", type=int, required=True)
    args = ap.parse_args()

    fn = run_emvp if args.method == "emvp" else run_ckks
    print(f"| docs | setup | query_enc | server | decrypt | online | ct |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for n in args.sizes:
        t0 = time.perf_counter()
        r = fn(n)
        total = time.perf_counter() - t0
        setup = r.get("setup", r.get("index_encrypt", 0.0))
        qe = r.get("query_enc", r.get("query_encrypt", 0.0))
        print(f"| {n} | {setup:.2f}s | {qe:.3f}s | {r['server']:.3f}s "
              f"| {r['decrypt']:.3f}s | {r['online']:.3f}s | {r['ct']} | "
              f"(total {total:.2f}s)")
        if args.method == "ckks":
            print(f"  error={r['error']:.2e}", end="")
        print(flush=True)


if __name__ == "__main__":
    main()
