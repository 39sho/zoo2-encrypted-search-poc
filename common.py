"""共通ユーティリティ: 日本語 bigram 化、BM25 ベクトル、CountSketch、合成コーパス。"""
import math
import random
from collections import Counter

import numpy as np

from emvp_poc.emvp import P


def bigrams(text: str) -> list[str]:
    chars = [c for c in text if not c.isspace()]
    return ["".join(chars[i : i + 2]) for i in range(len(chars) - 1)]


def generate_corpus(n_docs: int, vocab: list[str], terms_per_doc: int = 20, seed: int = 0):
    """規模計測用の合成文書（語のリスト）。固定語彙からサンプリングする。"""
    rng = random.Random(seed)
    return [rng.choices(vocab, k=rng.randint(max(2, terms_per_doc // 2), terms_per_doc))
            for _ in range(n_docs)]


def bm25_matrix(docs: list[list[str]], vocab: list[str], scale: int = 4096) -> np.ndarray:
    """文書 x 語彙 の BM25 重み行列（整数化して体 P に丸める）。"""
    n = len(docs)
    term_id = {t: i for i, t in enumerate(vocab)}
    tfs = [Counter(doc) for doc in docs]
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
            M[d, term_id[term]] = int(round(idf * cnt * (k1 + 1) / denom * scale)) % P
    return M


def query_vector(terms: list[str], vocab: list[str]) -> np.ndarray:
    vid = {t: i for i, t in enumerate(vocab)}
    v = np.zeros(len(vocab), dtype=np.int64)
    for t in terms:
        if t in vid:
            v[vid[t]] = 1
    return v


class CountSketch:
    """CountSketch でスパースベクトルを密ベクトルへ。identity=True で恒等写像。"""

    def __init__(self, vocab_size: int, dim: int, seed: int, identity: bool = False):
        self.dim = dim
        self.identity = identity
        if identity:
            assert vocab_size <= dim
            return
        rng = random.Random(seed)
        self.a = [rng.randrange(1, P) for _ in range(2)]
        self.g = [rng.randrange(1, P) for _ in range(2)]

    def _h(self, i: int) -> int:
        return ((self.a[0] * i + self.a[1]) % P) % self.dim

    def _g(self, i: int) -> int:
        return 1 if ((self.g[0] * i + self.g[1]) % 2) == 0 else -1

    def sketch(self, vec: np.ndarray) -> np.ndarray:
        if self.identity:
            out = np.zeros(self.dim, dtype=np.int64)
            out[: len(vec)] = vec
            return out
        out = np.zeros(self.dim, dtype=np.int64)
        for i in np.nonzero(vec)[0]:
            j = self._h(int(i))
            out[j] = (out[j] + self._g(int(i)) * int(vec[i])) % P
        return out
