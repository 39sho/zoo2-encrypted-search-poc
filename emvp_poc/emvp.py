"""EMVP (Encrypted Matrix-Vector Product) の小規模実装。

Lexicure §3.3.1 / Benhamouda et al. ePrint 2025/858 に沿う。
    C = [P^T | I_k],  D = [I_d | -P]   (C D^T = 0)
    M_hat = M D + Z
    q_tilde = (q | 0) + r C
    server: a'_i = q_hat_i * M_hat_i^T
    client: a = sum(α_i^-1 a'_i) - r' = q M^T

PoC の単純化:
  - 原本の Z はトラップドア付き擬似乱数行列。ここではシードから一様乱数として
    生成して直接掛ける（情報理論的マスクとしては有効だが、クライアントの
    r' 計算は O(N·n) で原本より重い）。
  - 体は 16 bit 素数 p=65521。積和が int64 に収まるため NumPy の整数
    行列演算をそのまま使える（ large scale ベンチマーク用）。
"""
import numpy as np

P = 65521  # 16-bit prime


def matmul_mod(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A.astype(np.int64) @ B.astype(np.int64)) % P


def matvec_mod(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (A.astype(np.int64) @ v.astype(np.int64)) % P


def _inv(x: int) -> int:
    return pow(int(x) % P, P - 2, P)


class EMVPClient:
    def __init__(self, n_docs: int, d: int, k: int, b: int, seed: int):
        self.n_docs = n_docs
        self.d = d
        self.k = k
        self.n = d + k
        self.b = b
        self.s = (self.n + b - 1) // b
        self.rng = np.random.default_rng(seed)
        self.Pmat = self.rng.integers(0, P, size=(d, k), dtype=np.int64)
        self.Z = self.rng.integers(0, P, size=(n_docs, self.n), dtype=np.int64)

        self.C = np.concatenate([self.Pmat.T, np.eye(k, dtype=np.int64)], axis=1) % P
        self.D = np.concatenate([np.eye(d, dtype=np.int64), (-self.Pmat) % P], axis=1) % P
        assert np.all(matmul_mod(self.C, self.D.T) == 0), "C D^T must be zero"

    def encrypt_matrix(self, M: np.ndarray) -> np.ndarray:
        """M (N x d) -> M_hat (N x n)。サーバーに渡す。"""
        return (matmul_mod(M, self.D) + self.Z) % P

    def encrypt_query(self, q: np.ndarray):
        """q (d,) -> (q_hat, state)。サーバーに q_hat を渡す。"""
        r = self.rng.integers(0, P, size=self.k, dtype=np.int64)
        qt = np.concatenate([q.astype(np.int64) % P, np.zeros(self.k, dtype=np.int64)])
        qt = (qt + matvec_mod(self.C.T, r)) % P
        rprime = matvec_mod(self.Z, qt)
        alphas = np.array([int(self.rng.integers(1, P)) for _ in range(self.s)], dtype=np.int64)
        qhat = []
        for i in range(self.s):
            block = qt[i * self.b : (i + 1) * self.b]
            if len(block) < self.b:
                block = np.concatenate([block, np.zeros(self.b - len(block), dtype=np.int64)])
            qhat.append((alphas[i] * block) % P)
        return qhat, (rprime, alphas)

    def decrypt(self, A: np.ndarray, state) -> np.ndarray:
        """A (s x N) -> scores (N,)。"""
        rprime, alphas = state
        total = np.zeros(self.n_docs, dtype=np.int64)
        for i in range(self.s):
            total = (total + _inv(int(alphas[i])) * A[i]) % P
        return (total - rprime) % P


def evaluate(M_hat: np.ndarray, q_hat, b: int) -> np.ndarray:
    """サーバー側。暗号文のまま全ドキュメントのスコアを計算する。"""
    out = np.zeros((len(q_hat), M_hat.shape[0]), dtype=np.int64)
    for i, qi in enumerate(q_hat):
        block = M_hat[:, i * b : (i + 1) * b]
        if block.shape[1] < b:
            block = np.concatenate(
                [block, np.zeros((M_hat.shape[0], b - block.shape[1]), dtype=np.int64)], axis=1
            )
        out[i] = matvec_mod(block, qi)
    return out
