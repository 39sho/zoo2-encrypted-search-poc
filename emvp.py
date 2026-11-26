"""EMVP (Encrypted Matrix-Vector Product) の小規模実装。

Lexicure §3.3.1 / Benhamouda et al. ePrint 2025/858 に沿う。
    C = [P | I_k],  D = [I_d | -P^T]  (C D^T = 0)
    M_hat = M D + Z
    q_tilde = (q | 0) + r C
    server: a'_i = q_hat_i * M_hat_i^T
    client: a = sum(α_i^-1 a'_i) - r' = q M^T

PoC の単純化:
  - 原本の Z はトラップドア付き擬似乱数行列で、クライアントは短い鍵から
    r' = q̃ Z^T を高速に計算する。ここでは Z を乱数から直接生成して掛ける
    （Z が一様乱数なら情報理論的マスクとして有効）。
  - 体は p = 2^31-1 (Mersenne prime)、d=64, k=192, n=256。
"""
import numpy as np

P = 2**31 - 1


def matmul_mod(A: np.ndarray, B: np.ndarray, p: int = P) -> np.ndarray:
    """int64 オーバーフローを避け、Python int で畳み込んで mod p を取る。"""
    A = A.astype(object)
    B = B.astype(object)
    return np.array(
        [[sum(int(a) * int(b) for a, b in zip(row, col)) % p for col in B.T] for row in A],
        dtype=np.int64,
    )


def matvec_mod(A: np.ndarray, v: np.ndarray, p: int = P) -> np.ndarray:
    return np.array(
        [sum(int(a) * int(b) for a, b in zip(row, v)) % p for row in A],
        dtype=np.int64,
    )


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
        # 秘密鍵（短いシード）から P と Z を導出。
        self.Pmat = self.rng.integers(0, P, size=(d, k), dtype=np.int64) % P
        self.Z = self.rng.integers(0, P, size=(n_docs, self.n), dtype=np.int64) % P

        # Pmat は d x k。C = [P^T | I_k] (k x n), D = [I_d | -P] (d x n)
        # すると C D^T = P^T - P^T = 0。
        self.C = np.concatenate([self.Pmat.T, np.eye(k, dtype=np.int64)], axis=1) % P
        self.D = np.concatenate([np.eye(d, dtype=np.int64), (-self.Pmat) % P], axis=1) % P
        assert np.all(matmul_mod(self.C, self.D.T) == 0), "C D^T must be zero"

    def encrypt_matrix(self, M: np.ndarray) -> np.ndarray:
        """M (N x d) -> M_hat (N x n)。サーバーに渡す。"""
        return (matmul_mod(M, self.D) + self.Z) % P

    def encrypt_query(self, q: np.ndarray):
        """q (d,) -> (q_hat, state)。サーバーに q_hat を渡す。"""
        r = self.rng.integers(0, P, size=self.k, dtype=np.int64) % P
        qt = np.concatenate([q.astype(np.int64) % P, np.zeros(self.k, dtype=np.int64)])
        qt = (qt + matvec_mod(self.C.T, r)) % P
        rprime = matvec_mod(self.Z, qt)
        alphas = np.array(
            [int(self.rng.integers(1, P)) for _ in range(self.s)], dtype=np.int64
        )
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
    """サーバー側。暗号化行列と暗号化クエリから暗号化スコアを計算する。"""
    out = np.zeros((len(q_hat), M_hat.shape[0]), dtype=np.int64)
    for i, qi in enumerate(q_hat):
        block = M_hat[:, i * b : (i + 1) * b]
        if block.shape[1] < b:
            block = np.concatenate(
                [block, np.zeros((M_hat.shape[0], b - block.shape[1]), dtype=np.int64)],
                axis=1,
            )
        out[i] = matvec_mod(block, qi)
    return out
