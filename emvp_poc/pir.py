"""単一サーバー PIR（Paillier 加法準同型）の小規模実装。

Lexicure ではスコア計算（EMVP）と本文取得（PIR）を分離する。
サーバーはどの項目を取得したかを知らずに、暗号化された選択ベクトルから
対象項目だけを復号結果として返す（サーバーは平文も位置も見ない）。

PoC なので選択ベクトルは O(N) 個の暗号文を送る素朴な形。
実用化には压缩 PIR（例: 多重 LINEAR/SimplePIR）が必要。
"""
import random


def _is_probable_prime(n: int, rounds: int = 24) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(rounds):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _random_prime(bits: int) -> int:
    while True:
        cand = random.getrandbits(bits) | (1 << (bits - 1)) | 1
        if _is_probable_prime(cand):
            return cand


class PaillierPIR:
    def __init__(self, bits: int = 256, seed: int = 7):
        random.seed(seed)
        p = _random_prime(bits)
        q = _random_prime(bits)
        while q == p:
            q = _random_prime(bits)
        self.n = p * q
        self.n2 = self.n * self.n
        self.g = self.n + 1
        lam = (p - 1) * (q - 1) // __import__("math").gcd(p - 1, q - 1)
        self.lam = lam
        self.mu = pow(lam, -1, self.n)

    def encrypt(self, m: int) -> int:
        r = random.randrange(1, self.n)
        return (pow(self.g, m, self.n2) * pow(r, self.n, self.n2)) % self.n2

    def decrypt(self, c: int) -> int:
        x = pow(c, self.lam, self.n2)
        return (((x - 1) // self.n) * self.mu) % self.n

    def query(self, index: int, size: int):
        """index 番目を取得するための暗号化選択ベクトル。"""
        return [self.encrypt(1 if i == index else 0) for i in range(size)]

    @staticmethod
    def answer_with_mod(items, encrypted_query, n2: int) -> int:
        """サーバー側: Π e_i^{item_i} = Enc(Σ e_i * item_i)。"""
        acc = 1
        for e, item in zip(encrypted_query, items):
            acc = (acc * pow(e, int(item), n2)) % n2
        return acc


def encode_text(text: str, n: int) -> int:
    data = text.encode("utf-8")
    if len(data) > n.bit_length() // 8 - 2:
        raise ValueError("message too long for this Paillier modulus")
    return int.from_bytes(data, "big")


def decode_text(value: int) -> str:
    length = (value.bit_length() + 7) // 8
    return value.to_bytes(length, "big").decode("utf-8")
