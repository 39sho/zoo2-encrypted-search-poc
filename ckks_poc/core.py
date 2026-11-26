"""CKKS による暗号文内積（比較用の汎用 FHE 実装）。

1 暗号文に複数の文書をパッキングし、暗号化クエリとの積を計算したあと
ローテーション＋マスクで各文書スロットに内積を溜める（Lexicure の EMVP に
相当する演算を FHE で素直にやった版）。
"""
import time

from openfhe import (
    CCParamsCKKSRNS,
    GenCryptoContext,
    HEStd_128_classic,
    PKESchemeFeature,
)


def setup(batch: int, depth: int, scale_bits: int = 50):
    p = CCParamsCKKSRNS()
    p.SetMultiplicativeDepth(depth)
    p.SetScalingModSize(scale_bits)
    p.SetBatchSize(batch)
    p.SetSecurityLevel(HEStd_128_classic)
    cc = GenCryptoContext(p)
    for f in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH,
              PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE):
        cc.Enable(f)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    return cc, keys


def prepare_vectors(cc, keys, doc_vectors, query, V, batch):
    """文書ベクトルを batch スロットにパッキングして暗号化する。"""
    docs_per_ct = batch // V
    index_enc = []
    for start in range(0, len(doc_vectors), docs_per_ct):
        flat = [0.0] * batch
        for j, row in enumerate(doc_vectors[start : start + docs_per_ct]):
            flat[j * V : (j + 1) * V] = [float(x) for x in row]
        index_enc.append(cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(flat)))
    query_slots = [float(x) for x in query] * docs_per_ct
    query_enc = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(query_slots))
    return index_enc, query_enc, docs_per_ct


def encrypt_query(cc, keys, query, V, batch):
    docs_per_ct = batch // V
    query_slots = [float(x) for x in query] * docs_per_ct
    return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(query_slots))


def prepare_masks(cc, V, batch, steps):
    masks = {}
    for s in steps:
        masks[s] = cc.MakeCKKSPackedPlaintext(
            [1.0 if (i % V) < V - s else 0.0 for i in range(batch)]
        )
    return masks


def evaluate(cc, index_enc, query_enc, masks, steps):
    """サーバー側。暗号文のまま全ドキュメントのスコアを計算する。"""
    scores = []
    for ct in index_enc:
        acc = cc.EvalMult(ct, query_enc)
        for s in steps:
            shifted = cc.EvalRotate(acc, s)
            acc = cc.EvalAdd(acc, cc.EvalMult(shifted, masks[s]))
        scores.append(acc)
    return scores


def decrypt_scores(cc, sk, scores_enc, batch, V, docs_per_ct):
    out = []
    for ct in scores_enc:
        pt = cc.Decrypt(sk, ct)
        pt.SetLength(batch)
        vals = pt.GetRealPackedValue()
        out.extend(vals[j * V] for j in range(docs_per_ct))
    return out


def run(doc_vectors, query, V=64, batch=512, scale_bits=50, seeds=(1,)):
    """1 回の検索を実行して (scores, timings) を返す。"""
    steps = [1 << i for i in range(V.bit_length() - 1)]
    depth = 2 + len(steps)
    t = time.perf_counter()
    cc, keys = setup(batch, depth, scale_bits)
    keygen = time.perf_counter() - t
    cc.EvalRotateKeyGen(keys.secretKey, steps)

    t = time.perf_counter()
    index_enc, _, docs_per_ct = prepare_vectors(
        cc, keys, doc_vectors, query, V, batch
    )
    index_enc_time = time.perf_counter() - t

    masks = prepare_masks(cc, V, batch, steps)
    t = time.perf_counter()
    q_enc = encrypt_query(cc, keys, query, V, batch)
    q_enc_time = time.perf_counter() - t

    t = time.perf_counter()
    scores_enc = evaluate(cc, index_enc, q_enc, masks, steps)
    server = time.perf_counter() - t

    t = time.perf_counter()
    scores = decrypt_scores(cc, sk=keys.secretKey, scores_enc=scores_enc,
                            batch=batch, V=V, docs_per_ct=docs_per_ct)
    dec = time.perf_counter() - t
    return scores, {
        "keygen": keygen,
        "index_encrypt": index_enc_time,
        "query_encrypt": q_enc_time,
        "server": server,
        "decrypt": dec,
    }, len(index_enc)
