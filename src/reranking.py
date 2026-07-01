"""K-reciprocal re-ranking for person search (Zhong et al., CVPR 2017).

Applied at *gallery-image level*: one score-weighted aggregate feature per
gallery image is used for the re-ranking step (memory: ~150 MB for PRW's
6,112 gallery images).  Within each image the original per-detection cosine
similarity is retained for the true-positive selection step.

Reference:
    Zhong et al. "Re-ranking Person Re-identification with k-reciprocal
    Encoding", CVPR 2017. https://arxiv.org/abs/1701.08398
"""

import numpy as np
from sklearn.metrics import average_precision_score


def _k_reciprocal_neigh(sorted_nn, i, k):
    """Reciprocal k-nearest neighbors of i (indices into sorted_nn[i])."""
    forward = sorted_nn[i, 1:k + 1]   # skip self (dist=0 at col 0)
    # Backward: which of forward also have i in their k-NN?
    mask = np.isin(sorted_nn[forward, 1:k + 1], [i]).any(axis=1)
    return forward[mask]


def k_reciprocal_rerank(q_feat, g_feat, k1=20, k2=6, lam=0.3):
    """K-reciprocal re-ranking.

    Args:
        q_feat: [n_q, D] float32 L2-normalised query features.
        g_feat: [n_g, D] float32 L2-normalised gallery features.
        k1: size of the k-reciprocal neighbourhood.
        k2: local query expansion neighbourhood size.
        lam: weight of the original cosine distance in the final distance.

    Returns:
        dist_final: [n_q, n_g] float32 re-ranked distance matrix.
    """
    n_q, D = q_feat.shape
    n_g = len(g_feat)
    n = n_q + n_g

    # ── Pairwise cosine distances ──────────────────────────────────────────
    feat = np.concatenate([q_feat, g_feat], axis=0).astype(np.float32)
    sim  = feat @ feat.T                          # [n, n]  cosine similarity
    dist = (1.0 - sim).clip(0, 2).astype(np.float16)   # [n, n]  float16

    # Pre-sort each row by distance once (ascending)
    sorted_nn = np.argsort(dist, axis=1).astype(np.int32)   # [n, n]

    # ── Build weighted Jaccard vectors V ───────────────────────────────────
    k1_half = max(1, int(np.ceil(k1 / 2)))
    V = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        R = _k_reciprocal_neigh(sorted_nn, i, k1)

        # Expand: add reciprocal neighbours of R members
        R_exp = list(R)
        for r in R:
            R_r = _k_reciprocal_neigh(sorted_nn, r, k1_half)
            if len(R_r) > 0 and len(np.intersect1d(R_r, R)) / len(R_r) >= 2 / 3:
                R_exp.extend(R_r)
        R_exp = np.unique(R_exp).astype(np.int32)
        if len(R_exp) == 0:
            continue

        # Gaussian weights proportional to similarity
        weights = np.exp(-dist[i, R_exp].astype(np.float32))
        weights /= weights.sum() + 1e-9
        V[i, R_exp] = weights

    # ── Local query expansion with k2 ─────────────────────────────────────
    V_qe = np.empty_like(V)
    for i in range(n):
        nn_k2 = sorted_nn[i, 1:k2 + 1]
        V_qe[i] = V[[i, *nn_k2]].mean(axis=0)

    # ── Jaccard distance via inverted index (sparse-aware, low memory) ───────
    # V_qe is effectively sparse: each row has at most ~k1 + k1_half*k1 ≈ 50
    # non-zero entries out of n=n_q+n_g.  Materialising [n_q, n_g, n] would
    # require tens of GB, so we use an inverted index instead.
    #
    # For each column k, inv_idx[k] = rows i where V_qe[i, k] > 0.
    # Then for each query i, min_sum[i, j-n_q] = sum_k min(V_qe[i,k], V_qe[j,k])
    # is accumulated by iterating over the non-zero columns of query i and
    # looking up the matching gallery rows in inv_idx[k].

    # Build inverted index column → rows with non-zero weight
    inv_idx = [None] * n
    for k in range(n):
        rows = np.flatnonzero(V_qe[:, k])
        inv_idx[k] = rows if len(rows) else np.empty(0, dtype=np.int32)

    jaccard = np.zeros((n_q, n_g), dtype=np.float32)
    for i in range(n_q):
        nz_k = np.flatnonzero(V_qe[i])    # non-zero columns for this query
        temp  = np.zeros(n_g, dtype=np.float32)
        for k in nz_k:
            v_i_k   = V_qe[i, k]
            j_full  = inv_idx[k]           # all rows with non-zero V_qe[j, k]
            gal_mask = j_full >= n_q
            if not gal_mask.any():
                continue
            j_gal   = j_full[gal_mask]
            v_j_k   = V_qe[j_gal, k]
            temp[j_gal - n_q] += np.minimum(v_i_k, v_j_k)
        jaccard[i] = 1.0 - temp / (2.0 - temp + 1e-6)

    orig_dist = (1.0 - sim[:n_q, n_q:].astype(np.float32)).clip(0, 2)
    dist_final = (1.0 - lam) * jaccard + lam * orig_dist
    return dist_final


def eval_search_prw_reranked(
    gallery_dataset,
    query_dataset,
    gallery_dets,
    gallery_feats_raw,
    query_feats,
    det_thresh=0.5,
    ignore_cam_id=True,
    k1=20,
    k2=6,
    lam=0.3,
):
    """PRW person-search evaluation with k-reciprocal re-ranking.

    Re-ranking is applied at **gallery-image level**: one score-weighted
    aggregate embedding per gallery image is used to compute re-ranked
    distances; within each image the original per-detection cosine similarity
    determines which detection is selected as the true-positive candidate.

    The final score for each detection is:
        score = (1 - lam) * (1 - reranked_dist[query, image])
              + lam * cosine_sim(det_feat, query_feat)

    Args and return value mirror ``eval_search_prw``.
    """
    annos = gallery_dataset.annotations

    # ── Filter detections and build per-image aggregate features ──────────
    name_to_info = {}   # img_name -> (det [n,5], feat [n,D], agg_feat [D])
    img_names_ordered = []
    for anno, det, feat in zip(annos, gallery_dets, gallery_feats_raw):
        name = anno["img_name"]
        scores = det[:, 4]
        inds = np.where(scores >= det_thresh)[0]
        if len(inds) == 0:
            continue
        d, f = det[inds], feat[inds].astype(np.float32)
        # Score-weighted aggregate feature (CWS-style), then L2-normalise
        w = scores[inds].astype(np.float32)
        agg = (f * w[:, None]).sum(axis=0)
        norm = np.linalg.norm(agg)
        agg = agg / norm if norm > 1e-9 else agg
        name_to_info[name] = (d, f, agg)
        img_names_ordered.append(name)

    # Stack aggregate gallery features and query features for re-ranking
    g_names  = list(name_to_info.keys())
    g_agg    = np.stack([name_to_info[n][2] for n in g_names])   # [n_g, D]
    q_feats  = np.stack([qf.ravel().astype(np.float32) for qf in query_feats])  # [n_q, D]

    print(f"  re-ranking {len(q_feats)} queries × {len(g_agg)} gallery images "
          f"(k1={k1}, k2={k2}, λ={lam}) …")
    dist_rr = k_reciprocal_rerank(q_feats, g_agg, k1=k1, k2=k2, lam=lam)  # [n_q, n_g]
    g_name_to_idx = {n: i for i, n in enumerate(g_names)}

    # ── Evaluation loop (mirrors eval_search_prw) ─────────────────────────
    def _iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / union if union > 0 else 0.0

    aps, accs = [], []
    ret = {"image_root": gallery_dataset.img_prefix, "results": []}

    for qi in range(len(query_dataset)):
        feat_p = q_feats[qi]
        q_anno = query_dataset.annotations[qi]
        q_name = q_anno["img_name"]
        q_pid  = q_anno["pids"]
        q_cam  = q_anno["cam_id"]

        # Ground-truth locations for this query
        query_gts = {}
        for x in annos:
            if q_pid in x["pids"] and x["img_name"] != q_name:
                query_gts[x["img_name"]] = x["boxes"][x["pids"] == q_pid]

        # Gallery scope
        if ignore_cam_id:
            gallery_imgs = [x for x in annos if x["img_name"] != q_name]
        else:
            gallery_imgs = [x for x in annos
                            if x["img_name"] != q_name and x["cam_id"] != q_cam]

        y_true, y_score = [], []
        imgs, rois = [], []
        count_gt = count_tp = 0
        seen = set()

        for item in gallery_imgs:
            gname = item["img_name"]
            count_gt += gname in query_gts
            if gname not in name_to_info or gname in seen:
                continue
            seen.add(gname)

            det, feat_g, _ = name_to_info[gname]
            g_idx = g_name_to_idx[gname]

            # Re-ranked image-level similarity
            rr_sim = float(1.0 - dist_rr[qi, g_idx])
            # Per-detection cosine similarity
            cos_sim = feat_g.dot(feat_p)   # [n_det]
            # Combined score
            combined = (1.0 - lam) * rr_sim + lam * cos_sim

            label = np.zeros(len(combined), dtype=np.int32)
            if gname in query_gts:
                gt = query_gts[gname].ravel()
                w, h = gt[2] - gt[0], gt[3] - gt[1]
                iou_thr = min(0.5, (w * h) / ((w + 10) * (h + 10)))
                order = np.argsort(combined)[::-1]
                for j in order:
                    if _iou(det[j, :4], gt) >= iou_thr:
                        label[j] = 1
                        count_tp += 1
                        break

            y_true.extend(list(label))
            y_score.extend(list(combined))
            imgs.extend([gname] * len(combined))
            rois.extend(list(det))

        y_score = np.asarray(y_score)
        y_true  = np.asarray(y_true)
        recall_rate = count_tp / max(count_gt, 1)
        ap = 0.0 if count_tp == 0 else average_precision_score(y_true, y_score) * recall_rate
        aps.append(ap)

        inds = np.argsort(y_score)[::-1]
        accs.append(int(y_true[inds[:1]].sum() > 0))

        new_entry = {
            "query_img": q_name,
            "query_roi": list(map(float, q_anno["boxes"].squeeze())),
            "query_gt":  query_gts,
            "gallery":   [],
        }
        for k in range(min(10, len(inds))):
            new_entry["gallery"].append({
                "img":     str(imgs[inds[k]]),
                "roi":     list(map(float, rois[inds[k]])),
                "score":   float(y_score[inds[k]]),
                "correct": int(y_true[inds[k]]),
            })
        ret["results"].append(new_entry)

    mAP  = float(np.mean(aps))
    top1 = float(np.mean(accs))
    print(f"search ranking (re-ranked):")
    print(f"  mAP   = {mAP:.2%}")
    print(f"  top-1 = {top1:.2%}")
    ret["mAP"]  = mAP
    ret["accs"] = np.array([top1])
    return ret
