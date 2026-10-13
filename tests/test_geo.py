import math

import numpy as np
import pytest
import torch

from bins import ancestors, cell_stats, haversine, parent_map, to_latlon, to_xyz
from eval import distance_metrics, ece, geoscore, leaf_logp, refine, retrieve
from model import device


# ── Bins ──────────────────────────────────────────────────────────────────────

def test_parent_map_nested():
    fine, coarse = np.array([0, 1, 2, 3, 0]), np.array([0, 0, 1, 1, 0])
    assert parent_map(fine, coarse, 4).tolist() == [0, 0, 1, 1]


def test_parent_map_rejects_non_nested():
    with pytest.raises(ValueError, match="not nested"):
        parent_map(np.array([0, 0]), np.array([0, 1]), 1)


def test_ancestors_chain_parents():
    parents = [np.array([0, 0, 1]), np.array([0, 1, 1, 2])]  # level1 -> level0, level2 -> level1
    assert ancestors(parents, 4).tolist() == [[0, 0, 0], [0, 1, 1], [0, 1, 2], [1, 2, 3]]


def test_cell_stats_spherical_centroid():
    xyz = to_xyz(np.array([10.0, -10.0, 50.0]), np.array([20.0, 20.0, 0.0]))
    cent, cnt, spread = cell_stats(xyz, np.array([0, 0, 1]), 2)
    lat, lon = to_latlon(cent)
    assert cnt.tolist() == [2, 1]
    assert abs(lat[0]) < 1e-9 and abs(lon[0] - 20) < 1e-9
    assert abs(spread[0] - haversine(10, 20, 0, 20)) < 1e-6 and spread[1] < 1e-6


def test_haversine_paris_london():
    assert abs(haversine(48.8566, 2.3522, 51.5074, -0.1278) - 343.5) < 5


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_geoscore_bounds():
    assert geoscore(0) == 5000
    assert geoscore(1492.7) == pytest.approx(5000 / math.e)


def test_distance_metrics_thresholds():
    m = distance_metrics(np.array([0.5, 10, 100, 1000, 3000]))
    assert [m[f"acc_{t}km"] for t in (1, 25, 200, 750, 2500)] == [0.2, 0.4, 0.6, 0.6, 0.8]
    assert m["median_km"] == 100


def test_ece_perfectly_calibrated_is_zero():
    conf = np.array([0.95] * 20 + [0.25] * 20)
    corr = np.array([1] * 19 + [0] + [1] * 5 + [0] * 15, float)
    assert ece(conf, corr) == pytest.approx(0.0, abs=1e-9)


# ── Product-of-experts scoring ────────────────────────────────────────────────

def test_leaf_logp_is_normalized_hierarchical_sum():
    anc = torch.tensor([[0, 0], [0, 1], [1, 2]])
    lg = [torch.tensor([[2.0, 0.0]]), torch.tensor([[0.0, 1.0, 3.0]])]
    lp = leaf_logp(lg, anc)
    raw = torch.log_softmax(lg[0], -1)[:, anc[:, 0]] + torch.log_softmax(lg[1], -1)
    assert torch.allclose(lp.exp().sum(), torch.tensor(1.0))
    assert torch.allclose(lp, raw - torch.logsumexp(raw, -1, keepdim=True))
    assert torch.allclose(leaf_logp(lg, anc, alpha=0.0), torch.log_softmax(lg[1], -1))


def test_refine_keeps_cluster_and_falls_back():
    xyz = torch.as_tensor(to_xyz(np.array([48.8566, 48.86, 40.7128]), np.array([2.3522, 2.36, -74.006])),
                          dtype=torch.float32)
    hs = torch.tensor([[0.9, 0.88, 0.89], [-math.inf, -math.inf, -math.inf]])
    hi = torch.tensor([[0, 1, 2], [0, 0, 0]])
    lat, lon = refine(hs, hi, xyz, 25.0, (np.array([0.0, 7.0]), np.array([0.0, 8.0])))
    assert haversine(lat[0], lon[0], 48.8566, 2.3522) < 2
    assert (lat[1], lon[1]) == (7.0, 8.0)


def _toy_gallery():
    vec = torch.nn.functional.normalize(torch.eye(4)[:, :4].repeat(1, 16).float(), dim=-1)
    return {"leaf": torch.tensor([0, 0, 1, 1], device=device), "vec": {"ae": vec.half().to(device)}}, vec


def test_retrieve_prior_gates_gallery_and_serve_mode_agrees():
    gal, vec = _toy_gallery()
    qv = {"ae": vec[[2]].half()}  # identical to a point in leaf 1
    prior = (torch.tensor([[0]], dtype=torch.int32), torch.tensor([[0.0]]))  # only leaf 0 allowed
    for serve_k in (0, 1):
        hs, hi = retrieve(qv, prior, gal, {"ae": 10.0}, n_leaf=2, k=2, serve_k=serve_k)
        assert set(hi[0].tolist()) == {0, 1}, f"serve_k={serve_k} leaked a hit outside the allowed leaf"
    hs, hi = retrieve(qv, None, gal, {"ae": 1.0}, n_leaf=2, k=1)
    assert hi[0, 0].item() == 2


# ── AlphaEarth COG extraction (row mapping and pooling must be exact) ─────────

def test_dequantize_matches_aef_formula_and_masks():
    from geo_lookup import dequantize
    d = dequantize(np.array([127, -127, 0, -128], np.int8))
    assert np.allclose(d[:3], [(127 / 127.5) ** 2, -(127 / 127.5) ** 2, 0.0]) and np.isnan(d[3])


def test_pooled_sums_dequantized_blocks_then_normalizes():
    from geo_lookup import dequantize, pooled
    q = np.full((64, 4, 4), -128, np.int8)
    q[0, :2, :2], q[1, 0, 0] = 100, 50
    p = pooled(dequantize(q), 2)
    v = np.zeros(64); v[0], v[1] = 4 * (100 / 127.5) ** 2, (50 / 127.5) ** 2
    assert p.shape == (64, 2, 2) and np.allclose(p[:, 0, 0], v / np.linalg.norm(v))
    assert np.isnan(p[:, 1, 1]).all()


def test_assign_maps_points_to_containing_tile_in_their_utm_zone():
    pytest.importorskip("rasterio")
    import pandas as pd
    from geo_lookup import assign
    ix = pd.DataFrame({"crs": ["EPSG:32631", "EPSG:32631"], "utm_west": [400000, 450000], "utm_east": [450000, 500000],
                       "utm_south": [5300000, 5300000], "utm_north": [5500000, 5500000]})
    # Paris (zone 31N, x ~452 km) -> tile 1; Berlin (zone 33N) and Sydney (south) have no tile here
    f, x, _ = assign(np.array([48.8566, 52.52, -33.87]), np.array([2.3522, 13.405, 151.21]), ix)
    assert f.tolist() == [1, -1, -1] and 450000 < x[0] < 455000


def test_claim_refuses_mismatched_tile_dir(tmp_path):
    from geo_lookup import claim
    (tmp_path / "SIGNATURE").write_text("stale")
    with pytest.raises(RuntimeError, match="different points"):
        claim(str(tmp_path), "fresh")


def test_paired_deltas_are_b_minus_a(tmp_path, monkeypatch):
    import config
    from eval import paired
    monkeypatch.setattr(config, "RUNS_DIR", str(tmp_path))
    for run, d in (("a", np.full(200, 100.0)), ("b", np.full(200, 10.0))):
        (tmp_path / run).mkdir()
        np.savez(tmp_path / run / "test_dists.npz", B2=d)
    ci = paired(("a", "B2"), ("b", "B2"))
    assert ci["geoscore"][0] > 0 and ci["acc_25km"] == [1.0, 1.0] and ci["median_km"] == [-90.0, -90.0]


def test_decoupled_takes_cells_from_prior_and_ae_from_ae_model():
    from eval import Decoupled
    prior = lambda x: {"cells": ["p"], "ae": "prior-ae", "aux_cat": "p"}
    ae = lambda x: {"cells": ["a"], "ae": "ae-ae"}
    out = Decoupled.forward(type("D", (), {"prior": staticmethod(prior), "ae": staticmethod(ae)})(), None)
    assert out["cells"] == ["p"] and out["ae"] == "ae-ae" and out["aux_cat"] == "p"


def test_window_takes_3x3_and_masks_outside_raster():
    from geo_lookup import window
    a = np.zeros((64, 4, 4), np.int8)
    a[0] = np.arange(16).reshape(4, 4)
    w = window(a, np.array([1, 0]), np.array([1, 3]))
    assert w.shape == (2, 9, 64)
    assert w[0, :, 0].tolist() == [0, 1, 2, 4, 5, 6, 8, 9, 10]
    assert w[1, :, 0].tolist() == [-128, -128, -128, 2, 3, -128, 6, 7, -128]


def test_load_feats_aligns_rows_and_fuses_unit_parts(tmp_path, monkeypatch):
    import bins
    from bins import load_feats
    monkeypatch.setattr(bins, "clip_path", lambda split, b: str(tmp_path / f"{b}.npy"))
    monkeypatch.setattr(bins, "encoded_rows", lambda split, b: np.load(tmp_path / f"{b}_rows.npy"))
    np.save(tmp_path / "a.npy", np.array([[3, 4], [0, 2], [6, 8]], np.float16))
    np.save(tmp_path / "a_rows.npy", np.array([5, 1, 9]))
    np.save(tmp_path / "b.npy", np.array([[0, 0, 1], [2, 0, 0]], np.float16))
    np.save(tmp_path / "b_rows.npy", np.array([9, 5]))
    assert load_feats("train", np.array([5, 1, 9]), ("a",)).shape == (3, 2)  # whole backbone: memmap as is
    assert load_feats("train", np.array([9, 5]), ("a",)).tolist() == [[6, 8], [3, 4]]
    f = load_feats("train", np.array([9, 5]), ("a", "b"))
    assert np.allclose(f.astype(np.float32), [[0.6, 0.8, 0, 0, 1], [0.6, 0.8, 1, 0, 0]], atol=1e-3)


def test_mixed_prior_spans_coarse_only_to_leaf_only(monkeypatch):
    import eval as ev
    # 4 leaves under 2 coarse cells; gallery points sit in leaves [0, 2, 2, 3] -> coarse cells [0, 1, 1, 1]
    env = type("E", (), {"anc": np.array([[0, 0], [0, 1], [1, 2], [1, 3]]), "n_leaf": 4, "cells": {"n_cells": np.array([2, 4])}})()
    gal = {"leaf": torch.tensor([0, 2, 2, 3]), "vec": {"loc": torch.eye(4, dtype=torch.float16)}}
    q = {"lp0": torch.log(torch.tensor([[0.5, 0.5]])).half(), "top_idx": torch.tensor([[2, 0]]),
         "top_lp": torch.log(torch.tensor([[0.7, 0.2]]))}
    qv = {"loc": torch.zeros(1, 4, dtype=torch.float16)}
    monkeypatch.setattr(ev, "device", "cpu")
    score = lambda beta: ev.method_hits("HT2", q, gal, qv, {"kappa": 1.0, "level": 0, "beta": beta}, env)
    hs, hi = score(0.0)  # equal coarse mass spread over 1 vs 3 points: the lone cell-0 point wins
    assert hi[0, 0] == 0 and torch.allclose(hs[0, :4], torch.log(torch.tensor([0.5, 0.5 / 3, 0.5 / 3, 0.5 / 3])), atol=1e-3)
    hs, hi = score(1.0)  # leaf prior / leaf count; leaf 3 falls outside top-M and takes the floor (0.2)
    assert hi[0, 0] in (1, 2) and torch.allclose(hs[0, :4], torch.log(torch.tensor([0.35, 0.35, 0.2, 0.2])), atol=1e-3)


def test_retrieve_scores_missing_feature_as_zero_and_applies_offset(monkeypatch):
    import eval as ev
    monkeypatch.setattr(ev, "device", "cpu")
    # 2 photos with clip, 2 map points without; all have loc; map points carry the flag
    gal = {"leaf": torch.zeros(4, dtype=torch.long), "vec": {
        "clip": torch.tensor([[1.0, 0.0], [0.0, 1.0]]).half(), "loc": torch.ones(4, 1).half(),
        "flag": torch.tensor([[0.0], [0.0], [1.0], [1.0]]).half()}}
    qv = {"clip": torch.tensor([[1.0, 0.0]]).half(), "loc": torch.ones(1, 1).half(), "flag": torch.ones(1, 1).half()}
    hs, hi = ev.retrieve(qv, None, gal, {"clip": 2.0, "loc": 1.0, "flag": 1.5}, 1, k=4, gb=3)
    assert hi[0].tolist()[:1] == [0] and torch.allclose(hs[0], torch.tensor([3.0, 2.5, 2.5, 1.0]))


def test_cell_partials_sum_vectors_and_count_codes_per_cell(monkeypatch):
    import config
    from bins import cell_partials, qt_key
    monkeypatch.setattr(config, "PYR_DEPTHS", (8,))
    lat, lon = np.array([10.0, 10.01, -40.0]), np.array([20.0, 20.01, 150.0])
    V, codes = np.eye(3, 64, dtype=np.float32), np.array([1, 1, 0])
    (keys, sums, hist), = cell_partials(lat, lon, V, codes, 2)
    assert keys.tolist() == sorted(set(qt_key(lat, lon, 8).tolist())) and len(keys) == 2
    same = keys == qt_key(10.0, 20.0, 8)
    assert np.allclose(sums[same][0, :2], [1, 1]) and hist[same][0].tolist() == [0, 2] and hist[~same][0].tolist() == [1, 0]


def test_pyramid_index_stacks_depths_and_maps_missing_to_masked_row(tmp_path):
    from bins import Pyramid, qt_key
    k8, k9 = np.sort(qt_key(np.array([10.0, -40.0]), np.array([20.0, 150.0]), 8)), qt_key(np.array([10.0]), np.array([20.0]), 9)
    np.savez(tmp_path / "p.npz", depths=np.array([8, 9]), k8=k8, k9=k9, m8=np.ones((2, 64), np.int8), m9=np.ones((1, 64), np.int8),
             h8=np.ones((2, 64), np.uint8), h9=np.ones((1, 64), np.uint8))
    P = Pyramid(str(tmp_path / "p.npz"))
    idx = P.index(np.array([10.0, -40.0, 60.0]), np.array([20.0, 150.0, 0.0]))
    assert P.M.shape == (4, 64) and (P.M[3] == -128).all()
    assert idx.tolist() == [[int(np.flatnonzero(k8 == qt_key(10.0, 20.0, 8))[0]), 2],
                            [int(np.flatnonzero(k8 == qt_key(-40.0, 150.0, 8))[0]), 3], [3, 3]]


def test_fuse_projects_shared_rows_with_train_fit_pca(tmp_path, monkeypatch):
    import bins, config
    rng = np.random.default_rng(0)
    paths = {("train", "streetclip"): "sc", ("train", "siglip2"): "sg", ("train", "fusion2"): "fu"}
    monkeypatch.setattr(config, "FEAT_DIR", str(tmp_path))
    monkeypatch.setattr(bins, "_root", lambda split: str(tmp_path))
    monkeypatch.setattr(bins, "clip_path", lambda split, b: str(tmp_path / f"{paths[split, b]}.npy"))
    monkeypatch.setattr(bins, "encoded_rows", lambda split, b: np.load(tmp_path / f"{paths[split, b]}_rows.npy"))
    np.save(tmp_path / "sc.npy", rng.normal(size=(6, 2)).astype(np.float16)); np.save(tmp_path / "sc_rows.npy", np.arange(6))
    np.save(tmp_path / "sg.npy", rng.normal(size=(5, 3)).astype(np.float16)); np.save(tmp_path / "sg_rows.npy", np.array([5, 4, 3, 2, 1]))
    bins.fuse(["train"], dim=2, n_fit=10)
    z, out = np.load(tmp_path / "fusion2_pca.npz"), np.load(tmp_path / "fu.npy")
    rows = np.load(tmp_path / "fusion2_train_rows.npy")
    want = (bins.load_feats("train", rows, ("streetclip", "siglip2")).astype(np.float32) - z["mean"]) @ z["W"]
    assert rows.tolist() == [1, 2, 3, 4, 5] and out.shape == (5, 2) and np.allclose(out, want, atol=1e-2)
    assert np.allclose(z["W"].T @ z["W"], np.eye(2), atol=1e-5)


def test_quantize_inverts_dequantize_and_keeps_mask():
    from geo_lookup import dequantize, quantize
    q = np.array([-127, -64, -1, 0, 1, 64, 127, -128], np.int8)
    assert (quantize(dequantize(q)) == q).all()


# ── Novel-location split ──────────────────────────────────────────────────────

def test_quadtree_leaves_nest_and_respect_threshold():
    from bins import qt_center, qt_counts, qt_leaf, QT_DEPTH
    lat = np.r_[np.full(6, 10.0) + np.arange(6) * 1e-3, -40.0]
    lon = np.r_[np.full(6, 20.0) + np.arange(6) * 1e-3, -100.0]
    counts = qt_counts(lat, lon)
    coarse, fine = qt_leaf(lat, lon, counts, 6), qt_leaf(lat, lon, counts, 2)
    d_c, d_f = coarse // 4 ** QT_DEPTH, fine // 4 ** QT_DEPTH
    assert (d_f >= d_c).all() and d_c[0] == 1 and d_f[0] > 1
    # a fine leaf's ancestor at the coarse depth is the coarse leaf; centers lie inside their quadrant
    k = fine % 4 ** QT_DEPTH
    shift = (d_f - d_c)
    n_f = np.left_shift(1, d_f)
    anc = (k // n_f >> shift) * np.left_shift(1, d_c) + ((k % n_f) >> shift)
    assert (anc == coarse % 4 ** QT_DEPTH).all()
    clat, clon = qt_center(fine[:1])
    assert abs(clat[0] - 10) < 180 / 2 ** d_f[0] and abs(clon[0] - 20) < 360 / 2 ** d_f[0]


def test_block_ids_are_roughly_equal_area():
    from bins import block_ids
    lon = np.linspace(-179.99, 179.99, 20000)
    eq, hi = (len(np.unique(block_ids(np.full_like(lon, la), lon))) for la in (0.1, 60.0))
    assert 390 <= eq <= 410 and 190 <= hi <= 210


def test_pick_blocks_holds_out_a_fraction_per_continent():
    from bins import HOLD_FRAC, pick_blocks
    blk, cont = np.arange(1000), np.repeat([0, 1], 500)
    held, tune = pick_blocks(blk, cont)
    assert all(abs((cont[held] == c).sum() - HOLD_FRAC * 500) <= 1 for c in (0, 1)) and np.isin(tune, held).all()
