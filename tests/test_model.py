import pytest
import torch

import numpy as np

from model import (
    lorentz_expmap0,
    lorentz_logmap0,
    lorentz_dist,
    lorentz_cdist,
    lorentz_project,
    ContrastiveQueue,
    info_nce_loss,
    haversine_soft_ce,
    cross_view_loss,
    dequantize_t,
    AlphaBridge,
)

N_CELLS = [2, 4, 8]
PARENTS = [np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1, 2, 2, 3, 3])]
N_AUX = [2, 5, 3, 4]


def _bridge(head="linear"):
    return AlphaBridge(N_CELLS, PARENTS, N_AUX, head=head).eval()


# ── Lorentz geometry ─────────────────────────────────────────────────────────

def test_lorentz_project_satisfies_constraint():
    """Projected points must lie on the hyperboloid: -x0² + ‖x₁..ₙ‖² = -1."""
    v = torch.randn(8, 5)
    x = lorentz_project(v)
    inner = -x[:, 0].pow(2) + x[:, 1:].pow(2).sum(dim=-1)
    assert torch.allclose(inner, torch.full((8,), -1.0), atol=1e-4)


def test_expmap_logmap_roundtrip():
    """logmap0(expmap0(v)) should recover v for small tangent vectors."""
    v = torch.randn(4, 8) * 0.3
    x = lorentz_expmap0(v)
    v_rec = lorentz_logmap0(x)
    assert torch.allclose(v, v_rec, atol=1e-4)


def test_lorentz_dist_nonnegative():
    v1 = torch.randn(6, 16) * 0.5
    v2 = torch.randn(6, 16) * 0.5
    x1 = lorentz_expmap0(v1)
    x2 = lorentz_expmap0(v2)
    d = lorentz_dist(x1, x2)
    assert (d >= 0).all()


def test_lorentz_dist_self_zero():
    v = torch.randn(4, 8) * 0.5
    x = lorentz_expmap0(v)
    d = lorentz_dist(x, x)
    assert torch.allclose(d, torch.zeros(4), atol=1e-3)


def test_lorentz_dist_triangle_inequality():
    """d(a,c) ≤ d(a,b) + d(b,c) for random triples."""
    v = torch.randn(3, 12) * 0.3
    pts = lorentz_expmap0(v)
    d_ab = lorentz_dist(pts[0:1], pts[1:2])
    d_bc = lorentz_dist(pts[1:2], pts[2:3])
    d_ac = lorentz_dist(pts[0:1], pts[2:3])
    assert (d_ac <= d_ab + d_bc + 1e-4).all()


# ── ContrastiveQueue ──────────────────────────────────────────────────────────

def test_queue_empty_returns_tensor_not_none():
    """Empty queue must return an empty tensor, not None (guards info_nce_loss)."""
    q = ContrastiveQueue(dim=64, size=128)
    result = q.get("cpu")
    assert result is not None
    assert isinstance(result, torch.Tensor)
    assert result.shape == (0, 64)


def test_queue_partial_fill():
    q = ContrastiveQueue(dim=8, size=100)
    q.push(torch.randn(10, 8))
    result = q.get("cpu")
    assert result.shape == (10, 8)


def test_queue_full_returns_all():
    q = ContrastiveQueue(dim=8, size=16)
    for _ in range(3):          # overfill to trigger wrap-around
        q.push(torch.randn(8, 8))
    assert q.full
    result = q.get("cpu")
    assert result.shape == (16, 8)


def test_queue_wrap_around_fills_correctly():
    """After wrap-around, all queue slots should hold non-zero values."""
    q = ContrastiveQueue(dim=4, size=8)
    q.push(torch.ones(8, 4))
    q.push(torch.ones(6, 4) * 2)   # triggers wrap
    result = q.get("cpu")
    assert result.shape == (8, 4)
    assert (result != 0).all()


# ── info_nce_loss ─────────────────────────────────────────────────────────────

def test_info_nce_loss_scalar():
    pred   = torch.randn(4, 64)
    pred   = pred / pred.norm(dim=-1, keepdim=True)
    target = torch.randn(4, 64)
    target = target / target.norm(dim=-1, keepdim=True)
    loss = info_nce_loss(pred, target, temperature=0.07)
    assert loss.shape == ()
    assert loss.item() > 0


def test_info_nce_loss_empty_queue_same_as_no_queue():
    pred   = torch.randn(4, 64)
    pred   = pred / pred.norm(dim=-1, keepdim=True)
    target = torch.randn(4, 64)
    target = target / target.norm(dim=-1, keepdim=True)
    empty  = torch.empty(0, 64)
    loss_no_queue    = info_nce_loss(pred, target, 0.07, queue_negs=None)
    loss_empty_queue = info_nce_loss(pred, target, 0.07, queue_negs=empty)
    assert torch.isclose(loss_no_queue, loss_empty_queue)


def test_lorentz_cdist_matches_pairwise():
    x, y = lorentz_expmap0(torch.randn(3, 8) * 0.5), lorentz_expmap0(torch.randn(5, 8) * 0.5)
    ref = lorentz_dist(x[:, None], y[None])
    assert torch.allclose(lorentz_cdist(x, y), ref, atol=1e-4)


def test_soft_info_nce_matches_hard_as_tau_to_zero():
    pred = torch.nn.functional.normalize(torch.randn(6, 64), dim=-1)
    tgt = torch.nn.functional.normalize(torch.randn(6, 64), dim=-1)
    hard = info_nce_loss(pred, tgt, 0.07)
    soft = info_nce_loss(pred, tgt, 0.07, target_tau=1e-4)
    assert torch.isclose(hard, soft, atol=1e-4)


def test_soft_info_nce_exact_prediction_is_stationary_with_near_duplicates():
    """With near-duplicate targets, hard InfoNCE still pushes a perfect prediction away from the twin; soft (tau_t = tau) does not."""
    tgt = torch.nn.functional.normalize(torch.randn(3, 64), dim=-1)
    tgt[1] = torch.nn.functional.normalize(tgt[0] + 0.05 * torch.randn(64), dim=-1)
    grads = []
    for tau_t in (0.0, 0.07):
        pred = tgt.clone().requires_grad_(True)
        info_nce_loss(pred, tgt, 0.07, target_tau=tau_t).backward()
        grads.append(pred.grad.norm())
    assert grads[1] < 1e-4 < grads[0]


def test_info_nce_weight_masks_rows():
    pred = torch.nn.functional.normalize(torch.randn(4, 64), dim=-1)
    tgt = torch.nn.functional.normalize(torch.randn(4, 64), dim=-1)
    w = torch.tensor([1.0, 1.0, 0.0, 0.0])
    per_row = torch.nn.functional.cross_entropy(pred @ tgt.T / 0.07, torch.arange(4), reduction="none")
    assert torch.isclose(info_nce_loss(pred, tgt, 0.07, weight=w), per_row[:2].mean(), atol=1e-5)


def test_haversine_soft_ce_prefers_nearest_cell():
    cent = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], dtype=torch.float64)
    x = torch.tensor([[0.999, 0.0447, 0]], dtype=torch.float64)
    x = x / x.norm()
    near = haversine_soft_ce(torch.tensor([[5.0, -5.0]]), x, cent, 500.0)
    far = haversine_soft_ce(torch.tensor([[-5.0, 5.0]]), x, cent, 500.0)
    assert near < far


# ── AlphaBridge forward shapes ────────────────────────────────────────────────

@pytest.mark.parametrize("head", ["linear", "euclid", "lorentz"])
def test_alphabridge_output_shapes(head):
    with torch.no_grad():
        out = _bridge(head)(torch.randn(2, 768))
    assert [lg.shape for lg in out['cells']] == [(2, n) for n in N_CELLS]
    assert out['ae'].shape == (2, 64)
    assert [lg.shape for lg in out['aux_cat']] == [(2, n) for n in N_AUX]
    assert out['aux_reg'].shape == (2, 2)
    if head != "linear":
        assert all((lg <= 0).all() for lg in out['cells']), "distance logits must be non-positive"


def test_ae_head_is_unit_norm():
    """PhysicalHead applies L2 normalisation — AE predictions must be unit vectors (cosine = dot)."""
    with torch.no_grad():
        out = _bridge()(torch.randn(4, 768))
    assert torch.allclose(out['ae'].norm(dim=-1), torch.ones(4), atol=1e-5)


def test_tree_prototypes_nest_by_construction():
    head = _bridge("euclid").cells
    protos = head.prototypes()
    for l in (1, 2):
        par = getattr(head, f"parent_{l}")
        assert torch.allclose(protos[l] - protos[l - 1][par], head.delta[l])


def test_ema_handles_integer_buffers():
    from train import EMA
    model = AlphaBridge(N_CELLS, PARENTS, N_AUX, head="lorentz")
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.encoder.skip.weight.add_(1.0)
    ema.update(model)
    assert ema.shadow["cells.parent_1"].dtype == torch.long


# ── Location tower ────────────────────────────────────────────────────────────

def test_dequantize_t_matches_numpy_with_mask_as_zero():
    from geo_lookup import dequantize
    q = np.array([127, -127, 3, 0, -128], np.int8)
    assert np.allclose(dequantize_t(torch.as_tensor(q)).numpy(), np.nan_to_num(dequantize(q)))


def test_cross_view_loss_tiny_tau_is_symmetric_hard_infonce():
    from bins import to_xyz
    g = torch.Generator().manual_seed(0)
    u, v = (torch.nn.functional.normalize(torch.randn(8, 16, generator=g), dim=-1) for _ in range(2))
    xyz = torch.as_tensor(to_xyz(np.linspace(-60, 60, 8), np.linspace(-150, 150, 8)))
    lg, y = u @ v.T / 0.1, torch.arange(8)
    hard = 0.5 * (torch.nn.functional.cross_entropy(lg, y) + torch.nn.functional.cross_entropy(lg.T, y))
    assert torch.allclose(cross_view_loss(u, v, xyz, 0.1, 1e-3), hard, atol=1e-5)


@pytest.mark.parametrize("loc", ["ae", "gps", "ae+gps", "ae+pyr", "pyr"])
def test_location_tower_unit_embeddings_and_image_head(loc):
    from bins import to_xyz
    m = AlphaBridge(N_CELLS, PARENTS, N_AUX, loc=loc).eval()
    ctx = torch.randint(-128, 128, (4, 18, 64), dtype=torch.int8)
    v = m.tower(ctx, torch.as_tensor(to_xyz(np.zeros(4), np.arange(4.0))), torch.randn(4, 5, 128))
    assert v.shape == (4, 64) and torch.allclose(v.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert m(torch.randn(4, 768))["loc"].shape == (4, 64)


def test_pyramid_feats_masked_row_is_zero():
    from model import pyramid_feats
    M = torch.tensor([[127] * 64, [-128] * 64], dtype=torch.int8)
    H = torch.tensor([[255] * 64, [0] * 64], dtype=torch.uint8)
    f = pyramid_feats(M, H, torch.tensor([[0, 1]]))
    assert f.shape == (1, 2, 128) and (f[0, 1] == 0).all() and torch.allclose(f[0, 0, 64:], torch.ones(64))
