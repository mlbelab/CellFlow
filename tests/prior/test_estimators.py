import numpy as np
import pandas as pd
import pytest

from cellflow.prior import BaselineSimilarity, FixedPrior, TransferPrior, effect_stats, transfer_effects

HELDOUT = "A|y"
WEIGHTING = BaselineSimilarity(tau=0.3, n0=50.0)


def _estimator(**kw):
    return TransferPrior(
        "context", "perturbation", weighting=WEIGHTING, batch_key="batch", group_key="tissue", projection="PCs", **kw
    )


def _split(adata):
    train = adata[(adata.obs.context != HELDOUT).to_numpy()].copy()
    ctrl = adata[((adata.obs.context == HELDOUT) & adata.obs.is_control).to_numpy()].copy()
    cov = pd.DataFrame({"context": HELDOUT, "perturbation": ["d0", "d1", "d2"], "batch": "bA"})
    return train, ctrl, cov


def _reference(adata, pairs, scale=1.0):
    """Transferred effects from statistics of the full data with the held-out context's treated cells excluded."""
    st = effect_stats(
        adata,
        "context",
        "perturbation",
        "is_control",
        batch_key="batch",
        group_key="tissue",
        heldout_contexts=[HELDOUT],
    )
    return transfer_effects(st, WEIGHTING, pairs, scale=scale)


class TestTransferPrior:
    def test_fit_training_pairs(self, adata_prior):
        train, _, _ = _split(adata_prior)
        est = _estimator(scale=0.7)
        pv = est.fit(train, "is_control", "X_pca")
        obs = train.obs[~train.obs.is_control]
        pairs = set(zip(obs.context.astype(str), obs.perturbation.astype(str), strict=True))
        ref, w = _reference(adata_prior, sorted(pairs), scale=0.7)
        assert set(pv.values) == set(ref) and set(ref) <= pairs
        P = adata_prior.varm["PCs"]
        assert pv.values_full is None  # not kept for the training pairs
        for k, v in ref.items():
            assert pv.weights[k][0] == w[k][0] and k[0] not in w[k][0]  # leave-own-context-out
            np.testing.assert_allclose(pv.values[k], v @ P, rtol=1e-12)
        np.testing.assert_array_equal(est.projection_, P)

    @pytest.mark.parametrize("weighting", [BaselineSimilarity(k=1), BaselineSimilarity(tau=0.05, n0=5.0, k=2)])
    def test_weighting_variants(self, adata_prior, weighting):
        train, ctrl, cov = _split(adata_prior)
        est = TransferPrior("context", "perturbation", weighting=weighting, batch_key="batch", group_key="tissue")
        est.fit(train, "is_control", "X_pca")
        pv = est.predict(ctrl, cov, "is_control", "X_pca")
        st = effect_stats(
            adata_prior,
            "context",
            "perturbation",
            "is_control",
            batch_key="batch",
            group_key="tissue",
            heldout_contexts=[HELDOUT],
        )
        ref, w = transfer_effects(st, weighting, list(pv.values))
        assert set(ref) == set(pv.values) and all(len(w[k][0]) <= weighting.k for k in w)
        for k, v in ref.items():
            np.testing.assert_allclose(pv.values[k], v, rtol=1e-10)

    def test_default_estimator(self, adata_prior):
        """Without group_key or batch_key the default weighting works (no group preference)."""
        train, ctrl, cov = _split(adata_prior)
        pv = TransferPrior("context", "perturbation").fit(train, "is_control", "X")
        assert len(pv.values) > 0 and all(np.isfinite(v).all() for v in pv.values.values())

    def test_predict_heldout_from_controls(self, adata_prior):
        train, ctrl, cov = _split(adata_prior)
        est = _estimator()
        est.fit(train, "is_control", "X_pca")
        pv = est.predict(ctrl, cov, "is_control", "X_pca")
        ref, _ = _reference(adata_prior, [(HELDOUT, d) for d in cov.perturbation])
        assert set(pv.values) == set(ref) == {(HELDOUT, d) for d in cov.perturbation}
        for k, v in ref.items():
            np.testing.assert_allclose(pv.values_full[k], v, rtol=1e-10)
        # treated cells of the held-out context in the source data are ignored, even if poisoned
        a = adata_prior[(adata_prior.obs.context == HELDOUT).to_numpy()].copy()
        a.X[~a.obs.is_control.to_numpy()] += 100.0
        pv2 = est.predict(a, cov, "is_control", "X_pca")
        assert all(np.array_equal(pv2.values[k], v) for k, v in pv.values.items())

    def test_predict_needs_controls(self, adata_prior):
        train, ctrl, cov = _split(adata_prior)
        est = _estimator()
        est.fit(train, "is_control", "X_pca")
        with pytest.raises(ValueError, match="No control cells"):
            est.predict(ctrl[:0].copy(), cov, "is_control", "X_pca")

    def test_predict_checks(self, adata_prior):
        train, ctrl, cov = _split(adata_prior)
        est = _estimator()
        est.fit(train, "is_control", "X_pca")
        with pytest.raises(ValueError, match="features"):
            est.predict(ctrl[:, :5].copy(), cov, "is_control", "X_pca")
        with pytest.raises(KeyError, match="context"):
            est.predict(ctrl, cov.drop(columns="context"), "is_control", "X_pca")
        bad = ctrl.copy()
        bad.X = None
        with pytest.raises(ValueError, match="adata.X"):
            est.predict(bad, cov, "is_control", "X_pca")
        with pytest.raises(KeyError, match="varm"):
            TransferPrior("context", "perturbation", projection="nope").fit(train, "is_control", "X_pca")


def test_fixed_prior():
    vals = {("a", "d"): np.ones(3)}
    est = FixedPrior("context", "perturbation", vals)
    pv = est.fit(None, "is_control", "X")
    assert pv.values == vals and est.predict(None, None, "is_control", "X").values == vals
    df = pd.DataFrame({"context": ["a", "b"], "perturbation": ["d", "d"]})
    codes, table = pv.lookup(df, "context", "perturbation", 3)
    np.testing.assert_array_equal(table[codes], [[1, 1, 1], [0, 0, 0]])
    with pytest.raises(ValueError, match="dimension"):
        pv.lookup(df, "context", "perturbation", 4)
