import numpy as np
import pytest

from cellflow.prior import BaselineSimilarity, effect_stats, transfer_effects

KEYS = {"context_key": "context", "perturbation_key": "perturbation", "control_key": "is_control"}


def _stats(adata, **kw):
    return effect_stats(adata, batch_key="batch", group_key="tissue", **KEYS, **kw)


class TestEffectStats:
    def test_matches_direct_means(self, adata_prior):
        st = _stats(adata_prior)
        obs = adata_prior.obs
        X = adata_prior.X.astype(np.float64)
        assert st.contexts == ("A|x", "A|y", "A|z", "B|u", "B|v")
        assert st.perturbations == ("d0", "d1", "d2")
        assert list(st.groups) == ["A", "A", "A", "B", "B"]
        for i, c in enumerate(st.contexts):
            ctrl = (obs.context == c).to_numpy() & obs.is_control.to_numpy()
            np.testing.assert_allclose(st.control_mean_pooled[i], X[ctrl].mean(0), rtol=1e-12)
            for j, d in enumerate(st.perturbations):
                t = ((obs.context == c) & (obs.perturbation == d)).to_numpy()
                if not t.any():
                    assert st.n[i, j] == 0 and not st.valid[i, j] and (c, d) not in st.delta_index
                    continue
                b = obs.batch[t].iloc[0]
                cb = ctrl & (obs.batch == b).to_numpy()
                assert st.batch[i, j] == b and st.n[i, j] == t.sum() and st.n_control[i, j] == cb.sum()
                assert st.valid[i, j] == (t.sum() >= 30 and cb.sum() >= 20)
                if st.valid[i, j]:
                    np.testing.assert_allclose(st.effect(c, d), X[t].mean(0) - X[cb].mean(0), rtol=1e-10, atol=1e-12)
        assert not st.valid[st.contexts.index("B|v"), st.perturbations.index("d2")]  # 10 treated cells
        assert set(st.delta_index) == {
            (c, d)
            for c in st.contexts
            for d in st.perturbations
            if st.valid[st.contexts.index(c), st.perturbations.index(d)]
        }
        assert st.delta.shape == (len(st.delta_index), adata_prior.n_vars)

    def test_rep_x_and_errors(self, adata_prior):
        a, b = _stats(adata_prior), _stats(adata_prior, rep="X")
        np.testing.assert_array_equal(a.delta, b.delta)
        with pytest.raises(KeyError, match="nope"):
            effect_stats(adata_prior, batch_key="nope", **KEYS)
        with pytest.raises(KeyError, match="obsm"):
            _stats(adata_prior, rep="X_umap")
        a = adata_prior.copy()
        a.X = None
        with pytest.raises(ValueError, match="adata.X"):
            _stats(a)
        with pytest.raises(ValueError, match="No control cells"):
            _stats(adata_prior[~adata_prior.obs.is_control.to_numpy()].copy())

    def test_heldout_uses_controls_only(self, adata_prior):
        st = _stats(adata_prior, heldout_contexts=["A|y"])
        i = st.contexts.index("A|y")
        assert st.heldout_contexts == ("A|y",)
        assert (st.n[i] == 0).all() and not st.valid[i].any()
        np.testing.assert_allclose(st.control_mean_pooled[i], _stats(adata_prior).control_mean_pooled[i])

    def test_obsm_rep(self, adata_prior):
        st = _stats(adata_prior, rep="X_pca")
        assert st.delta.shape[-1] == adata_prior.obsm["X_pca"].shape[1]

    def test_projection_is_linear(self, adata_prior):
        """Effects projected from gene space equal effects computed directly in X_pca (same weights)."""
        P, pcs = adata_prior.varm["PCs"], adata_prior.obsm["X_pca"].astype(np.float64)
        a = adata_prior.copy()
        a.obsm["X_pca"] = (a.X.astype(np.float64) - a.X.astype(np.float64).mean(0)) @ P  # exact, float64
        sg, sp_ = _stats(a), _stats(a, rep="X_pca")
        pairs = [(c, d) for c in sg.contexts for d in sg.perturbations]
        eff, w = transfer_effects(sg, BaselineSimilarity(tau=0.3, n0=20.0), pairs)
        for k, (cand, wt) in w.items():
            direct = sum(x * sp_.effect(t, k[1]) for x, t in zip(wt, cand, strict=True))
            np.testing.assert_allclose(eff[k] @ P, direct, rtol=1e-9, atol=1e-12)
        assert np.abs(pcs - a.obsm["X_pca"]).max() < 1e-4  # the fixture's float32 X_pca is the same projection

    def test_errors(self, adata_prior):
        a = adata_prior.copy()
        a.obs["tissue"] = a.obs["tissue"].astype(str)
        a.obs.iloc[0, a.obs.columns.get_loc("tissue")] = "B"
        with pytest.raises(ValueError, match="constant within each context"):
            _stats(a)
        a = adata_prior.copy()
        a.obs["batch"] = a.obs["batch"].astype(str)
        row = np.flatnonzero(((a.obs.context == "A|y") & (a.obs.perturbation == "d0")).to_numpy())[0]
        a.obs.iloc[row, a.obs.columns.get_loc("batch")] = "other"
        with pytest.raises(ValueError, match="spans several batches"):
            _stats(a)


class TestBaselineSimilarity:
    def _reference(self, st, c, d, tau, n0, k, same_group):
        """Direct transcription of the weighting rule."""
        ci, j = {x: i for i, x in enumerate(st.contexts)}, st.perturbations.index(d)
        cand = [t for t in st.contexts if t != c and st.valid[ci[t], j]]
        same = [t for t in cand if st.groups[ci[t]] == st.groups[ci[c]]]
        if same_group and same:
            cand = same
        S = np.corrcoef(st.control_mean_pooled)
        cand = sorted(cand, key=lambda t: -S[ci[c], ci[t]])[:k]
        s = np.array([S[ci[c], ci[t]] for t in cand])
        w = np.exp(s / tau) / np.exp(s / tau).sum()
        if n0:
            n = np.array([st.n[ci[t], j] for t in cand])
            w = w * n / (n + n0) / (w * n / (n + n0)).sum()
        return cand, w

    @pytest.mark.parametrize(
        "tau,n0,k,same_group", [(0.2, 0.0, None, True), (0.1, 30.0, 2, True), (0.5, 0, None, False)]
    )
    def test_weights(self, adata_prior, tau, n0, k, same_group):
        st = _stats(adata_prior)
        weighting = BaselineSimilarity(tau=tau, n0=n0, k=k, same_group=same_group)
        pairs = [(c, d) for c in st.contexts for d in st.perturbations]
        out = weighting.weights(st, pairs)
        for c, d in pairs:
            cand, w = self._reference(st, c, d, tau, n0, k, same_group)
            if not cand:
                assert (c, d) not in out
                continue
            assert out[(c, d)][0] == cand and c not in cand
            np.testing.assert_allclose(out[(c, d)][1], w, rtol=1e-12)
            np.testing.assert_allclose(out[(c, d)][1].sum(), 1.0, rtol=1e-12)

    def test_group_fallback(self, adata_prior):
        st = _stats(adata_prior)
        out = BaselineSimilarity().weights(st, [("B|v", "d0"), ("A|x", "d1")])
        assert set(out[("B|v", "d0")][0]) == {"B|u"}  # same tissue preferred
        # d1 in tissue A is only observed in A|x and A|y: A|x transfers from A|y
        assert out[("A|x", "d1")][0] == ["A|y"]
        a = adata_prior[~((adata_prior.obs.tissue == "B") & ~adata_prior.obs.is_control).to_numpy()].copy()
        out = BaselineSimilarity().weights(_stats(a), [("B|v", "d0")])
        assert set(out[("B|v", "d0")][0]) == {"A|x", "A|y", "A|z"}  # no same-tissue candidate: fall back to all

    def test_errors(self, adata_prior):
        with pytest.raises(ValueError, match="tau"):
            BaselineSimilarity(tau=0.0).weights(_stats(adata_prior), [("A|x", "d0")])
        # without groups, same_group has no effect (the default weighting works without group_key)
        st = effect_stats(adata_prior, batch_key="batch", **KEYS)
        a, b = (
            BaselineSimilarity().weights(st, [("A|x", "d0")]),
            BaselineSimilarity(same_group=False).weights(st, [("A|x", "d0")]),
        )
        assert a[("A|x", "d0")][0] == b[("A|x", "d0")][0]
        np.testing.assert_array_equal(a[("A|x", "d0")][1], b[("A|x", "d0")][1])

    def test_non_finite_similarity(self, adata_prior):
        """A context without controls (nan pooled mean) or with a constant pooled mean gets no prior and is no
        neighbour; the other weights stay finite."""
        a = adata_prior[~((adata_prior.obs.context == "A|z") & adata_prior.obs.is_control).to_numpy()].copy()
        st = _stats(a)
        assert np.isnan(st.control_mean_pooled[st.contexts.index("A|z")]).all()
        pairs = [(c, d) for c in st.contexts for d in st.perturbations]
        out = BaselineSimilarity().weights(st, pairs)
        assert not any(k[0] == "A|z" or "A|z" in v[0] for k, v in out.items())
        assert all(np.isfinite(v[1]).all() for v in out.values()) and len(out) > 0
        b = adata_prior.copy()
        b.X[((b.obs.context == "B|u") & b.obs.is_control).to_numpy()] = 1.0  # constant pooled control mean
        out = BaselineSimilarity().weights(_stats(b), pairs)
        assert not any(k[0] == "B|u" or "B|u" in v[0] for k, v in out.items())
        assert all(np.isfinite(v[1]).all() for v in out.values())


class TestTransferEffects:
    def test_values(self, adata_prior):
        st = _stats(adata_prior)
        weighting = BaselineSimilarity(tau=0.3, n0=50.0)
        pairs = [(c, d) for c in st.contexts for d in st.perturbations]
        eff, weights = transfer_effects(st, weighting, pairs, scale=0.5)
        assert weights.keys() == eff.keys() and ("A|x", "d0") in eff
        for (c, d), (cand, w) in weighting.weights(st, pairs).items():
            assert weights[(c, d)][0] == cand
            ref = 0.5 * sum(wi * st.effect(t, d) for wi, t in zip(w, cand, strict=True))
            np.testing.assert_allclose(eff[(c, d)], ref, rtol=1e-12)

    def test_heldout_poisoning(self, adata_prior):
        """Held-out treated cells never enter; a training context's prior never uses its own treated cells."""
        weighting = BaselineSimilarity(n0=100.0)
        pairs = [(c, d) for c in ["A|x", "A|y", "A|z", "B|u", "B|v"] for d in ["d0", "d1", "d2"]]
        base, _ = transfer_effects(_stats(adata_prior, heldout_contexts=["A|y"]), weighting, pairs)

        def poisoned(c):
            a = adata_prior.copy()
            a.X[((a.obs.context == c) & ~a.obs.is_control).to_numpy()] += 100.0
            return transfer_effects(_stats(a, heldout_contexts=["A|y"]), weighting, pairs)[0]

        assert any(k[0] == "A|y" for k in base)
        p = poisoned("A|y")
        assert p.keys() == base.keys() and all(np.array_equal(p[k], v) for k, v in base.items())
        p = poisoned("A|z")
        assert all(np.array_equal(p[k], v) for k, v in base.items() if k[0] == "A|z")
        assert not all(np.array_equal(p[k], v) for k, v in base.items())
