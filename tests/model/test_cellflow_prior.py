import jax
import numpy as np
import optax
import pytest

import cellflow
from cellflow.prior import BaselineSimilarity, FixedPrior, TransferPrior

HELDOUT = "A|y"


def _transfer():
    return TransferPrior(
        "context",
        "perturbation",
        weighting=BaselineSimilarity(n0=50.0),
        batch_key="batch",
        group_key="tissue",
        projection="PCs",
    )


def _fit(adata, prior=None, heldout=HELDOUT, solver="otfm", split=("context", "batch"), train=True, **model_kw):
    train_data = adata[(adata.obs.context != heldout).to_numpy()].copy()
    cf = cellflow.model.CellFlow(train_data, solver=solver)
    cf.prepare_data(
        sample_rep="X_pca",
        control_key="is_control",
        perturbation_covariates={"drug": ["perturbation"]},
        perturbation_covariate_reps={"drug": "drug_emb"},
        split_covariates=list(split) if split else None,
        prior=prior,
    )
    if solver == "genot":
        model_kw["vf_kwargs"] = {"genot_source_dims": (4, 4), "genot_source_dropout": 0.0}
    cf.prepare_model(
        condition_embedding_dim=2, hidden_dims=(4, 4), decoder_dims=(4, 4), optimizer=optax.adam(1e-3), **model_kw
    )
    if train:
        cf.train(num_iterations=3)
    return cf, train_data


def _heldout_inputs(adata, heldout=HELDOUT):
    obs = adata.obs
    ctrl = adata[((obs.context == heldout) & obs.is_control).to_numpy()].copy()
    t = obs[(obs.context == heldout) & ~obs.is_control]
    cov = t[["perturbation", "batch"]].astype(str).drop_duplicates().reset_index(drop=True)
    cov["context"], cov["is_control"] = heldout, False
    for k in ["perturbation", "context", "batch"]:
        cov[k] = cov[k].astype("category")
    cov["cond_id"] = [f"{heldout}@@{d}" for d in cov.perturbation]
    return ctrl, cov


def _predict(cf, adata, heldout=HELDOUT, **kw):
    ctrl, cov = _heldout_inputs(adata, heldout)
    return cf.predict(ctrl, covariate_data=cov, condition_id_key="cond_id", max_steps=3, throw=False, **kw)


def _without_prior(cf, adata, **kw):
    prior, cf.prior = cf.prior, None
    try:
        return _predict(cf, adata, **kw)
    finally:
        cf.prior = prior


class TestCellFlowPrior:
    def test_prepare_data_shift(self, adata_prior):
        cf, train = _fit(adata_prior, prior=_transfer())
        assert isinstance(cf.prior, TransferPrior) and cf.prior.stats_.contexts == ("A|x", "A|z", "B|u", "B|v")
        vals = cf.prior_train.values
        Z = np.asarray(train.obsm["X_pca"])
        np.testing.assert_array_equal(Z, adata_prior[train.obs_names].obsm["X_pca"])  # adata untouched
        got = np.asarray(cf.train_data.cell_data)
        obs = zip(train.obs.context, train.obs.perturbation, train.obs.is_control, strict=True)
        for i, (c, d, ctrl) in enumerate(obs):
            want = Z[i] if ctrl or (c, d) not in vals else Z[i] - vals[(c, d)].astype(np.float32)
            np.testing.assert_array_equal(got[i], want)
        assert sum(not c for c in train.obs.is_control) > 0 and len(vals) > 0

    def test_zero_prior_is_identity(self, adata_prior):
        zero = FixedPrior("context", "perturbation", {("A|x", "d0"): np.zeros(5), (HELDOUT, "d1"): np.zeros(5)})
        cf0, _ = _fit(adata_prior)
        cf1, _ = _fit(adata_prior, prior=zero)
        assert cf0.prior is None and cf0.prior_train is None
        np.testing.assert_array_equal(np.asarray(cf0.train_data.cell_data), np.asarray(cf1.train_data.cell_data))
        same = jax.tree.map(lambda a, b: bool((a == b).all()), cf0.solver.vf_state.params, cf1.solver.vf_state.params)
        assert jax.tree_util.tree_all(same)
        p0, p1 = _predict(cf0, adata_prior), _predict(cf1, adata_prior)
        assert cf0.prior_predict is None
        assert all(np.array_equal(p0[k], p1[k]) for k in p0)

    def test_predict_adds_prior(self, adata_prior):
        cf, _ = _fit(adata_prior, prior=_transfer())
        plain, shifted = _without_prior(cf, adata_prior), _predict(cf, adata_prior)
        ctrl, cov = _heldout_inputs(adata_prior)
        ref = cf.prior.predict(ctrl, cov, "is_control", "X_pca")
        assert cf.prior_predict.values.keys() == ref.values.keys() == {tuple(k.split("@@")) for k in plain}
        for k in plain:
            add = ref.values[tuple(k.split("@@"))].astype(np.float32)
            np.testing.assert_array_equal(np.asarray(shifted[k]), np.asarray(plain[k]) + add)

    def test_predict_prior_needs_condition_id(self, adata_prior):
        cf, _ = _fit(adata_prior, prior=_transfer())
        ctrl, cov = _heldout_inputs(adata_prior)
        with pytest.raises(ValueError, match="condition_id_key"):
            cf.predict(ctrl, covariate_data=cov, max_steps=3, throw=False)

    def test_prior_dim_mismatch(self, adata_prior):
        with pytest.raises(ValueError, match="dimension"):
            _fit(adata_prior, prior=FixedPrior("context", "perturbation", {("A|x", "d0"): np.zeros(3)}))

    def test_save_load(self, adata_prior, tmp_path):
        cf, _ = _fit(adata_prior, prior=_transfer())
        before = _predict(cf, adata_prior)
        cf.save(str(tmp_path))
        cf2 = cellflow.model.CellFlow.load(str(tmp_path / "CellFlow.pkl"))
        assert isinstance(cf2.prior, TransferPrior) and cf2.prior.stats_.contexts == cf.prior.stats_.contexts
        after = _predict(cf2, adata_prior)
        assert all(np.array_equal(before[k], after[k]) for k in before)

    def test_estimator_is_copied(self, adata_prior):
        est = _transfer()
        cf1, _ = _fit(adata_prior, prior=est, train=False)
        cf2, _ = _fit(adata_prior, prior=est, heldout="B|u", train=False)
        assert not hasattr(est, "stats_")  # the passed estimator is not fitted or shared
        assert cf1.prior is not est and cf2.prior is not cf1.prior
        assert "B|u" in cf1.prior.stats_.contexts and "B|u" not in cf2.prior.stats_.contexts

    def test_one_pair_per_condition(self, adata_prior):
        with pytest.raises(ValueError, match="share one model condition"):
            _fit(adata_prior, prior=_transfer(), split=None, train=False)

    def test_prepare_data_errors(self, adata_prior):
        with pytest.raises(TypeError, match="prior estimator"):
            _fit(adata_prior, prior=BaselineSimilarity(), train=False)
        with pytest.raises(KeyError, match="nope"):
            _fit(adata_prior, prior=TransferPrior("nope", "perturbation"), train=False)
        cf = cellflow.model.CellFlow(adata_prior, solver="otfm")
        with pytest.raises(ValueError, match="sample_rep"):
            cf.prepare_data(
                sample_rep={"obsm": "X_pca"},
                control_key="is_control",
                perturbation_covariates={"drug": ["perturbation"]},
                perturbation_covariate_reps={"drug": "drug_emb"},
                prior=_transfer(),
            )

    def test_predict_column_errors(self, adata_prior):
        cf, _ = _fit(adata_prior, prior=_transfer(), train=False)
        cf.solver.is_trained = True  # only the argument checks run before the solver
        ctrl, cov = _heldout_inputs(adata_prior)
        with pytest.raises(KeyError, match="context"):
            cf.predict(ctrl, covariate_data=cov.drop(columns="context"), condition_id_key="cond_id")

    def test_validation_data_shifted(self, adata_prior):
        cf, _ = _fit(adata_prior, prior=_transfer(), train=False)
        val = adata_prior[(adata_prior.obs.context == HELDOUT).to_numpy()].copy()
        cf.prepare_validation_data(val, name="val")
        got = np.asarray(cf.validation_data["val"].cell_data)
        pv = cf.prior.predict(val, val.obs[~val.obs.is_control], "is_control", "X_pca")
        Z = val.obsm["X_pca"]
        for i, (c, d, ctrl) in enumerate(zip(val.obs.context, val.obs.perturbation, val.obs.is_control, strict=True)):
            want = Z[i] if ctrl else Z[i] - pv.values[(c, d)].astype(np.float32)
            np.testing.assert_array_equal(got[i], want)
        cf.train(num_iterations=3, valid_freq=2)
        assert cf.solver.is_trained

    def test_validation_priors_ignore_treated_cells(self, adata_prior):
        """Validation priors of an unseen context come from its controls only (no leakage of its treated cells)."""
        cf, _ = _fit(adata_prior, prior=_transfer(), train=False)
        val = adata_prior[(adata_prior.obs.context == HELDOUT).to_numpy()].copy()
        treated = ~val.obs.is_control.to_numpy()
        cf.prepare_validation_data(val, name="val")
        base = np.asarray(cf.validation_data["val"].cell_data)
        poisoned = val.copy()
        poisoned.X[treated] += 100.0
        cf.prepare_validation_data(poisoned, name="val")
        np.testing.assert_array_equal(np.asarray(cf.validation_data["val"].cell_data), base)

    def test_genot(self, adata_prior):
        """GENOT models take a prior the same way; the prior is added to the generated cells."""
        cf, _ = _fit(adata_prior, prior=_transfer(), solver="genot")
        rng = jax.random.PRNGKey(0)
        plain, shifted = _without_prior(cf, adata_prior, rng=rng), _predict(cf, adata_prior, rng=rng)
        for k in plain:
            add = cf.prior_predict.values[tuple(k.split("@@"))].astype(np.float32)
            np.testing.assert_array_equal(np.asarray(shifted[k]), np.asarray(plain[k]) + add)

    def test_load_model_without_prior_attributes(self, adata_prior, tmp_path):
        cf, _ = _fit(adata_prior)
        for k in ("prior", "prior_train", "prior_predict"):
            delattr(cf, k)  # as pickled by a version without priors
        cf.save(str(tmp_path))
        cf2 = cellflow.model.CellFlow.load(str(tmp_path / "CellFlow.pkl"))
        assert cf2.prior is None and cf2.prior_train is None and cf2.prior_predict is None
        assert len(_predict(cf2, adata_prior)) == 3
