import anndata as ad
import numpy as np
import pytest


class TestWKNN:
    @pytest.mark.parametrize("n_neighbors", [50, 100])
    def test_compute_wknn_k(self, adata_perturbation: ad.AnnData, n_neighbors):
        import cellflow

        cellflow.pp.compute_wknn(
            ref_adata=adata_perturbation,
            query_adata=adata_perturbation,
            n_neighbors=n_neighbors,
            copy=False,
        )

        assert "wknn" in adata_perturbation.uns
        assert adata_perturbation.uns["wknn"].shape == (
            adata_perturbation.n_obs,
            adata_perturbation.n_obs,
        )
        assert np.all(np.sum(adata_perturbation.uns["wknn"] > 0, axis=1) == n_neighbors)

    @pytest.mark.parametrize("weighting_scheme", ["top_n", "jaccard", "jaccard_square"])
    def test_compute_wknn_weighting(self, adata_perturbation: ad.AnnData, weighting_scheme):
        import cellflow

        n_neighbors = 50
        top_n = 10

        cellflow.pp.compute_wknn(
            ref_adata=adata_perturbation,
            query_adata=adata_perturbation,
            n_neighbors=n_neighbors,
            weighting_scheme=weighting_scheme,
            top_n=top_n,
            copy=False,
        )

        assert "wknn" in adata_perturbation.uns
        assert adata_perturbation.uns["wknn"].shape == (
            adata_perturbation.n_obs,
            adata_perturbation.n_obs,
        )
        if weighting_scheme == "top_n":
            assert np.all(np.sum(adata_perturbation.uns["wknn"] > 0, axis=1) <= n_neighbors)
        else:
            assert np.all(np.sum(adata_perturbation.uns["wknn"] > 0, axis=1) == n_neighbors)

    @pytest.mark.parametrize("uns_key_added", ["wknn", "wknn2"])
    def test_compute_wknn_key_added(self, adata_perturbation: ad.AnnData, uns_key_added):
        import cellflow

        n_neighbors = 50

        cellflow.pp.compute_wknn(
            ref_adata=adata_perturbation,
            query_adata=adata_perturbation,
            n_neighbors=n_neighbors,
            uns_key_added=uns_key_added,
            copy=False,
        )

        assert uns_key_added in adata_perturbation.uns
        assert adata_perturbation.uns[uns_key_added].shape == (
            adata_perturbation.n_obs,
            adata_perturbation.n_obs,
        )
        assert np.all(np.sum(adata_perturbation.uns[uns_key_added] > 0, axis=1) == n_neighbors)

    @pytest.mark.parametrize("label_key", ["drug1", "cell_type"])
    def test_transfer_labels(self, adata_perturbation: ad.AnnData, label_key):
        import cellflow

        cellflow.pp.compute_wknn(
            ref_adata=adata_perturbation,
            query_adata=adata_perturbation,
            n_neighbors=50,
            copy=False,
        )

        cellflow.pp.transfer_labels(
            adata_perturbation,
            adata_perturbation,
            label_key=label_key,
            copy=False,
        )

        assert f"{label_key}_transfer" in adata_perturbation.obs
        assert f"{label_key}_transfer_score" in adata_perturbation.obs
        assert adata_perturbation.obs[f"{label_key}_transfer"].dtype.name == "category"
        assert adata_perturbation.obs[f"{label_key}_transfer_score"].dtype.name == "float64"

    def test_transfer_labels_matches_dense(self, adata_perturbation: ad.AnnData):
        import pandas as pd

        import cellflow

        cellflow.pp.compute_wknn(
            ref_adata=adata_perturbation,
            query_adata=adata_perturbation,
            n_neighbors=50,
            copy=False,
        )
        out = cellflow.pp.transfer_labels(adata_perturbation, adata_perturbation, label_key="cell_type", copy=True)

        onehot = pd.get_dummies(adata_perturbation.obs["cell_type"].astype(str))
        dense = np.asarray(adata_perturbation.uns["wknn"] @ onehot.values.astype(float))
        np.testing.assert_allclose(out.obs["cell_type_transfer_score"].values, dense.max(1))
        assert (out.obs["cell_type_transfer"].astype(str).values == onehot.columns[dense.argmax(1)]).all()
