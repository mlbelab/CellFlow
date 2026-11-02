import jax.tree_util as jtu
import numpy as np
import pytest

import cellflow


class TestMetrics:
    @pytest.mark.parametrize("prefix", ["", "test_"])
    def test_compute_metrics(self, metrics_data, prefix):
        x_test = metrics_data["x_test"]
        y_test = metrics_data["y_test"]

        metrics = jtu.tree_map(cellflow.metrics.compute_metrics, x_test, y_test)
        mean_metrics = cellflow.metrics.compute_mean_metrics(metrics, prefix)

        assert "Alvespimycin+Pirarubicin" in metrics.keys()
        assert {"r_squared", "sinkhorn_div_1", "sinkhorn_div_10", "sinkhorn_div_100", "e_distance", "energy_distance", "mmd"} == set(
            metrics["Alvespimycin+Pirarubicin"].keys()
        )
        assert {
            prefix + "r_squared",
            prefix + "sinkhorn_div_1",
            prefix + "sinkhorn_div_10",
            prefix + "sinkhorn_div_100",
            prefix + "e_distance",
            prefix + "energy_distance",
            prefix + "mmd",
        } == set(mean_metrics.keys())

    @pytest.mark.parametrize("epsilon", [0.1, 1, 10])
    def test_function_output(self, metrics_data, epsilon):
        x_test = metrics_data["x_test"]["Alvespimycin+Pirarubicin"]
        y_test = metrics_data["y_test"]["Alvespimycin+Pirarubicin"]

        r_squared = cellflow.metrics.compute_r_squared(x_test, y_test)
        sinkhorn_div = cellflow.metrics.compute_sinkhorn_div(x_test, y_test, epsilon=epsilon)
        e_distance = cellflow.metrics.compute_e_distance(x_test, y_test)
        e_distance_fast = cellflow.metrics.compute_e_distance_fast(x_test, y_test)
        scalar_mmd = cellflow.metrics.compute_scalar_mmd(x_test, y_test)
        mmd_fast = cellflow.metrics.maximum_mean_discrepancy(x_test, y_test, exact=False)

        assert -1000 <= r_squared <= 1
        assert sinkhorn_div >= 0
        assert e_distance >= 0
        assert e_distance_fast >= 0
        assert scalar_mmd >= 0
        assert mmd_fast >= 0

    @pytest.mark.parametrize("gamma", [0.1, 1, 2])
    def test_fast_metrics(self, metrics_data, gamma):
        x_test = metrics_data["x_test"]["Alvespimycin+Pirarubicin"]
        y_test = metrics_data["y_test"]["Alvespimycin+Pirarubicin"]

        e_distance = cellflow.metrics.compute_e_distance(x_test, y_test)
        e_distance_fast = cellflow.metrics.compute_e_distance_fast(x_test, y_test)

        mmd = cellflow.metrics.maximum_mean_discrepancy(x_test, y_test, gamma, exact=True)
        mmd_fast = cellflow.metrics.maximum_mean_discrepancy(x_test, y_test, gamma, exact=False)

        assert np.allclose(e_distance, e_distance_fast, rtol=1e-4, atol=1e-4)
        assert np.allclose(mmd, mmd_fast, rtol=1e-4, atol=1e-4)

    def test_e_distance_is_mean_only(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (200, 5)).astype(np.float32)
        y = rng.normal(2, 1, (150, 5)).astype(np.float32)
        expected = 2 * np.sum((x.mean(0) - y.mean(0)) ** 2)
        assert np.allclose(cellflow.metrics.compute_e_distance(x, y), expected, rtol=1e-4)

    def test_energy_distance_sees_shape(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (300, 5)).astype(np.float32)
        bimodal = np.concatenate([rng.normal(-3, 0.3, (150, 5)), rng.normal(3, 0.3, (150, 5))]).astype(np.float32)
        bimodal = bimodal - bimodal.mean(0) + x.mean(0)  # same mean as x
        same = rng.normal(0, 1, (300, 5)).astype(np.float32)

        assert np.isclose(cellflow.metrics.compute_e_distance(x, bimodal), 0, atol=1e-3)
        energy_bimodal = cellflow.metrics.compute_energy_distance(x, bimodal)
        energy_same = cellflow.metrics.compute_energy_distance(x, same)
        assert energy_bimodal > 10 * energy_same > 0
