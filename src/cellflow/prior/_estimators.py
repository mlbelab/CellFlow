from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

import anndata as ad
import numpy as np
import pandas as pd

from cellflow.prior._prior import (
    BaselineSimilarity,
    EffectStats,
    Pair,
    WeightingStrategy,
    _check_columns,
    _context_groups,
    _group_sums,
    _matrix,
    effect_stats,
    transfer_effects,
)

__all__ = ["PriorValues", "PriorEstimator", "TransferPrior", "FixedPrior"]


def _pairs(df: pd.DataFrame, context_key: str, perturbation_key: str) -> tuple[np.ndarray, pd.Index]:
    """Integer codes of each row's (context, perturbation) pair and the unique pairs (``str`` tuples)."""
    _check_columns(df, (context_key, perturbation_key), "the data")
    ctx, pert = df[context_key].astype(str).to_numpy(), df[perturbation_key].astype(str).to_numpy()
    return pd.factorize(pd.MultiIndex.from_arrays([ctx, pert]))


@dataclass(frozen=True)
class PriorValues:
    """Prior vectors of a set of (context, perturbation) pairs, as used by :class:`~cellflow.model.CellFlow`.

    Attributes
    ----------
    values
        ``{(context, perturbation): vector}`` in the model's representation (``sample_rep``). Pairs without an
        entry have a zero prior.
    values_full
        The same priors in the space the effects were computed in (e.g. genes) before the projection, or
        :obj:`None` (always for the training pairs). With orthonormal projection columns, a predicted mean is
        decoded including the prior's part outside the representation as
        ``ctrl_mean + (pred_mean - src_mean) @ projection.T + values_full - values @ projection.T``.
    weights
        ``{(context, perturbation): (neighbour contexts, weights)}`` of transferred priors, or :obj:`None`.
    """

    values: dict[Pair, np.ndarray]
    values_full: dict[Pair, np.ndarray] | None = None
    weights: dict[Pair, tuple[list[str], np.ndarray]] | None = None

    def lookup(
        self, df: pd.DataFrame, context_key: str, perturbation_key: str, dim: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Priors of the rows of ``df``.

        Returns
        -------
        ``(codes, table)``: each row's index into ``table`` and the ``[n_unique_pairs, dim]`` float64 priors
        (zeros for pairs without a prior).
        """
        codes, uniq = _pairs(df, context_key, perturbation_key)
        zero = np.zeros(dim)
        table = np.vstack([np.asarray(self.values.get(k, zero), np.float64) for k in uniq] or [zero])
        if table.shape[1] != dim:
            raise ValueError(f"The prior has dimension {table.shape[1]}, but `sample_rep` has {dim}.")
        return codes, table


@runtime_checkable
class PriorEstimator(Protocol):
    """Interface of a prior for :meth:`cellflow.model.CellFlow.prepare_data` (``prior=``).

    ``fit`` is called once, on a copy of the estimator, with the training data and returns the priors of the
    training pairs, by which their treated cells are shifted; it stores whatever ``predict`` needs on the
    estimator, which is saved with the model. ``predict`` returns the priors of the conditions in
    ``covariate_data``, using only the control cells in ``adata`` for contexts not seen in training. Pairs are
    identified by the ``context_key`` and ``perturbation_key`` columns: a single perturbation column, so
    combinations or doses need one combined label column in :attr:`~anndata.AnnData.obs` and ``covariate_data``.
    """

    context_key: str
    perturbation_key: str

    def fit(self, adata: ad.AnnData, control_key: str, sample_rep: str) -> PriorValues: ...

    def predict(
        self, adata: ad.AnnData, covariate_data: pd.DataFrame, control_key: str, sample_rep: str
    ) -> PriorValues: ...


@dataclass
class FixedPrior:
    """Bring-your-own prior: fixed vectors in the model's representation.

    The same ``values`` are used to shift the training targets and to add back at prediction. Unlike
    :class:`TransferPrior`, nothing guards against priors of training pairs that use their own treated cells.

    Parameters
    ----------
    context_key, perturbation_key
        Columns of :attr:`~anndata.AnnData.obs` and of ``covariate_data`` identifying a pair.
    values
        ``{(context, perturbation): vector}`` in ``sample_rep``.
    """

    context_key: str
    perturbation_key: str
    values: dict[Pair, np.ndarray]

    def fit(self, adata: ad.AnnData, control_key: str, sample_rep: str) -> PriorValues:
        """Return the fixed values."""
        return PriorValues(values=dict(self.values))

    def predict(
        self, adata: ad.AnnData, covariate_data: pd.DataFrame, control_key: str, sample_rep: str
    ) -> PriorValues:
        """Return the fixed values."""
        return PriorValues(values=dict(self.values))


@dataclass
class TransferPrior:
    """Transfer prior: a similarity-weighted transfer of the perturbation's observed effects in other contexts.

    At :meth:`fit` (called by :meth:`~cellflow.model.CellFlow.prepare_data` on the training data) the effects are
    computed with :func:`~cellflow.prior.effect_stats` and every training pair gets a leave-own-context-out prior
    :func:`~cellflow.prior.transfer_effects`. At :meth:`predict` a context not seen in training is weighted from
    the control cells given as prediction source only (never its treated cells).

    Parameters
    ----------
    context_key, perturbation_key
        :attr:`~anndata.AnnData.obs` columns (and ``covariate_data`` columns at prediction) of the context and
        perturbation of a cell. ``perturbation_key`` is a single column; combinations or doses need one
        combined label column.
    weighting
        Weighting strategy, see :class:`~cellflow.prior.WeightingStrategy`.
    scale
        Multiplier of the prior.
    batch_key
        :attr:`~anndata.AnnData.obs` column of the batch (controls of the same context and batch are the
        reference of an effect). :obj:`None`: one batch per context.
    group_key
        :attr:`~anndata.AnnData.obs` column of a group label per context (e.g. ``'tissue'``) for weighting;
        :obj:`None`: no groups (``same_group`` of the weighting then has no effect).
    rep
        Space the effects and similarities are computed in: key in :attr:`~anndata.AnnData.obsm`, or
        :obj:`None` / ``'X'`` for :attr:`~anndata.AnnData.X` (e.g. genes).
    projection
        ``[F, D]`` matrix (or key in :attr:`~anndata.AnnData.varm` of the training data) mapping the effects'
        space to the model's ``sample_rep``, e.g. PCA loadings; :obj:`None` if ``rep`` is ``sample_rep``.
    min_treated, min_control
        Validity thresholds of an observed effect.

    Attributes
    ----------
    stats_
        :class:`~cellflow.prior.EffectStats` of the training data (set by :meth:`fit`).
    projection_
        The projection matrix used (set by :meth:`fit`).
    features_
        ``var_names`` (effects in ``X``) or dimension (effects in ``obsm``) of the effects' space, checked at
        :meth:`predict` for new contexts.
    """

    context_key: str
    perturbation_key: str
    weighting: WeightingStrategy = field(default_factory=BaselineSimilarity)
    scale: float = 1.0
    batch_key: str | None = None
    group_key: str | None = None
    rep: str | None = None
    projection: np.ndarray | str | None = None
    min_treated: int = 30
    min_control: int = 20

    def _features(self, adata: ad.AnnData) -> list[str] | int:
        if self.rep is None or self.rep == "X":
            return list(adata.var_names)
        return int(_matrix(adata, self.rep).shape[1])

    def _effects(self, stats: EffectStats, pairs: list[Pair], full: bool) -> PriorValues:
        eff, weights = transfer_effects(stats, self.weighting, pairs, scale=self.scale)
        if self.projection_ is None:
            return PriorValues(values=eff, weights=weights)
        values = {k: v @ self.projection_ for k, v in eff.items()}
        return PriorValues(values, eff if full else None, weights)

    def fit(self, adata: ad.AnnData, control_key: str, sample_rep: str) -> PriorValues:
        """Compute the training effects and the leave-own-context-out priors of the training pairs."""
        self.stats_ = effect_stats(
            adata,
            self.context_key,
            self.perturbation_key,
            control_key,
            batch_key=self.batch_key,
            group_key=self.group_key,
            rep=self.rep,
            min_treated=self.min_treated,
            min_control=self.min_control,
        )
        self.features_ = self._features(adata)
        p = self.projection
        if isinstance(p, str) and p not in adata.varm:
            raise KeyError(f"Projection `{p}` not found in `adata.varm`.")
        self.projection_ = None if p is None else np.asarray(adata.varm[p] if isinstance(p, str) else p, np.float64)
        _, pairs = _pairs(adata.obs[~adata.obs[control_key].to_numpy(bool)], self.context_key, self.perturbation_key)
        return self._effects(self.stats_, list(pairs), full=False)

    def _with_contexts(self, adata: ad.AnnData, contexts: list[str], control_key: str) -> EffectStats:
        """``stats_`` extended by contexts not seen in training, from their control cells in ``adata`` only."""
        st = self.stats_
        new = sorted(set(contexts) - set(st.contexts))
        if not new:
            return st
        _check_columns(adata.obs, [self.context_key, control_key, self.group_key])
        if self._features(adata) != self.features_:
            raise ValueError(
                "The features of `adata` (var_names, or the obsm dimension of `rep`) differ from the training "
                "data; the prior of new contexts needs the same features."
            )
        ctx = adata.obs[self.context_key].astype(str).to_numpy()
        code = pd.Categorical(ctx, categories=new).codes
        n, s = _group_sums(
            _matrix(adata, self.rep), np.where(adata.obs[control_key].to_numpy(bool), code, -1), len(new)
        )
        if (n == 0).any():
            raise ValueError(
                f"No control cells in `adata` for contexts {[c for c, m in zip(new, n, strict=True) if not m]}."
            )
        groups = st.groups
        if groups is not None:
            sub = adata.obs[np.isin(ctx, new)]
            groups = np.concatenate([groups, _context_groups(sub, self.context_key, self.group_key, new)])
        C, P = len(new), len(st.perturbations)
        # new contexts have no observed (valid) effects, so delta / delta_index are shared unchanged
        return replace(
            st,
            contexts=st.contexts + tuple(new),
            groups=groups,
            n=np.vstack([st.n, np.zeros((C, P), st.n.dtype)]),
            batch=np.vstack([st.batch, np.full((C, P), "", dtype=object)]),
            n_control=np.vstack([st.n_control, np.zeros((C, P), st.n_control.dtype)]),
            control_mean_pooled=np.vstack([st.control_mean_pooled, s / n[:, None]]),
            valid=np.vstack([st.valid, np.zeros((C, P), bool)]),
        )

    def predict(
        self, adata: ad.AnnData, covariate_data: pd.DataFrame, control_key: str, sample_rep: str
    ) -> PriorValues:
        """Priors of the pairs in ``covariate_data``; new contexts are weighted from ``adata``'s control cells."""
        _, pairs = _pairs(covariate_data, self.context_key, self.perturbation_key)
        stats = self._with_contexts(adata, [c for c, _ in pairs], control_key)
        return self._effects(stats, list(pairs), full=True)
