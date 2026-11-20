from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "EffectStats",
    "effect_stats",
    "WeightingStrategy",
    "BaselineSimilarity",
    "transfer_effects",
]

Pair = tuple[str, str]


@dataclass(frozen=True)
class EffectStats:
    """Observed perturbation effects per (context, perturbation), from :func:`effect_stats`.

    The ``[C, P]`` arrays are indexed by ``contexts`` and ``perturbations``. Effects are stored only for valid
    pairs: row ``delta_index[(c, d)]`` of ``delta`` (``[V, F]``, ``F`` features of the space the statistics
    were computed in, e.g. genes); use :meth:`effect`.

    Attributes
    ----------
    context_key, perturbation_key, control_key, batch_key
        The :attr:`~anndata.AnnData.obs` columns the statistics were computed from.
    contexts, perturbations
        Sorted context and perturbation labels.
    groups
        ``[C]`` group label of each context (e.g. tissue), or :obj:`None`.
    n
        ``[C, P]`` number of treated cells.
    batch
        ``[C, P]`` batch in which the pair was profiled (``''`` if it was not observed).
    n_control
        ``[C, P]`` number of control cells of the pair's (context, batch).
    delta
        ``[V, F]`` float64 treated mean minus the control mean of the same (context, batch) of the valid pairs.
    delta_index
        ``{(context, perturbation): row of delta}`` for the valid pairs.
    control_mean_pooled
        ``[C, F]`` float64 mean of all control cells of each context (``nan`` for a context without controls).
    valid
        ``[C, P]`` pairs with at least ``min_treated`` treated and ``min_control`` control cells.
    heldout_contexts
        Contexts whose treated cells were ignored (only their controls entered).
    """

    context_key: str
    perturbation_key: str
    control_key: str
    batch_key: str | None
    contexts: tuple[str, ...]
    perturbations: tuple[str, ...]
    groups: np.ndarray | None
    n: np.ndarray
    batch: np.ndarray
    n_control: np.ndarray
    delta: np.ndarray
    delta_index: dict[Pair, int]
    control_mean_pooled: np.ndarray
    valid: np.ndarray
    heldout_contexts: tuple[str, ...]

    def effect(self, context: str, perturbation: str) -> np.ndarray:
        """``[F]`` observed effect of a valid pair."""
        return self.delta[self.delta_index[(context, perturbation)]]


def _matrix(adata: ad.AnnData, rep: str | None) -> Any:
    """``adata.X`` for ``rep`` :obj:`None` or ``'X'``, else ``adata.obsm[rep]``, with clear errors."""
    if rep is None or rep == "X":
        if adata.X is None:
            raise ValueError("`adata.X` is None, but the statistics are computed in `adata.X`.")
        return adata.X
    if rep not in adata.obsm:
        raise KeyError(f"Representation `{rep}` not found in `adata.obsm`.")
    return adata.obsm[rep]


def _check_columns(df: pd.DataFrame, keys: Iterable[str | None], where: str = "`adata.obs`") -> None:
    """Raise a clear :class:`KeyError` for columns (``None`` entries ignored) missing from ``df``."""
    missing = [k for k in keys if k is not None and k not in df.columns]
    if missing:
        raise KeyError(f"Columns {missing} not found in {where}.")


def _group_sums(X: Any, codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-group row counts and float64 row sums of ``X`` (dense or sparse); ``codes < 0`` are ignored."""
    ok = codes >= 0
    ind = sp.csr_matrix((np.ones(ok.sum()), (codes[ok], np.flatnonzero(ok))), shape=(n_groups, X.shape[0]))
    s = ind @ X
    s = s.toarray() if sp.issparse(s) else np.asarray(s)
    return np.asarray(ind.sum(1)).ravel(), s.astype(np.float64)


def _context_groups(obs: pd.DataFrame, context_key: str, group_key: str, contexts: Sequence[str]) -> np.ndarray:
    g = pd.DataFrame({"c": obs[context_key].astype(str), "g": obs[group_key].astype(str)}).drop_duplicates()
    if g["c"].duplicated().any():
        raise ValueError(f"`{group_key}` must be constant within each context.")
    return g.set_index("c").loc[list(contexts), "g"].to_numpy()


def effect_stats(
    adata: ad.AnnData,
    context_key: str,
    perturbation_key: str,
    control_key: str,
    batch_key: str | None = None,
    group_key: str | None = None,
    rep: str | None = None,
    heldout_contexts: Iterable[str] = (),
    min_treated: int = 30,
    min_control: int = 20,
) -> EffectStats:
    """Compute observed perturbation effects per (context, perturbation).

    The effect of perturbation ``d`` in context ``c`` is the mean of its treated cells minus the mean of the
    control cells of the same context and batch. Means are accumulated in float64; only the effects of valid
    pairs are kept. ``perturbation_key`` is a single column: combinations or doses need one combined label
    column (e.g. ``'drugA+drugB'``).

    Parameters
    ----------
    adata
        Cells with treated and control cells.
    context_key
        :attr:`~anndata.AnnData.obs` column with the context (e.g. cell type) of each cell.
    perturbation_key
        :attr:`~anndata.AnnData.obs` column with the perturbation of each treated cell.
    control_key
        Boolean :attr:`~anndata.AnnData.obs` column marking control cells.
    batch_key
        :attr:`~anndata.AnnData.obs` column with the batch. Each (context, perturbation) pair must lie in a single
        batch. If :obj:`None`, every context is one batch.
    group_key
        :attr:`~anndata.AnnData.obs` column with a group label per context (e.g. ``'tissue'``), used by weighting
        strategies to prefer contexts of the same group. Must be constant within each context.
    rep
        Key in :attr:`~anndata.AnnData.obsm` of the space to compute the effects in, or :obj:`None` / ``'X'``
        for :attr:`~anndata.AnnData.X`.
    heldout_contexts
        Contexts whose treated cells are ignored; only their control cells enter (``control_mean_pooled``).
    min_treated, min_control
        Validity thresholds of a pair (treated cells; controls of the pair's context and batch).

    Returns
    -------
    The :class:`EffectStats`.
    """
    obs = adata.obs
    _check_columns(obs, [context_key, perturbation_key, control_key, batch_key, group_key])
    X = _matrix(adata, rep)
    heldout = tuple(sorted({str(c) for c in heldout_contexts}))
    ctx = obs[context_key].astype(str).to_numpy()
    is_ctrl = obs[control_key].to_numpy(bool)
    treated = ~is_ctrl & ~np.isin(ctx, heldout)
    bat = obs[batch_key].astype(str).to_numpy() if batch_key is not None else np.full(len(obs), "", dtype=object)
    pert = obs[perturbation_key].astype(str).to_numpy()

    contexts = tuple(sorted(set(ctx)))
    perts = tuple(sorted(set(pert[treated])))
    C, P = len(contexts), len(perts)
    c_code = pd.Categorical(ctx, categories=contexts).codes.astype(np.int64)
    pair = np.where(treated, c_code * P + pd.Categorical(pert, categories=perts).codes, -1)
    groups = None if group_key is None else _context_groups(obs, context_key, group_key, contexts)

    n = np.bincount(pair[treated], minlength=C * P).reshape(C, P)
    pb = pd.DataFrame({"pair": pair[treated], "b": bat[treated]}).drop_duplicates()
    if pb["pair"].duplicated().any():
        raise ValueError("A (context, perturbation) pair spans several batches.")
    batch = np.full(C * P, "", dtype=object)
    batch[pb["pair"].to_numpy()] = pb["b"].to_numpy()
    batch = batch.reshape(C, P)

    if not is_ctrl.any():
        raise ValueError("No control cells in `adata`; effects need control cells.")
    cb_code, cb_uniq = pd.factorize(pd.MultiIndex.from_arrays([ctx[is_ctrl], bat[is_ctrl]]))
    cb_full = np.full(len(obs), -1)
    cb_full[is_ctrl] = cb_code
    cbi = {t: i for i, t in enumerate(cb_uniq)}
    k = np.array([cbi.get((contexts[f // P], batch.flat[f]), -1) if n.flat[f] else -1 for f in range(C * P)], int)
    cn, cs = _group_sums(X, cb_full, len(cb_uniq))
    n_control = np.where(k >= 0, cn[np.maximum(k, 0)], 0).reshape(C, P)
    valid = (k.reshape(C, P) >= 0) & (n >= min_treated) & (n_control >= min_control)

    vi = np.flatnonzero(valid)  # flat (context, perturbation) codes of the valid pairs
    row = np.full(C * P, -1)
    row[vi] = np.arange(len(vi))
    nv, sv = _group_sums(X, np.where(pair >= 0, row[np.maximum(pair, 0)], -1), len(vi))
    pn, ps = _group_sums(X, np.where(is_ctrl, c_code, -1), C)
    with np.errstate(invalid="ignore", divide="ignore"):
        control_mean_pooled = ps / pn[:, None]  # nan for a context without controls
    kv = k[vi]
    delta = sv / nv[:, None] - cs[kv] / cn[kv][:, None]
    return EffectStats(
        context_key=context_key,
        perturbation_key=perturbation_key,
        control_key=control_key,
        batch_key=batch_key,
        contexts=contexts,
        perturbations=perts,
        groups=groups,
        n=n,
        batch=batch,
        n_control=n_control,
        delta=delta,
        delta_index={(contexts[f // P], perts[f % P]): r for r, f in enumerate(vi)},
        control_mean_pooled=control_mean_pooled,
        valid=valid,
        heldout_contexts=heldout,
    )


class WeightingStrategy(Protocol):
    """Weights over the same perturbation's observed effects in other contexts, per (context, perturbation).

    A strategy maps ``pairs`` to ``{(context, perturbation): (neighbour contexts, weights)}``. Neighbours must be
    contexts ``t != context`` with ``stats.valid[t, perturbation]``; weights are non-negative and sum to one.
    Pairs without any neighbour are left out (they get no prior).

    :class:`BaselineSimilarity` (zero-shot: similarity of the contexts' control populations) is the only
    strategy implemented. A few-shot strategy weighting contexts by the similarity of their responses to
    perturbations observed in both fits this interface, provided those shots are part of the statistics (i.e.
    of the training data). Priors along the perturbation axis (e.g. a ridge over perturbation features, which
    can serve perturbations never observed) do not: they weight other perturbations, not contexts, and are
    separate :class:`~cellflow.prior.PriorEstimator` implementations.
    """

    def weights(self, stats: EffectStats, pairs: Sequence[Pair]) -> dict[Pair, tuple[list[str], np.ndarray]]: ...


@dataclass(frozen=True)
class BaselineSimilarity:
    """Zero-shot weighting by similarity of the contexts' control populations.

    For a pair ``(c, d)`` the candidates are the contexts ``t != c`` with a valid observed effect of ``d``
    (leave-own-context-out, so training and held-out contexts are treated the same way). If ``same_group``,
    the statistics have groups and any candidate shares ``c``'s group, only those are kept, otherwise all.
    Similarity ``s_t`` is the Pearson correlation of the pooled control means of ``c`` and ``t`` (float64);
    candidates with a non-finite similarity (a context without control cells, or with a constant pooled
    control mean) are dropped, so such a context gets no prior and is no neighbour. The ``k`` most similar
    candidates get weights ``w = softmax((s - max s) / tau)``, multiplied by ``n_t / (n_t + n0)`` (``n_t``
    treated cells of ``(t, d)``; credibility weight) if ``n0 > 0`` and renormalised.

    Parameters
    ----------
    tau
        Softmax temperature (> 0).
    n0
        Credibility pseudo-count; ``0`` disables it.
    k
        Number of most similar candidates to keep; :obj:`None` keeps all.
    same_group
        Prefer candidates of the same group (``group_key`` of :func:`effect_stats`); a no-op if the statistics
        have no groups (no ``group_key``).
    """

    tau: float = 0.2
    n0: float = 0.0
    k: int | None = None
    same_group: bool = True

    def weights(self, stats: EffectStats, pairs: Sequence[Pair]) -> dict[Pair, tuple[list[str], np.ndarray]]:
        """Neighbour contexts and weights for each pair, see :class:`WeightingStrategy`."""
        if self.tau <= 0:
            raise ValueError("`tau` must be > 0.")
        groups = stats.groups if self.same_group else None
        ci = {c: i for i, c in enumerate(stats.contexts)}
        pi = {d: i for i, d in enumerate(stats.perturbations)}
        with np.errstate(invalid="ignore", divide="ignore"):
            S = np.corrcoef(stats.control_mean_pooled.astype(np.float64))
        S = np.atleast_2d(S)
        out = {}
        for c, d in pairs:
            if d not in pi or c not in ci:
                continue
            i, j = ci[c], pi[d]
            cand = [t for t in stats.contexts if t != c and stats.valid[ci[t], j] and np.isfinite(S[i, ci[t]])]
            if groups is not None and any(groups[ci[t]] == groups[i] for t in cand):
                cand = [t for t in cand if groups[ci[t]] == groups[i]]
            cand = sorted(cand, key=lambda t: -S[i, ci[t]])[: self.k]
            if not cand:
                continue
            s = np.array([S[i, ci[t]] for t in cand])
            w = np.exp((s - s.max()) / self.tau) / np.exp((s - s.max()) / self.tau).sum()
            if self.n0 > 0:
                n = np.array([stats.n[ci[t], j] for t in cand], np.float64)
                w = w * n / (n + self.n0)
                w = w / w.sum()
            out[(c, d)] = (cand, w)
        return out


def transfer_effects(
    stats: EffectStats,
    weighting: WeightingStrategy,
    pairs: Iterable[Pair],
    scale: float = 1.0,
) -> tuple[dict[Pair, np.ndarray], dict[Pair, tuple[list[str], np.ndarray]]]:
    """Transferred prior effects ``prior(c, d) = scale * sum_t w_t delta(t, d)``.

    Parameters
    ----------
    stats
        Observed effects, see :func:`effect_stats`.
    weighting
        Weighting strategy giving the neighbours ``t`` and weights ``w_t``, e.g. :class:`BaselineSimilarity`.
    pairs
        (context, perturbation) pairs to compute.
    scale
        Multiplier of the prior.

    Returns
    -------
    ``(effects, weights)``: ``{pair: [F] effect}`` in the space of ``stats`` and ``{pair: (neighbours, weights)}``,
    both only for pairs with at least one neighbour.
    """
    weights = weighting.weights(stats, list(pairs))
    effects = {
        (c, d): scale * sum(wi * stats.effect(t, d) for wi, t in zip(w, cand, strict=True))
        for (c, d), (cand, w) in weights.items()
    }
    return effects, weights
