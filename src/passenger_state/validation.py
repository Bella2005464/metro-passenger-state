"""Validation: synthetic worlds, coverage metrics, and rank statistics.

Two synthetic worlds test the reliability layer, differing in how their
generating mechanism relates to the inference model:

* the *shared-mechanism* synthetic world anchors real target-day entry counts
  to a known transfer destination-share vector and a known logit route choice
  (:func:`simulate_s1_world`, :func:`simulate_s1_online_world`), so the true
  transfer-out flow is exactly computable and coverage/precision of the
  posterior is measurable (:func:`evaluate_s1_coverage`). Because it shares the
  logit route-choice mechanism with the inference model, it bounds precision
  from above and must not be the only primary result.
* the *mechanism-divergent* synthetic world generates a known transfer-split
  truth with a different functional form, a gravity model
  ``p(k) ∝ attraction[k] / cost^γ`` (:func:`gravity_transfer_probs`,
  :func:`simulate_s2_world`), so the posterior can be tested against a true
  error it was not built to match.

In both worlds the candidate set and cost are the same physical quantities as
in the inference model; only the decision rule (gravity vs softmax) diverges.
The module also provides accuracy and discrimination metrics (:func:`wape`,
:func:`s2_relative_error`, :func:`rank_correlation`) and a day-block
bootstrap.

All functions depend only on NumPy.
"""

import numpy as np

from passenger_state.network import (
    TRANSFER_STATION_THRESHOLD,
    downstream_interchanges,
    route_choice_cost,
)


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted Absolute Percentage Error.

    wape = sum(|actual - predicted|) / sum(actual)

    Raises ValueError if actual contains negative values or sums to zero.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if (actual < 0).any():
        raise ValueError("actual values must be non-negative")
    total_actual = actual.sum()
    if total_actual == 0:
        raise ValueError("sum of actual values must be positive for WAPE")
    return float(np.abs(actual - predicted).sum() / total_actual)


def day_block_bootstrap(
    per_day_deltas: np.ndarray,
    B: int = 2000,
    seed: int = 42,
) -> dict:
    """Day-block bootstrap for paired per-day differences.

    Resamples days with replacement, computes mean delta per sample,
    returns bootstrap distribution statistics.

    Args:
        per_day_deltas: 1-D array of per-day differences.
        B: Number of bootstrap resamples (default 2000).
        seed: RNG seed for reproducibility.

    Returns:
        dict with keys: mean_difference, ci_95_lower, ci_95_upper, B, n_days.
    """
    deltas = np.asarray(per_day_deltas, dtype=np.float64)
    if deltas.ndim != 1:
        raise ValueError("per_day_deltas must be a 1-D array")
    n_days = len(deltas)
    if n_days < 2:
        raise ValueError("need at least 2 days for bootstrap")
    rng = np.random.RandomState(seed)
    bootstrap_means = np.empty(B, dtype=np.float64)
    for b in range(B):
        indices = rng.randint(0, n_days, size=n_days)
        bootstrap_means[b] = deltas[indices].mean()
    return {
        "mean_difference": float(np.mean(deltas)),
        "ci_95_lower": float(np.percentile(bootstrap_means, 2.5)),
        "ci_95_upper": float(np.percentile(bootstrap_means, 97.5)),
        "B": B,
        "n_days": n_days,
    }


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between two equal-length 1-D arrays (no scipy).

    Ranks each array and returns the Pearson correlation of the ranks. Ties are
    averaged. Returns NaN when either array has zero variance.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D")
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shape")
    if x.size < 2:
        raise ValueError("need at least two points for correlation")

    def _rank(a):
        order = a.argsort()
        ranks = np.empty_like(a, dtype=np.float64)
        ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
        # Average ties.
        uniq, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
        if counts.max() > 1:
            for u in np.where(counts > 1)[0]:
                idx = np.where(inverse == u)[0]
                ranks[idx] = ranks[idx].mean()
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def rank_correlation(x, y, B: int = 2000, seed: int = 42) -> dict:
    """Spearman rank correlation + two-sided permutation p-value.

    Args:
        x: [n] feature values.
        y: [n] target values (e.g. transfer-split CV).
        B: permutation iterations.
        seed: RNG seed.

    Returns:
        dict with ``spearman``, ``permutation_pvalue`` and ``n``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be equal-length 1-D arrays")
    n = int(x.size)
    if n < 2:
        raise ValueError("need at least two points")
    observed = spearman_correlation(x, y)
    rng = np.random.RandomState(seed)
    count_extreme = 0
    for _ in range(B):
        perm = rng.permutation(x)
        if abs(spearman_correlation(perm, y)) >= abs(observed):
            count_extreme += 1
    return {
        "spearman": float(observed),
        "permutation_pvalue": float((count_extreme + 1) / (B + 1)),
        "n": n,
    }


def simulate_offline_counts(off_totals: np.ndarray, phi_true: np.ndarray, rng) -> np.ndarray:
    """Simulate off-line destination counts from true shares.

    Args:
        off_totals: [S] integer off-line totals per origin.
        phi_true: [S, n_off] true off-line destination shares (row-simplex).
        rng: ``numpy.random.RandomState``.

    Returns:
        [S, n_off] integer counts with row sums equal to ``off_totals``.
    """
    off_totals = np.asarray(off_totals, dtype=np.int64)
    phi_true = np.asarray(phi_true, dtype=np.float64)
    S, n_off = phi_true.shape
    n = np.empty((S, n_off), dtype=np.int64)
    for i in range(S):
        n[i] = rng.multinomial(int(off_totals[i]), phi_true[i])
    return n


def true_transfer_flow(entry, off_share, phi_true, route_probs) -> np.ndarray:
    """True per-interchange transfer-out flow given truth.

    Args:
        entry: [S] target-day entry counts.
        off_share: [S] true off-line (transfer-out) proportions.
        phi_true: [S, n_off] true off-line destination shares.
        route_probs: [S, n_off, K] route-choice probabilities.

    Returns:
        [K] true transfer-out flow per interchange.
    """
    entry = np.asarray(entry, dtype=np.float64)
    off_share = np.asarray(off_share, dtype=np.float64)
    phi_true = np.asarray(phi_true, dtype=np.float64)
    route_probs = np.asarray(route_probs, dtype=np.float64)
    off_mass = entry * off_share  # [S]
    dest_mass = off_mass[:, None] * phi_true  # [S, n_off]
    return (dest_mass[:, :, None] * route_probs).sum(axis=(0, 1))


def simulate_online_counts(on_totals, q_on_true, rng):
    """Simulate on-line destination counts from true shares.

    Args:
        on_totals: [S] integer on-line totals per origin.
        q_on_true: [S, S] true on-line destination shares (row-simplex, diag 0).
        rng: ``numpy.random.RandomState``.

    Returns:
        [S, S] integer counts with row sums equal to ``on_totals``.
    """
    on_totals = np.asarray(on_totals, dtype=np.int64)
    q_on_true = np.asarray(q_on_true, dtype=np.float64)
    S = q_on_true.shape[0]
    n = np.empty((S, S), dtype=np.int64)
    for i in range(S):
        n[i] = rng.multinomial(int(on_totals[i]), q_on_true[i])
    return n


def true_online_alighting(entry, on_share, q_on_true):
    """True on-line alighting (column sums of the on-line OD) given truth.

    Args:
        entry: [S] target-day entry counts.
        on_share: [S] true on-line (as opposed to off-line/transfer) proportions.
        q_on_true: [S, S] true on-line destination shares.

    Returns:
        [S] true on-line alighting per station.
    """
    entry = np.asarray(entry, dtype=np.float64)
    on_share = np.asarray(on_share, dtype=np.float64)
    q_on_true = np.asarray(q_on_true, dtype=np.float64)
    on_mass = entry * on_share
    return (on_mass[:, None] * q_on_true).sum(axis=0)


def simulate_s1_world(
    entry: np.ndarray,
    phi_true: np.ndarray,
    off_share_true: np.ndarray,
    route_probs: np.ndarray,
    train_totals: np.ndarray,
    rng,
):
    """Simulate one shared-mechanism world with known truth.

    Args:
        entry: [S] real target-day entry counts.
        phi_true: [S, n_off] true off-line destination shares.
        off_share_true: [S] true off-line proportions.
        route_probs: [S, n_off, K] true route-choice probabilities.
        train_totals: [S] training-total entry counts (data scale).
        rng: ``numpy.random.RandomState``.

    Returns:
        ``(T_true, on_totals, off_totals, n_off_counts)`` where ``T_true`` is
        the [K] true transfer-out flow and the remaining arrays are the training
        data the inference will consume.
    """
    entry = np.asarray(entry, dtype=np.float64)
    phi_true = np.asarray(phi_true, dtype=np.float64)
    off_share_true = np.asarray(off_share_true, dtype=np.float64)
    route_probs = np.asarray(route_probs, dtype=np.float64)
    train_totals = np.asarray(train_totals, dtype=np.float64)

    off_totals = np.array(
        [rng.binomial(int(tt), off_share_true[i]) for i, tt in enumerate(train_totals)],
        dtype=np.float64,
    )
    on_totals = train_totals - off_totals
    n_off_counts = simulate_offline_counts(off_totals, phi_true, rng)
    T_true = true_transfer_flow(entry, off_share_true, phi_true, route_probs)
    return T_true, on_totals, off_totals, n_off_counts


def evaluate_s1_coverage(
    T_true: np.ndarray,
    posterior_samples: np.ndarray,
    level: float = 0.95,
) -> dict:
    """Coverage and precision of the transfer posterior against known truth.

    Args:
        T_true: [K] true transfer-out flow.
        posterior_samples: [n_samples, K] posterior draws.
        level: credible-interval level.

    Returns:
        dict with ``coverage`` (fraction of interchanges whose true value falls
        in the interval), ``covered`` ([K] bool), ``bias`` ([K] relative bias)
        and ``interval_width`` ([K]).
    """
    T_true = np.asarray(T_true, dtype=np.float64)
    posterior_samples = np.asarray(posterior_samples, dtype=np.float64)
    if T_true.ndim != 1 or posterior_samples.ndim != 2:
        raise ValueError("T_true must be [K] and posterior_samples [n_samples, K]")
    if posterior_samples.shape[1] != T_true.shape[0]:
        raise ValueError("posterior_samples and T_true must share the interchange dim")

    alpha = (1.0 - level) / 2.0
    lo = np.percentile(posterior_samples, 100.0 * alpha, axis=0)
    hi = np.percentile(posterior_samples, 100.0 * (1.0 - alpha), axis=0)
    mean = posterior_samples.mean(axis=0)

    covered = (T_true >= lo) & (T_true <= hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        bias = (mean - T_true) / np.where(T_true > 0, T_true, 1.0)
    return {
        "coverage": float(covered.mean()),
        "covered": covered,
        "bias": bias,
        "interval_width": hi - lo,
    }


def simulate_s1_online_world(entry, q_on_true, on_share_true, train_totals, rng):
    """Simulate one shared-mechanism world for the on-line alighting posterior.

    Args:
        entry: [S] real target-day entry counts.
        q_on_true: [S, S] true on-line destination shares (row-simplex, diag 0).
        on_share_true: [S] true on-line proportions.
        train_totals: [S] training-total entry counts.
        rng: ``numpy.random.RandomState``.

    Returns:
        ``(A_true, on_totals, off_totals, on_counts)`` where ``A_true`` is the
        [S] true on-line alighting and the remaining arrays are the training
        data the inference consumes.
    """
    entry = np.asarray(entry, dtype=np.float64)
    q_on_true = np.asarray(q_on_true, dtype=np.float64)
    on_share_true = np.asarray(on_share_true, dtype=np.float64)
    train_totals = np.asarray(train_totals, dtype=np.float64)

    on_totals = np.array(
        [rng.binomial(int(tt), on_share_true[i]) for i, tt in enumerate(train_totals)],
        dtype=np.float64,
    )
    off_totals = train_totals - on_totals
    on_counts = simulate_online_counts(on_totals, q_on_true, rng)
    A_true = true_online_alighting(entry, on_share_true, q_on_true)
    return A_true, on_totals, off_totals, on_counts


def gravity_transfer_probs(
    line_stations,
    order,
    off_line_dests,
    pred_trees,
    k_dist,
    attraction,
    gamma,
    section_time=2.5,
    transfer_time=4.0,
    interchanges=None,
):
    """Build the [S, n_off, K] gravity route-choice probability tensor.

    For every origin ``i`` and off-line destination ``d``, the transfer-out
    interchange is drawn from ``p(k) ∝ attraction[k] / cost(i,k,d)^γ`` over the
    downstream candidates (``gamma = 0`` is cost-independent, i.e. purely
    attraction-driven; ``gamma -> inf`` recovers the minimum-cost interchange).

    Args:
        line_stations: ordered canonical station IDs of the line.
        order: station ID -> sequential order along the line.
        off_line_dests: ordered list of off-line destination station IDs.
        pred_trees: predecessor tree per line origin (candidate direction).
        k_dist: per-interchange distance dict (continuation time to each dest).
        attraction: interchange ID -> non-negative pull (e.g. number of lines
            served). Interchanges absent from this dict get pull 0.
        gamma: gravity decay exponent (non-negative).
        section_time, transfer_time: cost parameters (shared with inference).
        interchanges: the line's interchange list.

    Returns:
        ``(probs, interchanges)`` where ``probs`` is [S, n_off, K].
    """
    S = len(line_stations)
    n_off = len(off_line_dests)
    if interchanges is None:
        interchanges = [s for s in line_stations if s >= TRANSFER_STATION_THRESHOLD]
    interchanges = list(interchanges)
    k_idx = {k: j for j, k in enumerate(interchanges)}
    probs = np.zeros((S, n_off, len(interchanges)), dtype=np.float64)
    for i, origin in enumerate(line_stations):
        pred = pred_trees[origin]
        for d_idx, d in enumerate(off_line_dests):
            cands = downstream_interchanges(
                line_stations, order, origin, d, pred, interchanges
            )
            if not cands:
                continue
            weights = np.zeros(len(cands), dtype=np.float64)
            for j, k in enumerate(cands):
                dist = k_dist[k].get(d)
                if dist is None:
                    continue
                cost = route_choice_cost(
                    order, origin, k, dist, section_time, transfer_time
                )
                a = float(attraction.get(k, 0.0))
                if a < 0:
                    raise ValueError("attraction must be non-negative")
                if gamma == 0.0:
                    w = a
                else:
                    w = a / (cost ** gamma)
                weights[j] = w
            total = weights.sum()
            if total > 0:
                probs[i, d_idx, [k_idx[k] for k in cands]] = weights / total
    return probs, interchanges


def simulate_s2_world(entry, off_share_true, phi_true, p_gravity):
    """Deterministic [K] transfer-out truth under the gravity generator.

    ``T_true[k] = sum_{i,d} entry_i * off_share_i * phi_true[i,d] *
    p_gravity[k|i,d]``. The off-line shares/split are held at their fitted point
    values so the only divergence from the inference is the transfer-split rule
    (gravity vs logit).
    """
    entry = np.asarray(entry, dtype=np.float64)
    off_share_true = np.asarray(off_share_true, dtype=np.float64)
    phi_true = np.asarray(phi_true, dtype=np.float64)
    p_gravity = np.asarray(p_gravity, dtype=np.float64)
    off_mass = entry * off_share_true  # [S]
    dest_mass = off_mass[:, None] * phi_true  # [S, n_off]
    return (dest_mass[:, :, None] * p_gravity).sum(axis=(0, 1))


def s2_relative_error(T_true, posterior_samples):
    """Per-interchange relative error ``|mean - truth| / truth``.

    Returns ``(relative_error, active_mask)``: ``active_mask[k]`` is False where
    ``T_true[k] == 0`` (the interface carries no transfer flow, so relative
    error is undefined and reported as 0.0).
    """
    T_true = np.asarray(T_true, dtype=np.float64)
    posterior_samples = np.asarray(posterior_samples, dtype=np.float64)
    if T_true.ndim != 1 or posterior_samples.ndim != 2:
        raise ValueError("T_true must be [K] and posterior_samples [n, K]")
    if posterior_samples.shape[1] != T_true.shape[0]:
        raise ValueError("posterior_samples and T_true must share K")
    mean = posterior_samples.mean(axis=0)
    active = T_true > 0
    rel = np.zeros_like(T_true, dtype=np.float64)
    rel[active] = np.abs(mean[active] - T_true[active]) / T_true[active]
    return rel, active


def s2_rank_correlation(posterior_cv, relative_error, B=2000, seed=42):
    """Spearman rank correlation + permutation p between CV and relative error.

    Wraps :func:`rank_correlation`. ``posterior_cv`` and ``relative_error`` must
    be equal-length
    1-D arrays over the same interfaces (zero-truth interfaces already
    excluded by the caller via the active mask).
    """
    return rank_correlation(posterior_cv, relative_error, B=B, seed=seed)
