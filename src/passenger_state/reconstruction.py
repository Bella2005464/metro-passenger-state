"""Line-level OD reconstruction: closed-form shares and full posterior.

Two layers of the reconstruction. The point layer assumes within-line alighting
probabilities are stable across service days, so a service day's line-level OD
is recoverable from its observable entry counts and historically fitted
destination shares (:func:`fit_destination_shares`, :func:`predict_od`).

The posterior layer upgrades those shares to conjugate posteriors: a Dirichlet
for the on-line destination shares, a Beta for the on-line/transfer split, and
a Dirichlet for the transfer destination shares, combined with the
route-choice tensor into a single full line OD posterior ``[n_samples, S, S]``
(:func:`sample_line_od_posterior`). The primary object is the column sums (the
per-station alighting posterior): interchange columns carry the unidentifiable
transfer-out split uncertainty, on-line-only columns are near-deterministic.

Conjugate structure (per origin i, independent given data):
  s_i    ~ Beta(a_on + on_total_i, a_off + off_total_i)   on-line fraction
  q_on   ~ Dirichlet(on_od_sum[i] + alpha, diagonal 0)    on-line dest shares
  q_off  ~ Dirichlet(off_counts[i] + alpha)               transfer dest shares
  route = fixed logit / eta-mixture (see :mod:`passenger_state.network`)

Only numpy.random.RandomState is used; there is no MCMC dependency.
"""

import numpy as np

from passenger_state.network import assign_alighting_station
from passenger_state.reliability import fit_online_split


def build_line_od(
    line_stations: tuple[int, ...],
    trips,
    pred_trees: dict[int, dict[int, int | None]],
) -> tuple[np.ndarray, float]:
    """Build the full line-level OD matrix from origin-on-line trips.

    Assigns each trip to its alighting station via exact shortest path (see
    :func:`passenger_state.network.assign_alighting_station`): the destination itself
    for direct trips, or the last line station for transfer-out trips. The
    result has row sums equal to each origin's entry count (unlike the on-line
    OD, which covers only the on-line fraction).

    Args:
        line_stations: ordered canonical station IDs of the line.
        trips: iterable of ``(origin, dest, flow)``; every origin must be in
            ``line_stations``.
        pred_trees: predecessor tree (from
            :func:`passenger_state.network.dijkstra_predecessors`) per line origin.

    Returns:
        (od, unassigned): ``od`` is the [S, S] float matrix (origin rows,
        destination columns); ``unassigned`` is the total flow whose
        destination was unreachable (expected zero on the connected network).
    """
    order = {sid: i for i, sid in enumerate(line_stations)}
    line_set = set(line_stations)
    od = np.zeros((len(line_stations), len(line_stations)), dtype=np.float64)
    unassigned = 0.0
    for origin, dest, flow in trips:
        origin = int(origin)
        dest = int(dest)
        if origin not in order:
            raise ValueError(f"trip origin {origin} is not on the line")
        pred = pred_trees[origin]
        j = assign_alighting_station(line_set, origin, dest, pred)
        if j is None:
            unassigned += float(flow)
            continue
        od[order[origin], order[j]] += float(flow)
    return od, unassigned


def _validate_od_entry(train_od: np.ndarray, train_entry: np.ndarray) -> None:
    od = np.asarray(train_od, dtype=np.float64)
    entry = np.asarray(train_entry, dtype=np.float64)
    if od.ndim != 2 or od.shape[0] != od.shape[1]:
        raise ValueError("train_od must be a square [S, S] array")
    if entry.ndim != 1 or entry.shape[0] != od.shape[0]:
        raise ValueError("train_entry must be a 1-D array matching train_od rows")
    if (od < 0).any():
        raise ValueError("train_od flows must be non-negative")
    if (entry < 0).any():
        raise ValueError("train_entry must be non-negative")


def fit_destination_shares(
    train_od: np.ndarray,
    train_entry: np.ndarray,
) -> np.ndarray:
    """Fit origin→destination shares from aggregated training flow.

    Args:
        train_od: [S, S] float array; ``train_od[i, j]`` is total flow i→j over
            the training days.
        train_entry: [S] float array; total entry count at each origin over the
            same training days.

    Returns:
        shares: [S, S] float array with ``shares[i, j] = train_od[i, j] /
        train_entry[i]`` (zero row where entry is zero). Each row sums to at
        most one; the residual is off-line / transfer-out mass.
    """
    _validate_od_entry(train_od, train_entry)
    od = np.asarray(train_od, dtype=np.float64)
    entry = np.asarray(train_entry, dtype=np.float64)
    shares = np.zeros_like(od)
    positive = entry > 0
    shares[positive] = od[positive] / entry[positive, None]
    return shares


def predict_od(entry: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """Predict a service day's line-level OD from its entry counts and shares.

    Args:
        entry: [S] entry counts for the target day.
        shares: [S, S] stable destination-share matrix.

    Returns:
        predicted: [S, S] with ``predicted[i, j] = entry[i] * shares[i, j]``.
    """
    entry_arr = np.asarray(entry, dtype=np.float64)
    shares_arr = np.asarray(shares, dtype=np.float64)
    if shares_arr.ndim != 2 or shares_arr.shape[0] != shares_arr.shape[1]:
        raise ValueError("shares must be a square [S, S] array")
    if entry_arr.ndim != 1 or entry_arr.shape[0] != shares_arr.shape[0]:
        raise ValueError("entry must be a 1-D array matching shares rows")
    if (entry_arr < 0).any():
        raise ValueError("entry must be non-negative")
    if (shares_arr < 0).any():
        raise ValueError("shares must be non-negative")
    return entry_arr[:, None] * shares_arr


def fit_online_dest_alpha(on_od_sum: np.ndarray, alpha_prior: float = 1.0) -> np.ndarray:
    """Dirichlet concentration for on-line destination shares per origin.

    Args:
        on_od_sum: [S, S] aggregated on-line OD counts over training days
            (the diagonal must already be zero, with self-loops excluded
            upstream).
        alpha_prior: symmetric Dirichlet prior (Laplace smoothing).

    Returns:
        [S, S] concentration ``on_od_sum + alpha_prior`` with the diagonal
        zeroed (self-loops are forbidden in the main OD).
    """
    counts = np.asarray(on_od_sum, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("on_od_sum must be a square [S, S] array")
    if (counts < 0).any():
        raise ValueError("on_od_sum must be non-negative")
    alpha = counts + alpha_prior
    np.fill_diagonal(alpha, 0.0)
    return alpha


def fit_overdispersion_phi(
    alphas,
    daily_shares,
    min_phi: float = 1e-4,
    max_phi: float = 1.0,
) -> float:
    """Fit a single shared Dirichlet over-dispersion factor ``phi`` by moments.

    The static on-line destination-share posterior is ``q ~ Dirichlet(alpha)``
    with ``alpha = train_counts + prior``, so its concentration ``A = sum(alpha)``
    is huge and its variance tiny, which is the source of the static
    posterior's on-line over-confidence. Over-dispersion replaces it with
    ``q ~ Dirichlet(alpha * phi)`` (companion paper, Eq. (3)), ``phi`` in
    (0, 1], which keeps the mean ``alpha / A`` but inflates the variance by
    shrinking the concentration.

    ``phi`` is fit by matching the Dirichlet's *total* variance
        ``f(phi) = sum_i c_i / (A_i * phi + 1)``,  ``c_i = 1 - sum_j (alpha_ij/A_i)^2``
    to the observed *across-day* total variance
        ``V = sum_i mean_d sum_j (q_dij - pi_ij)^2``
    of the destination-share vectors, where ``pi = alpha / A`` is the fitted
    mean (companion paper, Eq. (4)). A single ``phi`` solves ``f(phi) = V`` by
    bisection (``f`` is strictly decreasing in ``phi``); the result is clamped
    to ``[min_phi, max_phi]``.

    A single shared ``phi`` is the calibration layer's only between-day
    variance parameter. ``phi`` must be fit from training days only, never from
    the evaluation split.

    Args:
        alphas: list of 1-D non-negative concentration rows (one per origin,
            diagonal already zero). Each must have positive total.
        daily_shares: list of ``[n_days, S]`` observed daily destination-share
            matrices, one per origin, aligned to ``alphas``. Days where the
            origin had no on-line flow must be excluded by the caller (a
            zero-flow row would spuriously inflate the variance).
        min_phi: lower clamp (``phi -> 0`` is the maximally over-dispersed limit).
        max_phi: upper clamp (``1.0`` = no over-dispersion).

    Returns:
        ``phi`` in ``[min_phi, max_phi]``.
    """
    if len(alphas) != len(daily_shares):
        raise ValueError("alphas and daily_shares must have matching length")
    if not alphas:
        raise ValueError("need at least one origin")

    terms = []  # (c_i, A_i)
    v_total = 0.0
    for alpha, shares in zip(alphas, daily_shares):
        alpha = np.asarray(alpha, dtype=np.float64)
        shares = np.asarray(shares, dtype=np.float64)
        if alpha.ndim != 1:
            raise ValueError("each alpha must be 1-D")
        if (alpha < 0).any():
            raise ValueError("alpha must be non-negative")
        a = float(alpha.sum())
        if a <= 0:
            raise ValueError("each alpha must have positive total concentration")
        if shares.ndim != 2 or shares.shape[1] != alpha.shape[0]:
            raise ValueError("each daily_shares must be [n_days, len(alpha)]")
        pi = alpha / a
        c = float(1.0 - (pi ** 2).sum())
        terms.append((c, a))
        dev = shares - pi
        v_total += float((dev ** 2).sum(axis=1).mean())

    def f(phi: float) -> float:
        return sum(c / (a * phi + 1.0) for c, a in terms)

    if v_total <= f(max_phi):
        return float(max_phi)
    if v_total >= f(min_phi):
        return float(min_phi)
    lo, hi = min_phi, max_phi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < v_total:  # f decreasing: mid too large
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def bootstrap_overdispersion_phi(
    alphas,
    daily_shares,
    n_bootstrap: int = 1000,
    seed: int = 0,
    min_phi: float = 1e-4,
    max_phi: float = 1.0,
) -> dict:
    """Day-block bootstrap confidence interval for the over-dispersion factor.

    Resamples days (with replacement) within each origin's daily-share matrix,
    refits ``phi`` by method of moments on the resampled day-to-day variance, and
    returns the point estimate plus a 2.5/97.5 percentile interval. This gives
    ``phi`` an uncertainty interval instead of a bare scalar, so the coverage
    claim is not conditioned on a single point estimate.

    Each origin's fitted concentration ``alpha`` is held fixed across resamples:
    ``alpha`` derives from the *aggregated* training counts (far more stable than
    the day-to-day variance), so the interval reflects the dominant source of
    ``phi`` uncertainty: the across-day variance estimate (whose effective
    sample size is the per-weekday day count, not the full training span).

    Args:
        alphas: list of 1-D concentration rows (aligned to ``daily_shares``).
        daily_shares: list of ``[n_days, S]`` observed daily share matrices.
        n_bootstrap: number of bootstrap resamples.
        seed: RNG seed.
        min_phi, max_phi: clamps passed to :func:`fit_overdispersion_phi`.

    Returns:
        dict with ``phi`` (point estimate on the original data), ``ci_low``,
        ``ci_high`` (2.5/97.5 percentile) and ``n_bootstrap``.
    """
    n_orig = [np.asarray(s).shape[0] for s in daily_shares]
    if any(n < 2 for n in n_orig):
        raise ValueError("each origin needs at least two days for a bootstrap")
    rng = np.random.RandomState(seed)
    estimates = []
    for _ in range(n_bootstrap):
        resampled = []
        for shares, n in zip(daily_shares, n_orig):
            idx = rng.randint(0, n, size=n)
            resampled.append(np.asarray(shares)[idx])
        estimates.append(fit_overdispersion_phi(alphas, resampled, min_phi, max_phi))
    estimates = np.asarray(estimates, dtype=np.float64)
    point = fit_overdispersion_phi(alphas, daily_shares, min_phi, max_phi)
    return {
        "phi": float(point),
        "ci_low": float(np.percentile(estimates, 2.5)),
        "ci_high": float(np.percentile(estimates, 97.5)),
        "n_bootstrap": int(n_bootstrap),
    }


def fit_overdispersion_split_psi(
    beta_params,
    daily_split,
    min_psi: float = 1e-4,
    max_psi: float = 1.0,
) -> float:
    """Fit a single shared Beta over-dispersion factor ``psi`` by moments.

    The static on-line/off-line split posterior is ``s_i ~ Beta(a_on_i, a_off_i)``
    with huge concentrations (aggregated training counts), so its variance is
    tiny. Over-dispersion replaces it with
    ``s_i ~ Beta(a_on_i * psi, a_off_i * psi)`` (companion paper, Eq. (2)),
    ``psi`` in (0, 1], which keeps the on-line mean
    ``a_on_i / (a_on_i + a_off_i)`` but inflates the variance by shrinking the
    concentration, the between-day split calibration that the
    destination-share ``phi`` (:func:`fit_overdispersion_phi`) does not cover.

    ``psi`` is fit by matching the Beta's *total* variance
        ``f(psi) = sum_i c_i / (A_i * psi + 1)``,
        ``c_i = mu_i (1 - mu_i)``, ``mu_i = a_on_i / A_i``, ``A_i = a_on_i + a_off_i``
    to the observed *across-day* total variance of the on-line fractions
        ``V = sum_i mean_d (s_di - mu_i)^2``.
    A single ``psi`` solves ``f(psi) = V`` by bisection (``f`` is strictly
    decreasing in ``psi``); the result is clamped to ``[min_psi, max_psi]``.

    ``psi`` must be fit from training days only, never from the evaluation split.

    Args:
        beta_params: list of length-2 arrays ``(a_on, a_off)``, one per origin.
        daily_split: list of 1-D arrays of observed daily on-line fractions
            ``s_d = on_d / (on_d + off_d)``, one per origin, aligned to
            ``beta_params``. Days where the origin had zero total flow must be
            excluded by the caller (a zero-flow day would spuriously deflate s_d).
        min_psi: lower clamp (``psi -> 0`` is the maximally over-dispersed limit).
        max_psi: upper clamp (``1.0`` = no over-dispersion).

    Returns:
        ``psi`` in ``[min_psi, max_psi]``.
    """
    if len(beta_params) != len(daily_split):
        raise ValueError("beta_params and daily_split must have matching length")
    if not beta_params:
        raise ValueError("need at least one origin")

    terms = []  # (c_i, A_i)
    v_total = 0.0
    for beta, fracs in zip(beta_params, daily_split):
        beta = np.asarray(beta, dtype=np.float64)
        fracs = np.asarray(fracs, dtype=np.float64)
        if beta.ndim != 1 or beta.shape[0] != 2:
            raise ValueError("each beta_params entry must be a length-2 (a_on, a_off) array")
        if (beta < 0).any():
            raise ValueError("beta_params must be non-negative")
        A = float(beta.sum())
        if A <= 0:
            raise ValueError("each beta_params entry must have positive total concentration")
        if fracs.ndim != 1:
            raise ValueError("each daily_split must be a 1-D array of on-line fractions")
        mu = beta[0] / A
        c = float(mu * (1.0 - mu))
        terms.append((c, A))
        v_total += float(((fracs - mu) ** 2).mean())

    def f(psi: float) -> float:
        return sum(c / (A * psi + 1.0) for c, A in terms)

    if v_total <= f(max_psi):
        return float(max_psi)
    if v_total >= f(min_psi):
        return float(min_psi)
    lo, hi = min_psi, max_psi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < v_total:  # f decreasing: mid too large
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def collect_overdispersion_terms(
    on_od_sum,
    daily_on_od,
    daily_on_total,
    alpha_prior: float = 1.0,
):
    """Collect ``(alphas, daily_shares)`` inputs for :func:`fit_overdispersion_phi`.

    For each weekday and origin with on-line training flow, returns the fitted
    concentration row (``fit_online_dest_alpha``) and the observed daily
    destination-share vectors. Days where the origin had zero on-line flow are
    excluded (they would otherwise inflate the variance); origins with fewer
    than two valid days carry no day-to-day variance and are skipped.

    Args:
        on_od_sum: weekday -> ``[S, S]`` aggregated on-line OD counts.
        daily_on_od: weekday -> list of ``[S, S]`` per-day on-line OD counts.
        daily_on_total: weekday -> list of ``[S]`` per-day on-line totals per
            origin (must be aligned with ``daily_on_od``).
        alpha_prior: symmetric Dirichlet prior passed to ``fit_online_dest_alpha``.

    Returns:
        ``(alphas, daily_shares)`` where ``alphas`` is a list of ``[S]``
        concentration rows and ``daily_shares`` is a list of ``[n_valid, S]``
        observed daily share matrices, one entry per surviving origin.
    """
    alphas = []
    daily_shares = []
    for w in on_od_sum:
        alpha_w = fit_online_dest_alpha(on_od_sum[w], alpha_prior)
        days_od = daily_on_od.get(w, [])
        days_total = daily_on_total.get(w, [])
        for i in range(alpha_w.shape[0]):
            if float(on_od_sum[w][i].sum()) <= 0:
                continue
            shares = []
            for od, tot in zip(days_od, days_total):
                if float(tot[i]) > 0:
                    shares.append(od[i] / tot[i])
            if len(shares) < 2:
                continue
            alphas.append(alpha_w[i])
            daily_shares.append(np.asarray(shares, dtype=np.float64))
    return alphas, daily_shares


def collect_overdispersion_split_terms(
    on_total,
    off_total,
    daily_on_total,
    daily_off_total,
    a_on: float = 1.0,
    a_off: float = 1.0,
):
    """Collect ``(beta_params, daily_split)`` inputs for :func:`fit_overdispersion_split_psi`.

    For each weekday and origin with on-line + off-line training flow, returns the
    fitted Beta split parameters (``fit_online_split``) and the observed daily
    on-line fractions ``s_d = on_d / (on_d + off_d)``. Days where the origin had
    zero total flow are excluded (they would spuriously deflate ``s_d``); origins
    with fewer than two valid days carry no day-to-day variance and are skipped.

    Args:
        on_total: weekday -> ``[S]`` aggregated on-line totals per origin.
        off_total: weekday -> ``[S]`` aggregated off-line totals per origin.
        daily_on_total: weekday -> list of ``[S]`` per-day on-line totals.
        daily_off_total: weekday -> list of ``[S]`` per-day off-line totals
            (aligned with ``daily_on_total``).
        a_on, a_off: Beta prior parameters passed to ``fit_online_split``.

    Returns:
        ``(beta_params, daily_split)`` where ``beta_params`` is a list of length-2
        arrays ``(a_on, a_off)`` and ``daily_split`` is a list of ``[n_valid]``
        on-line-fraction arrays, one entry per surviving origin.
    """
    beta_params = []
    daily_split = []
    for w in on_total:
        split_w = fit_online_split(on_total[w], off_total[w], a_on, a_off)
        days_on = daily_on_total.get(w, [])
        days_off = daily_off_total.get(w, [])
        for i in range(split_w.shape[0]):
            total = float(on_total[w][i]) + float(off_total[w][i])
            if total <= 0:
                continue
            fracs = []
            for on_d, off_d in zip(days_on, days_off):
                tot = float(on_d[i]) + float(off_d[i])
                if tot > 0:
                    fracs.append(float(on_d[i]) / tot)
            if len(fracs) < 2:
                continue
            beta_params.append(split_w[i])
            daily_split.append(np.asarray(fracs, dtype=np.float64))
    return beta_params, daily_split


def _sample_dirichlet_nonzero(alpha_row: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Dirichlet sample over the positive-concentration entries, zeros elsewhere.

    ``numpy.random.RandomState.dirichlet`` raises on ``alpha <= 0``, so the
    forbidden (zero-concentration) categories (the diagonal self-loop) are
    excluded from the draw and left at exactly zero. The positive entries
    remain a proper simplex (they sum to one).
    """
    alpha_row = np.asarray(alpha_row, dtype=np.float64)
    pos = alpha_row > 0
    out = np.zeros_like(alpha_row, dtype=np.float64)
    if pos.any():
        out[pos] = rng.dirichlet(alpha_row[pos])
    return out


def sample_online_alighting(
    entry: np.ndarray,
    online_split_params: np.ndarray,
    online_dest_alpha: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
    dest_overdispersion: float = 1.0,
    split_overdispersion: float = 1.0,
) -> np.ndarray:
    """On-line alighting posterior (column sums of the on-line OD).

    Args:
        entry: [S] target-day entry counts.
        online_split_params: [S, 2] ``(a_on, a_off)`` Beta parameters.
        online_dest_alpha: [S, S] Dirichlet concentration (diagonal zero).
        n_samples: number of posterior draws.
        rng: ``numpy.random.RandomState``.
        dest_overdispersion: shared between-day over-dispersion factor ``phi``
            in (0, 1]. Scales the destination-share concentration to
            ``Dirichlet(alpha * phi)``, keeping the mean and inflating the
            variance. ``1.0`` (default) reproduces the static posterior.
        split_overdispersion: shared between-day over-dispersion factor ``psi``
            in (0, 1]. Scales the on-line/off-line Beta split to
            ``Beta(a_on * psi, a_off * psi)``, keeping the on-line mean and
            inflating the split variance. ``1.0`` (default) reproduces the static
            split posterior.

    Returns:
        [n_samples, S]; ``A[m, j]`` is the m-th draw of on-line alighting at
        station ``j``.
    """
    entry = np.asarray(entry, dtype=np.float64)
    online_split_params = np.asarray(online_split_params, dtype=np.float64)
    online_dest_alpha = np.asarray(online_dest_alpha, dtype=np.float64)
    if dest_overdispersion <= 0.0:
        raise ValueError("dest_overdispersion must be positive")
    if dest_overdispersion > 1.0:
        raise ValueError("dest_overdispersion must be <= 1.0 (shrink only)")
    if split_overdispersion <= 0.0:
        raise ValueError("split_overdispersion must be positive")
    if split_overdispersion > 1.0:
        raise ValueError("split_overdispersion must be <= 1.0 (shrink only)")
    dest_alpha = online_dest_alpha * dest_overdispersion
    S = entry.shape[0]
    if entry.ndim != 1:
        raise ValueError("entry must be 1-D")
    if online_split_params.shape != (S, 2):
        raise ValueError("online_split_params must be [S, 2]")
    if online_dest_alpha.ndim != 2 or online_dest_alpha.shape != (S, S):
        raise ValueError("online_dest_alpha must be [S, S]")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    a_on = online_split_params[:, 0] * split_overdispersion
    a_off = online_split_params[:, 1] * split_overdispersion
    A = np.empty((n_samples, S), dtype=np.float64)
    for m in range(n_samples):
        s = rng.beta(a_on, a_off)  # [S]
        on_od = np.empty_like(dest_alpha)
        for i in range(S):
            q = _sample_dirichlet_nonzero(dest_alpha[i], rng)
            on_od[i] = entry[i] * s[i] * q
        A[m] = on_od.sum(axis=0)
    return A


def sample_line_od_posterior(
    entry: np.ndarray,
    online_split_params: np.ndarray,
    online_dest_alpha: np.ndarray,
    offline_alpha: np.ndarray,
    route_probs: np.ndarray,
    interchange_cols: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Full line OD posterior (on-line + transfer-out), fixed route choice.

    Args:
        entry: [S] target-day entry counts.
        online_split_params: [S, 2] ``(a_on, a_off)`` Beta parameters.
        online_dest_alpha: [S, S] on-line Dirichlet concentration (diag zero).
        offline_alpha: [S, n_off] off-line Dirichlet concentration.
        route_probs: [S, n_off, K] ``p(interchange k | origin i, dest d)``.
        interchange_cols: [K] int column indices (into S) of the interchanges.
        n_samples: number of posterior draws.
        rng: ``numpy.random.RandomState``.

    Returns:
        [n_samples, S, S]; ``od[m, i, j]`` is the m-th draw of flow i -> j.
        Row sums equal ``entry`` exactly when ``route_probs`` rows sum to one
        for every (i, d) with positive off-line mass (mass is otherwise left
        unassigned, matching the conservation accounting).
    """
    entry = np.asarray(entry, dtype=np.float64)
    online_split_params = np.asarray(online_split_params, dtype=np.float64)
    online_dest_alpha = np.asarray(online_dest_alpha, dtype=np.float64)
    offline_alpha = np.asarray(offline_alpha, dtype=np.float64)
    route_probs = np.asarray(route_probs, dtype=np.float64)
    interchange_cols = np.asarray(interchange_cols, dtype=np.int64)

    S = entry.shape[0]
    if entry.ndim != 1:
        raise ValueError("entry must be 1-D")
    if online_split_params.shape != (S, 2):
        raise ValueError("online_split_params must be [S, 2]")
    if online_dest_alpha.shape != (S, S):
        raise ValueError("online_dest_alpha must be [S, S]")
    if offline_alpha.ndim != 2 or offline_alpha.shape[0] != S:
        raise ValueError("offline_alpha must be [S, n_off]")
    if route_probs.ndim != 3 or route_probs.shape[0] != S:
        raise ValueError("route_probs must be [S, n_off, K]")
    if route_probs.shape[1] != offline_alpha.shape[1]:
        raise ValueError("route_probs and offline_alpha must share the destination dim")
    if interchange_cols.shape != (route_probs.shape[2],):
        raise ValueError("interchange_cols must have one entry per route_probs K")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    a_on = online_split_params[:, 0]
    a_off = online_split_params[:, 1]
    od = np.empty((n_samples, S, S), dtype=np.float64)
    for m in range(n_samples):
        s = rng.beta(a_on, a_off)  # [S]
        on_mass = entry * s
        off_mass = entry * (1.0 - s)
        od_m = np.empty((S, S), dtype=np.float64)
        phi = np.empty_like(offline_alpha)
        for i in range(S):
            od_m[i] = on_mass[i] * _sample_dirichlet_nonzero(online_dest_alpha[i], rng)
            phi[i] = rng.dirichlet(offline_alpha[i])
        dest_mass = off_mass[:, None] * phi  # [S, n_off]
        # Per-origin transfer-out flow: sum over off-line destinations only,
        # keeping the origin axis, so each origin's off-line mass is attributed
        # to its own interchange cells (not the network total).
        transfer = (dest_mass[:, :, None] * route_probs).sum(axis=1)  # [S, K]
        od_m[:, interchange_cols] += transfer
        od[m] = od_m
    return od


def alighting_posterior(od_samples: np.ndarray) -> np.ndarray:
    """Column sums (per-station alighting) of a full OD posterior.

    Args:
        od_samples: [n_samples, S, S].

    Returns:
        [n_samples, S]; ``A[m, j]`` is the m-th draw of total alighting at
        station ``j`` (on-line + transfer-out where applicable).
    """
    samples = np.asarray(od_samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[1] != samples.shape[2]:
        raise ValueError("od_samples must be [n_samples, S, S]")
    return samples.sum(axis=1)


def sample_line_od_posterior_eta_mixture(
    entry: np.ndarray,
    online_split_params: np.ndarray,
    online_dest_alpha: np.ndarray,
    offline_alpha: np.ndarray,
    route_probs_list: list,
    eta_weights: np.ndarray,
    interchange_cols: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Full line OD posterior marginalized over a discrete eta prior.

    Marginalizes the unidentifiable route-choice dispersion (posterior == prior)
    so the interchange split uncertainty appears inside the full OD posterior as
    variance at the interchange columns, not as a separate sensitivity analysis.

    Args:
        entry, online_split_params, online_dest_alpha, offline_alpha: as in
            :func:`sample_line_od_posterior`.
        route_probs_list: list of [S, n_off, K] tensors, one per eta grid point.
        eta_weights: [n_eta] prior masses (normalized internally).
        interchange_cols: [K] int column indices of the interchanges.
        n_samples: total posterior draws.
        rng: ``numpy.random.RandomState``.

    Returns:
        [n_samples, S, S] full OD posterior integrated over the eta prior.
    """
    route_probs_list = [np.asarray(rp, dtype=np.float64) for rp in route_probs_list]
    weights = np.asarray(eta_weights, dtype=np.float64)
    weights = weights / weights.sum()
    n_eta = len(route_probs_list)
    if n_eta != weights.shape[0]:
        raise ValueError("route_probs_list and eta_weights must have matching length")
    eta_idx = rng.choice(n_eta, size=n_samples, p=weights)
    S = np.asarray(entry, dtype=np.float64).shape[0]
    od = np.empty((n_samples, S, S), dtype=np.float64)
    for j in range(n_eta):
        mask = eta_idx == j
        nj = int(mask.sum())
        if nj == 0:
            continue
        od[mask] = sample_line_od_posterior(
            entry, online_split_params, online_dest_alpha, offline_alpha,
            route_probs_list[j], interchange_cols, n_samples=nj, rng=rng,
        )
    return od
