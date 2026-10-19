"""Interface-level transfer-flow reliability: attribution, coupling, assembly.

Conjugate Bayesian attribution of a line's transfer destination flow to its
interchange stations. For each origin the transfer destination shares follow a
Dirichlet posterior and the on-line/transfer split follows a Beta posterior;
the interchange attribution is a logit route-choice probability (see
:mod:`passenger_state.network`). The transfer-flow posterior is sampled by
Monte Carlo and reported as a distribution (variance is the primary object).

The route-choice dispersion ``eta`` is unidentifiable from fare-collection
counts (the interchange split is unobservable), so its posterior equals its
prior; marginalizing over a discrete log-spaced grid propagates that
uncertainty into every downstream posterior (companion paper, Eq. (5)).

Sampling uses only NumPy's ``RandomState``; there is no MCMC dependency.
"""

import numpy as np

# Log-uniform grid for the eta-marginalized posterior: uniform weights on a
# log-spaced grid approximate a log-uniform prior over the dispersion
# (companion paper, Eq. (5)).
MIXTURE_ETA_GRID = list(np.geomspace(0.1, 10.0, 9))

# Default number of posterior draws.
N_SAMPLES = 2000

# Default conjugate priors shared by the transfer and reconstruction layers.
ALPHA_PRIOR = 1.0
A_ON = 1.0
A_OFF = 1.0


def fit_offline_shares(train_off_counts: np.ndarray, alpha_prior: float = ALPHA_PRIOR) -> np.ndarray:
    """Dirichlet posterior concentration for transfer destination shares.

    Args:
        train_off_counts: [S, n_off] aggregated transfer destination counts per
            origin over training days.
        alpha_prior: symmetric Dirichlet prior (Laplace smoothing).

    Returns:
        [S, n_off] concentration parameters ``alpha_prior + counts``.
    """
    counts = np.asarray(train_off_counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError("train_off_counts must be a 2-D [S, n_off] array")
    if (counts < 0).any():
        raise ValueError("train_off_counts must be non-negative")
    return counts + alpha_prior


def fit_online_split(
    train_on_total: np.ndarray,
    train_off_total: np.ndarray,
    a_on: float = A_ON,
    a_off: float = A_OFF,
) -> np.ndarray:
    """Beta posterior parameters for the on-line/transfer split per origin.

    Args:
        train_on_total: [S] total on-line flow per origin over training days.
        train_off_total: [S] total transfer flow per origin over training days.
        a_on, a_off: Beta prior parameters.

    Returns:
        [S, 2] array of ``(a_on + on_total, a_off + off_total)`` per origin.
    """
    on = np.asarray(train_on_total, dtype=np.float64)
    off = np.asarray(train_off_total, dtype=np.float64)
    if on.ndim != 1 or off.shape != on.shape:
        raise ValueError("train_on_total and train_off_total must be equal-length 1-D arrays")
    if (on < 0).any() or (off < 0).any():
        raise ValueError("training totals must be non-negative")
    return np.stack([on + a_on, off + a_off], axis=1)


def sample_transfer_flow(
    entry: np.ndarray,
    online_split_params: np.ndarray,
    offline_alpha: np.ndarray,
    route_probs: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Monte Carlo samples of the per-interchange transfer-out flow.

    For each sample, draws the transfer split ``s_i ~ Beta`` and the transfer
    destination shares ``q_i ~ Dirichlet``, then attributes the resulting
    off-line mass over interchanges via the fixed route-choice probabilities.

    Args:
        entry: [S] target-day entry counts.
        online_split_params: [S, 2] ``(a_on, a_off)`` Beta parameters.
        offline_alpha: [S, n_off] Dirichlet concentrations.
        route_probs: [S, n_off, K] ``p(interchange k | origin i, dest d)``;
            rows sum to at most one (zero rows leave mass unassigned).
        n_samples: number of posterior draws.
        rng: ``numpy.random.RandomState`` for reproducibility.

    Returns:
        [n_samples, K] array; ``T[m, k]`` is the m-th draw of transfer-out flow
        at interchange ``k``.
    """
    entry = np.asarray(entry, dtype=np.float64)
    online_split_params = np.asarray(online_split_params, dtype=np.float64)
    offline_alpha = np.asarray(offline_alpha, dtype=np.float64)
    route_probs = np.asarray(route_probs, dtype=np.float64)

    S = entry.shape[0]
    if entry.ndim != 1:
        raise ValueError("entry must be 1-D")
    if online_split_params.shape != (S, 2):
        raise ValueError("online_split_params must be [S, 2]")
    if offline_alpha.ndim != 2 or offline_alpha.shape[0] != S:
        raise ValueError("offline_alpha must be [S, n_off]")
    if route_probs.ndim != 3 or route_probs.shape[0] != S:
        raise ValueError("route_probs must be [S, n_off, K]")
    if route_probs.shape[1] != offline_alpha.shape[1]:
        raise ValueError("route_probs and offline_alpha must share the destination dim")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    a_on = online_split_params[:, 0]
    a_off = online_split_params[:, 1]
    K = route_probs.shape[2]

    T = np.empty((n_samples, K), dtype=np.float64)
    for m in range(n_samples):
        s = rng.beta(a_on, a_off)  # [S]
        phi = np.empty_like(offline_alpha)
        for i in range(S):
            phi[i] = rng.dirichlet(offline_alpha[i])  # [n_off]
        off_mass = entry * (1.0 - s)  # [S]
        dest_mass = off_mass[:, None] * phi  # [S, n_off]
        T[m] = (dest_mass[:, :, None] * route_probs).sum(axis=(0, 1))
    return T


def sample_transfer_flow_eta_mixture(
    entry: np.ndarray,
    online_split_params: np.ndarray,
    offline_alpha: np.ndarray,
    route_probs_list: list,
    eta_weights: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sample transfer-out flow marginalized over a discrete eta prior.

    The route-choice dispersion ``eta`` is unidentifiable from fare-collection
    counts (the interchange split is unobservable), so its posterior equals its
    prior. Marginalizing over a discrete prior on ``eta`` propagates that
    route-choice uncertainty into the transfer-flow posterior, making the
    unobservable split visible as posterior variance rather than as a separate
    sensitivity analysis.

    Args:
        entry, online_split_params, offline_alpha: as in
            :func:`sample_transfer_flow`.
        route_probs_list: list of [S, n_off, K] tensors, one per eta grid point.
        eta_weights: [n_eta] prior masses (normalized internally).
        n_samples: total posterior draws.
        rng: ``numpy.random.RandomState``.

    Returns:
        [n_samples, K] transfer-out flow draws integrated over the eta prior.
    """
    route_probs_list = [np.asarray(rp, dtype=np.float64) for rp in route_probs_list]
    weights = np.asarray(eta_weights, dtype=np.float64)
    weights = weights / weights.sum()
    n_eta = len(route_probs_list)
    if n_eta != weights.shape[0]:
        raise ValueError("route_probs_list and eta_weights must have matching length")
    eta_idx = rng.choice(n_eta, size=n_samples, p=weights)
    K = route_probs_list[0].shape[2]
    T = np.empty((n_samples, K), dtype=np.float64)
    for j in range(n_eta):
        mask = eta_idx == j
        nj = int(mask.sum())
        if nj == 0:
            continue
        T[mask] = sample_transfer_flow(
            entry, online_split_params, offline_alpha, route_probs_list[j],
            n_samples=nj, rng=rng,
        )
    return T


def transfer_flow_summary(samples: np.ndarray) -> dict:
    """Posterior summaries for per-interchange transfer-out flow.

    Args:
        samples: [n_samples, K] transfer-out flow draws.

    Returns:
        dict with array-valued keys ``mean``, ``std``, ``cv``, ``ci95_lower``
        and ``ci95_upper``, each of shape [K]. ``cv`` is ``std/mean`` and is
        NaN where the mean is zero.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError("samples must be a 2-D [n_samples, K] array")
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = std / mean
    return {
        "mean": mean,
        "std": std,
        "cv": cv,
        "ci95_lower": np.percentile(samples, 2.5, axis=0),
        "ci95_upper": np.percentile(samples, 97.5, axis=0),
    }


def sample_transfer_coupling(
    entry: np.ndarray,
    online_split_params: np.ndarray,
    offline_alpha: np.ndarray,
    route_probs: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Monte Carlo destination-station transfer coupling ``[n_samples, n_off, K]``.

    Identical draws to :func:`sample_transfer_flow`, but the destination axis is
    retained (sum over origins only), so ``C[m, d, k]`` is the m-th draw of flow
    boarding the line and alighting at interchange ``k`` with transfer
    destination ``d``. ``C.sum(axis=1)`` reproduces ``sample_transfer_flow``.
    """
    entry = np.asarray(entry, dtype=np.float64)
    online_split_params = np.asarray(online_split_params, dtype=np.float64)
    offline_alpha = np.asarray(offline_alpha, dtype=np.float64)
    route_probs = np.asarray(route_probs, dtype=np.float64)

    S = entry.shape[0]
    if entry.ndim != 1:
        raise ValueError("entry must be 1-D")
    if online_split_params.shape != (S, 2):
        raise ValueError("online_split_params must be [S, 2]")
    if offline_alpha.ndim != 2 or offline_alpha.shape[0] != S:
        raise ValueError("offline_alpha must be [S, n_off]")
    if route_probs.ndim != 3 or route_probs.shape[0] != S:
        raise ValueError("route_probs must be [S, n_off, K]")
    if route_probs.shape[1] != offline_alpha.shape[1]:
        raise ValueError("route_probs and offline_alpha must share the destination dim")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    a_on = online_split_params[:, 0]
    a_off = online_split_params[:, 1]
    n_off = offline_alpha.shape[1]
    K = route_probs.shape[2]

    C = np.empty((n_samples, n_off, K), dtype=np.float64)
    for m in range(n_samples):
        s = rng.beta(a_on, a_off)  # [S]
        phi = np.empty_like(offline_alpha)
        for i in range(S):
            phi[i] = rng.dirichlet(offline_alpha[i])  # [n_off]
        off_mass = entry * (1.0 - s)  # [S]
        dest_mass = off_mass[:, None] * phi  # [S, n_off]
        C[m] = (dest_mass[:, :, None] * route_probs).sum(axis=0)  # [n_off, K]
    return C


def sample_transfer_coupling_eta_mixture(
    entry: np.ndarray,
    online_split_params: np.ndarray,
    offline_alpha: np.ndarray,
    route_probs_list: list,
    eta_weights: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Destination-station coupling marginalized over a discrete eta prior.

    Mirrors :func:`sample_transfer_flow_eta_mixture` with the destination axis
    retained; ``C.sum(axis=1)`` reproduces the flow marginal.
    """
    route_probs_list = [np.asarray(rp, dtype=np.float64) for rp in route_probs_list]
    weights = np.asarray(eta_weights, dtype=np.float64)
    weights = weights / weights.sum()
    n_eta = len(route_probs_list)
    if n_eta != weights.shape[0]:
        raise ValueError("route_probs_list and eta_weights must have matching length")
    eta_idx = rng.choice(n_eta, size=n_samples, p=weights)
    n_off = np.asarray(offline_alpha, dtype=np.float64).shape[1]
    K = route_probs_list[0].shape[2]
    C = np.empty((n_samples, n_off, K), dtype=np.float64)
    for j in range(n_eta):
        mask = eta_idx == j
        nj = int(mask.sum())
        if nj == 0:
            continue
        C[mask] = sample_transfer_coupling(
            entry, online_split_params, offline_alpha, route_probs_list[j],
            n_samples=nj, rng=rng,
        )
    return C


def _project(coupling_samples, dest_line_sets, line_numbers, weights_by_dest):
    """Shared projection: scatter destination mass over serving lines."""
    coupling_samples = np.asarray(coupling_samples, dtype=np.float64)
    if coupling_samples.ndim != 3:
        raise ValueError("coupling_samples must be [n_samples, n_off, K]")
    if len(dest_line_sets) != coupling_samples.shape[1]:
        raise ValueError("dest_line_sets must have one entry per destination")
    line_numbers = tuple(line_numbers)
    line_idx = {L: j for j, L in enumerate(line_numbers)}
    pairs_by_dest = []
    for lines in dest_line_sets:
        lines = tuple(lines)
        if not lines:
            raise ValueError("empty line set for a destination; cannot conserve mass")
        pairs_by_dest.append([(line_idx[L], float(w)) for L, w in zip(lines, weights_by_dest(lines))])
    n_lines = len(line_numbers)
    K = coupling_samples.shape[2]
    T = np.zeros((coupling_samples.shape[0], n_lines, K), dtype=np.float64)
    for d, pairs in enumerate(pairs_by_dest):
        for j, w in pairs:
            T[:, j, :] += w * coupling_samples[:, d, :]
    return T


def project_coupling_to_lines(coupling_samples, dest_line_sets, line_numbers):
    """Line-level coupling with a uniform soft split (1/m) over each destination's
    serving lines. Weights sum to one per destination, so ``T.sum(axis=1)`` equals
    ``coupling_samples.sum(axis=1)`` exactly (mass conserved)."""
    return _project(
        coupling_samples, dest_line_sets, line_numbers,
        lambda lines: [1.0 / len(lines) for _ in lines],
    )


def coupling_summary(coupling_samples, line_numbers, interchanges):
    """Per-(line, interchange) posterior summary of ``[n_samples, n_lines, K]``."""
    samples = np.asarray(coupling_samples, dtype=np.float64)
    if samples.ndim != 3:
        raise ValueError("coupling_samples must be [n_samples, n_lines, K]")
    if samples.shape[1] != len(line_numbers):
        raise ValueError("coupling_samples second axis must match line_numbers")
    if samples.shape[2] != len(interchanges):
        raise ValueError("coupling_samples third axis must match interchanges")
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = std / mean
    return {
        "mean": mean,
        "std": std,
        "cv": cv,
        "ci95_lower": np.percentile(samples, 2.5, axis=0),
        "ci95_upper": np.percentile(samples, 97.5, axis=0),
    }


def assemble_network(line_samples, line_interchanges):
    """Sum per-line transfer-out samples into per-interchange network flow.

    Each passenger boards exactly one line, so summing the per-line transfer-out
    marginals over the boarding lines gives the network-level transfer flow at
    each interchange without double counting. Per-line draws are independent, so
    the element-wise sum is a valid Monte Carlo estimate of the aggregate.

    Args:
        line_samples: list of [n_samples, K_L] arrays (one per line).
        line_interchanges: list of [K_L] interchange-ID lists aligned with
            ``line_samples``.

    Returns:
        dict mapping interchange ID -> [n_samples] summed draws. Every line's
        ``n_samples`` must match.
    """
    if len(line_samples) != len(line_interchanges):
        raise ValueError("line_samples and line_interchanges must have matching length")
    if not line_samples:
        return {}
    n_samples = line_samples[0].shape[0]
    net = {}
    for samples, interchanges in zip(line_samples, line_interchanges):
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] != n_samples:
            raise ValueError(
                "all line samples must be [n_samples, K_L] with matching n_samples"
            )
        if len(interchanges) != samples.shape[1]:
            raise ValueError("each line_interchanges must match its samples' K")
        for j, k in enumerate(interchanges):
            col = samples[:, j]
            if k in net:
                net[k] = net[k] + col
            else:
                net[k] = col.copy()
    return net
