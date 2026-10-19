"""End-to-end demonstration of the passenger-state pipeline.

Two parts covering the companion paper's three layers:

1. Real-format path (MetroFlow CSV -> canonical parquet -> Layers 1-2).
   The bundled ``tests/fixtures/metroflow_tiny`` CSVs are canonicalized, the
   3-station demo line is reconstructed with closed-form destination shares
   (Layer 1), and the between-day over-dispersion calibration ``phi`` is fitted
   on simulated training days and shown to restore interval coverage (Layer 2).

2. Mechanism path (Layers 3 + validation, on a small example network).
   A 7-station line with three interchanges and three off-line destinations
   exercises the eta-marginalized transfer posterior, the gravity
   mechanism-divergence check, and the mass-conserving coupling projection.

Run: python scripts/run_demo.py [--raw-root DIR] [--output-root DIR]
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import duckdb
import numpy as np

from passenger_state.data import (
    CanonicalizeConfig,
    canonicalize,
    load_station_graph,
    read_day_flows,
    read_day_online_alighting,
    read_day_trips,
    weekday,
)
from passenger_state.network import (
    build_graph,
    build_route_probs,
    dijkstra_predecessors,
)
from passenger_state.reconstruction import (
    build_line_od,
    fit_destination_shares,
    fit_online_dest_alpha,
    fit_overdispersion_phi,
    predict_od,
    sample_online_alighting,
)
from passenger_state.reliability import (
    MIXTURE_ETA_GRID,
    N_SAMPLES,
    fit_offline_shares,
    fit_online_split,
    project_coupling_to_lines,
    sample_transfer_coupling_eta_mixture,
    sample_transfer_flow_eta_mixture,
    transfer_flow_summary,
)
from passenger_state.validation import (
    gravity_transfer_probs,
    simulate_s2_world,
    s2_rank_correlation,
    s2_relative_error,
    wape,
)

TRAIN_DATE = "2017-05-01"
EVAL_DATE = "2017-08-01"

# Demo network for the mechanism path: a 7-station line with interchanges
# 2009/2020/2033 and off-line destinations 1121 (via 2009), 771 (via 2020)
# and 99 (via 2033).
DEMO_LINE = (1321, 1322, 2009, 1329, 2020, 2033, 1331)
DEMO_NODES = [1321, 1322, 2009, 1329, 2020, 2033, 1331, 1121, 771, 99]
DEMO_NEIGHBOURS = [
    [1322],                    # 1321
    [1321, 2009],              # 1322
    [1322, 1329, 1121],        # 2009
    [2009, 2020],              # 1329
    [1329, 2033, 771],         # 2020
    [2020, 1331, 99],          # 2033
    [2033],                    # 1331
    [2009],                    # 1121
    [2020],                    # 771
    [2033],                    # 99
]
DEMO_OFF_DESTS = [1121, 771, 99]
DEMO_DEST_LINES = {1121: (4,), 771: (7, 2), 99: (2,)}

DEMO_ON_TOTAL = np.array([60.0, 40.0, 0.0, 0.0, 30.0, 0.0, 0.0])
DEMO_OFF_TOTAL = np.array([40.0, 10.0, 0.0, 0.0, 20.0, 0.0, 0.0])
DEMO_OFF_COUNTS = np.array(
    [
        [30.0, 10.0, 0.0],
        [5.0, 5.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 10.0, 10.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
)
DEMO_ENTRY = np.array([100.0, 50.0, 0.0, 0.0, 50.0, 0.0, 0.0])


def demo_layer1(con, canonical_root, raw_root):
    """Layer 1: canonicalize -> graph -> closed-form line OD -> held-out WAPE."""
    station_ids, graph = load_station_graph(con, str(raw_root / "stationInfo.csv"))
    stations = tuple(station_ids)
    order = {sid: i for i, sid in enumerate(stations)}
    pred_trees = {s: dijkstra_predecessors(graph, s)[1] for s in stations}

    train_trips = read_day_trips(con, canonical_root, TRAIN_DATE, "train", stations)
    train_od, unassigned = build_line_od(stations, train_trips, pred_trees)
    train_entry = read_day_flows(con, canonical_root, TRAIN_DATE, "entry_count", stations, order)
    shares = fit_destination_shares(train_od, train_entry)

    eval_entry = read_day_flows(con, canonical_root, EVAL_DATE, "entry_count", stations, order)
    eval_alighting = read_day_online_alighting(
        con, canonical_root, EVAL_DATE, "test", stations, order
    )
    predicted = predict_od(eval_entry, shares).sum(axis=0)
    error = wape(eval_alighting, predicted)
    print(f"[layer 1] stations={stations} train day={TRAIN_DATE} (weekday "
          f"{weekday(TRAIN_DATE)}) eval day={EVAL_DATE} (weekday {weekday(EVAL_DATE)})")
    print(f"[layer 1] unassigned flow={unassigned:.0f}  held-out alighting WAPE={error:.3f}")
    return stations, order, train_od, train_entry


def _draw_offdiag_dirichlet(alpha_row, rng):
    """Dirichlet draw over the positive (non-self-loop) entries, zeros elsewhere."""
    q = np.zeros_like(alpha_row, dtype=np.float64)
    mask = alpha_row > 0
    q[mask] = rng.dirichlet(alpha_row[mask])
    return q


def demo_layer2(stations, train_od, train_entry):
    """Layer 2: between-day over-dispersion phi restores interval coverage."""
    rng = np.random.RandomState(7)
    # The fixture's raw counts are so concentrated that day-to-day share
    # variation is invisible; scale the counts down (keeping the share shape)
    # to a level where between-day dispersion is identifiable.
    alpha = fit_online_dest_alpha(train_od / 40.0)
    on_share = 0.8
    phi_true = 0.05
    n_train_days = 9

    # Simulated training days: each day draws its own over-dispersed share
    # vector per origin, the mechanism the static posterior misses.
    daily_shares = []
    for _ in range(n_train_days):
        q = np.array([
            _draw_offdiag_dirichlet(alpha[i] * phi_true, rng)
            for i in range(len(stations))
        ])
        daily_shares.append(q)
    phi_hat = fit_overdispersion_phi(
        [alpha[i] for i in range(len(stations)) if train_entry[i] > 0],
        [np.asarray([d[i] for d in daily_shares]) for i in range(len(stations))
         if train_entry[i] > 0],
    )

    # Coverage of the held-out day's alighting: static (phi=1) vs calibrated.
    n_rep, n_draws = 40, 400
    split = np.stack([train_entry * on_share, train_entry * (1 - on_share)], axis=1)
    coverage = {"static": [], "calibrated": []}
    for rep in range(n_rep):
        q_eval = np.array([
            _draw_offdiag_dirichlet(alpha[i] * phi_true, rng)
            for i in range(len(stations))
        ])
        truth = (train_entry[:, None] * on_share * q_eval).sum(axis=0)
        for label, factor in (("static", 1.0), ("calibrated", phi_hat)):
            draws = sample_online_alighting(
                train_entry, split, alpha, n_draws,
                np.random.RandomState(100 + rep),
                dest_overdispersion=factor,
            )
            lo = np.percentile(draws, 2.5, axis=0)
            hi = np.percentile(draws, 97.5, axis=0)
            coverage[label].append(float(((truth >= lo) & (truth <= hi)).mean()))
    cov_static = float(np.mean(coverage["static"]))
    cov_calib = float(np.mean(coverage["calibrated"]))
    print(f"[layer 2] phi (true {phi_true}) estimated at {phi_hat:.3f} from "
          f"{n_train_days} simulated days")
    print(f"[layer 2] 95% interval coverage: static {cov_static:.2f} -> "
          f"calibrated {cov_calib:.2f}")
    return phi_hat, cov_static, cov_calib


def demo_layer3():
    """Layer 3 + validation: transfer posterior, gravity check, coupling."""
    rng = np.random.RandomState(0)
    order = {s: i for i, s in enumerate(DEMO_LINE)}
    graph = build_graph(DEMO_NODES, DEMO_NEIGHBOURS, [121.0] * len(DEMO_NODES), [31.0] * len(DEMO_NODES))
    line_interchanges = [k for k in DEMO_LINE if k >= 2000]
    pred_trees = {o: dijkstra_predecessors(graph, o)[1] for o in DEMO_LINE}
    k_dist = {k: dijkstra_predecessors(graph, k)[0] for k in line_interchanges}

    route_list, weights = [], np.ones(len(MIXTURE_ETA_GRID))
    for eta in MIXTURE_ETA_GRID:
        probs, interchanges = build_route_probs(
            DEMO_LINE, order, DEMO_OFF_DESTS, pred_trees, k_dist, eta=eta,
            interchanges=line_interchanges,
        )
        route_list.append(probs)

    split = fit_online_split(DEMO_ON_TOTAL, DEMO_OFF_TOTAL)
    alpha = fit_offline_shares(DEMO_OFF_COUNTS)
    T = sample_transfer_flow_eta_mixture(
        DEMO_ENTRY, split, alpha, route_list, weights, N_SAMPLES, rng
    )
    summary = transfer_flow_summary(T)
    print(f"[layer 3] transfer-out posterior at interchanges {interchanges}: "
          f"mean={np.round(summary['mean'], 1)}, cv={np.round(summary['cv'], 3)}")

    # Mechanism-divergent check: gravity truth, logit inference.
    attraction = {2009: 1.6, 2020: 1.0, 2033: 1.3}
    p_gravity, _ = gravity_transfer_probs(
        DEMO_LINE, order, DEMO_OFF_DESTS, pred_trees, k_dist, attraction,
        gamma=5.0, interchanges=line_interchanges,
    )
    totals = DEMO_ON_TOTAL + DEMO_OFF_TOTAL
    off_share = np.divide(
        DEMO_OFF_TOTAL, totals, out=np.zeros_like(totals), where=totals > 0
    )
    phi_point = alpha / alpha.sum(axis=1, keepdims=True)
    T_true = simulate_s2_world(DEMO_ENTRY, off_share, phi_point, p_gravity)
    rel_err, active = s2_relative_error(T_true, T)
    corr = s2_rank_correlation(summary["cv"][active], rel_err[active])
    print(f"[layer 3] mechanism-divergent world: relative error "
          f"{np.round(rel_err[active], 3)}, rank corr (cv vs error) "
          f"{corr['spearman']:.2f} (p={corr['permutation_pvalue']:.3f})")

    # Coupling projection: destination mass scattered over serving lines,
    # conserved exactly.
    C = sample_transfer_coupling_eta_mixture(
        DEMO_ENTRY, split, alpha, route_list, weights, N_SAMPLES,
        np.random.RandomState(1),
    )
    T_lines = project_coupling_to_lines(C, [DEMO_DEST_LINES[d] for d in DEMO_OFF_DESTS], (2, 4, 7))
    residual = float(
        np.abs(T_lines.sum(axis=(1, 2)) - C.sum(axis=(1, 2))).max()
    )
    print(f"[layer 3] coupling projected to lines: shape {T_lines.shape}, "
          f"max mass-conservation residual {residual:.2e}")
    return corr, residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--raw-root", type=Path,
        default=repo_root / "tests" / "fixtures" / "metroflow_tiny",
        help="directory with the three raw MetroFlow CSVs",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="canonical parquet output (default: a temporary directory)",
    )
    args = parser.parse_args()

    tmp = None
    if args.output_root is None:
        tmp = tempfile.mkdtemp(prefix="passenger_state_demo_")
        output_root = Path(tmp) / "canonical"
    else:
        output_root = args.output_root
    try:
        out = canonicalize(CanonicalizeConfig(args.raw_root, output_root))
        con = duckdb.connect()
        print(f"[data] canonicalized fixture -> {out}")

        stations, order, train_od, train_entry = demo_layer1(con, out, args.raw_root)
        phi, cov_static, cov_calib = demo_layer2(stations, train_od, train_entry)
        corr, residual = demo_layer3()

        checks = [
            ("phi recovers the known truth", abs(phi - 0.05) < 0.02),
            ("calibration raises coverage", cov_calib > cov_static),
            ("rank correlation positive", corr["spearman"] > 0),
            ("mass conserved", residual < 1e-9),
        ]
        print("\nsummary:")
        for name, ok in checks:
            print(f"  {'ok   ' if ok else 'not ok'} {name}")
        if all(ok for _, ok in checks):
            print("\ndemo complete.")
            return 0
        return 1
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
