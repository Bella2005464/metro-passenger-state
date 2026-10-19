"""Smoke tests: core invariants of the passenger-state pipeline.

Nine small assertions: mass conservation of the coupling projection under a
multi-line split, the Layer-1 row constraint, over-dispersion moment matching
at both extremes, eta-mixture marginalization under non-uniform weights,
zero-truth exclusion, WAPE on a worked example, the split policy, a
canonicalize/read-back roundtrip on the bundled fixture, and the primary-arm
selection for branch lines.

Run: python -m unittest tests.test_smoke
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np

from passenger_state.data import (
    CanonicalizeConfig,
    canonicalize,
    read_day_flows,
    read_day_trips,
)
from passenger_state.network import (
    build_graph,
    build_route_probs,
    dijkstra_predecessors,
    line_spec,
)
from passenger_state.reconstruction import (
    build_line_od,
    fit_overdispersion_phi,
)
from passenger_state.reliability import (
    MIXTURE_ETA_GRID,
    fit_offline_shares,
    fit_online_split,
    project_coupling_to_lines,
    sample_transfer_coupling,
    sample_transfer_flow,
    sample_transfer_flow_eta_mixture,
)
from passenger_state.validation import s2_relative_error, wape

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "metroflow_tiny"

# Line 101 -- 2001 -- 102 with off-line destinations 201, 202 (both via 2001).
LINE = (101, 2001, 102)
NODES = [101, 2001, 102, 201, 202]
NEIGHBOURS = [[2001], [101, 102, 201, 202], [2001], [2001], [2001]]
OFF_DESTS = [201, 202]
DEST_LINES = {201: (3, 5), 202: (3,)}


def _demo_network():
    order = {s: i for i, s in enumerate(LINE)}
    graph = build_graph(NODES, NEIGHBOURS, [121.0] * len(NODES), [31.0] * len(NODES))
    pred_trees = {o: dijkstra_predecessors(graph, o)[1] for o in LINE}
    interchanges = [s for s in LINE if s >= 2000]
    k_dist = {k: dijkstra_predecessors(graph, k)[0] for k in interchanges}
    return order, pred_trees, interchanges, k_dist


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        raw = Path(cls.tmp.name) / "raw"
        shutil.copytree(FIXTURE, raw)
        cls.canonical_root = canonicalize(CanonicalizeConfig(raw, Path(cls.tmp.name) / "canonical"))
        cls.con = duckdb.connect()

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.tmp.cleanup()

    def test_coupling_projection_conserves_mass(self):
        order, pred_trees, interchanges, k_dist = _demo_network()
        probs, _ = build_route_probs(LINE, order, OFF_DESTS, pred_trees, k_dist, eta=1.0)
        entry = np.array([100.0, 0.0, 50.0])
        on_total = np.array([60.0, 0.0, 40.0])
        off_total = np.array([40.0, 0.0, 10.0])
        off_counts = np.array([[30.0, 10.0], [0.0, 0.0], [5.0, 5.0]])
        C = sample_transfer_coupling(
            entry, fit_online_split(on_total, off_total),
            fit_offline_shares(off_counts), probs,
            n_samples=500, rng=np.random.RandomState(0),
        )
        T_lines = project_coupling_to_lines(C, [DEST_LINES[d] for d in OFF_DESTS], (3, 5))
        residual = np.abs(T_lines.sum(axis=(1, 2)) - C.sum(axis=(1, 2))).max()
        self.assertLess(residual, 1e-9)
        # Per-destination conservation: each destination's line-projected mass
        # equals its coupling mass even under the 1/2 + 1/2 split of dest 201.
        self.assertTrue(np.allclose(
            T_lines.sum(axis=1), C.sum(axis=1), atol=1e-12
        ))

    def test_build_line_od_row_sums_equal_trip_flow(self):
        graph = build_graph(NODES, NEIGHBOURS, [121.0] * len(NODES), [31.0] * len(NODES))
        pred_trees = {o: dijkstra_predecessors(graph, o)[1] for o in LINE}
        trips = [(101, 102, 60.0), (101, 201, 30.0), (102, 101, 40.0)]
        od, unassigned = build_line_od(LINE, trips, pred_trees)
        row_sums = od.sum(axis=1)
        expected = np.array([90.0, 0.0, 40.0])
        self.assertEqual(unassigned, 0.0)
        self.assertTrue(np.allclose(row_sums, expected))

    def test_overdispersion_phi_extremes(self):
        alpha = [np.array([1.0, 1.0, 1.0])]
        # Near-maximal day-to-day variation -> phi collapses toward zero.
        wild = [np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])]
        self.assertLess(fit_overdispersion_phi(alpha, wild), 0.01)
        # No day-to-day variation -> phi stays at the static ceiling.
        calm = [np.tile(np.array([1 / 3, 1 / 3, 1 / 3]), (5, 1))]
        self.assertEqual(fit_overdispersion_phi(alpha, calm), 1.0)

    def test_eta_mixture_mean_equals_per_eta_average(self):
        order, pred_trees, interchanges, k_dist = _demo_network()
        route_list = [
            build_route_probs(LINE, order, OFF_DESTS, pred_trees, k_dist, eta=e)[0]
            for e in MIXTURE_ETA_GRID
        ]
        entry = np.array([300.0, 0.0, 150.0])
        on_total = np.array([180.0, 0.0, 120.0])
        off_total = np.array([120.0, 0.0, 30.0])
        off_counts = np.array([[90.0, 30.0], [0.0, 0.0], [15.0, 15.0]])
        split = fit_online_split(on_total, off_total)
        alpha = fit_offline_shares(off_counts)
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        mixture = sample_transfer_flow_eta_mixture(
            entry, split, alpha, route_list, weights, 27000, np.random.RandomState(0)
        ).mean(axis=0)
        per_eta = np.average([
            sample_transfer_flow(
                entry, split, alpha, rp, n_samples=3000, rng=np.random.RandomState(100 + j)
            ).mean(axis=0)
            for j, rp in enumerate(route_list)
        ], axis=0, weights=weights)
        scale = np.abs(per_eta).max()
        self.assertTrue(np.allclose(mixture, per_eta, atol=0.05 * scale))

    def test_relative_error_excludes_zero_truth(self):
        T_true = np.array([10.0, 0.0])
        samples = np.array([[8.0, 3.0], [8.0, 3.0]])
        rel, active = s2_relative_error(T_true, samples)
        self.assertEqual(list(active), [True, False])
        self.assertAlmostEqual(rel[0], 0.2)
        self.assertEqual(rel[1], 0.0)

    def test_wape_worked_example(self):
        self.assertAlmostEqual(wape(np.array([100.0, 100.0]), np.array([90.0, 80.0])), 0.15)

    def test_split_policy_covers_and_separates(self):
        from passenger_state.data import SplitPolicy

        policy = SplitPolicy.metroflow()
        self.assertEqual(policy.split_for_date("2017-05-01"), "train")
        self.assertEqual(policy.split_for_date("2017-08-01"), "test")
        self.assertFalse(set(policy.train_dates) & set(policy.val_dates))
        self.assertEqual(
            len(policy.train_dates) + len(policy.val_dates)
            + len(policy.test_dates) + len(policy.missingness_stress_dates),
            123,
        )

    def test_canonicalize_roundtrip_reads_back(self):
        stations = (20, 3, 11)
        order = {20: 0, 3: 1, 11: 2}
        trips = read_day_trips(self.con, self.canonical_root, "2017-05-01", "train", stations)
        entry = read_day_flows(self.con, self.canonical_root, "2017-05-01", "entry_count", stations, order)
        self.assertIsNotNone(trips)
        self.assertGreater(len(trips), 0)
        self.assertEqual(entry.shape, (3,))
        self.assertGreater(entry.sum(), 0.0)

    def test_line_spec_returns_primary_arm(self):
        spec = line_spec(11)
        self.assertIn(1163, spec.stations)  # Disney Resort, main-east arm
        self.assertNotIn(1131, spec.stations)  # Jiading North spur


if __name__ == "__main__":
    unittest.main()
