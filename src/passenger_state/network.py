"""Metro network topology: curated line membership, shortest paths, route choice.

Three layers of network knowledge, in increasing order of behavioral content:

* the curated line membership table (:data:`LINE_MEMBERSHIP`);
* a weighted undirected track graph with Dijkstra shortest paths
  (:func:`build_graph`, :func:`dijkstra_predecessors`), used to assign each
  boarding trip its alighting station;
* a logit route-choice model over candidate interchange stations
  (:func:`build_route_probs`), which turns the deterministic shortest-path
  alighting assignment into a probability distribution.

Curated membership for the 14 Shanghai Metro lines (companion paper).

Twelve lines are simple paths (``kind == "path"``). Lines 10 and 11 are genuine
branches (``kind == "branch"``):

* Line 10 forks at Longxi Road ``1045`` into the main line (Hongqiao Railway
  Station ``2014`` <-> New Jiangwan City ``1068``) and the Hangzhong Road branch
  (Longxi Road ``1045`` <-> Hangzhong Road ``1018``).
* Line 11 forks at Jiading New City ``1134`` into the Jiading North arm
  (``1131``), the Huaqiao arm (``1114``) and the main-east arm to Disney Resort
  (``1163``) via Luoshan Road ``2042``.

All 18 Line 10/11 interchange interfaces sit on the main arm; the branch spurs
carry no interchanges. ``line_spec`` returns the *primary* arm (the one holding
the interchanges); ``line_arms`` returns every arm.

Known data gap: ``stationInfo.csv`` omits Kangheng Road (``1158``) on Line 11
between Pusan Road ``1157`` and Yuqiao ``1159``, so the curated Line 11 main-east
order bridges ``1157 -> 1159`` as a virtual edge. The gap is documented here;
the raw ``stationInfo.csv`` is left unmodified.
"""

import heapq
import math
from dataclasses import dataclass, field

import numpy as np


def _path(stations, interchanges):
    return {
        "kind": "path",
        "stations": tuple(stations),
        "interchanges": tuple(interchanges),
    }


def _branch(*arms):
    return {
        "kind": "branch",
        "arms": [
            {"name": name, "stations": tuple(stations), "interchanges": tuple(interchanges)}
            for name, stations, interchanges in arms
        ],
    }


# Authoritative ordered membership. ``stations`` is the full sequential stop order
# (including non-transfer 2xxx members); ``interchanges`` is the transfer-interface
# subset, in along-line order.
LINE_MEMBERSHIP = {
    1: _path(
        (138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 2034, 2039,
         124, 2035, 2028, 2056, 2017, 119, 2025, 2029, 2038, 2027, 114, 113, 112,
         2011),
        (2034, 2039, 2035, 2056, 2017, 2025, 2029, 2038, 2027, 2011),
    ),
    2: _path(
        (263, 262, 261, 260, 259, 258, 257, 256, 255, 254, 253, 2052, 251, 250,
         2010, 248, 247, 2019, 2035, 2041, 2050, 2054, 2043, 240, 239, 238, 237,
         2013, 2014, 234),
        (2052, 2010, 2019, 2035, 2041, 2050, 2054, 2043, 2013, 2014),
    ),
    3: _path(
        (339, 338, 337, 336, 335, 334, 333, 332, 331, 330, 329, 328, 327, 2005,
         325, 2030, 2034, 2001, 2024, 2055, 2009, 2043, 2046, 2044, 2051, 314,
         2048, 312, 2027),
        (2005, 2034, 2024, 2055, 2009, 2043, 2044, 2051, 2048, 2027),
    ),
    4: _path(
        (426, 2006, 2007, 423, 2002, 421, 420, 2015, 2049, 2010, 416, 415, 2032,
         413, 2047, 2030, 2034, 2001, 2024, 2055, 2009, 2043, 2046, 2044, 2051,
         2029),
        (2006, 2007, 2002, 2015, 2010, 2032, 2047, 2034, 2024, 2055, 2009, 2043,
         2044, 2051, 2029),
    ),
    5: _path(
        (2011, 502, 503, 505, 507, 508, 509, 510, 511, 512, 513),
        (2011,),
    ),
    6: _path(
        (648, 647, 646, 645, 644, 643, 642, 2004, 640, 639, 638, 637, 636, 635,
         634, 633, 2010, 2049, 2015, 629, 628, 2023, 626, 625, 624, 623, 622,
         2040),
        (2004, 2010, 2015, 2023, 2040),
    ),
    7: _path(
        (721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734,
         735, 2024, 2020, 738, 2050, 2017, 2003, 2006, 2045, 744, 745, 2016, 747,
         2023, 749, 750, 751, 2052, 753),
        (2024, 2020, 2050, 2017, 2003, 2006, 2045, 2016, 2023, 2052),
    ),
    8: _path(
        (849, 848, 847, 846, 845, 844, 843, 842, 2018, 840, 2005, 838, 837, 2021,
         2035, 834, 2008, 2036, 2002, 830, 2016, 828, 827, 2040, 825, 824, 823,
         822, 821, 820),
        (2018, 2005, 2021, 2035, 2008, 2036, 2002, 2016, 2040),
    ),
    9: _path(
        (918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931,
         932, 2051, 2025, 2003, 2033, 937, 2031, 2036, 940, 941, 2010, 943),
        (2051, 2025, 2003, 2033, 2036, 2010),
    ),
    10: _branch(
        ("main",
         (2014, 2013, 1043, 1044, 1045, 1046, 1047, 1048, 2044, 2037, 1051, 2056,
          2053, 2008, 1055, 2019, 2026, 1058, 2047, 1060, 2018, 1062, 1063, 1064,
          1065, 1066, 1067, 1068),
         (2014, 2013, 2044, 2037, 2056, 2008, 2019, 2026, 2047, 2018)),
        ("hangzhong-branch",
         (1045, 1020, 1019, 1018),
         ()),
    ),
    11: _branch(
        ("main-east",
         (1134, 1135, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 2055, 2057,
          2054, 2037, 2025, 1150, 2012, 1152, 1153, 2040, 1155, 1156, 1157, 1159,
          2042, 1161, 1162, 1163),
         (2055, 2057, 2054, 2037, 2025, 2012, 2040, 2042)),
        ("jiading-north",
         (1131, 1132, 1133, 1134),
         ()),
        ("huaqiao",
         (1114, 1115, 1116, 1117, 1118, 1119, 1120, 1134),
         ()),
    ),
    12: _path(
        (1220, 1221, 1222, 1223, 1224, 1225, 1226, 2038, 2048, 2012, 2045, 2007,
         2033, 2056, 2041, 2039, 2021, 2026, 1238, 1239, 2032, 1241, 1242, 1243,
         1244, 1245, 1246, 2004, 1248, 1249, 1250, 2022),
        (2038, 2048, 2012, 2045, 2007, 2033, 2056, 2041, 2039, 2021, 2026, 2032,
         2004),
    ),
    13: _path(
        (1321, 1322, 1323, 1324, 1325, 1326, 2009, 2057, 1329, 2020, 1331, 2039,
         1333, 2041),
        (2009, 2057, 2020, 2039, 2041),
    ),
    16: _path(
        (1633, 1632, 1631, 1630, 1629, 1628, 1627, 1626, 1625, 1624, 2042, 1622,
         2052),
        (2042, 2052),
    ),
}

LINE_NUMBERS = tuple(sorted(LINE_MEMBERSHIP))


@dataclass(frozen=True)
class LineArm:
    """One sequential arm of a line (the whole line for a path, one branch for a fork)."""

    name: str
    stations: tuple[int, ...]
    interchanges: tuple[int, ...]

    @property
    def S(self) -> int:
        return len(self.stations)


@dataclass(frozen=True)
class LineSpec:
    """A line's ordered stations and interchange set for the inference machinery.

    ``order`` maps a canonical station ID to its sequential index; it is passed
    as a mutable dict for O(1) lookups (do not mutate it after construction).
    """

    line_number: int
    stations: tuple[int, ...]
    order: dict[int, int] = field(compare=False)
    interchanges: tuple[int, ...]

    @property
    def S(self) -> int:
        return len(self.stations)

    @property
    def station_list_sql(self) -> str:
        return ",".join(str(s) for s in self.stations)


def _membership(line_number: int) -> dict:
    entry = LINE_MEMBERSHIP.get(line_number)
    if entry is None:
        raise ValueError(f"no membership entry for line {line_number}")
    return entry


def _primary_arm(line_number: int) -> LineArm:
    entry = _membership(line_number)
    if entry["kind"] == "path":
        return LineArm("main", entry["stations"], entry["interchanges"])
    arms = entry["arms"]
    return LineArm(arms[0]["name"], arms[0]["stations"], arms[0]["interchanges"])


def line_arms(line_number: int) -> list[LineArm]:
    """All sequential arms of a line (one for a path, several for a branch)."""
    entry = _membership(line_number)
    if entry["kind"] == "path":
        return [LineArm("main", entry["stations"], entry["interchanges"])]
    return [
        LineArm(a["name"], a["stations"], a["interchanges"]) for a in entry["arms"]
    ]


def derive_line(line_number: int, neighbours=None) -> tuple[list[int], list[int]]:
    """The line's ordered stations and interchanges (from the membership table).

    ``neighbours`` is accepted for API compatibility and is ignored. Returns the
    *primary* arm for branch lines (the arm carrying the interchange
    interfaces).

    Raises ``ValueError`` for a line number without a membership entry.
    """
    arm = _primary_arm(line_number)
    return list(arm.stations), list(arm.interchanges)


def line_spec(line_number: int, neighbours=None) -> LineSpec:
    """Build a :class:`LineSpec` from the membership table.

    ``neighbours`` is accepted for API compatibility and is ignored.
    """
    arm = _primary_arm(line_number)
    order = {sid: i for i, sid in enumerate(arm.stations)}
    return LineSpec(line_number, arm.stations, order, arm.interchanges)


def station_line_sets() -> dict[int, tuple[int, ...]]:
    """Canonical station ID -> sorted tuple of line numbers serving it (all arms).

    A station serves every line that contains it on any arm, so interchange
    hubs map to all their lines and branch spurs map to their own line. This
    is the destination->line mapping for the transfer coupling.
    """
    sets: dict[int, set[int]] = {}
    for line_number, entry in LINE_MEMBERSHIP.items():
        if entry["kind"] == "branch":
            arms = entry["arms"]
        else:
            arms = [entry]
        for arm in arms:
            for sid in arm["stations"]:
                sets.setdefault(sid, set()).add(line_number)
    return {sid: tuple(sorted(lines)) for sid, lines in sets.items()}


# Station-ID conventions (canonical IDs).
TRANSFER_STATION_THRESHOLD = 2000  # stationID >= 2000 is an interchange hub
TRANSFER_TIME_MIN = 4.0           # minutes for a transfer (interchange) edge
AVG_SPEED_KMH = 35.0              # average train speed for track edges
MIN_TRACK_TIME_MIN = 0.5          # floor on track-edge travel time
EARTH_RADIUS_KM = 6371.0


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in km between two (lon, lat) points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        dlambda / 2.0
    ) ** 2
    return EARTH_RADIUS_KM * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def edge_weight(u: int, v: int, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Edge weight in minutes.

    Both endpoints interchange hubs (>= 2000) -> fixed transfer time.
    Otherwise a track edge -> haversine distance at average speed, floored.
    """
    if u >= TRANSFER_STATION_THRESHOLD and v >= TRANSFER_STATION_THRESHOLD:
        return TRANSFER_TIME_MIN
    travel_min = haversine_km(lon1, lat1, lon2, lat2) / AVG_SPEED_KMH * 60.0
    return max(travel_min, MIN_TRACK_TIME_MIN)


def build_graph(
    station_ids,
    neighbour_ids,
    lons,
    lats,
) -> dict[int, list[tuple[int, float]]]:
    """Build an undirected adjacency dict {node: [(neighbor, weight_min), ...]}.

    Args:
        station_ids: list of canonical station IDs (int).
        neighbour_ids: list of neighbour-ID lists aligned with station_ids.
        lons, lats: list of coordinates aligned with station_ids.

    Returns:
        adjacency: dict mapping each station ID to its (neighbor, weight) pairs.
        Edge weights use :func:`edge_weight`; each undirected edge appears once
        per direction.
    """
    ids = [int(s) for s in station_ids]
    pos = {sid: (float(lon), float(lat)) for sid, lon, lat in zip(ids, lons, lats)}
    adjacency: dict[int, list[tuple[int, float]]] = {sid: [] for sid in ids}
    for sid, nbrs in zip(ids, neighbour_ids):
        for nbr in nbrs:
            nbr = int(nbr)
            if nbr not in pos:
                continue
            lon1, lat1 = pos[sid]
            lon2, lat2 = pos[nbr]
            w = edge_weight(sid, nbr, lon1, lat1, lon2, lat2)
            adjacency[sid].append((nbr, w))
    return adjacency


def dijkstra_predecessors(
    adjacency: dict[int, list[tuple[int, float]]], source: int
) -> tuple[dict[int, float], dict[int, int | None]]:
    """Single-source Dijkstra returning (distance, predecessor) dicts.

    Args:
        adjacency: undirected adjacency from :func:`build_graph`.
        source: source station ID.

    Returns:
        (dist, pred): dist[node] = shortest distance; pred[node] = predecessor
        (None for the source, absent for unreachable nodes).
    """
    dist: dict[int, float] = {source: 0.0}
    pred: dict[int, int | None] = {source: None}
    heap = [(0.0, source)]
    visited: set[int] = set()
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node not in adjacency:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = d + weight
            if neighbor not in dist or candidate < dist[neighbor]:
                dist[neighbor] = candidate
                pred[neighbor] = node
                heapq.heappush(heap, (candidate, neighbor))
    return dist, pred


def reconstruct_path(
    pred: dict[int, int | None], source: int, target: int
) -> list[int] | None:
    """Reconstruct the shortest path source -> target from a predecessor tree.

    Returns the node list (both endpoints included), or None if ``target``
    is unreachable (not in the predecessor tree and not the source).
    """
    if target == source:
        return [source]
    if target not in pred:
        return None
    path = [target]
    current = target
    while current != source:
        current = pred[current]
        if current is None:
            return None
        path.append(current)
    path.reverse()
    return path


def assign_alighting_station(
    line_stations: set[int],
    origin: int,
    dest: int,
    pred: dict[int, int | None],
) -> int | None:
    """Assign the alighting station for an origin-on-line trip.

    Args:
        line_stations: set of canonical station IDs on the line.
        origin: origin station (must be in ``line_stations``).
        dest: destination station ID.
        pred: predecessor tree from :func:`dijkstra_predecessors` rooted at
            ``origin``.

    Returns:
        ``dest`` if it lies on the line (direct trip); otherwise the last
        line station along the shortest path ``origin -> dest`` (the transfer
        station where the passenger leaves the line). Returns None if ``dest``
        is unreachable.
    """
    if dest in line_stations:
        return dest
    path = reconstruct_path(pred, origin, dest)
    if path is None:
        return None
    last = None
    for node in path:
        if node in line_stations:
            last = node
    return last


def downstream_interchanges(
    line_stations,
    order: dict[int, int],
    origin: int,
    dest: int,
    pred: dict[int, int | None],
    interchanges=None,
) -> list[int]:
    """Candidate interchange stations for a transfer-out trip.

    Args:
        line_stations: ordered canonical station IDs of the line (a tuple).
        order: station ID -> sequential order along the line.
        origin: boarding station (must be on the line).
        dest: final destination station ID.
        pred: predecessor tree from :func:`dijkstra_predecessors` rooted at
            ``origin``.
        interchanges: the line's interchange list; defaults to the
            ``stationID >= 2000`` proxy (the curated membership already
            excludes non-transfer 2xxx members such as 2028 and 2031).

    Returns:
        The line's interchange stations strictly downstream of ``origin`` in the
        direction of travel toward ``dest``. An on-line ``dest`` yields ``[]``
        (no transfer). If the shortest path leaves the line immediately at an
        interchange origin, returns ``[origin]``. Returns ``[]`` if ``dest`` is
        unreachable.
    """
    if interchanges is None:
        interchanges = [
            s for s in line_stations if s >= TRANSFER_STATION_THRESHOLD
        ]
    inter_set = set(interchanges)
    if origin not in order:
        raise ValueError(f"origin {origin} is not on the line")
    if dest in set(line_stations):
        return []
    path = reconstruct_path(pred, origin, dest)
    if path is None:
        return []
    oi = order[origin]
    direction = 0
    for node in path[1:]:
        if node in order:
            direction = 1 if order[node] > oi else -1
            break
    if direction == 0:
        # The path leaves the line immediately: the transfer point is origin.
        return [origin] if origin in inter_set else []
    return [s for s in interchanges if (order[s] - oi) * direction > 0]


def build_route_probs(
    line_stations,
    order: dict[int, int],
    off_line_dests,
    pred_trees: dict[int, dict[int, int | None]],
    k_dist: dict[int, dict[int, float]],
    eta: float,
    section_time: float = 2.5,
    transfer_time: float = 4.0,
    interchanges=None,
):
    """Build the [S, n_off, K] route-choice probability tensor.

    For every origin ``i`` and off-line destination ``d``, computes the
    downstream interchange candidates and their logit probabilities, laying the
    result out as a dense tensor consumed by
    :func:`passenger_state.reliability.sample_transfer_flow`.

    Args:
        line_stations: ordered canonical station IDs of the line.
        order: station ID -> sequential order along the line.
        off_line_dests: ordered list of destination station IDs not served by
            the line.
        pred_trees: predecessor tree per line origin (for candidate direction).
        k_dist: per-interchange distance dict (from
            :func:`dijkstra_predecessors`) keyed by the interchange station ID.
        eta: route-choice dispersion.
        section_time, transfer_time: cost parameters.
        interchanges: the line's interchange list; defaults to the
            ``stationID >= 2000`` proxy.

    Returns:
        ``(probs, interchanges)`` where ``probs`` is [S, n_off, K] and
        ``interchanges`` is the ordered list of the line's interchange IDs.
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
            costs: dict[int, float] = {}
            for k in cands:
                dist = k_dist[k].get(d)
                if dist is None:
                    continue
                costs[k] = route_choice_cost(
                    order, origin, k, dist, section_time, transfer_time
                )
            for k, prob in route_choice_probs(costs, eta).items():
                probs[i, d_idx, k_idx[k]] = prob
    return probs, interchanges


def route_choice_cost(
    order: dict[int, int],
    origin: int,
    k: int,
    dist_from_k: float,
    section_time: float = 2.5,
    transfer_time: float = 4.0,
) -> float:
    """Route cost of alighting at interchange ``k`` for an off-line trip.

    cost = |order(k) - order(origin)| * section_time + transfer_time +
    dist_from_k, where ``dist_from_k`` is the shortest off-line continuation
    time from ``k`` to the destination (computed externally by the caller).

    Args:
        order: station ID -> sequential order along the line.
        origin: boarding station.
        k: candidate interchange station.
        dist_from_k: shortest-path time (minutes) from ``k`` to the destination.
        section_time: on-line travel time per station (minutes).
        transfer_time: transfer walking penalty (minutes).

    Returns:
        The route cost in minutes.
    """
    track = abs(order[k] - order[origin]) * section_time
    return track + transfer_time + float(dist_from_k)


def route_choice_probs(
    costs: dict[int, float],
    eta: float = 1.0,
) -> dict[int, float]:
    """Softmax route-choice probabilities over candidate interchanges.

    p(k) ∝ exp(-eta * cost[k]). ``eta -> inf`` recovers the deterministic
    minimum-cost interchange (the shortest-path assignment); ``eta -> 0`` is
    uniform over candidates.

    Args:
        costs: interchange station ID -> route cost (minutes). Interchanges
            with infinite cost (unreachable) must be filtered out by the caller.
        eta: route-choice dispersion (non-negative).

    Returns:
        interchange station ID -> probability, summing to one. Empty dict for
        empty input; ``{k: 1.0}`` for a single candidate.
    """
    if eta < 0:
        raise ValueError("eta must be non-negative")
    if not costs:
        return {}
    keys = list(costs)
    if len(keys) == 1:
        return {keys[0]: 1.0}
    if eta == 0.0:
        return {k: 1.0 / len(keys) for k in keys}
    c = np.array([float(costs[k]) for k in keys], dtype=np.float64)
    logits = -eta * (c - c.min())
    mx = logits.max()
    e = np.exp(logits - mx)
    p = e / e.sum()
    return {k: float(pi) for k, pi in zip(keys, p)}
