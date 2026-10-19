"""Data pipeline: canonicalization, date splits, and parquet readers.

``canonicalize`` streams the three raw MetroFlow CSVs (station info, per-slot
station in/out flow, per-slot OD flow) into a date-partitioned parquet dataset
via DuckDB: a station crosswalk, observed station flow, the non-diagonal OD
labels, and the derived self-loop candidate residual. The output directory is
built in a staging folder and moved into place atomically.

``SplitPolicy`` fixes the train/validation/test date split. The readers load a
single service day of trips or flows back from the parquet dataset for one
line's stations.
"""

from __future__ import annotations

import ast
import csv
import glob
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import numpy as np

from passenger_state.network import build_graph


KNOWN_SPLITS = frozenset({"train", "val", "test"})


def _date_range(start: date, end: date) -> tuple[str, ...]:
    days = (end - start).days
    return tuple((start + timedelta(days=offset)).isoformat() for offset in range(days + 1))


@dataclass(frozen=True)
class SplitPolicy:
    train_dates: tuple[str, ...]
    val_dates: tuple[str, ...]
    test_dates: tuple[str, ...]
    missingness_stress_dates: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = [
            set(self.train_dates),
            set(self.val_dates),
            set(self.test_dates),
            set(self.missingness_stress_dates),
        ]
        if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i + 1, len(groups))):
            raise ValueError("split date groups must be disjoint")
        expected = set(_date_range(date(2017, 5, 1), date(2017, 8, 31)))
        if set().union(*groups) != expected:
            raise ValueError("split date union must match all 123 MetroFlow service dates")

    @classmethod
    def metroflow(cls) -> "SplitPolicy":
        stress = (
            "2017-05-04",
            "2017-05-08",
            "2017-05-09",
            "2017-06-16",
            "2017-06-27",
            "2017-06-28",
        )
        stress_set = set(stress)
        train = tuple(
            day
            for day in _date_range(date(2017, 5, 1), date(2017, 6, 30))
            if day not in stress_set
        )
        return cls(
            train_dates=train,
            val_dates=_date_range(date(2017, 7, 1), date(2017, 7, 31)),
            test_dates=_date_range(date(2017, 8, 1), date(2017, 8, 31)),
            missingness_stress_dates=stress,
        )

    def dates_for(self, split: str) -> tuple[str, ...]:
        if split not in KNOWN_SPLITS:
            raise ValueError(f"unknown split: {split}")
        return getattr(self, f"{split}_dates")

    def split_for_date(self, service_date: str) -> str:
        for split in ("train", "val", "test"):
            if service_date in self.dates_for(split):
                return split
        if service_date in self.missingness_stress_dates:
            return "missingness_stress"
        raise ValueError(f"date is outside the MetroFlow service calendar: {service_date}")


@dataclass(frozen=True)
class SplitLabelStore:
    root: Path
    allowed_splits: frozenset[str]

    def __post_init__(self) -> None:
        unknown = self.allowed_splits - KNOWN_SPLITS
        if unknown:
            raise ValueError(f"unknown allowed splits: {sorted(unknown)}")

    def partitions(self, split: str) -> tuple[Path, ...]:
        if split not in KNOWN_SPLITS:
            raise ValueError(f"unknown split: {split}")
        if split not in self.allowed_splits:
            raise PermissionError(f"split {split!r} is not in this store's allowed_splits")
        directory = self.root / "labels" / split / "od_entry_time"
        if not directory.is_dir():
            return ()
        return tuple(sorted(path for path in directory.glob("service_date=*") if path.is_dir()))


def _non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _required(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} is required")


@dataclass(frozen=True)
class StationCrosswalkRecord:
    reported_station_id: str
    canonical_station_id: str
    matrix_index: int
    mapping_version: str

    def validate(self) -> None:
        _required("reported_station_id", self.reported_station_id)
        _required("canonical_station_id", self.canonical_station_id)
        _non_negative("matrix_index", self.matrix_index)
        _required("mapping_version", self.mapping_version)


METROFLOW_MAPPING_VERSION = "station_crosswalk"


def numeric_station_order(ids: Sequence[str]) -> list[str]:
    """Return unique station identifiers in numeric, not lexical, order."""
    if len(ids) != len(set(ids)):
        raise ValueError("station identifiers must be unique")
    try:
        return sorted(ids, key=int)
    except ValueError as error:
        raise ValueError("station identifiers must be numeric") from error


def derive_metroflow_crosswalk(
    original_ids: Sequence[str],
) -> tuple[StationCrosswalkRecord, ...]:
    """Map sorted reported positions to stationInfo original row-order IDs."""
    reported_ids = numeric_station_order(original_ids)
    rows = tuple(
        StationCrosswalkRecord(
            reported_station_id=reported_id,
            canonical_station_id=canonical_id,
            matrix_index=matrix_index,
            mapping_version=METROFLOW_MAPPING_VERSION,
        )
        for matrix_index, (reported_id, canonical_id) in enumerate(
            zip(reported_ids, original_ids, strict=True)
        )
    )
    for row in rows:
        row.validate()
    return rows


@dataclass(frozen=True)
class CanonicalizeConfig:
    raw_root: Path
    output_root: Path


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def copy_partitioned(connection: Any, select_sql: str, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    connection.execute(
        f"COPY ({select_sql}) TO '{_sql_path(target)}' "
        "(FORMAT PARQUET, PARTITION_BY (service_date), COMPRESSION ZSTD)"
    )


def _station_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("stationInfo.csv has no header")
        normalized = {name.strip(): name for name in reader.fieldnames}
        source_name = normalized.get("stationID")
        if source_name is None:
            raise ValueError("stationInfo.csv is missing stationID")
        ids = [str(row[source_name]).strip() for row in reader]
    if not ids or any(not station_id for station_id in ids):
        raise ValueError("stationInfo.csv contains no usable station IDs")
    return ids


def _date_predicate(dates: tuple[str, ...]) -> str:
    if not dates:
        return "FALSE"
    values = ", ".join(f"DATE '{item}'" for item in dates)
    return f"service_date IN ({values})"


def canonicalize(config: CanonicalizeConfig) -> Path:
    """Canonicalize the raw MetroFlow CSVs into a parquet dataset.

    Reads ``stationInfo.csv``, ``metroData_InOutFlow.csv`` and
    ``metroData_ODFlow.csv`` from ``config.raw_root`` and writes, under
    ``config.output_root``:

    * ``identity/station_crosswalk.parquet``: reported -> canonical station ID;
    * ``observations/station_flow/``: per-slot entry/exit counts by date;
    * ``labels/{train,val,test}/od_entry_time/``: non-diagonal OD flow by date;
    * ``nuisance/{split}/self_loop_candidate/``: the entry-minus-OD residual;
    * ``stress/missingness/``: the designated stress dates.

    Args:
        config: :class:`CanonicalizeConfig` with the raw CSV directory and the
            output directory (must not exist).

    Returns:
        The resolved output directory.

    Raises:
        FileExistsError: if the output directory already exists.
        FileNotFoundError: if one of the three input CSVs is missing.
        ValueError: on unmapped stations/OD rows or a negative self-loop
            residual.
    """
    raw_root = Path(config.raw_root).resolve()
    output_root = Path(config.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"canonical output already exists: {output_root}")
    if not raw_root.is_dir():
        raise ValueError(f"raw_root is not a directory: {raw_root}")
    raw_paths = [
        raw_root / "stationInfo.csv",
        raw_root / "metroData_InOutFlow.csv",
        raw_root / "metroData_ODFlow.csv",
    ]
    missing = [path for path in raw_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing canonical sources: {missing}")
    station_path, inout_path, od_path = raw_paths

    staging = output_root.with_name(f"{output_root.name}.staging-{uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    connection = duckdb.connect()
    try:
        split_policy = SplitPolicy.metroflow()
        crosswalk = derive_metroflow_crosswalk(_station_ids(station_path))
        connection.execute(
            "CREATE TABLE station_crosswalk("
            "reported_station_id VARCHAR, canonical_station_id VARCHAR, "
            "matrix_index INTEGER, mapping_version VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO station_crosswalk VALUES (?, ?, ?, ?)",
            [(r.reported_station_id, r.canonical_station_id, r.matrix_index, r.mapping_version) for r in crosswalk],
        )
        identity_dir = staging / "identity"
        identity_dir.mkdir(parents=True)
        connection.execute(
            f"COPY station_crosswalk TO '{_sql_path(identity_dir / 'station_crosswalk.parquet')}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )

        inout_sql = (
            f"SELECT strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE AS service_date, "
            "CAST(timeslot - MIN(timeslot) OVER (PARTITION BY date) AS INTEGER) AS slot, "
            "CAST(station AS VARCHAR) AS station_id, CAST(inFlow AS BIGINT) AS entry_count, "
            "CAST(outFlow AS BIGINT) AS exit_count "
            f"FROM read_csv_auto('{_sql_path(inout_path)}', header=true)"
        )
        connection.execute(f"CREATE TEMP TABLE station_flow AS {inout_sql}")
        invalid_station_count = connection.execute(
            "SELECT COUNT(*) FROM station_flow s LEFT JOIN station_crosswalk m "
            "ON s.station_id = m.canonical_station_id WHERE m.canonical_station_id IS NULL"
        ).fetchone()[0]
        if invalid_station_count:
            raise ValueError(f"unmapped station-flow rows: {invalid_station_count}")
        copy_partitioned(connection, "SELECT * FROM station_flow", staging / "observations" / "station_flow")

        od_source = f"read_csv_auto('{_sql_path(od_path)}', header=true)"
        unmapped_od = connection.execute(
            "SELECT COUNT(*) FROM " + od_source + " od "
            "LEFT JOIN station_crosswalk origin_map ON CAST(od.originStation AS VARCHAR) = origin_map.reported_station_id "
            "LEFT JOIN station_crosswalk destination_map ON CAST(od.destinationStation AS VARCHAR) = destination_map.reported_station_id "
            "WHERE origin_map.canonical_station_id IS NULL OR destination_map.canonical_station_id IS NULL"
        ).fetchone()[0]
        if unmapped_od:
            raise ValueError(f"unmapped OD rows: {unmapped_od}")
        canonical_od_sql = (
            "SELECT strptime(CAST(od.date AS VARCHAR), '%Y%m%d')::DATE AS service_date, "
            "CAST(od.timeslot - MIN(od.timeslot) OVER (PARTITION BY od.date) AS INTEGER) AS entry_slot, "
            "CAST(od.originStation AS VARCHAR) AS reported_origin_id, CAST(od.destinationStation AS VARCHAR) AS reported_destination_id, "
            "origin_map.canonical_station_id AS origin_id, destination_map.canonical_station_id AS destination_id, "
            "CAST(od.Flow AS BIGINT) AS flow, 'station_crosswalk' AS label_version, 'excluded' AS diagonal_policy "
            f"FROM {od_source} od JOIN station_crosswalk origin_map ON CAST(od.originStation AS VARCHAR) = origin_map.reported_station_id "
            "JOIN station_crosswalk destination_map ON CAST(od.destinationStation AS VARCHAR) = destination_map.reported_station_id "
            "WHERE origin_map.canonical_station_id <> destination_map.canonical_station_id"
        )
        connection.execute(f"CREATE TEMP TABLE canonical_od AS {canonical_od_sql}")
        connection.execute(
            "CREATE TEMP TABLE self_loop_candidate AS WITH od_rows AS ("
            "SELECT service_date, entry_slot, origin_id, SUM(flow)::BIGINT AS non_diagonal_flow FROM canonical_od GROUP BY ALL) "
            "SELECT s.service_date, s.slot AS entry_slot, s.station_id, "
            "(s.entry_count - COALESCE(o.non_diagonal_flow, 0))::BIGINT AS candidate_flow, "
            "'structural_residual_high' AS confidence, 'self_loop_residual' AS derivation_version "
            "FROM station_flow s LEFT JOIN od_rows o ON s.service_date = o.service_date "
            "AND s.slot = o.entry_slot AND s.station_id = o.origin_id"
        )
        minimum_residual = connection.execute("SELECT MIN(candidate_flow) FROM self_loop_candidate").fetchone()[0]
        if minimum_residual is None or minimum_residual < 0:
            raise ValueError(f"non-diagonal OD flow exceeds observed entry; minimum residual={minimum_residual}")

        for split in ("train", "val", "test"):
            predicate = _date_predicate(split_policy.dates_for(split))
            if connection.execute(f"SELECT COUNT(*) FROM canonical_od WHERE {predicate}").fetchone()[0]:
                copy_partitioned(connection, f"SELECT * FROM canonical_od WHERE {predicate}", staging / "labels" / split / "od_entry_time")
            if connection.execute(f"SELECT COUNT(*) FROM self_loop_candidate WHERE {predicate}").fetchone()[0]:
                copy_partitioned(connection, f"SELECT * FROM self_loop_candidate WHERE {predicate}", staging / "nuisance" / split / "self_loop_candidate")

        stress_predicate = _date_predicate(split_policy.missingness_stress_dates)
        if connection.execute(
            f"SELECT COUNT(*) FROM canonical_od WHERE {stress_predicate}"
        ).fetchone()[0]:
            copy_partitioned(
                connection,
                f"SELECT * FROM canonical_od WHERE {stress_predicate}",
                staging / "stress" / "missingness" / "od_entry_time",
            )
        if connection.execute(
            f"SELECT COUNT(*) FROM self_loop_candidate WHERE {stress_predicate}"
        ).fetchone()[0]:
            copy_partitioned(
                connection,
                f"SELECT * FROM self_loop_candidate WHERE {stress_predicate}",
                staging / "stress" / "missingness" / "self_loop_candidate",
            )

        connection.close()
        staging.replace(output_root)
        return output_root
    except Exception:
        print(f"canonicalization staging retained at: {staging}", file=sys.stderr)
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass


def read_day_trips(con, canonical_root: Path, date_str: str, split: str, stations):
    """One service day's origin-on-line trips as ``(origin, dest, flow)`` rows.

    Args:
        con: DuckDB connection.
        canonical_root: canonical parquet root (from :func:`canonicalize`).
        date_str: ISO service date.
        split: ``"train"``, ``"val"`` or ``"test"``.
        stations: ordered canonical station IDs of the line.

    Returns:
        List of ``(origin, dest, flow)`` tuples (diagonal excluded), or None if
        the date has no label partition.
    """
    station_list = ",".join(str(s) for s in stations)
    files = sorted(glob.glob(
        str(Path(canonical_root) / "labels" / split / "od_entry_time"
            / f"service_date={date_str}" / "*.parquet")
    ))
    if not files:
        return None
    quoted = ",".join(f"'{_sql_path(Path(f))}'" for f in files)
    rows = con.execute(
        f"SELECT origin_id, destination_id, SUM(flow)::DOUBLE AS f "
        f"FROM read_parquet([{quoted}]) "
        f"WHERE origin_id IN ({station_list}) "
        f"AND origin_id != destination_id "
        f"GROUP BY origin_id, destination_id"
    ).fetchall()
    return [(int(o), int(d), float(f)) for o, d, f in rows]


def read_day_flows(con, canonical_root: Path, date_str: str, measure: str, stations, order):
    """One service day's per-station entry or exit counts, order-aligned.

    Args:
        con: DuckDB connection.
        canonical_root: canonical parquet root.
        date_str: ISO service date.
        measure: ``"entry_count"`` or ``"exit_count"``.
        stations: ordered canonical station IDs of the line.
        order: station ID -> row index (from :class:`passenger_state.network.LineSpec`).

    Returns:
        [S] float array of summed daily counts, or None if the date has no
        observation partition.
    """
    if measure not in ("entry_count", "exit_count"):
        raise ValueError(f"measure must be 'entry_count' or 'exit_count', got {measure!r}")
    station_list = ",".join(str(s) for s in stations)
    files = sorted(glob.glob(
        str(Path(canonical_root) / "observations" / "station_flow"
            / f"service_date={date_str}" / "*.parquet")
    ))
    if not files:
        return None
    quoted = ",".join(f"'{_sql_path(Path(f))}'" for f in files)
    rows = con.execute(
        f"SELECT station_id, SUM({measure})::DOUBLE AS m "
        f"FROM read_parquet([{quoted}]) "
        f"WHERE station_id IN ({station_list}) "
        f"GROUP BY station_id"
    ).fetchall()
    flow = np.zeros(len(stations), dtype=np.float64)
    by_id = {r[0]: r[1] for r in rows}
    for sid in stations:
        flow[order[sid]] = by_id.get(str(sid), 0.0)
    return flow


def read_day_online_alighting(con, canonical_root: Path, date_str: str, split: str, stations, order):
    """One service day's on-line alighting (destination-side sums), order-aligned.

    Args:
        con: DuckDB connection.
        canonical_root: canonical parquet root.
        date_str: ISO service date.
        split: label split holding the date.
        stations: ordered canonical station IDs of the line.
        order: station ID -> row index.

    Returns:
        [S] float array of on-line alighting counts, or None if the date has no
        label partition.
    """
    station_list = ",".join(str(s) for s in stations)
    files = sorted(glob.glob(
        str(Path(canonical_root) / "labels" / split / "od_entry_time"
            / f"service_date={date_str}" / "*.parquet")
    ))
    if not files:
        return None
    quoted = ",".join(f"'{_sql_path(Path(f))}'" for f in files)
    rows = con.execute(
        f"SELECT destination_id, SUM(flow)::DOUBLE AS f "
        f"FROM read_parquet([{quoted}]) "
        f"WHERE origin_id IN ({station_list}) "
        f"AND destination_id IN ({station_list}) "
        f"AND origin_id != destination_id "
        f"GROUP BY destination_id"
    ).fetchall()
    alighting = np.zeros(len(stations), dtype=np.float64)
    for d, f in rows:
        alighting[order[int(d)]] = f
    return alighting


def load_station_graph(con, station_info_path):
    """Load stationInfo.csv and build (station_ids, graph)."""
    df = con.execute(
        f"SELECT stationID, name, lon, lat, neighbour "
        f"FROM read_csv_auto('{_sql_path(Path(station_info_path))}')"
    ).fetchdf()
    station_ids = [int(s) for s in df["stationID"]]
    neighbours = [ast.literal_eval(n) for n in df["neighbour"]]
    lons = [float(x) for x in df["lon"]]
    lats = [float(x) for x in df["lat"]]
    graph = build_graph(station_ids, neighbours, lons, lats)
    return station_ids, graph


def weekday(date_str: str) -> int:
    """Day of week (0=Monday) for an ISO date string."""
    return date.fromisoformat(date_str).weekday()
