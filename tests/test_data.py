"""Checks on the derived Parquet against the documented schema and Phase 1 findings."""

import polars as pl
import pytest

from src.config import ALL_TABLES
from src.io import BOOL_AS_FLOAT, SCHEMAS, csv_path, load, parse_time, scan, scan_game_tracking

pytestmark = pytest.mark.data

EXPECTED_ROWS = {
    "players": 510, "combine_results": 510, "combine_tracking": 463_189,
    "player_career_successes": 510, "player_play": 314_197, "games": 1_002,
    "game_tracking_2023": 3_665_443, "game_tracking_2024": 7_827_944,
    "game_tracking_2025": 11_160_903,
}


@pytest.mark.parametrize("name", ALL_TABLES)
def test_csv_header_matches_schema(name):
    header = csv_path(name).open().readline().strip().replace('"', "").split(",")
    assert set(header) == set(SCHEMAS[name])


@pytest.mark.parametrize("name", ALL_TABLES)
def test_parquet_dtypes_match_schema(name):
    schema = scan(name).collect_schema()
    for col, dtype in SCHEMAS[name].items():
        assert schema[col] == dtype, f"{name}.{col}: {schema[col]} != {dtype}"


@pytest.mark.parametrize("name", ALL_TABLES)
def test_row_counts(name):
    assert scan(name).select(pl.len()).collect().item() == EXPECTED_ROWS[name]


def test_bool_as_float_columns_are_0_1_or_null():
    pp = scan("player_play")
    for col in BOOL_AS_FLOAT["player_play"]:
        vals = pp.select(pl.col(col).drop_nulls().unique()).collect()[col].to_list()
        assert set(vals) <= {0.0, 1.0}, col


def test_primary_keys_unique():
    assert load("players")["nfl_id"].is_unique().all()
    assert load("games")["game_id"].is_unique().all()
    pp = scan("player_play").select("game_id", "play_id", "nfl_id").collect()
    assert not pp.is_duplicated().any()


def test_foreign_keys_resolve():
    ids = load("players")["nfl_id"].implode()
    games = load("games")["game_id"].implode()
    pp = scan("player_play").select("nfl_id", "game_id").collect()
    assert pp["nfl_id"].is_in(ids).all()
    assert pp["game_id"].is_in(games).all()
    assert load("combine_results")["nfl_id"].is_in(ids).all()


def test_combine_tracking_sampling_and_one_player_per_event():
    ct = load("combine_tracking")
    dt = (ct.sort("event_id", "entity_type", "time")
            .with_columns(parse_time().alias("ts"))
            .select(pl.col("ts").diff().over("event_id", "entity_type").dt.total_milliseconds())
            .drop_nulls().to_series())
    assert (dt == 100).all()
    per_event = ct.filter(pl.col("entity_type") == "PLAYER").group_by("event_id").agg(pl.col("nfl_id").n_unique())
    assert (per_event["nfl_id"] == 1).all()


def test_combine_accel_is_unsigned():
    # Phase 2 depends on this: deceleration must come from Δs, not from `a`.
    assert load("combine_tracking")["a"].min() >= 0


def test_load_refuses_game_tracking():
    with pytest.raises(ValueError):
        load("game_tracking_2023")


def test_scan_game_tracking_filters_push_down():
    some_id = load("players")["nfl_id"][0]
    out = scan_game_tracking(years=[2023], nfl_ids=[some_id], events=["ball_snap"],
                             columns=["nfl_id", "event"]).collect()
    assert set(out.columns) == {"nfl_id", "event", "season_file"}
    if out.height:
        assert (out["nfl_id"] == some_id).all()
        assert (out["event"] == "ball_snap").all()


def test_combine_dis_is_not_step_distance():
    # Path length must be computed from x/y: Combine `dis` behaves like the squared step.
    ct = load("combine_tracking").filter(pl.col("entity_type") == "PLAYER").sort("event_id", "time")
    d = (ct.with_columns(((pl.col("x").diff() ** 2 + pl.col("y").diff() ** 2).sqrt()).over("event_id").alias("step"))
           .filter(pl.col("step") > 0.05))
    assert (d["dis"] / d["step"]).median() < 0.7
    assert (d["dis"] / d["step"] ** 2).median() == pytest.approx(1.0, abs=0.1)
