"""CSV -> Parquet conversion and lazy readers.

Dtypes follow Data_Desctiption.MD. Raw CSVs use the literal ``NA`` for nulls
and ``TRUE``/``FALSE`` for booleans. Timestamps stay as ISO-8601 strings in
Parquet (as documented); use ``parse_time`` when time arithmetic is needed.
"""

import polars as pl

from src.config import ALL_TABLES, DATA_PATH, PARQUET_PATH, TRACKING_YEARS

I, F, S, B = pl.Int64, pl.Float64, pl.String, pl.Boolean

# Documented as float64 (1.0/0.0) but stored as TRUE/FALSE in the CSV.
# Parsed as Boolean, then cast to Float64 so the Parquet matches the doc.
BOOL_AS_FLOAT = {
    "player_play": ["quick_pressure", "unblocked_pressure", "recovered_fumble", "tackle_for_loss"],
}

SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "players": {
        "nfl_id": I, "display_name": S, "draft_year": I, "nfl_position": S,
        "birth_date": S, "college_name": S, "college_conference": S,
        "draft_round": I, "draft_pick_within_round": I, "draft_overall_pick": I,
    },
    "combine_results": {
        "draft_year": I, "nfl_id": I, "combine_position": S,
        "combine_height": F, "combine_weight": I, "hand_size": F,
        "arm_length": F, "wing_span": F, "ten_yd_split": F, "forty": F,
        "vertical": F, "broad_jump": F, "three_cone": F, "short_shuttle": F,
        "bench_reps": F, "ngs_athleticism_score": I,
        "ngs_college_production_score": F, "ngs_final_score": F,
    },
    "combine_tracking": {
        "draft_year": I, "event_id": S, "nfl_id": I, "entity_type": S,
        "time": S, "drill_type": S, "drill_name": S, "attempt": I,
        "x": F, "y": F, "s": F, "a": F, "dis": F, "dir": F,
    },
    "player_career_successes": {
        "nfl_id": I, "career_offensive_snaps": I, "career_defensive_snaps": I,
        "career_special_teams_snaps": I, "career_games_active": I,
        "career_games_started": I, "ap_all_pro_1st_team": I,
        "ap_all_pro_2nd_team": I, "pro_bowl_original_ballot": I,
    },
    "games": {
        "game_id": I, "game_key": I, "season": I, "season_type": S,
        "week": I, "home_team_abbr": S, "visitor_team_abbr": S,
    },
    "player_play": {
        # A. context
        "game_id": I, "play_id": I, "team_abbr": S, "nfl_id": I,
        "lined_up_position": S, "play_description": S, "quarter": I,
        "down": I, "yards_to_go": I, "possession_team": S,
        "yardline_side": S, "yardline_number": I, "game_clock": S,
        "pre_snap_home_score": I, "pre_snap_visitor_score": I,
        "play_direction": S,
        # B. win prob / EP
        "pre_snap_home_team_win_probability": F,
        "pre_snap_visitor_team_win_probability": F,
        "home_team_win_probability_added": F,
        "visitor_team_win_probility_added": F,
        "expected_points_added": F, "expected_points": F,
        # C. scheme / result
        "offense_formation": S, "receiver_alignment": S, "pass_result": S,
        "pass_length": F, "play_nullified_by_penalty": S, "penalty_yards": F,
        "pre_penalty_yards_gained": I, "yards_gained": I,
        "team_coverage_man_zone": S, "team_coverage_type": S,
        "in_motion_at_ball_snap": B,
        # D. receiver
        "target": B, "rec_yards": F, "rush_yards": F, "yards_after_catch": F,
        "fumble": F, "fumble_lost": F, "route_ran": S,
        "separation_at_pass_forward": F, "cushion": F,
        "expected_yards_after_catch": F,
        # E. OL
        "dropback_duration": F, "extended_sack_allowed": F,
        "pass_rushers_encountered": F, "peak_pressure_probability_allowed": F,
        "pressure_allowed": B, "sack_allowed": F,
        "time_to_pressure_allowed": F, "quick_pressure": F,
        "unblocked_pressure": F,
        # F. defense
        "blitzing": B, "player_get_off": F, "sack": F, "tackle": F,
        "assist": F, "caused_forced_fumble": F, "recovered_fumble": F,
        "tackle_for_loss": F, "time_to_pressure": F, "time_to_qb_hurry": F,
        "coverage_assignment": S, "coverage_assignment_at_snap": S,
    },
}
_GAME_TRACKING = {
    "game_id": I, "play_id": I, "nfl_id": I, "time": S, "x": F, "y": F,
    "s": F, "a": F, "dis": F, "o": F, "dir": F, "event": S,
}
for _y in TRACKING_YEARS:
    SCHEMAS[f"game_tracking_{_y}"] = _GAME_TRACKING


def csv_path(name: str):
    return DATA_PATH / f"{name}.csv"


def parquet_path(name: str):
    return PARQUET_PATH / f"{name}.parquet"


def check_raw_files() -> None:
    """Stop if any official CSV is missing or empty."""
    bad = [n for n in ALL_TABLES if not csv_path(n).exists() or csv_path(n).stat().st_size == 0]
    if bad:
        raise FileNotFoundError(f"Missing or empty CSVs in {DATA_PATH}: {bad}")


def convert_one(name: str, overwrite: bool = False) -> int:
    """Stream one CSV to Parquet with the documented schema. Returns row count."""
    out = parquet_path(name)
    if out.exists() and not overwrite:
        return pl.scan_parquet(out).select(pl.len()).collect().item()
    src = csv_path(name)
    header = src.open().readline().strip().replace('"', "").split(",")
    schema = SCHEMAS[name]
    if set(header) != set(schema):
        raise ValueError(
            f"{name}: header/schema mismatch. "
            f"extra={set(header) - set(schema)} missing={set(schema) - set(header)}"
        )
    PARQUET_PATH.mkdir(parents=True, exist_ok=True)
    as_bool = BOOL_AS_FLOAT.get(name, [])
    lf = pl.scan_csv(
        src,
        schema={c: B if c in as_bool else schema[c] for c in header},
        null_values=["NA"],
    )
    if as_bool:
        lf = lf.with_columns(pl.col(as_bool).cast(F))
    tmp = out.with_suffix(".parquet.tmp")
    lf.sink_parquet(tmp, compression="zstd")
    tmp.replace(out)  # atomic: a failed run never leaves a partial file
    return pl.scan_parquet(out).select(pl.len()).collect().item()


def convert_all(overwrite: bool = False) -> dict[str, int]:
    check_raw_files()
    return {n: convert_one(n, overwrite) for n in ALL_TABLES}


def scan(name: str) -> pl.LazyFrame:
    """Lazy scan of a derived Parquet table (run ``convert_all`` first)."""
    p = parquet_path(name)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run scripts/convert_to_parquet.py")
    return pl.scan_parquet(p)


def load(name: str) -> pl.DataFrame:
    """Eager load. Intended for the small tables only."""
    if name.startswith("game_tracking"):
        raise ValueError("Use scan_game_tracking() for game tracking; do not load fully.")
    return scan(name).collect()


def scan_game_tracking(
    years=TRACKING_YEARS,
    nfl_ids=None,
    events=None,
    columns=None,
) -> pl.LazyFrame:
    """Year-wise lazy scan with optional player/event filters pushed down."""
    frames = []
    for y in years:
        lf = scan(f"game_tracking_{y}").with_columns(pl.lit(y).alias("season_file"))
        if nfl_ids is not None:
            lf = lf.filter(pl.col("nfl_id").is_in(list(nfl_ids)))
        if events is not None:
            lf = lf.filter(pl.col("event").is_in(list(events)))
        if columns is not None:
            lf = lf.select([*columns, "season_file"])
        frames.append(lf)
    return pl.concat(frames)


def parse_time(col: str = "time") -> pl.Expr:
    return pl.col(col).str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", time_unit="ms")
