"""NFL play-level filters and outcome tables (Phase 1: filters only)."""

import polars as pl

from src.io import load


def play_frame() -> pl.DataFrame:
    """player_play joined to season/season_type/week and roster position."""
    games = load("games").select("game_id", "season", "season_type", "week")
    players = load("players").select("nfl_id", "display_name", "nfl_position", "draft_year")
    return load("player_play").join(games, on="game_id", how="left").join(
        players, on="nfl_id", how="left"
    )


def regular_season(df: pl.DataFrame) -> pl.DataFrame:
    """Proposed base filter: REG games only, plays not wiped out by penalty.

    player_play contains PRE and POST games even though the brief targets
    regular-season performance.
    """
    return df.filter(
        (pl.col("season_type") == "REG") & (pl.col("play_nullified_by_penalty") == "N")
    )


def wr_route_snaps(df: pl.DataFrame, positions=("WR",)) -> pl.DataFrame:
    """Route-running snaps with a separation value (targeted or not)."""
    return df.filter(
        pl.col("nfl_position").is_in(list(positions))
        & pl.col("route_ran").is_not_null()
        & pl.col("separation_at_pass_forward").is_not_null()
    )


BREAKING_ROUTES = ["OUT", "SLANT", "HITCH", "CORNER", "IN"]
VERTICAL_ROUTES = ["GO", "POST"]
