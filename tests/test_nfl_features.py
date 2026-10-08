"""Unit tests for the NFL play filters."""

import polars as pl

from src.nfl_features import regular_season, wr_route_snaps


def plays():
    return pl.DataFrame({
        "season_type": ["REG", "REG", "PRE", "POST", "REG"],
        "play_nullified_by_penalty": ["N", "Y", "N", "N", "N"],
        "nfl_position": ["WR", "WR", "WR", "WR", "CB"],
        "route_ran": ["OUT", "GO", "SLANT", "POST", None],
        "separation_at_pass_forward": [2.1, 3.0, 1.0, 4.0, None],
    })


def test_regular_season_drops_pre_post_and_nullified():
    out = regular_season(plays())
    assert out.height == 2
    assert out["season_type"].unique().to_list() == ["REG"]
    assert (out["play_nullified_by_penalty"] == "N").all()


def test_wr_route_snaps_needs_route_and_separation():
    df = pl.DataFrame({
        "nfl_position": ["WR", "WR", "WR", "TE"],
        "route_ran": ["OUT", None, "GO", "OUT"],
        "separation_at_pass_forward": [2.0, 1.0, None, 3.0],
    })
    assert wr_route_snaps(df).height == 1
    assert wr_route_snaps(df, positions=("WR", "TE")).height == 2
