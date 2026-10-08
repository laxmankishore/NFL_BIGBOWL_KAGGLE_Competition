"""Unit tests for the sketch break metric on synthetic speed profiles."""

import numpy as np
import polars as pl
import pytest

from src.combine_features import _break_frame, player_best, sketch_break_metrics, with_signed_accel


def comeback_profile() -> np.ndarray:
    """Accelerate to 8.5, drop to 3.3 at the break, re-accelerate to 5.8, then slow
    to 3.2 at the end (slightly below the break speed)."""
    up = np.linspace(0.4, 8.5, 22)
    down = np.linspace(8.5, 3.3, 13)[1:]
    reaccel = np.linspace(3.3, 5.8, 10)[1:]
    tail = np.linspace(5.8, 3.2, 11)[1:]
    return np.concatenate([up, down, reaccel, tail])


def as_frames(s: np.ndarray, event_id="e1", nfl_id=1, drill="COMEBACK_ROUTE_RIGHT", attempt=1):
    n = len(s)
    return pl.DataFrame({
        "event_id": [event_id] * n,
        "nfl_id": [nfl_id] * n,
        "entity_type": ["PLAYER"] * n,
        "time": [f"2024-03-02T10:00:{i // 10:02d}.{i % 10}00" for i in range(n)],
        "drill_name": [drill] * n,
        "attempt": [attempt] * n,
        "s": s,
    })


def test_break_is_first_local_min_not_end_of_drill():
    # Regression: the global minimum after the peak is the end of the drill.
    s = comeback_profile()
    peak, brk = _break_frame(s)
    assert peak == 21
    assert brk == 33
    assert s[brk] == pytest.approx(3.3)


def test_no_break_on_monotonic_sprint():
    assert _break_frame(np.linspace(0, 9, 40)) is None


def test_no_break_on_shallow_dip():
    s = np.concatenate([np.linspace(0, 8, 20), [7.6, 7.4, 7.6], np.linspace(7.8, 8.5, 15)])
    assert _break_frame(s) is None  # dip < 1 yd/s is not a break


def test_too_short_returns_none():
    assert _break_frame(np.array([1.0, 5.0, 2.0, 4.0])) is None


def test_signed_accel_is_negative_when_slowing():
    df = with_signed_accel(as_frames(comeback_profile()))
    a = df["ds_dt"].to_numpy()
    assert a[5] > 0          # accelerating in the stem
    assert a[27] < 0         # decelerating into the break
    assert df["t"][10] == pytest.approx(1.0)


def test_sketch_metrics_on_synthetic_event():
    out = sketch_break_metrics(as_frames(comeback_profile()))
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["break_frame"] == 33
    assert row["entry_speed"] == pytest.approx(8.5)
    assert row["break_speed"] == pytest.approx(3.3)
    # (8.5 - 3.3) over 1.2 s ≈ 4.3 yd/s²; smoothing keeps it close
    assert 3.5 < row["peak_decel"] < 5.0
    assert row["exit_accel"] > 0
    assert row["break_time_s"] > 0


def test_sketch_metrics_ignores_other_drills_and_ball():
    frames = pl.concat([
        as_frames(comeback_profile(), event_id="slant", drill="SLANT_ROUTE_LEFT"),
        as_frames(comeback_profile(), event_id="ball").with_columns(pl.lit("BALL").alias("entity_type")),
    ])
    assert sketch_break_metrics(frames).height == 0


def test_player_best_keeps_shortest_break_time():
    ev = pl.DataFrame({
        "nfl_id": [1, 1, 2],
        "drill_name": ["CURL_ROUTE_RIGHT"] * 3,
        "attempt": [1, 2, 1],
        "break_time_s": [2.4, 1.8, 3.0],
    })
    best = player_best(ev).sort("nfl_id")
    assert best["attempt"].to_list() == [2, 1]
