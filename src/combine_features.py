"""Combine tracking features.

Never sum Combine `dis` for path length: it behaves like the squared frame
step. Use x/y differences instead.

Phase 1 holds only a *sketch* break metric, used by the thesis-lock check
"tracking is not a 1:1 proxy of the stopwatch". The real feature set is
Phase 2 work.

Note: ``a`` in combine_tracking is an unsigned magnitude (min 0), so
deceleration is derived from the frame-to-frame change in ``s``.
"""

import numpy as np
import polars as pl

DT = 0.1  # seconds between frames

# Drills whose break is a single speed minimum (decel into a stop, re-accel out).
SPEED_DIP_DRILLS = ["COMEBACK_ROUTE_RIGHT", "CURL_ROUTE_RIGHT"]


def with_signed_accel(ct: pl.DataFrame, smooth: int = 3) -> pl.DataFrame:
    """Add frame index, elapsed seconds and signed tangential accel ds/dt."""
    return (
        ct.sort("event_id", "time")
        .with_columns(pl.int_range(pl.len()).over("event_id").alias("frame"))
        .with_columns(
            (pl.col("frame") * DT).alias("t"),
            (pl.col("s").diff().over("event_id") / DT)
            .rolling_mean(smooth, center=True, min_samples=1)
            .over("event_id")
            .alias("ds_dt"),
        )
    )


def _break_frame(s: np.ndarray) -> tuple[int, int] | None:
    """(peak_idx, break_idx): top speed in the stem, then the first local speed
    minimum after it that is at least 1 yd/s below the peak.

    The first local minimum (not the global one) avoids picking the player
    slowing down at the end of the drill.
    """
    if len(s) < 15:
        return None
    peak = int(np.argmax(s[: int(len(s) * 0.7)]))
    for i in range(peak + 1, len(s) - 3):
        if s[i] <= s[i - 1] and s[i] < s[i + 1] and s[peak] - s[i] >= 1.0:
            return peak, i
    return None


def sketch_break_metrics(ct: pl.DataFrame, drills=SPEED_DIP_DRILLS) -> pl.DataFrame:
    """One row per event: entry speed, decel into break, exit accel, break time.

    break_time_s: seconds from the last frame >= 90% of entry speed before the
    break to the first frame back at >= 60% of entry speed after it. Shorter
    means the player sheds and regains speed faster.
    """
    df = with_signed_accel(
        ct.filter((pl.col("entity_type") == "PLAYER") & pl.col("drill_name").is_in(drills))
    )
    rows = []
    for (event_id,), g in df.group_by("event_id", maintain_order=True):
        s = g["s"].to_numpy()
        a = g["ds_dt"].to_numpy()
        found = _break_frame(s)
        if found is None:
            continue
        peak, brk = found
        entry = s[peak]
        pre = np.where(s[: brk + 1] >= 0.9 * entry)[0]
        post = np.where(s[brk:] >= 0.6 * entry)[0]
        if len(pre) == 0 or len(post) == 0:
            continue
        exit_win = a[brk : brk + 6]  # 0.5 s after the break
        rows.append(
            {
                "event_id": event_id,
                "nfl_id": g["nfl_id"][0],
                "drill_name": g["drill_name"][0],
                "attempt": g["attempt"][0],
                "entry_speed": float(entry),
                "break_speed": float(s[brk]),
                "peak_decel": float(-np.nanmin(a[peak : brk + 1])),
                "exit_accel": float(np.nanmax(exit_win)),
                "break_time_s": float((brk + post[0] - pre[-1]) * DT),
                "break_frame": brk,
            }
        )
    return pl.DataFrame(rows)


def player_best(events: pl.DataFrame) -> pl.DataFrame:
    """Best attempt per player x drill (shortest break time)."""
    return (
        events.sort("break_time_s")
        .group_by("nfl_id", "drill_name", maintain_order=True)
        .first()
    )
