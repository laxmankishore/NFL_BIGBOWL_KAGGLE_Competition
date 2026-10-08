"""Project paths and constants. All IO goes through these paths."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Relative paths (per NEXT_STEPS.md); resolved against the repo root so
# notebooks work regardless of their working directory.
DATA_DIR = Path("data/nfl-big-data-bowl-2027")
PARQUET_DIR = Path("data/parquet")

DATA_PATH = REPO_ROOT / DATA_DIR
PARQUET_PATH = REPO_ROOT / PARQUET_DIR
FIG_PATH = REPO_ROOT / "figures"

SEED = 2027

SMALL_TABLES = [
    "players",
    "combine_results",
    "combine_tracking",
    "player_career_successes",
    "player_play",
    "games",
]
TRACKING_YEARS = [2023, 2024, 2025]
ALL_TABLES = SMALL_TABLES + [f"game_tracking_{y}" for y in TRACKING_YEARS]
