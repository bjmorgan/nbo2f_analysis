"""Configuration dataclasses and YAML loader for the REWL driver."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOWED_SCHEDULES = {"1_over_t"}
ALLOWED_FLATNESS_MODES = {"pooled", "per_walker"}
ALLOWED_MERGE_CADENCES = {"at_halve", "never"}


@dataclass(frozen=True)
class SystemCfg:
    n_sc: int
    ce: str | None = None
    ce_path: Path | None = None


@dataclass(frozen=True)
class Window:
    lo: float
    hi: float
    walkers: int


@dataclass(frozen=True)
class WindowsCfg:
    energy_spacing: float
    list: tuple[Window, ...]

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(w.lo, w.hi) for w in self.list]

    @property
    def walkers_per_window(self) -> list[int]:
        return [w.walkers for w in self.list]


@dataclass(frozen=True)
class WlCfg:
    flatness_limit: float
    fill_factor_limit: float
    schedule: str
    flatness_mode: str
    merge_cadence: str
    n_trials_per_walker: int
    block_size_sweeps: int
    trajectory_write_interval_sweeps: int


@dataclass(frozen=True)
class ConfigSearchCfg:
    n_workers: int | None
    max_swaps: tuple[int, ...]
    attempts_per_swap_count: int
    random_attempts: int


@dataclass(frozen=True)
class CheckpointCfg:
    filename: str
    interval_cycles: int


@dataclass(frozen=True)
class RewlConfig:
    random_seed: int
    system: SystemCfg
    windows: WindowsCfg
    wl: WlCfg
    config_search: ConfigSearchCfg
    checkpoint: CheckpointCfg
    source_path: Path = field(default=Path("."))


def load_yaml(path: str | Path) -> RewlConfig:
    """Parse a YAML config file into a validated `RewlConfig`."""
    path = Path(path).resolve()
    with open(path, "r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    system_raw = raw["system"]
    ce = system_raw.get("ce")
    ce_path_raw = system_raw.get("ce_path")
    if (ce is None) == (ce_path_raw is None):
        raise ValueError(
            "system: exactly one of 'ce' (named preset) and 'ce_path' "
            "(filesystem path) must be set."
        )
    if ce_path_raw is not None:
        ce_path = Path(ce_path_raw)
        if not ce_path.is_absolute():
            ce_path = (path.parent / ce_path).resolve()
    else:
        ce_path = None
    system = SystemCfg(n_sc=int(system_raw["n_sc"]), ce=ce, ce_path=ce_path)
    if system.n_sc <= 0:
        raise ValueError(f"system.n_sc must be > 0, got {system.n_sc}")

    win_raw = raw["windows"]
    spacing = float(win_raw["energy_spacing"])
    if spacing <= 0:
        raise ValueError(
            f"windows.energy_spacing must be > 0, got {spacing}"
        )
    window_entries = []
    for entry in win_raw["list"]:
        if len(entry) != 3:
            raise ValueError(
                f"windows.list entries must be [lo, hi, walkers], "
                f"got {entry!r}"
            )
        lo, hi, walkers = entry
        if not (float(lo) < float(hi)):
            raise ValueError(
                f"windows.list entry {entry!r}: require lo < hi"
            )
        if int(walkers) < 1:
            raise ValueError(
                f"windows.list entry {entry!r}: walkers must be >= 1"
            )
        window_entries.append(
            Window(lo=float(lo), hi=float(hi), walkers=int(walkers))
        )
    if len(window_entries) < 1:
        raise ValueError("windows.list must contain at least one window")
    for i in range(1, len(window_entries)):
        prev = window_entries[i - 1]
        cur = window_entries[i]
        if cur.lo >= prev.hi:
            raise ValueError(
                f"windows {i - 1} and {i} do not overlap: "
                f"prev.hi={prev.hi}, cur.lo={cur.lo}"
            )
    windows = WindowsCfg(
        energy_spacing=spacing,
        list=tuple(window_entries),
    )

    wl_raw = raw["wl"]
    wl = WlCfg(
        flatness_limit=float(wl_raw["flatness_limit"]),
        fill_factor_limit=float(wl_raw["fill_factor_limit"]),
        schedule=str(wl_raw["schedule"]),
        flatness_mode=str(wl_raw["flatness_mode"]),
        merge_cadence=str(wl_raw["merge_cadence"]),
        n_trials_per_walker=int(wl_raw["n_trials_per_walker"]),
        block_size_sweeps=int(wl_raw["block_size_sweeps"]),
        trajectory_write_interval_sweeps=int(
            wl_raw["trajectory_write_interval_sweeps"]
        ),
    )
    if not (0.0 < wl.flatness_limit < 1.0):
        raise ValueError(
            f"wl.flatness_limit must be in (0, 1), got {wl.flatness_limit}"
        )
    if wl.fill_factor_limit <= 0:
        raise ValueError(
            f"wl.fill_factor_limit must be > 0, got {wl.fill_factor_limit}"
        )
    if wl.schedule not in ALLOWED_SCHEDULES:
        raise ValueError(
            f"wl.schedule={wl.schedule!r} not in {sorted(ALLOWED_SCHEDULES)}"
        )
    if wl.flatness_mode not in ALLOWED_FLATNESS_MODES:
        raise ValueError(
            f"wl.flatness_mode={wl.flatness_mode!r} not in "
            f"{sorted(ALLOWED_FLATNESS_MODES)}"
        )
    if wl.merge_cadence not in ALLOWED_MERGE_CADENCES:
        raise ValueError(
            f"wl.merge_cadence={wl.merge_cadence!r} not in "
            f"{sorted(ALLOWED_MERGE_CADENCES)}"
        )
    if wl.n_trials_per_walker <= 0:
        raise ValueError(
            f"wl.n_trials_per_walker must be > 0, "
            f"got {wl.n_trials_per_walker}"
        )
    if wl.block_size_sweeps <= 0:
        raise ValueError(
            f"wl.block_size_sweeps must be > 0, got {wl.block_size_sweeps}"
        )
    if wl.trajectory_write_interval_sweeps < 0:
        raise ValueError(
            f"wl.trajectory_write_interval_sweeps must be >= 0, "
            f"got {wl.trajectory_write_interval_sweeps}"
        )

    cs_raw = raw["config_search"]
    raw_max_swaps = cs_raw["max_swaps"]
    if not isinstance(raw_max_swaps, list) or not all(
        isinstance(x, int) and not isinstance(x, bool) for x in raw_max_swaps
    ):
        raise ValueError(
            f"config_search.max_swaps must be a list of ints, "
            f"got {raw_max_swaps!r}"
        )
    if not raw_max_swaps:
        raise ValueError(
            "config_search.max_swaps must contain at least one entry"
        )
    if any(x < 1 for x in raw_max_swaps):
        raise ValueError(
            f"config_search.max_swaps entries must all be >= 1, "
            f"got {raw_max_swaps!r}"
        )
    cs = ConfigSearchCfg(
        n_workers=(
            int(cs_raw["n_workers"])
            if cs_raw.get("n_workers") is not None
            else None
        ),
        max_swaps=tuple(raw_max_swaps),
        attempts_per_swap_count=int(cs_raw["attempts_per_swap_count"]),
        random_attempts=int(cs_raw["random_attempts"]),
    )

    ck_raw = raw["checkpoint"]
    ckpt = CheckpointCfg(
        filename=str(ck_raw["filename"]),
        interval_cycles=int(ck_raw["interval_cycles"]),
    )
    if ckpt.interval_cycles < 0:
        raise ValueError(
            f"checkpoint.interval_cycles must be >= 0, "
            f"got {ckpt.interval_cycles}"
        )

    return RewlConfig(
        random_seed=int(raw["random_seed"]),
        system=system,
        windows=windows,
        wl=wl,
        config_search=cs,
        checkpoint=ckpt,
        source_path=path,
    )


def resolve_ce_path(cfg: RewlConfig) -> Path:
    """Return an absolute filesystem path to the CE file referenced by `cfg`."""
    if cfg.system.ce_path is None and cfg.system.ce is None:
        raise ValueError(
            "system: neither 'ce' nor 'ce_path' is set; cannot resolve CE."
        )
    if cfg.system.ce_path is not None:
        if not cfg.system.ce_path.is_file():
            raise FileNotFoundError(
                f"CE path not found: {cfg.system.ce_path}"
            )
        return cfg.system.ce_path
    # Bundled preset: resolve via the package's bundled data directory.
    pkg_data_root = Path(__file__).resolve().parent.parent / "data" / "ces"
    candidate = pkg_data_root / f"{cfg.system.ce}.ce"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Bundled CE preset {cfg.system.ce!r} not found at {candidate}"
        )
    return candidate
