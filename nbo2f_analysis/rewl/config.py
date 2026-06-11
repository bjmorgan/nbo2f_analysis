"""Configuration dataclasses and YAML loader for the REWL driver."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOWED_SCHEDULES = {"1_over_t"}
ALLOWED_FLATNESS_MODES = {"pooled", "per_walker"}
ALLOWED_MERGE_CADENCES = {"at_halve", "never"}
ALLOWED_ONE_OVER_T_GATES = {"visit_once", "flatness"}
ALLOWED_ONE_OVER_T_ENTRIES = {"window_clock", "f_continuous"}


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
    one_over_t_gate: str = "visit_once"
    bp_stall_multiple: float = 4.0
    one_over_t_entry: str = "window_clock"


@dataclass(frozen=True)
class MoveSpec:
    type: str
    weight: float


@dataclass(frozen=True)
class MovesCfg:
    list: tuple[MoveSpec, ...]


@dataclass(frozen=True)
class ConfigSearchCfg:
    n_workers: int | None
    window_search_penalty: float
    walk_sweeps: int
    max_walks_per_window: int


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
    moves: MovesCfg
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
        one_over_t_gate=str(wl_raw.get("one_over_t_gate", "visit_once")),
        bp_stall_multiple=float(wl_raw.get("bp_stall_multiple", 4.0)),
        one_over_t_entry=str(wl_raw.get("one_over_t_entry", "window_clock")),
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
    if wl.one_over_t_gate not in ALLOWED_ONE_OVER_T_GATES:
        raise ValueError(
            f"wl.one_over_t_gate={wl.one_over_t_gate!r} not in "
            f"{sorted(ALLOWED_ONE_OVER_T_GATES)}"
        )
    if not math.isfinite(wl.bp_stall_multiple) or wl.bp_stall_multiple <= 0:
        raise ValueError(
            f"wl.bp_stall_multiple must be a finite positive number, "
            f"got {wl.bp_stall_multiple}"
        )
    if wl.one_over_t_entry not in ALLOWED_ONE_OVER_T_ENTRIES:
        raise ValueError(
            f"wl.one_over_t_entry={wl.one_over_t_entry!r} not in "
            f"{sorted(ALLOWED_ONE_OVER_T_ENTRIES)}"
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

    # Avoid a top-level circular import — the nbo2f module imports from
    # chain_geometry/mchammer_moves which are heavy. Lazy import here
    # keeps `load_yaml` cheap for tests that don't need icet loaded.
    from nbo2f_analysis.rewl.nbo2f import ALLOWED_MOVE_TYPES

    moves_raw = raw.get("moves")
    if moves_raw is None:
        raise ValueError("moves: section is required in the config")
    if not isinstance(moves_raw, list) or not moves_raw:
        raise ValueError(
            f"moves: must be a non-empty list of {{type, weight}} entries, "
            f"got {moves_raw!r}"
        )
    seen_types: set[str] = set()
    move_specs: list[MoveSpec] = []
    for i, entry in enumerate(moves_raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"moves[{i}] must be a {{type, weight}} dict, got {entry!r}"
            )
        extra = set(entry) - {"type", "weight"}
        if extra or set(entry) != {"type", "weight"}:
            raise ValueError(
                f"moves[{i}] must have exactly the keys 'type' and 'weight', "
                f"got {sorted(entry)}"
            )
        type_ = str(entry["type"])
        if type_ not in ALLOWED_MOVE_TYPES:
            raise ValueError(
                f"moves[{i}].type={type_!r} not recognised. "
                f"Allowed: {sorted(ALLOWED_MOVE_TYPES)}"
            )
        if type_ in seen_types:
            raise ValueError(
                f"moves[{i}].type={type_!r} appears more than once"
            )
        seen_types.add(type_)
        weight = float(entry["weight"])
        if not (weight > 0.0):
            raise ValueError(
                f"moves[{i}].weight must be > 0, got {weight}"
            )
        move_specs.append(MoveSpec(type=type_, weight=weight))
    moves_cfg = MovesCfg(list=tuple(move_specs))

    cs_raw = raw["config_search"]
    cs = ConfigSearchCfg(
        n_workers=(
            int(cs_raw["n_workers"])
            if cs_raw.get("n_workers") is not None
            else None
        ),
        window_search_penalty=float(cs_raw["window_search_penalty"]),
        walk_sweeps=int(cs_raw["walk_sweeps"]),
        max_walks_per_window=int(cs_raw["max_walks_per_window"]),
    )
    if cs.n_workers is not None and cs.n_workers < 1:
        raise ValueError(
            f"config_search.n_workers must be >= 1 when set, "
            f"got {cs.n_workers}"
        )
    if not (cs.window_search_penalty > 0):
        raise ValueError(
            f"config_search.window_search_penalty must be > 0, "
            f"got {cs.window_search_penalty}"
        )
    if cs.walk_sweeps < 1:
        raise ValueError(
            f"config_search.walk_sweeps must be >= 1, got {cs.walk_sweeps}"
        )
    if cs.max_walks_per_window < 1:
        raise ValueError(
            f"config_search.max_walks_per_window must be >= 1, "
            f"got {cs.max_walks_per_window}"
        )
    max_walkers: int = max(windows.walkers_per_window)
    if cs.max_walks_per_window < max_walkers:
        raise ValueError(
            f"config_search.max_walks_per_window ({cs.max_walks_per_window}) "
            f"must be >= the largest window walker count ({max_walkers}); "
            f"a window gains at most one config per walk round."
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
        moves=moves_cfg,
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
