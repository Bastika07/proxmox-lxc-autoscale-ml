"""Replay of real metrics from a Proxmox host.

`fixtures/pve_metrics_2026-11-25.json.gz` is /var/log/lxc_metrics.json as the
monitor wrote it on a 17-container host: 1000 cycles, 2026-11-24 22:46 to
2026-11-25 19:16 UTC, one sample about every 74 s. I/O and network counters
are zeroed and values rounded; everything else is unchanged, including two
incomplete cycles (indices 463 and 464) written while the monitor restarted.

Synthetic fixtures hid three faults that only a dry run on that host found
(timezones, one container per cycle, swap runaway). These tests run the model
on the real thing instead.

HOST_ALLOCATIONS and HOST_SCALING are the host's `pct config` and the
`scaling` section of its model config, read at 2026-11-25 19:20 UTC.
"""
import gzip
import json
import os
from datetime import UTC, datetime

import pandas as pd
import pytest
import yaml

from scaling_decisions import determine_scaling_action
from test_scaling_stability import SHIPPED_CONFIG, direction_changes

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                       "pve_metrics_2026-11-25.json.gz")

HOST_ALLOCATIONS = {  # container: (cores, memory MB, swap MB)
    "100": (2, 512, 512), "101": (2, 8192, 0), "102": (2, 512, 512),
    "103": (2, 512, 512), "104": (2, 1024, 512), "105": (2, 512, 512),
    "106": (2, 8704, 512), "111": (2, 512, 512), "113": (2, 2560, 4096),
    "114": (2, 512, 512), "115": (2, 1024, 512), "117": (2, 512, 512),
    "120": (2, 512, 512), "121": (2, 512, 256), "122": (2, 1536, 512),
    "123": (2, 512, 1024), "124": (2, 512, 512),
}

HOST_SCALING = {
    "max_cpu_cores": 8, "min_cpu_cores": 2, "max_ram_mb": 16384, "min_ram_mb": 512,
    "cpu_scale_step": 1, "ram_scale_step_mb": 512,
    "cpu_scale_up_threshold": 75, "cpu_scale_down_threshold": 30,
    "ram_scale_up_threshold": 75, "ram_scale_down_threshold": 30,
    "min_confidence": 0, "dry_run": False,
}


def host_snapshots():
    with gzip.open(FIXTURE, "rt") as fh:
        return json.load(fh)


def host_config(tmp_path, data_file):
    with open(SHIPPED_CONFIG) as fh:
        config = yaml.safe_load(fh)
    config["scaling"].update(HOST_SCALING)
    config.update(data_file=str(data_file), log_file=str(tmp_path / "model.log"),
                  lock_file=str(tmp_path / "model.lock"))
    return config


def write_as_of_now(snapshots, path):
    """The snapshots, shifted so the last one is from now: older metrics are
    refused as stale, and that is not what these tests are about."""
    last = max(pd.Timestamp(m["timestamp"]) for m in snapshots[-1].values() if "timestamp" in m)
    shift = pd.Timestamp(datetime.now(UTC)) - last
    shifted = []
    for snapshot in snapshots:
        moved = {}
        for cid, metrics in snapshot.items():
            if "timestamp" in metrics:
                metrics = {**metrics, "timestamp": (pd.Timestamp(metrics["timestamp"]) + shift).isoformat()}
            moved[cid] = metrics
        shifted.append(moved)
    path.write_text(json.dumps(shifted))
    return path


@pytest.fixture
def orchestrator(monkeypatch):
    import lxc_autoscale_ml as orch

    applied, evaluated = [], []
    monkeypatch.setattr(
        orch, "apply_scaling",
        lambda cid, cores, ram, config, **kw: applied.append((cid, cores, ram)))
    monkeypatch.setattr(
        orch, "fetch_container_configs_sync",
        lambda ids, url, **kw: {cid: {"cores": HOST_ALLOCATIONS[cid][0],
                                      "memory_mb": HOST_ALLOCATIONS[cid][1],
                                      "swap_mb": HOST_ALLOCATIONS[cid][2]} for cid in ids})
    real = orch.evaluate_container

    def record(container_id, **kwargs):
        evaluated.append(container_id)
        return real(container_id=container_id, **kwargs)

    monkeypatch.setattr(orch, "evaluate_container", record)
    orch.applied, orch.evaluated = applied, evaluated
    return orch


def test_a_full_cycle_on_the_host_metrics_evaluates_every_container_and_scales_none(
        orchestrator, tmp_path):
    """The dry run on the host at 18:51 UTC changed nothing; the last cycle of
    the export, 25 minutes later, must not either."""
    path = write_as_of_now(host_snapshots(), tmp_path / "metrics.json")
    assert orchestrator.run_cycle(host_config(tmp_path, path)) is True
    assert sorted(orchestrator.evaluated) == sorted(HOST_ALLOCATIONS)
    assert orchestrator.applied == []


def test_a_cycle_on_an_incomplete_snapshot_evaluates_the_containers_it_has(orchestrator, tmp_path):
    """Cycle 464 was written while the monitor restarted and holds 13 of 17."""
    snapshots = host_snapshots()[:465]
    present = [cid for cid in snapshots[-1] if cid != "summary"]
    assert len(present) == 13
    path = write_as_of_now(snapshots, tmp_path / "metrics.json")
    assert orchestrator.run_cycle(host_config(tmp_path, path)) is True
    assert sorted(orchestrator.evaluated) == sorted(present)


def replay(container_id):
    """Close the loop over the whole export for one container.

    CPU usage is relative to the allocation, so the recorded value is turned
    back into cores' worth of work at the 2 cores the host ran with and
    divided by whatever the replay allocated since. RAM and swap in use do not
    depend on the limit while it is above them.
    """
    config = {"scaling": HOST_SCALING}
    cores, ram, _ = HOST_ALLOCATIONS[container_id]
    cores_history, ram_history = [cores], [ram]
    previous = None
    for snapshot in host_snapshots():
        metrics = snapshot.get(container_id)
        if metrics is None:
            continue
        demand = metrics["cpu_usage_percent"] / 100 * 2
        usage = min(100.0, demand / cores * 100)
        latest = {
            "cpu_usage_percent": usage,
            "previous_cpu_usage_percent": previous,
            "memory_usage_mb": metrics["memory_usage_mb"],
            "swap_usage_mb": metrics["swap_usage_mb"],
            "swap_total_mb": metrics["swap_total_mb"],
            "current_cores": cores,
            "current_ram_mb": ram,
        }
        _, _, new_cores, new_ram = determine_scaling_action(latest, 1, 100.0, config)
        previous = usage
        cores = new_cores if new_cores is not None else cores
        ram = new_ram if new_ram is not None else ram
        cores_history.append(cores)
        ram_history.append(ram)
    return cores_history, ram_history


def quick_reversals(history, within):
    """Changes undone again within `within` samples."""
    changes = [(i, b - a) for i, (a, b) in enumerate(zip(history, history[1:], strict=False)) if b != a]
    return [(i, j) for (i, d), (j, e) in zip(changes, changes[1:], strict=False)
            if (d > 0) != (e > 0) and j - i <= within]


@pytest.mark.parametrize("container_id", sorted(HOST_ALLOCATIONS))
def test_no_container_flaps_over_a_day_of_host_metrics(container_id):
    """Five samples is about six minutes, two model cycles on the host."""
    cores, ram = replay(container_id)
    assert quick_reversals(cores, within=5) == [], f"CPU flapped: {cores}"
    assert quick_reversals(ram, within=5) == [], f"RAM flapped: {ram}"


@pytest.mark.parametrize("container_id", sorted(HOST_ALLOCATIONS))
def test_no_container_runs_away_over_a_day_of_host_metrics(container_id):
    """Swap runaway added 512 MB every cycle up to the ceiling."""
    _, ram = replay(container_id)
    start = HOST_ALLOCATIONS[container_id][1]
    assert max(ram) <= start + 2 * HOST_SCALING["ram_scale_step_mb"], ram


def test_container_106_keeps_its_cores_through_its_half_hourly_bursts():
    """On the host it went 2 -> 3 -> 2 cores three times on 2026-11-25
    (11:34, 12:05, 12:35 UTC), each time on a single busy sample."""
    cores, _ = replay("106")
    assert direction_changes(cores) == 0 and set(cores) == {2}, cores
