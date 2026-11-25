"""Regressions found by a dry run against real metrics on a Proxmox host.

Three faults were hidden by the test fixtures, which wrote naive timestamps and
gave every container of a cycle the same one -- unlike the real monitor:

1. The monitor writes timezone-aware UTC timestamps, the staleness check
   compared them with a naive "now" and every cycle died with
   "Cannot subtract tz-naive and tz-aware".
2. The monitor stamped every container separately, so "rows with the newest
   timestamp" was a single container and only that one was evaluated.
3. A container whose swap stayed occupied after a scale-up was scaled up again
   every cycle until max_ram_mb, because swapped pages are not paged back in.
"""
import asyncio
import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from data_manager import load_data
from scaling_decisions import determine_scaling_action


CONFIG = {
    "api": {"api_url": "http://127.0.0.1:5000"},
    "spike_detection": {"spike_threshold": 2, "rolling_window": 3},
    "model": {"contamination": 0.05, "n_estimators": 20, "max_samples": 8, "random_state": 42},
    "scaling": {
        "min_cpu_cores": 1, "max_cpu_cores": 8,
        "min_ram_mb": 512, "max_ram_mb": 16384,
        "cpu_scale_up_threshold": 75, "cpu_scale_down_threshold": 30,
        "ram_scale_up_threshold": 75, "ram_scale_down_threshold": 30,
        "cpu_scale_step": 1, "ram_scale_step_mb": 512,
        "swap_scale_up_threshold": 20, "swap_scale_down_block_threshold": 5,
    },
}


def container_metrics(stamp, **overrides):
    metrics = {
        "timestamp": stamp,
        "cpu_usage_percent": 50.0,
        "memory_usage_mb": 300.0,
        "swap_usage_mb": 0.0,
        "swap_total_mb": 0.0,
        "process_count": 20,
        "io_stats": {"reads": 1, "writes": 1},
        "network_usage": {"rx_bytes": 1, "tx_bytes": 1},
        "filesystem_usage_gb": 2.0,
        "filesystem_total_gb": 8.0,
        "filesystem_free_gb": 6.0,
    }
    metrics.update(overrides)
    return metrics


def monitor_like_file(path, container_ids, cycles=8, per_container_stamps=True):
    """Metrics as the real monitor writes them: aware UTC isoformat, and (in
    monitors before the fix) a separate timestamp per container."""
    data = []
    base = datetime.now(UTC) - timedelta(seconds=60 * (cycles - 1))
    for cycle in range(cycles):
        snapshot = {}
        for n, cid in enumerate(container_ids):
            offset = timedelta(milliseconds=37 * n) if per_container_stamps else timedelta()
            stamp = (base + timedelta(seconds=60 * cycle) + offset).isoformat()
            snapshot[cid] = container_metrics(
                stamp, cpu_usage_percent=40.0 + cycle, memory_usage_mb=300.0 + cycle,
                process_count=20 + cycle)
        snapshot["summary"] = {}
        data.append(snapshot)
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def orchestrator(monkeypatch):
    import lxc_autoscale_ml as orch

    applied = []
    monkeypatch.setattr(
        orch, "apply_scaling",
        lambda cid, cores, ram, config, **kw: applied.append((cid, cores, ram)))
    monkeypatch.setattr(
        orch, "fetch_container_configs_sync",
        lambda ids, url, **kw: {cid: {"cores": 2, "memory_mb": 1024} for cid in ids})
    orch.applied = applied
    return orch


def evaluated_ids(orchestrator, monkeypatch, config):
    seen = []
    real = orchestrator.evaluate_container

    def record(container_id, **kwargs):
        seen.append(container_id)
        return real(container_id=container_id, **kwargs)

    monkeypatch.setattr(orchestrator, "evaluate_container", record)
    assert orchestrator.run_cycle(config) is True
    return seen


class TestTimezones:
    def test_a_cycle_on_monitor_timestamps_does_not_crash(self, orchestrator, tmp_path, caplog):
        path = monitor_like_file(tmp_path / "m.json", ["104"])
        assert orchestrator.run_cycle({**CONFIG, "data_file": path}) is True
        assert "tz-naive" not in caplog.text

    def test_timestamps_are_normalised_to_utc(self, tmp_path):
        stamps = [
            "2026-11-25T08:00:00.123456+00:00",   # current monitor
            "2026-11-25T09:00:01+01:00",          # other offset, no microseconds
            "2026-11-25T08:00:02Z",
        ]
        data = [{"104": container_metrics(s), "summary": {}} for s in stamps]
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data))
        df = load_data(str(path))
        assert str(df["timestamp"].dt.tz) == "UTC"
        assert list(df["timestamp"].dt.second) == [0, 1, 2]

    def test_naive_rows_from_an_old_monitor_mix_with_aware_ones(self, tmp_path):
        """Old naive rows stay in the file until they age out."""
        naive_local = datetime(2026, 11, 25, 9, 0, 0)
        aware = naive_local.astimezone().astimezone(UTC) + timedelta(seconds=60)
        data = [{"104": container_metrics(naive_local.isoformat()), "summary": {}},
                {"104": container_metrics(aware.isoformat()), "summary": {}}]
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data))
        df = load_data(str(path))
        assert df is not None
        # The naive value is read as local time, so the two are 60 s apart.
        assert (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds() == 60

    def test_staleness_is_measured_on_aware_timestamps(self, orchestrator):
        fresh = pd.DataFrame({"timestamp": [pd.Timestamp.now(tz="UTC")]})
        old = pd.DataFrame({"timestamp": [pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)]})
        assert orchestrator.metrics_are_stale(fresh, CONFIG) is False
        assert orchestrator.metrics_are_stale(old, CONFIG) is True


class TestEveryContainerOfTheNewestSnapshot:
    def test_per_container_timestamps_still_evaluate_the_whole_fleet(
            self, orchestrator, tmp_path, monkeypatch):
        """Files written by the unfixed monitor are still in use on the host."""
        ids = ["103", "104", "105", "113"]
        path = monitor_like_file(tmp_path / "m.json", ids, per_container_stamps=True)
        assert sorted(evaluated_ids(orchestrator, monkeypatch,
                                    {**CONFIG, "data_file": path})) == ids

    def test_a_container_missing_from_the_newest_snapshot_is_not_evaluated(
            self, orchestrator, tmp_path, monkeypatch):
        path = tmp_path / "m.json"
        monitor_like_file(path, ["104", "105"])
        data = json.loads(path.read_text())
        del data[-1]["105"]
        path.write_text(json.dumps(data))
        assert evaluated_ids(orchestrator, monkeypatch,
                             {**CONFIG, "data_file": str(path)}) == ["104"]

    def test_snapshot_is_not_a_model_feature(self, tmp_path):
        from data_manager import preprocess_data
        from model import train_anomaly_models

        df = preprocess_data(load_data(monitor_like_file(tmp_path / "m.json", ["104"])), CONFIG)
        _, features = train_anomaly_models(df, CONFIG)
        assert "snapshot" not in features


class TestMonitorSharesOneTimestampPerCycle:
    def test_all_containers_of_a_cycle_carry_the_same_timestamp(self, monkeypatch, tmp_path):
        import lxc_monitor
        from lxc_monitor import Settings

        for name in dir(Settings):
            if not name.startswith("_"):
                setattr(lxc_monitor.settings, name, getattr(Settings, name))
        lxc_monitor.settings.export_file = str(tmp_path / "metrics.json")
        monkeypatch.setattr(lxc_monitor, "get_running_lxc_containers",
                            lambda: ["103", "104", "105"])

        async def probe(func, *args, **kwargs):
            await asyncio.sleep(0.01)  # real probes take time; stamps would drift
            if func is lxc_monitor.get_container_cpu_usage:
                return 10.0
            return None

        monkeypatch.setattr(lxc_monitor, "retry_on_failure", probe)
        for parallel in (True, False):
            lxc_monitor.settings.parallel_processing = parallel
            asyncio.run(lxc_monitor.collect_and_export_metrics())

        written = json.loads((tmp_path / "metrics.json").read_text())
        assert len(written) == 2
        for snapshot in written:
            stamps = {m["timestamp"] for cid, m in snapshot.items() if cid != "summary"}
            assert len(stamps) == 1, stamps
            assert datetime.fromisoformat(stamps.pop()).tzinfo is not None


def decide(memory_usage_mb, swap_usage_mb, swap_total_mb, current_ram_mb, **scaling):
    config = {"scaling": {**CONFIG["scaling"], **scaling}}
    metrics = pd.Series({
        "cpu_usage_percent": 50.0, "memory_usage_mb": memory_usage_mb,
        "swap_usage_mb": swap_usage_mb, "swap_total_mb": swap_total_mb,
        "current_cores": 2, "current_ram_mb": current_ram_mb,
    })
    _, ram_action, _, new_ram = determine_scaling_action(metrics, 1, 90.0, config)
    return ram_action, new_ram


class TestSwapDoesNotRunAway:
    def test_swap_that_stays_after_a_scale_up_stops_the_series(self):
        """The observed case: swap stays above 20% because the kernel does not
        page it back in, so each cycle added 512 MB up to the ceiling. Replay
        the loop with the container's RAM use and swap unchanged."""
        ram = 512
        steps = 0
        for _ in range(40):  # 40 cycles = 2 hours at the host's 3 minutes
            action, new_ram = decide(300, 200, 512, ram)
            if action != "Scale Up":
                break
            ram, steps = new_ram, steps + 1
        assert ram == 1024 and steps == 1

    def test_swap_that_would_fit_in_ram_does_not_scale_up(self):
        # 60% of swap in use, but 1000 + 300 MB is only 32% of 4096 MB.
        assert decide(1000, 300, 512, 4096) == ("No Scaling", None)

    def test_swap_with_ram_too_small_to_hold_it_scales_up(self):
        # 300 MB in RAM + 200 MB in swap is 98% of 512 MB.
        assert decide(300, 200, 512, 512) == ("Scale Up", 1024)

    def test_growing_swap_keeps_scaling_while_it_does_not_fit(self):
        """Real pressure is still answered: if what is in swap keeps growing,
        the threshold is reached again at the larger limit."""
        assert decide(600, 300, 1024, 1024) == ("Scale Up", 1536)
        assert decide(600, 600, 1024, 1536) == ("Scale Up", 2048)
        assert decide(600, 600, 1024, 2048) == ("No Scaling", None)

    def test_swap_scale_up_and_scale_down_guard_do_not_oscillate(self):
        """After a swap-driven step the container must not qualify for a
        scale-down that would undo it."""
        action, ram = decide(300, 200, 512, 512)
        assert action == "Scale Up"
        assert decide(300, 200, 512, ram)[0] == "No Scaling"


class TestClampsNeverReverseTheDirection:
    def test_scale_down_below_the_floor_does_not_raise_ram(self):
        # 512 MB with min_ram_mb 1024 and 10% use: a "scale down" to 1024 MB.
        assert decide(51, 0, 0, 512, min_ram_mb=1024) == ("No Scaling", None)

    def test_scale_up_above_the_ceiling_does_not_lower_ram(self):
        assert decide(9000, 0, 0, 10240, max_ram_mb=8192) == ("No Scaling", None)

    def test_cpu_clamps_do_not_reverse_either(self):
        config = {"scaling": {**CONFIG["scaling"], "min_cpu_cores": 2, "max_cpu_cores": 4}}
        low = pd.Series({"cpu_usage_percent": 5.0, "memory_usage_mb": 500.0,
                         "current_cores": 1, "current_ram_mb": 1024})
        high = pd.Series({"cpu_usage_percent": 99.0, "memory_usage_mb": 500.0,
                          "current_cores": 6, "current_ram_mb": 1024})
        assert determine_scaling_action(low, 1, 90.0, config)[0] == "No Scaling"
        assert determine_scaling_action(high, 1, 90.0, config)[0] == "No Scaling"
