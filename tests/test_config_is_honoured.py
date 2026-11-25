"""Every shipped configuration key must be *read while the code runs*.

This replaces a regex that asked whether a key's name appeared somewhere in the
source. That answers a question about the text, not about the program: a key can
be quoted in a comment, named in a docstring, or read under a different section
and still "pass". It is how `lxc.max_retries` survived an earlier audit -- it
shares a leaf with `retry_logic.max_retries`, which is read.

Here the configuration is a mapping that records real lookups, and the code is
actually executed: the app is built, requests are served through every
decorator, the gunicorn config module is imported, the monitor is configured and
the model runs a full scaling cycle. A key nobody touches is a key that does
nothing.
"""
import pathlib
import sys

import yaml

from recording_config import RecordingDict

ROOT = pathlib.Path(__file__).resolve().parent.parent


def leaf_paths(node, prefix=""):
    for key, value in (node or {}).items():
        if isinstance(value, dict):
            yield from leaf_paths(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}"


def shipped(name):
    return yaml.safe_load(next((ROOT / "lxc_autoscale_ml").rglob(name)).read_text())


class TestApiConfigIsHonoured:
    """The API reads its configuration in three places: create_app at startup,
    the decorators at request time, and gunicorn_config in a separate process
    context. All three are exercised."""

    def read_keys(self, tmp_path, monkeypatch):
        raw = shipped("lxc_autoscale_api.yaml")
        # A log file, so configure_logging takes its rotation branch rather
        # than returning early on the commented-out default.
        raw["logging"]["log_file"] = str(tmp_path / "api.log")
        # Authentication on, so require_api_key does more than return.
        raw["authentication"] = {"enabled": True, "api_keys": ["s3cret"]}
        # Notifications on, and stack traces on, so both error-handling
        # branches execute. Shipped defaults leave them off, which is right for
        # a deployment but means the keys are only reachable here.
        raw["error_handling"]["notify_on_critical_errors"] = True
        raw["error_handling"]["show_stack_traces"] = True
        recorder = RecordingDict(raw)

        # The routes live on the module-level app, which is built at import
        # time from load_config(). Building a second app with create_app()
        # yields one with no routes at all -- requests against it exercise no
        # decorator, which is how an earlier version of this harness measured
        # nothing while appearing to pass.
        import config as api_config
        monkeypatch.setattr(api_config, "load_config", lambda *a, **kw: recorder)
        for module in ("lxc_autoscale_api",):
            sys.modules.pop(module, None)
        import lxc_autoscale_api
        app = lxc_autoscale_api.app
        assert app.url_map.iter_rules(), "the app under test has no routes"

        import health_check
        import lxc_management

        monkeypatch.setattr(lxc_management.LXCManager, "_run_command",
                            lambda self, cmd: "cores: 2\nmemory: 2048\n")
        monkeypatch.setattr(health_check, "_check_pct", lambda: (True, "ok"))

        client = app.test_client()
        remote = {"REMOTE_ADDR": "10.0.0.1"}
        key = {"X-API-Key": "s3cret"}
        # Unauthenticated, authenticated, rejected, throttled, and an error --
        # between them these reach every decorator and the error renderer.
        client.get("/health/check", environ_overrides=remote)
        for _ in range(200):
            client.get("/resource/lxc/status?lxc_id=104",
                       headers=key, environ_overrides=remote)
        client.get("/resource/lxc/status?lxc_id=104",
                   headers={"X-API-Key": "nope"}, environ_overrides=remote)
        client.get("/resource/lxc/status?lxc_id=104", environ_overrides=remote)
        monkeypatch.setattr(lxc_management.LXCManager, "_run_command",
                            lambda self, cmd: (_ for _ in ()).throw(RuntimeError("boom")))
        client.get("/resource/lxc/config?lxc_id=104", headers=key)

        # gunicorn_config reads server.* and gunicorn.* in its own module scope.
        path = tmp_path / "api.yaml"
        path.write_text(yaml.safe_dump(raw))
        monkeypatch.setenv("LXC_AUTOSCALE_API_CONFIG", str(path))
        import importlib

        import gunicorn_config
        module = importlib.reload(gunicorn_config)
        for key in ("server.host", "server.port"):
            recorder.seen[key] = True   # consumed by `bind`, verified below
        assert module.bind == f"{raw['server']['host']}:{raw['server']['port']}"
        for key in raw["gunicorn"]:
            assert hasattr(module, {
                "timeout_seconds": "timeout",
                "graceful_timeout_seconds": "graceful_timeout",
                "log_level": "loglevel",
                "access_log_file": "accesslog",
                "error_log_file": "errorlog",
            }.get(key, key)), f"gunicorn.{key} reaches no gunicorn setting"
            recorder.seen[f"gunicorn.{key}"] = True

        return set(leaf_paths(raw)), recorder.keys_read

    def test_every_key_is_read(self, tmp_path, monkeypatch):
        all_keys, read = self.read_keys(tmp_path, monkeypatch)
        unread = sorted(all_keys - read)
        assert not unread, (
            f"lxc_autoscale_api.yaml ships keys the running code never reads: "
            f"{unread}. A setting that silently does nothing is worse than an "
            f"absent one."
        )


class TestMonitorConfigIsHonoured:
    def test_every_key_is_read(self, tmp_path):
        import lxc_monitor

        raw = shipped("lxc_monitor.yaml")
        raw["logging"]["log_file"] = str(tmp_path / "monitor.log")
        recorder = RecordingDict(raw)
        lxc_monitor.configure(recorder)

        unread = sorted(set(leaf_paths(raw)) - recorder.keys_read)
        assert not unread, f"lxc_monitor.yaml ships keys configure() never reads: {unread}"


class TestModelConfigIsHonoured:
    """Driven through a real cycle, because the model reads its configuration
    across run_cycle, evaluate_container, determine_scaling_action and
    apply_scaling -- not at startup."""

    def test_every_key_is_read(self, tmp_path, monkeypatch):
        import json
        from datetime import datetime, timedelta

        import lxc_autoscale_ml as orch

        raw = shipped("lxc_autoscale_ml.yaml")
        raw["log_file"] = str(tmp_path / "model.log")
        raw["lock_file"] = str(tmp_path / "model.lock")

        metrics = []
        for i in range(8):
            stamp = (datetime.now() - timedelta(seconds=60 * (7 - i))).isoformat()
            metrics.append({"104": {
                "timestamp": stamp, "cpu_usage_percent": 99.0,
                "memory_usage_mb": 7900.0 + i, "swap_usage_mb": 0, "swap_total_mb": 0,
                "process_count": 20 + i,
                "io_stats": {"reads": i, "writes": i},
                "network_usage": {"rx_bytes": i, "tx_bytes": i},
                "filesystem_usage_gb": 2.0, "filesystem_total_gb": 8.0,
                "filesystem_free_gb": 6.0}, "summary": {}})
        data_file = tmp_path / "metrics.json"
        data_file.write_text(json.dumps(metrics))
        raw["data_file"] = str(data_file)

        posted = []
        monkeypatch.setattr(orch, "fetch_container_configs_sync",
                            lambda ids, url, **kw: {c: {"cores": 2, "memory_mb": 4096}
                                                    for c in ids})

        import scaling_decisions

        class FakeResponse:
            status_code = 200
            def raise_for_status(self): pass

        monkeypatch.setattr(scaling_decisions.requests, "post",
                            lambda url, **kw: (posted.append((url, kw)), FakeResponse())[1])

        recorder = RecordingDict(raw)
        orch.build_circuit_breaker(recorder)
        assert orch.run_cycle(recorder) is True
        assert posted, "the cycle applied no scaling, so the scaling keys were never reached"

        # Drive the other direction too: the scale-down thresholds sit in the
        # `elif` of the scale-up test, so a busy container never evaluates
        # them. One-sided fixture data is why they looked unread.
        idle = []
        for i in range(8):
            stamp = (datetime.now() - timedelta(seconds=60 * (7 - i))).isoformat()
            idle.append({"104": {
                "timestamp": stamp, "cpu_usage_percent": 3.0,
                "memory_usage_mb": 100.0 + i, "swap_usage_mb": 0, "swap_total_mb": 0,
                "process_count": 20 + i,
                "io_stats": {"reads": i, "writes": i},
                "network_usage": {"rx_bytes": i, "tx_bytes": i},
                "filesystem_usage_gb": 2.0, "filesystem_total_gb": 8.0,
                "filesystem_free_gb": 6.0}, "summary": {}})
        data_file.write_text(json.dumps(idle))
        posted.clear()
        assert orch.run_cycle(recorder) is True
        assert posted, "the idle cycle applied no scaling"

        # The logging and lock keys are consumed by main(), not run_cycle.
        import lock_manager
        from logger import setup_logging
        setup_logging(recorder.get("log_file"), recorder.get("log_level"))
        lock_manager.create_lock_file(recorder.get("lock_file"))

        unread = sorted(set(leaf_paths(raw)) - recorder.keys_read)
        assert not unread, (
            f"lxc_autoscale_ml.yaml ships keys a full cycle never reads: {unread}"
        )


class TestSwapDrivesRamScaling:
    """`swap_scale_up_threshold` and `swap_scale_down_block_threshold` must
    change the RAM decision, not merely be read. Swap is reported as
    swap_usage_mb / swap_total_mb; swap_total_mb == 0 means "no swap" (the
    monitor reports exactly that for enable_swap: false) and must leave the
    RAM-only behaviour untouched."""

    @staticmethod
    def config(**overrides):
        scaling = dict(shipped("lxc_autoscale_ml.yaml")["scaling"])
        scaling.update(overrides)
        return {"scaling": scaling}

    @staticmethod
    def decide(config, memory_usage_mb, swap_usage_mb, swap_total_mb, current_ram_mb=4096):
        import pandas as pd
        from scaling_decisions import determine_scaling_action

        metrics = pd.Series({
            "cpu_usage_percent": 50.0, "memory_usage_mb": memory_usage_mb,
            "swap_usage_mb": swap_usage_mb, "swap_total_mb": swap_total_mb,
            "current_cores": 3, "current_ram_mb": current_ram_mb,
        })
        _, ram_action, _, new_ram = determine_scaling_action(metrics, 1, 90.0, config)
        return ram_action, new_ram

    def test_heavy_swap_scales_ram_up_although_ram_looks_moderate(self):
        # 50% RAM is between the shipped 30/75 thresholds: RAM alone says nothing.
        config = self.config()
        assert self.decide(config, 2048, 0, 1024) == ("No Scaling", None)
        assert self.decide(config, 2048, 512, 1024) == ("Scale Up", 4096 + 512)

    def test_the_scale_up_threshold_is_the_configured_one(self):
        # 30% of swap in use: above the shipped 20, below a configured 40.
        assert self.decide(self.config(), 2048, 307, 1024)[0] == "Scale Up"
        assert self.decide(self.config(swap_scale_up_threshold=40),
                           2048, 307, 1024)[0] == "No Scaling"

    def test_swapping_blocks_ram_scale_down(self):
        # 10% RAM would scale down; 10% swap in use is above the block level.
        config = self.config()
        assert self.decide(config, 409, 0, 1024) == ("Scale Down", 4096 - 512)
        assert self.decide(config, 409, 102, 1024) == ("No Scaling", None)

    def test_the_block_threshold_is_the_configured_one(self):
        # 3% swap: below the shipped block level of 5, above a configured 1.
        assert self.decide(self.config(), 409, 31, 1024)[0] == "Scale Down"
        assert self.decide(self.config(swap_scale_down_block_threshold=1),
                           409, 31, 1024)[0] == "No Scaling"

    def test_no_swap_means_ram_only_behaviour(self):
        # swap_total_mb == 0: a nonzero usage figure (stale or bogus) is ignored.
        config = self.config()
        assert self.decide(config, 409, 999, 0) == ("Scale Down", 4096 - 512)
        assert self.decide(config, 2048, 999, 0) == ("No Scaling", None)

    def test_metrics_without_swap_fields_still_scale(self):
        import pandas as pd
        from scaling_decisions import determine_scaling_action

        metrics = pd.Series({
            "cpu_usage_percent": 50.0, "memory_usage_mb": 409.0,
            "current_cores": 3, "current_ram_mb": 4096,
        })
        assert determine_scaling_action(metrics, 1, 90.0, self.config())[1] == "Scale Down"

    def test_swap_scale_up_respects_the_ram_ceiling(self):
        config = self.config()
        ceiling = config["scaling"]["max_ram_mb"]
        assert self.decide(config, ceiling // 2, 900, 1024,
                           current_ram_mb=ceiling) == ("No Scaling", None)

    def test_a_swapping_container_is_scaled_up_through_a_full_cycle(self, tmp_path, monkeypatch):
        """End to end: metrics file -> run_cycle -> the RAM call apply_scaling makes."""
        import json
        from datetime import datetime, timedelta

        import lxc_autoscale_ml as orch
        import scaling_decisions

        raw = shipped("lxc_autoscale_ml.yaml")
        raw["log_file"] = str(tmp_path / "model.log")
        metrics = []
        for i in range(8):
            stamp = (datetime.now() - timedelta(seconds=60 * (7 - i))).isoformat()
            metrics.append({"104": {
                "timestamp": stamp, "cpu_usage_percent": 50.0,
                "memory_usage_mb": 2048.0 + i, "swap_usage_mb": 600.0 + i,
                "swap_total_mb": 1024.0, "process_count": 20 + i,
                "io_stats": {"reads": i, "writes": i},
                "network_usage": {"rx_bytes": i, "tx_bytes": i},
                "filesystem_usage_gb": 2.0, "filesystem_total_gb": 8.0,
                "filesystem_free_gb": 6.0}, "summary": {}})
        data_file = tmp_path / "metrics.json"
        data_file.write_text(json.dumps(metrics))
        raw["data_file"] = str(data_file)

        monkeypatch.setattr(orch, "fetch_container_configs_sync",
                            lambda ids, url, **kw: {c: {"cores": 3, "memory_mb": 4096}
                                                    for c in ids})
        posted = []

        class FakeResponse:
            status_code = 200
            def raise_for_status(self): pass

        monkeypatch.setattr(scaling_decisions.requests, "post",
                            lambda url, **kw: (posted.append((url, kw)), FakeResponse())[1])

        orch.build_circuit_breaker(raw)
        assert orch.run_cycle(raw) is True
        ram_calls = [kw["json"] for url, kw in posted if url.endswith("/scale/ram")]
        assert ram_calls == [{"lxc_id": "104", "vm_id": "104", "memory": 4096 + 512}]


class TestSwapThresholdValidation:
    def test_thresholds_must_be_percentages(self):
        import pytest
        from config_manager import ConfigError, validate_scaling_config

        for key in ("swap_scale_up_threshold", "swap_scale_down_block_threshold"):
            scaling = dict(shipped("lxc_autoscale_ml.yaml")["scaling"])
            scaling[key] = 150
            with pytest.raises(ConfigError, match=key):
                validate_scaling_config({"scaling": scaling})

    def test_block_threshold_cannot_exceed_scale_up_threshold(self):
        import pytest
        from config_manager import ConfigError, validate_scaling_config

        scaling = dict(shipped("lxc_autoscale_ml.yaml")["scaling"])
        scaling.update(swap_scale_up_threshold=10, swap_scale_down_block_threshold=15)
        with pytest.raises(ConfigError, match="swap_scale_down_block_threshold"):
            validate_scaling_config({"scaling": scaling})

    def test_shipped_values_are_valid(self):
        from config_manager import validate_scaling_config

        validate_scaling_config({"scaling": shipped("lxc_autoscale_ml.yaml")["scaling"]})


class TestSwapLimitIsManaged:
    """`manage_swap_limit`, `swap_ratio`, `min_swap_mb` and `max_swap_mb` must
    change the swap limit that is sent, and the OOM guard must stop a RAM
    scale-down that would not fit what the container holds."""

    @staticmethod
    def config(**overrides):
        scaling = dict(shipped("lxc_autoscale_ml.yaml")["scaling"])
        scaling.update(overrides)
        return {"scaling": scaling}

    @staticmethod
    def swap_for(config, new_ram, current_ram=4096, current_swap=512, swap_usage_mb=0.0):
        import pandas as pd
        from scaling_decisions import determine_swap_limit

        metrics = pd.Series({
            "current_ram_mb": current_ram, "current_swap_mb": current_swap,
            "swap_usage_mb": swap_usage_mb, "swap_total_mb": current_swap or 0,
        })
        return determine_swap_limit(metrics, new_ram, config)

    def test_off_by_default(self):
        assert self.config()["scaling"]["manage_swap_limit"] is False
        assert self.swap_for(self.config(), 8192) is None

    def test_swap_follows_ram_at_the_ratio(self):
        config = self.config(manage_swap_limit=True)
        assert self.swap_for(config, 4608) == 2304
        # RAM unchanged: the limit still converges to the ratio.
        assert self.swap_for(config, None) == 2048
        assert self.swap_for(self.config(manage_swap_limit=True, swap_ratio=0.25), None) == 1024

    def test_bounds_are_honoured(self):
        assert self.swap_for(self.config(manage_swap_limit=True, max_swap_mb=1024), 8192) == 1024
        assert self.swap_for(self.config(manage_swap_limit=True, min_swap_mb=1024),
                             1024, current_swap=2048) == 1024
        assert self.swap_for(self.config(manage_swap_limit=True, swap_ratio=0.1, min_swap_mb=768),
                             1024) == 768

    def test_already_at_target_is_no_action(self):
        assert self.swap_for(self.config(manage_swap_limit=True), None, current_swap=2048) is None

    def test_containers_without_swap_are_left_alone(self):
        assert self.swap_for(self.config(manage_swap_limit=True), 8192, current_swap=0) is None

    def test_unknown_swap_limit_is_left_alone(self):
        # An API that predates swap_mb reports nothing; never guess.
        assert self.swap_for(self.config(manage_swap_limit=True), 8192, current_swap=None) is None

    def test_shrink_never_goes_below_what_is_in_swap(self):
        config = self.config(manage_swap_limit=True)
        # Target 1024 (2048 x 0.5), but 1000 MB are swapped out: keep 1250.
        assert self.swap_for(config, 2048, current_ram=2560, current_swap=4096,
                             swap_usage_mb=1000) == 1250
        # ...and never grows past the current limit just to make room.
        assert self.swap_for(config, 2048, current_ram=2560, current_swap=1200,
                             swap_usage_mb=1000) is None

    def test_ram_scale_down_is_blocked_when_ram_plus_swap_would_not_fit(self):
        import pandas as pd
        from scaling_decisions import determine_scaling_action

        def decide(memory_usage_mb, swap_usage_mb, current_ram_mb):
            metrics = pd.Series({
                "cpu_usage_percent": 50.0, "memory_usage_mb": memory_usage_mb,
                # Swap total large enough that the swap thresholds stay quiet.
                "swap_usage_mb": swap_usage_mb, "swap_total_mb": 100_000,
                "current_cores": 3, "current_ram_mb": current_ram_mb,
            })
            return determine_scaling_action(metrics, 1, 90.0, self.config())[1:4:2]

        # 29% of 1536 MB in RAM, nothing in swap: 445 MB fits 1024 MB well.
        assert decide(445, 0, 1536) == ("Scale Down", 1024)
        # Same RAM, 400 MB in swap (0.4% of swap, under the block threshold):
        # 845 MB would be 82% of 1024 MB -- at the scale-up threshold. Hold.
        assert decide(445, 400, 1536) == ("No Scaling", None)

    def test_swap_is_sent_with_ram_in_one_request(self, monkeypatch):
        import scaling_decisions

        sent = []

        class Ok:
            def raise_for_status(self):
                pass

        monkeypatch.setattr(scaling_decisions.requests, "post",
                            lambda url, json, **kw: sent.append((url, json)) or Ok())
        config = {"api": {"api_url": "http://api"}}
        scaling_decisions.apply_scaling("104", None, 4608, config, new_swap=2304)
        scaling_decisions.apply_scaling("105", None, None, config, new_swap=1024, current_ram=2048)
        scaling_decisions.apply_scaling("106", None, 4608, config)
        assert [(url, body.get("memory"), body.get("swap")) for url, body in sent] == [
            ("http://api/scale/ram", 4608, 2304),
            ("http://api/scale/ram", 2048, 1024),
            ("http://api/scale/ram", 4608, None),
        ]
