"""Scaling decisions must settle under a steady workload.

Three faults found on a real host were the same kind of fault: the decision
never came to rest or turned around.

- Swap that stayed occupied after a scale-up triggered another scale-up every
  cycle until max_ram_mb (+512 MB every 3 minutes up to 16 GB).
- A "Scale Down" of a container below min_ram_mb was clamped up to the floor,
  so a scale-down raised RAM.
- Container 106 went from 2 to 3 cores and back within one cycle, three times
  a day, on short CPU bursts.

Each test here closes the loop instead of checking one decision: it feeds the
decision's result back in as the next cycle's allocation, the way the
orchestrator and Proxmox do, and checks the whole run.
"""
import os

import pytest
import yaml

from scaling_decisions import determine_scaling_action

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_CONFIG = os.path.join(ROOT, "lxc_autoscale_ml", "model", "lxc_autoscale_ml.yaml")

CYCLES = 40
SETTLED_AFTER = 20


def shipped_config():
    with open(SHIPPED_CONFIG) as fh:
        return yaml.safe_load(fh)


def host_like_config():
    """The shipped thresholds with the floors of a small host (512 MB, 1 core)."""
    config = shipped_config()
    config["scaling"].update(min_ram_mb=512, min_cpu_cores=1, max_ram_mb=16384, max_cpu_cores=8)
    return config


def run(config, cores, ram_mb, cpu_demand, ram_used_mb, swap_used_mb=0, swap_total_mb=0,
        cycles=CYCLES):
    """Close the loop for `cycles` cycles; return the (cores, ram) history and actions.

    cpu_demand is in cores' worth of work, a list for a changing load or a
    number for a steady one; usage is relative to the allocation, as the
    monitor measures it inside the container. Swapped pages stay in swap: the
    kernel does not page them back in when RAM grows.
    """
    history = [(cores, ram_mb)]
    actions = []
    previous_cpu = None
    for cycle in range(cycles):
        demand = cpu_demand[cycle] if isinstance(cpu_demand, list) else cpu_demand
        cpu_usage = min(100.0, demand / cores * 100)
        metrics = {
            "cpu_usage_percent": cpu_usage,
            "previous_cpu_usage_percent": previous_cpu,
            "memory_usage_mb": min(ram_used_mb, ram_mb * 0.98),
            "swap_usage_mb": swap_used_mb,
            "swap_total_mb": swap_total_mb,
            "current_cores": cores,
            "current_ram_mb": ram_mb,
        }
        cpu_action, ram_action, new_cores, new_ram = determine_scaling_action(metrics, 1, 100.0, config)
        previous_cpu = cpu_usage
        actions.append((cpu_action, cores, new_cores, ram_action, ram_mb, new_ram))
        cores = new_cores if new_cores is not None else cores
        ram_mb = new_ram if new_ram is not None else ram_mb
        history.append((cores, ram_mb))
    return history, actions


def direction_changes(values):
    steps = [b - a for a, b in zip(values, values[1:], strict=False) if b != a]
    return sum(1 for a, b in zip(steps, steps[1:], strict=False) if (a > 0) != (b > 0))


def assert_stable(history, actions):
    cores = [c for c, _ in history]
    ram = [r for _, r in history]
    assert direction_changes(cores) == 0, f"CPU turned around: {cores}"
    assert direction_changes(ram) == 0, f"RAM turned around: {ram}"
    assert len(set(history[SETTLED_AFTER:])) == 1, f"never settled: {history}"
    for cpu_action, cores_before, new_cores, ram_action, ram_before, new_ram in actions:
        if new_cores is not None:
            assert (new_cores > cores_before) == (cpu_action == "Scale Up"), actions
        if new_ram is not None:
            assert (new_ram > ram_before) == (ram_action == "Scale Up"), actions


STEADY_WORKLOADS = [
    # (cores, ram_mb, cpu_demand, ram_used_mb, swap_used_mb, swap_total_mb)
    pytest.param(2, 2048, 0.2, 300, 0, 0, id="idle"),
    pytest.param(2, 2048, 1.9, 1900, 0, 512, id="busy-no-swap"),
    pytest.param(2, 1024, 0.5, 200, 380, 512, id="low-ram-high-swap"),
    pytest.param(1, 512, 0.1, 40, 210, 512, id="ct117-like"),
    pytest.param(2, 1024, 0.3, 200, 380, 512, id="ct104-like"),
    pytest.param(3, 2560, 1.0, 825, 820, 4096, id="ct113-like"),
    pytest.param(2, 1536, 0.6, 680, 195, 512, id="ct122-like"),
    pytest.param(1, 512, 0.9, 480, 500, 512, id="ram-and-swap-full"),
    pytest.param(4, 8192, 3.9, 8000, 4000, 4096, id="at-the-ceiling"),
    pytest.param(1, 256, 0.05, 30, 0, 0, id="below-the-floor"),
]


@pytest.mark.parametrize("make_config", [shipped_config, host_like_config], ids=["shipped", "host-like"])
@pytest.mark.parametrize("cores,ram_mb,cpu_demand,ram_used_mb,swap_used_mb,swap_total_mb", STEADY_WORKLOADS)
def test_a_steady_workload_settles_without_turning_around(
        make_config, cores, ram_mb, cpu_demand, ram_used_mb, swap_used_mb, swap_total_mb):
    history, actions = run(make_config(), cores, ram_mb, cpu_demand, ram_used_mb,
                           swap_used_mb, swap_total_mb)
    assert_stable(history, actions)


@pytest.mark.parametrize("cpu_demand", [x / 10 for x in range(1, 80, 3)])
@pytest.mark.parametrize("ram_used_mb", [100, 400, 900, 1500, 3000, 6000])
@pytest.mark.parametrize("swap_used_mb", [0, 150, 450])
def test_no_steady_workload_oscillates(cpu_demand, ram_used_mb, swap_used_mb):
    history, actions = run(host_like_config(), 2, 1024, cpu_demand, ram_used_mb,
                           swap_used_mb, 512 if swap_used_mb else 0)
    assert_stable(history, actions)


def test_a_short_cpu_burst_does_not_add_and_remove_a_core_in_back_to_back_cycles():
    """Container 106: one cycle of 2 cores' worth of work on 2 cores, then idle.

    The burst adds a core. Taking it away again the very next cycle, on the
    first quiet sample, is the flap seen three times a day on the host.
    """
    config = shipped_config()  # min_cpu_cores 2, as on the host: 106 never drops below 2
    demand = [0.4] * 5 + [2.0] + [0.4] * 34
    history, _ = run(config, 2, 2048, demand, 600)
    cores = [c for c, _ in history]
    assert direction_changes(cores) == 0, f"the burst added a core and took it away again: {cores}"


def test_sustained_cpu_load_still_adds_a_core():
    demand = [0.4] * 5 + [2.0] * 35
    history, _ = run(shipped_config(), 2, 2048, demand, 600)
    cores = [c for c, _ in history]
    assert cores[-1] > 2, cores
    assert cores.index(3) == 7, f"a core should come on the second busy sample: {cores}"
