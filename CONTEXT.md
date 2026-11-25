# Context

Glossary for the scaling model. Decisions behind these terms are in `docs/adr/`.

- **Allocation**: a container's limits in Proxmox (`cores`, `memory`, `swap`), read from the API every cycle. Scaling targets are absolute values derived from it.
- **Cycle**: one model pass (load metrics, train, decide, apply). The monitor writes one **snapshot** per collection; a cycle decides on the newest snapshot.
- **Swap pressure**: swap in use while RAM could not hold it, i.e. RAM used plus swap used reaches `ram_scale_up_threshold` of the RAM limit. Swap in use alone is not pressure: swapped pages stay in swap after RAM is added. See ADR 0001.
- **OOM guard**: RAM is never scaled down below what RAM used plus swap used needs to stay under `ram_scale_up_threshold`. Always on. See ADR 0002.
- **Burst**: a single CPU sample above `cpu_scale_up_threshold`. It adds no core; two samples in a row do. See ADR 0003.
- **Flap**: a change undone again within a few cycles (2 -> 3 -> 2 cores). Tested by `tests/test_scaling_stability.py` and `tests/test_host_metrics_replay.py`.
- **Runaway**: a scale-up that triggers the next one every cycle until the ceiling.
- **Dry run**: `scaling.dry_run: true`; decisions are logged with their inputs, nothing is applied.
