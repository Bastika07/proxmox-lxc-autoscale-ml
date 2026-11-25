# 2. Swap limit management is off by default; the OOM guard is always on

Date: 2026-11-25. Status: accepted (PR #4).

## Decision

- `scaling.manage_swap_limit` (default `false`) sets the swap limit to `swap_ratio` x RAM in the same `pct set -memory -swap`. Containers with swap 0 keep it; an API that does not report `swap_mb` leaves swap alone.
- A RAM scale-down is refused when RAM used plus swap used would reach `ram_scale_up_threshold` of the smaller limit, whether or not the swap limit is managed. Lowering limits below what is held sends the kernel to the OOM killer.

## Why off by default

Changing swap limits on a running host is new behaviour; it needs an updated API and an explicit opt-in.
