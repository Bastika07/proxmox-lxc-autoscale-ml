# 1. Swap scales RAM up only under swap pressure

Date: 2026-11-25. Status: accepted (PR #5).

## Context

Swap usage above `swap_scale_up_threshold` used to scale RAM up on its own. The kernel does not page swapped memory back in when RAM grows, so swap usage stayed high and RAM grew by `ram_scale_step_mb` every cycle up to `max_ram_mb` (seen on a real host: +512 MB every 3 minutes towards 16 GB).

## Decision

Swap drives a RAM scale-up only while RAM used plus swap used reaches `ram_scale_up_threshold` of the current RAM limit. This is the same test as the OOM guard (ADR 0002), so up and down cannot oscillate, and each step lowers the ratio, which ends the series. No new config key, no cooldown state.

## Rejected

"Swap above 50 % and growing" (closed PR #6): a second rule for the same problem, with a key name already used by this one. A cooldown: needs state across cycles and only slows the runaway.
