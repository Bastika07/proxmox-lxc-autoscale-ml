# 3. CPU scales up on two busy samples in a row

Date: 2026-11-25. Status: accepted.

## Context

Container 106 runs a job for about a minute every half hour. Its one busy sample (up to 91 %) crossed `cpu_scale_up_threshold`, the model added a core, and the next quiet sample took it away: 2 -> 3 -> 2 cores within one cycle, three times on 2026-11-25. The monitor samples about every 74 s.

## Decision

CPU scales up only when the current sample and the one before it are both above `cpu_scale_up_threshold`. The orchestrator passes the previous sample as `previous_cpu_usage_percent`; without it a single sample decides, as before. Sustained load gets its core one sample (about 75 s) later.

## Rejected

A scale-down cooldown: still adds and removes a core for every burst, just slower, and needs state across cycles.
