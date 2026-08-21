# Performance benchmarking

## Routine gate (use this day-to-day)

`scripts/bench_perf_gate.py` is the **canonical long-term check**:

* Synthetic scene (no G1 MJCF)
* Fixed **192×144**, **N=256**, `quality="fast"` (product default)
* Paths: `lambert`, `pbr_gbuffer`, `pbr_fast`
* Versioned JSON (`schema_version`) + optional `--compare` regression exit

```bash
# Measure
.venv/bin/python scripts/bench_perf_gate.py --json-out bench/results/latest.json

# Save a baseline on a known box (once per GPU model / driver stack)
cp bench/results/latest.json bench/baselines/rtx4090.json

# Later: fail CI/local if >15% slower (also checks pbr_fast vs_lambert ratio)
.venv/bin/python scripts/bench_perf_gate.py \
  --json-out bench/results/latest.json \
  --compare bench/baselines/rtx4090.json
```

`--quick` lowers warmup/iters for agent loops; **do not** refresh baselines from `--quick` runs.

## Deep-dive benches (not the daily gate)

| Script | When |
| --- | --- |
| `bench_pbr_camera.py` | G1 scene, N-sweep, pretty / smap / nosray |
| `bench_tiled_filters.py` | FXAA/SSAO torch vs tiled vs compile |
| `probe_pbr_*.py` | One-off optimization probes |

Keep deep dives for design notes (`_pbr_camera.md`); keep the gate green for merges that touch `pbr/` or raycast kernels.

## Automation recommendations

1. **Self-hosted GPU runner (best)** — on PRs that touch `src/simple_raycaster/pbr/**` or `kernels.py`, run the gate with `--compare` against a committed baseline for that machine class. Absolute ms are not portable across GPU models; keep one baseline file per class (`rtx4090.json`, …).
2. **Prefer relative signals** — the gate also fails if `pbr_fast` **vs_lambert** regresses beyond `--fail-pct`. That is stabler than raw ms when clocks/power fluctuate.
3. **Do not put absolute ms thresholds on CPU-only cloud CI** — skip or `allow_failure` without CUDA. Smoke (`smoke_pbr_camera.py`) stays the correctness check everywhere CUDA exists.
4. **Nightly (optional)** — full `bench_pbr_camera.py --pretty` + `bench_tiled_filters.py` on a fixed box; store JSON under `bench/results/` (gitignored) or an artifact store; chart trends, don’t fail the day-to-day gate on probe noise.
5. **Baseline hygiene** — refresh baselines only after intentional perf work or stack upgrades (Warp/Torch/driver), on an idle machine, without `--quick`. Record `git_sha`, `warp`, `torch`, `gpu` from the JSON in the PR.

`bench/results/` is gitignored. Commit baselines under `bench/baselines/` when you have a stable reference machine.
