"""`python -m telemetry.context_dist` — assembled-context size distribution → cost regime.

Closes the loop the cost report names: drop the *production* assembled-context
distribution onto the MEASURED cost curve and read the operating regime straight off
the decision table. Reads the gateway's hash-chained audit JSONL, each record of
which now carries `included_context_tokens` (the privacy-filtered context the
assembler produced — counts only, never content), computes p50/p90/p95/p99, and maps
each percentile to its batch-width regime + pooled cost (rendered from
`telemetry/cost.py`, not hand-typed).

    python -m telemetry.context_dist --audit state/gateway_audit.jsonl

Records written before this observability field existed simply lack it and are
skipped; the tool reports how many it found.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from telemetry.cost import PooledDiurnalFleetCostModel

# Decision-table regimes — match docs/REPORT_cost_model_v2.md. Contiguous, no gap.
# Bounds trace to the measured curve (MEASURED_BATCH_BY_CONTEXT): 1,937 tok → batch 2,
# 3,367 tok → batch 1, so 1k–3k is the batch-2 caution band and >3k is batch-1.


@dataclass(frozen=True)
class Regime:
    upper_exclusive_tokens: float
    batch: int
    label: str


_REGIMES = (
    Regime(250, 16, "cheapest"),
    Regime(1_000, 4, "realistic interactive"),
    Regime(3_000, 2, "caution"),
    Regime(math.inf, 1, "avoid full-context unless necessary"),
)

# Cost basis for the rendered $/day column (the curve scales ~linearly with density).
_WARM_SERVICE_SEC = 0.93   # measured Modal L4 warm latency
_FAMILIES = 1_000_000
_DENSITY = 4               # req/family/day
_DIURNAL_PEAK = 3.0


def regime_for(tokens: float) -> Regime:
    for regime in _REGIMES:
        if tokens < regime.upper_exclusive_tokens:
            return regime
    return _REGIMES[-1]


def percentile(sorted_vals: list[int], q: float) -> int:
    """Nearest-rank percentile (rank = ceil(q/100 * N)) over sorted token counts."""
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(q / 100.0 * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def _pooled_cost_band(batch: int) -> tuple[float, float]:
    result = PooledDiurnalFleetCostModel(
        _FAMILIES, _DENSITY, _DIURNAL_PEAK, _WARM_SERVICE_SEC, batch, True
    ).project()
    return result.fleet_cost_per_day_usd_low, result.fleet_cost_per_day_usd_high


def read_context_tokens(audit_path: str | Path) -> list[int]:
    """Extract `included_context_tokens` from every audit record that carries it (counts only)."""
    path = Path(audit_path)
    if not path.exists():
        return []
    tokens: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        response = json.loads(line).get("response", {})
        if "included_context_tokens" in response:
            tokens.append(int(response["included_context_tokens"]))
    return tokens


def _print_distribution(tokens: list[int], source: str) -> None:
    ordered = sorted(tokens)
    print(f"assembled-context size distribution  (n={len(tokens)}, source={source})")
    print(f"  min={ordered[0]}  max={ordered[-1]}  mean={sum(tokens) / len(tokens):.0f} tokens\n")
    print(f"  {'pctile':7} {'tokens':>7}  {'batch':>5}  {'regime':36} pooled $/day (1M fam, {_DENSITY} req/day)")
    for q in (50, 90, 95, 99):
        value = percentile(ordered, q)
        regime = regime_for(value)
        low, high = _pooled_cost_band(regime.batch)
        print(f"  p{q:<6} {value:>7}  {regime.batch:>5}  {regime.label:36} ${low:,.0f}–{high:,.0f}")
    print("\n  cost basis: warm 0.93 s, scales ~linearly with req/family/day. Regime bounds trace to")
    print("  the measured curve (telemetry/cost.py MEASURED_BATCH_BY_CONTEXT); counts only, no content.")


def run(args: argparse.Namespace) -> int:
    tokens = read_context_tokens(args.audit)
    if not tokens:
        print(f"No records with included_context_tokens in {args.audit}.")
        print("(Audit records written before the observability field existed do not carry it.)")
        return 1
    _print_distribution(tokens, args.audit)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assembled-context size distribution -> cost regime (drop production traffic onto the curve)"
    )
    parser.add_argument("--audit", default="state/gateway_audit.jsonl",
                        help="gateway audit JSONL (records carry included_context_tokens)")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
