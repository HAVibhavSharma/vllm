# HBM Timeline — Visualisation Spec

**What this is:** how to turn the `kv_hbm` log lines from
[`hbm-logging-changes.md`](hbm-logging-changes.md) into a single self-contained
HTML page that shows both runs on one timeline.

**Who this is for:** whoever builds the page — human or agent. It is a spec,
not a description of something that exists. Nothing here is built yet.

---

## 1. What the page has to answer

In priority order. A panel that does not serve one of these does not belong on
the page.

1. **How much KV did each run have to rebuild?** `remat_mb` over time, both
   runs. This is the claim; everything else is supporting evidence. See
   §2 of [`hbm-logging-changes.md`](hbm-logging-changes.md) for why this and
   not the miss count.
2. **Did the cache work at all?** Hit rate, both runs. Necessary context — a
   movement reduction bought by serving fewer hits is not a win.
3. **Did the policy change what HBM held?** Occupancy over time.
4. **Did it actually do anything?** Splice volume. A flat `spliced_blocks=0`
   on the policy run means the tick never reached the splice — the silent
   failure the line was added to catch.
5. **Did it evict less, or just differently?** Eviction rate, both runs.
6. **Whose blocks went?** `job:node` attribution over time. This is the panel
   that turns "the hit rate got worse" into "the forecast for `research` is
   wrong".
7. **Was it right?** Regret over time, policy run only.

---

## 2. Input

Two files, produced by `grep 'kv_hbm ' <log>`:

```
baseline.kv_hbm
policy.kv_hbm
```

The page holds up to **three** at once, in slots A, B and C — not in fixed
arms. Any set of `variant=` values is a legal comparison, and each column is
labelled with the variant its own lines name. Keying the runs by variant
instead of by slot is what made a `variant=continuum` log parse cleanly and
then render nothing at all.

Three rather than two because that is the comparison that actually gets run:
`baseline` vs `node_eviction` vs `continuum`. Splitting it across two two-way
pages means eyeballing a scroll between them, which §6 already rules out for
two columns and is worse for three.

**Slot A is the reference.** Every Δ in the summary is measured against it,
never between the non-reference arms. Pairwise deltas across three arms is six
delta columns with no reading order; against a fixed reference it is two, and
"which arm won" is answerable by scanning one column. Slots fill in drop
order, except that a run naming `variant=baseline` claims A when A is free.

A is drawn dashed, B solid, C dotted — separable in a greyscale screenshot and
without relying on hue. The policy-only panels (splice volume, regret, scored
coverage) still render for `node_eviction` alone: every other variant emits a
constant zero there, and a flat zero reads as "no regret" rather than "not
measured".

Slots that hold nothing do not appear: no empty column, no solo button, no
delta of dashes.

Optionally, one harness report per slot (`armD_rep1.jsonl` /
`armE_rep1.jsonl` / …) for client-measured end-to-end latency. These carry no
variant field, so the slot comes from which input the file was dropped in —
never from the filename, because a mislabelled arm silently inverts the
result. The paired E2E figure is computed once per non-reference arm, each
against the reference on the keys those two have in common; the matched-key
count differs per pair and is printed on its own row.

**The report is the only input that is not warm-only by default, and the page
has to say so.** A run is two phases —
`cold phase -> POST /v1/kv_metrics/reset -> warm phase`
(`run_evaluate_node_eviction.py`) — and the reset is what makes every `kv_hbm`
field describe the warm phase alone. Nothing equivalent happens to the harness
report: `ODR_TRACE_REPORT` resolves once in `TraceTransport.__init__`, the
httpx client is a module-level singleton, so both phases append to one file
with no timestamp and no phase marker on the record.

Last-occurrence-wins already recovers most of it, because the cold phase runs
the same keys first and the warm record overwrites it. What leaks is a key the
cold phase ran and the warm phase never reached — the warm loop can stop on
`all_queries_satisfied()` and cancel in-flight instances — and that record is a
pure cold latency sitting in the total.

The default cut is **positional: the back half of each report**. Both phases
replay the same pinned trace, so a report that ran both holds every key twice
and the warm pass is the second half of the file — 24 records means the 12
warm calls. It needs nothing from the record beyond its order, so it works on
a report written before `job_id` existed, and it is a rule a reader can check
by eye against the line count.

It is an assumption, not a measurement, and the page says so on every load:
the halves line up only when both phases ran the same trace to completion. The
warm loop can stop on `all_queries_satisfied()` and cancel in-flight
instances, and then the cold phase is the longer of the two and the overhang
survives the cut. An odd record count is that case showing on the surface, so
the banner names the arms it applies to.

`job_id` is the exact discriminator when the halves are not trustworthy, and
it is on every record (`trace_store.py`). The harness hands job ids out from
one `itertools.count(1)` in start order and the cold phase completes before
the warm phase begins, so cold owns `1..N` and everything above is warm.
Typing a **first warm `job_id`** overrides the halves rule; the value applies
to all slots — per-arm cuts would let two different phases be compared to
each other. `N` is `instances_per_query x examples`, or read the lowest
`job_id` in the warm `job_instance_e2e_latency.csv`. Emptying the box falls
back to the halves rule rather than to no cut at all, so clearing an override
never quietly puts the cold phase back in the totals.

**Both phases** is the third setting, for looking at the warmup deliberately.

Filtering is a rebuild from the retained raw records, not a mutation, so
switching between the three settings is lossless in both directions and
**Both phases** restores the original figures exactly. Under a `job_id` cut,
records with no `job_id` are kept and counted separately rather than dropped,
so an older report format does not silently empty the block. The E2E block
states which phase it is showing and how many cold records the cut removed.

**Both phases does not collapse repeated keys, and that is the whole reason
the setting is legible.** A cut mode collapses first — within one phase a
repeated key is a re-appended run, and averaging those together is the mistake
the collapsing exists to prevent. Applying it under **Both phases** would keep
only the warm record of every key the warm phase also ran, so the page would
report the warm phase under a label saying both, and on a symmetric run the
two settings would print byte-identical figures. Observed exactly that on a
44-record report: 22 calls and the same mean in both settings, which reads as
a broken control. Uncollapsed, the same file gives 22 calls at 5.5 s warm
against 44 calls at 7.9 s across both phases, and the cold inflation is on
screen where it can be argued with.

`byKey` stays deduped in every mode regardless, because the paired comparison
needs exactly one record per key per arm.

A **Records used of file** row sits under the call count in every mode. Without
it a cut that halved the file and a cut that did nothing print the same call
count whenever the phases were symmetric, and there is no way to see from the
table that the setting took effect.

### Phase

A run is two phases —
`cold phase -> POST /v1/kv_metrics/reset -> warm phase` — and only the second
is being measured. `reset_measurement()` zeroes every counter, increments
`epoch` and restarts `epoch_age_s`; it does not remove the cold lines from the
log. So **the run to report on is the highest `epoch` present**, and the page
shows that alone by default.

This matters unevenly across the page, which is why it was easy to miss:

- The **summary table** was already warm-only for free. It reads the last line,
  and the counters restarted at the reset.
- The **panels** were not. Plotting every line draws the cold ramp and the warm
  series on one axis, with a discontinuity where the counters were zeroed that
  reads as a collapse in hit rate rather than a reset.

Toggling back to both phases is offered for looking at the warmup deliberately,
and banners say which phase is on screen. A log with no `epoch` field predates
the reset endpoint; the filter cannot apply and the page says so rather than
silently showing everything as if it were warm.

The **time axis is rebased per phase**: in the warm view `t=0` is the reset,
not server boot. Leaving the origin at boot would put the warm phase at
`t=40min` on an axis whose first 40 minutes are blank, and the two arms' resets
do not land at the same offset, so the arms would be misaligned against each
other as well as against zero.

Each line carries the log's own timestamp prefix followed by the `kv_hbm`
payload. The payload is flat `key=value` pairs separated by single spaces,
identical field order on both sides — see the field table in
[`hbm-logging-changes.md`](hbm-logging-changes.md) §2.

The page must accept the files by **drag-and-drop or file picker**, not a
fetch. A strict CSP blocks requests to any external host, and more practically
these logs live wherever the benchmark ran. Read them with `FileReader`.

---

## 3. Parsing

```js
// Timestamp prefix varies with vLLM's logger config, so anchor on `kv_hbm `
// and take everything before it as the timestamp blob.
const AT = line.indexOf("kv_hbm ");
const stamp = line.slice(0, AT);
const fields = Object.fromEntries(
  line.slice(AT).split(" ").filter(p => p.includes("="))
      .map(p => { const i = p.indexOf("="); return [p.slice(0,i), p.slice(i+1)]; })
);
```

Note `indexOf("=")` rather than `split("=")`: `top_evicted`'s value contains
`=` characters. Splitting on all of them silently truncates the attribution.

Numeric coercion: everything except `variant` and `top_evicted` is a number.
`usage` has a trailing `%`. Strip it and store the fraction. `hit_rate`,
`hit_rate_win`, `remat_ratio` and `regret` are already fractions — do not
divide them by 100.

`top_evicted` is `-` when empty, otherwise
`label=count,label=count,...` where `label` is `job_id:node` — and `job_id`
may itself contain a colon. Split each entry on the **last** `=`, then take
the label whole. Do not split the label on `:`.

### Time

Parse the timestamp prefix if you can; fall back to line index if you cannot.
Then **convert to elapsed seconds from the first line of that file**. The two
runs happened at different wall-clock times and comparing them on absolute
time produces two disjoint traces on one axis.

---

## 4. Data model

```
Run {
  slot: "A" | "B" | "C",      // A is the reference every Δ is taken against
  variant: string,            // "baseline" | "node_eviction" | "continuum" | …
  samples: Sample[],          // ordered by t
}

Sample {
  t: number,                  // seconds since that run's first line
  total, used, free: number,
  usage: number,              // fraction, 0..1
  queue: number,
  splices, splicedBlocks: number,      // cumulative
  evicted, evictedByScore: number,     // cumulative
  regret: number,                      // ratio, already windowed by the engine
  hitRate: number,                     // cumulative, 0..1
  hitRateWin: number,                  // THIS window only
  hitTokens, queryTokens: number,      // cumulative
  rematBlocks: number,                 // cumulative
  rematMb: number,                     // cumulative
  rematRatio: number,                  // cumulative
  blocksCached: number,                // cumulative
  indexKeys, indexBlocks: number,
  topEvicted: Map<label, count>,       // THIS window only
}
```

### Derived series

Cumulative fields need differencing before they mean anything on a chart. A
monotonically rising `evicted` line looks the same for a healthy run and a
thrashing one.

```
rate(field, i) = (s[i][field] - s[i-1][field]) / (s[i].t - s[i-1].t)
```

Emit rates in **blocks per second**, not per window — window lengths are not
uniform (§5).

`topEvicted`, `hitRateWin` and `regret` are already per window. Do **not**
difference them.

`hitRate`, `rematRatio` and friends are cumulative **ratios**, not counters.
Differencing a ratio is meaningless. To get a windowed `rematRatio`, difference
the two counters and divide:

```
rematRatioWin(i) = ΔrematBlocks(i) / Δblocks_cached(i)   // guard /0
```

The same applies to a windowed hit rate — except the engine already computes
it, so use `hitRateWin` and do not recompute it from
`ΔhitTokens / ΔqueryTokens`. They agree, but only the engine's version is
correct across a `reset_prefix_cache`.

---

## 5. The three traps

Get these wrong and the page is confidently misleading.

### 5.1 Samples are not evenly spaced

The line is change-gated: an idle server emits nothing. So gaps in the series
mean "nothing moved", not "no data". Three consequences:

- Interpolate **step-after**, never linear. A linear segment across a 10-minute
  idle gap draws a smooth ramp through occupancy that never happened.
- Divide by the actual `Δt` when computing rates, never by the nominal period.
- Draw the gaps. A faint marker or a dashed segment where `Δt` exceeds ~3×
  the nominal period, so a reader can tell silence from missing data.

### 5.2 The attribution stack does not sum to the total

`top_evicted` names the top *N* keys (default 5), so
`sum(topEvicted) ≤ Δevicted`. Stacking only the named keys silently
under-reports.

Add an explicit remainder band:

```
other = max(0, Δevicted - sum(topEvicted.values()))
```

Render `other` in a muted neutral, always at the top of the stack, always
labelled `other (untruncated)`. If it dominates, the reader needs to know to
raise `hbm_summary_top_keys` — not to conclude that five nodes did everything.

`<untracked>` is a real category, not a remainder — requests carrying no
`job_id`. Give it its own band, distinct from `other`.

### 5.3 `remat_mb` is zero when the page size was never configured

The engine reports `remat_mb=0.0` rather than guessing when it does not know
the KV page size. A page that plots that as "no movement" inverts the
conclusion.

Detect it: `rematBlocks > 0 && rematMb === 0` for a whole run means the byte
size was missing. Fall back to plotting `rematBlocks` and label the axis
"blocks", with a note on the panel. Never plot a zero line.

### 5.4 The two runs have different key sets

Colour assignment must be **stable across both runs**: build one label →
colour map from the union of labels in both files, sorted, before rendering
either. If `run-42:research` is teal on the left panel and amber on the right,
the comparison is unreadable.

---

## 6. Panels

Single page, vertically stacked, **one shared x-axis** with a synchronised
crosshair. Baseline and policy are compared *within* each panel, not in two
separate columns — the eye cannot align two columns across a scroll.

| # | Panel | Encoding |
|---|---|---|
| 1 | **KV rebuilt (cumulative)** | `rematMb`, two lines, MB. **The headline.** The gap between the two curves at the right edge *is* the result. Shade it and label the final delta in the panel. |
| 2 | **Rebuild rate** | `ΔrematBlocks / Δt`, blocks/s, two lines. Shows *when* the policy helped — a gap that opens only under pressure is a different story from a constant offset. |
| 3 | **Hit rate** | `hitRateWin`, two lines, 0–1. Read against panel 1: movement down with hit rate flat or up is the win. Movement down *and* hit rate down means it served less, not better. |
| 4 | **TTFT** | `ttft_win_ms`, two lines, ms. Engine-side: scheduler queueing plus prefill, excluding front-end queueing and detokenization. Read against panel 3 — hit rate says how often the cache worked, this says whether it mattered. A hit-rate gain with TTFT flat means the hits landed on blocks that were cheap to rebuild. |
| 5 | **TTFT median** | `ttft_p50_ms`, one line per arm, ms. What a *typical* request saw. TTFT is right-skewed, so panel 4's mean can sit where no request was — nine 100 ms prefills and one 30 s cold miss average to 3.09 s. Read against panel 4: **a mean that moved without the median is a tail effect**, not a faster common case. Same ring as panel 6, so the median-to-p95 spread is one population. |
| 6 | **TTFT p95** | `ttft_p95_ms`, two lines, ms. The tail the mean hides: destroying one large prefix can leave the mean flat and still make a minority of requests much worse. |
| 7 | **Redo share** | Windowed `ΔrematBlocks / Δblocks_cached`, two lines, 0–1. Normalises panel 2 for throughput, so a quiet run cannot fake an improvement. |
| 8 | **Occupancy** | `used`/`total` as a percentage. Baseline dashed, policy solid. Y-axis 0–100%, fixed. |
| 9 | **Free queue depth** | `queue`, two lines. The eviction-candidate pool; a policy that keeps it deeper is holding more evictable-but-cached blocks. |
| 10 | **Eviction rate** | `Δevicted / Δt`, blocks/s, two lines. |
| 11 | **Splice activity** | `ΔsplicedBlocks / Δt`, policy only, filled area. Annotate `splices=0 for the whole run` in red if the policy series is flat zero — that is a bug, not a result. |
| 12 | **Attribution — policy** | Stacked area, one band per `job:node`, plus `<untracked>` and `other`. Y is blocks evicted per window. |
| 13 | **Attribution — baseline** | Same, same colours, same y-scale as panel 11. Locking the y-scale across the two is what makes them comparable at a glance. |
| 14 | **Regret** | `regret`, policy only. Line, 0–1. Baseline emits a constant `0.000` — do not plot it; a flat line at zero reads as "no regret" rather than "not measured". |
| 15 | **Scored coverage** | `evictedByScore / evicted`, policy only. Near zero means the policy is running but has no opinion on what it evicts — degraded to LRU without saying so. |

### Summary header

Above the panels, a compact table: one value column per loaded arm, then one Δ
column per non-reference arm. Movement first, because that is the claim:

- **KV rebuilt (`remat_mb`), and the delta as a percentage** — the headline
- **TTFT mean (`ttft_ms`)** — the second headline, because it is the only
  figure that says the movement mattered
- TTFT median (`ttft_p50_ms`), directly under the mean — the pair *is* the
  finding, and a mean that moved without the median must not be quoted as
  "requests got faster"
- TTFT p95 (`ttft_p95_ms`), and `ttft_n` beside it: means over different
  request counts are not a comparison
- redo share (`remat_ratio`)
- hit rate (cumulative)
- total evictions
- mean occupancy
- blocks spliced
- final regret
- distinct `job:node` keys seen

This is what gets screenshotted into a writeup. Make it copyable as text.

**Guard against the flattering comparison.** If a run differs from the
reference in `query_tokens` by more than ~10%, the two did not serve the same
work and the absolute `remat_mb` delta is not a fair number. Detect it and
print a warning banner naming *that arm*: *"<arm> served a different amount of
work from <reference> — N% apart."* One banner per offending arm, not one
rolled-up banner, which would name none of them. A visualisation that lets
someone quote a win they did not earn is worse than no visualisation.

The same guard applies to TTFT, for a different reason: it is the one
per-*request* number on a page of per-*block* numbers, so an unmatched
workload corrupts it silently. Warn when `ttft_n` differs by more than ~10%
from the reference.

**And against comparing an arm with itself.** Two slots holding the same
`variant=` is almost always a mis-drop, and their delta is run-to-run variance
dressed as a policy effect. Name the slots and say so.

---

## 7. Interaction

- **Crosshair** synchronised across every panel, with a single tooltip showing
  all series at that `t` for both runs.
- **Brush to zoom** on any panel, applied to all. Double-click resets.
- **Legend toggles** per `job:node` label, applied to both attribution panels
  at once.
- **Solo toggle** per loaded arm, plus "all runs". A slot with no run loaded
  hides its button rather than offering a filter that empties the page.

Nothing else. Every additional control is one more thing to explain in the
writeup.

---

## 8. Implementation constraints

- **Single self-contained HTML file.** No CDN, no external stylesheet, no
  webfont, no fetch. Inline every byte of CSS and JS. This is not a
  preference — a strict CSP blocks external hosts, and the page needs to work
  from a `file://` URL on someone else's laptop.
- **No charting library.** The panels are line and stacked-area charts on a
  shared linear x-axis. Hand-rolled SVG is a few hundred lines and avoids
  vendoring a minified bundle into the repo. Canvas only if a run produces
  more than ~5k samples, which at a 30s period means a 40-hour run.
- **Theme-aware.** `@media (prefers-color-scheme: dark)` plus
  `:root[data-theme="dark"]` / `:root[data-theme="light"]` overrides, so an
  explicit toggle wins in both directions.
- **Responsive.** Panels scale to width; the summary header wraps. Any wide
  element scrolls inside its own `overflow-x: auto` container — the page body
  must never scroll horizontally.
- **Degrade honestly.** One file loaded → render that run alone and say so in
  the header, rather than drawing empty comparison series.

---

## 9. Where to put it

`plan/hbm-timeline.html`, built from these two docs. It is a development tool,
not a shipped artefact — it does not belong under `vllm/`.

If it is instead published as a hosted artifact, note that the log files stay
local: the page reads them client-side and never uploads them.
