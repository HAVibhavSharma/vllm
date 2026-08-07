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
  variant: "baseline" | "node_eviction",
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
| 4 | **Redo share** | Windowed `ΔrematBlocks / Δblocks_cached`, two lines, 0–1. Normalises panel 2 for throughput, so a quiet run cannot fake an improvement. |
| 5 | **Occupancy** | `used`/`total` as a percentage. Baseline dashed, policy solid. Y-axis 0–100%, fixed. |
| 6 | **Free queue depth** | `queue`, two lines. The eviction-candidate pool; a policy that keeps it deeper is holding more evictable-but-cached blocks. |
| 7 | **Eviction rate** | `Δevicted / Δt`, blocks/s, two lines. |
| 8 | **Splice activity** | `ΔsplicedBlocks / Δt`, policy only, filled area. Annotate `splices=0 for the whole run` in red if the policy series is flat zero — that is a bug, not a result. |
| 9 | **Attribution — policy** | Stacked area, one band per `job:node`, plus `<untracked>` and `other`. Y is blocks evicted per window. |
| 10 | **Attribution — baseline** | Same, same colours, same y-scale as panel 9. Locking the y-scale across the two is what makes them comparable at a glance. |
| 11 | **Regret** | `regret`, policy only. Line, 0–1. Baseline emits a constant `0.000` — do not plot it; a flat line at zero reads as "no regret" rather than "not measured". |
| 12 | **Scored coverage** | `evictedByScore / evicted`, policy only. Near zero means the policy is running but has no opinion on what it evicts — degraded to LRU without saying so. |

### Summary header

Above the panels, a compact row of paired numbers — baseline vs policy, with
the delta. Movement first, because that is the claim:

- **KV rebuilt (`remat_mb`), and the delta as a percentage** — the headline
- redo share (`remat_ratio`)
- hit rate (cumulative)
- total evictions
- mean occupancy
- blocks spliced
- final regret
- distinct `job:node` keys seen

This is what gets screenshotted into a writeup. Make it copyable as text.

**Guard against the flattering comparison.** If the two runs differ in
`query_tokens` by more than ~10%, they did not serve the same work and the
absolute `remat_mb` delta is not a fair number. Detect it and print a warning
banner: *"runs differ in served tokens by N% — compare `remat_ratio`, not
`remat_mb`."* A visualisation that lets someone quote a win they did not earn
is worse than no visualisation.

---

## 7. Interaction

- **Crosshair** synchronised across every panel, with a single tooltip showing
  all series at that `t` for both runs.
- **Brush to zoom** on any panel, applied to all. Double-click resets.
- **Legend toggles** per `job:node` label, applied to both attribution panels
  at once.
- **Variant toggle** to solo baseline or policy.

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
