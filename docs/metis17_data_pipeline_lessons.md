# Metis-1.7 data pipeline: lessons from the 1.6 build

Written 2026-08-04, during the Metis-1.6 `metis-1.6-data-r1` build on Portage.
Everything here was measured against real data on Lustre, not reasoned about.

This file lived outside the repository for most of the 1.6 build, because
`require_clean_repository` treated an untracked file as dirty and committing it
moved HEAD, which un-pinned the source lock (§0). Writing down a lesson could
cost a rebuild, which is itself the lesson in §0. That is fixed: stage identity
is now a fingerprint of the modules a stage runs, so a file under `docs/`
changes nothing, and the document lives with the code it is about.

---

## 0. The worst defect in the pipeline: cleanup makes the build unrestartable

**This cost a 1.37 TB re-download and a full rebuild from normalize. It is the
most expensive single thing in this document, and it is not a performance bug.**

Three rules in the 1.6 pipeline are individually reasonable and jointly fatal.

1. `cleanup_raw` is a **stage inside the build graph**. After normalize is
   verified, it deletes `raw/`, `cache/huggingface`, `cache/common-crawl` and
   `cache/tmp/materializers` — the entire acquisition output.
2. `verify_acquisition_handoff` runs on **every `submit` and every `resume`**,
   and calls `_artifact_record` on every file the download tasks recorded. A
   missing file is a hard `RuntimeError: Acquisition output is missing`.
3. The source lock is bound to a repository commit, and
   `_validate_existing_lock` refuses to reuse it from any other commit:
   *"resume with the original commit or create a new data release."*

Read them together. The moment `cleanup_raw` completes, the build can only run
**forward**. It cannot be resubmitted, because rule 2 demands files rule 1 just
deleted. It cannot be fixed and resumed, because rule 3 pins it to the commit
whose inputs are gone. And it cannot be restarted at a new commit, because a new
commit means a new lock, a new execution contract, and therefore a re-run of
normalize — which needs the raw data that no longer exists.

A single interruption after that point — a cancelled job, a node failure, an
operator deciding to deploy a one-line speedup — costs the entire acquisition.

**How this actually played out.** Span dedup was running slowly. We cancelled
the array to deploy a file-pool fix (§1d). `cleanup_raw` had completed ~16 hours
earlier, silently, as a normal graph stage. Cancelling was therefore already
irreversible; we simply did not know it yet. Every recovery route was checked
and every one was closed:

| Route | Why it fails |
|---|---|
| `submit build` again | handoff verifier: 1,862 raw files missing |
| `resume build` | same verifier, same call site |
| restore the old lock + completions | works, but then the checkout must be the old commit, so the fix cannot ship |
| flip `require_acquisition_handoff` off | `gates` is *inside* the execution contract (§6a), so this invalidates normalize, which then needs raw |
| patch the verifier to honour cleanup receipts | a code change is a new commit is a new lock is a new contract — same wall |

The state *was* fully recoverable in the sense that matters for correctness: the
archived `sources.lock` / `build.inputs` / `ACQUISITION_READY` triple
reproduced the original `execution_contract_sha256` for `normalize`,
`exact_filter` and `span_prefilter_signature` exactly, and 923 GB of
exact-deduped corpus plus 1,202 span tasks were still on disk. It was recovery
into a **frozen** build: correct, resumable, and permanently unmodifiable.

There is a tempting fake fix here. `submit` calls the verifier with
`verify_artifact_hashes=False`, so it checks only `is_file()` and `st_size` —
1.37 TB of correctly-sized empty files would satisfy it. **Do not.** That
converts an integrity check into a lie, and nothing downstream would catch it.
If the only way to make a gate pass is to fabricate what it inspects, the gate
is telling you the truth about the state you are in.

### What 1.7 must do

- **Never let a graph stage delete a precondition that a later `submit`
  re-validates.** If cleanup must exist, the handoff verifier has to consult the
  cleanup receipt and accept documented, hash-recorded retirement. The receipt
  already contains everything needed — `verified-content/<stage>.jsonl` carries
  per-file size and SHA-256 for the whole tree.
- **Default to retaining.** The 1.6 fix was a `retain_stage_inputs` gate that
  makes verified cleanup skip its deletions. Portage reported **4.7 PB free**
  against a **1.37 TB** input set. The stage was solving a problem we did not
  have, at the cost of the ability to recover. At 30 T tokens the inputs get
  bigger, but so does the cost of re-fetching them — the ratio does not improve.
- **Separate "what the data is" from "what the code is" in the contract.**
  Binding stage completions to the repository commit means a typo fix in a
  comment invalidates a week of dedup. The contract should hash the manifest,
  the lock's *source content*, and the policy knobs that change output — not the
  commit SHA, and not the whole scheduler block (§6a). Code identity belongs in
  provenance, recorded and reported, not in the resume key.
- **Make the trap loud.** `cleanup_raw` should refuse to run, or at minimum log
  a one-line warning, when it is about to delete files that the active profile's
  `require_acquisition_handoff` gate will demand back.

### The general rule

An irreversible step and a revalidated precondition must never be the same
files. Before writing any stage that deletes, ask the question that would have
caught this in review: **"after this runs, what does `resume` check?"**

---

### 0a. The source lock was not the only over-binding

Section 0 fixed the source lock: stage identity became a fingerprint of the
modules a stage runs, so unrelated commits stopped invalidating finished work.
That paid off repeatedly during the 1.6 rebuild -- walltime raises, array
throttle changes and a reverted commit all cost nothing, and one of those raises
(decontam_filter 18h to 72h) is the only reason two 14-hour shards were not
killed mid-flight.

It did not help with the change that mattered most.

The decontamination policy lives in `manifests/contamination/eval-holdouts.yaml`
and is hashed into the evaluation holdout bundle. The bundle is immutable within
a data release. So changing a *matching threshold* -- a number that cannot alter
which benchmarks are withheld, only how overlap is detected -- required cutting
metis-1.6-data-r2 and rebuilding every stage from normalize. Three guards
enforced it in sequence, and each was right to: decontam_index refused the
mismatched policy hash, prepare_holdouts refused to rebuild in place, and
rehandoff refused to re-attest changed acquisition data.

The binding conflates two different things: *what is held out* (the benchmark
inventory, which must be immutable within a release) and *how overlap is
detected* (thresholds, which are tuning). The first deserves the guard. The
second does not, and paying a full rebuild to retune a threshold is what makes
operators guess at values instead of measuring them.

**For 1.7:** split the holdout bundle's identity. Hash the benchmark inventory
and the extracted fragments into the release, since changing those changes what
the model is allowed to see. Keep detection thresholds beside them as tunable
policy, versioned and recorded in the release manifest but not part of the
bundle's immutability. Then a threshold sweep costs a decontam re-run rather
than a corpus rebuild -- which is the difference between measuring the right
value and guessing at it twice.

---

### 0b. The dominant failure mode: configured, and silently inert

Four times in one build, a setting was configured correctly, persisted
correctly, and had no effect, with nothing failing to say so.

| where | what happened |
|---|---|
| `eval-holdouts.yaml` | thresholds pinned to the holdout bundle; editing them changed nothing until the bundle was rebuilt |
| disk contamination index | `save`/`load` did not round-trip the tuning, so `decontam_index` built with it and `decontam_filter` ran without it |
| `_policy_contract` | cast every field through `int()`, flooring `match_fraction` 0.002 to 0 while still recording the field as present |
| monitor error probe | grepped a log path the running supervisor had stopped writing, reporting `errors=0` for hours from a stale file |

Add to those the audit that read `repo_path` when the field was named `path`
and reported a confident zero suspect files, and the marker check that looked
up `task-<index:06d>` when the stage wrote `task-<index:08d>-<digest>.json`.

The shape is always the same: **the absence of a signal is read as the absence
of a problem.** A zero, a default, a missing key, an empty grep. None of them
can distinguish "nothing is wrong" from "nothing was measured", and every one
of them defaults to the reassuring reading.

**What to do about it in 1.7:**

- **Make defaults loud.** A tuning value that falls back to zero should log the
  fallback, or refuse. `getattr(self, "match_fraction", 0.0)` is a silent
  revert to old behaviour dressed as a safe default.
- **Round-trip anything that crosses a process boundary, and assert it.** The
  index knew its own thresholds; nothing checked that the reader saw them.
- **Verify a probe can produce a non-zero before trusting its zero.** Every
  audit above would have been caught by testing it against a case that must
  fail. The habit is cheap: measure something you know is broken first.
- **Prefer failing to defaulting** anywhere a value changes what the data
  becomes. A missing threshold should stop a build, not quietly restore the
  behaviour someone was trying to change.

The pipeline's fail-closed gates are good at catching *wrong* values. Nothing
in it catches a *value that never arrived*, and that is the more common defect.

---

## 1. The single biggest performance defect: YAML parsed once per record

**Measured: 82% of normalize CPU time, 5.6x speedup available.**

```
as the build runs today (YAML per record):  0.27 MB/s
with quality profiles cached             :  1.52 MB/s
```

`evaluate_quality` opens with:

```python
profiles = profiles or load_quality_profiles()
```

`stage_runner.py:1348` calls it **without** `profiles=`, so every record in the
build re-reads and re-parses `configs/metis16/quality-profiles.yaml` off Lustre
to make one accept/reject decision against a ~100-line file that never changes.
A cProfile run over 300 records showed `yaml.compose_node` called 103,500 times
— the top eleven entries by `tottime` were all `yaml/`.

`profile_preflight.py` already does it correctly (`profiles=profiles`), which is
why the preflight sweeps 3,000 records in a minute while the build spent ~2
hours per input file.

**Fix:** `functools.lru_cache` on `load_quality_profiles`, or thread the cached
dict through from the caller. One line either way.

**Scope:** normalize only. `evaluate_quality` has exactly one call site in the
whole build. Dedup stages hash and compare; they never evaluate quality. Do not
expect this to speed up dedup.

**Confirmed in production, not just in the profiler.** After the fix the stage
ran at **77.4 markers/min against the previous run's 12.9/min peak — 6.0x**,
matching the predicted 5.6x. A 9-hour stage became roughly 30 minutes of work
plus a 4-hour single-file tail (see 10d). The cache itself: 2,000 calls in
0.089s, ~44us each, against ~10ms uncached.

**Related but minor:** `_manifest()` is uncached and `_stage_execution_contract`
calls it once per task — 0.109 s per call, about 14.6 CPU-minutes across the
whole graph. Negligible; note it only so nobody re-discovers it as a suspect.

**Lesson for 1.7:** profile the per-record path before assuming the bottleneck is
I/O or node count. The wall-clock story was never "two hours of work per file";
it was ~20 minutes of work wrapped in 100 minutes of re-parsing the same file.

---

## 1b. The next 2x is per-character Python loops — TOP PRIORITY AT 30T

With the YAML parse gone, the profile is dominated by walking each document
character by character in the interpreter. Measured on `pes2o`, 200 rows /
0.27 MB, post-cache: **1.32 MB/s, 1.0 ms per 1 KB document — about one
microsecond per character.**

| cost | calls for 200 rows |
|---|---|
| `sum(c.isalpha() for c in text)` — quality.py:150 | 269,535 |
| `sum(not c.isalnum() and not c.isspace() ...)` — quality.py:151 | 269,535 |
| `[c for c in text if c.isalpha()]` — evidence.py:400 | per char |
| `sum("LATIN" in unicodedata.name(c,"") for c in letters)` | **218,223 `unicodedata.name` calls** |
| `str.isalpha` total | 538,670 |
| `typing.__subclasscheck__` (isinstance overhead) | 36,400 |

Every document is traversed **four separate times**: alpha fraction, symbol
fraction, building a list of letters, then a Unicode-database lookup per letter
to decide whether it is Latin. `unicodedata.name()` per character is the worst
of them and is entirely avoidable.

**Fixes, cheapest first:**
- the Latin test needs no `unicodedata.name()`. An `str.isascii()` fast path
  settles the overwhelming majority; reserve the lookup for the remainder.
- `text_features` walks the string twice for alpha and symbol fractions. One
  pass, or `re.findall` / `str.translate` counting, gets both.
- `sum(map(str.isalpha, text))` is roughly 2x over the genexpr for free; a
  translate table or regex count is far more.
- `_computed_english_probability` truncates `words` to 250k characters but
  **not** `letters`, so a large document pays the full per-character cost on the
  single most expensive loop. Truncate both.

**Why this outranks everything else here at 30T.** Metis-1.6 normalized 1.37 TB
for a 1 T-token target; a 30 T-token corpus is roughly thirty times that input.
At today's rate normalize alone scales into days of pure CPU, and combined with
the file-size tail in 10d it becomes the dominant cost of the entire build. The
YAML fix gave a measured 6x. This is plausibly another 2x or better, putting
normalize **10x+** off where 1.6 began — the difference between a stage you plan
the week around and a stage you stop thinking about.

Do it before the first 30 T build, not during: once a build is in flight every
fix costs a full re-run (see 10e).

---

## 1c. Span dedup was NOT the sentence splitter -- it was a thrashing file pool

**This section originally recommended rewriting `sentence_spans` with a regex.
That recommendation was wrong, was tested, and is retracted. Read 1d first: the
real cause was a file-handle pool sized below its bucket count.**

The profile below is accurate as far as it goes, and it is a good example of how
a correct profile can still point at the wrong fix.

`span_prefilter_signature` ran at **7.53 MB/s** and took **11.5 hours to reach
1,190 of 1,862 tasks**, with node CPULoad between 18 and 62 out of 192 -- 10-32%
utilisation, the same "not saturated because it is stuck in the interpreter"
signature as the YAML bug.

Profiled on 300 real post-exact-dedup documents (9.13 MB):

| | time | share |
|---|---|---|
| `sentence_spans` (character-by-character Python loop) | **1.541s** | **68%** |
| `re.findall` in `_canonical_sentence` | 0.198s | 9% |
| `span_digest` | 0.066s | 3% |
| **SHA-256 itself** | **0.026s** | **1%** |

The cryptographic hashing everyone assumes is expensive is **1%**. The stage is
its sentence splitter, which walks every character of every document in Python:

```python
while cursor < length:
    character = text[cursor]
    if character == "\n": boundary = cursor
    elif character in ".!?": ...
```

The rule it implements -- a newline is a boundary, and a run of `.!?` followed by
whitespace or end-of-text is a boundary -- is one regex:
`re.finditer(r'[.!?]+(?=\s|$)|\n', text)`. That runs in C and should be 10-50x
faster, taking the stage to roughly 3x overall.

**This compounds across the span chain.** `sentence_spans` is called from
`iter_span_signatures` (used by `span_prefilter_signature` and
`span_signature`), and from `strip_duplicate_spans` and `build_span_dedup_filter`
(used by `span_filter`) -- three of the five span stages.

**Tested, and it does not work.** The regex rewrite is exactly equivalent --
5,823 documents, 2,800 of them real corpus rows, plus adversarial cases like
`"..!?.."`, `".\n."` and `"U.S.A."`, with zero mismatches -- and it is **not
faster**: 8.7 MB/s before, 8.3 MB/s after, i.e. 0.9x.

The profiler was right that `sentence_spans` holds 68% of the stage's `tottime`,
and the wrong conclusion was mine. That time is the **per-sentence** work inside
the loop -- `_canonical_sentence` doing NFKC normalisation and casefolding, the
`WORD_RE.findall` word count, the slicing, the `SentenceSpan` construction --
not the character scan that finds the boundaries. Replacing the scan removed the
part that was not costing anything.

**The lesson is about profiling, not about splitting.** `tottime` on a function
containing a loop attributes the loop body's inline work to the function, so a
hot function is not the same as a hot line. Before rewriting, measure the
candidate replacement in isolation -- here, timing `sentence_spans` with the
body stubbed out would have shown the scan was cheap in about a minute.

If this is revisited for 1.7, the target is the per-sentence work: memoise
`_canonical_sentence` (documents repeat sentences), or compute `words` from the
already-normalised string without a second regex pass. Do not touch the scan.

## 1d. What span dedup actually was: a file pool smaller than its bucket count

Three passes bucket signatures by `digest % finder_workers` and write each
bucket through an LRU pool of open file handles. All three sized that pool at
**32** while `finder_tasks` is **64**:

```
span_dedup._BoundedFilePool        64 buckets, 32 handles (config)
code_dedup.write_code_signatures   64 buckets, 32 handles (hardcoded)
final_dedup._FilePool              64 buckets, 32 handles (default argument)
```

The bucket for a signature is a hash, so the target is effectively random per
record. With half the buckets resident, roughly **every second write evicted a
handle and reopened another file**, and on Lustre an open and a close are
metadata round trips. This corpus emits about 6.4 billion span signatures.

Measured on the live run: `span_prefilter_signature` reached 1,190 of 1,862
tasks in **11 hours 31 minutes** at 18-62 CPULoad out of 192. The same work
profiled single-threaded runs at 7.45 MB/s, which across the ~640 resident
workers puts the compute cost of the whole 923GB corpus at **minutes**. Two
orders of magnitude apart, and the gap was entirely the pool.

**Confirmed by direct measurement on the same Lustre filesystem.** 64 buckets,
60,000 records, hash-random bucket order, one pool size against the other:

| pool | wall | throughput | opens |
|---|---|---|---|
| 32 handles | 35.17 s | 1,706 rec/s | 29,870 (**49.8%** of writes) |
| 96 handles | 0.13 s | 479,779 rec/s | 64 (one per bucket) |

**281x on the writer path.** The 49.8% miss rate is exactly the K/N = 32/64
predicted for an LRU under uniform random access, and the difference works out
to **1.18 ms per open-and-close** — one Lustre metadata round trip, paid on
every second record, 6.4 billion times.

Note what this measurement does and does not say. It isolates the writer. The
stage also reads inputs and computes signatures, which is CPU work the pool size
cannot touch, so the end-to-end stage speedup is bounded by the writer's share
of stage time and will be smaller than 281x. The mechanism and its magnitude are
settled; the stage-level multiplier still has to be measured in production.

**How to spot this class:** aggregate throughput far below single-threaded
throughput times worker count, with CPU unsaturated. That combination means the
workers are blocked on something, and on a shared filesystem the first suspect
is metadata operations, not bandwidth.

**For 1.7:** assert `maximum_open_files >= finder_tasks` at profile-validation
time. It is a one-line invariant and it silently cost more than every other
defect in this document combined.

---

## 2. Do not confuse node count with parallelism

I initially reported "we're using 3% of the cluster, 3–6x available." **That was
wrong** and acting on it would have destroyed ~1,300 completed tasks.

We occupied 17–32 nodes of 124, but each node ran ~48 worker processes. Tasks in
flight were `32 x 48 = 1,536` against a total of `1,862` tasks. Tasks are input
files; there are no more to hand out. **The real ceiling was 1.21x, not 3-6x.**

Nothing was saturated because nothing needed to be: CPULoad ~36 of 192, 497 GB
free of 512 GB RAM, ~74 MB/s writes.

The `*_find` stages look worse (6 elements for `span_find`, 4 for `code_find`)
but are the same story — their task counts come from bucket counts and
`tasks_per_job` already puts every task in flight.

**Lesson for 1.7:** compute `total_tasks` vs `tasks_in_flight` before proposing a
parallelism change. If they are close, the only lever left is making each task
faster (see §1).

**What was genuinely recoverable:** arrays whose element count exceeded their
`%N` throttle. Raised live with `scontrol update JobId=<id> ArrayTaskThrottle=<n>`
— no profile edit, no commit, no contract change:

| stage | was | now |
|---|---|---|
| `decontam_filter` | 32 | 47 |
| `verify_shard`, `pack` | 32 | 42 |
| `token_count`, `tokenizer_sample` | 32 | 39 |
| span/minhash/code signature+filter | 32 | 34 |

Caveat: `ArrayTaskThrottle` is not persistent. A cancel-and-resubmit reverts to
the profile's `max_concurrent`. For 1.7, fix `max_concurrent` in the profile
instead — but only at submission time, never mid-build (§6).

---

## 2a. Acquisition is link-limited, so its parallelism belongs across hosts

Acquisition ran at 126 MB/s with `max_workers: 4`. Raising it to 24 — 157 live
threads — produced **124 MB/s**. No throttling from the Hub, load average 1.97
on 384 cores. The extra concurrency bought exactly nothing, and the reason was
visible in one number the whole time: 126 MB/s is 1008 Mbps.

```
bond0        25000 Mbps  (up)   internal fabric
hsn0/hsn1   200000 Mbps  (up)   Slingshot interconnect
ens2f3        1000 Mbps  (up)   default route  <- all external traffic
```

A login node with 384 cores, a 25 Gbps bond and a 200 Gbps Slingshot fabric
reaches the internet through a 1 Gbps management NIC. The fast interfaces are
for Lustre and MPI. **Check `ip route` against interface speeds before tuning
any download concurrency**, because saturation looks identical to a tuning
problem, and every knob inside the host is the wrong knob.

The right axis is more hosts. login1 is an identical node with its own 1 Gbps
uplink, and it needed no code: `metisctl download-task --profile P --task-index N`
is a per-task entry point that skips completed tasks and does not take the
supervisor's singleton lock. Running it on login1 descending from the last task
while login2's supervisor ascends from the first took aggregate throughput to
**231 MB/s**, and the two never collide because `StateStore.task_lock` is a
mkdir mutex on Lustre — atomic, and deliberately unwilling to reclaim a lock
whose `OWNER.json` names a different host.

**The caveat that matters operationally:** that same refusal means a helper
process dying on login1 leaves a lock its own host can clear and login2 cannot.
`metisctl unlock-stale` exists for exactly this, and a multi-host acquisition
should expect to need it.

**For 1.7 at 30 T:** 1 Gbps is 10.8 TB/day per host. A 30 T-token corpus is
tens of terabytes of candidates, so acquisition is measured in host-days and the
only lever is how many hosts pull at once. Design the supervisor for it —
`--shard i/n` as a first-class flag rather than a hand-driven loop — and confirm
whether a data-transfer node with a real uplink exists before assuming the login
nodes are the ceiling.

---

## 3. Profiles demanded evidence publishers never ship

This was the dominant *correctness* failure class, worth ~45B usable tokens.
Each case is the same shape: a gate asserts something about the data that nobody
checked against the data, and fail-closed turns "we could not measure this" into
"this is bad."

| source | symptom | actual cause |
|---|---|---|
| `nemotron_specialized_fact_seeking` | 44/60 `missing_language_probability` | detector returned `None` below 100 letters / 30 words; rows run 18–55 words. Short is *uncertain*, not *unmeasurable* |
| `openstax` | reported `0/1` | preflight sampled only the **first input file** per source; openstax is 76 books in 76 files, so "one file" was one record |
| `finepdfs_edu_english` | 38.3% vs a 76.7% ceiling | `reading_order_passed` internally required `repeated_page_edges <= 0.08` **and** the profile gated the same fraction again. Relaxing either alone changed nothing, and the redundancy hid the cause |
| `open_law_usgpo` | 38/60 `personal_data` | agency office numbers in Federal Register notices — (817) 222-5110 is FAA Fort Worth |
| `nemotron_math_proofs` | 7/60 language rejections | ships `lean.jsonl` with `formal_statement`/`lean_header` and **no** `ext` field, so the formal-language test never fired and Lean was scored as English prose |
| `megamath_unique` | 54/60 `math_score_minimum` | **units**: MegaMath-web states a 0–1 probability, FineMath a 0–5 integer, gate threshold is 3. A row rated 1.00 scored 1.0 against 3 |

**Lesson for 1.7:** before writing a gate, dump the actual columns of the pinned
config and confirm the field exists, is populated, and is on the scale the
threshold assumes. `preflight-profiles` catches this in a minute; it was written
for exactly this and is the highest-value tool in the repo.

**Corollary — categories that repeat lines by construction:** code, worked
mathematics, books, and LaTeX papers all repeat lines structurally. The repo had
already raised `maximum_repeated_line_fraction` for the first three;
`scientific_paper_v1` still sat at the 0.20 default and passed only 51.1% of
arXiv papers (0.45 passes 97.2%). Check the whole family when fixing one.

---

## 4. `allow_patterns` on multi-config Hugging Face repos

**Seven of the pinned sources used unrestricted patterns.** Every one either was
broken or needed auditing. This was the largest single class of defect.

- **`cosmopedia_v2`** — the id 307-redirects to `HuggingFaceTB/smollm-corpus`,
  which holds three configs. `**/*.parquet` matched all 673 GB and the resolver
  filled its byte target from `fineweb-edu-dedup`: the raw CommonCrawl corpus
  Cosmopedia was *generated from*, not Cosmopedia. Zero yield, and a duplicate of
  the separately pinned `fineweb_edu`.
- **`proof_pile2_math` / `proof_pile2_science`** — same repo, same revision, same
  pattern. Both resolved against the identical 482-file set; all 48 of science's
  files were also among math's 73.
- **`finemath_unique`** — `finemath-4plus` is a nested subset of `finemath-3plus`
  (confirmed against the data: every `int_score>=4` row carries raw score >= 3.5,
  exactly 4plus's published floor). The pattern counted those rows twice.
- **`megamath_unique`** — redirects to `IFM/MegaMath`; `megamath-web-pro` is a
  model-rewritten refinement of `megamath-web`, so taking both double-counts and
  contradicts `provenance.generated=false`.

**Lesson for 1.7:** pin the config, never the repo. Check for redirects — two of
seven repo ids silently redirected elsewhere.

### 4a. The glob trap that produces a silent zero-token source

`manifest.matches_any` is `fnmatch`-based, and `fnmatch` treats `/` literally.
The `**/` fallback only strips a **leading** `**/`.

```
finemath-3plus/**            -> 128 files   CORRECT
finemath-3plus/**/*.parquet  ->   0 files   MATCHES NOTHING
```

Whether the second form works depends on whether files sit directly under the
config dir or one level deeper — `megamath-web/**/*.parquet` *does* match,
because that config nests. **Always validate a pattern against the real tree
using the repo's own matcher before committing it.** A pattern that matches
nothing produces a zero-token source, not an error.

---

## 5. Silent-failure mechanics worth knowing

**`_row_metadata` pre-seeds metadata and `_set_evidence` will not overwrite.**
Any adjustment in the profile block was discarded for exactly the rows that ship
the field the adjustment is for. This had silently disabled the
`nemotron_cc_math_4plus` and `_unique_3` partition floors long before I touched
anything. Fixed by marking derived scores and passing `overwrite=`.

**A self-confirming constant.** `compressed_bytes_per_token: 0.75` for Cosmopedia
was wrong by 5x (measured 3.86 from parquet column-chunk metadata). It cannot
self-detect: `_select_files` stops once selected bytes reach
`candidate_tokens * ratio`, and the shortfall check divides those same bytes by
the same ratio. It would have taken 8 of 104 shards, landed ~2.4B tokens against
an 8B budget, and reported `candidate_target_met`. **Measure bytes-per-token from
the actual files, per source.**

**Attestation vs. rubber stamp.** Cosmopedia ships no generator column at all, so
`require_genealogy` rejected 705 of 720 rows. The fix is a manifest attestation
naming the pinned generator with a written basis — legitimate, because the
generator is a property of the release. But note `seed_data` is the seed *corpus
name* (9 distinct values across 720 rows), so hashing it into
`source_document_id` satisfies the grounding gate with evidence that grounds
nothing. The repo's own comment warns about this: "one value shared by every row
identifies nothing." **For 1.7, prefer hashing the actual seed text (in `prompt`)
over a corpus label.**

**Directory names can lie.** In `nvidia/Nemotron-Pretraining-Legal-v1`, the
payloads of the two largest subsets are swapped: `Case-Law-Summary/` contains the
CaseHOLD *task* ("select the correct holding statement"), and `CaseHOLD/`
contains ordinary case-law narrative. Read payloads, never names.

---

## 6. Immutability ordering — the rule that cost three failed submissions

Three consecutive `submit build` attempts failed, each on a different binding:

1. `The data manifest changed after acquisition` — a stale `ACQUISITION_READY.json`
   short-circuits `write_acquisition_handoff`, which validates the *old* handoff
   against the new manifest instead of writing a new one.
2. `The repository commit changed after the immutable source lock was created` —
   the lock binds `HEAD`.
3. `The immutable source lock changed after acquisition` — the handoff binds
   `sha256(sources.lock.json)`.

**The rule: make every code and manifest change first, then `resolve` LAST, then
rebind the handoff, then submit. Any commit after resolve invalidates the lock.**

**`rehandoff` semantics.** It seals `artifact_count` / `artifact_bytes` and
refuses when the acquired data itself moved. It is:
- the **wrong** tool after a re-pin that downloaded new files (it correctly refuses)
- the **right** tool when only the commit/lock moved and no bytes changed

When bytes genuinely changed, archive `ACQUISITION_READY.json`, `HANDOFF_VERIFIED.json`
and the `handoff_signature`/`handoff_verify` completion markers (they embed
`handoff_sha256` and fail two commands later otherwise), then re-run acquisition;
it re-verifies and attests without re-downloading.

### 6a. The scheduler block is inside the execution contract

`_stage_execution_contract` hashes:

```python
"state_artifacts": {sha256 of sources.lock.json, build.inputs.json, ACQUISITION_READY.json},
"scheduler": profile.get("scheduler", {}),
"gates": profile.get("gates", {}),
```

Consequences:
- **Changing any `tasks_per_job` or `max_concurrent` invalidates every completed
  task in every stage.** normalize deletes its output and redoes the task
  (line 1267); filtering stages raise "completion belongs to stale inputs or
  policy" (line 387).
- **Re-resolving mid-build does the same**, because the lock's sha256 is in the
  contract.

**Tune concurrency at submission time only.** Mid-build, `ArrayTaskThrottle` is
the only safe lever. Source-code changes are safe (not hashed) but pulling under
running jobs risks partially-written files — there are lazy imports in the hot
path (`from .handoff_verification import ...` inside the task function).

**For 1.7:** consider hashing only the scheduler keys that affect *results*
(none of them do) rather than the whole block. Concurrency is an operational
knob and should not be provenance.

---

## 7. Slurm operational notes

**Held tasks are invisible to failure monitoring.** Tasks failed to launch with
`user_env_retrieval_failed_requeued_held` (Slurm runs a login shell to retrieve
the environment under `--export=ALL`; it timed out), were requeued, and **held**.
A held task is neither `RUNNING` nor `FAILED`. Under an `afterok` graph it stalls
every downstream stage in silence — a monitor that greps for failures shows
nothing while the build sits dead. One task hit `Restarts=4`.

**Any watcher must check for held/launch-failed states and release them**
(`scontrol release <jobid>`), and must alarm on "nothing running while jobs are
queued." Match `PENDING` only — a `COMPLETING` job keeps the reason string that
held it, and acting on that fires forever.

**Environment:** Slurm reports local time while `date -u` reports UTC; a
`StartTime` that looks five hours stale is probably a timezone confusion.

---

## 8. Contamination: reformulations defeat n-gram decontamination

`nemotron_legal_v1` pulled five subsets that are Qwen3 **reformulations** of tasks
from `legalbench` and `lex_glue` — both pinned in `eval-holdouts.yaml`.

The decontamination policy matches on 13-grams (`explicit_genealogy_match`,
`remove_entire_document`). **A reformulation need share no 13-gram with the
benchmark it came from.** Decontamination is not a backstop for
model-rewritten benchmark data; exclusion at the pin is.

**For 1.7:** treat "synthetic data derived from a task in our eval set" as a
distinct risk from "text that overlaps our eval set," and audit synthetic corpora
against the holdout registry by *provenance*, not by n-gram.

---

## 9. Gate design principles that earned their keep

- **Measure the thing the gate is aimed at.** Repeated page edges do not measure
  reading order; they measure pagination. Currency `$` is not a math delimiter.
  A role mailbox (`support@openstax.org`) is not a person's contact details. An
  RFC 2606 placeholder (`firstname.lastname@example.org`) can never route to
  anyone.
- **Count distinct, not occurrences.** A running footer repeating one address on
  every page is one contact, not forty. `personal_data_maximum_contacts: 4`
  admits a document with a contact block and still rejects a staff directory —
  measured shape: 258 of 300 records carry no contact, 31 carry one or two, a
  thin tail runs higher.
- **Scope exemptions to the profile that needs them,** and test that they do not
  leak. `legal_primary_v1` may keep agency phone numbers; `web_general_v1` must
  not.
- **Never rewrite the training text to satisfy a gate.** The currency fix changes
  only the balance *count*; a test pins that `extract_training_text` is unchanged.
- **Regex needs testing, not reading.** A negative lookahead only suppresses the
  match starting where it is anchored, so `no-reply@x.org` was skipped at `no`
  and matched again at `reply`. And `\$\d[\d,]*` backtracks: `$20` matches as
  `$2`, so the test inspects `0 \times` instead of ` \times` and real mathematics
  reads as a price. Both were found by tests, not by review.

---

## 10a. A producer and a consumer that disagreed about the same flag

`stage_runner` demands the frozen Common Crawl opt-out snapshot whenever a
source declares `provenance.common_crawl_derived` -- deliberately, and the code
says why: a packaged extraction like FreshWeb is Common Crawl text a third party
filtered at build time, which makes it *more* exposed to a later publisher
withdrawal, not less. But `handoff._uses_common_crawl` only wrote the snapshot
when some source had `driver == "common_crawl_ranges"`.

Those two conditions agreed for as long as any source used that driver. When the
`common_crawl_ranges` sources were withdrawn, no source had the driver, so no
snapshot was ever written -- while `metis_freshweb_2025` still declared
`common_crawl_derived` and failed all 34 of its normalization tasks on the
missing block. **Every archived handoff on disk lacks it**, twelve of them.

The part that should worry you most: `verify-handoff` is *supposed* to catch
exactly this (`_verify_final_common_crawl_policy` raises when the block is
missing and Common Crawl is used) -- but it is gated on the same wrong
`_uses_common_crawl`. **A guard and the thing it guards shared a broken
predicate, so the guard passed twelve times on handoffs that were missing the
data.** 2,983 publisher opt-out domains and 118 URL rules were silently not
being honoured.

**For 1.7:** when a check and the code it checks derive from the same predicate,
they cannot catch each other. Assert the *outcome* independently -- "if any
source is common_crawl_derived, the handoff has the block" -- rather than
re-deriving the condition.

## 10b. Fail-closed at task level is the wrong granularity

`normalize` raised when a task accepted zero records. The intent was to catch a
profile demanding evidence no publisher ships. In practice it caught that
**zero** times and stopped the build **three** times on properties of one file:

- a 13-byte zstd frame the publisher released empty
- an OpenStax book, because that source is one document per task, so one
  rejected book is a task that accepted nothing
- `lean_proofsteps.jsonl.zst`, where a proof state plus one tactic repeats lines
  by construction: median `repeated_line_fraction` 0.795, so 4% of rows clear
  `proof_v1`'s 0.50

Each time, a failed task failed its array element and `afterok` stopped all 49
downstream jobs -- in the last case over 12.7MB of a 7.58GB source.

"Every record rejected" is a **source**-level signal and is meaningless at task
level, where a task may hold one document. The zero-yield fact is now recorded
(`zero_yield` plus the rejection histogram, in every report) and the systematic
case is caught where it is visible: `preflight-profiles` before submission, and
`minimum_unique_tokens` at `select`.

**For 1.7:** keep the check, move it. After normalize, assert per *source* that
not every task yielded zero. That catches the real failure without letting one
file stop 49 jobs.

## 10c. An array element reported 34 failures and exited 0

Element 14's own summary said `'failed': 34, 'ran': 48`, and `sacct` recorded
`COMPLETED 0:0`.

**This was originally written up as an exit-code bug. That diagnosis was
probably wrong, and chasing it is instructive.** The batch runner does return
non-zero on failures, and `stage.sbatch` uses `exec`, so the interpreter's
status *is* the job's status — no pipe to swallow it, no trailing command to
overwrite it. Both predate the incident, so neither can explain it. The likely
story is `--requeue`: the element was requeued, the retry recomputed its pending
set from completion markers, ran only the 34 still-unfinished tasks, and passed.
`sacct` reports the final attempt. The log showing `failed: 34` was the first.

Two lessons, and the second is the one worth keeping.

**Reading a requeued array element's log is not reading its outcome.** Slurm
reports the last attempt; `%A_%a.out` may be from any of them. Compare
`sacct -j <id> --duplicates` before concluding anything about an array.

**The exposure was real even though the bug was not.** Everything deciding the
exit code was in-process accounting: a worker reports ok, the parent counts it,
the count becomes the status. Every link is somewhere a task can be lost without
anyone lying — a swallowed exception, a requeued attempt, a refactor that
returns early. `afterok` sees only the status, so a stage that drops tasks and
exits 0 advances the graph over a corpus with holes, and nothing downstream
looks again.

**Fixed by making the exit code an observation rather than a tally.** After a
task group runs, every index it owned is re-stat'ed; a task counts as done when
its completion marker exists on disk, not when it reports that it does. A worker
that returns `ok` and writes nothing now fails the element and names the tasks.

**For 1.7:** prefer deriving status from durable artifacts over in-process
counters anywhere a scheduler makes a branching decision on it. The counter and
the artifact agree right up until the moment it matters.

### 10c-postscript. The fix was worse than the bug, and why

The marker check above was written, shipped, and reverted within a day, because
on the first real submission it failed a stage that had done all of its work:

```
FAIL stage handoff_signature: 16 of 16 tasks left no completion marker
```

against a stage holding **3,280 completion markers**. Completion markers are not
named uniformly, and `StateStore.is_complete` is an exact filename match:

```
handoff_signature   task-00000000-117b1cffc8bb3828.json   8 digits + digest
download            download-000000-d5b1bb2fd544f305.json stage-prefixed
holdouts            task-000000.json                      plain 6 digits
```

The check looked up `task-<index:06d>` and could never find the first two forms.
The whole 37-stage graph then sat on `DependencyNeverSatisfied`.

The same wrong assumption was already present, one line above, in the `pending`
computation that decides which tasks to skip — so stages whose markers carry a
digest have always re-run instead of resuming. Wasteful, harmless, and years
old. Promoting it to an exit code made it fatal.

Three things worth keeping:

- **A safety check is production code.** This one was reasoned about carefully
  and tested against a synthetic state store that used the naming the check
  assumed. The test confirmed the check's own premise instead of the codebase's
  behaviour, which is the easiest test to write and the least useful.
- **Weight the evidence for the bug you are defending against.** This defended
  against 10c, which the section above had already downgraded to "probably a
  requeue artifact." Defending hard against a bug that likely does not exist,
  with a mechanism that can fail closed, is a bad trade.
- **Before asserting on a convention, verify the codebase actually guarantees
  it.** `ls` on one completion directory would have shown the digest suffix in
  seconds. Any future attempt needs one canonical completion key per stage —
  ask the state store what a task's marker is called rather than guess its
  shape.

## 10d. Task granularity is one file, and files are not the same size

Input files range from 13 bytes to 29.53GB. Task granularity is one file, so the
tail of every parallel stage is set by the largest single file, and no amount of
cluster is relevant to it. Measured: after the YAML fix the bulk of normalize
finished in about 30 minutes, while the tail ran for hours on a handful of
monsters -- one element spent **4h14m** on a single file.

The skew is inherited from upstream packaging, not created here:

```
median      356 MB        44 files (2.4%) hold 15.7% of all bytes
p90       2,146 MB        nvidia/Nemotron-Math-Proofs-v1 ships its entire
p99       3,378 MB          corpus as one 29.53GB data/lean.jsonl
max      29.53 GB        txt360 ships a single 22.85GB chunk
```

`nemotron_math_proofs` has no sharded alternative in the repo at all -- the
publisher offers exactly one file. So sharding has to happen on our side.

Nothing about the work is serial: a 22.85GB file is ~10 million independent
per-row decisions. It is the *task contract* that is serial, not the
computation.

This also breaks rate-based ETAs. Aggregate throughput says nothing about
completion when the finish time is set by one indivisible unit; I predicted "30
minutes" twice and was wrong twice for this reason.

**For 1.7:** shard large inputs at acquisition, or make task granularity
sub-file (byte ranges / row groups). Either removes the tail and makes ETAs
meaningful.

## 10f. The dedup stages do not record how much they removed

`exact_filter` completion markers carry `stage`, `task_index`, `completed_at`
and the execution-contract hash. No counts. `cleanup_exact` then deletes the
intermediate artifacts, so after a pass completes **there is no way to answer
"how many documents did exact dedup remove"** -- not from the receipts, not from
the leftovers.

Nor can it be inferred from directory sizes: `normalized/` is reclaimed by
cleanup, so there is no before-and-after to compare.

This is the single most important operational number in the whole build. The
`minimum_unique_tokens` gate fires at `select`, stage 33, and whether it passes
depends entirely on how much the four dedup passes remove. Today that is
unknowable until `token_count` at stage 27 -- by which point 26 stages and
several days have already been spent.

**For 1.7:** every filtering stage should write `records_in`, `records_out` and
`bytes_in`, `bytes_out` into its completion marker. `normalize` already writes
`counts` and `rejection_reasons`; the dedup filters should do the same. It costs
nothing and turns a late hard gate into an early prediction.

## 10e. Every manifest edit cascades through three identities

A single-line manifest change invalidates, in order:

1. `manifest_contract_sha256` -> every stage completion is stale by contract
2. the source lock -> the handoff no longer matches it
3. **download task IDs** -> acquisition markers no longer match, and `rehandoff`
   refuses with "Acquisition task is incomplete" even though every byte is
   present and size-matched

The third is the surprising one. The recovery is always: archive handoff +
handoff-bound markers -> resolve -> re-run acquisition (idempotent; it
re-verifies and re-attests without downloading) -> submit.

Regenerating the handoff **also** invalidates stage completions, because
`sha256(ACQUISITION_READY.json)` is inside the execution contract. So there is no
way to fix a manifest bug without redoing normalize. **Batch manifest changes;
never make them one at a time.**

This is what made 1.6 feel like an endless loop: fail-fast means bugs surface
strictly one at a time, and full invalidation means each one costs a complete
re-run. Four bugs became four restarts of a multi-hour stage. There is no
fix-and-resume.

**The fix for 1.7, and it is cheap now and enormous later:** put only what
determines a record's *content* into `_stage_execution_contract`. It currently
hashes the entire manifest and the entire `scheduler` block, so correcting a
licence expression -- metadata about how a record is treated, never what it
contains -- or raising a concurrency limit discards the whole corpus. Hash the
quality profiles, the gates that actually filter, and the input identities;
leave licence text, labels and scheduler tuning out. At 1 T that mistake costs
an afternoon. At 30 T it costs days per typo.

## 11a. Measured stage rates from the rebuilt 1.6 run

Wall times from completion-marker timestamps on the 2.41 TB rebuild, 3,274
tasks per full-corpus stage, array throttle 112, 128-node parry partition.
These are the numbers to extrapolate from for 30 T.

| stage | tasks | wall | rate |
|---|---|---|---|
| normalize | 3,274 | 4.1 h | 13.4/min |
| exact_signature | 3,274 | 15 min | 216.1/min |
| exact_find | 256 | 101 s | 152.1/min |
| exact_filter | 3,274 | 26 min | 124.5/min |
| span_prefilter_signature | 3,274 | 54 min | 60.2/min |
| span_prefilter_find | 64 | 7 min | 9.1/min |
| span_signature | 3,274 | 53 min | 62.3/min |
| span_find | 64 | 2 min | 31.0/min |
| span_filter | 3,274 | 112 min | 29.3/min |
| minhash_signature | 3,274 | 2.1 h | 25.6/min |

**Span dedup, all five stages: 3.8 hours.** Before the file-pool fix, stage one
alone ran 11 h 31 m and reached 64% of a corpus 1.76x smaller.

The stage-level speedup on span_prefilter_signature is **35x** (1.72 -> 60.2
tasks/min), of which about 1.6x is the throttle raise, leaving roughly **22x**
for the pool fix. The isolated writer benchmark in 1d measured **281x**. Both
numbers are correct and they measure different things: the benchmark timed the
writer alone, and the stage also reads inputs and computes signatures, which the
pool size cannot touch. When quoting a micro-benchmark, say what fraction of the
stage it covers, or it will be read as a stage-level promise.

**Retention cost:** raw 2.3 TB + normalized 1.4 TB + eligible 3.0 TB + dedup
3.9 TB = 10.6 TB live, against 4.7 PB free. Keeping every intermediate is what
makes a code fix cost a rehandoff instead of a re-download, and at 1.6 scale it
is free. At 30 T it is roughly 130 TB, still cheap against re-fetching.

---

## 11. Numbers from this build, for calibration

| | start | end |
|---|---|---|
| zero-yield sources | 2 | **0** |
| usable tokens | 928.3 B | **973.7 B** |
| margin vs `minimum_unique_tokens: 950B` | **-21.7 B** | **+23.7 B** |

The build was **below its own gate** at the start and nobody knew, because the
gate fires at `select` — stage 33 of 37, days in. **For 1.7: compute projected
usable tokens (candidate x measured yield, capped at exposure) before submitting,
and treat it as a submission gate rather than a late one.**

Remaining gaps at submission, for reference: `finepdfs_edu_english` 10.3 B,
`nemotron_cc_v2_organic` 4.5 B (the only source that genuinely needs more data
rather than better gates), `megamath_unique` 3.7 B, `nemotron_math_proofs` 2.5 B.

---

## 12. Shard size is the stage makespan, and shard index predicts shard size

Added 2026-08-14, from the `metis-1.6-data-r2` decontamination stall.

`decontam_filter` finished 80 of its 82 array entries and then sat for a further
day and a half on the remaining two. Nothing was wrong with them. One acquired
file becomes one build input, becomes one task, becomes one shard, and **nothing
inside a shard is parallel**, so a stage cannot finish before its single largest
file does.

Measured on the r2 eligible corpus:

| | compressed bytes |
|---|---|
| median shard | 0.20 GB |
| p90 | 1.31 GB |
| p99 | 2.29 GB |
| **largest ten** | **6.83 - 6.86 GB** |

The ten largest are all `s2orc/train`: full academic papers, ~28 KB per document
against a corpus mean nearer 3 KB. They are 34x the median.

Two separate defects made that fatal rather than merely slow.

**Shards were sized by document count, not bytes.** A source whose documents are
ten times longer produces files ten times larger. `materializers.py` already has
`repository_output_shard_bytes`, but `acquisition.mode: external_complete` means
the file sizes are whatever the upstream handed over, and nothing re-cut them.

**Adjacent shards are correlated, and task assignment was contiguous.** Build
inputs sort by `(source_id, relative_path)`, so one source occupies a contiguous
run of indices. `_run_task_group` handed each array entry
`range(first, first + tasks_per_job)`, so all ten giants landed in **two** array
entries. Byte load per entry: max 71.9 GB against a mean of 18.6 GB, **3.87x**.

Both are fixed in the code and **both are off for 1.6**:

- `_task_indices` takes a `task_stride`. Striding deals indices round-robin, so a
  run of large shards spreads one per entry: measured **3.87x -> 1.30x**. Ships
  disabled because an array already submitted was sized for the contiguous
  layout.
- `maximum_input_bytes` splits an oversized input across several tasks, by record
  position (`index % n == p`) rather than byte offset, because a gzip member
  cannot be seeked into and a document must never be divided. Ships at `0`
  because `build.inputs.json` is frozen per release and re-derived on every
  submission, so raising it mid-release makes the frozen-input comparison fail
  and blocks resubmission.

**For 1.7: set `maximum_input_bytes` to about 1 GB and leave
`stride_task_assignment` on, both before the first submission.** On the r2
corpus that turns the ten 6.9 GB shards into seventy ~1 GB tasks and takes
`decontam_filter` from a 40.7 h makespan to roughly throughput-bound. Splitting
costs one extra decompression pass per part, about ninety seconds, against the
tens of hours the imbalance costs.

This is the same lesson as §10d, which recorded that task granularity is one file
and files differ in size. §10d observed it. This is what it cost.

---

## 13. The execution contract was too coarse to retune anything

`_stage_execution_contract` deliberately binds detection tuning, so that a
retuned threshold cannot be silently ignored by a stage already holding a
completion marker. The comment above it says tuning lives outside the holdout
bundle "so it can be retuned without a new release".

It could not. The tuning went into **every** stage's contract. Measured against
the real profile, changing one decontamination threshold invalidated all eleven
stages checked, including `normalize`, `exact_filter`, `span_filter`,
`minhash_filter` and `code_filter` -- stages that never read the tuning and whose
output it cannot possibly change. An invalidated contract makes a stage delete
its output and re-run, so "retune without a new release" actually meant a rebuild
from raw: **82.7 h of critical path, of which decontamination was 40.7 h.**

Now bound only to `decontam_index`, `decontam_filter` and `cleanup_decontam`.

The same coarseness exists one level up and is **not** fixed. `COMMON_MODULES` in
`stage_code.py` contains `stage_runner.py` and `slurm.py`, so any edit to either
changes every stage's code hash. During this session a scheduling change --
which cannot alter any stage's output -- invalidated the completed
decontamination and failed `cleanup_decontam` with *"completion belongs to stale
inputs or policy"*. The rollback was to the previous commit; the fix would be
splitting `stage_runner.py` per stage, which §0 of this document already names.

**For 1.7: treat any edit to `stage_runner.py` or `slurm.py` as a full
invalidation, and do not deploy one mid-build.** Deploy by commit and pull, check
`stage_code_sha256` for an affected stage before and after, and if it moves,
do not ship it until the build is finished.

---

## 14. Decontamination's length bias, and two rules that are not decontamination

Recorded after §8, which noted that reformulations defeat n-gram matching. The
opposite failure is also present: rules that fire on documents nobody copied.

r2 decontamination kept **97.2% of documents but only 92.0% of their bytes**. The
documents it removed averaged **three times the length** of the ones it kept.
Drop rate against document size, over 4,800 sampled documents scored with the
real index:

| size | drop rate | dominant reason |
|---|---|---|
| <1 KB | 0.97% | short_ngram |
| 1-2 KB | 4.57% | short_ngram |
| 4-8 KB | 7.13% | short_ngram |
| 8-16 KB | 12.32% | short_ngram |
| 16-32 KB | 9.52% | code_skeleton |
| 32-64 KB | **20.00%** | contiguous_run, code_skeleton |
| >64 KB | **16.54%** | code_skeleton, contiguous_run |

A corpus that has to supply an 18 B-token long-context extension, with
`minimum_long_document_tokens: 8192`, cannot be biased against its own longest
documents.

Corpus-wide removals, 29.3 M documents in total:

| rule | removed | share |
|---|---|---|
| short_ngram (8-gram, min 4) | 13.5 M | 46% |
| ngram (13-gram, proportional) | 6.7 M | 23% |
| contiguous_run (8 x 13-gram) | 4.0 M | 13% |
| code_skeleton (16-gram, min 32) | 3.3 M | 11% |
| code_ngram (12-gram, min 16) | 1.9 M | 6% |
| **exact** | **117** | 0.0004% |

Current open practice is 13-gram overlap against the evaluation set. That is the
`ngram` family, and it stays. The two largest contributors are not that:

- **short_ngram at 8 tokens is below the standard.** Eight words is ordinary
  phrasing. Because `reason()` tests it last, every one of those 13.5 M documents
  had already passed every 13-gram test -- they are removals *beyond* the
  standard, and there are 115,000 of them for every document exact match caught.
- **code_skeleton erases identifiers and literals**, so it matches structure
  rather than copying. Two unrelated files that loop and branch alike collide,
  and long files collide most, which is why it dominates above 16 KB. Raising its
  threshold to 32 helped and did not change the shape.

Both are now disabled by `0`, which builds no postings and is never consulted.
`minimum_matching_ngrams` has no such switch: the 13-gram rule is the
decontamination contract and a release may not silently ship without it. On the
same sample, disabling both takes 32-64 KB documents from 20.0% to about 9% and
>64 KB from 16.5% to about 7%.

**For 1.7: ship with short_ngram and code_skeleton off, and re-measure the
drop-rate-by-size table before accepting the corpus.** The table, not the
headline retention percentage, is what shows this class of defect -- 97.2%
retention looks fine and hides a 20% loss at the sizes that matter most.

---

## 15. An open disagreement about the r2 token budget

§11 records r1 finishing at **973.7 B usable tokens** against the 950 B gate, a
+23.7 B margin, from a projection of candidate x measured yield.

A direct measurement of r2 disagrees. Post-decontamination text is 3.245 TB over
the 2,989 shards that had logged statistics, extrapolating to **3.503 TB** for
all 3,227. Bytes per token was measured, not assumed: a 65,536-vocab byte-level
BPE trained on a stratified sample of the r2 corpus and evaluated on held-out
shards gives **3.817 bytes/token**, which puts the whole corpus at about
**918 B tokens** -- below the 950 B unique target, before `final_hash` removes
anything.

Neither number should be trusted yet. The projection is a projection. The
measurement extrapolates over the 238 shards without statistics, and its
tokenizer saw 50 MB where the real one sees 160 GB -- a better-trained tokenizer
compresses harder, raises bytes/token, and *lowers* the token count, so the
measurement is if anything optimistic.

`token_count` settles it and is cheap. **For 1.7: run it, and do not treat the
submission-time projection as the answer.** The gate that matters fires at
`select`, stage 33 of 37; a two-hour measurement immediately after
`final_hash_filter` is worth more than a projection at submission.

---

## 16. Status of the 1.7 changes, and what is deliberately not done

Added 2026-08-14, while the 1.6 tokenizer trained.

**Landed on `main`, shipped disabled, safe to enable before the next release:**

| change | where | enable by |
|---|---|---|
| striped task assignment | `_task_indices`, `stage.sbatch` | `scheduler.stride_task_assignment` (already defaults on for new submissions) |
| byte-bounded input splitting | `_split_oversized` | `storage.maximum_input_bytes: 1_000_000_000` |
| tuning scoped to decontam stages | `_DECONTAMINATION_TUNED_STAGES` | already scoped |
| 8-gram and code-skeleton rules disablable | `ContaminationIndex.build` | set both minimums to `0` |
| tokenizer sample shortfall tolerance | `_tokenizer_sample_plan` | `gates.tokenizer_sample_shortfall_tolerance` |
| recorded contract-drift override | `_filter_chain_drift_allowed` | `gates.allow_filter_chain_contract_drift` + a written reason |

**Landed on branch `metis17-fixes`:**

*§10f, filtering stages record what they removed.* DataTrove already writes
per-rank counts to `logging_dir/stats/{rank}.json`; the completion marker simply
never read them. `_datatrove_task_counts` folds `records_in/out`,
`bytes_in/out`, `records_removed` and `removed_by_reason` into every filtering
stage's marker, where they outlive the corpus the paired cleanup retires.

Validated against the r2 decontamination stats across all 3,274 ranks:

```
records_in   1,081,933,951
records_out  1,047,469,393
removed         34,464,558   (3.185%)
   short_ngram          15,024,711     ngram           8,388,537
   contiguous_run        4,825,402     code_skeleton   3,918,560
   code_ngram            2,307,231     exact                 117
```

Those supersede the figures in §14, which came from parsing `.err` files and
missed the 294 shards whose stat blocks had not been flushed. The direction is
unchanged and short_ngram is still the largest contributor at 43.6%.

**Deliberately not done: splitting `stage_runner.py` per stage.**

This is the root cause of §0, §13, and the four separate refusals that blocked
the 1.6 tokenizer sample -- the plan gate, the filter-chain receipt, the
acquisition handoff fingerprint, and a frozen-input identity that moved because
an unrelated function in the same file did. `COMMON_MODULES` contains
`stage_runner.py`, so every stage's identity moves when any stage's
implementation does, and no guard downstream can tell a dangerous change from an
irrelevant one.

It is 4,500 lines holding every stage in the build, and doing it hastily while a
release is mid-flight is how the next silent defect gets introduced. It wants a
deliberate pass with the per-stage tests written first, at the start of a release
cycle rather than the end of one. Until it lands, the drift override is the
pressure valve and its reason string says so.

---

## 17. Tokenizer decisions for 1.7, and one that is not about vocabulary size

### 17a. Split digits. This is the highest-leverage tokenizer change available.

1.6 trains with the GPT-2 byte-level regex, which matches *runs* of digits, so
BPE is free to merge a whole literal into one id:

```
'The year 2026 cost $147832.55'
  -> ['The', 'Ġyear', 'Ġ2026', 'Ġcost', 'Ġ$', '147832', '.', '55']
'x = 1234567'
  -> ['x', 'Ġ=', 'Ġ1234567']
```

`147832` and `1234567` are single pre-tokens. A model given one id for a
seven-digit number has no positional handle on its digits, so place value has to
be memorised per literal instead of learned once. Current practice is individual
digit tokens, and it is listed as settled rather than contested.

`manifest.tokenizer.split_digits` now controls this and defaults to `false`,
which reproduces 1.6 exactly. **For 1.7: set it true.** It is worth more than the
vocabulary increase and costs only the merges it prevents.

### 17b. A 131,072 vocabulary does not fit in uint16

`storage.final_token_dtype: uint16` holds 0-65,535, and 1.6's vocabulary is
65,536: ids 0-65,535, every value used, nothing wasted. That is not a
coincidence, it is the tightest packing available, and it is why the packed
corpus is ~1.82 TiB rather than twice that.

1.7's 131,072 needs ids up to 131,071, which is 17 bits. **For 1.7: set
`final_token_dtype: uint32` at the same time as the vocabulary, and budget
~3.64 TiB for the packed corpus instead of ~1.82 TiB.** Decided rather than
discovered at `pack`.

### 17c. A larger vocabulary makes character-level tasks slightly worse

Worth stating because the intuition runs the other way. A bigger vocabulary
means a word is more likely to be one token, and a token is atomic: nothing in
the embedding for `strawberry` encodes that it contains three `r`s. A smaller
vocabulary splits words into more pieces and leaks marginally more
character-level signal.

So 1.7 at 131k will be marginally *worse* than 1.6 at 65k on letter-counting,
and better on nearly everything else -- multilingual coverage, code, tokens per
document. This is not an argument against the increase. It is an argument for
covering character-level composition in post-training rather than expecting the
tokenizer to supply it, and for not confusing 17a with a fix for it: splitting
digits helps arithmetic, and does nothing for letters.
