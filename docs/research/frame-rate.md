# Frame rate: where the time goes and what to change

Local research notes (2026-11-25). Not committed, not pushed. Ten read-only
passes: five over this tree (`main` at 8a595d49 plus the uncommitted mesh
fallback), five over other ports. Code findings are from reading, not
profiling; every "payoff" is an estimate until measured.

Today: the lit world runs at about 2-3 fps (other ports on the same base
report the same). One reported breakdown on a fast card (fxceless fork,
issue 696, 232 ms/frame): 74 ms preparing (~150 MB of copies), 150 ms GPU,
37 ms copy-back, and the three barely overlap.

## Step 0: measure before changing anything

The tools already exist:

- The fps line (`crates/victory-gpu/src/host.rs:3196-3220`) splits ms/frame
  into preparing, GPU wait and copy-back, with MB for each. It also reports
  mesh direct/indirect fallback counts per frame (`host.rs:3219`).
- Diagnostics build: `copystat` counters (Checked/Unchanged, Watches,
  Fallbacks, TexHashed), `SUBMISSION_US`, `COPY_BACK`, `VICTORY_GPU_TIMING`
  (`host.rs:774`) for per-list GPU time, `VICTORY_COPY_MODE=none|plain`.
- Pipeline log: `libraries made/loaded/started_early` (`library.rs` `save()`)
  says whether a run is on a cold cache.

Run a fixed stretch of the world twice (warm cache), keep the fps lines,
and compare mean and worst frames, not only the average. TuxBox's
`tools/compare_scarlett_performance.py` (Apache-2.0) is a good model for a
paired before/after gate.

Add two counters first: mesh parts and empty mesh parts per frame.

## Measured on the GTX 1080 (2026-11-25)

Diag build of this tree, `VICTORY_LOG=info`, no input given. The title sits
in its pre-world phase (13 draws/frame). Logs are in `D:\victory-data\perf-run*.log`.

- **Cold cache costs minutes.** A fresh build of `victory.exe` changes the
  exe hash, so the first fps lines read 0.3 fps with 642 pipelines being made.
  The cache took about 2-3 minutes to warm (item 5).
- **Warm baseline: 13-17 fps (73 ms/frame), but only ~6 ms of it is in the
  measured submit path.** The rest was one AboveNormal thread at ~0.86 of a
  core, and every other thread was idle.
- **Sampling that thread** (300 suspend-and-read samples, symbols from
  `victory.pdb`) found 72% of samples in `HostHeap::sync_slots`, called by
  `Heap::sync_direct_copies` from `State::bind`.
  - The title's view heap holds 1,245,184 descriptors and is not
    write-watched memory, so `watch::writes(.., false)` returns `None`.
  - Every bind then compares every slot.
  - Another 6% was `ZwQueryVirtualMemory`, from `by_allocation` failing on
    each call.
- **Change tried:** `descriptor.rs` `sync_direct_copies` asks
  `watch::writes(start, end, true)`, which takes the guarded path that
  shadows already use for mapped views.
  - Result: a flat **30.0 fps (33.3 ms)**, the title's own cap in this phase.
  - The hot thread fell from ~4.3 s to ~0.4 s of CPU per 5 s.
  - The warnings are the same as before the change.
  - `victory-gpu` tests: 255 pass. The mesh-library test fails because this
    card has no mesh shaders. `watch::tests::a_range_never_committed_has_no_writes`
    failed once in three full runs and passes alone.
- **The world never runs on this card: video memory runs out.** This is from
  runs 3-5. I added diagnostics-only logging (`log_video_memory` and
  `VRAM_MADE` in `host.rs`, noted in `resource.rs` `host_texture` and
  `Host::default_buffer`) that reports DXGI local and non-local usage
  against budget every 5 s.
  - Before any input: 4.8 GB of video memory, of which 3.4 GB is render
    targets and 1.4 GB textures. Shared memory is at 1.8 GB.
  - Through the menus, video memory goes over its 7.1-7.3 GB budget (8.7 GB
    used), and Windows pages it out. That matches dips to 2.8 fps.
  - At world load: 11,983 MB used against a 7,187 MB budget. That is 3,361 MB
    of render targets plus 8,601 MB of textures plus 16 MB of buffers. What
    was made ≈ what is in use, so the title really keeps it alive (released
    host textures are freed).
  - Then tens of thousands of `0x8007000E` failures, for shared buffers,
    upload buffers and textures, and the device is removed (`0x887A0005`)
    within seconds.
  - Shared (non-local) memory never passes 3.5 of 7.4 GB.
- **Consequence:** on an 8 GB card, world frame rate can't be measured or
  improved until video memory use drops by about 5 GB. The research plan's
  CPU items still apply, but the first job on the 1080 is a lower-memory
  mode, for example:
  1. Read-only textures (not render targets, not UAV) above a size get host
     copies without their top mip, which saves about 75% each. Views shift
     MostDetailedMip by one, and uploads skip subresource 0. Explicit-mip
     loads in shaders would read one level off, so make it a switch.
  2. Place the least-used textures in system memory (a custom heap on L0)
     once the local budget is reached; shared memory has about 4 GB spare.
  3. Make render targets smaller by running the title's internal
     resolution lower, if it allows that.
- **Not measured:** in-world fps, mesh fallback counts and copy-back cost.
  All of them wait on the memory problem.
- **Risk of the change:** guarded blocks rely on the access-violation handler
  catching every CPU write to the heap. Kernel writes are handled through
  the existing announcement path. If the title writes descriptors through
  some path the handler doesn't see, a slot goes stale. Watch for missing
  textures or wrong draws compared with an unguarded build.

## Ranked plan

### Tier 1: biggest expected wins

1. **Dirty-page copy-back (upstream `main` has it, we don't).**
   Upstream commits `cd94d353` and `940d32f6`: shaders mark the 64 KiB pages
   they write through views, and copy-back compares only those pages.
   - Our `shadow.rs:232-340` snapshots and compares the whole declared UAV
     range on every submission. The comment at `shadow.rs:228-231` estimates
     about 2.3M page faults a frame. One 174 MB buffer gets compared twice a
     frame for about 2.3 MB of real writes; upstream measured 0.45-0.7 s of a
     1.5-1.7 s frame on it.
   - Both commits are already in our local `upstream/main` ref. They touch 20
     files (emit + gpu), and the fxceless fork reports about 85 conflicts.
     Fork `main` is 319 commits behind upstream, so expect real merge work,
     including against the mesh fallback's `emit` changes.
   - Payoff: high. Effort: medium.

2. **Stop the CPU waiting on the GPU after every submission.**
   `host.rs:1141-1145`: ExecuteCommandLists, Signal, then
   `WaitForSingleObject(INFINITE)`, all under the `submitted` mutex.
   Copy-back and retirement run only after that wait, so CPU prep, GPU work
   and copy-back never overlap (frame ≈ sum, not max).
   - Fix: submit and signal without waiting. Wait only where the title needs
     a result: a fence value or memory it polls, or a range we must copy
     back. That needs a ring of in-flight snapshots in place of the single
     locked one.
   - Coalesce consecutive title submissions into one host submission when no
     wait sits between them (`queue.rs:156-217`, `timeline.rs:141`).
   - Fold the present copy (`host.rs:3393`, its own synchronous submit) into
     the preceding batch.
   - Payoff: high. Effort: medium-high.

3. **Per-draw descriptor sync costs a syscall.**
   `draw.rs:760` calls `sync_direct_copies(true)` for each heap on every
   bind. Each call takes the global journal mutex and does
   `GetWriteWatch(RESET)` over the whole heap (`watch.rs:63-96`). On a miss
   it falls back to `sync_slots` over every slot.
   - Fix: sync once per list epoch, after the title's own descriptor-copy
     calls, or through the guarded fast path.
   - Payoff: high (1-3k draws a frame). Effort: low-medium.

4. **Pipeline wait budget of 500 ms a frame.**
   `pipeline/make.rs:228`: a draw that meets a pipeline still being made
   blocks the recording thread for up to 500 ms per frame.
   - Fix: wait about 0-16 ms for graphics, then skip the draw; the drop path
     already exists (`pipeline.rs:480-506`). Keep a longer wait for compute
     only, because a skipped dispatch corrupts results. xcl caps waits at
     250 ms per pipeline and 2 s per frame and moves the compute wait from
     bind to submit (about 2x cold throughput). rebox and OpenScarlett do
     the same.
   - Also make the budget monotonic (fxceless PR 709).
   - Payoff: high while pipelines are cold. Effort: low.

### Tier 2: solid, cheaper

5. **Shader cache survives rebuilds.** The cache key is the BLAKE3 of the
   whole exe (`shader_cache.rs:56-80`), so every rebuild of `victory.exe`
   throws away all translations. Key on a translator version or crate
   source hash instead.
   - Persist the mesh fallback translation; today it runs again every run
     (`pipeline.rs:206`), and on the 1080 every mesh pipeline takes it.
   - Add a prewarm pass over the saved early-start list.
6. **Pipeline library save.** `save()` serializes the whole library under
   its mutex after every pipeline (`make.rs:441-448`). Save on drain and at
   exit instead.
7. **`bindless_textures` cache misses after every barrier.** The key
   includes `list.state_gen()` (`draw.rs:1139-1143`). It also allocates a
   HashSet and Vecs per draw.
8. **`tables_with_strides` per-draw allocations and locks.**
   - Six Vecs per call, SipHash on every slot, and a global VIEWS lock plus
     a clone per slot (`draw.rs:1730-2070`).
   - Fix: scratch buffers kept in State, FxHash (upstream PR 2102, mergeable,
     low risk), and one lock per table.
9. **Copy-back merge and batching.** Upstream PR 2054 makes the byte merge
   branch-free (0.392 to 0.088 ms/MB, low risk). PR 1124 batches copy-back
   writes; it claims 555 to 133 ms/frame, but conflicts.
10. **Join adjacent title-memory copies** (fxceless PR 705): about 14k kernel
    copy calls a frame of about 480 B each, roughly 19 ms.
11. **Refresh granularity.** Ranges are merged into one min/max span and pull
    in untouched gaps (`host.rs:2475-2485`). `Batch::title_written` flushes
    every refresh and clears the texture-checked set on every `Step::Run`
    (`host.rs:2469-2499`). Unwatchable textures are read and hashed with
    BLAKE3 whole on every check (`texture.rs:439-464`).

### Tier 3: mesh fallback (1080-specific, local only)

12. **Batch parts.** Each part does 2 transitions and 2 PSO switches
    (`draw.rs:1478-1500`, indirect `1388-1418`). One shared scratch buffer
    (`host.rs:2195`) serializes compute and raster.
    - Fix: a ring buffer with a bump offset; run several compute passes,
      then the draws, with one transition pair per batch.
13. **Empty triangles.** `draw_vertices = 3 * max_primitives`
    (`meshbuf.rs:80`): every group draws its maximum even when it emits
    fewer. Have the compute half write a primitive count and draw through
    indirect or compaction.
14. **Indirect path.**
    - Clamp and `mesh_draws` are two passes, followed by a global UAV barrier
      (`indirect.rs:178`); fuse them and narrow the barrier.
    - Every execution records the maximum of 16 parts, empty or not
      (`draw.rs:1295-1312`).
    - Correctness gap: only the first indirect record runs
      (`mesh_draws.hlsl`).
15. Check whether amplification (task) stages are handled at all.

### Tier 4: bigger designs, later

- **Card-side change detection** (fxceless: `VICTORY_LATE_WAIT`,
  `WIDE_BACK`, `CARD_BUFFERS_*`, `RELAY`, PR 688 `CARD_COMPARE`): reported
  488 to 216 ms on an RTX 3070 Ti. GPL-3.0, same lineage as ours. It
  overlaps item 1, so do item 1 first and then compare.
- **`VICTORY_TEXTURE_NO_WAIT`** (fxceless issue 470): remap-lock holds of
  about 400 ms go to 0.
- **Present:** 3 buffers, ALLOW_TEARING, and a frame-latency waitable
  (`host.rs:3036`, `3413`); upstream PR 250 removes the fps cap.
- **Separate compute and copy queues:** everything goes to one DIRECT queue
  (`host.rs:486`). Low payoff on a 1080.
- **Emitted shader cost:**
  - Every loop carries an iteration counter (`lower.rs:304-318`).
  - Waterfall loops add cost.
  - Wave64 runs as two wave32 halves on NVIDIA.
  - Profile these before changing them.
- **Ray tracing:** xcl's proposal 1751 builds host DXR structures in place
  of per-node emulation. Idea only.

## Sources and licences

| Source | Licence | Use |
|---|---|---|
| Marvisak/VIctory (upstream) | ours | commits/PRs: merge or port; read-only, no comments |
| fxceless/victory-fork | GPL-3.0 | same lineage; code portable |
| xcl-dev/xcl | GPL-2.0-only | ideas only |
| puffnfresh/rebox | GPL-2.0 | ideas only |
| OpenCommunityEdition/OpenScarlett | AGPL-3.0 | ideas only |
| brcly/TuxBox | Apache-2.0 | code reusable with attribution |
| LooperSalty/AnyXboxX | GPL-3.0 | VIctory fork; reports 4.8-7.1 fps on an RTX 4070 SUPER with offline precompile |
| pablogonz12/Scarlight | GPL-3.0 | no record/replay benchmark found, despite the description |

Other reported numbers: xcl reports 14 fps on average on an RTX 5090,
mostly limited by cold driver PSO compiles. Scarlight reports title-memory
copies capping the world at 0.2-0.8 fps on an RX 9070 XT.

## Suggested order

1. Measure (step 0) on the 1080, warm cache.
2. Items 3 and 4: small changes, likely visible quickly.
3. Item 1 (port upstream dirty pages) and item 5.
4. Item 2 (async submission), the largest structural change.
5. Mesh fallback items 12-14, guided by the new part counters.

## Low video memory: getting into the world on the GTX 1080 (2026-11-25, later)

Local only, uncommitted. Five read-only research agents (one workflow):
mip drop is what other projects use (upstream issues 737/1226/1302, fork
issue 983); `--console series-s` already exists and shrinks render
targets; textures in system memory only move the out-of-memory point.

What works now:
- `VICTORY_NO_TEXTURES=1` (new, `resource.rs` `host_texture`): read-only
  textures get no host copy; views fall back to null SRVs.
- `--console series-s`: render targets 3.36 GB → 1.93 GB.
- Together: video memory 4.8 → 2.2 GB before the world and about 2.5 GB in
  it (was 12 GB and device removal). No out-of-memory errors.
- `VICTORY_SKIP_PIPELINES=5d113aa6830760b6`: that compute dispatch
  (40x23x1) hung the GPU (`0x887A0006`) at world entry, with or without a
  loop cap (`VICTORY_SHADER_LOOP_LIMIT=4096` did not help).
- `timeline.rs` `committed_recently`: memory waits asked `VirtualQuery` on
  every write notification (63% of the busiest thread). Caching pages for
  1 s took world frames from about 5.1 s to about 1.7 s.

Still open:
- 0.2-0.6 fps in the world (1.7-2.6 s/frame), of which only about 0.5 s is in
  our measured submit path. A whole-process sample showed 97.7% of thread
  samples waiting. Next: `VICTORY_PROFILE=4` (built-in unwinding profiler)
  to see where they wait. DirectStorage reads sometimes queue with idle
  workers (`waiting=274`, `busy_workers=0.45`).
- The window stays black in this configuration, even with the black-screen
  session's structured-SRV fix built in.
- Intermittent failures: another GPU hang about 1 min into the world (run 16);
  an invalid command list (run 14); the title's own fatal error after 9 min
  in its intro (run 13).
- `VICTORY_D3D12_DRED=1` makes the world far slower (per-command notes call
  `VirtualQuery`), so leave it off when measuring.

Later runs (18-20):
- Run 18 (`VICTORY_PROFILE=4`): no thread above 51% of a core in the world.
  So frames are not CPU-bound; threads wait. The title then crashed through
  its own crash reporter (its upload was blocked, as it should be).
- Runs 16, 19 and 20 hung the GPU (`0x887A0006`) at different points: run 20
  before the world, with indirect mesh dispatches skipped (new switch
  `VICTORY_SKIP_INDIRECT_MESH=1` in `draw.rs`). With `VICTORY_D3D12_DRED=1`
  (run 17), 10 minutes passed without a hang, so the hang depends on timing.
  Every run since 7 used `--console series-s` plus no textures, and runs 1-6
  (Series X profile) never hung. Next experiments: Series X profile with no
  textures, and `VICTORY_D3D12_DEBUG=1` to name an invalid command.

Runs 21-27 (mip drop ported, hangs traced):
- Ported the mip-drop branch from a public upstream fork (Xvania7,
  `fix/issue-1226-auto-mip-drop`, GPL-3.0, same lineage), fetched as the local
  ref `refs/remotes/xvania/mipdrop`. Adapted to our lazy `Resource::host()`.
  `VICTORY_TEXTURE_MIP_DROP=2` with `VICTORY_TEXTURE_MIP_DROP_MIN_SIZE=256`:
  world textures 14 GB → 5 GB, video memory about 6.2 of 7.2 GB, no
  out-of-memory errors (run 23: 6 min in the world without failure).
- Series X without textures (run 21) hung too, so hangs are not the Series S
  profile. With textures plus mip drop, run 24 also hung, so they are not
  from dropping textures either.
- With the debug layer (run 25), the removal reason is `DXGI_ERROR_DEVICE_HUNG`:
  a submission ran past Windows' 2-second GPU timeout (TDR). It is not an
  invalid command.
- Per-command GPU timing (run 27, `VICTORY_GPU_TIMING=1` plus DRED): the slowest
  commands are indirect compute dispatches (compute `8bbcbda6008eaa37`,
  `c7feccae4e01e454`, `776d835da5ffa646`, 60-150 ms each) and the indirect
  mesh fallback (14 parts of 5041 groups, up to 153 ms). Lists run up to
  about 0.38 s, and spikes past 2 s cause the hang.
- GPU utilization in the world is 6-48% and CPU is low, yet frames take
  1.2-2.6 s, so most of each frame is latency between submissions, not work.

Next step (user is raising TdrDelay to 10 s and TdrDdiDelay to 20 s, then rebooting):
re-run with
`$env:VICTORY_TEXTURE_MIP_DROP='2'; $env:VICTORY_TEXTURE_MIP_DROP_MIN_SIZE='256'; $env:VICTORY_SKIP_PIPELINES='5d113aa6830760b6'; $env:VICTORY_LOG='info'; target\diag\victory.exe run <exe> --console series-s --execute`
(drop the skip once the timeout is raised, to test whether `5d113…` was just slow).
Then measure world fps and find the latency between submissions (GPU 6-48% busy).
Input: click the window, hold Enter about 0.3 s.

Runs 28-32 (crash fixed, frame time located):
- Debug layer (run 29): two errors from our code.
  1. 86-87 compute shaders refused: `CreateComputeShader: Shader wants wave
     lane count of 64 but maximum wave lane count for device is 32`. The
     translator pins WaveSize(64) for shaders that depend on the console's
     lane layout (`lower.rs` `Lower::new`, ~line 586); NVIDIA has only 32.
     Their dispatches are dropped. A real fix emulates a 64-lane wave as two
     32-lane halves (ticket 40; `docs/research/wave-size-rules.md` on the
     `research/wave-size-rules` branch).
  2. `ExecuteIndirect`: the argument buffer was still in UNORDERED_ACCESS
     state. `bound()`/the clamp sets INDIRECT_ARGUMENT, then `prepare`/`bind`
     moves the same shadow buffer back when the pass's shader also binds it.
     Fixed in `draw.rs`: `INDIRECT_ARGUMENT` is re-asserted for the argument
     and count buffers right before `ExecuteIndirect` (general path and mesh
     fallback). After the fix (runs 30-31): 0 such errors, and 10+ minutes in
     the world with no crash or hang (before: every run died within 1-3 min).
- Run 31 (no debug layer): stable, 0.5-0.6 fps (1.6-2.0 s/frame), video
  memory 6.2 of 7.2 GB.
- Run 32 (`VICTORY_PROFILE=4`, resolved offline with dbghelp; the module
  size must be passed to `SymLoadModuleExW`): the title's render thread
  spends its frame inside its own `ExecuteCommandLists` →
  `queue::execute` → `Timeline::submit` → `Host::execute`, in the NVIDIA
  driver's submission path (208 of 534 samples), the driver waiting (86) and
  our `WaitForSingleObject` after each submission (75), plus copy-back.
  So every host submission is synchronous on the title's thread: plan item 2.
- The title then crashed through its own reporter. At 0.2 fps this may be its
  stall watchdog.

Asynchronous submission (runs 33-34), local, behind `VICTORY_ASYNC_SUBMIT=1`:
- `host.rs`: `Host::execute` is split into `begin` (prepare,
  ExecuteCommandLists, Signal) and `complete` (fence wait, device-removed
  check, copy-back, readbacks, the writes that tell the title its work is done,
  epoch release). `execute` = begin + complete (unchanged behaviour).
  `execute_async` hands `complete` to a "victory gpu completion" thread. A
  `busy` flag + condvar keeps one submission in flight (the next `begin`
  waits), because snapshots are per host copy, not per submission.
- `timeline.rs`: new `Op::RunAsync` gets a resume callback. The queue's later
  ops (fence signals, waits) are held until the completion thread resumes it,
  so the title never sees its fence before copy-back.
- `queue.rs`: the title's ExecuteCommandLists uses `Op::RunAsync` only with
  `VICTORY_ASYNC_SUBMIT=1`. The default stays synchronous; tests always use
  the synchronous path.
- Result: with it on, the title crashes through its own reporter about a
  minute into loading, in both runs. A first version that let fence signals run
  before completion crashed the same way, and holding them back did not fix
  it. Suspected next: work the title records while the GPU still runs the
  previous submission writes shadow buffers (or descriptor slots) the GPU is
  reading. Before, recording always happened with the GPU idle. Needs an
  audit of record-time writes (`draw.rs` refresh calls) before async can be
  on by default.
- Tests: `victory-gpu` 258-260 pass. The known mesh-library failure (no mesh
  shaders on this card) remains, and two `draw::depth_tests` now fail in about
  half of the full parallel runs (pass alone; a retired object still pinned by
  another test's epoch). Not yet checked whether this came from the async
  split or the earlier mip-drop port.

## State at end of 2026-11-25 (runs 33-45)

Stable configuration on the GTX 1080 (picture, loading, intro credits, world,
10+ min without a crash):
`$env:VICTORY_TEXTURE_MIP_DROP='2'; $env:VICTORY_TEXTURE_MIP_DROP_MIN_SIZE='256'; $env:VICTORY_SKIP_PIPELINES='5d113aa6830760b6'; target\diag\victory.exe run <exe> --console series-s --execute`
plus, optionally, `$env:VICTORY_ASYNC_SUBMIT='1'` set first.

- Black screen: fixed by the black-screen session (Series S presents two
  planes; `planes.rs` + `shaders/plane_compose.hlsl` blend them).
- Async submission (subagent): `VICTORY_ASYNC_SUBMIT=1` now leaves
  completion pending for the title's next queue call, `GetCompletedValue`,
  `SetEventOnCompletion`, or a fallback thread. Bisection (modes 2-7 kept,
  see `host.rs` `async_mode`) showed that the loading crash needed the next
  `begin` or woken work to run on the completion thread. With it on: stable,
  0.3-0.7 fps (the default path is about 0.4).
- 64-lane shaders (subagent): `victory-shader` `emit/lower/split.rs` runs
  64-thread wave64 groups as two WaveSize(32) halves (11 hardware tests pass).
  With it on, the debug layer shows 0 "wave lane count of 64" errors, but the
  title then reads -1 from GPU data and crashes during loading (runs 42, 43).
  Gated behind `VICTORY_WAVE64_SPLIT=1` (off by default) while the subagent
  adds a per-shader bisect filter.
- Frame time (run 45, async on): 0.7 fps, 1.4 s/frame; GPU 8-16% busy; no
  thread above 18% CPU. The busiest threads are title workers parked in waits
  and NVIDIA driver threads. The remaining gap to 15 fps (20-50x) is latency
  between title threads, our timelines and submissions, not raw CPU or GPU
  work. Next: trace which waits a frame's critical path crosses (title
  fence and memory waits vs. our queue timelines), ideally with
  per-queue timeline instrumentation.

Wave64 split bisection (runs 46-49, `VICTORY_WAVE64_SPLIT=1`, after the
subagent's four correctness fixes):
- all shaders (46): the title reads -1 (`0xc0000005 ... read 0xffffffffffffffff`)
  and crashes during loading.
- `VICTORY_WAVE64_SPLIT_ONLY=0-7` (47): the same -1 read.
- `0-3` (48) and `4-7` (49): the GPU hangs (device removed) instead.
So the failures are spread across many split shaders, not one: probably the
two halves falling out of step (a barrier one half never reaches), or chains
where only some passes run (groups of more than 64 threads are still refused).
Next for the split: GPU timing with DRED on a run with `0-3` to name the hung
dispatch; check every barrier in `emit/lower/split.rs` is reached by both
halves on every path. It stays off by default.

Frame-time breakdown (run 50, async on, world):
- New diagnostics line "queues per frame" (`timeline.rs` `HELD_US`, logged
  next to the fps line): per frame, summed over the title's queues, about
  1.6-7.9 s held behind GPU memory waits, 1.4-3.0 s behind submissions still
  finishing, 0 behind fences, 20-50 ms running work.
- copystat: about 25 host submissions per frame, each finished before the
  next starts (one in flight). At 1.5-3.5 s/frame that is 60-140 ms per
  submission, while the GPU is busy only about 12 ms of each.
- So the frame is a chain: a queue's GPU memory wait is satisfied only after
  another queue's submission completes (fence wait, copy-back, done-writes),
  25 times a frame. On the console these waits stay on the GPU.
- Run 51 adds the time inside the driver's ExecuteCommandLists to that line
  (the earlier profile put much of the render thread there); it is not yet
  measured in the world.
- Directions:
  - Let several submissions be in flight, which needs per-submission
    snapshots.
  - Satisfy a memory wait from a done-write that an earlier submission on
    the same in-order host queue will make. Its value is known when the
    submission begins, and the host queue runs in order.
  - Merge chains of small submissions.

Runs 52-53:
- The driver's ExecuteCommandLists takes only 2-5 ms per frame (run 52). Work
  measured per frame (preparing + GPU + copy-back) is 650-950 ms of a
  1.4-3.0 s frame; the rest are gaps between submissions.
- Gap source found: GPU memory waits satisfied by shader output (copy-back)
  were only noticed by the fence watcher's 10 ms poll. Fixes, unmeasured:
  - `complete_writes` now calls `crate::fence::written(0, 0)` after the
    done-writes, so satisfied waiters wake at once;
  - `fence.rs` `WATCH_POLL` 10 ms → 1 ms.
  Run 53 ended silently before the world (no error and no crash record, like
  run 50; someone may have closed it), so the effect is still unmeasured.

Runs 54-56 (wake fixes):
- Run 54: a command list failed to close at world entry (also seen in runs 24
  and 41, so intermittent and not new). Run 55 with the debug layer showed no
  invalid command (still 107 wave-64 refusals, split off), crawled at 0.1 fps
  and ended.
- Run 56: about 1 fps (1.0-1.4 s/frame) in a light scene (300-500
  draws/frame). Queues are still held about 0.7-1.0 s/frame behind memory
  waits and 0.8-1.0 s behind finishing submissions (summed over queues).
  Then the title's fatal error.
- Conclusion so far: the wake latency was a small part. The chain itself
  (about 25-34 submissions a frame, one in flight, each needing CPU prep,
  GPU, copy-back and done-writes before the next queue's wait clears) is the
  wall. Getting near 15 fps needs several submissions in flight
  (per-submission snapshots), or far fewer host round trips per frame.

Note (user, 2026-11-25): the D: drive (cache, saves and logs) disconnected and
reconnected during these runs. The silent exits with no error (runs 50, 53)
and the Explorer windows that appeared match that. Some unexplained crashes
may have the same cause; treat single crashes with care.

In-flight submissions (subagent; `VICTORY_IN_FLIGHT=<n>`, runs 57-59):
- Built behind the switch, with per-submission snapshots, completions in
  order, early-met waits, and in-place indirect-argument ranges protected.
- Run 59 (n=4): world at 0.5-1.0 fps with 1,100-1,400 draws/frame; 33
  submissions/frame, 20-27 begun behind others (about 1.9 in flight).
  10-14 waits/frame met early by done-writes, 0 by shader writes. Queues still
  held 3.7-4.3 s/frame behind memory waits (summed over queues, so partly
  parallel waiting). Device removed (`0x887A0006`) about 1 min into the world;
  run 57 also lost the device. Not better than the default path yet.
- Next: find what produces the values the remaining memory waits wait for
  (title CPU writes? not-yet-submitted work on another queue?). Log per wait
  which address and the time until it was satisfied, and by what.

Subagent round 3 (2026-11-25 evening):
- Title self-crash: its watchdog kills the process when one main-loop pass
  takes over 30 s (scene transitions with heavy first-use translation and
  pipeline work). `VICTORY_NO_WATCHDOG=1` (loader hook, existing) disarms it;
  now used in all test runs. Other self-crashes: an object pool running out
  (int3 at two sites), bad GPU data (-1 family), one in our own profiler
  (`profile.rs` stack walker; leave VICTORY_PROFILE off).
- Dropped work: about 700 of roughly 900 dropped dispatches/frame are the
  console's own ray-tracing BVH builder (`XboxBvhBuilder.dll`). Three gaps
  fixed: spill-table loads, narrow buffer stores, indexed root views. But the
  three BVH shaders that now translate (`3b90165a77d95cdf`, `caa78b8ca62bc3d6`,
  `5552dac6a2b7a1fd`) hang the GPU at world entry (run 60). With them skipped
  (run 63) the world is stable again at about 0.7 fps. They are added to the
  default skip list in test runs.
- Wave64 split: multi-wave groups (128-1024 threads) are now supported, with
  26 hardware tests, no bug found by audit. In the world (run 64) the title
  still reads -1. It stays off.

Runs 65-71:
- With the current tree, world entry fails in 4 of 5 runs (65, 66, 68-70:
  device hung or a command list that fails to close). The three new
  translator changes are off by default now (`VICTORY_SHADER_SPILL_FIX`,
  `VICTORY_SHADER_ROOT_INDEX`, `VICTORY_SHADER_NARROW_STORE`), and that
  didn't help. Run 67 passed once.
- The 16:31 build (saved as `D:\victory-data\bin\victory.exe`) reached the world
  normally (run 71). So something added to the tree since then made world
  entry unstable. Suspects: the split agent's `lower.rs` hooks (they may touch
  non-split shaders), the in-flight agent's default-path changes
  (`refresh_in_place` at the indirect clamp sites), and the wait-stats or
  wake changes.
- What renders in the world (run 71 screenshot): only the interface plane.
  Cutscene subtitles show; the 3D game plane is black.

World-entry regression (runs 72-73): the two wake changes (immediate
waiter scan in `complete_writes`, 1 ms watcher poll) were the only default-path
changes since 16:31 (the regression subagent checked every other change for
neutrality). With both reverted (run 73) world entry is normal again. They are
now opt-in through `VICTORY_FAST_WAKE=1`; the default is the old 10 ms watcher
with no completion scan.

Black 3D world (run 72): the console's FSR4 upscaler compute shaders come in
containers with no DXIL part. The machine code is in an `ACCC` part
(`SFI0 ISG1 OSG1 PSV0 RTS0 HASH ACCC`). They are refused, so the 1280x720 scene
never reaches the 1920x1080 game plane. A subagent is adding ACCC parsing.

## Upscaler shaders (ACCC containers)

- The 3D game plane stayed black because the upscaler DLL's 29 programs (ACCC containers, no DXIL part) were refused.
- Now all 29 emit and validate under `victory translate --validate`. The DXIL of the other 25,247 shaders is byte-identical.
- Support added for:
  - descriptors made raw after `s_load`;
  - `v_pk_fmac_f16`;
  - `v_dot4c_i32_i8`, `v_dot4_i32_i8` and `v_dot4_u32_u8`;
  - scc/vccz/execz read as data;
  - the SCC of a mask op as a wave-any;
  - trailing readlanes in wave sections;
  - partial image stores (read-modify-write, a shortcut);
  - constant-built samplers;
  - spill readlane through the writelane chain.
- 5 of the 29 are wave64. They also validate on the 32-lane path. Their spin loops poll other groups' memory, and whether that makes progress on the host is unverified.
- Not yet checked in a world run. Run 74 died out of memory at the start screen, with another app holding about 25 GB of RAM.

## Refused shaders in the title directory

- Refused programs went from 1,335 to 1,109. The 226 new ones are 163 mesh, 45 pixel, 15 vertex and 3 compute, and all validate.
- The 25,253 shaders that emitted before are byte-identical. The new fixes run only as a last chance on programs that were otherwise refused.
- The fixes:
  - a dead lane-mask edge joined with data reads as undefined (a shortcut, unproven for every program);
  - mesh programs may read mask counts and compares;
  - a descriptor size read as data falls back to a raw view.
- Biggest remaining reasons:
  - computed barycentrics (207 ps), which need per-vertex attributes;
  - structured buffers of unknown stride (204, likely offline-only);
  - ES-stage addresses (195);
  - mesh output above the host size limit (71);
  - descriptors chosen by control flow (40);
  - first-lane descriptors from a material loop (37 ms).
- Second round: refused 1,109 to 984 (125 more emit, all validate, old output byte-identical).
  - Fixed:
    - lanes holding float data where a descriptor is expected (they sample heap index 0, a shortcut);
    - samplers chosen by control flow (sampled once per sampler, then selected);
    - cube gradient samples;
    - 8 of the first-half interpolation cases.
  - "Descriptor read only as data" rose to 30.

## Wave64 split in the world (runs 86-108)

- The "Unsignaled CrossQueueBarrier" line in the title's crash context is
  routine: it is in crash logs of unrelated faults from before the split too.
  The split crashes were the title reading -1 or empty bounds (+inf/-inf)
  from GPU data.
- Bisection with `VICTORY_WAVE64_SPLIT_ONLY` (now cached per shader, so a new
  filter re-translates nothing already seen; `VICTORY_KEYS` presses Enter
  without the window) named `28918b7150a7aad5`, a pass of the console's BVH
  builder (`XboxBvhBuilder.dll`). Most of the builder's passes do not
  translate and three hang the GPU, so with the split four of its wave64
  passes ran alone and left half-built structures the title reads back.
  `emit::split_takes` now leaves those four out (they stay refused, as
  without the split). No crash of that kind since (runs 97, 98, 108).
- With the split, 1 pipeline is refused instead of about 100, and the 3D
  plane in the intro is solid white instead of black (UI and credits draw over
  it). Probably exposure or an upscaler input; not yet looked at.
- Still open:
  - with every split program ready (warm shader cache), the GPU hangs
    (`DXGI_ERROR_DEVICE_HUNG`) at the start of loading (runs 99, 101, 102, 104).
    Neither half of the `8-f` programs alone hangs (106, 107); with the debug
    layer or DRED it does not happen, so it looks like a long submission or a
    pair spin (`SPIN_LIMIT`) rather than an invalid command.
  - an int3 at the same site as runs 13, 52 and 67 (no split) about 8 min of
    game time into the intro (runs 97, 98): the pool crash, not split-specific.
  - run 108 ended when the D: drive dropped (Ntfs "delayed write failed" on
    its log).

Hang at the start of loading (runs 109-130):
- It is not a GPU timeout. Each time the NVIDIA driver logs "Graphics TEX
  Exception ... TEX LAYOUT" (System log, nvlddmkm event 13): a texture
  instruction met a descriptor of the wrong kind, and the device hangs.
  `VICTORY_GPU_TIMING` shows no list above 340 ms (the slowest commands are
  the upscaler's ML passes, 30-70 ms each).
- With the two 1024-thread shadow classification programs
  (`b8f3f7979d6bbc50`, `eae3498e6553afa8`) skipped, loading works (111,
  112). `emit::split_takes` now leaves them out (`SPLIT_LEFT_OUT_TILES`),
  so they are refused as without the split. After that, warm runs 128,
  129 and 130 got through loading. The cold run 127 still faulted once, so
  another producer or consumer of the tiled-shadow lists may be wrong too.
- Also changed, but these did not fix the hang: one lane polls in
  `pair_exchange_waves`; `SPIN_LIMIT` is 2^30, so halves never give up on
  a slow partner; a run of plain LDS writes in one console block gets one
  pair sync (one classification program went from 430 exchanges to 19).
- Run 131: the 3D scene draws in the intro (sky, silhouettes, a building
  with credits over it), with odd colors and dark halos.
- Runs 131 and 132 (and 97, 98) end about 8-10 min into the intro with
  the int3 at +0x5149aec (the pool crash also seen in run 67 without the
  split). With the split it now happens every time, so a split program
  may feed a count the title allocates from; the next bisection target.

Intro pool crash bisection (runs 133-145, `VICTORY_WAVE64_SPLIT_ONLY`):
- Each run needs about 10 min to show the crash, and the GPU faults at the
  start of loading come and go between runs of the same filter (more on
  the first run of a new filter, when pipelines are still being made and
  their dispatches dropped while the rest of a chain runs). So results are
  noisy.
- The int3 at +0x5149aec came back with half the split programs (133, 63),
  a quarter (135, 32), and with every single-wave program and no
  multi-wave one (144, 114 programs). So it is not one program and not
  the multi-wave exchanges.
- `02384a60fda6f2d0` (a VFX heap defragmenter, two waves walking the heap)
  split on its own makes the GPU fault at the start of loading (138). It
  is on a new left-out list, `SPLIT_LEFT_OUT_WRONG`. Its 15 neighbours in
  that bisection step faulted too (139), so it is not the only one.
- Another quarter (137) crashed on a -1 pointer instead. The full set
  with every pipeline waited for (143) crashed on a -1 pointer at a third
  site.
- New diagnostics switch `VICTORY_PIPELINE_WAIT_MS`: the per-frame budget
  for waiting on pipelines still being made (default 500 ms). A large
  value makes every first use wait, so no dispatch of a chain is dropped
  while the rest of it runs. Loading is much slower with it on a cold
  cache, because some programs take minutes to translate.

## After merging main (2026-11-27)

Three things described above no longer exist in the tree; the notes are kept
as a record of what was tried.

- `planes.rs` and `shaders/plane_compose.hlsl` are gone. Main's present
  overlay shows the second plane (`VICTORY_PRESENT_TWO_PLANES=1`, opt-in).
- `VICTORY_IN_FLIGHT` and this branch's asynchronous submission are gone.
  Main's relay path is the default, and `VICTORY_ASYNC_SUBMIT` is now main's
  switch of that name.
- The in-place indirect clamp is replaced by main's clamp into a scratch
  buffer, which leaves title memory alone.

Kept and carried onto main's code: the three-way copy-back rule (the CPU
wins when title memory changed since the submission's snapshot;
`VICTORY_COPY_BACK_GPU_WINS=1` restores the old rule) on the copy-back that
runs with a submission, and `VICTORY_TEXTURE_MIP_DROP` on top of main's mip
pressure mechanism.

First runs of the merged tree on the GTX 1080 (runs 193-198), three faults,
all from the two sides meeting:

- Loading made 16 GB of host textures and ran out of memory (193). Main
  exempts textures the title places in its own memory from giving up mip
  levels under pressure, and `VICTORY_TEXTURE_MIP_DROP` went through the
  same gate; the title's streamed textures are placed ones. The switch now
  reaches them (196: 4 GB, as before the merge).
- World frames took 2 to 6 minutes (196). Main remakes a streamed texture's
  view when more of its levels arrive; with levels dropped, the check
  compared the title's level numbers with the host copy's and remade every
  such view at every draw. Both sides now use the host copy's numbers
  (197: about 1 s a frame).
- The title stopped on a bad pointer at frame 2090 (197), with 4 to 97
  copy-back conflicts every 5 s where run 192 had 3 in all. On main's late
  paths (relay, card copies, lazy readback) the three-way rule finds CPU
  writes all the time, and a pointer can keep a half from each side. With
  the old rule (198) the world ran 15 minutes. The rule is now opt-in
  there (`VICTORY_COPY_BACK_LATE_RULE=1`).
