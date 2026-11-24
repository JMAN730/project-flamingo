# Which fixes from the parent project's open pull requests main already carries

Surveyed on 2026-11-24 against `main` at `39c58e0e`. `main` moved four times
while this was written (from `93f2c1fa`); every status below is for `39c58e0e`.

## Answer

Of the 25 fixes the ticket lists:

| Status | Count | Fixes |
| --- | --- | --- |
| On main | 6 | root SRVs as bounded one-view tables; mesh pipelines from the pipeline library; copy-back without a query per copy; the opt-in watchdog hook; byte and halfword local-data-share access; 64-bit image atomics |
| Partly on main | 5 | Agility SDK runtime on Windows 10; adapter choice; flat byte-offset accesses to structured buffers; buffer views off their host copy's elements; descriptors read only as data |
| In flight only | 6 | mirroring descriptor slots the title copies itself; spin-wait loop limit; wave64 compute on 32-lane hosts; dynamic descriptor range on a capped heap; rectangle lists placed by vertex number; console flow for groups of several console waves |
| Absent | 8 | null descriptors of the kind the shader reads; caching the error of a container that does not translate; refusing wave64 programs without 64-lane waves; texture mip drop; segregating single-slot and table dynamic descriptors; depth targets with unordered access through a twin texture; BVH intersection instructions; asynchronous submissions |

Five of the six "in flight" fixes are in flight only inside one open pull
request, "Arena/59fabb00 victory": a branch of 188 commits from another fork
that GitHub reports as conflicting with `main`. They are not close to landing.
The sixth (rectangle lists) is in a small open pull request of its own.

## Method

1. Each fix was found among the parent project's open pull requests by title
   and its diff read (`gh pr diff`, read-only). One fix (mirroring descriptor
   slots) has no pull request of its own there; it is one commit inside a
   larger pull request about running under Proton, and that commit's diff was
   read instead.
2. For each diff, every added line of 28 or more non-blank characters was
   looked for, whitespace removed, in the same file on `main` and on the head
   of every open pull request on this fork; the names of added functions,
   constants, tests and environment switches were looked for across the tree.
   Percentages below are the share of such lines found.
3. A low percentage does not prove absence (ports and `cargo fmt` change
   lines), so every fix that was not a near-complete match was then read by
   hand in the code on `main`: the evidence column names the file and what it
   does today.
4. Commits were located with `git log -S` on a distinctive added name.

Nothing was built or run. Statuses say what the source contains, not whether
it works on the reference machine.

Open pull requests on this fork when surveyed: "Arena/59fabb00 victory",
"GPU: compact fresh descriptor tables, dedupe by content, and fix sampler
stride", "GPU: retire freed host resources by active list and submission
epochs", "GPU: support typeless default views and uncapped bindless texture
transitions", "Keep the exact host pipeline bypass in diagnostics builds",
"GPU: draw indexed and indirect rectangle lists with a geometry stage", and
"Add pinned Nix development environment and improve Linux tooling". Only the
first and the sixth carry any fix on the list.

## Findings

"Arena" below means the open pull request "Arena/59fabb00 victory". Short
SHAs are commits in this fork: on `main` unless the row says they are on a
pull request's branch.

| # | Fix | Status | Evidence | Note |
| --- | --- | --- | --- | --- |
| 1 | Loading the Agility SDK runtime on Windows 10 | Partly | `b5b21f32`; `crates/victory-shader/build.rs`, `crates/victory-shader/src/agility.rs` | Only the shader crate's test binaries opt in, and only when `D3D12Core.dll` already sits in `target/<profile>/D3D12/`. `apps/victory/build.rs` exports nothing, so `victory.exe` still gets the in-box runtime. No code fetches the runtime. The four parent variants that cover the application (exports from the executable, an app-local runtime when present, the configuration-interface route, a downloader in `xtask`) match 3-8% and are absent. |
| 2 | Choosing an adapter that can run the title | Partly | `bd64dd9d`; `pick_hardware_adapter` in `crates/victory-gpu/src/host.rs` | A different, smaller change: adapters are tried in high-performance order, software ones skipped, `VICTORY_ADAPTER` filters by name, and the first that creates a feature level 12.0 device wins. It does not test shader model 6.6 or mesh shaders, and does not name the adapters it turned down. The parent's capability-ranked choice (0% of its new file) is absent. |
| 3 | Root SRVs bound as one-view tables bounded to their host copy | On main | `d866dd9e`, `9801045b` (merge `37ca4d1f`) | 89% of lines; the unmatched ones are in heap-sizing code `main` had already changed. `VICTORY_ROOT_SRV_DIRECT=1` goes back. Merged after the ticket was written. |
| 4 | Mirroring descriptor slots the title copies inside its own heap memory | In flight | Arena, `5785b6df` | Absent on `main`: `crates/victory-gpu/src/descriptor.rs` and `queue.rs` have no resync of title-written slots at submission. 79% of the parent commit's lines are on the Arena branch. |
| 5 | The spin-wait loop limit in release builds | In flight | Arena, `b7e8d83c` (`emit/spin.rs`) | On `main` there is one limit, 2^20 (`LOOP_LIMIT` in `crates/victory-shader/src/emit/lower.rs`), no spin-wait detection, and `VICTORY_SHADER_LOOP_LIMIT` is read only in diagnostics builds (`loop_limit` in `emit/mod.rs`). The parent's separate one-change fix that makes the switch work in every build is also absent. |
| 6 | Null descriptors of the kind the shader reads for table slots without a view | Absent | `null_view` in `crates/victory-gpu/src/descriptor/host.rs`, marked `// shortcut:` | A slot with no view still gets a null 2D shader view. `main` does type a null from the title's own view description when a view exists but cannot be made (`null_for`, `null_like`; `cbb50b80`), which is a different case. The parent's sibling change (typed nulls plus a depth scratch stand-in) is absent too. |
| 7 | Splitting flat byte-offset accesses to structured buffers | Partly; the rest in flight | `0b9b629c` on `main`; Arena, `69c6a65b` | `main` splits only scalar loads from a listed structured SRV in a table (this fork's own change). Scalar loads from unordered views, and vector loads, stores and atomics without an index, still go through a raw view (`crates/victory-shader/src/emit/lower.rs`). The parent's change covers those and is on the Arena branch. |
| 8 | Buffer views off their host copy's elements | Partly | `8cf8d2a7` on `main` (`UnalignedCopy` in `crates/victory-gpu/src/descriptor/host.rs`); Arena, `c2a13f65` | `main` has this fork's own fix: a shader view off an element gets an upload copy of its bytes, up to 1 MiB, refreshed at submission. Larger views and every unordered view off an element are still left null (the two `once!` warnings). Neither parent variant (a raw view of the same bytes; a copy of just the view's bytes, unordered views included) is on `main`. |
| 9 | Mesh pipelines loading from the pipeline library | On main | `e411b710` | 98% of lines. |
| 10 | Caching the error of a container that does not translate | Absent | `crates/victory-gpu/src/shader_cache.rs`; its test says "Failures are not stored" | 9% of lines. A container that does not translate is translated again on every run. |
| 11 | Wave64 compute on 32-lane hosts (each approach) | In flight | see rows 11a-11e | Nothing on `main` runs a wave64 compute program with more than 32 threads on a 32-lane host. |
| 11a | ...by emulating the wave in group-shared memory | Absent | none | 8% of lines on `main`, no better on any branch. The Arena branch added an emulation of this kind (`11f3dd86`) and later took it back (`4061790d`) in favour of the console flow. |
| 11b | ...in halves and in the console flow | In flight | Arena, `4b2e982d` | 7% on `main`, 76% on Arena. |
| 11c | ...one console wave sharing local data share in the console flow | In flight | Arena, `4b2e982d` | 9% on `main`, 85% on Arena. |
| 11d | ...barriers in place of lockstep for one wave, 32-lane swizzle and DPP at wave size 32 | Absent | none | 6% everywhere. `main` has a narrower change of this fork's own in the same spirit: wave size 32 for whole-wave reductions of 32 threads or fewer (`013e0ae7`, `54be6ebd`, `722cebcf`). |
| 11e | ...translating for the host's wave sizes | In flight | Arena (95% of lines) | The base the console-flow changes stack on. |
| 12 | Refusing wave64 programs on hosts without 64-lane waves | Absent | `crates/victory-gpu/src/pipeline.rs` | Only the opt-in diagnostics switch `VICTORY_SKIP_WAVE64` is on `main` (`957e2784`, a sibling change, 100% of lines). No code compares a program's wave size with the host's lane range. |
| 13 | Texture mip drop under memory pressure | Absent | none | Both parent variants (automatic on out-of-memory; a switch with a default for small cards) match 3-15%; no mip-drop name exists on `main` or any branch. |
| 14 | Segregating single-slot and table dynamic descriptors | Absent | `HostHeap::dynamic` in `crates/victory-gpu/src/descriptor/host.rs` | One pool with one rotating cursor. 3% of lines. |
| 15 | Preserving the dynamic descriptor range when the view heap is capped | In flight, as a different implementation | Arena, `ac62f72d` and `ce44ed6c` (`view_heap_sizes`) | On `main` a capped view heap keeps 2^16 dynamic slots, not 2^18 (`HostHeap::new`, the `PORTABLE_VIEWS - MIN_DYNAMIC` arm). The parent has two variants; neither is here by content. The Arena branch reaches the same result its own way. |
| 16 | Rectangle lists placed by vertex number | In flight, as a different implementation | open pull request "GPU: draw indexed and indirect rectangle lists with a geometry stage", `062be5d1` | On `main` rectangles are expanded into vertex buffers, which cannot supply a fourth corner the vertex stage computes from the vertex number; indexed rectangles are not drawn (`crates/victory-gpu/src/draw.rs`). The open pull request gives every rectangle pipeline a geometry stage emitted as DXIL; the parent compiles one from HLSL and only for vertex stages that read no vertex buffer. |
| 17 | Depth targets given unordered access through a twin texture | Absent | `crates/victory-gpu/src/resource.rs`, marked `// shortcut:` | The host copy of a depth target drops the unordered-access flag. |
| 18 | Console flow for groups of several console waves | In flight | Arena, `f12a7cb2` | 8% on `main`, 79% on Arena. Stacks on rows 11b, 11c and 11e. |
| 19 | Emulating the BVH intersection instructions | Absent | `crates/victory-shader/src/gfx10/opcodes.rs` | The two instructions are decoded and named; nothing lowers them. 8% of lines. |
| 20 | Copy-back without a query per copy | On main | `87354e41`; `write_mapped` in `crates/victory-gpu/src/host.rs` | 100% of lines. The parent has a second attempt at the same problem (a direct system call for the write); it is absent and not needed. The parent's two further copy-back changes (writing only the GPU's bytes; batching the writes) are absent. |
| 21 | Asynchronous submissions on a host fence timeline | Absent | `crates/victory-gpu/src/timeline.rs` | 7% of lines. The timeline it builds on is on `main` (from the earlier fence-wait change), but work still runs on the submitting thread unless the queue is held back. The second bulk merge left this one out because it was a draft. |
| 22 | The opt-in watchdog hook | On main | `de1aff9e` | 97% of lines. |
| 23 | Byte and halfword local-data-share access (four attempts) | On main, twice | `fd3f567c`, then `f0a5c345` | Two of the four attempts are on `main`, in the same function; see Surprises. The other two (one with its own test file, one that refuses reads into half a register) are absent. |
| 24 | Descriptors read only as data taking their kind from their table | Partly; the rest in flight | `57c98c77` on `main`; Arena, `be98ae6e` | `main` accepts only eight dwords, taken as a 2D shader view, and refuses the rest with "a descriptor read only as data" (`crates/victory-shader/src/emit/lower.rs`). That narrow case came with the change for twin descriptor loads in compute samples. Taking the kind (unordered or not, buffer stride) from the table is on the Arena branch, reworked. |
| 25 | 64-bit image atomics | On main | `77dd599f` | By a sixth change, not one of the parent's five open attempts (each matches 15-28%, their flag names are all absent). `main` lowers them as 64-bit typed atomics and sets both 64-bit atomic feature flags. |

## Surprises

- **Two implementations of sub-dword local data share live in one function.**
  `fd3f567c` added byte and halfword arms to the read and write paths;
  `f0a5c345` then added a block above them that handles every byte and
  halfword form and returns. The first change's arms are now unreachable, and
  its test exercises the second change's code. The tests the second change's
  parent pull request carries are not on `main`.
- **64-bit image atomics arrived by a side door.** The parent has five open
  attempts under that subject; `main` has none of them and already has the
  behaviour.
- **"The Agility SDK code is in the tree" is true only for tests.** The
  exports are emitted by the shader crate's build script, which applies to
  that crate's own test binaries. The executable is not opted in.
- **Three answers to buffer views off an element.** `main` has this fork's
  upload copy; the Arena branch has the parent's scratch-memory copy; the
  parent's later "copy of just their bytes" (the only one covering unordered
  views) is in neither. Merging the Arena branch would put two of them side by
  side in `descriptor/host.rs`.
- **Two geometry-stage answers to rectangle lists**, the parent's (HLSL,
  compiled at run time, needs a compiler on the machine) and the open pull
  request's (DXIL emitted by our own writer).
- **The Arena branch reverses itself on wave64.** It adds a group-shared
  emulation and later takes it back for the console flow. Anyone
  cherry-picking from it should take the later commits, not the earlier one.
- **The Arena branch is where most in-flight work sits, and it conflicts.**
  188 commits ahead of `main`, 382 behind, reported as conflicting. Five fixes
  on the list are "in flight" only there.
- **Duplicates on the parent that need one pick each:** Agility SDK loading
  (four variants plus the test opt-in), the capped-heap dynamic range (two),
  mip drop (two), copy-back without a query (two), sub-dword local data share
  (four, plus two bundles), descriptors read only as data (two), 64-bit image
  atomics (five).
- **Root SRVs moved during the survey** from an open pull request to `main`.

## Not determined

- Whether a built `victory.exe` exports the Agility SDK symbols. The answer
  above is read from the two build scripts; no binary was inspected.
- Whether any fix works on the reference machine. Nothing was run.
- Whether the Arena branch's reworked versions (rows 15 and 24) behave like
  the parent's. They were compared by reading, not by test.
- The parent's bundle pull requests were not checked one by one; each bundles
  changes that are listed here singly.
