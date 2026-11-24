# What a lit public tree changes that main lacks

Research for ticket 39. Question: a public third-party tree of the parent
project is reported lit; which of its differences from this fork's `main`
plausibly produce the lit picture? The tree is identified below by the commit
that was read.

Method: read-only fetch of that tree, then reading. Nothing was built or run,
and no code was taken. Every claim below names the file and function it was
read from; "reference" means that tree, "main" means this fork.

## Commits compared

| Side | Commit | Note |
|------|--------|------|
| main | `c1052279` | 456 commits past the common ancestor |
| reference | `7b0f736d` | 6 commits past the common ancestor |
| common ancestor | `ffe310eb` | a parent-project commit both sides contain |

The reference is one large commit (plus five small ones) on top of the
ancestor: 65 files, about 34,900 lines added, 1,400 removed. It has no history
to bisect. So the work was done in two steps: read what the reference changed
since the ancestor, then look in main for an equivalent.

Sources of the claims about what the title does are the reference's own code
comments and its README. They were not checked against the title.

## What the reference itself claims

From its README (status dated 2026-11-25, one machine, one NVIDIA card):

- The opening cutscene and the first gameplay scene render lit by the sun.
- "Everything in shadow is black": its ray-traced bounce light and ray-traced
  shadows do not run. The ray-tracing work is off by default and its own
  write-up says nothing is lit by it yet.
- 2 to 4 frames per second in the world.

So "lit" there means direct light only. The ray-tracing work (about a quarter
of the shader diff) cannot be what lights it.

## Differences by area

Ratings: **likely** (likely needed for light), **possibly**, **unrelated**.

### 1. Depth and stencil textures written through unordered access

Reference: a depth texture the title also asks unordered access for gets a
colour "twin" that shaders write instead, and the two are copied into each
other when one side is used after the other was written.

- `crates/victory-gpu/src/resource.rs`: `Twin`, `twin`, `depth_and_unordered`,
  and the function that builds the twin from the description. A depth format
  with a stencil plane gets two twins: an 8-bit one for the stencil plane
  (copied through a buffer) and a 16-bit or 32-bit float one for the depth
  plane.
- `crates/victory-gpu/src/host.rs`: `HostList::twin_use` (which side is ahead,
  when to copy), `HostList::pass_plane` and `PlanePass` (a pass of the layer's
  own that draws the depth twin into the depth plane, because no colour format
  holds the bits of a 24-bit depth plane), `twins_to_depth` at list close.
- `crates/victory-gpu/src/descriptor/host.rs`: `twin_uav` turns the title's
  unordered access view of the depth texture into a view of the twin.
- `crates/victory-gpu/src/draw.rs`: `readable` calls `twin_use` for shader
  reads and for unordered access; `depth_state` picks the read or write state.
- Switch: `VICTORY_DEPTH_TWINS` (`0`, `stencil`, `nodepth`).

Why it matters, per the reference's comments in `resource.rs` and `host.rs`:
the title's tiled lighting writes each pixel's material class into the depth
plane and masks into the stencil plane from compute, then draws one class per
draw with the depth test "equal". Without the write, no pixel passes.

Main: no equivalent. `host_texture` in `crates/victory-gpu/src/resource.rs`
removes the unordered access flag from every depth texture (its own comment
calls this a shortcut and names the second copy as the fix). `fits` in
`crates/victory-gpu/src/descriptor/host.rs` then refuses the view, the slot
gets a null view, and the writes are dropped.

Rating: **likely**. It is the only difference that, on its own, predicts a
lighting target of exactly zero while the G-buffer fills.

### 2. Rectangle lists

Reference: a pipeline the title makes for rectangles gets a geometry stage
between the translated vertex and pixel shaders. The draw goes out as a
triangle list with the title's own vertex counts, vertex numbers, indices and
indirect arguments; the stage completes each rectangle from the three vertex
shader outputs.

- `crates/victory-shader/src/emit/rect.rs`: `rect_list_gs`. It finds which of
  the three vertices is the corner (the one sharing x with one neighbour and y
  with the other) and puts the fourth opposite it.
- `crates/victory-gpu/src/pipeline/host.rs`: `HostPipeline::rects`.
- `crates/victory-gpu/src/draw.rs`: `completes_rects`; indexed rectangles and
  indirect rectangles are drawn when the pipeline has the stage.

Its comments say the title's full-screen rectangles give the corner as the
second vertex, the tiled lighting gives it as the first, and the tiled
lighting issues 256 indirect rectangle draws a frame whose corners come from
the vertex number.

Main: a different design. `crates/victory-gpu/src/primitive.rs`
(`expand_rects`) and `crates/victory-gpu/src/rectangles.rs` (two compute
passes before `ExecuteIndirect`) expand vertex inputs to six vertices and
compute the fourth as `v0 + v2 - v1`, which assumes the second vertex is the
corner. For a pipeline with no per-vertex inputs, `rectangles.rs` documents
that it draws six vertices per rectangle from vertex 0; a vertex shader that
derives corners from the vertex number, three per rectangle, then gets the
wrong numbers. Indexed rectangles are not drawn at all (`draw.rs`: "indexed
rectangles not drawn", "indirect indexed rectangles not drawn").

Rating: **likely**. If the reference's description of the tiled lighting draws
is right, main draws them with the wrong corner rule and the wrong vertex
numbers even once item 1 is fixed.

### 3. A view placed over a texture the host writes

Reference: when the title places a view, in another format, at the address of
a texture that the host renders to or writes, the placed view's texture is
filled by a host-side copy from that texture instead of from title memory.

- `crates/victory-gpu/src/resource.rs`: `alias_of` and the table of targets by
  address. Covers every mip level and layer, and volumes.
- `crates/victory-gpu/src/texture.rs`: `alias` (copy through a buffer, because
  the host copies between two formats of equal texel width in no other way).
- `crates/victory-gpu/src/draw.rs`: `readable` tries the alias copy before the
  upload from title memory.
- Switch: `VICTORY_ALIAS_COPY` (`0` off, `1` single level and layer only).

Per its comment: the title packs light into 32-bit integer textures from
compute and samples them through shared-exponent views placed at the same
address (sky, sky and light probes, the bounced light the lighting pass adds).
On the host these are two textures; the sampled one was filled from title
memory nobody writes, so it read black.

Main: no equivalent. `texture_at` in `crates/victory-gpu/src/resource.rs`
makes a separate resource per (address, description) and `readable` uploads it
from title memory. Main does have a different fix for a different case:
`target_format` substitutes a 16-bit float format when the title creates a
shared-exponent texture as a target.

Rating: **likely** for sky and indirect terms; not required for a non-zero
direct-light target.

### 4. Console 64-lane compute programs on a 32-lane host

Reference: a large body of work so that wave64 compute programs with
cross-lane code run in two host waves of 32.

- `crates/victory-gpu/src/host.rs` reports the host's widest wave to the
  translator (`set_host_max_wave`).
- `crates/victory-shader/src/emit/mod.rs`: a ladder of retries. Exact forms
  first (the two halves exchange through group-shared memory at group
  barriers; early-outs and skips left out so more code is run by every lane;
  per-lane branches taken wave-wide via `widen` and
  `structure::structure_wide`), then each host wave as a wave of its own.
- `crates/victory-shader/src/emit/own_bits.rs`: proves that a wave-wide OR
  only steers a loop over each lane's own bits, so a half-wave OR is exact. Its
  module comment names the title's light loops as the case.
- `crates/victory-shader/src/emit/lower.rs`: `with_local_waves`, `wave_local`,
  `lds_meetings`, `half_sum`.
- `crates/victory-shader/src/emit/wave.rs`, `wave64_tests.rs`.
- Switches: `VICTORY_WAVE_WIDE`, `VICTORY_WAVE_LOCAL`, `VICTORY_WAVE_OWN_BITS`,
  `VICTORY_WAVE_DEEP_SKIPS`, `VICTORY_GROUP_SYNC`, `VICTORY_HOST_WAVE`.

Main: partial. `translate_with_wave32` in
`crates/victory-shader/src/emit/mod.rs` and the entry code in `emit/lower.rs`
lower a wave64 whole-wave reduction with a 32-lane wave only when the thread
group is exactly 32 threads, uses no group-shared memory and holds no mask.
Every other compute program with cross-lane code declares a 64-lane wave,
which a 32-lane host refuses, so its dispatches do not run.

Mesh programs: the same in both trees. `cross_lane` refuses a cross-lane
operation in a mesh shader on both sides.

Rating: **likely** on 32-lane hosts (the report is about NVIDIA cards),
**unrelated** on a host with 64-lane waves.

### 5. Descriptor mirroring

Several separate changes in `crates/victory-gpu/src/descriptor/host.rs` and
`crates/victory-gpu/src/draw.rs`.

| Change in the reference | Main | Rating |
|---|---|---|
| Per-draw table slots on a capped heap: `DYNAMIC` is 2^18 there too, taken out of the title's slots | 2^18 on an uncapped heap, `MIN_DYNAMIC` 2^16 on a capped one | possibly (capped heaps only) |
| Slots given back when the list's submission has run, not at the title's reset (`draw.rs`, the list's `ran` path) | Held until reset or release | possibly |
| A table that cannot be made is logged ("a table is not made"); no drop of the draw was found | The draw is dropped ("no dynamic slots for its tables") | possibly: on main a starved frame loses its late draws |
| Heap cap told to the translator before the first shader (`probe_view_limit`); indices past the cap read a null slot | Limit fixed at 2^24 (`VIEW_HEAP_LIMIT`), so an index past a capped heap is not clamped | possibly (stability on capped heaps) |
| Buffers a shader picks from the heap by a computed index are made and brought up to date once per list (`State::heap_buffers`, the stride table `SLOT_STRIDES`) | Not done. `crates/victory-gpu/src/shadow.rs` documents it: such buffers "keep what their first use copied" | possibly |
| Textures picked by a computed index are created and uploaded within a budget (`State::heap_textures`, `TextureQueue`, `Spans`) | Not done; a placed texture is uploaded only when a table reads it | possibly (materials more than light) |
| Structured buffer views made raw where possible (`raw_window`); views the host would refuse left null (`buffer_view_ok`) | Different approach: structured aliases and aligned copies for views off their stride | unrelated to the gap |
| The copy a view's slot shows is the copy that is refreshed (`buffer_copy`, `buffer_extent`) | Not checked | possibly |
| Unordered access view of a texture the title placed itself | Not mirrored on either side | - |
| View table bounded (constant buffer and placed views named whole in their records) | Not present | unrelated (memory) |

Descriptors the title copies as bytes inside heap memory: no scan of heap
memory was found in either tree. Both make such a slot from its record when a
draw reads it through a table (`HostHeap::mirror` called from `draw.rs`).

Rating for the group: **possibly**. The slot supply items decide whether late
draws of a frame run at all on a capped heap; the heap-indexed passes decide
whether data read without a table is current.

### 6. Queue waits and cross-queue ordering

Reference: `crates/victory-gpu/src/queue.rs` (`Held`, `split`, `cuts`,
`release_held`) and `crates/victory-gpu/src/list.rs` (`ValueWait`,
`unmet_waits`, `waits_by_list`, `cut`). A queue holds work behind a fence wait;
a list is cut into host lists at its waits for memory values. Its comment in
`list.rs`: the compute queue's lighting passes wait for a value the graphics
queue's depth pass writes; run at once they read a cleared depth buffer.

Main: equivalent present, differently built. `crates/victory-gpu/src/timeline.rs`
(`MemoryWait`, `Timeline`), `fence.rs`, and wait segments in `list.rs`.

One difference in what happens to a wait nothing meets: the reference runs
everything still held at each present; main lets a memory wait time out after
2 seconds (`memory_wait_timeout` in `fence.rs`) and continue.

Rating: **likely** needed, but main has it. Not a gap.

### 7. Target formats the host refuses

Reference: `target_format` in `resource.rs` makes targets in two 16-bit colour
formats as 8-bit RGBA when the host refuses them as a target or for unordered
access. Its comment: the title has full-screen targets in both, and a refused
texture meant every pass writing it drew nothing.

Main: `host_texture` retries without the refused use, so the texture exists
but writes through the refused use are lost.

Rating: **possibly**. Which passes use those targets was not determined.

### 8. Translator coverage

Refusal messages present in the ancestor that the reference removed and main
still has (compared as strings, so main may cover part of a class under the
same message):

- a descriptor chosen by control flow
- a lane mask holding lanes elsewhere read as data
- a root unordered access view or typed root view
- a cube sample with texel coordinates, gradients or a level query
- a sample offset past -8..7
- an interpolation with computed barycentrics; pull-model interpolation
- an output modifier on a 16-bit operation

Named cases in the reference's comments:

- `crates/victory-shader/src/resource.rs`, `code_join`: a table address joined
  from two tables in the code. The comment says the fog reconstruction reads
  its sample offsets this way and that, refused, "the lit frame was multiplied
  by a fog transmittance nothing had written". No equivalent was found in main
  by search. Rating: **possibly**, and it acts after lighting, on the final
  picture.
- `resource.rs`, `only_edge`: a descriptor loaded inside a branch that only
  some lanes run. **possibly**.
- `crates/victory-shader/src/spill.rs`: the spilled parameter table. Main has
  its own (`ir::build_with` with a spill flag). Used by the structure builder
  only. **unrelated**.

Present on both sides, so not a gap:

- A gather at a named mip level, with and without an offset: reference inline
  in `lower.rs`; main `gather_at_level`.
- Implicit-level samples outside pixel shaders: the reference reads level 0;
  main computes derivatives across the quad of lanes, which is closer to the
  console.
- A heap view of no listed kind read as a buffer view, not a constant buffer:
  the reference decides by the stride table, main by how the program uses it.
- Root shader views bound as one-view tables bounded to their copy.
- Texture-to-buffer read-backs.

### 9. Ray tracing

`crates/victory-shader/src/bvh.rs`, `emit/intersect.rs`,
`crates/victory-gpu/src/memory.rs`, and the builder rule in
`crates/victory-gpu/src/pipeline.rs` (`Builder`, `is_builder`, `builder_runs`).
Main decodes the two intersect instructions and lowers neither.

Rating: **unrelated** to the lit picture today. Off by default, and the
reference's own write-up says nothing is lit by it.

### 10. Speed and memory

Late waits and the relay thread (`host.rs`), title memory opened in place
(`inplace.rs`), card-side buffer copies (`shadow.rs`, `copystat.rs`), host
textures made on first use and let go when idle (`resource.rs`), the pipeline
library mapped instead of read (`pipeline/library.rs`), cheaper page queries
(`texture.rs`, `watch.rs`), configurable pipeline waits.

Rating: **unrelated**. One caveat: making host textures on first use changes
when slots become valid, and the reference added `make_waiting` in
`descriptor.rs` to cover that.

### 11. Loader and system layer

`crates/victory-loader/src/console_insn.rs` and `console_insn_decode.rs`
emulate processor instructions an Intel host lacks. Main has its own handling
of the main case in `crates/victory-loader/src/crash.rs`. The HLE changes are
a lock around remapping (`xmem.rs`) and small input, audio and storage edits.

Rating: **unrelated**.

## Ranked shortlist

1. **Colour twins for depth and stencil textures that compute writes** (area
   1). Main drops those writes. Directly gates the depth test the lighting
   draws use. Verify first: does main log "view the host does not allow" with
   the unordered flag for a depth format during a world frame?
2. **Rectangles completed in a geometry stage, including indexed and indirect
   ones with corners from the vertex number** (area 2). Main's expansion
   assumes the wrong corner and renumbers vertices for exactly the kind of
   draw the reference says tiled lighting uses.
3. **Host-side copy for a view placed over a host-written texture** (area 3).
   Without it sky, probes and bounced light read as black.
4. **wave64 compute on 32-lane hosts** (area 4). On NVIDIA, main refuses every
   such program beyond one narrow shape; the reference's comments tie light
   loops to this.
5. **Descriptor slot supply on capped heaps and current data for heap-indexed
   reads** (area 5). Decides whether late draws run and what they read.

Runner-up: the fog reconstruction program (area 8), which the reference says
blacks out an already lit frame.

## Blunt workarounds found

None of these is a hard-coded skip list; no shader hashes or pipeline names
are special-cased by default. They are approximations that may make a picture
appear without being right.

- **Held work forced at each present** (`queue.rs`, `VICTORY_WAIT_FORCE`, on by
  default): a wait nothing meets is dropped at the frame boundary.
- **Loops cut short** (`emit/lower.rs`, `count_iteration`): 128 iterations in
  pixel shaders, 256 in vertex and mesh, 4,096 in compute; a lane that hits
  the limit is killed. Main cuts at 2^20.
- **Approximate wave64 forms as a last resort** (`with_local_waves_any`): each
  half-wave gives its own result, so a tile-wide value becomes two half-tile
  values. The exact barrier form is off by default after it lost the device.
- **8-bit stand-in for 16-bit colour targets**, with copies between a stand-in
  and title-format texels skipped (`unlike` in `list.rs`).
- **Indirect rectangles without the geometry stage** are drawn as their first
  triangle only.
- **Structure builder dispatches dropped** unless every builder pipeline
  exists and a switch is set.
- **Views the host would refuse left null**, and view indices past a capped
  heap sent to a null slot.
- **Implicit-level samples outside pixel shaders read level 0.**
- **The depth plane's twin is drawn at list close**, so a use later in the
  same list sees the old plane; every bound depth target counts as written.
- **Root views without a size** are taken as 64 KiB (constants) or the rest of
  the buffer.
- **Draws omitted while their pipeline is still being made.**
- The workspace lint that asks for a safety comment on each unsafe block is
  switched off.

Also to note before taking anything from it: the instruction emulation in the
loader says in its header that it is ported from a project under GPL-2.0-only,
into a tree under GPL-3.0-or-later.

## What could not be determined by reading

- **Which change lit it.** The reference is one commit with no history. Its
  switches (`VICTORY_DEPTH_TWINS=0`, `VICTORY_ALIAS_COPY=0`,
  `VICTORY_QUEUE_WAIT=0`, the wave switches) would allow an ablation, but
  nothing was run.
- **Whether the report of three NVIDIA cards is accurate.** The tree's README
  names one machine and one card.
- **What hardware main's all-zero lighting target was seen on.** Areas 4 and
  part of 5 only apply to 32-lane hosts and capped heaps.
- **Whether the tiled lighting programs are among the wave64 compute
  programs.** The reference's comments name light loops and a particle light
  atlas, not the lighting pass itself.
- **Whether main's rectangle path really misdraws vertex-number rectangles.**
  It follows from `rectangles.rs`'s own description; it was not observed.
- **Whether main's translator already accepts the fog reconstruction shape**
  under another route. Only a search for an equivalent was done.
- **How much of the remaining refusal classes main covers.** Messages were
  compared as strings.
- **The reference's statements about the title** (what tiled lighting writes
  where, vertex order, packed light formats) come from its comments.
- **Portability.** The reference sits on an ancestor 456 commits behind main,
  and main has since built different designs for waits, rectangles, target
  formats and root views. None of the reference's changes would apply as they
  are; each would have to be written again against main.
