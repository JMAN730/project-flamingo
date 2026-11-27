#define RS "RootConstants(num32BitConstants=7, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint args_offset;
    // Bit 0: a count buffer bounds the records. Bit 1: the records are a
    // window clamped ahead of their draws: one that asks for more than can
    // run is left empty and costs nothing. Bit 2: dispatch records split at
    // 65,535 groups in x (second records written `records` records on).
    // Bit 3: the records are gathered (`gather`): record `index` lies in the
    // source where word `index` of the table behind `count_buf` says (from
    // `count_offset` on), and goes to slot `index`.
    // Bits 8-15: dispatches, host thread groups per console thread group
    // along z (a console group of several waves run as one host group per
    // wave; 0 or 1: as asked). Bit 16: mesh records as rows (`ROWS`):
    // each written as 20 bytes, the row width and last group (the root
    // constants the mesh shader reads), then the groups as rows of at most
    // 65,535.
    uint has_count;
    uint count_offset;
    // The first record this dispatch clamps: the caller dispatches one group
    // per 64 records (see `record_clamp_with`), so a chunk's charges land in
    // lane order rather than in whatever order the groups ran in.
    uint base;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
// Records are bounded one by one to 2^26 each, so 2^31 is the most the sum
// can reach without wrapping.
static const uint BUDGET = 1u << 31;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    uint eff_count = records;
    bool ahead = (has_count & 2u) != 0;
    if ((has_count & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
    // The record this thread clamps: the caller runs one group per 64
    // records, and `base` says which 64.
    uint index = base + gi;
    if (index >= eff_count) return;
    bool rows = (has_count & (1u << 16)) != 0;
    uint src_at = args_offset + index * stride;
    // (A branch, not a select: without bit 3 the view behind `count_buf`
    // is a count of four bytes, or the source itself, and is not read here.)
    [branch]
    if ((has_count & 8u) != 0) {
        src_at = count_buf.Load(count_offset + index * 4u);
    }
    uint dst_at = index * (rows ? 20u : stride);
    uint2 row_words = uint2(0, 0);
    uint cost;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint2 asked = src_args.Load2(src_at);
        if (ahead && (asked.x > (1u << 24) || asked.y > (1u << 24))) {
            asked = uint2(0, 0);
        }
        uint count = min(asked.x, 1u << 24);
        uint instances = min(asked.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw_counts = uint2(count, instances);
    } else if (kind == 2) {
        uint3 asked = src_args.Load3(src_at);
        // Diagnostics, in the budget buffer's words 6 and 7 (after the mesh
        // ones): the largest x a compute record asked for, and the records that
        // asked for more groups in x than one dispatch runs.
        if (asked.x <= (1u << 24) && asked.y <= 64u && asked.z <= 64u) {
            uint noted;
            budget.InterlockedMax(24, asked.x, noted);
            if (asked.x > 65535u) {
                budget.InterlockedAdd(28, 1u, noted);
            }
        }
        uint3 g = min(asked, uint3(65535u, 65535u, 65535u));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        // Split (bit 2): the groups past 65,535 in x, up to 65,535 more, run
        // as a second record (below), charged with the first.
        uint rest = ((has_count & 4u) != 0 && g.x == 65535u) ? min(asked.x - 65535u, 65535u) : 0u;
        if (rest != 0 && g.y * g.z != 0 && g.x + rest > GROUPS / (g.y * g.z)) {
            g = uint3(0, 0, 0);
            rest = 0;
        }
        // One host group per console wave of a group split by the
        // translation: z counts the waves too.
        uint split = max((has_count >> 8) & 0xffu, 1u);
        if (g.z * split > 65535u) {
            g = uint3(0, 0, 0);
            rest = 0;
        }
        g.z *= split;
        cost = (g.x + rest) * g.y * g.z * 64u;
        groups = g;
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        // Diagnostics: records asking for more groups than a dispatch runs
        // (65,535) yet under 2^24, counted in budget word 4, largest in word 5.
        if (g.x > 65535u && g.x < (1u << 24)) {
            uint seen;
            budget.InterlockedAdd(16, 1u, seen);
            budget.InterlockedMax(20, g.x, seen);
        }
        // Ahead of its draw, a record over the most a dispatch runs is an
        // unfilled slot: empty, and not counted below either.
        if (ahead && g.x > (rows ? (1u << 22) : 65535u)) {
            g = uint3(0, 0, 0);
        }
        // Diagnostics, in the budget buffer's next words: the largest
        // group count a mesh record asked for, and the groups asked in all.
        uint was;
        budget.InterlockedMax(4, g.x, was);
        budget.InterlockedAdd(8, min(g.x, 1u << 24), was);
        if (rows && g.x > 65535u && g.x <= (1u << 22) && g.y != 0 && g.z != 0) {
            uint n = (g.x + 65534u) / 65535u;
            uint width = (g.x + n - 1u) / n;
            row_words = uint2(width, g.x - 1u);
            cost = g.x * 128u;
            groups = uint3(width, n, 1u);
        } else {
            g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
            cost = g.x * g.y * g.z * 128u;
            groups = g;
        }
    }
    bool over = false;
    if (cost > 0) {
        // Diagnostics: the largest single record's cost.
        uint largest;
        budget.InterlockedMax(12, cost, largest);
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = (BUDGET - before <= cost) ? BUDGET : (before + cost);
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        over = before >= BUDGET || BUDGET - before < cost;
    }
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
            row_words = uint2(0, 0);
        }
        if (rows) {
            dst_args.Store2(dst_at, row_words);
            dst_args.Store3(dst_at + 8, groups);
        } else {
            // Split: the second record, `records` records on, takes the groups
            // past 65,535 in x (none where the first is not full).
            if (kind == 2 && (has_count & 4u) != 0) {
                uint rest = groups.x == 65535u ? min(src_args.Load(src_at) - 65535u, 65535u) : 0u;
                dst_args.Store3(records * stride + dst_at, uint3(rest, groups.yz));
            }
            dst_args.Store3(dst_at, groups);
            for (uint off = 12; off < stride; off += 4) {
                dst_args.Store(dst_at + off, src_args.Load(src_at + off));
            }
        }
    }
}
