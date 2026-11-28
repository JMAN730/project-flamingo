// The arguments of an indirect mesh dispatch on a host without mesh shaders
// (`State::execute_indirect`), which runs it in `parts` parts of at most
// `capacity` thread groups: per part, the draw of its groups (one instance
// each) and the dispatch of the compute pass that fills the mesh buffer for
// it. Parts past the title's group count are empty.
//
// Only the title's first record is run: the records of one execution share
// every binding and number their groups from 0, so more of them would draw
// the first one's geometry again.
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint parts;
    uint capacity;
    uint vertices;
    uint unused;
};
// The title's records, clamped: thread groups in x, then 0 or 1 twice.
RWByteAddressBuffer args : register(u0);
// Ours: `parts` draws (vertices per instance, instances, two zeros), then
// `parts` dispatches.
RWByteAddressBuffer made : register(u1);

[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint3 g = args.Load3(0);
    uint groups = g.y * g.z == 0 ? 0 : g.x;
    for (uint k = 0; k < parts; k++) {
        uint first = k * capacity;
        uint n = groups > first ? min(groups - first, capacity) : 0;
        uint one = n == 0 ? 0 : 1;
        made.Store4(k * 16, uint4(vertices, n, 0, 0));
        made.Store3(parts * 16 + k * 12, uint3(n, one, one));
    }
}
