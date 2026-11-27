// The console's second display plane blended over the first, as its display
// hardware does: both planes stretched to the output (shortcut: their
// source rectangles and placements are not read), the overlay taken as
// premultiplied alpha.
#define RS "DescriptorTable(SRV(t0, numDescriptors=2), UAV(u0)), " \
    "RootConstants(num32BitConstants=2, b0), " \
    "StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, " \
    "addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP)"

Texture2D<float4> base : register(t0);
Texture2D<float4> overlay : register(t1);
RWTexture2D<float4> output : register(u0);
SamplerState linear_clamp : register(s0);
cbuffer Size : register(b0) {
    uint width;
    uint height;
};

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height) {
        return;
    }
    float2 uv = (float2(id.xy) + 0.5) / float2(width, height);
    float4 b = base.SampleLevel(linear_clamp, uv, 0);
    float4 o = overlay.SampleLevel(linear_clamp, uv, 0);
    output[id.xy] = float4(o.rgb + b.rgb * (1.0 - o.a), 1.0);
}
