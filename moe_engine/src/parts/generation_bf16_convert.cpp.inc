// bf16 conversion helpers for KV cache
static inline uint16_t moe_f32_to_bf16(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    // Round-to-nearest-even: add 0x7FFF + bit[16] for tie-breaking
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static inline float moe_bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float v;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}
