float storage_f8_e4m3_to_f32(uint8_t value);

int storage_f8_e4m3_decode(
    const uint8_t* input,
    uint64_t count,
    float* out_values
);
