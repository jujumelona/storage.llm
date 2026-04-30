int moe_pc_engine_prepare_moe_activation(moe_pc_engine_t* e, int layer) {
    return prefetch_moe_activation_raw(e, layer);
}
