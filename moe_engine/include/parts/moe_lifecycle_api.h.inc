
moe_pc_engine_config_t moe_pc_default_config(void);
moe_io_config_t moe_pc_default_io_config(void);
moe_model_shape_t moe_pc_Moe1_model_shape(void);

moe_pc_engine_t* moe_pc_engine_create(const moe_pc_engine_config_t* config);
void moe_pc_engine_destroy(moe_pc_engine_t* engine);

int moe_pc_detect_backend_caps(
    moe_backend_t preferred_backend,
    moe_platform_t platform,
    moe_backend_caps_t* out_caps
);

int moe_pc_engine_get_backend_caps(const moe_pc_engine_t* engine, moe_backend_caps_t* out_caps);

int moe_pc_engine_get_optimization_plan(
    const moe_pc_engine_t* engine,
    moe_inference_phase_t phase,
    moe_runtime_optimization_plan_t* out_plan
);


