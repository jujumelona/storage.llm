
glm5_pc_engine_config_t glm5_pc_default_config(void);
glm5_io_config_t glm5_pc_default_io_config(void);
glm5_model_shape_t glm5_pc_glm51_model_shape(void);

glm5_pc_engine_t* glm5_pc_engine_create(const glm5_pc_engine_config_t* config);
void glm5_pc_engine_destroy(glm5_pc_engine_t* engine);

int glm5_pc_detect_backend_caps(
    glm5_backend_t preferred_backend,
    glm5_platform_t platform,
    glm5_backend_caps_t* out_caps
);

int glm5_pc_engine_get_backend_caps(const glm5_pc_engine_t* engine, glm5_backend_caps_t* out_caps);

int glm5_pc_engine_get_optimization_plan(
    const glm5_pc_engine_t* engine,
    glm5_inference_phase_t phase,
    glm5_runtime_optimization_plan_t* out_plan
);


