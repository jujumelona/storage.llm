static void glm5_generation_set_error(glm5_generation_stats_t* stats, const char* message) {
    if (!stats || !message) {
        return;
    }
    std::snprintf(stats->error, sizeof(stats->error), "%s", message);
}

static void glm5_eval_set_error(glm5_eval_stats_t* stats, const char* message) {
    if (!stats || !message) {
        return;
    }
    std::snprintf(stats->error, sizeof(stats->error), "%s", message);
}
