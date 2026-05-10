static void moe_generation_set_error(moe_generation_stats_t* stats, const char* message) {
    if (!stats || !message) {
        return;
    }
    std::snprintf(stats->error, sizeof(stats->error), "%s", message);
}

static void moe_eval_set_error(moe_eval_stats_t* stats, const char* message) {
    if (!stats || !message) {
        return;
    }
    std::snprintf(stats->error, sizeof(stats->error), "%s", message);
}
