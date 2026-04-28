int glm5_pc_group_tokens_by_expert(
    const glm5_token_expert_assignment_t* assignments,
    uint32_t assignment_count,
    glm5_token_expert_assignment_t* out_sorted_assignments,
    glm5_expert_batch_t* out_batches,
    uint32_t max_batches,
    uint32_t* out_batch_count
) {
    if (!assignments || assignment_count == 0 || !out_batch_count) {
        if (out_batch_count) {
            *out_batch_count = 0;
        }
        return 0;
    }

    if (assignment_count == 1) {
        if (out_sorted_assignments) {
            out_sorted_assignments[0] = assignments[0];
        }
        if (out_batches && max_batches >= 1) {
            out_batches[0].layer = assignments[0].layer;
            out_batches[0].expert = assignments[0].expert;
            out_batches[0].first_sorted_index = 0;
            out_batches[0].token_count = 1;
        }
        *out_batch_count = 1;
        return 1;
    }

    // Bug #17: Sort directly into caller-owned buffer — no heap allocation per call
    // If out_sorted_assignments is provided, copy+sort in-place; else use temp vector
    glm5_token_expert_assignment_t* sort_buf = out_sorted_assignments;
    std::vector<glm5_token_expert_assignment_t> fallback;
    if (!sort_buf) {
        fallback.assign(assignments, assignments + assignment_count);
        sort_buf = fallback.data();
    } else {
        std::memcpy(sort_buf, assignments, sizeof(glm5_token_expert_assignment_t) * assignment_count);
    }
    std::sort(sort_buf, sort_buf + assignment_count, [](const auto& a, const auto& b) {
        if (a.layer != b.layer) {
            return a.layer < b.layer;
        }
        if (a.expert != b.expert) {
            return a.expert < b.expert;
        }
        return a.token_index < b.token_index;
    });

    uint32_t batch_count = 0;
    uint32_t start = 0;
    while (start < assignment_count) {
        const uint32_t layer = sort_buf[start].layer;
        const uint32_t expert = sort_buf[start].expert;
        uint32_t end = start + 1;
        while (end < assignment_count && sort_buf[end].layer == layer && sort_buf[end].expert == expert) {
            ++end;
        }
        if (out_batches && batch_count < max_batches) {
            out_batches[batch_count].layer = layer;
            out_batches[batch_count].expert = expert;
            out_batches[batch_count].first_sorted_index = start;
            out_batches[batch_count].token_count = end - start;
        }
        ++batch_count;
        start = end;
    }

    *out_batch_count = batch_count;
    return (!out_batches || batch_count <= max_batches) ? 1 : 0;
}

