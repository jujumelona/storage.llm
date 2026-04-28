enum glm5_codec_col : size_t {
    GLM5_CODEC_COL_PART = 0,
    GLM5_CODEC_COL_SHARD = 1,
    GLM5_CODEC_COL_SHARD_FILE = 2,
    GLM5_CODEC_COL_SOURCE_FILE = 3,
    GLM5_CODEC_COL_WEIGHT_KEY = 4,
    GLM5_CODEC_COL_LAYER = 5,
    GLM5_CODEC_COL_EXPERT = 6,
    GLM5_CODEC_COL_PROJ = 7,
    GLM5_CODEC_COL_ROWS = 8,
    GLM5_CODEC_COL_COLS = 9,
    GLM5_CODEC_COL_GROUPS = 10,
    GLM5_CODEC_COL_GROUP_SIZE = 11,
    GLM5_CODEC_COL_WEIGHT_OFFSET = 15,
    GLM5_CODEC_COL_WEIGHT_LENGTH = 16,
    GLM5_CODEC_COL_SCALE_OFFSET = 20,
    GLM5_CODEC_COL_SCALE_LENGTH = 21,
    GLM5_CODEC_COL_SCALE2_OFFSET = 24,
    GLM5_CODEC_COL_SCALE2_LENGTH = 25,
    GLM5_CODEC_COL_REQUIRED = 26
};

static const std::string& codec_col(const std::vector<std::string>& row, size_t index) {
    static const std::string empty;
    return index < row.size() ? row[index] : empty;
}

static int codec_header_supported(const std::string& line) {
    const std::vector<std::string> header = split_csv_line(line);
    return header.size() >= GLM5_CODEC_COL_REQUIRED &&
        header[GLM5_CODEC_COL_PART] == "part" &&
        header[GLM5_CODEC_COL_SHARD_FILE] == "shard_file" &&
        header[GLM5_CODEC_COL_WEIGHT_KEY] == "weight_key" &&
        header[GLM5_CODEC_COL_WEIGHT_OFFSET] == "weight_byte_offset" &&
        header[GLM5_CODEC_COL_SCALE2_LENGTH] == "scale2_byte_length";
}
