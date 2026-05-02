enum moe_codec_col : size_t {
    moe_CODEC_COL_PART = 0,
    moe_CODEC_COL_SHARD = 1,
    moe_CODEC_COL_SHARD_FILE = 2,
    moe_CODEC_COL_SOURCE_FILE = 3,
    moe_CODEC_COL_WEIGHT_KEY = 4,
    moe_CODEC_COL_LAYER = 5,
    moe_CODEC_COL_EXPERT = 6,
    moe_CODEC_COL_PROJ = 7,
    moe_CODEC_COL_ROWS = 8,
    moe_CODEC_COL_COLS = 9,
    moe_CODEC_COL_GROUPS = 10,
    moe_CODEC_COL_GROUP_SIZE = 11,
    moe_CODEC_COL_WEIGHT_OFFSET = 15,
    moe_CODEC_COL_WEIGHT_LENGTH = 16,
    moe_CODEC_COL_SCALE_OFFSET = 20,
    moe_CODEC_COL_SCALE_LENGTH = 21,
    moe_CODEC_COL_SCALE2_OFFSET = 24,
    moe_CODEC_COL_SCALE2_LENGTH = 25,
    moe_CODEC_COL_REQUIRED = 26
};

static const std::string& codec_col(const std::vector<std::string>& row, size_t index) {
    static const std::string empty;
    return index < row.size() ? row[index] : empty;
}

static int codec_header_supported(const std::string& line) {
    const std::vector<std::string> header = split_csv_line(line);
    // BUGFIX 477: header.size() 범위 체크 강화
    if (header.size() < moe_CODEC_COL_REQUIRED) return 0;
    // BUGFIX 478: 각 컬럼 인덱스 범위 체크
    if (moe_CODEC_COL_PART >= header.size() ||
        moe_CODEC_COL_SHARD_FILE >= header.size() ||
        moe_CODEC_COL_WEIGHT_KEY >= header.size() ||
        moe_CODEC_COL_WEIGHT_OFFSET >= header.size() ||
        moe_CODEC_COL_SCALE2_LENGTH >= header.size()) {
        return 0;
    }
    return header[moe_CODEC_COL_PART] == "part" &&
        header[moe_CODEC_COL_SHARD_FILE] == "shard_file" &&
        header[moe_CODEC_COL_WEIGHT_KEY] == "weight_key" &&
        header[moe_CODEC_COL_WEIGHT_OFFSET] == "weight_byte_offset" &&
        header[moe_CODEC_COL_SCALE2_LENGTH] == "scale2_byte_length";
}
