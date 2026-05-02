/*
 * bridge_text.cpp - Text-shaped libcommon helpers.
 *
 * Copyright (C) 2026 Ion7 Project Contributors
 * SPDX-License-Identifier: MIT
 *
 * Three loosely related concerns are colocated here because they all
 * operate on text strings and pull in nlohmann/json one way or another:
 *
 *   - Partial regex      : `common_regex` for streaming stop-string detection.
 *   - JSON Schema → GBNF : the libcommon converter, more thorough than the
 *                          pure-Lua ion7-grammar fallback.
 *   - JSON validate /    : nlohmann/json wrappers exposing a string-in /
 *     format / merge       string-out C ABI for the Lua side.
 */

#include "ion7_bridge.h"

#include "regex-partial.h"
#include "json-schema-to-grammar.h"

/* nlohmann/json is needed for ordered_json::parse() in schema → GBNF
 * and for the JSON utilities below. Pulled in once for the whole file. */
#include "nlohmann/json.hpp"

#include <cstdint>
#include <cstring>
#include <string>

/* =========================================================================
 * Partial regex (streaming stop-string detection)
 * ======================================================================= */

/**
 * Owning wrapper around `common_regex`. Keeping a dedicated handle
 * leaves room to stash precomputed state (anchored variants, length
 * hints, ...) in a future revision without breaking the C ABI.
 */
struct ion7_regex {
    common_regex rx;
    explicit ion7_regex(const char* pattern) : rx(pattern) {}
};

extern "C"
ion7_regex_t* ion7_regex_new(const char* pattern)
{
    if (!pattern) return nullptr;
    try {
        return new ion7_regex(pattern);
    } catch (...) {
        return nullptr;
    }
}

extern "C"
void ion7_regex_free(ion7_regex_t* r)
{
    delete r;
}

extern "C"
int ion7_regex_search(ion7_regex_t* r,
                      const char*   text,
                      size_t        len,
                      int           partial)
{
    if (!r || !text) return 0;
    std::string s(text, len);
    common_regex_match m = r->rx.search(s, /*from=*/0, /*as_match=*/false);
    if (m.type == COMMON_REGEX_MATCH_TYPE_FULL)               return 2;
    if (partial && m.type == COMMON_REGEX_MATCH_TYPE_PARTIAL) return 1;
    return 0;
}

/* =========================================================================
 * JSON Schema → GBNF grammar
 * ======================================================================= */

extern "C"
int ion7_json_schema_to_grammar(const char* schema_json,
                                char*       out,
                                size_t      out_len)
{
    if (!schema_json) return -1;
    try {
        auto schema = nlohmann::ordered_json::parse(schema_json);
        std::string gbnf = json_schema_to_grammar(schema);
        size_t needed = gbnf.size() + 1;
        if (out && out_len >= needed) {
            std::memcpy(out, gbnf.c_str(), needed);
        }
        return (int)needed;
    } catch (...) {
        return -1;
    }
}

/* =========================================================================
 * JSON utilities (validate / format / merge)
 * ======================================================================= */

extern "C"
int ion7_json_validate(const char* json_str)
{
    if (!json_str) return 0;
    return nlohmann::ordered_json::accept(json_str) ? 1 : 0;
}

extern "C"
int ion7_json_format(const char* json_str, char* out, size_t out_len)
{
    if (!json_str) return -1;
    try {
        auto j = nlohmann::ordered_json::parse(json_str);
        std::string s = j.dump(/*indent=*/2);
        size_t needed = s.size() + 1;
        if (out && out_len >= needed) {
            std::memcpy(out, s.c_str(), needed);
        }
        return (int)needed;
    } catch (...) {
        return -1;
    }
}

extern "C"
int ion7_json_merge(const char* base,
                    const char* overlay,
                    char*       out,
                    size_t      out_len)
{
    if (!base || !overlay) return -1;
    try {
        auto j = nlohmann::ordered_json::parse(base);
        j.merge_patch(nlohmann::ordered_json::parse(overlay));
        std::string s = j.dump();
        size_t needed = s.size() + 1;
        if (out && out_len >= needed) {
            std::memcpy(out, s.c_str(), needed);
        }
        return (int)needed;
    } catch (...) {
        return -1;
    }
}
