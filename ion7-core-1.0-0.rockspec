package = "ion7-core"
version = "1.0-0"

source = {
    url = "git+https://github.com/Ion7-Labs/ion7-core.git",
    tag = "v1.0.0",
}

description = {
    summary  = "Silicon-level LuaJIT FFI bindings for llama.cpp",
    detailed = [[
        ion7-core provides direct, zero-overhead access to llama.cpp from LuaJIT.
        Includes Model, Context, Vocab, Sampler, and Threadpool primitives.
        No business logic - just the metal.
    ]],
    homepage = "https://github.com/Ion7-Labs/ion7-core",
    license  = "MIT-or-later",
}

dependencies = {
    "lua >= 5.1",
}

build = {
    type    = "builtin",
    modules = {
        ["ion7.core"]                    = "src/ion7/core/init.lua",
        ["ion7.core.model"]              = "src/ion7/core/model.lua",
        ["ion7.core.context"]            = "src/ion7/core/context.lua",
        ["ion7.core.vocab"]              = "src/ion7/core/vocab.lua",
        ["ion7.core.sampler"]            = "src/ion7/core/sampler.lua",
        ["ion7.core.custom_sampler"]     = "src/ion7/core/custom_sampler.lua",
        ["ion7.core.threadpool"]         = "src/ion7/core/threadpool.lua",
        ["ion7.core.ffi.loader"]         = "src/ion7/core/ffi/loader.lua",
        ["ion7.core.ffi.types"]          = "src/ion7/core/ffi/types.lua",
    },
}
