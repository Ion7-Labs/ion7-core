package = "ion7-core"
version = "0.1.0beta2-1"

source = {
    url = "git+https://github.com/Ion7-Labs/ion7-core.git",
    tag = "v0.1.0-beta2",
}

description = {
    summary  = "Silicon-level LuaJIT FFI bindings for llama.cpp",
    detailed = [[
        ion7-core gives LuaJIT direct, zero-overhead access to llama.cpp.
        Model loading, decode, KV cache, sampler chains, custom samplers,
        threadpool sharing, speculative decoding, training, GBNF
        constraints — all driven from Lua via FFI plus a small libcommon
        bridge. The rockspec build vendors llama.cpp as a submodule and
        compiles it together with `ion7_bridge.so`. Backend selection is
        driven by the ION7_BACKEND environment variable :

            ION7_BACKEND=cpu      (default — pure CPU, AVX2 / NEON)
            ION7_BACKEND=vulkan   (cross-vendor GPU)
            ION7_BACKEND=cuda     (NVIDIA — also reads ION7_CUDA_ARCH)
            ION7_BACKEND=rocm     (AMD)
            ION7_BACKEND=metal    (Apple Silicon)

        The build leaves `libllama`, `libggml*`, and `ion7_bridge` under
        `<rocktree>/share/lua/<ver>/ion7/core/_libs/`. The FFI loader
        probes that directory automatically — no LD_LIBRARY_PATH or
        ION7_LIBLLAMA_PATH gymnastics required.
    ]],
    homepage = "https://github.com/Ion7-Labs/ion7-core",
    license  = "MIT-or-later",
}

dependencies = {
    "lua >= 5.1",
    "lua-cjson >= 2.1",
}

external_dependencies = {
    -- The build needs cmake (for vendor/llama.cpp), a C++17 toolchain
    -- (for the bridge), and the headers / libraries of any optional
    -- backend the user selects. luarocks cannot validate those at
    -- declaration time ; the build_command fails loudly when missing.
}

build = {
    type = "command",

    -- Build vendor/llama.cpp + bridge/ion7_bridge.so.
    --
    -- The Makefile-driven build (`make build`) is reused verbatim so
    -- the development and rockspec-install paths agree. The submodule
    -- is initialised on demand (idempotent : a no-op when already
    -- populated by the user's clone).
    build_command = [[
set -e
echo "[ion7-core] backend = ${ION7_BACKEND:-cpu}"

if [ ! -f vendor/llama.cpp/CMakeLists.txt ]; then
    echo "[ion7-core] initialising vendor/llama.cpp submodule..."
    git submodule update --init --recursive vendor/llama.cpp
fi

# Backend → cmake flags
case "${ION7_BACKEND:-cpu}" in
    cpu)
        BACKEND_FLAGS=""
        ;;
    vulkan)
        BACKEND_FLAGS="-DGGML_VULKAN=ON"
        ;;
    cuda)
        BACKEND_FLAGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${ION7_CUDA_ARCH:-86}"
        ;;
    rocm)
        BACKEND_FLAGS="-DGGML_HIP=ON"
        ;;
    metal)
        BACKEND_FLAGS="-DGGML_METAL=ON"
        ;;
    *)
        echo "[ion7-core] unknown ION7_BACKEND='${ION7_BACKEND}' (cpu|vulkan|cuda|rocm|metal)" >&2
        exit 1
        ;;
esac

if [ -n "${ION7_LLAMA_CMAKE_EXTRA:-}" ]; then
    BACKEND_FLAGS="$BACKEND_FLAGS $ION7_LLAMA_CMAKE_EXTRA"
fi

echo "[ion7-core] cmake configure: $BACKEND_FLAGS"
cmake -B vendor/llama.cpp/build -S vendor/llama.cpp \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    $BACKEND_FLAGS

echo "[ion7-core] cmake build (this can take a while)..."
cmake --build vendor/llama.cpp/build --config Release -j

echo "[ion7-core] building ion7_bridge..."
$(MAKE) -C bridge \
    LIB_DIR="$(pwd)/vendor/llama.cpp/build/bin" \
    COMMON_LIB_DIR="$(pwd)/vendor/llama.cpp/build/common" \
    LLAMA_SRC="$(pwd)/vendor/llama.cpp" \
    ION7_RELEASE=1
]],

    -- Lay out the rock tree :
    --   <prefix>/share/lua/<ver>/ion7/                  Lua sources
    --   <prefix>/share/lua/<ver>/ion7/core/_libs/       libllama* + ion7_bridge*
    --   <prefix>/bin/ion7-load.lua                      tarball preamble
    --
    -- The libs ship with $ORIGIN-relative rpath (ION7_RELEASE=1) so
    -- ion7_bridge resolves libllama from the same _libs/ directory
    -- regardless of where the rock tree is mounted.
    install_command = [[
set -e
mkdir -p "$(LUADIR)/ion7"
cp -r src/ion7/* "$(LUADIR)/ion7/"

mkdir -p "$(LUADIR)/ion7/core/_libs"
# Copy every shared / dynamic library produced by the build, including
# version symlinks (`libllama.so.0`) and platform variants. `cp -d`
# preserves symlinks ; the `|| true` keeps install going on platforms
# that produce a different filename set (e.g. .dylib only on macOS).
cp -d vendor/llama.cpp/build/bin/lib*.so*    "$(LUADIR)/ion7/core/_libs/" 2>/dev/null || true
cp -d vendor/llama.cpp/build/bin/lib*.dylib* "$(LUADIR)/ion7/core/_libs/" 2>/dev/null || true
cp -d vendor/llama.cpp/build/bin/*.dll       "$(LUADIR)/ion7/core/_libs/" 2>/dev/null || true
cp bridge/ion7_bridge.so    "$(LUADIR)/ion7/core/_libs/" 2>/dev/null || true
cp bridge/ion7_bridge.dylib "$(LUADIR)/ion7/core/_libs/" 2>/dev/null || true
cp bridge/ion7_bridge.dll   "$(LUADIR)/ion7/core/_libs/" 2>/dev/null || true

mkdir -p "$(BINDIR)"
cp bin/ion7-load.lua "$(BINDIR)/"
chmod 0755 "$(BINDIR)/ion7-load.lua"
]],
}
