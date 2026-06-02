
# DEPRECATED

# Building vllm-graph from source 

## Building the llvm-project

First clone the llvm-project 
```
cd /llvm-build
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout d418a03e01e6a31b51b0c9dd42ba46da6c47f89d
```

Build llvm
```

cmake -B build \
      -G Ninja \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_TARGETS_TO_BUILD=host \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      llvm

cmake --build build -j$(nproc)
```


