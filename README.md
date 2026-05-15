# Lensieve
Private AI Photo Search and Curation

Default:
uv sync

Optional acceleration:
macOS Apple Silicon:
CMAKE_ARGS="-DLLAMA_METAL=on" uv pip install --no-binary=llama-cpp-python llama-cpp-python

NVIDIA CUDA:
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --no-binary=llama-cpp-python llama-cpp-python