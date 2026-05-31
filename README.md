# Lensieve

![Demo](assets/demo.gif)

Lensieve is a local, privacy-first tool for semantic search and metadata querying over personal photo collections.  
It runs entirely offline and does not require external services.

Lensieve supports free-form text queries with optional time references, for example:

`Show me photos of nature from last summer`

It also supports metadata queries, for example:

`How many portrait photos did I take in 2024?`

Image search results can be grouped by visual similarity in the UI.

The full pipeline (ingestion and search) can run on CPU-only machines, though performance will be slower than on hardware-accelerated systems.

For technical details, see the accompanying article:
https://dmitryfrumkin.substack.com/p/lensieve

## Setup

### 1. Install uv

Install `uv`: https://github.com/astral-sh/uv

### 2. Create the environment

```bash
uv sync
```

#### Linux notes

Most dependencies work out of the box on Linux. On some systems, additional packages may be required:

```bash
sudo apt-get install libraw-dev libheif-dev libde265-dev
```

This is mainly needed for RAW image support (`rawpy`) and HEIF/HEIC images.

#### Hardware acceleration

By default, `uv sync` installs `llama-cpp-python` with CPU-only support.

To rebuild `llama-cpp-python` with hardware acceleration, run one of the following commands after `uv sync`.

##### macOS (Apple Silicon)
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" uv sync --reinstall-package llama-cpp-python
```

##### Linux (NVIDIA CUDA)
```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv sync --reinstall-package llama-cpp-python
```

#### Development (optional)

To work on the codebase, install development dependencies:

```bash
uv sync --group dev
```

This includes tools for:

- Linting and formatting (`ruff`)
- Notebooks (`jupyterlab`, `ipywidgets`)
- Notebook diffing (`nbdime`)

### 3. Configure Lensieve

Edit `configs/config.yaml` and set `root` to the directory containing your images.

- Nested directories are supported  
- Duplicate filenames across directories are allowed  
- Multiple image formats are supported  
- Video files and animated images (e.g. GIFs) are not supported 

You may also adjust model selection depending on your hardware.

### 4. Download models

```bash
uv run python src/lensieve/cli/download_models.py
```

## Ingestion

Process images and build the index:

```bash
uv run python src/lensieve/cli/ingest.py
```

## Run

Start the UI:

```bash
uv run python src/lensieve/cli/run_gradio.py
```

Enter a query, review results, and group similar images.