# Tutorial: How to Finetune Llama-3 and Use In Ollama

![unsloth](images/unsloth.png) ![ollama](images/ollama.png)

See https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama.

## Installation

### Prerequisites

* Install CUDA 12.4 (https://developer.nvidia.com/cuda-12-4-1-download-archive)
* Install Git (https://git-scm.com/downloads)
* Install CMake (https://cmake.org/download/)
* Install Ollama (https://ollama.com/download)
* Install UV and Python 3.12:
  * Add the following environment variables (to keep Symantec happy):
    ```
    UV_INSTALL_DIR=C:\UV
    UV_CACHE_DIR=C:\UV\.cache
    UV_PYTHON_INSTALL_DIR=C:\UV\python-installs
    UV_TOOL_DIR=C:\UV\tools
    ```
  * Add `C:\UV` to path.
  * Install UV: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  * Install Python: `uv python install 3.12`

# Run

The following command will also automatically create a virtual environment:

```
uv run main.py
```

Note that this will error, and that `fix.bat` will then need to be run. See [this Unsloth issue](https://github.com/unslothai/unsloth/issues/748) for details. It should work after that.
