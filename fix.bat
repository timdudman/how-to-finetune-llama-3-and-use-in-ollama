@echo off
REM https://github.com/unslothai/unsloth/issues/748#issuecomment-2750091501
tar -xf fix/llama-b4981-bin-win-cuda-cu12.4-x64.zip -C llama.cpp
copy /Y fix\save.py .venv\Lib\site-packages\unsloth\save.py