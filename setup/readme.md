# Setup

As detailed in [FSDL 2022](https://github.com/the-full-stack/fsdl-text-recognizer-2022-labs/tree/main/setup).

- `environment.yml` specifies Python and optionally CUDA/CUDNN
- `make conda-update` creates/updates a virtual environment
- `conda activate lemoncake` activates the virtual environment
- `requirements/prod.in` and `requirements/dev.in` specify core Python packages in that environment
- `make pip-tools` resolves all other Python dependencies and installs them
- `export PYTHONPATH=.:$PYTHONPATH` makes the current directory visible on your Python path -- add it to your
    - `~/.zshrc` and `source ~/.zshrc` OR
    - `~/.bashrc` and `source ~/.bashrc`