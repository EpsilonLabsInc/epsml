# Setup

1. [Install `uv`](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)
2. Install the required python version (defined in `.python-version`) with uv
    ```bash
    uv python install 3.11.13
    ```

3. Create a virtualenv and activate it
    ```bash
    uv venv
    source .venv/bin/activate
    ```

4. Install dependencies (only do this within the virtualenv)
    - Install only main dependencies
      ```bash
      uv sync --no-dev
      ```
    - Install only dev dependencies
      ```bash
      uv sync --group dev
      ```
    - Install all dependencies
      ```bash
      uv sync
      ```