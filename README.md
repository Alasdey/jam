# jam
A jam trying to reach escape velocity.

## Setup

### 1. Python dependencies
```bash
uv sync
```

### 2. Subleq C library
```bash
gcc -shared -fPIC -o interpreters/subleq/libsubleq.so interpreters/subleq/subleq.c
```

### 3. Treemo Rust extension (optional but strongly recommended — ~200× faster)

Requires the Rust toolchain. Install it once with:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then build and install the extension into the project venv:
```bash
cd interpreters/treemo/treemo_rs
VIRTUAL_ENV=../../.venv maturin develop --release
cd ../../..
```

The interpreter auto-detects the extension at import time and falls back to pure
Python if it is not present.

To verify which backend is active:
```python
from interpreters.treemo.treemo import _RUST
print("Rust backend:", _RUST)
```

### 4. Run
```bash
uv run python main.py
```

## Interpreter layout

| File | Role |
|---|---|
| `interpreters/treemo/treemo.py` | Entry point — delegates entirely to the Rust extension |
| `interpreters/treemo/treemo_rs/` | Rust crate (PyO3 + memchr Two-Way SIMD search) |
| `interpreters/treemo_python/treemo_python.py` | Pure-Python reference implementation |

## Todo's
Make a wrapper for the interpreters to allow for more UISC or RISC languages (be it marginal improvements). \
Make a wrapper for the rewards \
Build a logging system \
Improve the payoff computation \
Import the Nash set computation \
Import the Genetic Programming
