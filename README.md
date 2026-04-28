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

### 3. Treemo Rust extension

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

### 4. Run
```bash
uv run python main.py
```

## Interpreter layout

| File | Role |
|---|---|
| `interpreters/treemo/treemo.py` | Entry point — delegates entirely to the Rust extension |
| `interpreters/treemo/treemo_rs/` | Rust crate (PyO3 + memchr Two-Way SIMD search) |

## Todo's
Make a wrapper for the interpreters to allow for more UISC or RISC languages (be it marginal improvements). **(Done)**\
Make a wrapper for the rewards **(Done)**\
Build a logging system **(Done)**\
Improve the payoff computation **(Done)**\
Import the Nash set computation **(Done)**\
Import the Genetic Programming **(Done)**\
The subleq is broken in some way because the rewards with it make matrix that make no sens. \
Unit testing or the like to check sanity of the code (interpreters, reward, selection ect) \