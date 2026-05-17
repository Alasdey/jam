# treemo_c

C implementation of the Treemo interpreter, exposed to Python via ctypes.

## Build

```bash
cc -O3 -shared -fPIC -o libtreemo.so treemo.c   # shared library (Python binding)
cc -O3 -o treemo treemo.c                         # standalone CLI
```

## CLI

```bash
./treemo <program_bits> <input_bits> [max_step]
# e.g. ./treemo 11011000 1010 50
```

## Python

```python
from interpreters.treemo_c.treemo import TreemoInterpreter
interp = TreemoInterpreter(max_step=50)
output, code = interp.run(prog, inp)   # prog and inp are lists of 0/1
```

Each unique program is compiled once and the handle cached for the lifetime of the interpreter instance.

## Compile vs exec overhead

Measured on trees of 500–3000 nodes, 500 samples:

| max_step | compile avg | exec avg | ratio |
|---|---|---|---|
| 200 | 261 µs | 107 µs | compile is **3× slower** |
| 20 000 | 261 µs | 36 356 µs | compile is **139× faster** |

Compile cost (rule extraction) is constant in `max_step`. At low step counts it dominates and caching it per-program is the main lever. At high step counts it is negligible and execution dominates. Caching is implemented at the instance level; whether a module-level global cache is worth adding depends on the `max_step` regime the experiments run in.
