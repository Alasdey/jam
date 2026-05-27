"""
Python binding for libtreemo.so via ctypes.

Build the shared library first:
    cc -O3 -shared -fPIC -o libtreemo.so treemo.c

Usage:
    from treemo import TreemoInterpreter
    interp = TreemoInterpreter(max_step=50, pass_mode=True, first_mode=False)
    result, code = interp.run(code, inp)   # compiles code on first call, cached thereafter

pass_mode : True  → a rule fires at most once before the interpreter moves on
            False → a rule fires until it no longer matches before moving on
first_mode: False → after advancing, continue to the next rule in sequence (default)
            True  → after advancing, restart from rule 0
"""

import ctypes
import pathlib
from typing import List, Tuple

_lib = ctypes.CDLL(pathlib.Path(__file__).parent / "libtreemo.so")

_lib.treemo_compile.restype  = ctypes.c_void_p
_lib.treemo_compile.argtypes = [ctypes.c_void_p, ctypes.c_int]

# treemo_exec(prog, inp, ilen, max_step, pass_mode, first_mode, *out_len)
_lib.treemo_exec.restype  = ctypes.c_void_p
_lib.treemo_exec.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_int)]

_lib.treemo_free_prog.restype  = None
_lib.treemo_free_prog.argtypes = [ctypes.c_void_p]

_lib.treemo_free_buf.restype  = None
_lib.treemo_free_buf.argtypes = [ctypes.c_void_p]


class TreemoInterpreter:
    """
    Treemo interpreter backed by libtreemo.so.
    Each unique program is compiled once and cached for subsequent runs.
    """

    def __init__(self, max_step: int = 50,
                 pass_mode: bool = True, first_mode: bool = False):
        self.max_step  = max_step
        self.pass_mode  = int(pass_mode)
        self.first_mode = int(first_mode)
        self._cache: dict[bytes, int] = {}  # code bytes -> compiled handle (c_void_p)

    def run(self, code: List[int], inp: List[int]) -> Tuple[List[int], List[int]]:
        key = bytes(code)
        if key not in self._cache:
            self._cache[key] = _lib.treemo_compile(key, len(key))
        handle = self._cache[key]

        inp_bytes = bytes(inp)
        out_len = ctypes.c_int(0)
        ptr = _lib.treemo_exec(handle, inp_bytes, len(inp_bytes),
                               self.max_step, self.pass_mode, self.first_mode,
                               ctypes.byref(out_len))
        result = list(ctypes.string_at(ptr, out_len.value))
        _lib.treemo_free_buf(ptr)
        return result, code

    def __del__(self):
        for handle in self._cache.values():
            _lib.treemo_free_prog(handle)
        self._cache.clear()
