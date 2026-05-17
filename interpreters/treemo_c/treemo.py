"""
Python binding for treemo.so via ctypes.

Build the shared library first:
    cc -O3 -shared -fPIC -o treemo.so treemo.c

Usage:
    from treemo import TreemoInterpreter
    interp = TreemoInterpreter(max_step=50)
    result, code = interp.run(code, inp)   # compiles code on first call, cached thereafter
"""

import ctypes
import pathlib
from typing import List, Tuple

_lib = ctypes.CDLL(pathlib.Path(__file__).parent / "libtreemo.so")

_lib.treemo_compile.restype  = ctypes.c_void_p
_lib.treemo_compile.argtypes = [ctypes.c_void_p, ctypes.c_int]

_lib.treemo_exec.restype  = ctypes.c_void_p
_lib.treemo_exec.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                              ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

_lib.treemo_free_prog.restype  = None
_lib.treemo_free_prog.argtypes = [ctypes.c_void_p]

_lib.treemo_free_buf.restype  = None
_lib.treemo_free_buf.argtypes = [ctypes.c_void_p]


class TreemoInterpreter:
    """
    Treemo interpreter backed by treemo.so.
    Each unique program is compiled once and cached for subsequent runs.
    """

    def __init__(self, max_step: int = 50):
        self.max_step = max_step
        self._cache: dict[bytes, int] = {}  # code bytes -> compiled handle (c_void_p)

    def run(self, code: List[int], inp: List[int]) -> Tuple[List[int], List[int]]:
        key = bytes(code)
        if key not in self._cache:
            self._cache[key] = _lib.treemo_compile(key, len(key))
        handle = self._cache[key]

        inp_bytes = bytes(inp)
        out_len = ctypes.c_int(0)
        ptr = _lib.treemo_exec(handle, inp_bytes, len(inp_bytes),
                               self.max_step, ctypes.byref(out_len))
        result = list(ctypes.string_at(ptr, out_len.value))
        _lib.treemo_free_buf(ptr)
        return result, code

    def __del__(self):
        for handle in self._cache.values():
            _lib.treemo_free_prog(handle)
        self._cache.clear()
