"""
Python binding for treemo.so via ctypes.

Build the shared library first:
    cc -O3 -shared -fPIC -o treemo.so treemo.c

Usage:
    from treemo import TreemoProgram
    prog = TreemoProgram([1,1,0,1,1,0,0,0])
    result = prog.run([1,0,1,0], max_step=50)  # -> list of 0/1
"""

import ctypes
import pathlib

_lib = ctypes.CDLL(pathlib.Path(__file__).parent / "treemo.so")

_lib.treemo_compile.restype  = ctypes.c_void_p
_lib.treemo_compile.argtypes = [ctypes.c_void_p, ctypes.c_int]

_lib.treemo_exec.restype  = ctypes.c_void_p
_lib.treemo_exec.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                              ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

_lib.treemo_free_prog.restype  = None
_lib.treemo_free_prog.argtypes = [ctypes.c_void_p]

_lib.treemo_free_buf.restype  = None
_lib.treemo_free_buf.argtypes = [ctypes.c_void_p]


class TreemoProgram:
    """Compiled Treemo program. Compile once, run on many inputs."""

    def __init__(self, prog):
        data = bytes(prog)
        self._handle = _lib.treemo_compile(data, len(data))

    def run(self, inp, max_step=50):
        data = bytes(inp)
        out_len = ctypes.c_int(0)
        ptr = _lib.treemo_exec(self._handle, data, len(data),
                               max_step, ctypes.byref(out_len))
        result = list(ctypes.string_at(ptr, out_len.value))
        _lib.treemo_free_buf(ptr)
        return result

    def __del__(self):
        if self._handle:
            _lib.treemo_free_prog(self._handle)
            self._handle = None
