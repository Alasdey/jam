from typing import List, Tuple

try:
    from treemo_rs import treemo as _treemo_rs
except ImportError as e:
    raise ImportError(
        "Rust extension 'treemo_rs' not found. "
        "Build it with:\n"
        "  cd interpreters/treemo/treemo_rs && VIRTUAL_ENV=../../.venv maturin develop --release"
    ) from e


def treemo(code: List[int], inp: List[int], max_step: int = 5) -> List[int]:
    return list(_treemo_rs(code, inp, max_step))


class TreemoInterpreter:
    def __init__(self, max_step: int = 50):
        self.max_step = max_step

    def run(self, code: List[int], inp: List[int]) -> Tuple[List[int], List[int]]:
        result = treemo(code, inp, max_step=self.max_step)
        return result, code
