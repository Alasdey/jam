from typing import Callable, Dict, List, Optional

Program = List[int]
HomoiconicFn = Callable[..., Optional[Program]]  # (interp, prog_a, prog_b) -> Program | None


def homoiconic_output(interp, prog_a: Program, prog_b: Program) -> Optional[Program]:
    output, _ = interp.run(prog_a, prog_b)
    return output if output else None


def homoiconic_memory(interp, prog_a: Program, prog_b: Program) -> Optional[Program]:
    _, memory = interp.run(prog_a, prog_b)
    return memory if memory else None


HOMOICONIC_OPS: Dict[str, HomoiconicFn] = {
    "output": homoiconic_output,
    "memory": homoiconic_memory,
}
