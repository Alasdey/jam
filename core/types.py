from dataclasses import dataclass
from typing import Protocol

Program = list[int]


@dataclass
class Individual:
    id: int
    genome: Program
    method: str  # "random" | "mutate" | "crossover" | "homoiconic" | "seeded"
    parents: list[int]
    born_gen: int


class Interpreter(Protocol):
    def run(self, code: Program, inp: Program) -> tuple[Program, Program]: ...
