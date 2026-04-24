from abc import ABC, abstractmethod
from typing import Optional

Program = list  # list[int] for all current interpreters


class Creator(ABC):

    @abstractmethod
    def random(self) -> Program: ...

    @abstractmethod
    def mutate(self, prog: Program) -> Program: ...

    @abstractmethod
    def crossover(self, prog_a: Program, prog_b: Program) -> Program: ...

    @abstractmethod
    def homoiconic(self, interp, prog_a: Program, prog_b: Program) -> Optional[Program]: ...
