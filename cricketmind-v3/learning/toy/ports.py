from typing import Protocol

from toy.domain import Ball


class BallRepository(Protocol):
    def balls_for(self, batter: str) -> list[Ball]: ...
