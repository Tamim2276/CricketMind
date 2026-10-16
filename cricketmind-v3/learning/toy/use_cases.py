from toy.domain import strike_rate
from toy.ports import BallRepository


class GetStrikeRate:
    def __init__(self, repo: BallRepository):
        self.repo = repo

    def run(self, batter: str) -> float | None:
        balls = self.repo.balls_for(batter)
        return strike_rate(balls)
