from toy.domain import Ball


class FakeBallRepository:
    def __init__(self, balls: list[Ball]):
        self.balls = balls
        self.asked = []  # remembers who we were asked about

    def balls_for(self, batter: str) -> list[Ball]:
        self.asked.append(batter)
        found = []
        for ball in self.balls:
            if ball.batter == batter:
                found.append(ball)
        return found
