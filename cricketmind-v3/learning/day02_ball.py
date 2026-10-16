from dataclasses import dataclass


@dataclass(frozen=True)
class Ball:
    batter: str
    runs: int
    is_wicket: bool = False


if __name__ == "__main__":
    four = Ball("V Kohli", 4)
    print(four)
    print(four == Ball("V Kohli", 4))  # same values, so equal
    print(Ball("V Kohli", "four"))  # type hints are not checked!
    four.runs = 6  # frozen, so this line crashes
