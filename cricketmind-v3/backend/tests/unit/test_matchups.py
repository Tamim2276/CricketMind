"""Days 51-52: one batter against two bowlers, and the difference.

SYNTHETIC: every batter and bowler here is made up.
"""

import random

from cricketmind_v3.domain import matchups
from cricketmind_v3.domain.matchups import (
    Pair,
    expected,
    matchup,
    meetings_worth,
    representative_pairs,
)
from cricketmind_v3.domain.shrinkage import population_prior


def field(meetings=None) -> list[Pair]:
    """40 other batters against 40 other bowlers, 100 balls a pair. A batter is careful
    (4%) or risky (6%), a bowler mild (4%) or sharp (6%); together, about as the odds
    rule says (3, 5 or 7 outs in 100). Then K, at exactly normal, and two named bowlers,
    each with 100,000 balls against the others: B is sharp (8%), S mild (2%), so normal
    stays 5%. meetings: K's (outs, balls) against B, if any."""
    pairs = []
    for i in range(40):
        for j in range(40):
            outs = 3 + 2 * (i % 2) + 2 * (j % 2)
            pairs.append(
                Pair(
                    batter=f"bat{i}",
                    bowler=f"bowl{j}",
                    order="top",
                    outs=outs,
                    faced=100,
                )
            )
    for j in range(40):
        pairs.append(
            Pair(batter="K", bowler=f"bowl{j}", order="top", outs=125, faced=2500)
        )
    for i in range(40):
        pairs.append(
            Pair(batter=f"bat{i}", bowler="B", order="top", outs=200, faced=2500)
        )
        pairs.append(
            Pair(batter=f"bat{i}", bowler="S", order="top", outs=50, faced=2500)
        )
    if meetings is not None:
        outs, faced = meetings
        pairs.append(Pair(batter="K", bowler="B", order="top", outs=outs, faced=faced))
    return pairs


# The odds rule


def test_the_odds_rule():
    # a normal batter against a bowler is just the bowler, and the other way round
    assert abs(expected(0.05, 0.08, 0.05) - 0.08) < 1e-12
    assert abs(expected(0.08, 0.05, 0.05) - 0.08) < 1e-12
    # by hand: odds 0.1111 (10%) and 0.0870 (8%), over normal's 0.0526 (5%): 0.1836,
    # so 0.1836 / 1.1836 = 15.5%
    assert abs(expected(0.10, 0.08, 0.05) - 0.1551) < 0.0001


# The plan's tests: the two sides and their difference


def test_against_a_sharp_bowler_and_a_mild_one():
    found = matchup(field(), "K", "B", "S", 6)
    # by hand: K is normal, so against B his risk is B's 8%, against S it is 2%
    this = 1 - 0.92**6  # 0.394
    that = 1 - 0.98**6  # 0.114
    assert abs(found.this.chance - this) < 0.01
    assert abs(found.that.chance - that) < 0.01
    assert abs(found.difference - (this - that)) < 0.015
    assert found.low > 0  # a real difference: the range leaves out zero
    assert found.low < found.difference < found.high
    assert found.n == 100_000  # the thinnest record it rests on


def test_the_same_batter_in_both_replays():
    # with only 4 outs in 50 balls, K is uncertain; both chances move with him, so the
    # difference is surer than either side
    pairs = field()
    small = []
    for pair in pairs:
        if pair.batter != "K":
            small.append(pair)
    small.append(Pair(batter="K", bowler="bowl0", order="top", outs=4, faced=50))
    found = matchup(small, "K", "B", "S", 6)
    assert found.high - found.low < found.this.high - found.this.low


def test_the_same_seed_gives_the_same_answer():
    assert matchup(field(), "K", "B", "S", 6) == matchup(field(), "K", "B", "S", 6)


def test_each_is_pulled_toward_the_others(monkeypatch):
    # the batters' normal leaves out K; the bowlers' leaves out B and S
    handed = []

    def recording(records):
        handed.append(list(records))
        return population_prior(records)

    monkeypatch.setattr(matchups, "population_prior", recording)
    matchup(field(), "K", "B", "S", 6)
    batters, bowlers = handed
    assert len(batters) == 40  # bat0 to bat39, not K
    assert len(bowlers) == 40  # bowl0 to bowl39, not B or S
    assert (125 * 40, 2500 * 40) not in batters  # K's record
    assert (200 * 40, 100_000) not in bowlers  # B's record


# Their meetings


def test_meetings_that_add_nothing_move_nothing():
    # here every pair is as the odds rule says, so meetings are worth a great deal and
    # 3 outs in 20 balls against B (15%) barely moves K from 8%
    plain = matchup(field(), "K", "B", "S", 6)
    met = matchup(field(meetings=(3, 20)), "K", "B", "S", 6)
    assert met.meetings_worth > 10_000
    assert abs(met.this.chance - plain.this.chance) < 0.02


def test_meetings_that_matter_move_the_answer():
    # now each pair is 3 outs in 100 off the odds rule, one way or the other, so meetings
    # do matter; and K was out 30 times in 100 balls against B
    pairs = []
    for pair in field():
        i = pair.batter[3:]
        j = pair.bowler[4:]
        if pair.batter.startswith("bat") and pair.bowler.startswith("bowl"):
            off = 3
            if (int(i) + 2 * int(j)) % 4 >= 2:
                off = -3
            pair = Pair(
                batter=pair.batter,
                bowler=pair.bowler,
                order="top",
                outs=pair.outs + off,
                faced=100,
            )
        pairs.append(pair)
    plain = matchup(pairs, "K", "B", "S", 6)
    pairs.append(Pair(batter="K", bowler="B", order="top", outs=30, faced=100))
    met = matchup(pairs, "K", "B", "S", 6)
    assert met.meetings_worth < 200
    assert met.this.chance > plain.this.chance + 0.2  # 30 in 100 is heard


def made_up_meetings(rng, worth):
    """Batters and bowlers whose meetings differ from the odds rule by a Beta curve of
    this worth (None: not at all). Each meeting, the batter bats until out or 1 to 12
    balls."""
    normal = 0.05
    batters = []
    bowlers = []
    for _ in range(300):
        batters.append(rng.betavariate(0.05 * 250, 0.95 * 250))
        bowlers.append(rng.betavariate(0.05 * 2000, 0.95 * 2000))
    true = {}
    records = {}
    for _ in range(60_000):
        i = rng.randrange(300)
        j = rng.randrange(300)
        if (i, j) not in true:
            rate = expected(batters[i], bowlers[j], normal)
            if worth is not None:
                rate = rng.betavariate(rate * worth, (1 - rate) * worth)
            true[(i, j)] = rate
            records[(i, j)] = [0, 0]
        record = records[(i, j)]
        for _ in range(rng.randint(1, 12)):
            record[1] += 1
            if rng.random() < true[(i, j)]:
                record[0] += 1
                break
    pairs = []
    for (i, j), (outs, faced) in records.items():
        pairs.append(
            Pair(batter=str(i), bowler=str(j), order="top", outs=outs, faced=faced)
        )
    batter_rates = {}
    bowler_rates = {}
    for i in range(300):
        batter_rates[str(i)] = batters[i]
        bowler_rates[str(i)] = bowlers[i]
    return pairs, batter_rates, bowler_rates


def test_how_much_meetings_matter_is_found_from_the_data():
    rng = random.Random(3)
    # meetings that add nothing: the fit puts them worth far more than any meeting
    pairs, batters, bowlers = made_up_meetings(rng, None)
    assert meetings_worth(pairs, batters, bowlers, 0.05) > 500
    # meetings worth 50 balls: found near 50 (the method of moments said 6, Day 51)
    pairs, batters, bowlers = made_up_meetings(rng, 50)
    assert 30 < meetings_worth(pairs, batters, bowlers, 0.05) < 80


# The pictures: paired overs


def test_paired_overs_stay_pairs():
    # replay by replay: (against this bowler, against that one); 0 is survived
    this = [0, 3, 1, 0, 2, 0]
    that = [0, 0, 1, 4, 0, 0]
    # sorted by this, then that: (1, 1), (2, 0), (3, 0), (0, 4), (0, 0), (0, 0);
    # every second kept, from the middle of each step
    assert representative_pairs(this, that, 3) == ((2, 0, 0), (0, 4, 0))


def test_the_pictures_come_with_both_sides():
    found = matchup(field(), "K", "B", "S", 6)
    assert len(found.this.overs) == 100
    assert len(found.that.overs) == 100
    out_this = 0
    out_that = 0
    for fell in found.this.overs:
        if fell:
            out_this += 1
    for fell in found.that.overs:
        if fell:
            out_that += 1
    assert abs(out_this - 39) <= 3  # 39% out against B
    assert abs(out_that - 11) <= 3  # 11% against S
    assert len(found.this.survival) == 7


def test_each_pair_of_overs_shares_its_luck():
    # the same balls against both: B (8%) gets him out on that ball or sooner whenever
    # S (2%) does, never later
    found = matchup(field(), "K", "B", "S", 6)
    for this, that in zip(found.this.overs, found.that.overs, strict=True):
        if that:
            assert this
            assert this <= that


# No answer


def test_no_answer_without_a_question_or_records():
    pairs = field()
    assert matchup(pairs, "K", "B", "B", 6) is None  # a bowler instead of himself
    assert matchup(pairs, "K", "B", "S", 0) is None  # within no balls
    assert matchup(pairs, "nobody", "B", "S", 6) is None  # a batter with no balls
    assert matchup(pairs, "K", "B", "nobody", 6) is None  # a bowler with no balls
    assert matchup(pairs[:100], "K", "B", "S", 6) is None  # too few to tell normal
