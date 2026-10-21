# Cricsheet JSON: notes from Day 28

The reference sheet for Days 29–32. Everything here was seen in the files, not guessed.

## The snapshot

| | |
|---|---|
| File | `data/raw/all_json.zip` (git-ignored) |
| Downloaded | 2026-10-21, from https://cricsheet.org/downloads/all_json.zip |
| Size | 147,884,302 bytes |
| SHA-256 | `2c0b9c7d4f86da1d1ba939539d40bb435917c360ecf71661daf6f0a2bd5f8573` |
| Matches | 23,132 (plus `README.txt` and `LICENSE.txt`) |
| Withheld by Cricsheet | 380 matches: the Afghanistan men's team, or played in the Afghanistan Premier League |
| Format versions | 1.2.0 and 1.3.0, both in the same zip |
| Licence | ODC Attribution License (ODC-By): free to use; the paper must credit Cricsheet |

## The five matches studied (`data/day28/`)

Each has at least one wide, one no-ball, one wicket and one review.

| Kind | File | Date | Match | Format |
|---|---|---|---|---|
| T20I, men | `1550709.json` | 2026-10-11 | Pakistan v Sri Lanka, Rawalpindi | 1.3.0 |
| IPL | `1535463.json` | 2026-05-27 | Rajasthan Royals v Sunrisers Hyderabad | 1.2.0 |
| ODI, men | `1529229.json` | 2026-10-03 | India v West Indies | 1.3.0 |
| Test, men | `1525659.json` | 2026-10-09 to 13 | Australia v South Africa (4 innings, 1 declared) | 1.3.0 |
| T20I, women | `1490709.json` | 2026-07-05 | England v Australia, Women's T20 World Cup | 1.2.0 |

## The shape of one match

```
match.json
├── meta      data_version "1.3.0", created, revision
├── info      who, where, when: teams, venue, city, dates (a list), match_type (T20, ODI, Test...),
│             gender, team_type (international or club), event, toss, outcome, overs,
│             balls_per_over, players (11 per team), registry (a short id per person), officials
└── innings   a list: 2 in a T20 or an ODI, up to 4 in a Test
     ├── team         "Pakistan"
     ├── powerplays   [{"from": 0.1, "to": 5.6, "type": "mandatory"}]   (limited overs)
     ├── declared     true                                               (a Test, when declared)
     └── overs        [{"over": 0, "deliveries": [ball, ball, ...]}, {"over": 1, ...}, ...]
```

## One ball

```
{
  "actual_delivery": "9.4",           <- over 9 (the 10th over), ball 4
  "batter": "Maaz Sadaqat",           <- on strike
  "bowler": "E Malinga",
  "non_striker": "Sahibzada Farhan",  <- at the other end
  "extras": {"wides": 1},             <- only when there are extras
  "runs": {
    "batter": 0,                      <- off the bat
    "extras": 1,                      <- wides, no-balls, byes, leg-byes, penalties
    "total": 1                        <- batter + extras
  },
  "wickets": [ ... ],                 <- only when someone is out
  "review": { ... }                   <- only when there was a DRS review
}
```

## What matters for Day 30 and the engines

1. **Overs count from 0.** Over 0 is the first over, so the T20 powerplay is overs 0–5; the
   file even says so (`powerplays` from 0.1 to 5.6).
2. **A wide or a no-ball is bowled again with the same number.** Over 9 of the T20I has seven
   entries: 9.1, 9.2, 9.3, 9.4 (a wide), 9.4 again, 9.5, 9.6. The number cannot identify a
   ball; its position in the list does. An over can have 7, 8 or more entries.
3. **A wide is not a ball faced; a no-ball is.** The batter can score off a no-ball: at 12.1,
   1 run off the bat plus 1 no-ball makes 2. A strike rate (Day 38) counts no-balls as balls
   faced, but not wides.
4. **`runs.total` = `runs.batter` + `runs.extras`.** Day 32 checks this, and that the balls add
   up to each innings total.
5. **`wickets` is a list.** Each wicket has `player_out`, `kind` ("caught", "bowled", "lbw",
   "run out", ...) and sometimes `fielders`. The player out need not be the striker (a run out
   at the other end).
6. **`review`** has `by` (the team), `umpire`, `batter` (the one reviewed: at 13.6 of the T20I
   it was the non-striker, not the striker), `decision` ("struck down" or "upheld") and `type`.
7. **Names** are Cricsheet register names ("B Fernando", "PHKD Mendis"), and `info.registry`
   gives each person a short id ("Abrar Ahmed" -> "abb7c76c"). Day 33 matches "Kohli" to
   "V Kohli" with them.
8. **Two format versions.** Both 1.2.0 and 1.3.0 have `actual_delivery`; the reader on Day 30 is
   tested on both.
