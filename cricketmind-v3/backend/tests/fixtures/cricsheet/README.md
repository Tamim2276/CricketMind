# Test fixtures: three Cricsheet matches

Copied unchanged from Cricsheet's `all_json.zip` of 2026-10-21 (D-032), for the pipeline tests.

| File | Match | Format |
|---|---|---|
| `1550709.json` | Pakistan v Sri Lanka, men's T20I, Rawalpindi, 2026-10-11 | 1.3.0 |
| `1535463.json` | Rajasthan Royals v Sunrisers Hyderabad, IPL, 2026-05-27 | 1.2.0 |
| `1490709.json` | England v Australia, women's T20 World Cup, London, 2026-07-05 | 1.2.0 |

Data from [Cricsheet](https://cricsheet.org), made available under the
[Open Data Commons Attribution License](https://opendatacommons.org/licenses/by/1-0/).

## `../player_meta_sample.csv`

Ten rows in the shape of the cricketdata package's `player_meta` table (Day 34). The
Cricsheet ids are real (from `1550709.json`); **the batting and bowling styles are made up**
for the tests, written in each of the styles the real file mixes: "Right-hand bat" and
"Right hand Bat", two bowling styles joined by a comma, a blank, a bowling style typed in the
batting column, a repeated id and a missing id.
