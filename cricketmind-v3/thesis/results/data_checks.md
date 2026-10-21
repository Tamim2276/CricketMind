# Data checks (Phase 3)

| Check | Rule | Rows | Verdict |
|---|---|---|---|
| ball totals | a ball's total is the batter's runs plus the extras | 0 | ok |
| extras add up | the extras are the wides, no-balls, byes, leg-byes and penalties | 0 | ok |
| targets | a chase's target is the first innings' score plus one (no rain rule) | 3 | PROBLEM |
| winning margins | 'won by N runs' is the first innings' score minus the second's | 4 | PROBLEM |
| ball counts | every loaded match has as many balls stored as were counted | 0 | ok |
| match rows | every loaded match has a row in the matches table | 0 | ok |
| empty names | every ball names a batter, a bowler and a non-striker | 0 | ok |
| names without an id | every batter and bowler has a register id in that match | 0 | ok |
| unreadable matches | every match file could be read | 0 | ok |
| overs with too many legal balls | (a count) overs with more legal balls than the format allows | 319 | counted |
| powerplays unlike the rule | (a count) innings whose own powerplay ends in another over than the rule says: shortened matches | 2901 | counted |
| balls with two wickets | (a count) balls on which more than one wicket fell | 17 | counted |

Examples, targets: [('1494140', 143, 144), ('1457218', 169, 167), ('1318395', 63, 64)]

Examples, winning margins: [('1506668', 12, 11), ('1495657', 46, 45), ('1536405', 83, 84), ('1318396', 102, 103)]

Examples, overs with too many legal balls: [('518071', 1, 9, 7), ('1152540', 2, 1, 7), ('1539515', 1, 4, 7), ('249217', 3, 63, 7), ('446967', 2, 36, 7)]

Examples, powerplays unlike the rule: [('1002147', 1, 7.6, 'ODM'), ('1002147', 2, 7.6, 'ODM'), ('1003861', 2, 1.6, 'T20'), ('1003869', 1, 2.6, 'T20'), ('1003869', 2, 0.6, 'T20')]

Examples, balls with two wickets: [('1068620', 3, 56, 6), ('1130684', 2, 12, 2), ('1208093', 2, 19, 5), ('1383095', 2, 14, 2), ('1384429', 1, 24, 2)]
