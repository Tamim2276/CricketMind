"""Day 62: which measures each agent may use."""

import re
from pathlib import Path

from cricketmind_v3.application.graph import nodes
from cricketmind_v3.application.harness import ROSTER as SYNTHETIC_ROSTER
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.registry import CLAIM_TYPES, check_registry, lookup
from cricketmind_v3.domain.render import METRICS
from cricketmind_v3.domain.roster import NAMES, ROSTER, STAKEHOLDERS

D = Provenance.DESCRIPTIVE
M = Provenance.MEASURED
P = Provenance.PREDICTIVE

# the Prov. column of the v4 roster table, written out by hand from the table
TABLE_PROVENANCE = {
    "N11a": {D},
    "N11b": {D, P},
    "N11c": {D, M},
    "N11d": {D},
    "N11e": {D},
    "N11f": {D},
    "N11g": {D, P},
    "N11h": {D},
    "N11i": {D},
    "N11j": {D, M},
    "N11k": {D},
    "N11l": {D},
    "N11m": set(),
}


def test_every_agent_node_is_on_the_roster_and_nobody_else():
    found = set()
    for agent in nodes.AGENTS:
        found.add(agent.__name__.split("_")[0].capitalize())  # n11b_... -> N11b
    assert set(ROSTER) == found


def test_every_measure_named_is_on_the_menu():
    for agent, names in ROSTER.items():
        for name in names:
            assert lookup(name) is not None, (agent, name)


def test_no_agent_lists_a_measure_twice():
    for agent, names in ROSTER.items():
        assert len(names) == len(set(names)), agent


def test_every_measure_on_the_menu_has_someone_to_say_it():
    # no owner: C1 would refuse it from every agent, and the question stays unanswered
    owned = set()
    for names in ROSTER.values():
        owned.update(names)
    for row in CLAIM_TYPES:
        assert row.name in owned, row.name


def test_each_agent_keeps_the_provenance_of_the_table():
    for agent, names in ROSTER.items():
        kinds = set()
        for name in names:
            kinds.add(lookup(name).provenance)
        assert kinds == TABLE_PROVENANCE[agent], agent


def test_the_start_up_check_passes_for_every_agent():
    # N6's engine comes in Phase 8; nothing else may be wrong
    sources = set()
    for row in CLAIM_TYPES:
        if row.source != "N6":
            sources.add(row.source)
    assert check_registry(ROSTER, sources, set(METRICS)) == [
        "weight_transfer: no engine provides N6"
    ]


def test_the_plans_example_has_both_its_agents():
    # "Kohli vs pace: cover drive or pull?" goes to N11b and N11c
    assert "sr_difference" in ROSTER["N11b"]
    assert "sr_difference" in ROSTER["N11c"]


def test_the_plans_own_stats_question_has_someone_to_say_it():
    # "Kohli's average against Australia since 2020?"
    assert "batting_average" in ROSTER["N11c"]


def test_the_agents_built_on_one_engine_keep_to_it():
    # the plan's Days 69-71: N11f uses M2, N11i M4, N11k the M3 flag
    expected = {"N11f": {"N8.M2"}, "N11i": {"N8.M4"}, "N11k": {"N8.M3"}}
    for agent, engines in expected.items():
        found = set()
        for name in ROSTER[agent]:
            found.add(lookup(name).source)
        assert found == engines, agent


def test_the_bowling_agent_has_the_what_if():
    assert "matchup_difference" in ROSTER["N11g"]  # the plan's Day 69: uses T2


def test_the_explainer_has_no_numbers():
    assert ROSTER["N11m"] == ()


def test_the_synthetic_harness_asks_only_for_what_the_real_roster_allows():
    for agent, names in SYNTHETIC_ROSTER.items():
        for name in names:
            assert name in ROSTER[agent], (agent, name)


def test_every_agent_has_a_name_and_people_it_works_for():
    assert set(NAMES) == set(ROSTER)
    assert set(STAKEHOLDERS) == set(ROSTER)


def test_the_names_are_the_architecture_documents():
    root = Path(__file__).resolve().parents[3]
    source = root / "docs" / "architecture" / "CricketMind_Architecture_v4.tex"
    table = source.read_text(encoding="utf-8")
    for agent in ROSTER:
        if agent == "N11m":
            assert "Anyone asking about rules & \\textbf{Explainer (N11m)}" in table
        else:
            assert f"{STAKEHOLDERS[agent]} & {NAMES[agent]} ({agent})" in table, agent


def test_the_architecture_document_says_the_same_as_the_code():
    # the v4 table once listed words no menu row has; if this fails, correct the table
    # in docs/architecture and compile the PDF again
    root = Path(__file__).resolve().parents[3]
    source = root / "docs" / "architecture" / "CricketMind_Architecture_v4.tex"
    table = source.read_text(encoding="utf-8")
    for agent, names in ROSTER.items():
        found = []
        for line in table.splitlines():
            if f"({agent})" in line and " & " in line:
                for name in re.findall(r"\\texttt\{([^}]*)\}", line):
                    found.append(name.replace("\\_", "_"))
        assert found == list(names), agent
