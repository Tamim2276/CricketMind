"""Day 60: Shot DNA (N6), before anything reads the clip."""

from cricketmind_v3.domain.shot_dna import NO_CLIP, NOT_READ_YET, stub_shot_dna
from cricketmind_v3.domain.subject import CricketSubject

KOHLI = CricketSubject(player="V Kohli", shot="Cover Drive")


def test_a_question_in_words_has_nothing_seen():
    dna = stub_shot_dna(KOHLI, None)
    assert dna.subject == KOHLI
    assert dna.clip is None
    assert dna.label is None
    assert dna.confidence is None
    assert dna.posture == ()
    assert dna.empty_because == NO_CLIP


def test_a_clip_is_kept_but_not_read_yet():
    dna = stub_shot_dna(KOHLI, "clip.mp4")
    assert dna.clip == "clip.mp4"
    assert dna.label is None  # never guessed from the question's own shot
    assert dna.posture == ()
    assert dna.empty_because == NOT_READ_YET
