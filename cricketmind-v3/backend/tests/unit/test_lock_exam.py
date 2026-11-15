"""The exam's lock (D-070): it joins the friends' files and never shows a question."""

import csv
import hashlib

import pytest

from cricketmind_v3.pipeline import lock_exam
from cricketmind_v3.pipeline.lock_exam import (
    COLUMNS,
    ExamLockedError,
    exam_is_locked,
    lock,
    require_locked_exam,
)

SECRET = "SECRETWORD"  # in every made-up question: it must never be printed


def friend_rows(friend: int, times: int = 2) -> list[list[str]]:
    # a made-up friend's sheet, 30 rows a time, as the guide's ready sheet lays it out
    rows = []
    kinds = ("stats", "prediction", "video", "rules", "cant_check")
    group = 1
    for _ in range(times):
        for kind in kinds:
            for wording in range(4):
                question = f"{SECRET} friend {friend} A{group} way {wording}"
                rows.append(
                    [f"A{group}", kind, question, f"{SECRET} answer it", "", ""]
                )
            group += 1
    for number in range(1, 5 * times + 1):
        question = f"{SECRET} friend {friend} B{number}"
        rows.append([f"B{number}", "", question, f"{SECRET} be honest", "", ""])
    for number in range(1, 5 * times + 1):
        kind = "false_premise" if number <= 3 * times else "trick"
        question = f"{SECRET} friend {friend} C{number}"
        rows.append([f"C{number}", kind, question, f"{SECRET} correct it", "", ""])
    return rows


def write_friend(folder, name, rows, encoding="utf-8-sig", delimiter=","):
    raw = folder / "raw"
    raw.mkdir(exist_ok=True)
    with open(raw / name, "w", encoding=encoding, newline="") as out:
        table = csv.writer(out, delimiter=delimiter)
        table.writerow(COLUMNS)
        table.writerows(rows)


def two_friends(folder):
    write_friend(folder, "friend1.csv", friend_rows(1))
    write_friend(folder, "friend2.csv", friend_rows(2))


def run(folder, capsys) -> tuple[int, str]:
    code = lock(folder, folder / "no_guide.html")
    return code, capsys.readouterr().out


def test_two_friends_of_sixty_lock_the_exam(tmp_path, capsys):
    two_friends(tmp_path)
    code, said = run(tmp_path, capsys)
    assert code == 0
    assert "2 files, 120 rows" in said
    assert "Part A: 80" in said
    assert "Part B: 20" in said
    assert "Part C: 20" in said
    assert "LOCKED" in said


def test_four_friends_of_thirty_lock_it_too(tmp_path, capsys):
    for friend in range(1, 5):
        write_friend(tmp_path, f"friend{friend}.csv", friend_rows(friend, times=1))
    code, said = run(tmp_path, capsys)
    assert code == 0
    assert "4 files, 120 rows" in said


def test_the_printed_sha_is_the_files_own(tmp_path, capsys):
    two_friends(tmp_path)
    _, said = run(tmp_path, capsys)
    content = (tmp_path / "final_exam.csv").read_bytes()
    assert f"SHA-256: {hashlib.sha256(content).hexdigest()}" in said


def shows_nothing_of_the_exam(said: str) -> bool:
    # in any case: a printed cell in lower case is still a leak (the first version of this
    # test looked for capitals, and missed the kind column being printed)
    return SECRET.lower() not in said.lower() and "friend 1" not in said.lower()


def test_it_never_prints_a_question_when_it_works(tmp_path, capsys):
    two_friends(tmp_path)
    _, said = run(tmp_path, capsys)
    assert shows_nothing_of_the_exam(said)


def test_it_never_prints_a_question_when_it_fails(tmp_path, capsys):
    # every kind of problem at once, with the secret word in every cell it could print
    rows = friend_rows(1)
    rows[0][1] = f"{SECRET} a wrong kind"  # Part A, a kind that is none
    rows[1][2] = ""  # no question, and its handling holds the word
    rows[2][3] = ""  # no handling, and its question holds the word
    rows[3][0] = f"{SECRET} not a group"  # a label that is no group
    rows[4][2] = rows[5][2]  # the same question twice
    rows[8][0] = f"A{SECRET}"  # starts like Part A: group_problems must not say it
    rows[40][1] = f"{SECRET} a wrong kind"  # Part B
    rows[50][1] = f"{SECRET} a wrong kind"  # Part C
    write_friend(tmp_path, "friend1.csv", rows)
    write_friend(tmp_path, "friend2.csv", friend_rows(2)[:50])  # too few rows
    code, said = run(tmp_path, capsys)
    assert code == 1
    assert "NOT LOCKED" in said
    assert shows_nothing_of_the_exam(said)


def test_a_kind_that_is_none_is_counted_and_not_said(tmp_path, capsys):
    rows = friend_rows(1)
    rows[40][1] = "something someone typed"
    write_friend(tmp_path, "friend1.csv", rows)
    write_friend(tmp_path, "friend2.csv", friend_rows(2))
    _, said = run(tmp_path, capsys)
    assert "(not a kind): 1" in said
    assert "typed" not in said


def test_a_file_that_appears_at_the_last_moment_is_not_written_over(
    tmp_path, capsys, monkeypatch
):
    # the check at the top passed; someone else locks it before the write
    two_friends(tmp_path)
    real = lock_exam.joined_exam

    def and_someone_else_locks_it(rows):
        (tmp_path / "final_exam.csv").write_bytes(b"theirs")
        return real(rows)

    monkeypatch.setattr(lock_exam, "joined_exam", and_someone_else_locks_it)
    with pytest.raises(FileExistsError):
        run(tmp_path, capsys)
    assert (tmp_path / "final_exam.csv").read_bytes() == b"theirs"


def test_each_friends_groups_are_kept_apart(tmp_path, capsys):
    two_friends(tmp_path)
    run(tmp_path, capsys)
    with open(tmp_path / "final_exam.csv", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 120
    assert rows[0]["writer"] == "w1"
    assert rows[0]["group"] == "w1-A1"
    assert rows[60]["writer"] == "w2"
    assert rows[60]["group"] == "w2-A1"
    assert rows[0]["id"] == "1"
    assert rows[119]["id"] == "120"


def test_the_file_has_the_same_bytes_on_every_machine(tmp_path, capsys):
    two_friends(tmp_path)
    run(tmp_path, capsys)
    content = (tmp_path / "final_exam.csv").read_bytes()
    assert b"\r" not in content
    assert not content.startswith(b"\xef\xbb\xbf")


def test_it_will_not_write_a_locked_exam_again(tmp_path, capsys):
    two_friends(tmp_path)
    run(tmp_path, capsys)
    first = (tmp_path / "final_exam.csv").read_bytes()
    code, said = run(tmp_path, capsys)
    assert code == 1
    assert "never written again" in said
    assert (tmp_path / "final_exam.csv").read_bytes() == first


def test_too_few_rows_writes_nothing(tmp_path, capsys):
    write_friend(tmp_path, "friend1.csv", friend_rows(1))
    code, said = run(tmp_path, capsys)
    assert code == 1
    assert "Part A has 40 rows, not 80" in said
    assert "NOT LOCKED" in said
    assert not (tmp_path / "final_exam.csv").exists()


def test_a_missing_question_names_its_row(tmp_path, capsys):
    rows = friend_rows(1)
    rows[6][2] = ""
    write_friend(tmp_path, "friend1.csv", rows)
    write_friend(tmp_path, "friend2.csv", friend_rows(2))
    code, said = run(tmp_path, capsys)
    assert code == 1
    assert "friend1.csv, row 8: no question" in said  # the column names are row 1


def test_a_missing_handling_names_its_row(tmp_path, capsys):
    rows = friend_rows(2)
    rows[0][3] = "  "
    write_friend(tmp_path, "friend1.csv", friend_rows(1))
    write_friend(tmp_path, "friend2.csv", rows)
    code, said = run(tmp_path, capsys)
    assert code == 1
    assert "friend2.csv, row 2: no expected_handling" in said


def test_a_group_with_three_wordings_is_named(tmp_path, capsys):
    rows = friend_rows(1)
    rows[0][0] = "A99"
    write_friend(tmp_path, "friend1.csv", rows)
    write_friend(tmp_path, "friend2.csv", friend_rows(2))
    code, said = run(tmp_path, capsys)
    assert code == 1
    assert "group A1 has 3 wordings, not 4" in said


def test_a_group_of_two_kinds_is_named(tmp_path, capsys):
    rows = friend_rows(1)
    rows[0][1] = "rules"
    write_friend(tmp_path, "friend1.csv", rows)
    write_friend(tmp_path, "friend2.csv", friend_rows(2))
    code, said = run(tmp_path, capsys)
    assert code == 1
    assert "group A1 mixes kinds" in said


def test_part_c_must_be_tricky(tmp_path, capsys):
    rows = friend_rows(1)
    rows[-1][1] = "stats"
    write_friend(tmp_path, "friend1.csv", rows)
    write_friend(tmp_path, "friend2.csv", friend_rows(2))
    code, said = run(tmp_path, capsys)
    assert code == 1
    assert "Part C is false_premise or trick" in said


def test_part_b_may_leave_its_kind_empty(tmp_path, capsys):
    # the ready sheet leaves it empty, and tells them only to fill the questions
    two_friends(tmp_path)
    code, said = run(tmp_path, capsys)
    assert code == 0
    assert "(left empty): 20" in said


def test_a_bad_group_label_is_named(tmp_path, capsys):
    rows = friend_rows(1)
    rows[40][0] = "question one"
    write_friend(tmp_path, "friend1.csv", rows)
    write_friend(tmp_path, "friend2.csv", friend_rows(2))
    code, said = run(tmp_path, capsys)
    assert code == 1
    assert "friend1.csv, row 42: the group must be like A1" in said


def test_the_same_question_twice_is_named(tmp_path, capsys):
    first = friend_rows(1)
    second = friend_rows(2)
    second[40][2] = first[40][2].upper()  # case and spaces do not make it new
    write_friend(tmp_path, "friend1.csv", first)
    write_friend(tmp_path, "friend2.csv", second)
    code, said = run(tmp_path, capsys)
    assert code == 1
    assert "friend2.csv, row 42: the same question as friend1.csv, row 42" in said


def test_a_question_copied_from_the_guide_is_named(tmp_path, capsys):
    guide = tmp_path / "guide.html"
    guide.write_text('<li class="q">"Can you be run out on a free hit?"</li>', "utf-8")
    rows = friend_rows(1)
    rows[40][2] = "can you be  run out on a FREE hit?"
    write_friend(tmp_path, "friend1.csv", rows)
    write_friend(tmp_path, "friend2.csv", friend_rows(2))
    code = lock(tmp_path, guide)
    said = capsys.readouterr().out
    assert code == 1
    assert "friend1.csv, row 42: copied from the guide's examples" in said


def test_the_real_guides_examples_are_noticed(tmp_path, capsys):
    rows = friend_rows(1)
    rows[40][2] = "Why is Shakib so bad at the death?"  # the guide's own
    write_friend(tmp_path, "friend1.csv", rows)
    write_friend(tmp_path, "friend2.csv", friend_rows(2))
    code = lock(tmp_path, lock_exam.GUIDE)
    assert code == 1
    assert "copied from the guide" in capsys.readouterr().out


def test_a_missing_column_is_named(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "friend1.csv").write_text("group,kind,question\nA1,stats,x\n", "utf-8")
    code, said = run(tmp_path, capsys)
    assert code == 1
    assert "no column called expected_handling, measure, filters" in said


def test_no_files_is_said(tmp_path, capsys):
    code, said = run(tmp_path, capsys)
    assert code == 1
    assert "no CSV files" in said


def test_excels_other_ways_of_saving_are_read(tmp_path, capsys):
    # a semicolon between columns, and Excel's own code page
    write_friend(tmp_path, "friend1.csv", friend_rows(1), "cp1252", ";")
    write_friend(tmp_path, "friend2.csv", friend_rows(2), "utf-8", ",")
    code, _ = run(tmp_path, capsys)
    assert code == 0


def test_empty_rows_at_the_end_of_a_sheet_are_left_out(tmp_path, capsys):
    rows = friend_rows(1)
    rows.append(["", "", "", "", "", ""])
    rows.append(["", "", "", "", "", ""])
    write_friend(tmp_path, "friend1.csv", rows)
    write_friend(tmp_path, "friend2.csv", friend_rows(2))
    code, said = run(tmp_path, capsys)
    assert code == 0
    assert "120 rows" in said


# The guard on real answers (D-070)


def test_a_real_run_is_refused_before_the_lock(tmp_path):
    assert not exam_is_locked(tmp_path)
    with pytest.raises(ExamLockedError, match="lock the exam first"):
        require_locked_exam(tmp_path)


def test_a_real_run_goes_ahead_after_the_lock(tmp_path, capsys):
    two_friends(tmp_path)
    run(tmp_path, capsys)
    assert exam_is_locked(tmp_path)
    require_locked_exam(tmp_path)
