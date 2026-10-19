"""tools/same_match.py: the k-NN check behind Day 16's answer."""
import numpy as np

from tools.same_match import compare, vote


def test_a_tie_goes_to_the_first_class():
    assert vote(np.array([2, 1, 2, 1]), 3) == 1
    assert vote(np.array([0, 2, 2]), 3) == 2


def test_dropping_a_wrong_own_match_neighbour_fixes_the_vote():
    # train: 0 and 1 from match "a" (shot 1), 2 and 3 from match "b" (shot 0)
    y_train = np.array([1, 1, 0, 0])
    m_train = np.array(["a", "a", "b", "b"])
    # one val clip, shot 0, from match "a": its own match is nearest
    neighbours = np.array([[0, 1, 2, 3]])
    found = compare(neighbours, np.array([0]), np.array(["a"]), y_train,
                    m_train, n_classes=2, k=2)
    assert found["top1_all"] == 0.0
    assert found["top1_without_own_match"] == 1.0
    assert found["nearest_is_own_match"] == 1.0
    assert found["own_match_share_of_top_k"] == 1.0
    assert found["own_match_neighbours_wrong"] == 1.0


def test_with_no_own_match_in_train_nothing_changes():
    y_train = np.array([0, 1, 0])
    m_train = np.array(["b", "c", "d"])
    neighbours = np.array([[0, 1, 2], [1, 0, 2]])
    found = compare(neighbours, np.array([0, 1]), np.array(["a", "a"]),
                    y_train, m_train, n_classes=2, k=1)
    assert found["top1_all"] == found["top1_without_own_match"] == 1.0
    assert found["nearest_is_own_match"] == 0.0
    assert found["own_match_neighbours_wrong"] is None
