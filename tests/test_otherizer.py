import pandas as pd
import pytest

from otherizer import Otherizer


def create_input_data():
    return pd.DataFrame(
        {
            "a": ["a", "a", "b", "b", "c", "d"],
            "b": ["x", "y", "x", "x", "z", "z"],
            "c": ["dog", "cat", "dog", "fish", "fish", "fish"],
            "d": ["blue", "red", "green", "green", "yellow", "yellow"],
            "e": ["sheep", "sheep", "sheep", "sheep", "sheep", "sheep"],
        }
    )


@pytest.mark.parametrize(
    "i, col, thresh, expected",
    [
        (0, "a", 0.2, ["a", "a", "b", "b", "other", "other"]),
        (1, "b", 0.2, ["x", "other", "x", "x", "z", "z"]),
        (2, "c", 0.2, ["dog", "other", "dog", "fish", "fish", "fish"]),
        (3, "d", 0.2, ["other", "other", "green", "green", "yellow", "yellow"]),
        (4, "e", 0.2, ["sheep", "sheep", "sheep", "sheep", "sheep", "sheep"]),
        (5, "a", 1, ["other", "other", "other", "other", "other", "other"]),
        (6, "b", 1, ["other", "other", "other", "other", "other", "other"]),
        (7, "c", 1, ["other", "other", "other", "other", "other", "other"]),
        (8, "d", 1, ["other", "other", "other", "other", "other", "other"]),
        (9, "e", 1, ["sheep", "sheep", "sheep", "sheep", "sheep", "sheep"]),
        (10, "e", 0.1, ["sheep", "sheep", "sheep", "sheep", "sheep", "sheep"]),
        (
            11,
            "a",
            0.1,
            ["a", "a", "b", "b", "c", "d"],
        ),
        (12, "b", 0.1, ["x", "y", "x", "x", "z", "z"]),
        (13, "c", 0.1, ["dog", "cat", "dog", "fish", "fish", "fish"]),
        (14, "d", 0.1, ["blue", "red", "green", "green", "yellow", "yellow"]),
    ],
)
def test_encodes_expected_vals(i, col, thresh, expected):
    input_data = create_input_data()
    encoder = Otherizer(threshold=thresh)
    transformed_data = encoder.fit_transform(input_data)

    actual = transformed_data[col].tolist()
    assert actual == expected
