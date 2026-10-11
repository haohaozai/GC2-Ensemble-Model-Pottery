import pytest

from text_branch.cones import ConeError, ConeScale
from text_branch.config import ORTON_CONE_ORDER


@pytest.fixture
def scale() -> ConeScale:
    return ConeScale(ORTON_CONE_ORDER)


def test_06_and_6_are_different_cones(scale):
    low_fire, mid_fire = scale.parse("06"), scale.parse("6")
    assert low_fire.label == "06" and mid_fire.label == "6"
    assert low_fire != mid_fire
    assert low_fire.high_index < mid_fire.low_index


def test_ordering_is_ceramic_not_numeric(scale):
    expected = ["010", "09", "06", "05 1/2", "05", "04", "01", "1", "5", "5 1/2", "6", "10"]
    assert sorted(expected, key=scale.index) == expected
    # Numerically 010 > 09 and 06 > 04, but they are cooler cones.
    assert scale.index("010") < scale.index("09")
    assert scale.index("06") < scale.index("04")


@pytest.mark.parametrize("raw, low, high", [
    ("9 - 11", "9", "11"),
    ("04 - 6", "04", "6"),
    ("08 - 05 1/2", "08", "05 1/2"),
    ("010 - 09", "010", "09"),
])
def test_valid_ranges(scale, raw, low, high):
    cone = scale.parse(raw)
    assert (cone.low, cone.high) == (low, high)
    assert cone.label == raw


@pytest.mark.parametrize("raw, code", [
    ("6 - 04", "cone_range_reversed"),
    ("09 - 010", "cone_range_reversed"),
    ("10 - 6", "cone_range_reversed"),
    ("15", "cone_unrecognised"),
    ("hot", "cone_unrecognised"),
    ("- 6", "cone_malformed"),
    ("6 - 8 - 10", "cone_malformed"),
])
def test_bad_values_raise_with_flag_code(scale, raw, code):
    with pytest.raises(ConeError) as excinfo:
        scale.parse(raw)
    assert excinfo.value.code == code


@pytest.mark.parametrize("raw, label", [
    ("cone 6", "6"), ("Cone 06", "06"), ("^6", "6"), ("5½", "5 1/2"), ("5.5", "5 1/2"),
    ("05 1/2", "05 1/2"), ("cone 9 to 10", "9 - 10"), ("  6  ", "6"),
])
def test_user_spellings(scale, raw, label):
    assert scale.parse(raw).label == label


def test_containment_and_overlap(scale):
    recipe = scale.parse("9 - 11")
    assert recipe.contains(scale.parse("10"))
    assert recipe.contains(scale.parse("9 - 10"))
    assert not recipe.contains(scale.parse("6"))
    assert not recipe.contains(scale.parse("8 - 10"))
    assert recipe.overlaps(scale.parse("8 - 10"))
    assert not scale.parse("6").contains(scale.parse("06"))


def test_text_wording(scale):
    assert scale.parse("9 - 10").text == "9 to 10"
    assert scale.parse("06").text == "06"


def test_duplicate_order_rejected():
    with pytest.raises(ValueError):
        ConeScale(["6", "6"])
