"""Tests for the HSV cap classifier and the QR sample-ID reader.

Every image used here is synthesised inside the test, so the suite needs no
fixture files, no camera and no simulator.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from samplesort.config import SampleSortConfig
from samplesort.perception.classifier import ColorClassifier, QRReader

#: Unambiguous BGR swatches for each configured class.
SWATCHES = {
    "red": (40, 40, 200),
    "blue": (200, 90, 40),
    "green": (60, 170, 60),
    "yellow": (50, 200, 220),
}
#: Colours that must not classify as any sample class.
NON_SAMPLE = {
    "white": (255, 255, 255),
    "black": (8, 8, 8),
    "mid grey": (128, 128, 128),
    "rack grey": (160, 160, 163),
}


@pytest.fixture
def classifier(config: SampleSortConfig) -> ColorClassifier:
    """A classifier bound to the shipped taxonomy."""
    return ColorClassifier(config.classes)


def test_labels_match_the_config(classifier: ColorClassifier, config: SampleSortConfig) -> None:
    assert classifier.labels == config.classes.labels


@pytest.mark.parametrize("label", sorted(SWATCHES))
def test_each_swatch_classifies_as_its_own_class(classifier: ColorClassifier, label: str) -> None:
    blue, green, red = SWATCHES[label]
    verdict = classifier.classify_bgr(blue, green, red)
    assert verdict is not None, f"{label} swatch was not classified at all"
    assert verdict[0] == label
    assert 0.0 <= verdict[1] <= 1.0


@pytest.mark.parametrize("name", sorted(NON_SAMPLE))
def test_non_sample_colours_are_rejected(classifier: ColorClassifier, name: str) -> None:
    blue, green, red = NON_SAMPLE[name]
    assert classifier.classify_bgr(blue, green, red) is None


def test_pure_red_scores_well_despite_the_wrapped_hue_range(
    classifier: ColorClassifier,
) -> None:
    # Hue 0 sits at the edge of the [0, 8] half of red's split range. After the
    # wrap-around merge it should land near the centre of [172, 188] instead.
    verdict = classifier.classify_bgr(0, 0, 255)
    assert verdict is not None
    assert verdict[0] == "red"
    assert verdict[1] > 0.6


def test_both_halves_of_the_red_range_classify_as_red(classifier: ColorClassifier) -> None:
    for hue in (0, 4, 8, 172, 176, 179):
        verdict = classifier.classify_hsv(hue, 200, 200)
        assert verdict is not None and verdict[0] == "red", f"hue {hue}"


def test_desaturated_colours_are_rejected(classifier: ColorClassifier) -> None:
    # Right hue, but not enough saturation to be a real cap.
    assert classifier.classify_hsv(115, 20, 200) is None


def test_dark_colours_are_rejected(classifier: ColorClassifier) -> None:
    assert classifier.classify_hsv(115, 200, 5) is None


def test_a_clean_cap_scores_higher_than_a_marginal_one(classifier: ColorClassifier) -> None:
    # Hue dead centre with strong saturation and brightness beats a washed-out
    # sample sitting right on every threshold.
    clean = classifier.classify_hsv(115, 250, 250)
    marginal = classifier.classify_hsv(101, 112, 62)
    assert clean is not None and marginal is not None
    assert clean[1] > marginal[1]


def test_saturation_is_not_penalised_for_being_high(classifier: ColorClassifier) -> None:
    strong = classifier.classify_hsv(115, 255, 255)
    weak = classifier.classify_hsv(115, 120, 255)
    assert strong is not None and weak is not None
    assert strong[1] > weak[1]


def test_masks_cover_every_class(classifier: ColorClassifier, config: SampleSortConfig) -> None:
    image = np.zeros((10, 40, 3), dtype=np.uint8)
    for index, label in enumerate(config.classes.labels):
        image[:, index * 10 : (index + 1) * 10] = SWATCHES[label]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    masks = classifier.masks(hsv)
    assert set(masks) == set(config.classes.labels)
    for index, label in enumerate(config.classes.labels):
        mask = masks[label]
        assert mask[:, index * 10 : (index + 1) * 10].all(), f"{label} band was not masked"
        others = np.delete(mask, slice(index * 10, (index + 1) * 10), axis=1)
        assert not others.any(), f"{label} mask leaked onto another band"


def test_classify_region_uses_the_median(classifier: ColorClassifier) -> None:
    # Mostly blue with a few red outliers: the median must still say blue.
    image = np.full((20, 20, 3), SWATCHES["blue"], dtype=np.uint8)
    image[0:3, 0:3] = SWATCHES["red"]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.ones((20, 20), dtype=np.uint8)

    verdict = classifier.classify_region(hsv, mask)
    assert verdict is not None and verdict[0] == "blue"


def test_classify_region_on_an_empty_mask_returns_none(classifier: ColorClassifier) -> None:
    hsv = cv2.cvtColor(np.zeros((8, 8, 3), dtype=np.uint8), cv2.COLOR_BGR2HSV)
    assert classifier.classify_region(hsv, np.zeros((8, 8), dtype=np.uint8)) is None


def test_disabled_qr_reader_returns_nothing() -> None:
    reader = QRReader(enabled=False)
    assert reader.read(np.zeros((64, 64, 3), dtype=np.uint8)) == []


def test_qr_reader_decodes_a_rendered_code() -> None:
    encoder = cv2.QRCodeEncoder.create()
    code = encoder.encode("SAMPLE-0042")
    canvas = np.full((360, 360), 255, dtype=np.uint8)
    resized = cv2.resize(code, (240, 240), interpolation=cv2.INTER_NEAREST)
    canvas[60:300, 60:300] = resized
    frame = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    results = QRReader(enabled=True).read(frame)
    assert [text for text, _ in results] == ["SAMPLE-0042"]
    (_, (u, v)) = results[0]
    assert 150 < u < 210 and 150 < v < 210


def test_qr_reader_on_a_blank_frame_returns_nothing() -> None:
    blank = np.full((128, 128, 3), 255, dtype=np.uint8)
    assert QRReader(enabled=True).read(blank) == []
