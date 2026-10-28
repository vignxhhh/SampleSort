"""Tests for rack occupancy tracking and slot allocation."""

from __future__ import annotations

import pytest

from samplesort.config import SampleSortConfig
from samplesort.planning.rack_state import RackFullError, RackState


@pytest.fixture
def rack_state(config: SampleSortConfig) -> RackState:
    """A fresh, empty rack state built from the shipped workspace."""
    return RackState(config.workspace, config.classes)


def test_starts_completely_empty(rack_state: RackState, config: SampleSortConfig) -> None:
    assert rack_state.total_occupied == 0
    assert rack_state.total_free == config.workspace.total_slots
    for rack in config.workspace.racks:
        assert rack_state.free_slots(rack.id) == list(range(rack.num_slots))


def test_occupy_fills_slots_in_order(rack_state: RackState, config: SampleSortConfig) -> None:
    rack_id = config.rack_for_class("red")
    for expected_index in range(config.workspace.rack(rack_id).num_slots):
        assignment = rack_state.occupy("red")
        assert assignment.rack_id == rack_id
        assert assignment.slot_index == expected_index
        assert assignment.class_label == "red"


def test_assignment_carries_the_slot_coordinates(
    rack_state: RackState, config: SampleSortConfig
) -> None:
    assignment = rack_state.occupy("blue")
    rack = config.workspace.rack(assignment.rack_id)
    assert assignment.xy == rack.slots[assignment.slot_index]


def test_next_free_slot_does_not_reserve(rack_state: RackState) -> None:
    first = rack_state.next_free_slot("red")
    second = rack_state.next_free_slot("red")
    assert first == second
    assert rack_state.total_occupied == 0


def test_occupancy_reflects_reservations(rack_state: RackState, config: SampleSortConfig) -> None:
    rack_id = config.rack_for_class("blue")
    rack_state.occupy("blue")
    occupancy = rack_state.occupancy(rack_id)
    assert occupancy[0] == "blue"
    assert occupancy[1] is None
    assert rack_state.is_occupied(rack_id, 0)
    assert not rack_state.is_occupied(rack_id, 1)


def test_occupancy_returns_a_copy(rack_state: RackState, config: SampleSortConfig) -> None:
    rack_id = config.rack_for_class("blue")
    snapshot = rack_state.occupancy(rack_id)
    snapshot[0] = "tampered"
    assert rack_state.occupancy(rack_id)[0] is None


def test_full_rack_raises_a_clear_error(rack_state: RackState, config: SampleSortConfig) -> None:
    rack_id = config.rack_for_class("red")
    rack = config.workspace.rack(rack_id)
    for _ in range(rack.num_slots):
        rack_state.occupy("red")

    assert not rack_state.has_room_for("red")
    with pytest.raises(RackFullError) as excinfo:
        rack_state.occupy("red")
    message = str(excinfo.value)
    assert rack.label in message
    assert "red" in message
    assert str(rack.num_slots) in message


def test_classes_sharing_a_rack_compete_for_its_slots(
    rack_state: RackState, config: SampleSortConfig
) -> None:
    # green and yellow both map to rack_c in the shipped config.
    shared = {label for label in config.classes.labels if config.rack_for_class(label) == "rack_c"}
    assert shared == {"green", "yellow"}, "this test assumes green and yellow share rack_c"

    capacity = config.workspace.rack("rack_c").num_slots
    for index in range(capacity):
        label = "green" if index % 2 == 0 else "yellow"
        rack_state.occupy(label)
    assert rack_state.count_free("rack_c") == 0
    with pytest.raises(RackFullError):
        rack_state.occupy("yellow")


def test_filling_one_rack_does_not_block_another(
    rack_state: RackState, config: SampleSortConfig
) -> None:
    for _ in range(config.workspace.rack(config.rack_for_class("red")).num_slots):
        rack_state.occupy("red")
    assert rack_state.has_room_for("blue")
    assert rack_state.occupy("blue").slot_index == 0


def test_release_frees_a_slot_for_reuse(rack_state: RackState) -> None:
    assignment = rack_state.occupy("red")
    rack_state.release(assignment.rack_id, assignment.slot_index)
    assert not rack_state.is_occupied(assignment.rack_id, assignment.slot_index)
    assert rack_state.occupy("red").slot_index == assignment.slot_index


def test_release_rejects_unknown_racks_and_slots(rack_state: RackState) -> None:
    with pytest.raises(KeyError, match="unknown rack id"):
        rack_state.release("rack_zzz", 0)
    with pytest.raises(IndexError, match="has no slot"):
        rack_state.release("rack_a", 99)


def test_occupancy_of_an_unknown_rack_raises(rack_state: RackState) -> None:
    with pytest.raises(KeyError, match="unknown rack id"):
        rack_state.occupancy("rack_zzz")


def test_reset_empties_everything(rack_state: RackState, config: SampleSortConfig) -> None:
    for label in config.classes.labels:
        rack_state.occupy(label)
    assert rack_state.total_occupied > 0
    rack_state.reset()
    assert rack_state.total_occupied == 0


def test_counts_stay_consistent(rack_state: RackState, config: SampleSortConfig) -> None:
    rack_id = config.rack_for_class("red")
    rack_state.occupy("red")
    rack_state.occupy("red")
    total = config.workspace.rack(rack_id).num_slots
    assert rack_state.count_occupied(rack_id) == 2
    assert rack_state.count_free(rack_id) == total - 2
    assert rack_state.total_occupied + rack_state.total_free == config.workspace.total_slots


def test_summary_reports_every_rack(rack_state: RackState, config: SampleSortConfig) -> None:
    rack_state.occupy("red")
    summary = rack_state.summary()
    assert set(summary) == {rack.id for rack in config.workspace.racks}
    assert summary[config.rack_for_class("red")].startswith("1/")
