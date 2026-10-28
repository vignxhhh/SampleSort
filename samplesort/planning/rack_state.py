"""Track which rack slots are occupied and hand out the next free one."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from samplesort.config import ClassesConfig, WorkspaceConfig

logger = logging.getLogger(__name__)


class RackFullError(RuntimeError):
    """Raised when a class's destination rack has no free slot left."""


@dataclass(frozen=True)
class SlotAssignment:
    """A reserved rack slot.

    Attributes:
        rack_id: The rack the slot belongs to.
        slot_index: Index of the slot within that rack.
        xy: Table-frame position of the slot centre, in metres.
        class_label: The sample class the slot was reserved for.
    """

    rack_id: str
    slot_index: int
    xy: tuple[float, float]
    class_label: str


@dataclass
class RackState:
    """Occupancy bookkeeping across every configured rack.

    Args:
        workspace: Rack layout.
        classes: Taxonomy providing the class-to-rack mapping.
    """

    workspace: WorkspaceConfig
    classes: ClassesConfig
    _occupancy: dict[str, list[str | None]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Start with every slot empty."""
        self._occupancy = {rack.id: [None] * rack.num_slots for rack in self.workspace.racks}

    # ---------------------------------------------------------------- inspection

    def occupancy(self, rack_id: str) -> list[str | None]:
        """Return a copy of one rack's slot contents.

        Args:
            rack_id: The rack to inspect.

        Returns:
            One entry per slot: the class label occupying it, or ``None``.

        Raises:
            KeyError: If the rack id is unknown.
        """
        if rack_id not in self._occupancy:
            raise KeyError(f"unknown rack id '{rack_id}'")
        return list(self._occupancy[rack_id])

    def is_occupied(self, rack_id: str, slot_index: int) -> bool:
        """Whether a specific slot currently holds a tube."""
        return self._occupancy[rack_id][slot_index] is not None

    def free_slots(self, rack_id: str) -> list[int]:
        """Indices of the empty slots in a rack, in order."""
        return [i for i, value in enumerate(self._occupancy[rack_id]) if value is None]

    def count_free(self, rack_id: str) -> int:
        """How many slots are still free in a rack."""
        return len(self.free_slots(rack_id))

    def count_occupied(self, rack_id: str) -> int:
        """How many slots are taken in a rack."""
        return self.workspace.rack(rack_id).num_slots - self.count_free(rack_id)

    @property
    def total_free(self) -> int:
        """How many slots are free across every rack."""
        return sum(self.count_free(rack_id) for rack_id in self._occupancy)

    @property
    def total_occupied(self) -> int:
        """How many slots are taken across every rack."""
        return self.workspace.total_slots - self.total_free

    def has_room_for(self, class_label: str) -> bool:
        """Whether a sample of this class could be placed right now."""
        rack_id = self.classes.by_label(class_label).rack_id
        return self.count_free(rack_id) > 0

    # ---------------------------------------------------------------- allocation

    def next_free_slot(self, class_label: str) -> SlotAssignment:
        """Reserve nothing, but report where a sample of this class would go.

        Args:
            class_label: The sample class to place.

        Returns:
            The slot that :meth:`occupy` would take.

        Raises:
            RackFullError: If the class's destination rack is full.
        """
        rack_id = self.classes.by_label(class_label).rack_id
        free = self.free_slots(rack_id)
        if not free:
            rack = self.workspace.rack(rack_id)
            raise RackFullError(
                f"{rack.label} ({rack_id}) is full: all {rack.num_slots} slots are "
                f"occupied, so the next '{class_label}' sample has nowhere to go"
            )
        slot_index = free[0]
        return SlotAssignment(
            rack_id=rack_id,
            slot_index=slot_index,
            xy=self.workspace.rack(rack_id).slots[slot_index],
            class_label=class_label,
        )

    def occupy(self, class_label: str) -> SlotAssignment:
        """Reserve the next free slot for a sample of this class.

        Args:
            class_label: The sample class to place.

        Returns:
            The newly reserved slot.

        Raises:
            RackFullError: If the class's destination rack is full.
        """
        assignment = self.next_free_slot(class_label)
        self._occupancy[assignment.rack_id][assignment.slot_index] = class_label
        logger.debug(
            "reserved %s slot %d for %s",
            assignment.rack_id,
            assignment.slot_index,
            class_label,
        )
        return assignment

    def release(self, rack_id: str, slot_index: int) -> None:
        """Mark a slot empty again, e.g. after a failed place.

        Args:
            rack_id: The rack holding the slot.
            slot_index: Index of the slot to free.

        Raises:
            KeyError: If the rack id is unknown.
            IndexError: If the slot index is out of range.
        """
        if rack_id not in self._occupancy:
            raise KeyError(f"unknown rack id '{rack_id}'")
        slots = self._occupancy[rack_id]
        if not 0 <= slot_index < len(slots):
            raise IndexError(f"{rack_id} has no slot {slot_index}; it has {len(slots)}")
        slots[slot_index] = None

    def reset(self) -> None:
        """Empty every slot in every rack."""
        for slots in self._occupancy.values():
            for index in range(len(slots)):
                slots[index] = None

    def summary(self) -> dict[str, str]:
        """A short per-rack occupancy summary, handy for CLI output and logs."""
        return {
            rack.id: f"{self.count_occupied(rack.id)}/{rack.num_slots}"
            for rack in self.workspace.racks
        }
