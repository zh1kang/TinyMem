"""Fixed eight-event recurrence streams over the four independent facts."""

from dataclasses import dataclass

from tinymem.research.independent_fact_data import ENTITIES, ROOM_PAIRS


@dataclass(frozen=True)
class RecurrenceEvent:
    step: int
    before_code: int
    after_code: int
    target_fact: int
    new_bit: int
    event_text: str
    prior_split: str


@dataclass(frozen=True)
class Stream:
    stream_id: str
    initial_code: int
    family: str
    order: tuple[int, ...]
    events: tuple[RecurrenceEvent, ...]


def _event_split(code: int, fact: int) -> str:
    untouched = code & ~(1 << fact)
    return "train" if untouched.bit_count() % 2 == 0 else "heldout"


def build_recurrence_streams() -> tuple[Stream, ...]:
    """Return all fixed repeat and toggle streams in code/family/order order."""
    streams = []
    for initial_code in range(16):
        for family in ("repeat", "toggle"):
            for order_name, order in (("forward", (0, 1, 2, 3)), ("reverse", (3, 2, 1, 0))):
                current = initial_code
                events = []
                for step, fact in enumerate(order * 2, start=1):
                    current_bit = (current >> fact) & 1
                    new_bit = current_bit if family == "repeat" else 1 - current_bit
                    after_code = (current & ~(1 << fact)) | (new_bit << fact)
                    events.append(RecurrenceEvent(
                        step=step,
                        before_code=current,
                        after_code=after_code,
                        target_fact=fact,
                        new_bit=new_bit,
                        event_text=f"{ENTITIES[fact]} moved to the {ROOM_PAIRS[fact][new_bit]}.",
                        prior_split=_event_split(current, fact),
                    ))
                    current = after_code
                streams.append(Stream(
                    stream_id=f"code-{initial_code:02d}:{family}:{order_name}",
                    initial_code=initial_code,
                    family=family,
                    order=order,
                    events=tuple(events),
                ))
    return tuple(streams)
