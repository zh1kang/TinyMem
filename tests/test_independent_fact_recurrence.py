import inspect
import re
from collections import Counter

import pytest

from tinymem.research.independent_fact_data import ENTITIES, ROOM_PAIRS
from tinymem.research.independent_fact_updates import build_update_cases


EVENT_TEXT = re.compile(r"^(.+) moved to the (.+)\.$")


def _code_from_rooms(rooms):
    code = 0
    for fact, pair in enumerate(ROOM_PAIRS):
        code |= pair.index(rooms[fact]) << fact
    return code


def test_streams_have_fixed_coverage_and_ordered_ids():
    from tinymem.research.independent_fact_recurrence import build_recurrence_streams

    streams = build_recurrence_streams()
    assert len(streams) == 64
    assert [stream.stream_id for stream in streams[:4]] == [
        "code-00:repeat:forward", "code-00:repeat:reverse",
        "code-00:toggle:forward", "code-00:toggle:reverse",
    ]
    assert {stream.initial_code for stream in streams} == set(range(16))
    assert {stream.family for stream in streams} == {"repeat", "toggle"}
    assert {stream.order for stream in streams} == {
        (0, 1, 2, 3), (3, 2, 1, 0),
    }
    for stream in streams:
        assert stream.events and len(stream.events) == 8
        assert tuple(event.target_fact for event in stream.events) == stream.order * 2
        order_name = "forward" if stream.order == (0, 1, 2, 3) else "reverse"
        assert stream.stream_id == f"code-{stream.initial_code:02d}:{stream.family}:{order_name}"


def test_independent_text_replay_validates_all_transitions_and_endpoints():
    from tinymem.research.independent_fact_recurrence import build_recurrence_streams

    for stream in build_recurrence_streams():
        rooms = [ROOM_PAIRS[fact][(stream.initial_code >> fact) & 1] for fact in range(4)]
        initial = _code_from_rooms(rooms)
        assert initial == stream.initial_code
        for step, event in enumerate(stream.events, start=1):
            assert event.step == step
            assert event.before_code == _code_from_rooms(rooms)
            match = EVENT_TEXT.fullmatch(event.event_text)
            assert match is not None
            fact = ENTITIES.index(match.group(1))
            bit = ROOM_PAIRS[fact].index(match.group(2))
            assert (event.target_fact, event.new_bit) == (fact, bit)
            untouched = event.before_code & ~(1 << fact)
            assert event.prior_split == ("train" if untouched.bit_count() % 2 == 0 else "heldout")
            rooms[fact] = ROOM_PAIRS[fact][bit]
            assert event.after_code == _code_from_rooms(rooms)
        assert _code_from_rooms(rooms) == stream.initial_code
        if stream.family == "toggle":
            halfway = stream.events[3].after_code
            assert halfway == (stream.initial_code ^ 15)
        else:
            assert all(event.before_code == event.after_code for event in stream.events)


def test_event_fact_and_bit_coverage_is_balanced():
    from tinymem.research.independent_fact_recurrence import build_recurrence_streams

    events = [event for stream in build_recurrence_streams() for event in stream.events]
    assert Counter(event.target_fact for event in events) == {fact: 128 for fact in range(4)}
    assert Counter(event.new_bit for event in events) == {0: 256, 1: 256}
    assert {fact: Counter(event.new_bit for event in events if event.target_fact == fact) for fact in range(4)} == {
        fact: {0: 64, 1: 64} for fact in range(4)
    }


def test_every_recurrence_event_is_one_of_the_frozen_128_update_cases():
    from tinymem.research.independent_fact_recurrence import build_recurrence_streams

    cases = {
        (case.before_code, case.target_fact, case.new_bit): case
        for case in build_update_cases()
    }
    first_steps = []
    for stream in build_recurrence_streams():
        for event in stream.events:
            case = cases[(event.before_code, event.target_fact, event.new_bit)]
            assert event.after_code == case.after_code
            assert event.event_text == case.event_text
            assert event.prior_split == case.split
        first_steps.append((stream.initial_code, stream.events[0].target_fact, stream.events[0].new_bit))
    assert len(first_steps) == 64
    assert all(key in cases for key in first_steps)


def test_recurrence_stream_builder_takes_no_arguments():
    from tinymem.research.independent_fact_recurrence import build_recurrence_streams

    assert inspect.signature(build_recurrence_streams).parameters == {}
    with pytest.raises(TypeError):
        build_recurrence_streams(0)
