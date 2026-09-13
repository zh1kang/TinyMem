"""Small deterministic manifests shared by independent fact tests."""

from tinymem.research.independent_fact_data import ENTITIES, ROOM_PAIRS


_TRAINING_ORDERS = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
_TRAINING_ACTIONS = (('C', 'C', 'C', 'C'), ('R', 'R', 'R', 'R'),
                     ('C', 'R', 'C', 'R'), ('R', 'C', 'R', 'C'))
_OLD_EVENTS = ((0, 'C'), (1, 'C'), (1, 'R'), (2, 'C'),
               (3, 'R'), (0, 'R'), (2, 'R'), (3, 'C'))


def _event(step, before, fact, action):
    old_bit = (before >> fact) & 1
    bit = old_bit if action == 'R' else 1 - old_bit
    after = (before & ~(1 << fact)) | (bit << fact)
    return {
        'step': step, 'before_code': before, 'after_code': after,
        'target_fact': fact, 'new_bit': bit, 'action': action,
        'text': f'{ENTITIES[fact]} moved to the {ROOM_PAIRS[fact][bit]}.',
    }


def _training_streams():
    streams = []
    for code in range(16):
        for order_index, facts in enumerate(_TRAINING_ORDERS):
            for action_index, actions in enumerate(_TRAINING_ACTIONS):
                current = code
                events = []
                for step, (fact, action) in enumerate(zip(facts, actions, strict=True), 1):
                    event = _event(step, current, fact, action)
                    events.append(event)
                    current = event['after_code']
                streams.append({
                    'id': f'train-{code:02d}-{order_index}-{action_index}',
                    'initial_code': code,
                    'events': events,
                })
    return streams


def _evaluation_streams():
    streams = []
    for code in range(16):
        for repeat in range(2):
            current = code
            events = []
            prefix = _OLD_EVENTS if repeat == 0 else tuple(reversed(_OLD_EVENTS))
            for step, (fact, action) in enumerate(prefix, 1):
                event = _event(step, current, fact, action)
                events.append(event)
                current = event['after_code']
            for step, (fact, _) in enumerate(prefix, 9):
                event = _event(step, current, fact, 'R')
                events.append(event)
                current = event['after_code']
            streams.append({
                'id': f'eval-{code:02d}-{repeat}',
                'initial_code': code,
                'events': events,
            })
    return streams


def recurrent_manifest():
    """Return a complete synthetic replacement for the sealed data manifest."""
    return {
        'training_streams': _training_streams(),
        'evaluation_streams': _evaluation_streams(),
    }


def answer_manifest():
    from tinymem.research.independent_fact_answer_protocol import build_answer_manifest

    return build_answer_manifest(recurrent_manifest())
