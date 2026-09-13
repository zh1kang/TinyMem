"""Fresh update programs and isolated truthful-refresh branches for frozen writers."""
from itertools import permutations
import random

import torch

from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_data import ENTITIES, ROOM_PAIRS
from tinymem.research.independent_fact_recurrent_training import _check_state


def _events(initial_code, program):
    current = initial_code
    result = []
    for step, (fact, action) in enumerate(program, 1):
        before = current
        if action == 'C':
            current ^= 1 << fact
        bit = (current >> fact) & 1
        result.append({'step': step, 'before_code': before, 'after_code': current,
            'target_fact': fact, 'new_bit': bit, 'action': action,
            'text': f'{ENTITIES[fact]} moved to the {ROOM_PAIRS[fact][bit]}.'})
    return result


def build_manifest(previous_manifest):
    """Select 64 unique order programs without consulting checkpoints or readouts."""
    excluded = set()
    for group in ('evaluation_streams', 'old_continuity_streams'):
        for stream in previous_manifest[group]:
            if len(stream['events']) >= 8:
                excluded.add(tuple((e['target_fact'], e['action']) for e in stream['events'][:8]))
    candidates = list(permutations([(fact, action) for fact in range(4) for action in 'CR']))
    random.Random(20260912).shuffle(candidates)
    selected = [p for p in candidates if p not in excluded][:64]
    if len(selected) != 64:
        raise ValueError('insufficient fresh update programs')
    streams = []
    for index, program in enumerate(selected):
        code, replicate = divmod(index, 4)
        prefix = _events(code, program)
        final = prefix[-1]['after_code']
        balanced = [((replicate + step) % 4, 'R') for step in range(8)]
        tails = {'balanced': _events(final, balanced)}
        tails.update({f'single{fact}': _events(final, [(fact, 'R')] * 8) for fact in range(4)})
        streams.append({'id': f'repeat-{code:02d}-{replicate}', 'initial_code': code,
                        'prefix': prefix, 'tails': tails})
    return {'kind': 'independent_fact_repeat_confirmation_v1', 'seed': 20260912,
            'prefix_horizon': 8, 'tail_horizon': 8, 'streams': streams}


def state_catalog(manifest):
    catalog = {f'initial:{code:02d}': {'kind': 'initial', 'code': code} for code in range(16)}
    for stream in manifest['streams']:
        for condition, events in [('prefix', stream['prefix']), *stream['tails'].items()]:
            for event in events:
                name = f"{stream['id']}/{condition}/{event['step']:02d}"
                if name in catalog:
                    raise ValueError('duplicate state name')
                catalog[name] = {'kind': 'event', 'code': event['after_code'],
                    'condition': condition, 'stream_id': stream['id'], 'step': event['step'],
                    'before_code': event['before_code'], 'fact': event['target_fact']}
    return catalog


def _copy(state):
    return LatentSlotState(state.values.detach().clone().contiguous(),
                           state.valid.detach().clone().contiguous())


def collect_states(initializer, updater, histories, features, manifest):
    """Copy the same own-prefix state into every branch; never reset from truth labels."""
    for module in (initializer, updater):
        if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise ValueError('checkpoint must be frozen and in evaluation mode')
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in module.parameters()):
            raise ValueError('checkpoint must use CPU FP32')
    snapshots = [{k: v.clone() for k, v in m.state_dict().items()} for m in (initializer, updater)]
    states = {}

    def write(module, state, feature):
        before = _copy(state)
        hidden = feature.unsqueeze(0)
        hidden_before = hidden.clone()
        with torch.inference_mode():
            result = module(state, hidden, torch.ones(hidden.shape[:2], dtype=torch.bool))
        _check_state(result, 1, module)
        if (not torch.equal(state.values, before.values) or not torch.equal(state.valid, before.valid)
                or not torch.equal(hidden, hidden_before)):
            raise ValueError('write mutated a source state or event feature')
        return _copy(result)

    for code in range(16):
        states[f'initial:{code:02d}'] = write(initializer, initializer.empty(1), histories[code])
    for stream in manifest['streams']:
        state = _copy(states[f"initial:{stream['initial_code']:02d}"])
        for event in stream['prefix']:
            state = write(updater, state, features[event['target_fact'], event['new_bit']])
            states[f"{stream['id']}/prefix/{event['step']:02d}"] = _copy(state)
        root = _copy(state)
        for condition, events in stream['tails'].items():
            state = _copy(root)
            for event in events:
                state = write(updater, state, features[event['target_fact'], event['new_bit']])
                states[f"{stream['id']}/{condition}/{event['step']:02d}"] = _copy(state)
    for module, snapshot in zip((initializer, updater), snapshots, strict=True):
        if any(not torch.equal(v, snapshot[k]) for k, v in module.state_dict().items()):
            raise ValueError('frozen checkpoint changed')
    if set(states) != set(state_catalog(manifest)):
        raise ValueError('state coverage differs')
    return states
