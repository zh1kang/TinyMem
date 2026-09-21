"""Training-only QA1 preparation with whole-story development splits."""

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import re
from collections.abc import Sequence

from tinymem.data.babi import load_babi_file
from tinymem.data.schema import ReasoningExample

_MOVEMENT = re.compile(r'([A-Z][a-z]+) (?:moved|went|went back|journeyed|travelled) to the ([a-z]+)\.')
_QUESTION = re.compile(r'Where is ([A-Z][a-z]+)\?')


@dataclass(frozen=True)
class Vocabulary:
    entities: tuple[str, ...]
    rooms: tuple[str, ...]

    def __post_init__(self) -> None:
        for values in (self.entities, self.rooms):
            if (not isinstance(values, tuple) or not 1 <= len(values) <= 8
                    or any(not isinstance(value, str) or not value for value in values)
                    or tuple(sorted(set(values))) != values):
                raise ValueError('vocabulary must contain one to eight sorted unique names')


def parse_movement(text: str) -> tuple[str, str]:
    if not isinstance(text, str):
        raise TypeError('movement must be text')
    match = _MOVEMENT.fullmatch(text)
    if match is None:
        raise ValueError(f'unsupported QA1 movement: {text!r}')
    return match[1], match[2]


def query_entity(question: str, vocabulary: Vocabulary) -> int:
    match = _QUESTION.fullmatch(question)
    if match is None or match[1] not in vocabulary.entities:
        raise ValueError('question must name a declared QA1 entity')
    return vocabulary.entities.index(match[1])


def story_id(example: ReasoningExample) -> str:
    story, separator, number = example.source_example_id.rpartition(':question-')
    if not separator or not number.isdecimal() or ':episode-' not in story:
        raise ValueError('expected official numbered-source story identity')
    return story


def fit_vocabulary(examples: Sequence[ReasoningExample]) -> Vocabulary:
    if not examples or any(e.dataset != 'babi' or e.task_id != 'qa1' or e.split != 'train' for e in examples):
        raise ValueError('vocabulary fitting accepts QA1 training examples only')
    events = {parse_movement(line) for e in examples for line in e.context.splitlines()}
    return Vocabulary(tuple(sorted({e for e, _ in events})), tuple(sorted({r for _, r in events})))


def replay_locations(history: Sequence[str], vocabulary: Vocabulary) -> tuple[int, ...]:
    """Replay only fact text; zero is unknown and room codes start at one."""
    if isinstance(history, (str, bytes)):
        raise TypeError('history must be a sequence of fact strings')
    codes = [0] * len(vocabulary.entities)
    for text in history:
        entity, room = parse_movement(text)
        if entity not in vocabulary.entities or room not in vocabulary.rooms:
            raise ValueError('movement contains an undeclared entity or room')
        codes[vocabulary.entities.index(entity)] = vocabulary.rooms.index(room) + 1
    return tuple(codes)


def validate_example(example: ReasoningExample, vocabulary: Vocabulary) -> None:
    if example.dataset != 'babi' or example.task_id != 'qa1':
        raise ValueError('expected official bAbI QA1 example')
    codes = replay_locations(example.context.splitlines(), vocabulary)
    code = codes[query_entity(example.question, vocabulary)]
    if code == 0 or vocabulary.rooms[code - 1] != example.answer:
        raise ValueError('official answer disagrees with independent fact replay')


def split_training(examples: Sequence[ReasoningExample], *, validation_stories: int, seed: int
                   ) -> tuple[tuple[ReasoningExample, ...], tuple[ReasoningExample, ...]]:
    if type(seed) is not int or type(validation_stories) is not int:
        raise TypeError('split seed and story count must be integers')
    if not examples or any(e.dataset != 'babi' or e.task_id != 'qa1' or e.split != 'train' for e in examples):
        raise ValueError('development split accepts official QA1 training data only')
    if len({e.source_example_id for e in examples}) != len(examples):
        raise ValueError('duplicate source question')
    stories = sorted({story_id(e) for e in examples},
                     key=lambda name: (hashlib.sha256(f'{seed}:{name}'.encode()).hexdigest(), name))
    if not 0 < validation_stories < len(stories):
        raise ValueError('development split must retain training and validation stories')
    heldout = set(stories[:validation_stories])
    training = tuple(e for e in examples if story_id(e) not in heldout)
    validation = tuple(replace(e, split='validation') for e in examples if story_id(e) in heldout)
    return training, validation


def load_training(path: Path, *, expected_sha256: str, validation_stories: int = 200,
                  seed: int = 2026091601
                  ) -> tuple[Vocabulary, tuple[ReasoningExample, ...], tuple[ReasoningExample, ...]]:
    if path.name != 'qa1_single-supporting-fact_train.txt':
        raise ValueError('this preparation stage accepts the official QA1 training filename only')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('official QA1 training source checksum differs')
    examples = load_babi_file(path, task_id='qa1', split='train')
    training, validation = split_training(examples, validation_stories=validation_stories, seed=seed)
    vocabulary = fit_vocabulary(training)
    for example in (*training, *validation):
        validate_example(example, vocabulary)
    return vocabulary, training, validation
