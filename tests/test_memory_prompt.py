from dataclasses import replace
from types import SimpleNamespace

import pytest

from tinymem.data.reader_gate import ReaderCase
from tinymem.evaluation.reader_gate import reader_messages
from tinymem.research.memory_prompt import encode_memory_example
from tinymem.research.pretrained import PretrainedReader


class BoundaryTokenizer:
    """Small greedy tokenizer that reproduces punctuation/newline merges."""

    eos_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return f"<system>{messages[0]['content']}<user>{messages[1]['content']}<assistant>"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        ids = []
        while text:
            match = next((piece for piece in (".\n\n\n", ".\n\n", "\n\n\n", "\n\n") if text.startswith(piece)), None)
            if match is None:
                ids.append(ord(text[0]) + 1)
                text = text[1:]
            else:
                ids.append({".\n\n\n": 1000, ".\n\n": 1001, "\n\n\n": 1002, "\n\n": 1003}[match])
                text = text[len(match):]
        return ids


@pytest.fixture
def reader():
    return PretrainedReader(SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096)), BoundaryTokenizer())


@pytest.mark.parametrize("ending", [".", ".\n", "code-0123"])
def test_history_owns_boundary_tokens_and_matches_native_prompt(reader, ending):
    case = ReaderCase("id", "babi_qa1", "history", "Mary moved to the kitchen" + ending, "Where is Mary?", "kitchen")
    encoded = encode_memory_example(reader, case)
    full = reader.tokenizer.apply_chat_template(reader_messages(case, condition="full_context"), tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
    assert (*encoded.before_ids, *encoded.history_ids, *encoded.after_ids) == tuple(reader.tokenizer.encode(full, add_special_tokens=False))
    assert encoded.history_ids == tuple(reader.tokenizer.encode(case.context + "\n\n", add_special_tokens=False))
    assert encoded.after_ids[:len("Question:")] == tuple(ord(char) + 1 for char in "Question:")
    assert encoded.answer_ids == (*reader.tokenizer.encode("kitchen", add_special_tokens=False), 0)


def test_query_fragments_do_not_depend_on_history_or_gold(reader):
    case = ReaderCase("id", "babi_qa1", "history", "Mary moved to the kitchen.", "Where is Mary?", "kitchen")
    encoded = encode_memory_example(reader, case)
    changed = encode_memory_example(reader, replace(case, context="Mary moved to the hallway.\n", answer="secret label"))
    assert changed.before_ids == encoded.before_ids
    assert changed.after_ids == encoded.after_ids
    assert changed.history_ids != encoded.history_ids
    assert changed.answer_ids != encoded.answer_ids
    different_question = encode_memory_example(reader, replace(case, question="Where is Sandra?"))
    assert different_question.history_ids == encoded.history_ids
    assert different_question.before_ids == encoded.before_ids
    assert different_question.after_ids != encoded.after_ids


def test_cross_boundary_merges_fail_instead_of_changing_the_prompt(reader):
    case = ReaderCase("id", "babi_qa1", "history", "\nMary moved to the kitchen.", "Where is Mary?", "kitchen")
    with pytest.raises(ValueError, match="merges across"):
        encode_memory_example(reader, case)


def test_memory_prompt_rejects_empty_reserved_or_overlength_envelopes(reader):
    case = ReaderCase("id", "babi_qa1", "history", "Mary moved to the kitchen.", "Where is Mary?", "kitchen")
    with pytest.raises(ValueError, match="nonempty"):
        encode_memory_example(reader, replace(case, context=""))
    with pytest.raises(ValueError, match="reserved"):
        encode_memory_example(reader, replace(case, question="__TINYMEM_NATIVE_HISTORY_BOUNDARY__"))
    reader.model.config.max_position_embeddings = 10
    with pytest.raises(ValueError, match="before memory"):
        encode_memory_example(reader, case)


def test_history_can_exceed_reader_context_but_query_cannot(reader):
    case = ReaderCase("id", "babi_qa1", "history", "Mary moved to the kitchen.\n" * 200,
                      "Where is Mary?", "kitchen")
    encoded = encode_memory_example(reader, case)
    assert len(encoded.history_ids) > reader.model.config.max_position_embeddings
