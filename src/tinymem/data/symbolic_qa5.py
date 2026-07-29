"""Symbolic replay for bAbI task 5 transfer relations."""

from dataclasses import dataclass

from tinymem.data.schema import ReasoningExample
from tinymem.data.symbolic_world import (
    ACQUISITION_SEPARATORS,
    DROP_SEPARATORS,
    MOVEMENT_SEPARATORS,
    OracleResult,
    iter_controlled_facts,
    validate_fact_inputs,
)


TRANSFER_SEPARATORS = (
    " passed the ",
    " gave the ",
    " handed the ",
)


@dataclass(frozen=True)
class Transfer:
    giver: str
    object_name: str
    recipient: str
    fact_id: int


def parse_qa5_transfer(sentence: str, fact_id: int) -> Transfer:
    """Parse one three-argument transfer fact."""
    validate_fact_inputs(sentence, fact_id)
    body = sentence[:-1]
    matches = tuple(separator for separator in TRANSFER_SEPARATORS if separator in body)
    if len(matches) != 1:
        raise ValueError("qa5 transfer must contain one supported transfer verb")
    giver, remainder = body.split(matches[0], maxsplit=1)
    if remainder.count(" to ") != 1:
        raise ValueError("qa5 transfer must identify one recipient")
    object_name, recipient = remainder.split(" to ", maxsplit=1)
    if not giver or not object_name or not recipient:
        raise ValueError("qa5 transfer must contain giver, object, and recipient")
    return Transfer(giver, object_name, recipient, fact_id)


def answer_qa5_question(question: str, transfers: list[Transfer]) -> OracleResult:
    """Answer one official qa5 question from transfer history."""
    if not isinstance(question, str) or not question.endswith("?"):
        raise ValueError("question must be a complete qa5 question")
    body = question[:-1]

    matcher = None
    answer_field = ""
    if body.startswith("What did ") and " give to " in body:
        giver, recipient = body[len("What did ") :].split(" give to ", maxsplit=1)
        matcher = lambda transfer: transfer.giver == giver and transfer.recipient == recipient
        answer_field = "object_name"
    elif body.startswith("Who gave the ") and " to " in body:
        object_name, recipient = body[len("Who gave the ") :].split(" to ", maxsplit=1)
        matcher = lambda transfer: transfer.object_name == object_name and transfer.recipient == recipient
        answer_field = "giver"
    elif body.startswith("Who gave the "):
        object_name = body[len("Who gave the ") :]
        matcher = lambda transfer: transfer.object_name == object_name
        answer_field = "giver"
    elif body.startswith("Who received the "):
        object_name = body[len("Who received the ") :]
        matcher = lambda transfer: transfer.object_name == object_name
        answer_field = "recipient"
    elif body.startswith("Who did ") and " give the " in body and body.endswith(" to"):
        giver, object_name = body[len("Who did ") : -len(" to")].split(
            " give the ",
            maxsplit=1,
        )
        matcher = lambda transfer: transfer.giver == giver and transfer.object_name == object_name
        answer_field = "recipient"
    else:
        raise ValueError("unsupported qa5 question structure")

    matching_transfers = [item for item in transfers if matcher(item)]
    if not matching_transfers:
        raise ValueError(f"no transfer matches question {question!r}")
    answer_values = {getattr(item, answer_field) for item in matching_transfers}
    if len(answer_values) != 1:
        rendered = ", ".join(sorted(answer_values))
        raise ValueError(f"ambiguous qa5 answers for {question!r}: {rendered}")
    transfer = matching_transfers[-1]
    return OracleResult(getattr(transfer, answer_field), (transfer.fact_id,))


def interpret_qa5(example: ReasoningExample) -> OracleResult:
    """Replay a qa5 example and answer one transfer query."""
    if example.task_id != "qa5":
        raise ValueError(f"interpret_qa5 does not support task {example.task_id!r}")
    transfers: list[Transfer] = []
    ignored_separators = MOVEMENT_SEPARATORS + ACQUISITION_SEPARATORS + DROP_SEPARATORS
    for sentence, fact_id in iter_controlled_facts(example):
        if any(separator in sentence for separator in TRANSFER_SEPARATORS):
            transfers.append(parse_qa5_transfer(sentence, fact_id))
        elif not any(separator in sentence for separator in ignored_separators):
            raise ValueError(f"unsupported qa5 fact: {sentence!r}")

    return answer_qa5_question(example.question, transfers)
