import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from tinymem.memory.quantized_slots import QuantizedSlotMemory
from tinymem.reader.adapter import configure_read_adapter
from tinymem.reader.pretrained import PretrainedReader
from tinymem.reader.lora import attach_reader_lora
from tinymem.studies.frontier.training import (
    AnswerTokens,
    answer_loss,
    optimizer_step,
    rollout,
)


def test_joint_native_answer_training_and_serialized_read():
    torch.manual_seed(11)
    torch.set_num_threads(1)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
    reader = PretrainedReader(Qwen3ForCausalLM(config), None)
    attach_reader_lora(reader, rank=2, checkpointing=False)
    adapters = configure_read_adapter(reader, trainable=True)
    memory = QuantizedSlotMemory(32, slots=2, memory_width=4, hidden_width=16, heads=2)
    features = {'a': torch.randn(3, 32), 'b': torch.randn(4, 32)}
    histories = (('a', 'b', 'a'), ('b',))
    tokens = (AnswerTokens((1, 2), (3, 4), (5, 6)),) * 2
    before = {n: p.detach().clone() for n, p in reader.model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([*memory.parameters(), *adapters], lr=.001)
    state = rollout(memory, histories, features)
    loss = answer_loss(reader, tokens, list(memory.memory_vectors(state)))
    result = optimizer_step(reader, memory, adapters, optimizer, loss)
    assert all(result[key] > 0 for key in ('writer_gradient_norm', 'bridge_gradient_norm', 'adapter_gradient_norm'))
    for name, parameter in reader.model.named_parameters():
        if name in before:
            assert torch.equal(parameter, before[name])
    memory.eval()
    with torch.no_grad():
        state = rollout(memory, histories, features)
        restored = torch.cat([memory.unpack(memory.pack(row[None])) for row in state])
        assert torch.equal(state, restored)
        assert torch.equal(answer_loss(reader, tokens, list(memory.memory_vectors(state))),
                           answer_loss(reader, tokens, list(memory.memory_vectors(restored))))
