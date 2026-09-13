"""Fresh-lineage ownership and stage transitions through a real tiny reader."""
import json

import torch
from safetensors.torch import load_file

from test_independent_fact_answer_training import _features, _setup
from test_readout_runner import tiny_reader
from tinymem.research.independent_fact_updates import new_update_writer


def test_fresh_training_lineage_has_separate_initializer_and_matched_branches(tiny_reader, tmp_path, monkeypatch):
    from scripts import fit_independent_fact_lineage as runner
    train_lineage = runner.train_lineage
    batch, worlds, bridge = _setup(tiny_reader)
    histories, _ = _features(16)
    bridge_before = {k: v.clone() for k,v in bridge.state_dict().items()}
    observed = []
    def observe(kernel, stage):
        def call(writer, batch, optimizer, **kwargs):
            if not kwargs.get('measure_only', False):
                name = stage if stage != 'continuation' else ('uniform' if kwargs['correction_multiplier'] == 1 else 'correction_weighted')
                parent = {'initial': 'fresh_initial', 'joint': 'initial_final',
                          'uniform': 'joint_final', 'correction_weighted': 'joint_final'}[name]
                expected = load_file(str(tmp_path/'training'/f'{parent}.safetensors'))
                assert all(torch.equal(v, expected[k]) for k,v in writer.state_dict().items())
                assert not optimizer.state
                assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in writer.parameters()}
                observed.append(name)
            return kernel(writer, batch, optimizer, **kwargs)
        return call
    for symbol,stage in [('train_answer_recurrent_batch','initial'),('train_joint_update_batch','joint'),('train_correction_weight_batch','continuation')]:
        monkeypatch.setattr(runner, symbol, observe(getattr(runner,symbol),stage))
    reader_before = {k: v.clone() for k, v in tiny_reader.model.state_dict().items()}
    initial, arms, record = train_lineage(
        output=tmp_path/'training', seed=2027, batches=[batch]*16, histories=histories,
        reader=tiny_reader, bridge=bridge, worlds=worlds, steps_per_stage=1)
    assert record['optimizer_steps'] == 4
    fresh = load_file(str(tmp_path/'training/fresh_initial.safetensors'))
    expected = new_update_writer(16, 2027).state_dict()
    assert all(torch.equal(fresh[k], v) for k, v in expected.items())
    other = new_update_writer(16, 2028).state_dict()
    assert any(not torch.equal(fresh[k], v) for k, v in other.items())
    teacher = load_file(str(tmp_path/'training/teacher_states.safetensors'))
    for code in range(16):
        hidden = histories[code].unsqueeze(0)
        with torch.no_grad(): state = initial(initial.empty(1), hidden, torch.ones(hidden.shape[:2], dtype=torch.bool))
        assert torch.equal(state.values, teacher['values'][code:code+1])
    stages = record['stages']
    assert stages['uniform']['start_sha256'] == stages['correction_weighted']['start_sha256'] == stages['joint']['final_sha256']
    assert stages['joint']['start_sha256'] == stages['initial']['final_sha256']
    assert stages['uniform']['state_weight'] == stages['correction_weighted']['state_weight'] == record['state_weight'] > 0
    metrics = [json.loads(line) for line in (tmp_path/'training/metrics.jsonl').read_text().splitlines()]
    assert [m['stage'] for m in metrics] == ['initial', 'joint', 'uniform', 'correction_weighted']
    assert all(not p.requires_grad and p.grad is None for m in [initial,*arms.values()] for p in m.parameters())
    assert all(torch.equal(reader_before[k], v) for k,v in tiny_reader.model.state_dict().items())
    assert observed == ['initial', 'joint', 'uniform', 'correction_weighted']
    assert all(torch.equal(bridge_before[k], v) for k,v in bridge.state_dict().items())
