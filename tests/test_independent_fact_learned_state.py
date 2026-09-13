"""Independent full-gradient and supervision-boundary tests for learned-state updates."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from independent_fact_fixtures import answer_manifest

from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_learned_state import train_learned_state_batch
from tinymem.research.independent_fact_recurrent_training import pack_recurrent_batch
from tinymem.research.independent_fact_updates import new_update_writer


def setup():
    manifest = answer_manifest()
    generator = torch.Generator().manual_seed(912)
    histories = {code: torch.randn(2+code%3, 4, generator=generator) for code in range(16)}
    features = {(fact, bit): torch.randn(1+fact+bit,4,generator=generator) for fact in range(4) for bit in range(2)}
    batch = pack_recurrent_batch(manifest['training_streams'][:2],histories,features)
    targets = LatentSlotState(torch.rand(16,2,8,generator=generator)*1.6-.8,torch.ones(16,2,dtype=torch.bool))
    return batch, targets


@pytest.mark.parametrize('arm', ['reset','recurrent'])
def test_matches_direct_gradient_and_optimizer_with_fixed_teacher(arm):
    batch, targets = setup(); writer = new_update_writer(4,31); oracle = deepcopy(writer)
    original = targets.values.clone()
    state = LatentSlotState(targets.values[batch.initial_codes],targets.valid[batch.initial_codes])
    terms = []
    for step in range(4):
        if arm=='reset': state=LatentSlotState(targets.values[batch.before_codes[:,step]],targets.valid[batch.before_codes[:,step]])
        state = oracle(state,batch.events[:,step],batch.event_valid[:,step])
        terms.append((state.values-targets.values[batch.after_codes[:,step]]).square().mean())
    expected = torch.stack(terms).mean(); expected.backward()
    torch.nn.utils.clip_grad_norm_(oracle.parameters(),1.0)
    torch.optim.SGD(oracle.parameters(),lr=.03).step()
    result = train_learned_state_batch(writer,batch,torch.optim.SGD(writer.parameters(),lr=.03),targets=targets,arm=arm)
    assert result['learned_state_mse']==pytest.approx(float(expected.detach()),abs=1e-7)
    for actual, reference in zip(writer.parameters(),oracle.parameters(),strict=True):
        torch.testing.assert_close(actual,reference,atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(actual.grad,reference.grad,atol=1e-7,rtol=1e-5)
    assert torch.equal(targets.values,original) and targets.values.grad is None


def test_assigned_coordinates_and_histories_do_not_supervise_the_updater():
    batch, targets = setup(); left=new_update_writer(4,73); right=deepcopy(left)
    changed=replace(batch, histories=torch.randn_like(batch.histories),
                    initial_targets=torch.randn_like(batch.initial_targets),
                    before_targets=torch.randn_like(batch.before_targets),after_targets=torch.randn_like(batch.after_targets))
    one=train_learned_state_batch(left,batch,torch.optim.SGD(left.parameters(),lr=.01),targets=targets,arm='recurrent')
    two=train_learned_state_batch(right,changed,torch.optim.SGD(right.parameters(),lr=.01),targets=targets,arm='recurrent')
    assert one==two
    assert all(torch.equal(a,b) for a,b in zip(left.parameters(),right.parameters(),strict=True))


def test_recurrence_gradient_is_not_detached():
    batch, targets=setup(); writer=new_update_writer(4,91); detached=deepcopy(writer)
    train_learned_state_batch(writer,batch,torch.optim.SGD(writer.parameters(),lr=0),targets=targets,arm='recurrent')
    state=LatentSlotState(targets.values[batch.initial_codes],targets.valid[batch.initial_codes]); terms=[]
    for step in range(4):
        state=detached(state,batch.events[:,step],batch.event_valid[:,step])
        terms.append((state.values-targets.values[batch.after_codes[:,step]]).square().mean())
        state=LatentSlotState(state.values.detach(),state.valid)
    torch.stack(terms).mean().backward();torch.nn.utils.clip_grad_norm_(detached.parameters(),1.0)
    assert any(not torch.allclose(a.grad,b.grad,atol=1e-6) for a,b in zip(writer.parameters(),detached.parameters(),strict=True))


def test_rejects_trainable_targets_or_wrong_optimizer():
    batch, targets=setup(); writer=new_update_writer(4,91)
    with pytest.raises(ValueError,match='detached'):
        train_learned_state_batch(writer,batch,torch.optim.SGD(writer.parameters(),lr=.01),
                                  targets=LatentSlotState(targets.values.requires_grad_(),targets.valid),arm='recurrent')
    targets=LatentSlotState(targets.values.detach(),targets.valid)
    with pytest.raises(ValueError,match='optimizer'):
        train_learned_state_batch(writer,batch,torch.optim.SGD([writer.queries],lr=.01),targets=targets,arm='recurrent')


def test_split_writer_keeps_initialization_fixed_and_routes_real_updates():
    from scripts.fit_independent_fact_learned_state import SplitWriter
    initializer=new_update_writer(4,11).requires_grad_(False).eval()
    updater=new_update_writer(4,13).requires_grad_(False).eval()
    split=SplitWriter(initializer,updater)
    hidden=torch.arange(12,dtype=torch.float32).reshape(1,3,4);valid=torch.ones(1,3,dtype=torch.bool)
    with torch.inference_mode():
        initial=split(split.empty(1),hidden,valid)
        assert torch.equal(initial.values,initializer(initializer.empty(1),hidden,valid).values)
        actual=split(initial,hidden,valid)
        assert torch.equal(actual.values,updater(initial,hidden,valid).values)
        assert not torch.equal(actual.values,initializer(initial,hidden,valid).values)
        with pytest.raises(ValueError,match='all-empty or all-valid'):
            split(LatentSlotState(initial.values,torch.tensor([[True,False]])),hidden,valid)
