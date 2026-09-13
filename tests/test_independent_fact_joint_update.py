"""Direct loss/gradient oracles for updater-only answer and learned-state training."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from test_readout_runner import tiny_reader
from test_independent_fact_answer_training import _setup, _query_ce
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_updates import new_update_writer


def teacher_states(batch):
    writer = new_update_writer(batch.events.shape[-1], 131).requires_grad_(False).eval()
    with torch.no_grad():
        return writer(writer.empty(16), batch.histories, batch.history_valid)


def direct_objective(reader, bridge, writer, batch, worlds, targets, weight, detach=False):
    state = LatentSlotState(targets.values[batch.initial_codes], targets.valid[batch.initial_codes])
    states = []
    losses = []
    for step in range(4):
        state = writer(state, batch.events[:, step], batch.event_valid[:, step])
        states.append(state.values)
        for row in range(batch.events.shape[0]):
            losses.append(_query_ce(reader, bridge, state.values, state.valid, row,
                                    worlds[int(batch.after_codes[row, step])]))
        if detach:
            state = state.detached()
    ce = torch.stack(losses).mean()
    mse = (torch.stack(states, dim=1) - targets.values[batch.after_codes]).square().mean()
    return ce + weight * mse, ce, mse


@pytest.mark.parametrize('weight', [0.0, 1.0])
@pytest.mark.parametrize('batch_size', [1, 2])
def test_joint_matches_direct_objective_and_optimizer(tiny_reader, weight, batch_size):
    from tinymem.research.independent_fact_joint_update import train_joint_update_batch

    batch, worlds, bridge = _setup(tiny_reader, batch_size)
    targets = teacher_states(batch)
    writer = new_update_writer(16, 151); oracle = deepcopy(writer)
    expected, ce, mse = direct_objective(tiny_reader, bridge, oracle, batch, worlds, targets, weight)
    expected.backward(); norm = torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0)
    torch.optim.SGD(oracle.parameters(), lr=.01).step()
    result = train_joint_update_batch(writer, batch, torch.optim.SGD(writer.parameters(), lr=.01),
        reader=tiny_reader, bridge=bridge, worlds=worlds, targets=targets, state_weight=weight)
    assert result['answer_ce'] == pytest.approx(float(ce.detach()), rel=1e-5)
    assert result['learned_state_mse'] == pytest.approx(float(mse.detach()), rel=1e-5)
    assert result['objective'] == pytest.approx(float(expected.detach()), rel=1e-5)
    assert result['gradient_norm'] == pytest.approx(float(norm), rel=1e-5)
    assert result['initial_encoder_examples'] == 0
    assert result['answer_sequences'] == batch_size * 4 * 6
    for actual, reference in zip(writer.parameters(), oracle.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, reference.grad, rtol=2e-4, atol=2e-6)
        torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-6)


def test_measurement_keeps_weights_optimizer_reader_and_teacher_fixed(tiny_reader):
    from tinymem.research.independent_fact_joint_update import train_joint_update_batch

    batch, worlds, bridge = _setup(tiny_reader)
    targets = teacher_states(batch)
    writer = new_update_writer(16, 171)
    before = deepcopy(writer.state_dict()); teacher_before = targets.values.clone()
    reader_before = deepcopy(tiny_reader.model.state_dict()); bridge_before = deepcopy(bridge.state_dict())
    optimizer = torch.optim.AdamW(writer.parameters(), lr=.01)
    result = train_joint_update_batch(writer, batch, optimizer, reader=tiny_reader, bridge=bridge,
        worlds=worlds, targets=targets, state_weight=.5, measure_only=True)
    assert result['optimizer_steps'] == 0 and not optimizer.state
    assert all(torch.equal(v,before[k]) for k,v in writer.state_dict().items())
    assert all(torch.equal(v,reader_before[k]) for k,v in tiny_reader.model.state_dict().items())
    assert all(torch.equal(v,bridge_before[k]) for k,v in bridge.state_dict().items())
    assert torch.equal(targets.values,teacher_before) and targets.values.grad is None
    assert all(p.grad is None for p in (*tiny_reader.model.parameters(),*bridge.parameters()))
    assert result['answer_parameter_gradient_norm'] > 0
    assert result['state_parameter_gradient_norm'] > 0
    oracle=deepcopy(writer)
    _,ce,mse=direct_objective(tiny_reader,bridge,oracle,batch,worlds,targets,.5)
    ce_grad=torch.autograd.grad(ce,tuple(oracle.parameters()),retain_graph=True)
    mse_grad=torch.autograd.grad(mse,tuple(oracle.parameters()))
    ce_norm=torch.cat([g.flatten() for g in ce_grad]).norm()
    mse_norm=torch.cat([g.flatten() for g in mse_grad]).norm()
    dot=sum((a*b).sum() for a,b in zip(ce_grad,mse_grad,strict=True))
    assert result['answer_parameter_gradient_norm']==pytest.approx(float(ce_norm),rel=1e-5)
    assert result['state_parameter_gradient_norm']==pytest.approx(float(mse_norm),rel=1e-5)
    assert result['parameter_gradient_dot']==pytest.approx(float(dot),rel=1e-4,abs=1e-7)


def test_uses_raw_recurrence_not_detached_or_teacher_reset(tiny_reader):
    from tinymem.research.independent_fact_joint_update import train_joint_update_batch

    batch, worlds, bridge = _setup(tiny_reader)
    targets = teacher_states(batch)
    writer = new_update_writer(16, 181); detached = deepcopy(writer)
    train_joint_update_batch(writer, batch, torch.optim.SGD(writer.parameters(), lr=0),
        reader=tiny_reader, bridge=bridge, worlds=worlds, targets=targets, state_weight=.5)
    loss,_,_ = direct_objective(tiny_reader,bridge,detached,batch,worlds,targets,.5,detach=True)
    loss.backward();torch.nn.utils.clip_grad_norm_(detached.parameters(),1.0)
    assert any(not torch.allclose(a.grad,b.grad,atol=1e-6) for a,b in zip(writer.parameters(),detached.parameters(),strict=True))


def test_histories_assigned_coordinates_and_teacher_labels_do_not_enter_update_forward(tiny_reader):
    from tinymem.research.independent_fact_joint_update import train_joint_update_batch

    batch, worlds, bridge = _setup(tiny_reader)
    targets = teacher_states(batch)
    changed = replace(batch,histories=torch.randn_like(batch.histories),
        initial_targets=torch.randn_like(batch.initial_targets),before_targets=torch.randn_like(batch.before_targets),
        after_targets=torch.randn_like(batch.after_targets))
    writer = new_update_writer(16,191);other=deepcopy(writer)
    calls=[]
    def observe(module,args,output):
        state,hidden,valid=args
        calls.append((state.values.detach().clone(),output.values.detach().clone()))
    handle=writer.register_forward_hook(observe)
    first=train_joint_update_batch(writer,batch,torch.optim.SGD(writer.parameters(),lr=.01),
        reader=tiny_reader,bridge=bridge,worlds=worlds,targets=targets,state_weight=.5)
    handle.remove()
    second=train_joint_update_batch(other,changed,torch.optim.SGD(other.parameters(),lr=.01),
        reader=tiny_reader,bridge=bridge,worlds=worlds,targets=targets,state_weight=.5)
    assert first==second and len(calls)==4
    assert torch.equal(calls[0][0],targets.values[batch.initial_codes])
    assert all(torch.equal(calls[i][0],calls[i-1][1]) for i in range(1,4))
    assert all(torch.equal(a,b) for a,b in zip(writer.parameters(),other.parameters(),strict=True))


def test_zero_weight_ignores_after_teacher_targets_but_joint_loss_uses_them(tiny_reader):
    from tinymem.research.independent_fact_joint_update import train_joint_update_batch

    batch,worlds,bridge=_setup(tiny_reader)
    targets=teacher_states(batch)
    altered=targets.values.clone();altered[1:]=-altered[1:]
    changed=LatentSlotState(altered,targets.valid.clone())
    for weight in (0.,.5):
        writer=new_update_writer(16,201);other=deepcopy(writer)
        a=train_joint_update_batch(writer,batch,torch.optim.SGD(writer.parameters(),lr=.01),
            reader=tiny_reader,bridge=bridge,worlds=worlds,targets=targets,state_weight=weight)
        b=train_joint_update_batch(other,batch,torch.optim.SGD(other.parameters(),lr=.01),
            reader=tiny_reader,bridge=bridge,worlds=worlds,targets=changed,state_weight=weight)
        assert a['answer_ce']==b['answer_ce']
        same=all(torch.equal(x,y) for x,y in zip(writer.parameters(),other.parameters(),strict=True))
        assert same==(weight==0)


def test_rejects_bad_weight_trainable_teacher_and_wrong_optimizer(tiny_reader):
    from tinymem.research.independent_fact_joint_update import train_joint_update_batch

    batch,worlds,bridge=_setup(tiny_reader);targets=teacher_states(batch);writer=new_update_writer(16,211)
    for weight in (-1,float('nan'),float('inf'),True):
        with pytest.raises(ValueError,match='state_weight'):
            train_joint_update_batch(writer,batch,torch.optim.SGD(writer.parameters(),lr=.01),
                reader=tiny_reader,bridge=bridge,worlds=worlds,targets=targets,state_weight=weight)
    with pytest.raises(ValueError,match='detached'):
        train_joint_update_batch(writer,batch,torch.optim.SGD(writer.parameters(),lr=.01),
            reader=tiny_reader,bridge=bridge,worlds=worlds,
            targets=LatentSlotState(targets.values.clone().requires_grad_(),targets.valid),state_weight=1.)
    with pytest.raises(ValueError,match='optimizer'):
        train_joint_update_batch(writer,batch,torch.optim.SGD([writer.queries],lr=.01),
            reader=tiny_reader,bridge=bridge,worlds=worlds,targets=targets,state_weight=0.)


def test_calibration_has_declared_rms_scale_and_rejects_bad_measurements():
    from tinymem.research.independent_fact_joint_update import calibrate_state_weight

    metrics=[{'optimizer_steps':0,'answer_parameter_gradient_norm':float(i),'state_parameter_gradient_norm':2.0*float(i)} for i in range(1,17)]
    assert calibrate_state_weight(metrics)==pytest.approx(.125)
    unequal=[dict(m,state_parameter_gradient_norm=2.) for m in metrics]
    assert calibrate_state_weight(unequal)==pytest.approx(.25*(sum(i*i for i in range(1,17))/16)**.5/2.)
    for changed in (metrics[:1],[dict(m,optimizer_steps=1) for m in metrics],
                    [dict(m,state_parameter_gradient_norm=0) for m in metrics],
                    [dict(m,answer_parameter_gradient_norm=float('nan')) for m in metrics]):
        with pytest.raises(ValueError):calibrate_state_weight(changed)
