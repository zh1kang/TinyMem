"""Post-fit identity checks and integrity gates independent of accuracy targets."""
from copy import deepcopy
import json

import pytest
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from test_readout_runner import tiny_reader
from scripts.fit_independent_facts import require_reference, verify_evaluation_identity, verify_reloaded_bridge
from scripts.profile_adapted_readout import frozen_hash
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.research.adapted_readout import configure_read_adapter
from tinymem.research.independent_fact_data import oracle_state
from tinymem.research.reader_adaptation import attach_reader_lora
from tinymem.research.readout_experiment import _reader_hash
from tinymem.research.update_protocol import file_sha256


def test_saved_adapter_reload_and_frozen_evaluation_use_stable_reader_identity(tiny_reader,tmp_path):
    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    adapters=configure_read_adapter(tiny_reader,trainable=True)
    base=frozen_hash(tiny_reader)
    # A real adapter update, then the runner's save, clear, reload and freeze path.
    with torch.no_grad():
        for parameter in adapters:
            parameter.add_(.01)
    assert frozen_hash(tiny_reader)==base
    trained=_reader_hash(tiny_reader)
    tiny_reader.model.save_pretrained(tmp_path/'adapter',save_embedding_layers=False)
    with torch.no_grad():
        for parameter in adapters:
            parameter.zero_()
    set_peft_model_state_dict(tiny_reader.model,load_file(str(tmp_path/'adapter/adapter_model.safetensors')),adapter_name='default')
    configure_read_adapter(tiny_reader,trainable=False)
    states={c:oracle_state(c,torch.device('cpu')) for c in range(16)}
    expected={f'{c}.{part}':getattr(state,part).clone() for c,state in states.items() for part in ('values','valid')}
    assert frozen_hash(tiny_reader)!=base  # Trainability changes this helper's membership.
    verify_evaluation_identity(tiny_reader,trained,expected,states)
    with torch.no_grad():
        next(tiny_reader.model.parameters()).add_(.1)
    with pytest.raises(ValueError,match='reader changed'):
        verify_evaluation_identity(tiny_reader,trained,expected,states)


def test_bridge_reload_detects_dormant_coordinate_corruption():
    original=ReadoutBridge(16,'affine')
    restored=deepcopy(original)
    verify_reloaded_bridge(original,restored)
    with torch.no_grad():
        restored.input_projection.weight[:,4:].add_(1)
    for code in range(16):
        state=oracle_state(code,torch.device('cpu'))
        assert torch.equal(original(state),restored(state))
    with pytest.raises(ValueError,match='tensor reload'):
        verify_reloaded_bridge(original,restored)


def test_reference_below_target_requires_interpretation_but_is_not_automatically_rejected(tmp_path):
    def write(name,data):
        (tmp_path/name).write_text(json.dumps(data))

    write('report.json',{'status':'complete','training_steps':0,'reader_unchanged':True,
                         'scores':{'all':{'known':{'correct':60,'total':64,'reliability_target_met':False}}}})
    write('protocol.json',{'declaration_sha256':'reference','reader':{'id':'reader'},'input_sha256':'inputs'})
    (tmp_path/'predictions.jsonl').write_text('retained evidence\n')
    write('complete.json',{'kind':'independent_fact_reference_complete_v1',
                          'files':{name:file_sha256(tmp_path/name) for name in ('report.json','protocol.json','predictions.jsonl')}})
    write('proof.json',{'verified':True,'report_sha256':file_sha256(tmp_path/'report.json'),'declaration_sha256':'reference','verifier_sha256':'v2','original_verifier_sha256':'original'})
    declaration={'reference_report_sha256':file_sha256(tmp_path/'report.json'),
                 'reference_complete_sha256':file_sha256(tmp_path/'complete.json'),
                 'reference_verification_sha256':file_sha256(tmp_path/'proof.json'),
                 'reference_declaration_sha256':'reference','reference_verifier_v2_sha256':'v2','reference_original_verifier_sha256':'original','reader':{'id':'reader'},'input_sha256':'inputs',
                 'reference_interpretation':{'training_informative':True}}
    assert require_reference(tmp_path,tmp_path/'proof.json',declaration)['scores']['all']['known']['correct']==60
    declaration['reference_interpretation']['training_informative']=False
    with pytest.raises(ValueError,match='interpretation'):
        require_reference(tmp_path,tmp_path/'proof.json',declaration)
    declaration['reference_interpretation']['training_informative']=True
    (tmp_path/'predictions.jsonl').write_text('changed evidence\n')
    with pytest.raises(ValueError,match='artifact changed'):
        require_reference(tmp_path,tmp_path/'proof.json',declaration)
