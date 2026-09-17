import ast,hashlib,io,json,keyword,sqlite3,subprocess,sys,textwrap,tokenize,zipfile
from importlib.metadata import distribution
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app import create_app
from qk import curriculum as c
import qk.course_api as api
from qk.lesson import LessonStore

@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path/'legacy.sqlite3',lesson_db_path=tmp_path/'learning.sqlite3')) as value:yield value

def passed(source):return dict(passed=True,error=None,tests=[dict(id='test',name='API test double',passed=True,detail='')],source_sha256=c.digest(source),wall_seconds=.01)

def test_course_is_numbered_and_training_follows_model_foundations():
    specs=c.catalogue();assert [item['number'] for item in specs]==list(range(1,23))
    assert specs[0]['id']=='mha_core'
    assert next(item['number'] for item in specs if item['id']=='pretrain_step')<next(item['number'] for item in specs if item['id']=='sft_collator')<next(item['number'] for item in specs if item['id']=='dpo_loss')
    assert {'Llama','Qwen3','Gemma2','Mixtral','DeepSeek-V3'} <= {item['family'] for item in specs}

@pytest.mark.parametrize('spec',c.catalogue(),ids=lambda spec:spec['id'])
def test_each_lesson_is_exact_explained_and_has_reconstructable_blanks(spec):
    key=spec['id'];proof=c.content(key,'provenance.json');source=c.reference(key);teaching=c.content(key,'lesson.json')
    module=Path(distribution(spec['module'].split('.')[0]).locate_file(spec['module'].replace('.','/')+'.py'))
    expected=b''.join(module.read_bytes().splitlines(keepends=True)[proof['first_line']-1:proof['last_line']])
    assert (c.folder(key)/'reference.py').read_bytes()==expected
    assert hashlib.sha256(expected).hexdigest()==proof['sha256']
    assert (c.folder(key)/'LICENSE').exists() and (c.folder(key)/'NOTICE').exists()
    names={token.string for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type==tokenize.NAME and not keyword.iskeyword(token.string)}
    assert not names-teaching['symbols'].keys()
    assert all(any(note['start']<=line<=note['end'] for note in teaching['line_notes']) for line,text in enumerate(source.splitlines(),1) if text.strip())
    assert len(teaching['gaps'])>=2
    text=c.template(key,'cloze');lines=text.splitlines();original=source.splitlines()
    for gap in teaching['gaps']:
        index=next(i for i,line in enumerate(lines) if f"__BLANK_{gap['id']}__" in line)
        lines[index:index+1]=original[gap['start']-1:gap['end']]
    assert '\n'.join(lines)+'\n'==source
    ast.parse(textwrap.dedent(c.template(key,'recall')))
    observations=c.content(key,'course-observations.json')
    assert observations['worker_sha256']==c.worker_digest()
    assert observations['source_sha256']==proof['sha256']
    assert len(observations['fixtures'])>=2

def test_new_learner_starts_at_mha_and_cannot_open_post_training(client):
    result=client.get('/api/course').json()
    assert result['next_lesson']=='mha_core'
    assert sum(item['unlocked'] for item in result['lessons'])==1
    assert client.get('/api/course/lesson/sft_collator').status_code==403
    assert client.post('/api/course/lesson/dpo_loss/verify',json=dict(stage='follow',source='')).status_code==403
    first=client.get('/api/course/lesson/mha_core').json()
    assert first['source']=='' and first['observations']
    assert first['lesson']['number']==1

def test_sequential_three_stage_progress_and_no_answer_payload(client,monkeypatch):
    monkeypatch.setattr(api,'execute_worker',lambda key,source:passed(source))
    assert client.get('/api/course/lesson/mha_core?stage=cloze').status_code==403
    for stage in c.STAGES:
        client.post('/api/course/lesson/mha_core/draft',json=dict(stage=stage,source=c.reference('mha_core')))
        response=client.post('/api/course/lesson/mha_core/verify',json=dict(stage=stage,source=c.reference('mha_core')))
        assert response.status_code==200 and response.json()['passed']
        if stage!='recall':assert client.get('/api/course/lesson/mha_projection').status_code==403
    assert client.get('/api/course/lesson/mha_projection').status_code==200
    cloze=client.get('/api/course/lesson/mha_core?stage=cloze').json()
    assert all(key not in cloze for key in ('reference','symbols','notes','observations','tokens','dependencies'))
    assert client.get('/api/course/lesson/pretrain_step').status_code==403

def test_migrates_prototype_without_changing_old_records(tmp_path):
    path=tmp_path/'learning.sqlite3';old=LessonStore(path);source=c.reference('llama_attention')
    old.save('follow',source);old.record('follow',passed(source))
    with sqlite3.connect(path) as db:before=db.execute('SELECT * FROM lesson_stages').fetchall()
    store=api.CourseStore(path);status=store.status();row=store.all()['llama_attention','follow']
    assert row['source']==source and row['passed_source']==source
    assert next(item for item in status if item['id']=='llama_attention')['unlocked']
    assert not next(item for item in status if item['id']=='decoder_layer')['unlocked']
    with sqlite3.connect(path) as db:assert db.execute('SELECT * FROM lesson_stages').fetchall()==before

def test_project_uses_verified_recall_snapshots_not_later_edits(tmp_path):
    store=api.CourseStore(tmp_path/'learning.sqlite3')
    for spec in c.catalogue():
        for stage in c.STAGES:
            source=c.reference(spec['id']);store.save(spec['id'],stage,source);store.record(spec['id'],stage,source,passed(source))
    store.save('mha_core','recall','later broken edit')
    assert store.project_sources()['mha_core']==c.reference('mha_core')

def test_separate_database_paths_keep_independent_local_progress(tmp_path):
    with TestClient(create_app(tmp_path/'first.sqlite3')) as first:
        first.post('/api/course/lesson/mha_core/draft',json=dict(stage='follow',source='private draft'))
        assert first.get('/api/course/lesson/mha_core').json()['source']=='private draft'
    with TestClient(create_app(tmp_path/'second.sqlite3')) as second:
        assert second.get('/api/course/lesson/mha_core').json()['source']==''

def test_late_validation_keeps_latest_draft_and_verified_snapshot(client,tmp_path,monkeypatch):
    store=api.CourseStore(tmp_path/'learning.sqlite3')
    def fake(key,source):store.save(key,'follow','new draft');return passed(source)
    monkeypatch.setattr(api,'execute_worker',fake)
    client.post('/api/course/lesson/mha_core/verify',json=dict(stage='follow',source='submitted snapshot'))
    row=store.all()['mha_core','follow']
    assert row['source']=='new draft' and row['passed_source']=='submitted snapshot'

def test_autosaved_draft_survives_a_full_application_restart(tmp_path):
    database=tmp_path/'persistent.sqlite3';source='def half_finished():\n    value = 1\n'
    with TestClient(create_app(database)) as first:
        result=first.post('/api/course/lesson/mha_core/draft',json=dict(stage='follow',source=source))
        assert result.status_code==200 and result.json()['saved_at']>0
    with TestClient(create_app(database)) as restarted:
        data=restarted.get('/api/course/lesson/mha_core').json()
        assert data['source']==source and data['updated_at']>0

def test_manual_and_auto_checkpoints_can_be_listed_and_restored(client):
    first='def first_version():\n    return 1\n';second='def second_version():\n    return 2\n'
    a=client.post('/api/course/lesson/mha_core/checkpoint',json=dict(stage='follow',source=first,origin='manual')).json()
    client.post('/api/course/lesson/mha_core/checkpoint',json=dict(stage='follow',source=second,origin='auto'))
    client.post('/api/course/lesson/mha_core/checkpoint',json=dict(stage='follow',source=second,origin='auto'))
    history=client.get('/api/course/lesson/mha_core/revisions?stage=follow').json()['revisions']
    assert len(history)==2 and {item['origin'] for item in history}=={'manual','auto'}
    assert sum(item['current'] for item in history)==1
    restored=client.post('/api/course/lesson/mha_core/restore',json=dict(stage='follow',revision_id=a['id'])).json()
    assert restored['source']==first
    assert client.get('/api/course/lesson/mha_core').json()['source']==first
    assert client.post('/api/course/lesson/mha_core/restore',json=dict(stage='follow',revision_id=999999)).status_code==404

def test_imported_local_draft_is_a_recoverable_revision_not_an_overwrite(tmp_path):
    store=api.CourseStore(tmp_path/'learning.sqlite3')
    store.save('mha_core','follow','current')
    imported=store.checkpoint('mha_core','follow','old local draft','imported-legacy',created_at=123.0)
    store.save('mha_core','follow','current')
    history=store.revisions('mha_core','follow')
    assert any(item['id']==imported['id'] and item['origin']=='imported-legacy' and not item['current'] for item in history)
    assert store.all()['mha_core','follow']['source']=='current'

def test_real_complete_reference_course_and_training_project(tmp_path):
    output=tmp_path/'reference-check.json'
    subprocess.run([sys.executable,'-X','utf8',str(c.ROOT/'qk/course_worker.py'),'check-all','--output',str(output)],check=True,timeout=180)
    result=json.loads(output.read_text(encoding='utf-8'));assert result['passed'],result
    assert len(result['lessons'])==22
    project=tmp_path/'project.json'
    subprocess.run([sys.executable,'-X','utf8',str(c.ROOT/'qk/course_project.py'),'--output',str(project),'--artifacts',str(tmp_path/'artifacts')],check=True,timeout=180)
    run=json.loads(project.read_text(encoding='utf-8'));assert run['passed'],run
    assert run['mha_calls']>0 and run['parameter_tensors_updated']>0 and run['dpo_gradient_norm']>0
    assert len(run['pretrain_losses'])==4 and len(run['sft_losses'])==2
    assert run['checkpoint_reloaded'] and (tmp_path/'artifacts/checkpoint.pt').exists()
    assert run['moe_expert_gradient_norm']>0 and len(run['moe_losses'])==2
    assert run['exported_implementations']==22
