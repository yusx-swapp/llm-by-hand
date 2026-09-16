"""Numbered course API. A small SQLite table, explicit ordering, no course-engine framework."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib,json,os,re,sqlite3,subprocess,sys,tempfile,threading,time,uuid,zipfile
from pathlib import Path
from typing import Literal
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field
from qk import curriculum as c

class CourseStore:
    def __init__(self,path):self.path=Path(path)
    @contextmanager
    def db(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(self.path,timeout=20);db.row_factory=sqlite3.Row
        try:
            with db:
                db.execute('CREATE TABLE IF NOT EXISTS course_stages (lesson TEXT,stage TEXT,source TEXT,passed INTEGER DEFAULT 0,passed_source TEXT,report TEXT,updated_at REAL,PRIMARY KEY(lesson,stage))')
                db.execute('CREATE TABLE IF NOT EXISTS course_meta (key TEXT PRIMARY KEY,value TEXT)')
                if not db.execute("SELECT 1 FROM course_meta WHERE key='prototype_migrated'").fetchone():
                    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='lesson_stages'").fetchone():
                        for row in db.execute('SELECT * FROM lesson_stages').fetchall():
                            report=json.loads(row['report']) if row['report'] else {}
                            verified=row['source'] if row['source'] is not None and report.get('source_sha256')==c.digest(row['source']) and report.get('passed') else None
                            db.execute('INSERT OR IGNORE INTO course_stages VALUES (?,?,?,?,?,?,?)',('llama_attention',row['stage'],row['source'],row['passed'],verified,row['report'],row['updated_at']))
                    db.execute("INSERT INTO course_meta VALUES ('prototype_migrated','1')")
                yield db
        finally:db.close()
    def all(self):
        with self.db() as db:rows=db.execute('SELECT * FROM course_stages').fetchall()
        return {(row['lesson'],row['stage']):dict(row) for row in rows}
    def status(self):
        rows=self.all();prior=True;result=[]
        for spec in c.catalogue():
            stages={stage:bool(rows.get((spec['id'],stage),{}).get('passed')) for stage in c.STAGES}
            completed=all(stages.values())
            migrated=spec['id']=='llama_attention' and any((spec['id'],stage) in rows for stage in c.STAGES)
            result.append({**spec,'completed':completed,'unlocked':prior or migrated,'stages':stages})
            prior=prior and completed
        return result
    def authorize(self,lesson,stage='follow'):
        status=next((item for item in self.status() if item['id']==lesson),None)
        if not status:raise HTTPException(404,'课程不存在。')
        if not status['unlocked']:raise HTTPException(403,'先按编号完成前面的课程，再进入本课；不能跳过基础直接训练。')
        index=c.STAGES.index(stage)
        if index and not status['stages'][c.STAGES[index-1]]:raise HTTPException(403,'先验证通过上一学习阶段。')
        return status
    def save(self,lesson,stage,source):
        with self.db() as db:db.execute('INSERT INTO course_stages(lesson,stage,source,updated_at) VALUES (?,?,?,?) ON CONFLICT(lesson,stage) DO UPDATE SET source=excluded.source,updated_at=excluded.updated_at',(lesson,stage,source,time.time()))
    def record(self,lesson,stage,source,report):
        with self.db() as db:
            db.execute('INSERT INTO course_stages(lesson,stage,passed,passed_source,report,updated_at) VALUES (?,?,?,?,?,?) ON CONFLICT(lesson,stage) DO UPDATE SET passed=MAX(course_stages.passed,excluded.passed),passed_source=CASE WHEN excluded.passed THEN excluded.passed_source ELSE course_stages.passed_source END,report=excluded.report,updated_at=excluded.updated_at',(lesson,stage,int(report['passed']),source if report['passed'] else None,json.dumps(report,ensure_ascii=False),time.time()))
    def project_sources(self):
        rows=self.all();needed=['mha_core','causal_mask','rms_norm','rope','swiglu','llama_attention','decoder_layer','causal_lm','moe_experts','moe_block','lm_loss','lm_collator','lr_schedule','pretrain_step','sft_collator','dpo_loss','top_p']
        missing=[key for key in needed if not rows.get((key,'recall'),{}).get('passed_source')]
        if missing:raise HTTPException(403,'训练串联需要先完成主线独立复现：'+', '.join(missing))
        return {spec['id']:rows[spec['id'],'recall']['passed_source'] for spec in c.catalogue() if rows.get((spec['id'],'recall'),{}).get('passed_source')}

class Submission(BaseModel):
    stage:Literal['follow','cloze','recall']
    source:str=Field(max_length=250_000)

class Reset(BaseModel):
    stage:Literal['follow','cloze','recall']

def execute_worker(lesson_id,source):
    started=time.monotonic();issue=c.quick_error(lesson_id,source)
    if issue:result=dict(passed=False,tests=[],**issue)
    else:
        with tempfile.TemporaryDirectory(prefix='qk-course-') as tmp:
            path=Path(tmp)/'learner.py';out=Path(tmp)/'result.json';path.write_text(source,encoding='utf-8')
            env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'}
            try:
                subprocess.run([sys.executable,str(c.ROOT/'qk/course_worker.py'),'verify','--lesson',lesson_id,'--source',str(path),'--output',str(out)],cwd=c.ROOT,env=env,capture_output=True,timeout=120)
                result=json.loads(out.read_text(encoding='utf-8')) if out.exists() else dict(passed=False,tests=[],error='执行器没有返回结果。')
            except subprocess.TimeoutExpired:result=dict(passed=False,tests=[],error='执行超过 120 秒；请检查无限循环或过大的张量。')
            except (OSError,ValueError) as exc:result=dict(passed=False,tests=[],error=str(exc))
    result.update(source_sha256=c.digest(source),wall_seconds=round(time.monotonic()-started,2),verified_at=time.time())
    return result

def create_course_router(database):
    router=APIRouter(prefix='/api/course',tags=['numbered curriculum']);store=CourseStore(database);lock=threading.Lock()
    runs=Path(database).parent/(Path(database).stem+'-runs')
    @router.get('')
    def catalogue():
        lessons=store.status();next_id=next((item['id'] for item in lessons if not item['completed']),lessons[-1]['id'])
        return dict(lessons=lessons,next_lesson=next_id)
    @router.get('/lesson/{lesson_id}')
    def lesson(lesson_id:str,stage:Literal['follow','cloze','recall']='follow'):
        status=store.authorize(lesson_id,stage);rows=store.all();row=rows.get((lesson_id,stage),{})
        data=c.content(lesson_id,'lesson.json');proof=c.content(lesson_id,'provenance.json');issue=c.runtime_issue(lesson_id)
        payload=dict(stage=stage,lesson=status,course=store.status(),provenance=proof,subtitle=data['subtitle'],context=data['context'],outcomes=data['outcomes'],runtime_context=data['runtime_context'],runtime_issue=issue,
                     progress={s:{'passed':status['stages'][s],'unlocked':i==0 or status['stages'][c.STAGES[i-1]]} for i,s in enumerate(c.STAGES)},
                     source=row['source'] if row.get('source') is not None else c.template(lesson_id,stage),report=json.loads(row['report']) if row.get('report') else None)
        if stage=='follow':
            ref=c.reference(lesson_id)
            if lesson_id=='llama_attention':
                observations=c.content(lesson_id,'observations.json')
                valid=observations.get('worker_sha256')==hashlib.sha256((c.ROOT/'qk/lesson_worker.py').read_bytes()).hexdigest()
            else:
                observations=c.content(lesson_id,'course-observations.json')
                valid=observations.get('worker_sha256')==c.worker_digest()
            valid=valid and observations.get('source_sha256')==proof['sha256']
            payload.update(reference=ref,notes=data['line_notes'],symbols=data['symbols'],tokens=c.token_locations(ref),line_map=c.line_map(ref),dependencies=c.content(lesson_id,'dependencies.json'),
                           observations=observations if valid and not issue else None,observation_issue=None if valid and not issue else issue or '参考运行记录需要重新生成。')
        if stage=='cloze':payload['gaps']=[{k:g[k] for k in ('id','title','goal')} for g in data['gaps']]
        return payload
    @router.post('/lesson/{lesson_id}/draft')
    def save(lesson_id:str,req:Submission):
        store.authorize(lesson_id,req.stage);store.save(lesson_id,req.stage,req.source);return dict(ok=True)
    @router.post('/lesson/{lesson_id}/verify')
    def verify(lesson_id:str,req:Submission):
        store.authorize(lesson_id,req.stage)
        issue=c.runtime_issue(lesson_id)
        if issue:raise HTTPException(409,issue)
        if not lock.acquire(blocking=False):raise HTTPException(409,'已有验证正在运行。')
        try:
            if lesson_id=='llama_attention':
                from qk.lesson import run_verification
                result=run_verification(req.source)
            else:result=execute_worker(lesson_id,req.source)
            store.record(lesson_id,req.stage,req.source,result)
            state=store.authorize(lesson_id,req.stage);result['progress']=state['stages'];result['course']=store.status()
            return result
        finally:lock.release()
    @router.post('/lesson/{lesson_id}/reset')
    def reset(lesson_id:str,req:Reset):
        store.authorize(lesson_id,req.stage);source=c.template(lesson_id,req.stage);store.save(lesson_id,req.stage,source)
        return dict(source=source,message='只恢复本阶段草稿，已验证的历史版本和其他课程记录保留。')
    @router.post('/project')
    def project():
        sources=store.project_sources()
        if not lock.acquire(blocking=False):raise HTTPException(409,'已有本地任务正在运行。')
        try:
            with tempfile.TemporaryDirectory(prefix='qk-project-') as tmp:
                path=Path(tmp)/'sources.json';out=Path(tmp)/'result.json';path.write_text(json.dumps(sources),encoding='utf-8')
                run_id=uuid.uuid4().hex;artifacts=runs/run_id;artifacts.mkdir(parents=True)
                env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'}
                subprocess.run([sys.executable,str(c.ROOT/'qk/course_project.py'),'--sources',str(path),'--output',str(out),'--artifacts',str(artifacts)],cwd=c.ROOT,env=env,capture_output=True,timeout=180)
                result=json.loads(out.read_text(encoding='utf-8')) if out.exists() else dict(passed=False,error='训练实验没有返回结果。')
                (artifacts/'student_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
                (artifacts/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
                if result.get('passed'):
                    bundle=artifacts/'student-project.zip'
                    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
                        for file in artifacts.glob('*'):
                            if file.is_file() and file!=bundle:archive.write(file,file.name)
                        for name in ('__init__.py','curriculum.py','course_project.py','course_fixtures.py'):
                            archive.write(c.ROOT/'qk'/name,'qk/'+name)
                        archive.write(c.LESSONS/'curriculum.json','lessons/curriculum.json')
                        for spec in c.catalogue():
                            for name in ('reference.py','provenance.json','LICENSE','NOTICE'):
                                file=c.folder(spec['id'])/name
                                if file.exists():archive.write(file,'lessons/'+spec['id']+'/'+name)
                        archive.write(c.ROOT/'requirements.txt','requirements.txt')
                        archive.writestr('RUN.md','# Local teaching experiment\n\nInstall requirements, then run:\n\npython -X utf8 qk/course_project.py --sources student_sources.json --output rerun.json --artifacts rerun-output\n\nThis bundle contains verified learner code and project-owned integration glue, not a trained production LLM. Execute trusted Python only.\n')
                    result['download_url']=f'/api/course/project/{run_id}/download'
                return result
        except subprocess.TimeoutExpired:return dict(passed=False,error='训练实验超过 180 秒。')
        finally:lock.release()
    @router.get('/project/{run_id}/download')
    def download(run_id:str):
        if not re.fullmatch('[0-9a-f]{32}',run_id):raise HTTPException(404,'实验不存在。')
        path=runs/run_id/'student-project.zip'
        if not path.is_file():raise HTTPException(404,'实验结果尚未生成。')
        return FileResponse(path,filename='llm-by-hand-project.zip',media_type='application/zip')
    return router
