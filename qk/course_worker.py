"""Execute one numbered lesson in a disposable process, or rebuild its reference observations."""
from __future__ import annotations
import argparse
from collections.abc import Mapping
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from qk import curriculum as course
from qk import course_fixtures as fixtures
from qk.lesson_worker import describe as basic_describe

torch.set_num_threads(1)

def describe(value,depth=0):
    if isinstance(value,torch.device):return {'kind':'scalar','value':str(value)}
    if isinstance(value,Mapping) and not isinstance(value,dict):value=dict(value)
    return basic_describe(value,depth)

class Observer:
    def __init__(self,path,source,offset):
        self.path=str(path);self.offset=offset;self.pending={};self.lines={};self.context={};self.observing=False
        self.paths=sorted({t['path'] for t in course.token_locations(source) if t.get('path') and '.' in t['path']})

    def snapshot(self,frame):
        self.observing=True
        try:
            values={name:describe(value) for name,value in frame.f_locals.items() if not name.startswith('.')}
            for path in self.paths:
                parts=path.split('.');value=frame.f_locals.get(parts[0])
                if value is None:continue
                try:
                    for part in parts[1:]:value=getattr(value,part)
                    if not callable(value) or isinstance(value,torch.nn.Module):values[path]=describe(value)
                except (AttributeError,RuntimeError):pass
            if frame.f_code.co_name=='__init__':
                for value in values.values():
                    if value.get('kind')=='tensor':value.pop('sample',None)
            return values
        finally:self.observing=False

    def event(self,frame,event,arg):
        if frame.f_code.co_filename!=self.path:return None
        if event not in {'line','return'}:return self.event
        values=self.snapshot(frame);key=id(frame);previous=self.pending.get(key)
        if previous:
            line,before,ops=previous
            old=self.lines.get(str(line),{})
            self.lines[str(line)]={'before':before,'after':values,'ops':ops,'method':frame.f_code.co_name,'executions':old.get('executions',0)+1}
        if event=='line':
            if key not in self.pending:self.context[frame.f_code.co_name]=values
            self.pending[key]=(frame.f_lineno-self.offset,values,[])
        else:self.pending.pop(key,None)
        return self.event

class Operations(TorchDispatchMode):
    def __init__(self,observer):self.observer=observer
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        output=func(*args,**(kwargs or {}));obs=self.observer
        name=func._schema.name.split('::')[-1]
        if obs.observing or name not in {'view','_unsafe_view','transpose','permute','reshape','mm','bmm','addmm','cat','split_with_sizes','_softmax','gather','index_add_','mean','sum','sum.dim_IntList'}:return output
        frame=sys._getframe(1);direct=frame.f_code.co_filename==obs.path
        while frame and frame.f_code.co_filename!=obs.path:frame=frame.f_back
        if frame and id(frame) in obs.pending:
            ops=obs.pending[id(frame)][2]
            if len(ops)<20 and isinstance(output,torch.Tensor):
                ops.append({'op':name,'inputs':[list(a.shape) for a in args if isinstance(a,torch.Tensor)],'output':list(output.shape),'direct':direct})
        return output

def compare(actual,expected,path='result'):
    if isinstance(expected,torch.Tensor):
        if not isinstance(actual,torch.Tensor):raise AssertionError(f'{path} 应为 Tensor，实际为 {type(actual).__name__}')
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6,equal_nan=False,msg=lambda message:f'{path}: {message}')
    elif isinstance(expected,Mapping):
        if not isinstance(actual,Mapping) or set(actual)!=set(expected):raise AssertionError(f'{path} 返回字段不同')
        for key in expected:compare(actual[key],expected[key],f'{path}.{key}')
    elif isinstance(expected,(list,tuple)):
        if not isinstance(actual,(list,tuple)) or len(actual)!=len(expected):raise AssertionError(f'{path} 序列长度不同')
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,f'{path}[{i}]')
    elif isinstance(expected,float):
        if not math.isclose(float(actual),expected,rel_tol=1e-5,abs_tol=1e-6):raise AssertionError(f'{path}: {actual} != {expected}')
    elif actual!=expected:raise AssertionError(f'{path}: {actual} != {expected}')

def verify(lesson_id,source,path):
    issue=course.quick_error(lesson_id,source)
    if issue:return dict(passed=False,tests=[],**issue)
    candidate,_=course.load_object(lesson_id,source,path);expected_type=course.upstream_object(lesson_id)
    tests=[]
    for name in fixtures.variants(lesson_id):
        expected=actual=None
        try:
            expected=fixtures.make_case(lesson_id,expected_type,name);actual=fixtures.make_case(lesson_id,candidate,name)
            fixtures.transfer_state(expected,actual)
            want=fixtures.execute(expected);got=fixtures.execute(actual)
            compare(got,want)
            info=describe(got['output'])
            detail=f"{expected.title} · 输出/副作用与上游一致 · 非空梯度 {sum(value is not None for value in got['gradients'].values())}"
            tests.append(dict(id=name,name=expected.title,passed=True,detail=detail,output=info))
        except Exception as exc:
            tests.append(dict(id=name,name=name,passed=False,detail=f'{type(exc).__name__}: {str(exc)[:2000]}'))
        finally:
            for scenario in (expected,actual):
                if scenario and scenario.cleanup:scenario.cleanup()
    return dict(passed=bool(tests) and all(test['passed'] for test in tests),tests=tests,error=None)

def observe(lesson_id):
    source=course.reference(lesson_id);path=course.folder(lesson_id)/'reference.py'
    obj,offset=course.load_object(lesson_id,source,path)
    records={}
    for case in fixtures.variants(lesson_id):
        observer=Observer(path,source,offset);scenario=None
        try:
            sys.settrace(observer.event)
            with Operations(observer):
                scenario=fixtures.make_case(lesson_id,obj,case)
                torch.manual_seed(808);output=scenario.call()
        finally:
            sys.settrace(None)
            if scenario and scenario.cleanup:scenario.cleanup()
        records[case]={'id':case,'title':scenario.title,'dimensions':scenario.dimensions,'lines':observer.lines,'context':observer.context,'output':describe(output)}
    return {'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'worker_sha256':course.worker_digest(),
            'observed_at':time.time(),'torch':torch.__version__,'fixtures':records}

def safe_json(value):
    return json.loads(json.dumps(value,ensure_ascii=False),parse_constant=str)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['verify','observe','check-all','observe-all'])
    parser.add_argument('--lesson');parser.add_argument('--source',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic()
    try:
        if args.action in {'check-all','observe-all'}:
            rows=[]
            for spec in course.catalogue():
                key=spec['id'];source=course.reference(key);path=course.folder(key)/'reference.py'
                try:
                    data=observe(key) if args.action=='observe-all' else verify(key,source,path)
                    if args.action=='observe-all':
                        (course.folder(key)/'course-observations.json').write_text(json.dumps(safe_json(data),ensure_ascii=False,separators=(',',':')),encoding='utf-8')
                        row=dict(id=key,passed=True,fixtures=len(data['fixtures']))
                    else:row=dict(id=key,**data)
                except Exception as exc:row=dict(id=key,passed=False,error=f'{type(exc).__name__}: {exc}');traceback.print_exc()
                rows.append(row);print(f"{spec['number']:03} {key}: {row['passed']}",flush=True)
            result={'passed':all(row['passed'] for row in rows),'lessons':rows}
        elif args.action=='observe':result=observe(args.lesson)
        else:
            source=args.source.read_text(encoding='utf-8') if args.source else course.reference(args.lesson)
            path=args.source or course.folder(args.lesson)/'reference.py'
            result=verify(args.lesson,source,path.resolve())
        result['wall_seconds']=round(time.monotonic()-started,2)
    except Exception as exc:
        traceback.print_exc();result=dict(passed=False,tests=[],error=f'{type(exc).__name__}: {exc}')
    args.output.write_text(json.dumps(safe_json(result),ensure_ascii=False,indent=2),encoding='utf-8')
