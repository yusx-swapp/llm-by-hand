"""A numbered source curriculum: plain files and a small amount of source plumbing."""
from __future__ import annotations
import ast
import hashlib
import importlib
from importlib.metadata import distribution, version
import inspect
import json
from pathlib import Path
import sys
import textwrap
import types

ROOT=Path(__file__).resolve().parents[1]
LESSONS=ROOT/'lessons'
STAGES=('follow','cloze','recall')

def catalogue():
    return json.loads((LESSONS/'curriculum.json').read_text(encoding='utf-8'))['lessons']

def entry(lesson_id):
    for item in catalogue():
        if item['id']==lesson_id: return item
    raise KeyError(lesson_id)

def folder(lesson_id):
    entry(lesson_id)  # never use an unchecked request as a filesystem path
    return LESSONS/lesson_id

def content(lesson_id,name):
    return json.loads((folder(lesson_id)/name).read_text(encoding='utf-8'))

def reference(lesson_id):
    return (folder(lesson_id)/'reference.py').read_text(encoding='utf-8')

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def runtime_issue(lesson_id):
    spec=entry(lesson_id); proof=content(lesson_id,'provenance.json')
    package=spec['module'].split('.')[0]
    try:
        path=Path(distribution(package).locate_file(spec['module'].replace('.','/')+'.py'))
        if version(package)!=proof['version'] or hashlib.sha256(path.read_bytes()).hexdigest()!=proof['module_sha256']:
            return f"本课固定使用 {package} {proof['version']} 的已核对源码；当前安装不同，请先检查环境。"
    except Exception as exc:
        return f'无法检查 {package} 环境：{exc}'
    return None

def worker_digest():
    return hashlib.sha256(b''.join((ROOT/'qk'/name).read_bytes() for name in
        ('curriculum.py','course_worker.py','course_fixtures.py','lesson_worker.py'))).hexdigest()

def parsed(source):
    normalized=textwrap.dedent(source)
    before,after=source.splitlines(),normalized.splitlines()
    offsets=[len(a)-len(a.lstrip())-(len(b)-len(b.lstrip())) for a,b in zip(before,after)]
    return ast.parse(normalized),normalized,offsets

def token_locations(source):
    tree,_,offsets=parsed(source)
    def dotted(node):
        if isinstance(node,ast.Name): return node.id
        if isinstance(node,ast.Attribute):
            root=dotted(node.value)
            return root+'.'+node.attr if root else None
    tokens=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.Name,ast.Attribute)) and node.lineno==node.end_lineno:
            name=node.id if isinstance(node,ast.Name) else node.attr
            shift=offsets[node.end_lineno-1]
            tokens.append(dict(line=node.end_lineno,start=node.end_col_offset-len(name)+1+shift,
                end=node.end_col_offset+1+shift,name=name,path=dotted(node),access='write' if isinstance(node.ctx,ast.Store) else 'read'))
        elif isinstance(node,ast.arg):
            tokens.append(dict(line=node.lineno,start=node.col_offset+1+offsets[node.lineno-1],
                end=node.col_offset+len(node.arg)+1+offsets[node.lineno-1],name=node.arg,path=node.arg,access='parameter'))
    return tokens

def line_map(source):
    """Map multiline statements to execution lines, not entire control-flow branches."""
    tree,_,_=parsed(source); result={}
    functions=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)]
    simple=(ast.Assign,ast.AnnAssign,ast.AugAssign,ast.Expr,ast.Return,ast.Raise,ast.Assert)
    for line,text in enumerate(source.splitlines(),1):
        method=next((n for n in sorted(functions,key=lambda n:n.end_lineno-n.lineno) if n.lineno<=line<=n.end_lineno),None)
        statements=[n for n in ast.walk(tree) if isinstance(n,simple) and n.lineno<=line<=n.end_lineno]
        statement=min(statements,key=lambda n:n.end_lineno-n.lineno) if statements else None
        is_doc=statement is not None and isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Constant) and isinstance(statement.value.value,str)
        header=method is None or line<method.body[0].lineno or is_doc or text.lstrip().startswith('#')
        result[str(line)]=dict(method=method.name if method else None,observe_line=statement.lineno if statement and not is_doc else line,header=header)
    return result

def template(lesson_id,stage):
    source=reference(lesson_id); teaching=content(lesson_id,'lesson.json')
    if stage=='follow': return ''
    if stage=='cloze':
        lines=source.splitlines()
        for gap in reversed(teaching['gaps']): lines[gap['start']-1:gap['end']]=gap['replacement'].splitlines()
        return '\n'.join(lines)+'\n'
    tree,_,_=parsed(source); lines=source.splitlines(); spec=entry(lesson_id)
    if spec['kind']=='class':
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        output=[lines[cls.lineno-1]]
        for node in cls.body:
            if isinstance(node,ast.FunctionDef):
                start=min([node.lineno]+[d.lineno for d in node.decorator_list])
                output+=lines[start-1:node.body[0].lineno-1]+['        raise NotImplementedError("请独立实现")','']
    else:
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
        output=lines[:node.body[0].lineno-1]
        body_line=lines[node.body[0].lineno-1]
        indent=body_line[:len(body_line)-len(body_line.lstrip())]
        output.append(indent+'raise NotImplementedError("请独立实现")')
    return '\n'.join(output)+'\n'

def quick_error(lesson_id,source):
    try:
        tree,_,_=parsed(source); spec=entry(lesson_id)
        kind=ast.ClassDef if spec['kind']=='class' else ast.FunctionDef
        name=spec['symbol'].split('.')[-1]
        nodes=[n for n in tree.body if isinstance(n,kind) and n.name==name]
        if len(nodes)!=1: raise ValueError(f'请自己定义 {name}；空代码或只导入上游答案不算实现。')
        if spec['kind']=='class' and not any(isinstance(n,ast.FunctionDef) and n.name=='forward' for n in nodes[0].body):
            raise ValueError('请实现 forward 方法。')
        if any(isinstance(n,ast.Name) and n.id.startswith('__BLANK_') for n in ast.walk(tree)):
            raise ValueError('还有 __BLANK_ 标记未填写。')
    except (SyntaxError,ValueError) as exc: return {'error':f'{type(exc).__name__}: {exc}','line':getattr(exc,'lineno',None)}
    return None

def load_object(lesson_id,source,path,overrides=None):
    issue=quick_error(lesson_id,source)
    if issue: raise ValueError(issue['error'])
    spec=entry(lesson_id); upstream=importlib.import_module(spec['module']); tree,normalized,_=parsed(source)
    namespace=types.ModuleType('_qk_'+lesson_id)
    namespace.__dict__.update(vars(upstream)); namespace.__dict__.update(overrides or {})
    namespace.__dict__.update(__name__='_qk_'+lesson_id,__file__=str(path))
    sys.modules[namespace.__name__]=namespace
    leaf=spec['symbol'].split('.')[-1]; offset=0
    if spec['kind'] in {'method','static'}:
        parent_name=spec['symbol'].split('.')[0]; parent=getattr(upstream,parent_name)
        namespace.__dict__['_CourseBase']=parent
        wrapper=f'class {parent_name}(_CourseBase):\n'+textwrap.indent(normalized,'    ')
        tree=ast.parse(wrapper); offset=1
        # Documentation generation is not the method's runtime behavior and assumes upstream file locations.
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef):
                node.decorator_list=[d for d in node.decorator_list if not ((isinstance(d,ast.Name) and d.id=='auto_docstring') or (isinstance(d,ast.Call) and isinstance(d.func,ast.Name) and d.func.id=='auto_docstring'))]
        ast.fix_missing_locations(tree); exec(compile(tree,str(path),'exec'),namespace.__dict__)
        result=namespace.__dict__[parent_name]
    else:
        namespace.__dict__.pop(leaf,None)
        exec(compile(tree,str(path),'exec'),namespace.__dict__)
        result=namespace.__dict__[leaf]
    return result,offset

def upstream_object(lesson_id):
    spec=entry(lesson_id); obj=importlib.import_module(spec['module'])
    if spec['kind'] in {'method','static'}: return getattr(obj,spec['symbol'].split('.')[0])
    for part in spec['symbol'].split('.'): obj=getattr(obj,part)
    return obj
