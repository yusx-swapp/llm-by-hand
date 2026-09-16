import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {makeHoverText,mapTypedLines} from '../web/course/hover.js';
const root=new URL('../lessons/',import.meta.url);
const read=(key,name)=>JSON.parse(readFileSync(new URL(`${key}/${name}`,root),'utf8'));
const text=key=>readFileSync(new URL(`${key}/reference.py`,root),'utf8');
const catalogue=JSON.parse(readFileSync(new URL('curriculum.json',root),'utf8')).lessons;
for(const spec of catalogue)test(`${spec.number}: exact type-along maps for ${spec.kind}`,()=>{
  const source=text(spec.id),lines=source.split('\n');
  const last=lines.map((line,index)=>({line,index})).filter(item=>item.line.trim()).at(-1).index+1;
  assert.equal(mapTypedLines(source,source).get(last),last);
});
test('MHA hover includes the actually observed QK shape',()=>{
  const key='mha_core',source=text(key),teaching=read(key,'lesson.json');
  const line=source.split('\n').findIndex(line=>line.includes('attn_weights = torch.matmul'))+1;
  const data={reference:source,notes:teaching.line_notes,symbols:teaching.symbols,provenance:read(key,'provenance.json'),observations:read(key,'course-observations.json'),tokens:[{line,start:5,end:17,path:'attn_weights',access:'write'}],line_map:{[line]:{method:'eager_attention_forward',observe_line:line,header:false}}};
  const result=makeHoverText(data,'basic',line,6,'attn_weights');
  assert.ok(result.includes('[2, 4, 4, 4]'));
  assert.ok(result.includes('参考运行'));
});
test('DPO branches not run by the current loss must not borrow same-name locals',()=>{
  const key='dpo_loss',source=text(key),teaching=read(key,'lesson.json');
  const line=source.split('\n').findIndex(line=>line.includes('per_sequence_loss = torch.relu'))+1;
  const data={reference:source,notes:teaching.line_notes,symbols:teaching.symbols,provenance:read(key,'provenance.json'),observations:read(key,'course-observations.json'),tokens:[],line_map:{[line]:{method:'_compute_loss',observe_line:line,header:false}}};
  const result=makeHoverText(data,'sigmoid',line,20,'per_sequence_loss');
  assert.ok(result.includes('当前示例未执行这条语句'));
  assert.ok(!result.includes('本次读取'));
});
