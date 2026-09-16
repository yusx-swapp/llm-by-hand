// Explanations are attached to the hovered source occurrence, never a global variable-name guess.
const shape=value=>`[${(value || []).join(', ')}]`;
const rounded=value=>typeof value==='number' && Number.isFinite(value) ? Number(value.toPrecision(4)) : value;
const code=value=>'`'+String(value).replace(/`/g,'')+'`';
const axes={B:'批次',T:'当前 token',S:'完整 KV 长度',D:'隐藏维度',H:'Query 头',KV:'KV 头',d:'每头宽度',V:'词表',E:'专家数',K:'每 token 选择',grad_accum:'梯度累积',warmup:'预热步数',total_steps:'总训练步数'};

function describe(value) {
  if (!value) return '尚未在这个方法中绑定';
  if (value.kind==='tensor') return `${shape(value.shape)} · ${value.dtype}`;
  if (value.kind==='linear') return `Linear(${value.in_features} → ${value.out_features}) · weight ${shape(value.weight)} · bias ${value.bias ? shape(value.bias) : 'None'}`;
  if (value.kind==='cache') return `DynamicCache · 已缓存 ${value.length} token`;
  if (value.kind==='sequence') return '('+value.value.map(describe).join(', ')+')';
  if (value.kind==='config') return 'LlamaConfig · '+Object.entries(value.value).map(([key,item])=>`${key}=${item}`).join(', ');
  if (value.kind==='mapping') return '{'+Object.keys(value.value).join(', ')+'}';
  if (value.value===null) return 'None';
  if (typeof value.value==='boolean') return value.value ? 'True' : 'False';
  return String(value.value);
}

// Only a verified, ordered prefix is mapped. After a changed statement we stop attaching
// reference runtime values to learner code. Generic concept text remains available.
export function mapTypedLines(source,reference) {
  const expected=reference.split('\n'), actual=source.split('\n'), result=new Map();
  let cursor=0;
  for (let i=0;i<actual.length;i++) {
    const text=actual[i].trim();
    if (!text) continue;
    while (cursor<expected.length && !expected[cursor].trim()) cursor++;
    if (text.startsWith('#') && expected[cursor]?.trim()!==text) continue;
    while (cursor<expected.length && expected[cursor].trim().startsWith('#') && expected[cursor].trim()!==text) cursor++;
    const target=expected[cursor]?.trim();
    if (text===target) {result.set(i+1,cursor+1);cursor++;}
    else {
      if (target?.startsWith(text)) result.set(i+1,cursor+1);
      break;
    }
  }
  return result;
}

export function makeHoverText(data,fixtureId,line,column,word,{mapped=true,learner=false}={}) {
  const definition=data.symbols[word];
  if (!mapped) return definition
    ? `**${word}**\n\n${definition}\n\n---\n这里只显示概念解释。你的这一行尚未与参考源码对应，因此不套用参考 shape。可悬停左侧源码查看对应位置的真实运行。`
    : '这处草稿尚未与参考语句对应。参考运行状态不会被冒充成你的运行结果；可悬停左侧源码查看逐行解释。';
  const note=data.notes.find(item=>item.start<=line && line<=item.end);
  if (!note) return null;
  const fixture=data.observations?.fixtures?.[fixtureId];
  const location=data.line_map?.[String(line)];
  const observedLine=location?.observe_line || note.observe_line || line;
  const observation=fixture?.lines?.[String(observedLine)];
  const context=fixture?.context?.[location?.method || note.method || (line<28?'__init__':'forward')] || {};
  const unexecuted=!!fixture && !observation && location && !location.header;
  const before=observation?.before || context, after=observation?.after || context;
  const token=data.tokens.find(token=>token.line===line && token.start<=column && column<token.end);
  const key=token?.path || (word in before || word in after ? word : null);
  const previous=key ? before[key] : null, next=key ? after[key] : null;
  const sections=[`**${word && definition ? word : `第 ${line} 行`}**${definition ? ` — ${definition}` : ''}`];
  if(fixture) sections.push(`**参考运行 · ${fixture.title}**${learner ? '（来自 reference，不是你的草稿运行结果）' : ''}`);
  if (!fixture) sections.push(`参考运行记录当前不可用：${data.observation_issue || '未生成'}。上面是源码解释，不是动态执行结果。`);
  else if (unexecuted || (note.branch==='cache' && fixtureId!=='decode')) sections.push('**当前示例未执行这条语句。** 不会把其他分支或同名变量的状态套到这里；可切换运行示例查看不同路径。');
  else {
    if (previous || next) {
      const access=token?.access;
      if (access==='write' && observation) sections.push(`**${key} · 本次赋值**\n\n执行前 ${code(describe(previous))}\n\n执行后 ${code(describe(next))}`);
      else sections.push(`**${key} · ${observation ? '本次读取' : '入参 / 已初始化的模块示例'}**\n\n${code(describe(previous || next))}`);
    }
    const writes=data.tokens.filter(token=>token.line===observedLine && token.access==='write' && token.path && token.path!==key);
    const targets=[...new Set(writes.map(token=>token.path))].filter(path=>after[path]);
    if (targets.length) sections.push('**本语句的结果**\n\n'+targets.slice(0,3).map(path=>`${code(path)} → ${code(describe(after[path]))}`).join('  \n'));
    let ops=(observation?.ops || []).filter(op=>op.direct);
    if (word==='transpose' || word==='view') ops=ops.filter(op=>op.op===word);
    if (ops.length) sections.push('**ATen 形状记录 · 含当前语句的底层分解**\n\n'+ops.slice(0,4).map(op=>`${code(op.op)} ${code(op.inputs.map(shape).join(' × '))} → ${code(shape(op.output))}`).join('  \n'));
    if(observation?.executions>1) sections.push(`本行出现 ${observation.executions} 次执行观察，以上显示最后一次完成时的前后状态，不混合不同专家或循环迭代。`);
  }
  sections.push(`**本行 · ${note.title}**\n\n${note.why}`);
  const sample=(next || previous)?.sample;
  if (sample?.length && !unexecuted && !(note.branch==='cache' && fixtureId!=='decode')) sections.push(`前 ${sample.length} 个样本值（展示时舍入）：${code(sample.map(rounded).join(', '))}`);
  if(fixture) sections.push(`---\n${Object.entries(fixture.dimensions).map(([key,value])=>`${key}=${value}（${axes[key] || key}）`).join(' · ')}\n\n本地 CPU 示例 · 数据来自参考源码的真实执行。`);
  sections.push(`本课第 ${line} 行 · 上游 ${data.provenance.source_file.split('/').pop()} 第 ${data.provenance.first_line+line-1} 行`);
  return sections.join('\n\n');
}

export function installHovers(monaco,data,referenceEditor,learnerEditor,getFixture) {
  return monaco.languages.registerHoverProvider('python',{
    provideHover(model,position) {
      const isReference=model===referenceEditor.model, isLearner=model===learnerEditor.model;
      if (!isReference && !isLearner) return null;
      const word=model.getWordAtPosition(position);
      let line=position.lineNumber, column=position.column, mapped=true;
      if (isLearner) {
        const correspondence=mapTypedLines(model.getValue(),data.reference);
        line=correspondence.get(position.lineNumber);
        mapped=!!line;
        if (mapped) {
          const referenceLine=data.reference.split('\n')[line-1], learnerLine=model.getLineContent(position.lineNumber);
          column=position.column-(learnerLine.length-learnerLine.trimStart().length)+(referenceLine.length-referenceLine.trimStart().length);
        }
      }
      const text=makeHoverText(data,getFixture(),line,column,word?.word || '',{mapped,learner:isLearner});
      if (!text) return null;
      return {range:new monaco.Range(position.lineNumber,word?.startColumn || 1,position.lineNumber,word?.endColumn || model.getLineMaxColumn(position.lineNumber)),
        contents:[{value:text,isTrusted:false,supportHtml:false}]};
    },
  });
}
