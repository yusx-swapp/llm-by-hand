import {$,api,escape as e,icon,dialog,toast,storage} from '../ui.js';
import {loadEditor,createCodeEditor} from './editor.js';
import {installHovers,mapTypedLines} from './hover.js';

const STAGES={
  follow:{number:'01',name:'跟敲理解',short:'看懂每一行，再亲手敲出来',instruction:'左侧是未经改写的真实源码，右侧由你来实现。悬停任意变量或语句，解释就在光标旁；也可以点「解释光标处」。'},
  cloze:{number:'02',name:'核心填空',short:'保留上下文，补回关键实现',instruction:'Reference 和过程解释已经收起。补齐本课的关键代码；可以改写周围代码，最终按实际行为验证，不按字符串对答案。'},
  recall:{number:'03',name:'独立复现',short:'不看答案，从接口写出完整实现',instruction:'只保留函数或类的接口，不提供 reference 或实现提示。独立写出本课实现，用真实输入验证数值、梯度或数据处理行为。'},
};
const main=$('#lesson-content');
let active,epoch=0,courseData=[];

const number=spec=>String(spec.number).padStart(3,'0');
const lessonLink=(spec,stage='follow')=>`#/${number(spec)}/${stage}`;
async function switchLesson(spec){if(!spec.unlocked){toast('先按编号完成前面的课程。');return;}if(!active || await active.save())location.hash=lessonLink(spec);}
function renderCourse(selected){
  let previous='';
  $('#course-map').innerHTML='<div class="course-map-heading"><strong>LLM 全流程</strong><span>001 → 022 · 顺序学习</span></div>'+courseData.map(item=>{
    const heading=item.chapter!==previous?`<h2>${e(item.chapter)}</h2>`:'';previous=item.chapter;
    return heading+`<button class="course-item ${item.id===selected?'current':''}" data-lesson="${item.id}" ${item.unlocked?'':'disabled'} title="${item.unlocked?e(item.title):'先完成之前的课程'}"><span class="course-number">${number(item)}</span><span><strong>${e(item.title)}</strong><small>${e(item.family)}</small></span>${icon(item.completed?'check':item.unlocked?'chevron':'lock')}</button>`;
  }).join('')+`<button class="button secondary project-launch" id="run-course-project" ${courseData.every(item=>item.completed)?'':'disabled'}>${icon('route')} 跑通我的训练链路</button><p class="course-map-note">主线使用你验证通过的独立实现，串起 Pretrain → SFT → DPO → 生成。</p>`;
  document.querySelectorAll('.course-item[data-lesson]').forEach(button=>button.onclick=()=>switchLesson(courseData.find(item=>item.id===button.dataset.lesson)));
  $('#run-course-project').onclick=runProject;
}
async function runProject(){
  if(!await dialog({title:'运行你的完整训练链路？',symbol:'route',confirm:'开始本地实验',body:'<p>使用已验证的独立复现版本，运行小型 CPU 模型的 Pretrain、SFT、DPO、KV-cache 生成及 checkpoint 保存/重载。这是教学实验，不代表大模型训练质量。</p>'}))return;
  const button=$('#run-course-project');button.disabled=true;button.textContent='训练实验运行中…';
  try{const result=await api('/course/project',{});await dialog({title:result.passed?'训练链路已跑通':'训练实验需要检查',symbol:result.passed?'check':'alert',confirm:'返回课程',cancel:'',body:`${result.download_url?`<p><a class="button secondary" href="${e(result.download_url)}">下载已验证代码与 checkpoint</a></p>`:''}<pre class="project-report">${e(JSON.stringify(result,null,2))}</pre>`});}
  catch(error){toast(error.message,true);}finally{if(button.isConnected){button.disabled=false;button.textContent='跑通我的训练链路';}}
}
function stageButtons(data){return Object.entries(STAGES).map(([key,stage])=>`<button class="stage-button ${key===data.stage?'active':''} ${data.progress[key].passed?'completed':''}" data-stage="${key}" ${data.progress[key].unlocked?'':'disabled'} ${key===data.stage?'aria-current="step"':''}><span class="stage-number">${data.progress[key].passed?icon('check'):stage.number}</span><span><strong>${stage.name}</strong><small>${stage.short}</small></span><span class="stage-mark">${!data.progress[key].unlocked?icon('lock'):key===data.stage?'<i></i>':''}</span></button>`).join('<span class="stage-connector" aria-hidden="true">→</span>');}

async function mount(data,token){
  const stage=data.stage, follow=stage==='follow',spec=data.lesson,endpoint=`/course/lesson/${spec.id}`;
  const observedCases=Object.values(data.observations?.fixtures || {});
  courseData=data.course;renderCourse(spec.id);
  $('#course-breadcrumb').innerHTML=`${e(spec.chapter)} <span>/</span> ${e(spec.family)} <span>/</span> <strong>${number(spec)}</strong>`;
  let disposed=false,ready=false,applying=false,learner,reference,hovers,saveTimer,periodicSaveTimer,checkpointTimer,saveQueue=Promise.resolve(true),busy=false;
  let source=data.source,persisted=data.source,lastSavedAt=data.updated_at || null,lastCheckpointSource=data.source,report=data.report,runError='',fixture=observedCases[0]?.id || 'basic';
  const alive=()=>!disposed && token===epoch;
  const pendingKey=`lesson:${data.provenance.sha256}:${stage}`;
  const pending=storage.get(pendingKey);
  if(pending!==null && pending!==source){
    const recover=await dialog({title:'恢复尚未保存的草稿？',symbol:'refresh',confirm:'恢复浏览器草稿',cancel:'使用服务器草稿',body:'<p>这个浏览器里还有一次未完成保存的编辑。三个阶段的草稿是分开的，恢复不会覆盖其他阶段。</p>'});
    if(recover) source=pending; else storage.remove(pendingKey);
  }
  if(!alive()) return {dispose:async()=>{},dirty:()=>false};
  document.title=`${number(spec)} ${spec.title} · ${STAGES[stage].name} — LLM by Hand`;
  main.innerHTML=`<section class="lesson" data-stage="${stage}" data-lesson="${spec.id}">
    <div class="lesson-heading"><div><div class="eyebrow"><span>源码课 ${number(spec)} / ${courseData.length}</span><i></i> ${e(spec.chapter)} / ${e(spec.family)}</div><h1>${e(spec.title)}</h1></div><button class="source-link" id="source-info">${icon('code')} ${e(data.provenance.package || 'transformers')} ${e(data.provenance.version)}<span>${data.provenance.last_line-data.provenance.first_line+1} 行真实源码</span>${icon('external')}</button></div>
    <nav class="stage-nav" aria-label="本课的三个学习阶段">${stageButtons(data)}</nav>
    <div class="stage-instruction"><span class="instruction-symbol">${icon(follow?'bulb':stage==='cloze'?'code':'terminal')}</span><p>${STAGES[stage].instruction}</p></div>
    ${data.runtime_issue?`<div class="warning">${icon('alert')}<p>${e(data.runtime_issue)}</p></div>`:''}
    <div class="workbench ${follow?'type-along':'practice'}">
      ${follow?`<section class="code-pane reference-pane" aria-label="对照源码"><header class="pane-header"><div><span class="file-icon">${icon('book')}</span><strong>Reference</strong><span class="pane-meta">上游原样保留 · 只读</span></div><button class="text-button" id="explain-reference">${icon('bulb')} 解释光标处</button></header><div class="reference-tools"><label>运行示例 <select id="fixture-select" aria-label="选择参考运行示例">${observedCases.map(item=>`<option value="${e(item.id)}">${e(item.title)}</option>`).join('')}</select></label><div class="jump-links" aria-label="源码位置">${data.notes.filter(note=>note.start>1).slice(0,3).map(note=>`<button data-line="${note.start}" title="${e(note.title)}">L${note.start}</button>`).join('')}</div></div><div id="reference-editor" class="editor-host"><div class="loading"><span class="spinner"></span> 加载本地编辑器…</div></div><footer class="source-footer"><span id="fixture-summary"></span><button class="text-button" id="support-source">课程上下文 ${icon('external')}</button></footer></section>`:
      `<aside class="exercise-context">${stage==='cloze'?`<div class="context-heading"><span class="eyebrow">RETRIEVE · 再想一遍</span><h2>补回实现中的关键决策</h2><p>不是随机挖空。每一处对应刚才跟敲时的一条数据流或配置关系。</p></div><div class="gap-list">${data.gaps.map(gap=>`<button class="gap-item" data-gap="${gap.id}"><span class="gap-number">${gap.id}</span><span><strong>${e(gap.title)}</strong><small>${e(gap.goal)}</small><em class="gap-state">待填写</em></span>${icon('chevron')}</button>`).join('')}</div><p class="context-footnote">「已填写」只表示占位符已替换，正确性仍需运行验证。</p>`:
      `<div class="context-heading"><span class="eyebrow">REBUILD · 独立写出来</span><h2>这一次，代码由你组织。</h2><p>不提供实现步骤，不展示参考答案。从接口写出当前课程的功能。</p></div><div class="recall-contract"><h3>本课在流程中的位置</h3><p>${e(data.context)}</p><h3>需要满足的行为</h3><ul>${data.outcomes.map(text=>`<li>${e(text)}</li>`).join('')}</ul><h3>如何验证</h3><p>使用真实上游对象和小输入，对照完整输出、状态变化和相关梯度。允许等价写法，不要求逐字复现。</p></div>`}<button class="button secondary back-to-learning" id="back-to-learning">${icon('back')} 回到上一阶段复习</button></aside>`}
      <section class="code-pane learner-pane" aria-label="自己动手实现"><header class="pane-header"><div><span class="file-icon">${icon('code')}</span><strong>你的实现</strong><span class="pane-meta">${stage==='follow'?'照着左侧，亲手敲一遍':stage==='cloze'?'补全 __BLANK_ 标记':'从接口独立复现'}</span></div><button class="save-status" id="save-status" title="点击立即保存" aria-live="polite">${source?(lastSavedAt?`已保存 · ${new Date(lastSavedAt*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'})}`:'草稿已载入'):'尚未输入'}</button></header><div class="learner-tools"><span>attention.py <i>·</i> Python</span><div><button id="runtime-context" class="text-button">已提供哪些运行依赖？</button>${follow?`<button id="explain-learner" class="text-button">${icon('bulb')} 解释光标处</button>`:''}</div></div><div class="learner-editor-wrap"><div id="learner-editor" class="editor-host"></div>${follow?`<div class="empty-editor-prompt" id="empty-editor-prompt"><span class="typing-cursor"></span><strong>从第一行开始，敲出自己的理解。</strong><p>对照左侧的装饰器和 class 开始。卡住时，把鼠标放到对应的变量或语句上。</p><span>这里不会自动填入 reference。</span></div>`:''}</div><footer class="editor-status"><span id="cursor-status">Ln 1, Col 1</span><span>${follow?'悬停 / 点按解释 · ':''}UTF-8 · 4 spaces</span></footer><div class="practice-actions"><button class="text-button" id="reset-draft">${icon('refresh')} 重置本阶段草稿</button><div class="save-actions"><button class="text-button" id="save-history">${icon('clock')} 保存记录</button><button class="button secondary" id="save-progress">${icon('check')} 保存进度 <kbd>Ctrl S</kbd></button><button class="button primary" id="verify-code">${icon('play')} 验证我的实现 <kbd>Ctrl ↵</kbd></button></div></div></section>
    </div>
    <section class="verification" aria-label="实现验证"><div class="verification-heading"><div><span class="section-symbol">${icon('layers')}</span><h2>让实现真正接回模型</h2><span>不是比对答案文本</span></div><button class="text-button" id="toggle-results" aria-expanded="true">收起结果 ${icon('chevron')}</button></div><div id="verification-body" aria-live="polite"></div></section>
    <div id="stage-completion"></div>
    <footer class="lesson-footer"><span>${icon('lock')} 本地保存 · 仅执行你信任的 Python 代码</span><span>从基础到训练按编号推进。完成独立复现后，再进入下一课或运行训练串联实验。</span></footer>
  </section>`;
  $('.learner-tools>span').innerHTML=`${e(spec.id)}.py <i>·</i> Python`;
  if($('#empty-editor-prompt p'))$('#empty-editor-prompt p').textContent='对照左侧的接口和代码，从第一行开始敲。卡住时，把鼠标放到对应变量或语句上。';

  const savedClock=timestamp=>new Date(timestamp*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
  function saveLabel(text,error=false){if(alive()){ $('#save-status').textContent=text;$('#save-status').classList.toggle('error',error); }}
  function savedLabel(prefix='已自动保存'){saveLabel(lastSavedAt?`${prefix} · ${savedClock(lastSavedAt)}`:prefix);}
  function persist(notify=false){
    clearTimeout(saveTimer);
    const snapshot=source;
    saveQueue=saveQueue.catch(()=>false).then(async()=>{
      if(snapshot===persisted){if(snapshot===source)storage.remove(pendingKey);if(notify)savedLabel('进度已保存');return true;}
      saveLabel('正在保存…');
      try{
        const result=await api(endpoint+'/draft',{stage,source:snapshot});persisted=snapshot;lastSavedAt=result.saved_at;
        if(source===snapshot)storage.remove(pendingKey);
        if(source===persisted)savedLabel();else saveLabel('尚有新编辑');
        if(notify && alive())toast('本阶段草稿已保存。');return true;
      }catch(error){saveLabel('保存失败 · 点此重试',true);if(notify && alive())toast(error.message,true);return false;}
    });return saveQueue;
  }

  async function checkpoint(notify=true,origin='manual'){
    const snapshot=source;
    if(!await persist(false))return false;
    try{
      const result=await api(endpoint+'/checkpoint',{stage,source:snapshot,origin});
      persisted=snapshot;lastSavedAt=result.saved_at;lastCheckpointSource=snapshot;
      if(source===snapshot){storage.remove(pendingKey);savedLabel(origin==='manual'?'已手动保存':'已建立自动快照');}
      if(notify)toast('已保存进度，并建立可恢复记录。');return true;
    }catch(error){saveLabel('保存失败 · 点此重试',true);if(notify)toast(error.message,true);return false;}
  }

  function changed(value){
    if(!alive())return;
    source=value;
    if(applying)return;
    storage.set(pendingKey,value);saveLabel('未保存的编辑…');
    clearTimeout(saveTimer);saveTimer=setTimeout(()=>persist(),400);
    if(follow)$('#empty-editor-prompt').hidden=!!value.length;
    if(stage==='cloze')updateGaps();
    if(report)markStale();
  }

  async function isCurrentReport(){return report?.source_sha256===await hash(source);}
  async function markStale(){
    const snapshot=source, result=await isCurrentReport();
    if(!alive() || snapshot!==source) return;
    const notice=$('#stale-result');if(notice)notice.hidden=result;
  }
  function updateGaps(){
    if(!learner) return;
    const remaining=new Set(learner.markGaps());
    document.querySelectorAll('[data-gap]').forEach(button=>{
      const filled=!!source.trim() && !remaining.has(Number(button.dataset.gap));button.classList.toggle('filled',filled);
      $('.gap-state',button).textContent=filled?'已填写 · 待验证':'待填写';
    });
  }

  async function showHistory(){
    if(!ready)return;
    let history;
    try{history=(await api(`${endpoint}/revisions?stage=${stage}`)).revisions;}
    catch(error){toast(error.message,true);return;}
    if(!history.length){
      await dialog({title:'还没有保存记录',symbol:'clock',confirm:'返回编辑',cancel:'',body:'<p>自动保存已经开启。点击「保存进度」会额外建立一个可恢复记录；持续编辑时也会定期建立快照。</p>'});return;
    }
    const labels={manual:'手动保存',auto:'自动快照','imported-legacy':'旧工程草稿','before-restore':'恢复前备份'};
    let selected=null;
    const opened=dialog({title:'保存记录',symbol:'clock',confirm:'关闭',cancel:'',body:`<p>当前草稿会自动保存。恢复旧记录前，当前版本也会先备份。</p><div class="save-history-list">${history.map(item=>`<button type="button" class="save-history-item ${item.current?'current':''}" data-revision="${item.id}" ${item.current?'disabled':''}><span><strong>${e(labels[item.origin] || item.origin)}</strong><small>${e(new Date(item.created_at*1000).toLocaleString())} · ${item.characters} 字符</small></span><em>${item.current?'当前版本':'恢复'}</em></button>`).join('')}</div>`});
    document.querySelectorAll('[data-revision]').forEach(button=>button.onclick=()=>{selected=Number(button.dataset.revision);$('#app-dialog').close('cancel');});
    await opened;
    if(!selected)return;
    if(!await dialog({title:'恢复这条保存记录？',symbol:'refresh',confirm:'恢复',cancel:'保留当前版本',body:'<p>当前代码会先自动备份，然后编辑器将切换到选择的历史版本。其他课程和阶段不受影响。</p>'}))return;
    try{
      const result=await api(endpoint+'/restore',{stage,revision_id:selected});
      applying=true;source=result.source;persisted=result.source;lastSavedAt=result.saved_at;lastCheckpointSource=result.source;
      learner.model.setValue(result.source);applying=false;storage.remove(pendingKey);savedLabel('已恢复并保存');report=null;runError='';learner.error(null,'');
      if(follow)$('#empty-editor-prompt').hidden=!!source.length;if(stage==='cloze')updateGaps();renderReport();toast('已恢复保存记录。');
    }catch(error){applying=false;toast(error.message,true);}
  }

  function renderProgress(){
    if(!alive())return;
    $('.stage-nav').innerHTML=stageButtons(data);bindStageButtons();
    const completion=$('#stage-completion'),next={follow:'cloze',cloze:'recall'}[stage],nextLesson=courseData[spec.number];
    completion.innerHTML=data.progress[stage].passed
      ? `<div class="completion ${stage==='recall'?'course-complete':''}"><span class="completion-symbol">${icon('check')}</span><div><strong>${stage==='recall'?`${number(spec)} · 本课独立复现已通过`:`${STAGES[stage].name}已有通过记录`}</strong><p>${stage==='recall'?'已保存本次通过的实现版本，后面的训练实验会使用已验证的代码。':'可以继续下一阶段；当前草稿若又有编辑，需重新验证才代表这个新版本。'}</p></div>${next?`<button class="button primary" id="continue-stage">${next==='cloze'?'进入核心填空':'进入独立复现'} ${icon('arrow')}</button>`:nextLesson?`<button class="button primary" id="continue-lesson" ${nextLesson.unlocked?'':'disabled'}>下一课 ${number(nextLesson)} ${icon('arrow')}</button>`:'<button class="button primary" id="final-project">运行训练串联实验</button>'}</div>`:'';
    if($('#continue-stage'))$('#continue-stage').onclick=()=>go(next);
    if($('#continue-lesson'))$('#continue-lesson').onclick=()=>switchLesson(nextLesson);
    if($('#final-project'))$('#final-project').onclick=runProject;
  }

  function renderReport(){
    if(!alive())return;
    const node=$('#verification-body');node.hidden=false;$('#toggle-results').setAttribute('aria-expanded','true');
    if(busy){node.innerHTML='<div class="running"><span class="spinner"></span><div><strong>正在运行你的实现…</strong><p>加载本地依赖，然后运行本课真实输入与行为检查。你可以继续编辑；验证使用点击时的代码快照。</p></div></div>';return;}
    if(runError){node.innerHTML=`<div class="run-error">${icon('alert')}<div><strong>验证暂时未完成</strong><p>${e(runError)}</p></div></div>`;return;}
    if(!report){node.innerHTML=`<div class="validation-intro"><p>本课练习与后续训练的联系：</p><p>${e(data.context)}</p><div>${data.outcomes.map(text=>`<span>${icon('circle')}${e(text)}</span>`).join('')}</div></div>`;return;}
    const passed=report.tests.filter(test=>test.passed).length;
    node.innerHTML=`<div class="validation-summary"><span class="result-symbol ${report.passed?'pass':'fail'}">${icon(report.passed?'check':'alert')}</span><div><h3>${report.passed?'实现验证通过':'实现还需要调整'}</h3><p>${report.tests.length?`${passed} / ${report.tests.length} 项行为检查通过`:'代码尚未进入行为检查'}${report.wall_seconds?` · 总用时 ${report.wall_seconds}s`:''}</p></div>${report.passed?'<span class="verified-label">本阶段已解锁后续学习</span>':''}</div><div class="stale-result" id="stale-result" hidden>这份结果对应上一次提交的代码；当前编辑还没有验证。</div>${report.error?`<pre class="error-detail">${e(report.error)}</pre>`:''}<div class="checks">${report.tests.map(test=>`<details class="check ${test.passed?'pass':'fail'}" ${!test.passed?'open':''}><summary>${icon(test.passed?'check':'alert')}<span>${e(test.name)}</span><strong>${test.passed?'通过':'需调整'}</strong></summary><pre>${e(test.detail)}</pre></details>`).join('')}</div>${report.tests.length?'<p class="validation-note">比较完整张量而非抽样；允许等价写法。数值容差 rtol=1e-5、atol=1e-6。</p>':''}`;
    markStale();
  }

  async function verify(){
    if(!ready || busy || !alive() || data.runtime_issue)return;
    const snapshot=source;busy=true;runError='';setBusy();
    try{
      if(!await persist())throw new Error('草稿尚未保存成功。请先重试保存，再进行验证。');
      if(!alive())return;
      const result=await api(endpoint+'/verify',{stage,source:snapshot});
      if(!alive())return;
      report=result;learner.error(source===snapshot ? result.line : null,result.error || '');
      Object.entries(result.progress || {}).forEach(([key,passed],index)=>{data.progress[key].passed=passed;});
      Object.keys(STAGES).forEach((key,index,keys)=>{data.progress[key].unlocked=!index || data.progress[keys[index-1]].passed;});
      if(result.course){courseData=result.course;data.course=result.course;renderCourse(spec.id);}
      renderProgress();
    }catch(error){if(alive())runError=error.message;}
    finally{if(alive()){busy=false;setBusy();}}
  }
  function setBusy(){
    $('#verify-code').disabled=!ready || busy || !!data.runtime_issue;$('#reset-draft').disabled=!ready || busy;
    $('#save-progress').disabled=!ready;$('#save-history').disabled=!ready;
    $('#verify-code').innerHTML=busy?'<span class="spinner"></span> 正在验证…':`${icon('play')} 验证我的实现 <kbd>Ctrl ↵</kbd>`;
    renderReport();
  }

  async function go(next){
    if(!ready || next===stage)return;
    if(!data.progress[next]?.unlocked){toast('先验证通过上一阶段。');return;}
    if(await persist(true))location.hash=lessonLink(spec,next);
  }
  function bindStageButtons(){document.querySelectorAll('.stage-nav button[data-stage]').forEach(button=>{button.disabled=!ready || !data.progress[button.dataset.stage].unlocked;button.onclick=()=>go(button.dataset.stage);});}
  function dimensions(){
    const current=data.observations?.fixtures?.[fixture];
    $('#fixture-summary').textContent=current ? Object.entries(current.dimensions).map(([name,value])=>`${name}=${value}`).join(' · ') : data.observation_issue || '运行观察不可用';
  }
  function dependencyInfo(){
    const body=`<p>${e(data.runtime_context)}</p><p><strong>运行的是你定义的实现，不是悄悄调用上游答案来替你通过。</strong>方法片段由运行器放回所属类；仅文档生成装饰器跳过副作用，源码本身不改。</p><p>验证器执行本地 Python，不是安全沙箱。只运行你信任的代码。</p>`;
    dialog({title:'运行上下文已经准备好',body,symbol:'box',confirm:'回到实现',cancel:''});
  }
  $('#runtime-context').onclick=dependencyInfo;
  $('#source-info').onclick=()=>dialog({title:'这份 reference 从哪里来？',symbol:'code',confirm:'返回课堂',cancel:'',body:`<p>直接取自本机安装的 <strong>${e(data.provenance.package || 'transformers')} ${e(data.provenance.version)}</strong>，不是改写后的伪代码。</p><code>${e(data.provenance.source_file)}</code><p>上游第 ${data.provenance.first_line}–${data.provenance.last_line} 行 · Apache-2.0。</p><p>导入、方法所属类和小型输入在运行器中单独提供。</p><p><a class="external-link" href="${e(data.provenance.upstream_url)}" target="_blank" rel="noopener noreferrer">打开上游源码 ${icon('external')}</a></p><details><summary>源码指纹</summary><code>${e(data.provenance.sha256)}</code></details>`});
  $('#verify-code').onclick=verify;
  $('#save-status').onclick=()=>checkpoint(true,'manual');
  $('#save-progress').onclick=()=>checkpoint(true,'manual');
  $('#save-history').onclick=showHistory;
  $('#toggle-results').onclick=()=>{const body=$('#verification-body');body.hidden=!body.hidden;$('#toggle-results').setAttribute('aria-expanded',String(!body.hidden));$('#toggle-results').innerHTML=`${body.hidden?'展开结果':'收起结果'} ${icon('chevron')}`;};
  $('#reset-draft').onclick=async()=>{
    if(!ready || busy || !await dialog({title:'重置这个阶段的草稿？',symbol:'refresh',confirm:'重置草稿',cancel:'保留我的代码',danger:true,body:'<p>本阶段恢复为空白或初始骨架。其他阶段的草稿和已有通过记录都保留。</p>'}))return;
    await checkpoint(false,'manual');
    try{const value=await api(endpoint+'/reset',{stage});if(!alive())return;applying=true;source=value.source;persisted=source;lastSavedAt=value.saved_at;lastCheckpointSource=source;learner.model.setValue(source);applying=false;clearTimeout(saveTimer);storage.remove(pendingKey);report=null;runError='';learner.error(null,'');savedLabel('本阶段草稿已重置');renderReport();}
    catch(error){if(alive())toast(error.message,true);}
  };
  if($('#back-to-learning'))$('#back-to-learning').onclick=()=>go(stage==='cloze'?'follow':'cloze');
  renderProgress();setBusy();

  const monaco=await loadEditor();
  if(!alive())return {dispose:async()=>{},dirty:()=>false};
  if(follow)reference=createCodeEditor(monaco,$('#reference-editor'),data.reference,{readOnly:true,id:`reference-${token}`,explanations:true});
  learner=createCodeEditor(monaco,$('#learner-editor'),source,{id:`learner-${token}`,explanations:follow,onChange:changed,onRun:verify,onCursor:position=>{
    if(!alive())return;$('#cursor-status').textContent=`Ln ${position.lineNumber}, Col ${position.column}`;
    if(follow){const line=mapTypedLines(source,data.reference).get(position.lineNumber);reference.highlight(line);if(line)reference.view.revealLineInCenterIfOutsideViewport(line);}
  }});
  if(follow){
    $('#empty-editor-prompt').hidden=!!source.length;
    hovers=installHovers(monaco,data,reference,learner,()=>fixture);
    $('#explain-reference').onclick=()=>reference.explain();$('#explain-learner').onclick=()=>learner.explain();
    $('#fixture-select').onchange=event=>{fixture=event.target.value;dimensions();reference.view.trigger('lesson','editor.action.hideHover',{});learner.view.trigger('lesson','editor.action.hideHover',{});};
    document.querySelectorAll('[data-line]').forEach(button=>button.onclick=()=>{const line=Number(button.dataset.line);reference.jump(line,9);reference.highlight(line);});
    $('#support-source').onclick=()=>dialog({title:'本课在整体流程中的位置',symbol:'layers',confirm:'返回跟敲',cancel:'',body:`<p>${e(data.context)}</p><p>${e(data.runtime_context)}</p>${Object.entries(data.dependencies).map(([name,item])=>`<details class="dependency-source"><summary>${e(name)} <small>上游 L${item.first_line}–L${item.last_line}</small></summary><pre>${e(item.source)}</pre></details>`).join('')}`});
    dimensions();
  }else if(stage==='cloze'){
    updateGaps();document.querySelectorAll('[data-gap]').forEach(button=>button.onclick=()=>{
      const matches=learner.model.findMatches(`__BLANK_${button.dataset.gap}__`,false,false,true,null,false);
      if(matches.length){learner.view.setSelection(matches[0].range);learner.view.revealRangeInCenter(matches[0].range);learner.view.focus();}
      else toast('这个标记已不在代码中；运行验证来检查实现，或重置骨架重新填写。');
    });
  }
  ready=true;renderProgress();setBusy();$('.lesson').dataset.ready='true';
  if(source!==persisted){storage.set(pendingKey,source);persist();}
  function emergencySave(){
    storage.set(pendingKey,source);
    if(source===persisted || !navigator.sendBeacon)return;
    const body=new Blob([JSON.stringify({stage,source})],{type:'application/json'});
    navigator.sendBeacon(`/api${endpoint}/draft`,body);
  }
  const onVisibility=()=>{if(document.hidden){emergencySave();persist();}};
  const onPageHide=()=>emergencySave();
  document.addEventListener('visibilitychange',onVisibility);window.addEventListener('pagehide',onPageHide);
  periodicSaveTimer=setInterval(()=>{if(source!==persisted)persist();},3000);
  checkpointTimer=setInterval(()=>{if(source && source===persisted && source!==lastCheckpointSource)checkpoint(false,'auto');},60000);
  return {verify,save:()=>checkpoint(true,'manual'),dirty:()=>source!==persisted,
    async dispose(){if(disposed)return;disposed=true;learner?.view.updateOptions({readOnly:true});clearTimeout(saveTimer);clearInterval(periodicSaveTimer);clearInterval(checkpointTimer);document.removeEventListener('visibilitychange',onVisibility);window.removeEventListener('pagehide',onPageHide);await persist();hovers?.dispose();reference?.dispose();learner?.dispose();},
  };
}

async function hash(source){const bytes=new TextEncoder().encode(source);const digest=await crypto.subtle.digest('SHA-256',bytes);return [...new Uint8Array(digest)].map(value=>value.toString(16).padStart(2,'0')).join('');}
async function route(){
  const token=++epoch,previous=active;active=null;await previous?.dispose();if(token!==epoch)return;
  closeMenu();window.scrollTo(0,0);
  main.innerHTML='<div class="loading"><span class="spinner"></span> 正在载入这个学习阶段…</div>';
  try{
    const catalogue=await api('/course');if(token!==epoch)return;courseData=catalogue.lessons;
    const parts=location.hash.replace(/^#\//,'').split('/');
    const chosen=courseData.find(item=>item.id===parts[0] || number(item)===parts[0]) || courseData.find(item=>item.id===catalogue.next_lesson);
    const stage=parts[1] in STAGES?parts[1]:'follow';history.replaceState(null,'',lessonLink(chosen,stage));renderCourse(chosen.id);
    const data=await api(`/course/lesson/${chosen.id}?stage=${stage}`);if(token!==epoch)return;
    const controller=await mount(data,token);if(token!==epoch){await controller.dispose();return;}active=controller;
  }catch(error){if(token!==epoch)return;main.innerHTML=`<div class="page-error">${icon('alert')}<h1>暂时无法打开这个阶段</h1><p>${e(error.message)}</p><div><button id="retry-lesson" class="button primary">重试</button><a class="button secondary" href="#/001/follow">回到基础第一课</a></div></div>`;$('#retry-lesson').onclick=route;}
}
$('#about-lesson').onclick=()=>dialog({title:'从基础到训练，按顺序连起来',symbol:'route',confirm:'回到课程',cancel:'',body:'<p>001 从 Multi-head Attention 数学开始，随后实现分头、mask、norm、RoPE、MLP、decoder 与 LM 输出，再比较 Qwen3、Gemma、MoE 和 DeepSeek MLA。</p><p>模型基础之后才进入 loss、数据组批、学习率与 Pretrain 训练步；接着是 SFT、DPO 和生成采样。</p><p>主线独立复现完成后，训练实验会使用已验证的用户代码，真正执行参数更新、SFT 监督 mask、偏好优化、KV-cache 生成及 checkpoint 重载。</p><p>这是小型 CPU 全流程教学，不是声称已经覆盖大规模数据、分布式训练、完整 RLHF 或训练出高质量大模型。</p>'});
const narrow=matchMedia('(max-width:1000px)');
function closeMenu(){document.body.classList.remove('course-menu-open');$('#course-overlay').hidden=true;main.inert=false;$('#course-map').inert=narrow.matches;}
$('#course-menu').onclick=()=>{if(!narrow.matches){document.body.classList.toggle('course-collapsed');return;}const open=document.body.classList.toggle('course-menu-open');$('#course-overlay').hidden=!open;main.inert=open;$('#course-map').inert=!open;};
$('#course-overlay').onclick=closeMenu;narrow.addEventListener('change',closeMenu);
$('.skip-link').onclick=event=>{event.preventDefault();main.focus();};
document.addEventListener('keydown',event=>{
  if(event.key==='Escape')closeMenu();
  if(document.body.classList.contains('course-menu-open'))return;
  if(event.defaultPrevented || $('#app-dialog').open || !active)return;
  if((event.ctrlKey || event.metaKey) && event.key==='Enter'){event.preventDefault();active.verify();}
  if((event.ctrlKey || event.metaKey) && event.key.toLowerCase()==='s'){event.preventDefault();active.save();}
});
window.addEventListener('hashchange',route);
window.addEventListener('beforeunload',event=>{if(active?.dirty()){event.preventDefault();event.returnValue='';}});
route();
