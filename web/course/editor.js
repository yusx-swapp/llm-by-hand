// Native Monaco editors; no component framework or editor state abstraction.
let ready;
export function loadEditor() {
  if (!ready) ready=new Promise((resolve,reject)=>{
    if (!window.require) { reject(new Error('本地 Monaco 未加载，请检查 /static/vendor/monaco。')); return; }
    window.require.config({paths:{vs:'/static/vendor/monaco/vs'}});
    window.MonacoEnvironment={baseUrl:'/static/vendor/monaco/'};
    window.require(['vs/editor/editor.main'],()=>{
      const monaco=window.monaco;
      monaco.editor.defineTheme('source-classroom',{base:'vs',inherit:true,rules:[
        {token:'comment',foreground:'88969A',fontStyle:'italic'},
        {token:'keyword',foreground:'785B9F'},{token:'string',foreground:'3C7B66'},{token:'number',foreground:'A17836'},
      ],colors:{'editor.background':'#FFFFFF','editor.foreground':'#26343B',
        'editorLineNumber.foreground':'#B4BDC1','editorLineNumber.activeForeground':'#71858D',
        'editor.selectionBackground':'#DDE9FD','editor.lineHighlightBackground':'#F8FAFE',
        'editorIndentGuide.background1':'#EFF2F5','editorHoverWidget.background':'#FFFFFF',
        'editorHoverWidget.border':'#CAD7EA','editorHoverWidget.foreground':'#2F3D4E',
        'editorHoverWidget.statusBarBackground':'#F3F6FC'}});
      resolve(monaco);
    },reject);
  });
  return ready;
}

export function createCodeEditor(monaco,host,source,{readOnly=false,id,onChange=()=>{},onRun=()=>{},onCursor=()=>{},explanations=false}={}) {
  host.replaceChildren();
  const model=monaco.editor.createModel(source,'python',monaco.Uri.parse(`inmemory://classroom/${id}.py`));
  const view=monaco.editor.create(host,{model,theme:'source-classroom',readOnly,domReadOnly:readOnly,
    ariaLabel:readOnly ? '上游参考源码' : '你的 Python 实现',
    automaticLayout:true,minimap:{enabled:false},fontFamily:'"Cascadia Code", "SFMono-Regular", Consolas, monospace',
    fontSize:13,lineHeight:24,tabSize:4,insertSpaces:true,detectIndentation:false,
    padding:{top:18,bottom:30},lineNumbersMinChars:3,folding:true,glyphMargin:false,
    wordWrap:'on',wrappingIndent:'indent',fixedOverflowWidgets:true,
    scrollBeyondLastLine:false,overviewRulerLanes:0,renderLineHighlight:readOnly?'none':'line',
    quickSuggestions:false,suggestOnTriggerCharacters:false,wordBasedSuggestions:'off',
    snippetSuggestions:'none',parameterHints:{enabled:false},inlineSuggest:{enabled:false},
    hover:{enabled:explanations,delay:240,sticky:true,above:true},
    scrollbar:{verticalScrollbarSize:7,horizontalScrollbarSize:7},
  });
  const activeLine=view.createDecorationsCollection();
  const gaps=view.createDecorationsCollection();
  const subscriptions=[view.onDidChangeModelContent(()=>onChange(model.getValue())),
    view.onDidChangeCursorPosition(event=>onCursor(event.position))];
  if (!readOnly) view.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter,onRun);
  if (explanations) subscriptions.push(view.onMouseDown(event=>{
    if (event.target.position && matchMedia('(pointer:coarse)').matches) {
      view.setPosition(event.target.position); view.trigger('lesson','editor.action.showHover',{});
    }
  }));
  return {view,model,
    jump(line,column=1){view.setPosition({lineNumber:line,column});view.revealLineInCenter(line);view.focus();},
    explain(){view.focus();view.trigger('lesson','editor.action.showHover',{});},
    highlight(line){activeLine.set(line?[{range:new monaco.Range(line,1,line,1),options:{isWholeLine:true,className:'source-active-line'}}]:[]);},
    markGaps(){
      const matches=model.findMatches('__BLANK_[0-9]+__',false,true,true,null,false);
      gaps.set(matches.map(match=>({range:match.range,options:{inlineClassName:'code-blank',hoverMessage:{value:'填写这部分核心实现。目标见左侧；这里不会弹出答案。'}}})));
      return matches.map(match=>Number(match.matches?.[0]?.match(/[0-9]+/)?.[0] || model.getValueInRange(match.range).match(/[0-9]+/)[0]));
    },
    error(line,message){monaco.editor.setModelMarkers(model,'lesson',line?[{startLineNumber:line,endLineNumber:line,startColumn:1,endColumn:Math.max(2,model.getLineMaxColumn(Math.min(line,model.getLineCount()))),message,severity:monaco.MarkerSeverity.Error}]:[]);},
    dispose(){subscriptions.forEach(item=>item.dispose());view.dispose();model.dispose();host.replaceChildren();},
  };
}
