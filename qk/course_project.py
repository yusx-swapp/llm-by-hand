"""Project-owned integration glue, NOT an upstream reference lesson.

Use verified learner implementations to run a tiny CPU pretrain → SFT → DPO → generation
experiment. No pretrained weights, external datasets or model-quality claims.
"""
from __future__ import annotations
import argparse,copy,json,os,sys,tempfile,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch
from transformers import TrainingArguments
from transformers.models.llama import modeling_llama as hf
from qk import curriculum as c
from qk import course_fixtures as f

torch.set_num_threads(1)

def run(sources,output_dir):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    for key,source in sources.items():
        c.entry(key)
        (output_dir/(key+'.py')).write_text(source,encoding='utf-8')
    compiled={};calls={'mha_core':0}
    def load(key,overrides=None):
        if key not in sources:raise ValueError(f'缺少已验证的独立复现：{key}')
        path=output_dir/(key+'.py');path.write_text(sources[key],encoding='utf-8')
        value,_=c.load_object(key,sources[key],path,overrides);compiled[key]=value;return value
    mha=load('mha_core');mask_class=load('causal_mask');norm=load('rms_norm');rope=load('rope');mlp=load('swiglu')
    def gqa_bridge(module,query,key,value,attention_mask,**kwargs):
        calls['mha_core']+=1
        key=hf.repeat_kv(key,module.num_key_value_groups);value=hf.repeat_kv(value,module.num_key_value_groups)
        return mha(module,query,key,value,attention_mask,**kwargs)
    attention=load('llama_attention',{'apply_rotary_pos_emb':rope,'eager_attention_forward':gqa_bridge})
    decoder=load('decoder_layer',{'LlamaAttention':attention,'LlamaMLP':mlp,'LlamaRMSNorm':norm})
    lm=load('causal_lm');loss_fn=load('lm_loss');collator_class=load('lm_collator');schedule=load('lr_schedule')
    trainer_class=load('pretrain_step');sft_class=load('sft_collator');dpo_class=load('dpo_loss');top_p=load('top_p')
    experts_class=load('moe_experts');moe_block=load('moe_block',{'MixtralExperts':experts_class})
    tok=f.tokenizer();cfg=f.tiny_config(vocab_size=len(tok),pad_token_id=tok.pad_token_id)
    class StudentModel(lm):
        def forward(self,input_ids=None,attention_mask=None,**kwargs):
            if input_ids is not None and (attention_mask is None or attention_mask.ndim==2):
                past=kwargs.get('past_key_values');past_length=past.get_seq_length() if past is not None else 0
                mask=mask_class._make_causal_mask(input_ids.shape,self.dtype,input_ids.device,past_length)
                if attention_mask is not None:mask=mask.masked_fill(attention_mask[:,None,None,:]==0,torch.finfo(self.dtype).min)
                attention_mask=mask
            return super().forward(input_ids=input_ids,attention_mask=attention_mask,**kwargs)
    def model_factory():
        previous_decoder,previous_norm=hf.LlamaDecoderLayer,hf.LlamaRMSNorm
        try:
            hf.LlamaDecoderLayer=decoder;hf.LlamaRMSNorm=norm
            model=StudentModel(copy.deepcopy(cfg));model.loss_function=loss_fn
            return model
        finally:hf.LlamaDecoderLayer=previous_decoder;hf.LlamaRMSNorm=previous_norm
    torch.manual_seed(902);model=model_factory()
    assert type(model.model.layers[0]) is decoder and type(model.model.layers[0].self_attn) is attention
    initial={name:p.detach().clone() for name,p in model.named_parameters()}
    rows=[tok(text,add_special_tokens=False) for text in ['the model learns attention from tokens [EOS]','a small model learns from tokens [EOS]']]
    batch=collator_class(tokenizer=tok,mlm=False)(rows)
    args=TrainingArguments(output_dir=str(output_dir/'trainer'),use_cpu=True,bf16=False,fp16=False,report_to='none',disable_tqdm=True,gradient_accumulation_steps=1)
    trainer=trainer_class(model=model,args=args);trainer.current_gradient_accumulation_steps=1
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    trainer.optimizer=optimizer
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:schedule(step,num_warmup_steps=1,num_training_steps=6,num_cycles=.5))
    pretrain=[]
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss=trainer.training_step(model,{k:v.clone() for k,v in batch.items()})
        optimizer.step();scheduler.step();pretrain.append(float(loss))
    changed=sum(not torch.equal(initial[name],p.detach()) for name,p in model.named_parameters())
    assert changed>0,'训练没有更新任何模型参数。'
    prompt=tok('user question assistant',add_special_tokens=False)['input_ids']
    responses=[tok(text,add_special_tokens=False)['input_ids'] for text in ['good answer [EOS]','the model learns [EOS]']]
    sft_rows=[dict(input_ids=prompt+answer,completion_mask=[0]*len(prompt)+[1]*len(answer)) for answer in responses]
    sft_batch=sft_class(pad_token_id=0,completion_only_loss=True).torch_call(sft_rows)
    assert (sft_batch['labels'][:,:len(prompt)]==-100).all(),'SFT prompt 没有从 loss 中排除。'
    sft=[]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss=trainer.training_step(model,{k:v.clone() for k,v in sft_batch.items()})
        optimizer.step();scheduler.step();sft.append(float(loss))
    chosen=prompt+tok('good answer [EOS]',add_special_tokens=False)['input_ids']
    rejected=prompt+tok('bad answer [EOS]',add_special_tokens=False)['input_ids']
    ids=torch.tensor([chosen,chosen,rejected,rejected]);completion=torch.tensor([[0]*len(prompt)+[1]*(len(chosen)-len(prompt))]*4)
    pref_batch=dict(input_ids=ids,attention_mask=torch.ones_like(ids),completion_mask=completion)
    optimizer.zero_grad(set_to_none=True);model.train()
    scenario=f.make_case('dpo_loss',dpo_class,'sigmoid',project_model=model,project_batch=pref_batch)
    dpo=scenario.call();dpo.backward()
    grad_norm=sum(float(p.grad.norm()) for p in model.parameters() if p.grad is not None)
    assert grad_norm>0,'偏好目标没有产生 policy 梯度。'
    optimizer.step()
    model.eval();generated=torch.tensor([tok('the model',add_special_tokens=False)['input_ids']]);cache=None;processor=top_p(top_p=.9,min_tokens_to_keep=2)
    with torch.no_grad():
        for _ in range(4):
            inputs=generated if cache is None else generated[:,-1:]
            output=model(input_ids=inputs,past_key_values=cache,use_cache=True)
            cache=output.past_key_values;logits=processor(generated,output.logits[:,-1])
            token=torch.multinomial(torch.softmax(logits,dim=-1),1);generated=torch.cat((generated,token),-1)
    checkpoint=output_dir/'checkpoint.pt';torch.save({'state_dict':model.state_dict(),'config':cfg.to_dict()},checkpoint)
    restored=model_factory().eval();restored.load_state_dict(torch.load(checkpoint,weights_only=True)['state_dict'])
    with torch.no_grad():torch.testing.assert_close(restored(input_ids=generated,use_cache=False).logits,model(input_ids=generated,use_cache=False).logits,rtol=1e-5,atol=1e-6)
    assert calls['mha_core']>0,'最底层的用户 MHA 实现没有被串联调用。'
    # Architectural branch: learner expert dispatch/aggregation inside a real Mixtral causal LM.
    from transformers import MixtralConfig
    from transformers.models.mixtral import modeling_mixtral as mix
    moe_cfg=MixtralConfig(vocab_size=len(tok),hidden_size=32,intermediate_size=48,num_hidden_layers=1,
                         num_attention_heads=4,num_key_value_heads=2,num_local_experts=4,num_experts_per_tok=2,max_position_embeddings=64)
    moe_cfg._attn_implementation='eager'
    old_block,old_attention=mix.MixtralSparseMoeBlock,mix.eager_attention_forward
    try:
        mix.MixtralSparseMoeBlock=moe_block;mix.eager_attention_forward=gqa_bridge
        torch.manual_seed(77);moe=mix.MixtralForCausalLM(moe_cfg)
        assert any(type(module) is moe_block for module in moe.modules())
        moe.loss_function=loss_fn
        moe_args=TrainingArguments(output_dir=str(output_dir/'moe-trainer'),use_cpu=True,bf16=False,fp16=False,report_to='none',disable_tqdm=True)
        moe_trainer=trainer_class(model=moe,args=moe_args);moe_trainer.current_gradient_accumulation_steps=1
        moe_opt=torch.optim.AdamW(moe.parameters(),lr=.003);moe_trainer.optimizer=moe_opt
        moe_losses=[];expert_grad=0.0
        for _ in range(2):
            moe_opt.zero_grad(set_to_none=True)
            value=moe_trainer.training_step(moe,{k:v.clone() for k,v in batch.items()})
            expert_grad=sum(float(p.grad.norm()) for name,p in moe.named_parameters() if 'gate_up_proj' in name and p.grad is not None)
            moe_opt.step();moe_losses.append(float(value))
        assert expert_grad>0,'MoE 专家没有收到训练梯度。'
        torch.save({'state_dict':moe.state_dict(),'config':moe_cfg.to_dict()},output_dir/'moe-checkpoint.pt')
    finally:mix.MixtralSparseMoeBlock=old_block;mix.eager_attention_forward=old_attention
    return dict(passed=True,scope='小型 CPU 教学实验，不代表大规模训练或模型质量',used_implementations=list(compiled),mha_calls=calls['mha_core'],
                parameter_tensors_updated=changed,pretrain_losses=pretrain,sft_losses=sft,prompt_labels_ignored=int((sft_batch['labels']==-100).sum()),
                dpo_loss=float(dpo.detach()),dpo_gradient_norm=grad_norm,generated_ids=generated.tolist(),generated_text=tok.decode(generated[0]),
                checkpoint_reloaded=True,moe_losses=moe_losses,moe_expert_gradient_norm=expert_grad,
                exported_implementations=len(sources),artifacts=['checkpoint.pt','moe-checkpoint.pt']+[key+'.py' for key in sources])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sources',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--artifacts',type=Path)
    args=parser.parse_args()
    sources=json.loads(args.sources.read_text(encoding='utf-8')) if args.sources else {spec['id']:c.reference(spec['id']) for spec in c.catalogue()}
    try:
        if args.artifacts:result=run(sources,args.artifacts)
        else:
            with tempfile.TemporaryDirectory(prefix='qk-project-output-') as tmp:result=run(sources,tmp)
    except Exception as exc:traceback.print_exc();result=dict(passed=False,error=f'{type(exc).__name__}: {exc}')
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
