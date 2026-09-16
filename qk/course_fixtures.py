"""Small real CPU cases. Models/data here are harness context, not claimed upstream lessons."""
from __future__ import annotations
from dataclasses import dataclass,field
from collections import defaultdict
from contextlib import nullcontext
import copy
import tempfile
from types import SimpleNamespace
import torch
from transformers import LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast,TrainingArguments,Trainer
from transformers.models.llama import modeling_llama as llama
from transformers import DynamicCache
from tokenizers import Tokenizer,models,pre_tokenizers

def tiny_config(**changes):
    values=dict(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,
                num_key_value_heads=2,head_dim=8,max_position_embeddings=64,attention_dropout=0.0)
    values.update(changes); cfg=LlamaConfig(**values);cfg._attn_implementation='eager';return cfg

def tokenizer():
    words=['[PAD]','[UNK]','[EOS]','the','model','learns','attention','from','tokens','a','small','example','hello','world','good','answer','bad','question','user','assistant']
    backend=Tokenizer(models.WordLevel({word:i for i,word in enumerate(words)},unk_token='[UNK]'))
    backend.pre_tokenizer=pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',unk_token='[UNK]',eos_token='[EOS]')

def attention_inputs(cfg,length=4,past=0,batch=2):
    x=torch.randn(batch,length,cfg.hidden_size)
    ids=torch.arange(past,past+length)[None,:].expand(batch,-1)
    cos,sin=llama.LlamaRotaryEmbedding(cfg)(x,ids)
    allowed=torch.arange(past+length)[None,:]<=torch.arange(past,past+length)[:,None]
    mask=torch.zeros(batch,1,length,past+length).masked_fill(~allowed[None,None],torch.finfo(torch.float32).min)
    return dict(hidden_states=x,position_embeddings=(cos,sin),attention_mask=mask,cache_position=torch.arange(past,past+length))

@dataclass
class Scenario:
    call: object
    modules: dict=field(default_factory=dict)
    tensors: dict=field(default_factory=dict)
    after: object=None
    backward: bool=True
    dimensions: dict=field(default_factory=dict)
    title: str='基础输入'
    cleanup: object=None

def variants(key):
    if key=='dpo_loss':return ['sigmoid','ipo_mix','weighted_js']
    if key=='pretrain_step':return ['one_step','accumulate']
    if key=='sft_collator':return ['padded','padding_free','assistant_mask']
    if key=='causal_mask':return ['causal','cached','sliding']
    if key in {'llama_attention','qwen_attention','mla_attention'}:return ['prefill','decode','alternate']
    return ['basic','alternate']

def make_case(key,target,variant='basic',*,project_model=None,project_batch=None):
    torch.manual_seed(120)
    alt=variant not in {'basic','prefill','causal','one_step','padded','sigmoid'}
    dims=dict(B=2,T=4,D=32,H=4,d=8)
    if key in {'mha_core','gemma_attention'}:
        kv=2 if key=='gemma_attention' else 4
        module=torch.nn.Module();module.head_dim=8;module.num_key_value_groups=4//kv;module.train(alt)
        q=torch.randn(2,4,4,8,requires_grad=True);k=torch.randn(2,kv,6 if alt else 4,8,requires_grad=True);v=torch.randn_like(k,requires_grad=True)
        mask=torch.zeros(2,1,4,k.shape[-2]);mask[:,:,:,-1]=torch.finfo(torch.float32).min
        kwargs=dict(scaling=None if not alt else .3,dropout=.2 if alt else 0)
        if key=='gemma_attention':kwargs['softcap']=1.5 if not alt else None
        title=('mask / softcap' if key=='gemma_attention' else '标准 MHA / additive mask') if not alt else '不同长度与 dropout'
        return Scenario(lambda:target(module,q,k,v,mask if not alt else None,**kwargs),tensors=dict(query=q,key=k,value=v),dimensions={**dims,'S':k.shape[-2],'KV':kv},title=title)
    if key=='mha_projection':
        from transformers import DistilBertConfig
        cfg=DistilBertConfig(dim=32,n_heads=4,hidden_dim=64,n_layers=1,attention_dropout=.2 if alt else 0);cfg._attn_implementation='eager'
        module=target(cfg).train(alt)
        x=torch.randn(2,6 if alt else 4,32,requires_grad=True)
        mask=torch.zeros(2,1,x.shape[1],x.shape[1]);mask[:,:,:,-1]=torch.finfo(torch.float32).min
        return Scenario(lambda:module(x,mask,output_attentions=True),dict(module=module),dict(hidden_states=x),dimensions={**dims,'T':x.shape[1],'S':x.shape[1]},title='标准 MHA' if not alt else '不同 token 长度 / dropout')
    if key=='causal_mask':
        past=3 if variant!='causal' else 0;window=2 if variant=='sliding' else None
        return Scenario(lambda:target._make_causal_mask(torch.Size([2,4]),torch.float32,torch.device('cpu'),past,window),backward=False,
                        dimensions=dict(B=2,T=4,S=4+past),title={'causal':'只看当前及过去','cached':'带 3 个历史 token','sliding':'滑动窗口 2'}[variant])
    if key in {'rms_norm','swiglu','moe_router','moe_experts','moe_block'}:
        cfg=tiny_config(hidden_size=24 if alt else 32,intermediate_size=48 if alt else 64)
        if key=='rms_norm':module=target(cfg.hidden_size,eps=1e-5 if alt else 1e-6)
        elif key=='swiglu':module=target(cfg)
        elif key=='moe_router':
            from transformers import Qwen3MoeConfig
            cfg=Qwen3MoeConfig(hidden_size=24 if alt else 32,intermediate_size=64,moe_intermediate_size=48,num_experts=4,num_experts_per_tok=2,norm_topk_prob=not alt,num_hidden_layers=1)
            module=target(cfg)
        else:
            from transformers import MixtralConfig
            cfg=MixtralConfig(hidden_size=24 if alt else 32,intermediate_size=48,num_local_experts=4,num_experts_per_tok=2,num_hidden_layers=1,num_attention_heads=4,router_jitter_noise=.1 if alt else 0)
            module=target(cfg)
        if key.startswith('moe_'):
            with torch.no_grad():
                for parameter in module.parameters():parameter.normal_(mean=0.0,std=0.1)
        module.train(alt)
        x=torch.randn(2,4,cfg.hidden_size,requires_grad=True)
        if key=='moe_router':x=x.reshape(8,cfg.hidden_size).detach().requires_grad_(True)
        if key=='moe_experts':
            x=x.reshape(8,cfg.hidden_size).detach().requires_grad_(True)
            indices=torch.tensor([[0,1],[1,2],[0,2],[2,3],[1,3],[0,3],[0,1],[2,3]])
            weights=torch.softmax(torch.randn(8,2),-1).requires_grad_(True)
            call=lambda:module(x,indices,weights)
        else:call=lambda:module(x*1.0)
        tensors=dict(hidden_states=x)
        if key=='moe_experts':tensors['top_k_weights']=weights
        return Scenario(call,dict(module=module),tensors,dimensions=dict(B=2,T=4,D=cfg.hidden_size,E=4,K=2),title='基础形状' if not alt else '不同维度 / 训练态')
    if key=='rope':
        q=torch.randn(2,4,4,8,requires_grad=True);k=torch.randn(2,2 if not alt else 4,4,8,requires_grad=True)
        angles=torch.randn(2,4,8);cos,sin=angles.cos(),angles.sin()
        return Scenario(lambda:target(q,k,cos,sin),tensors=dict(q=q,k=k),dimensions={**dims,'KV':k.shape[1]},title='GQA Q/K' if not alt else '相同头数 Q/K')
    if key in {'llama_attention','qwen_attention','decoder_layer','mla_attention'}:
        cfg=tiny_config()
        if key=='qwen_attention':
            from transformers import Qwen3Config
            cfg=Qwen3Config(vocab_size=32,hidden_size=32,head_dim=8,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,attention_dropout=.2 if variant=='alternate' else 0)
        if key=='mla_attention':
            from transformers import DeepseekV3Config
            cfg=DeepseekV3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=4,
                q_lora_rank=None if variant=='alternate' else 8,kv_lora_rank=8,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=8,n_routed_experts=None,first_k_dense_replace=1,max_position_embeddings=64,rope_interleave=variant=='alternate')
        cfg._attn_implementation='eager'
        module=target(cfg,layer_idx=0).train(variant=='alternate')
        def args(length=4,past=0):
            if key!='mla_attention':return attention_inputs(cfg,length,past)
            from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding
            x=torch.randn(2,length,32);pos=torch.arange(past,past+length)[None,:].expand(2,-1)
            cos,sin=DeepseekV3RotaryEmbedding(cfg)(x,pos)
            allowed=torch.arange(past+length)[None,:]<=torch.arange(past,past+length)[:,None]
            mask=torch.zeros(2,1,length,past+length).masked_fill(~allowed[None,None],torch.finfo(torch.float32).min)
            return dict(hidden_states=x,position_embeddings=(cos,sin),attention_mask=mask,cache_position=torch.arange(past,past+length))
        inputs=args(1,4) if variant=='decode' else args();inputs['hidden_states'].requires_grad_(True)
        cache=DynamicCache(config=cfg) if variant=='decode' else None
        warmup=args() if cache is not None else None
        def call():
            if warmup is not None:module(**warmup,past_key_values=cache)
            return module(**inputs,past_key_values=cache)
        tensors=dict(hidden_states=inputs['hidden_states'])
        if warmup is not None:tensors['cache_prefix']=warmup['hidden_states']
        return Scenario(call,dict(module=module),tensors,after=lambda:dict(cache_length=cache.get_seq_length() if cache is not None else 0),
                        dimensions=dict(B=2,T=1 if cache is not None else 4,S=5 if cache is not None else 4,D=32,H=4,d=8,KV=2),title={'prefill':'Prefill','decode':'先缓存 4 token，再 decode 1','alternate':'不同投影 / 训练配置'}.get(variant,'Decoder 输出'))
    if key=='causal_lm':
        cfg=tiny_config();module=target(cfg).eval();ids=torch.randint(3,32,(2,5))
        return Scenario(lambda:module(input_ids=ids,labels=ids if not alt else None,use_cache=False,logits_to_keep=0 if not alt else 2),dict(model=module),dimensions=dict(B=2,T=5,V=32),title='完整 logits 与 loss' if not alt else '只保留最后 2 个 logits')
    if key=='lm_loss':
        logits=torch.randn(2,5,32,requires_grad=True);labels=torch.randint(0,32,(2,5));labels[0,:2]=-100
        return Scenario(lambda:target(logits,labels,32,num_items_in_batch=torch.tensor(5) if alt else None),tensors=dict(logits=logits),dimensions=dict(B=2,T=5,V=32),title='shift / ignore_index' if not alt else '有效 token 归一化')
    if key=='lm_collator':
        tok=tokenizer();module=target(tokenizer=tok,mlm=False,pad_to_multiple_of=8 if alt else None)
        rows=[{'input_ids':[3,4,5,6,2]},{'input_ids':[3,6,2]}] if not alt else [[3,4,2],[10,11,5,6,7,2]]
        return Scenario(lambda:module.torch_call(copy.deepcopy(rows)),backward=False,dimensions=dict(B=2,T=8 if alt else 5),title='字典输入 / padding label' if not alt else '列表输入 / 对齐 8')
    if key=='lr_schedule':
        steps=[0,1,2,3,5,8,10,12];cycles=.5 if not alt else 1.0
        return Scenario(lambda:torch.tensor([target(step,num_warmup_steps=2,num_training_steps=10,num_cycles=cycles) for step in steps]),backward=False,dimensions=dict(warmup=2,total_steps=10),title='半周期 cosine' if not alt else '完整 cosine 周期')
    if key=='pretrain_step':
        cfg=tiny_config();model=LlamaForCausalLM(cfg);tmp=tempfile.TemporaryDirectory(prefix='qk-trainer-')
        args=TrainingArguments(output_dir=tmp.name,use_cpu=True,bf16=False,fp16=False,report_to='none',disable_tqdm=True,gradient_accumulation_steps=2 if alt else 1)
        trainer=target(model=model,args=args);trainer.current_gradient_accumulation_steps=args.gradient_accumulation_steps
        inputs=dict(input_ids=torch.randint(3,32,(2,5)),labels=torch.randint(3,32,(2,5)))
        return Scenario(lambda:trainer.training_step(model,copy.deepcopy(inputs)),dict(model=model),backward=False,
                        dimensions=dict(B=2,T=5,grad_accum=args.gradient_accumulation_steps),title='单步 backward' if not alt else '梯度累积缩放',cleanup=tmp.cleanup)
    if key=='sft_collator':
        module=target(pad_token_id=0,completion_only_loss=True,padding_free=variant=='padding_free',max_length=6 if variant=='assistant_mask' else None)
        rows=[dict(input_ids=[3,4,5,6,2],completion_mask=[0,0,1,1,1]),dict(input_ids=[9,10,11,2],completion_mask=[0,1,1,1])]
        if variant=='padding_free':
            rows[0]['seq_lengths']=[2,3];rows[1]['seq_lengths']=[4]
        if variant=='assistant_mask':
            rows[0]['assistant_masks']=[0,0,0,1,1];rows[1]['assistant_masks']=[0,0,1,1]
        return Scenario(lambda:module.torch_call(copy.deepcopy(rows)),backward=False,dimensions=dict(B=2,T=5),title={'padded':'仅 completion labels','padding_free':'拼接与 position_ids 重置','assistant_mask':'叠加 assistant mask'}[variant])
    if key=='dpo_loss':
        from accelerate import Accelerator
        from trl.trainer.utils import selective_log_softmax
        policy=project_model if project_model is not None else LlamaForCausalLM(tiny_config()).train()
        reference=copy.deepcopy(policy).eval()
        for p in reference.parameters():p.requires_grad_(False)
        if project_model is None:
            with torch.no_grad():policy.lm_head.weight.add_(.03*torch.randn_like(policy.lm_head.weight))
        trainer=object.__new__(target)
        trainer.model=policy;trainer.ref_model=reference;trainer.accelerator=Accelerator(cpu=True)
        trainer.args=SimpleNamespace(gradient_checkpointing_kwargs={'use_reentrant':False},discopop_tau=.1)
        trainer.ld_alpha=None;trainer.precompute_ref_logps=variant=='ipo_mix'
        trainer.f_divergence_type='js_divergence' if variant=='weighted_js' else 'reverse_kl'
        trainer.loss_types=['sigmoid','ipo'] if variant=='ipo_mix' else ['sigmoid'];trainer.loss_weights=[.7,.3] if variant=='ipo_mix' else [1.0]
        trainer.beta=.2 if alt else .1;trainer.label_smoothing=.1;trainer.use_weighting=variant=='weighted_js'
        trainer._metrics=defaultdict(lambda:defaultdict(list));trainer._total_train_tokens=0
        inputs=dict(input_ids=torch.randint(3,32,(4,6)),attention_mask=torch.ones(4,6,dtype=torch.long),completion_mask=torch.tensor([[0,0,1,1,1,1],[0,0,0,1,1,1],[0,0,1,1,1,1],[0,0,1,1,1,0]]))
        if project_batch is not None:inputs=project_batch
        def call():
            if trainer.precompute_ref_logps:
                with torch.no_grad():
                    out=reference(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],use_cache=False)
                    per=selective_log_softmax(out.logits[:,:-1],inputs['input_ids'][:,1:]);per[inputs['completion_mask'][:,1:]==0]=0
                    inputs['ref_chosen_logps'],inputs['ref_rejected_logps']=per.sum(1).chunk(2)
            return trainer._compute_loss(policy,copy.deepcopy(inputs),return_outputs=False)
        return Scenario(call,dict(policy=policy,reference=reference),after=lambda:dict(metrics=dict(trainer._metrics['train'])),dimensions=dict(B=2,T=6,V=32),title={'sigmoid':'标准 sigmoid DPO','ipo_mix':'预计算 reference / IPO 混合','weighted_js':'JS 散度 / WPO 加权'}[variant])
    if key=='top_p':
        module=target(top_p=.6 if alt else .9,min_tokens_to_keep=2 if alt else 1)
        scores=torch.randn(2,12);ids=torch.ones(2,4,dtype=torch.long)
        return Scenario(lambda:module(ids,scores.clone()),backward=False,dimensions=dict(B=2,V=12),title='累计概率 0.9' if not alt else '最少保留 2 token')
    raise KeyError(key)

def transfer_state(expected,actual):
    if set(expected.modules)!=set(actual.modules):raise AssertionError('运行环境的模型集合不一致。')
    for name,module in actual.modules.items():module.load_state_dict(expected.modules[name].state_dict(),strict=True)
    if set(expected.tensors)!=set(actual.tensors):raise AssertionError('运行输入集合不一致。')
    with torch.no_grad():
        for name,tensor in actual.tensors.items():tensor.copy_(expected.tensors[name])

def first_loss(value):
    if isinstance(value,torch.Tensor) and value.is_floating_point() and value.requires_grad:return value.float().square().mean()
    if isinstance(value,dict):
        if isinstance(value.get('loss'),torch.Tensor):return value['loss']
        for item in value.values():
            loss=first_loss(item)
            if loss is not None:return loss
    if isinstance(value,(list,tuple)):
        for item in value:
            loss=first_loss(item)
            if loss is not None:return loss
    return None

def execute(scenario):
    torch.manual_seed(808)
    output=scenario.call()
    if scenario.backward:
        loss=first_loss(output)
        if loss is not None:loss.backward()
    grads={f'{name}.{key}':p.grad.detach().clone() if p.grad is not None else None for name,module in scenario.modules.items() for key,p in module.named_parameters()}
    grads.update({name:t.grad.detach().clone() if t.grad is not None else None for name,t in scenario.tensors.items() if t.is_leaf})
    return dict(output=output,gradients=grads,after=scenario.after() if scenario.after else {})
