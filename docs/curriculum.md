# 001–022：从 MHA 到训练与生成

每课依次完成跟敲、核心填空、独立复现。数字表示学习顺序，不是模型热度排行。新用户必须先完成基础，不能直接进入 Post-train。

| 编号 | 课程 | 上游来源 |
|---|---|---|
| 001 | MHA：QKᵀ、缩放、mask、softmax、V | DistilBERT `eager_attention_forward` |
| 002 | MHA：投影、分头、合头 | DistilBERT `DistilBertSelfAttention.forward` |
| 003 | 因果 mask、缓存前缀、滑动窗口 | `AttentionMaskConverter._make_causal_mask` |
| 004 | RMSNorm 与数值精度 | `LlamaRMSNorm` |
| 005 | RoPE 与 Q/K 位置旋转 | Llama `apply_rotary_pos_emb` |
| 006 | SwiGLU 门控前馈 | `LlamaMLP` |
| 007 | GQA、RoPE 与 KV cache 的完整 Attention | `LlamaAttention`（已验收的样板） |
| 008 | decoder：残差、norm、Attention、MLP | `LlamaDecoderLayer` |
| 009 | LM head、logits 与可选监督 loss | `LlamaForCausalLM.forward` |
| 010 | Qwen3 的逐头 QK-Norm | `Qwen3Attention` |
| 011 | Gemma 的 logits softcap | Gemma2 `eager_attention_forward` |
| 012 | 专家路由与 Top-K 重归一化 | `Qwen3MoeTopKRouter` |
| 013 | token 分发、专家执行与加权聚合 | `MixtralExperts.forward` |
| 014 | 完整 MoE block | `MixtralSparseMoeBlock` |
| 015 | 低秩 Q/KV、解耦位置分量的 MLA | `DeepseekV3Attention.forward` |
| 016 | 下一 token loss、shift、ignore_index | `ForCausalLMLoss` |
| 017 | Pretrain 数据组批与 labels | Transformers `DataCollatorForLanguageModeling.torch_call` |
| 018 | warmup 与 cosine 学习率倍率 | `_get_cosine_schedule_with_warmup_lr_lambda` |
| 019 | Pretrain 训练步、梯度累积与 backward | `Trainer.training_step` |
| 020 | SFT completion/assistant mask | TRL `DataCollatorForLanguageModeling.torch_call` |
| 021 | policy/reference 的 DPO 偏好目标 | TRL `DPOTrainer._compute_loss` |
| 022 | Top-p 生成采样分布 | `TopPLogitsWarper.__call__` |

模型和主训练源码固定在 Transformers 5.1.0；SFT/DPO 固定在 TRL 1.1.0。详细模块、文件、行范围与源码指纹见各课 `provenance.json`。

## 训练联系不是一句介绍

课程实验将用户已通过验证的独立实现串起来：

- 001 数学 Attention 经明确的 GQA 头对齐 adapter，进入 007 用户版 LlamaAttention。
- 003–009 的 mask、norm、RoPE、MLP、Attention、decoder、LM 输出进入小型 Llama。
- 016–019 的损失、数据、学习率和训练步用于真实参数更新。
- 020 的监督 mask 决定 SFT 哪些 token 产生 loss。
- 021 的 DPO 实现作用于训练后的 policy 和冻结 reference，并实际 backward/update。
- 022 处理生成 logits，使用 cache 逐 token 生成。
- 013/014 的用户版专家和 MoE block 另接入真实小型 Mixtral，验证训练梯度与更新。

Qwen、Gemma、DeepSeek 等是架构比较分支：分别运行、分别验证，不把所有不兼容结构硬拼成一份模型。实验只用已验证的 `passed_source`，不会使用之后又被编辑坏的草稿。

## 边界

采用微型随机初始化模型和本地教学样本，目的是看懂并跑通软件链路，不代表完成大规模预训练。Tokenizer、优化器、模型未作为本课目标的外围代码仍由真实库与标注清楚的集成工具提供。

未覆盖完整 LoRA/PEFT、分布式/FSDP、多模态、RL rollout 或所有上游后端/损失变体。DPO 保留上游完整函数，当前运行场景集中在 sigmoid、IPO 混合和 JS/WPO；未执行的分支不会伪造运行值。
