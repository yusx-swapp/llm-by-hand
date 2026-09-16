"""Isolated, trusted-code runner for the single LlamaAttention lesson.

The upstream class is not edited. Its imports/helpers are supplied by its real module.
Reference observations are recorded once; verification always executes the learner's class.
"""
from __future__ import annotations

import argparse
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
import types

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM
from transformers.models.llama import modeling_llama as hf

torch.set_num_threads(1)
LESSON = Path(__file__).resolve().parents[1] / "lessons" / "llama_attention"
CONFIG_FIELDS = ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
                 "attention_dropout", "attention_bias", "_attn_implementation")
SELF_FIELDS = ("head_dim", "num_key_value_groups", "scaling", "attention_dropout", "is_causal",
               "layer_idx", "training", "q_proj", "k_proj", "v_proj", "o_proj")


def config(**changes):
    values = dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                  max_position_embeddings=64, attention_dropout=0.0, attention_bias=False)
    values.update(changes)
    value = LlamaConfig(**values)
    value._attn_implementation = "eager"
    return value


def load_class(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LlamaAttention"]
    if len(classes) != 1:
        raise ValueError("请定义一个 LlamaAttention 类；空草稿或只导入上游答案不算实现。")
    methods = {node.name for node in classes[0].body if isinstance(node, ast.FunctionDef)}
    if not {"__init__", "forward"} <= methods:
        raise ValueError("需要自己实现 __init__ 和 forward 两个方法。")
    if any(isinstance(node, ast.Name) and node.id.startswith("__BLANK_") for node in ast.walk(tree)):
        raise ValueError("还有 __BLANK_ 空缺未填写。先补全这些核心位置，再验证。")
    module = types.ModuleType("qk_learner_attention")
    module.__dict__.update(vars(hf))
    module.__dict__.update(__name__="qk_learner_attention", __file__=str(path))
    module.__dict__.pop("LlamaAttention", None)  # empty code must never fall back to the teacher's class
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    cls = module.__dict__.get("LlamaAttention")
    if not isinstance(cls, type) or not issubclass(cls, torch.nn.Module):
        raise ValueError("LlamaAttention 必须是 torch.nn.Module。")
    if cls.forward.__code__.co_filename != str(path):
        raise ValueError("forward 必须来自你提交的实现，而不是直接转用上游类。")
    return cls


def input_data(cfg, batch=2, length=5, start=0, *, mask=True, padded=False):
    x = torch.randn(batch, length, cfg.hidden_size)
    positions = torch.arange(start, start + length).unsqueeze(0).expand(batch, -1)
    cos, sin = hf.LlamaRotaryEmbedding(cfg)(x, positions)
    attention_mask = None
    if mask:
        allowed = torch.arange(start + length)[None, :] <= torch.arange(start, start + length)[:, None]
        attention_mask = torch.zeros(batch, 1, length, start + length)
        attention_mask.masked_fill_(~allowed[None, None], torch.finfo(torch.float32).min)
        if padded:
            attention_mask[0, :, :, -1] = torch.finfo(torch.float32).min
    return {"hidden_states": x, "position_embeddings": (cos, sin), "attention_mask": attention_mask,
            "cache_position": torch.arange(start, start + length)}


def describe(value, depth=0):
    if isinstance(value, torch.Tensor):
        flat = value.detach().reshape(-1)
        return {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", ""),
                "device": str(value.device), "requires_grad": value.requires_grad,
                "sample": flat[:4].float().tolist(), "numel": value.numel()}
    if isinstance(value, torch.nn.Linear):
        return {"kind": "linear", "in_features": value.in_features, "out_features": value.out_features,
                "weight": list(value.weight.shape), "bias": list(value.bias.shape) if value.bias is not None else None}
    if isinstance(value, DynamicCache):
        return {"kind": "cache", "length": value.get_seq_length()}
    if isinstance(value, LlamaConfig):
        return {"kind": "config", "value": {name: getattr(value, name, None) for name in CONFIG_FIELDS}}
    if isinstance(value, torch.dtype):
        return {"kind": "dtype", "value": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"kind": "scalar", "value": value}
    if isinstance(value, (tuple, list)) and depth < 2:
        return {"kind": "sequence", "value": [describe(item, depth + 1) for item in value[:6]]}
    if isinstance(value, dict) and depth < 2:
        return {"kind": "mapping", "value": {str(key): describe(item, depth + 1) for key, item in list(value.items())[:8]}}
    if callable(value):
        return {"kind": "callable", "value": getattr(value, "__name__", type(value).__name__)}
    return {"kind": "object", "value": type(value).__name__}


class Observation:
    """Source-line observations plus actual ATen view/transpose/linear operations."""
    def __init__(self, path):
        self.path = str(path)
        self.enabled = True
        self.observing = False
        self.pending = {}
        self.lines = {}
        self.context = {}

    def snapshot(self, frame):
        self.observing = True
        try:
            values = {name: describe(value) for name, value in frame.f_locals.items() if name != "self"}
            module = frame.f_locals.get("self")
            if module is not None:
                values["self"] = {"kind": "object", "value": "LlamaAttention"}
                for field in SELF_FIELDS:
                    if hasattr(module, field):
                        values[f"self.{field}"] = describe(getattr(module, field))
                cfg = getattr(module, "config", frame.f_locals.get("config"))
                if cfg:
                    values["self.config"] = describe(cfg)
                    for field in CONFIG_FIELDS:
                        values[f"config.{field}"] = describe(getattr(cfg, field, None))
                        values[f"self.config.{field}"] = values[f"config.{field}"]
            return values
        finally:
            self.observing = False

    def event(self, frame, event, arg):
        if frame.f_code.co_filename != self.path or frame.f_code.co_name not in {"__init__", "forward"}:
            return None
        if not self.enabled:
            return None
        if event not in {"line", "return"}:
            return self.event
        values = self.snapshot(frame)
        key = id(frame)
        previous = self.pending.get(key)
        if previous:
            line, before = previous
            observed = self.lines.setdefault(str(line), {"before": before, "method": frame.f_code.co_name, "ops": []})
            observed["after"] = values
        if event == "line":
            self.context.setdefault(frame.f_code.co_name, values)
            self.pending[key] = (frame.f_lineno, values)
        else:
            self.pending.pop(key, None)
        return self.event


class ShapeOperations(TorchDispatchMode):
    def __init__(self, observation):
        self.observation = observation

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        obs = self.observation
        name = func._schema.name.split("::")[-1]
        if not obs.enabled or obs.observing or name not in {"view", "_unsafe_view", "transpose", "addmm", "mm", "bmm", "cat", "_softmax"}:
            return result
        frame = sys._getframe(1)
        direct = frame.f_code.co_filename == obs.path
        while frame and frame.f_code.co_filename != obs.path:
            frame = frame.f_back
        if frame and frame.f_code.co_name == "forward":
            pending = obs.pending.get(id(frame))
            if pending:
                line, before = pending
                record = obs.lines.setdefault(str(line), {"before": before, "method": "forward", "ops": []})
                inputs = [list(value.shape) for value in args if isinstance(value, torch.Tensor)]
                output = list(result.shape) if isinstance(result, torch.Tensor) else None
                if output is not None and len(record["ops"]) < 24:
                    record["ops"].append({"op": name, "inputs": inputs, "output": output, "direct": direct})
        return result


def build_observations():
    path = LESSON / "reference.py"
    provenance = json.loads((LESSON / "provenance.json").read_text(encoding="utf-8"))
    assert transformers.__version__ == provenance["version"]
    cls = load_class(path)
    fixtures = {}
    for key, title in (("prefill", "GQA · 5 token 输入"), ("decode", "KV Cache · 第 6 个 token"), ("mha", "MHA · 不共享 KV 头")):
        torch.manual_seed(17)
        cfg = config(num_key_value_heads=4) if key == "mha" else config()
        observation = Observation(path)
        try:
            sys.settrace(observation.event)
            with ShapeOperations(observation):
                module = cls(cfg, layer_idx=0).eval()
                observation.enabled = False
                cache = DynamicCache(config=cfg) if key == "decode" else None
                if cache is not None:
                    module(**input_data(cfg), past_key_values=cache)
                args = input_data(cfg, length=1 if cache is not None else 5, start=5 if cache is not None else 0)
                args["past_key_values"] = cache
                observation.enabled = True
                outputs = module(**args)
        finally:
            sys.settrace(None)
        fixtures[key] = {"id": key, "title": title,
                         "dimensions": {"B": 2, "T": 1 if cache is not None else 5, "S": 6 if cache is not None else 5,
                                        "D": cfg.hidden_size, "H": cfg.num_attention_heads,
                                        "KV": cfg.num_key_value_heads, "d": cfg.head_dim},
                         "lines": observation.lines, "context": observation.context,
                         "output": describe(outputs), "cache_length": cache.get_seq_length() if cache else None}
    return {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "transformers": transformers.__version__, "torch": torch.__version__,
            "observed_at": time.time(), "fixtures": fixtures}


def assert_close(actual, expected, message=""):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=message or None)


def pair(cls, cfg, train=False):
    torch.manual_seed(23)
    reference = hf.LlamaAttention(copy.deepcopy(cfg), layer_idx=0)
    candidate = cls(copy.deepcopy(cfg), layer_idx=0)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    for field in ("head_dim", "num_key_value_groups", "scaling", "attention_dropout", "is_causal"):
        actual, expected = getattr(candidate, field), getattr(reference, field)
        equivalent = math.isclose(float(actual), float(expected), rel_tol=1e-5, abs_tol=1e-6) if field in {"scaling", "attention_dropout"} else actual == expected
        assert equivalent, f"{field} 与 config 的关系不正确。"
    reference.train(train)
    candidate.train(train)
    return reference, candidate


def compare_forward(cls, cfg=None, *, mask=True, padded=False, train=False, backward=False):
    cfg = cfg or config()
    reference, candidate = pair(cls, cfg, train)
    args = input_data(cfg, mask=mask, padded=padded)
    expected_args, actual_args = copy.deepcopy(args), copy.deepcopy(args)
    expected_args["hidden_states"].requires_grad_(backward)
    actual_args["hidden_states"].requires_grad_(backward)
    torch.manual_seed(31)
    expected = reference(**expected_args)
    torch.manual_seed(31)
    actual = candidate(**actual_args)
    assert isinstance(actual, tuple) and len(actual) == 2, "forward 要返回 (attn_output, attn_weights)。"
    assert_close(actual[0], expected[0])
    assert_close(actual[1], expected[1])
    if backward:
        expected[0].square().mean().backward()
        actual[0].square().mean().backward()
        assert_close(actual_args["hidden_states"].grad, expected_args["hidden_states"].grad, "输入梯度不一致；不要 detach 隐藏状态。")
        expected_parameters = dict(reference.named_parameters())
        for name, a in candidate.named_parameters():
            assert a.grad is not None, f"{name} 没有梯度。"
            assert_close(a.grad, expected_parameters[name].grad, f"{name} 的梯度不一致。")
    return f"output {list(actual[0].shape)} · weights {list(actual[1].shape)}" + (" · 输入及全部参数梯度一致" if backward else "")


def compare_cache(cls):
    cfg = config()
    reference, candidate = pair(cls, cfg)
    expected_cache, actual_cache = DynamicCache(config=cfg), DynamicCache(config=cfg)
    for start, length in ((0, 5), (5, 1), (6, 1)):
        args = input_data(cfg, length=length, start=start)
        expected = reference(**copy.deepcopy(args), past_key_values=expected_cache)
        actual = candidate(**copy.deepcopy(args), past_key_values=actual_cache)
        assert_close(actual[0], expected[0])
        assert_close(actual[1], expected[1])
        assert actual_cache.get_seq_length() == start + length, "KV cache 没有按新 token 数增长。"
    return "5 token prefill → 1 token decode → 1 token decode；cache 长度 5 → 6 → 7"


def compare_model(cls):
    cfg = config()
    torch.manual_seed(47)
    reference = LlamaForCausalLM(copy.deepcopy(cfg)).eval()
    candidate = copy.deepcopy(reference)
    attention = cls(copy.deepcopy(cfg), layer_idx=0)
    attention.load_state_dict(reference.model.layers[0].self_attn.state_dict())
    candidate.model.layers[0].self_attn = attention
    candidate.eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 6))
    expected = reference(input_ids=tokens, labels=tokens, use_cache=False)
    actual = candidate(input_ids=tokens, labels=tokens, use_cache=False)
    assert_close(actual.logits, expected.logits)
    assert_close(actual.loss, expected.loss)
    actual.loss.backward()
    assert all(parameter.grad is not None for parameter in attention.parameters()), "插入完整模型后参数没有收到训练梯度。"
    return f"真实 LlamaForCausalLM · logits {list(actual.logits.shape)} · causal-LM loss {actual.loss.item():.5f} · backward 可运行"


def verify(path):
    started = time.monotonic()
    tests = []
    try:
        cls = load_class(path)
    except Exception as error:
        return {"passed": False, "tests": [], "error": f"{type(error).__name__}: {error}",
                "line": getattr(error, "lineno", None), "duration": time.monotonic() - started}
    cases = [
        ("gqa", "GQA 投影、分头与输出", lambda: compare_forward(cls)),
        ("mha", "标准多头与 bias 配置", lambda: compare_forward(cls, config(hidden_size=48, num_attention_heads=3, num_key_value_heads=3, attention_bias=True))),
        ("head_dim", "不同 hidden_size 与显式 head_dim", lambda: compare_forward(cls, config(hidden_size=48, head_dim=8))),
        ("mask", "因果 / padding mask", lambda: compare_forward(cls, padded=True)),
        ("no_mask", "attention_mask=None 的真实后端行为", lambda: compare_forward(cls, mask=False)),
        ("cache", "KV Cache：prefill 接连续 decode", lambda: compare_cache(cls)),
        ("backward", "训练态 dropout 与输入、参数梯度", lambda: compare_forward(cls, config(attention_dropout=0.2), train=True, backward=True)),
        ("integration", "接回真实 LlamaForCausalLM 并计算训练 loss", lambda: compare_model(cls)),
    ]
    for key, name, execute in cases:
        try:
            detail = execute()
            tests.append({"id": key, "name": name, "passed": True, "detail": detail})
        except Exception as error:
            tests.append({"id": key, "name": name, "passed": False,
                          "detail": f"{type(error).__name__}: {str(error)[:1800]}"})
    return {"passed": all(test["passed"] for test in tests), "tests": tests, "error": None,
            "duration": time.monotonic() - started, "tolerance": "rtol=1e-5, atol=1e-6; full tensors, not samples"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["observe", "verify"])
    parser.add_argument("--source", type=Path, default=LESSON / "reference.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build_observations() if args.action == "observe" else verify(args.source.resolve())
    except Exception as error:
        result = {"passed": False, "error": f"{type(error).__name__}: {error}", "tests": []}
        traceback.print_exc()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
