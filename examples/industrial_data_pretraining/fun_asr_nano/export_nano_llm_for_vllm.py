#!/usr/bin/env python3
"""
从已合并权重的 Fun-ASR-Nano 模型目录加载 FunASRNano，将 HuggingFace tokenizer 与 LLM 导出到同一目录，
供 Fun-ASR-deploy 中 vLLM（AsyncLLMEngine）加载。

示例:
  python export_nano_llm_for_vllm.py \\
    --asr_model_dir /path/to/checkpoints/FunAudioLLM/Fun-ASR-Nano-2512 \\
    --output_dir /path/to/checkpoints/yuekai/Fun-ASR-Nano-2512-vllm

依赖: 已安装 funasr、transformers；若使用 LoRA，会尝试 merge_and_unload 再保存。
"""
from __future__ import annotations

import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(description="Export tokenizer + LLM for vLLM from Fun-ASR-Nano checkpoint dir.")
    p.add_argument(
        "--asr_model_dir",
        required=True,
        help="含 configuration.json、model.pt（已合并）的 Nano 模型目录",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="输出目录（tokenizer 与 causal LM 均 save_pretrained 到此）",
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="是否将 LLM 权重转换为 bf16 格式保存",
    )
    args = p.parse_args()

    if not os.path.isdir(args.asr_model_dir):
        print(f"ERROR: asr_model_dir 不存在: {args.asr_model_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    from funasr import AutoModel

    model, kwargs = AutoModel.build_model(
        model=args.asr_model_dir,
        trust_remote_code=True,
        device="cpu",
        disable_log=True,
    )

    tokenizer = kwargs.get("tokenizer")
    if tokenizer is None:
        print("ERROR: build_model 未返回 tokenizer", file=sys.stderr)
        sys.exit(1)

    llm = model.llm
    try:
        from peft import PeftModel

        if isinstance(llm, PeftModel):
            llm = llm.merge_and_unload()
    except Exception as e:
        print(f"提示: Peft merge 跳过或非 Peft 模型: {e}", file=sys.stderr)

    tokenizer.save_pretrained(args.output_dir)

    # 转换为 bf16 格式（如果指定）
    if args.bf16:
        import torch
        llm = llm.to(torch.bfloat16)

    llm.save_pretrained(args.output_dir)

    out_abs = os.path.abspath(args.output_dir)
    print(out_abs)
    print(f"已导出 tokenizer + LLM 至: {out_abs}", file=sys.stderr)


if __name__ == "__main__":
    main()
