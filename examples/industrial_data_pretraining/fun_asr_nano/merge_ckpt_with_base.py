#!/usr/bin/env python3
"""
将「基座完整 model.pt」与「微调时因 effective_save_name_excludes 未关闭而缺键的 ckpt」合并，
得到可用于 AutoModel 解码的完整权重。

当微调使用了 LoRA 时（--lora_rank / --lora_alpha），会自动：
  1. 检测 checkpoint 中的 lora_A / lora_B 权重
  2. 将 LoRA 权重合并回 base weight: W = W_base + (alpha/r) * B @ A
  3. 将 PEFT 格式的 key（含 base_model.model.）映射回原始 key
  4. 输出与基座结构完全一致的 clean checkpoint

用法示例（无 LoRA）:
  python merge_ckpt_with_base.py \\
    --base_model_dir ~/.cache/huggingface/hub/models--FunAudioLLM--Fun-ASR-Nano-2512/snapshots/<hash> \\
    --finetuned_ckpt /path/to/exp/model.pt \\
    --output_ckpt /path/to/exp/model.pt

用法示例（有 LoRA）:
  python merge_ckpt_with_base.py \\
    --base_model_dir ... \\
    --finetuned_ckpt ... \\
    --output_ckpt ... \\
    --lora_rank 8 \\
    --lora_alpha 16

说明:
  - base_model_dir: 与训练时 model_name 一致的 Hub 缓存目录（内含完整 model.pt）。
  - finetuned_ckpt: 微调产出目录下的 model.pt（或某个 ep 的 checkpoint）。
  - 合并规则: 以基座 state_dict 为底，用微调文件里存在的同名张量覆盖。
  - LoRA 合并: W_merged = W_base + (alpha/r) * lora_B @ lora_A
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys


def _pick_state_dict(obj: dict):
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        return obj["model_state_dict"]
    if "model" in obj and isinstance(obj["model"], dict) and obj["model"]:
        return obj["model"]
    if obj and all(hasattr(v, "shape") for v in obj.values()):
        return obj
    raise KeyError("无法在 checkpoint 中找到 state_dict / model_state_dict / model")


def _strip_module_prefix(sd: dict) -> dict:
    """与 resume 逻辑一致：部分 ckpt 可能带 DDP 的 module. 前缀。"""
    out = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def _peft_key_to_original(key: str) -> str:
    """将 PEFT 包装后的 key 映射回原始 key。

    例如:
      llm.base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight
      -> llm.model.model.layers.0.self_attn.q_proj.weight

      llm.base_model.model.model.embed_tokens.weight
      -> llm.model.model.embed_tokens.weight
    """
    # 只处理 llm 下的 PEFT key
    if key.startswith("llm.base_model.model."):
        key = key.replace("llm.base_model.model.", "llm.model.", 1)
    # PEFT v0.6+ 将 base weight 存为 base_layer.weight
    key = key.replace(".base_layer.", ".")
    return key


def _extract_lora_target_path(lora_key: str) -> str:
    """从 lora_A / lora_B key 中提取目标层的路径（不含 lora_A/B 后缀和 weight）。

    例如:
      llm.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
      -> llm.base_model.model.model.layers.0.self_attn.q_proj
    """
    for suffix in (".lora_A.default.weight", ".lora_B.default.weight"):
        if lora_key.endswith(suffix):
            return lora_key[: -len(suffix)]
    return lora_key


def _has_lora_keys(sd: dict) -> bool:
    """检测 state_dict 中是否包含 LoRA 权重。"""
    return any("lora_A" in k or "lora_B" in k for k in sd.keys())


def _merge_lora_weights(ft_sd: dict, base_sd: dict, lora_rank: int, lora_alpha: int) -> dict:
    """将 LoRA 权重合并回 base weight，返回 clean state_dict。

    合并公式: W_merged = W_base + (alpha / r) * lora_B @ lora_A
    """
    scaling = lora_alpha / lora_rank
    merged = dict(base_sd)

    # 1. 找到所有 lora_A / lora_B 配对
    lora_a_keys = sorted(k for k in ft_sd if ".lora_A.default.weight" in k)
    lora_pairs = {}
    for a_key in lora_a_keys:
        b_key = a_key.replace(".lora_A.default.weight", ".lora_B.default.weight")
        if b_key in ft_sd:
            target_path = _extract_lora_target_path(a_key)
            lora_pairs[target_path] = (a_key, b_key)
        else:
            print(f"警告: 找到 lora_A 但缺少对应 lora_B: {a_key}", file=sys.stderr)

    print(f"检测到 {len(lora_pairs)} 组 LoRA 权重 (rank={lora_rank}, alpha={lora_alpha}, scaling={scaling})", file=sys.stderr)

    # 2. 合并每组 LoRA
    for target_path, (a_key, b_key) in lora_pairs.items():
        lora_a = ft_sd[a_key].float()
        lora_b = ft_sd[b_key].float()

        # 映射到原始 key
        original_path = _peft_key_to_original(target_path)
        original_weight_key = original_path + ".weight"

        # 查找 base weight：先从 ft_sd 找（PEFT 格式），再从 base_sd 找
        peft_base_key_new = target_path + ".base_layer.weight"  # PEFT v0.6+
        peft_base_key_old = target_path + ".weight"             # PEFT v0.5

        if peft_base_key_new in ft_sd:
            base_weight = ft_sd[peft_base_key_new].float()
        elif peft_base_key_old in ft_sd and "lora_A" not in peft_base_key_old and "lora_B" not in peft_base_key_old:
            base_weight = ft_sd[peft_base_key_old].float()
        elif original_weight_key in merged:
            base_weight = merged[original_weight_key].float()
        else:
            print(f"警告: 未找到 LoRA 目标 {original_weight_key} 的 base weight，跳过", file=sys.stderr)
            continue

        merged_weight = base_weight + scaling * (lora_b @ lora_a)
        merged[original_weight_key] = merged_weight.to(base_weight.dtype)
        print(f"  合并: {a_key} -> {original_weight_key}", file=sys.stderr)

    # 3. 处理 ft_sd 中的其他 key（encoder, adaptor, 非 LoRA LLM weight）
    for k, v in ft_sd.items():
        # 跳过已处理的 LoRA key
        if ".lora_A." in k or ".lora_B." in k:
            continue

        # PEFT 格式的 LLM key -> 映射回原始 key
        if k.startswith("llm.base_model.model."):
            original_key = _peft_key_to_original(k)
            # 跳过已经通过 LoRA 合并处理过的 base_layer weight
            if ".base_layer." in k:
                # base_layer weight 已经在 LoRA 合并中处理过，如果没有对应的 lora pair 则直接映射
                if original_key in merged:
                    # 已经通过 LoRA 合并写入了，用 ft_sd 的值覆盖（仅当该层没有 LoRA 时）
                    target_path_check = k.replace(".base_layer.weight", "").replace(".weight", "")
                    if target_path_check not in lora_pairs:
                        if merged[original_key].shape != v.shape:
                            print(f"警告: key {original_key} 形状不一致，基座 {tuple(merged[original_key].shape)} vs 微调 {tuple(v.shape)}", file=sys.stderr)
                        merged[original_key] = v
                else:
                    merged[original_key] = v
            else:
                # 非 base_layer 的 PEFT key（如 embed_tokens, layer_norm 等）
                if original_key in merged and merged[original_key].shape != v.shape:
                    print(f"警告: key {original_key} 形状不一致，基座 {tuple(merged[original_key].shape)} vs 微调 {tuple(v.shape)}", file=sys.stderr)
                merged[original_key] = v
        else:
            # 非 LLM key（encoder, adaptor）直接覆盖
            if k in merged and merged[k].shape != v.shape:
                print(f"警告: key {k} 形状不一致，基座 {tuple(merged[k].shape)} vs 微调 {tuple(v.shape)}，使用微调张量", file=sys.stderr)
            merged[k] = v

    return merged


def main():
    p = argparse.ArgumentParser(description="Merge base Fun-ASR-Nano weights with finetuned partial ckpt. Supports LoRA merging.")
    p.add_argument(
        "--base_model_dir",
        required=True,
        help="含完整 model.pt 的基座目录（Hub 下载目录或官方包解压目录）",
    )
    p.add_argument(
        "--finetuned_ckpt",
        required=True,
        help="微调保存的 model.pt（可能缺部分 key）",
    )
    out = p.add_mutually_exclusive_group(required=True)
    out.add_argument(
        "--output_ckpt",
        metavar="PATH",
        help="合并后 checkpoint 的完整路径（可直接指向原训练 output 下的 model.pt，便于覆盖）",
    )
    out.add_argument(
        "--output_dir",
        metavar="DIR",
        help="输出目录：复制基座非权重文件，并写入合并后的 model.pt",
    )
    p.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="LoRA rank (r)，当 checkpoint 含 LoRA 权重时必须提供",
    )
    p.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha，当 checkpoint 含 LoRA 权重时必须提供",
    )
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        print("需要安装 torch", file=sys.stderr)
        sys.exit(1)

    base_pt = os.path.join(args.base_model_dir, "model.pt")
    if not os.path.isfile(base_pt):
        print(f"未找到基座权重: {base_pt}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.finetuned_ckpt):
        print(f"未找到微调 ckpt: {args.finetuned_ckpt}", file=sys.stderr)
        sys.exit(1)

    base_obj = torch.load(base_pt, map_location="cpu")
    ft_obj = torch.load(args.finetuned_ckpt, map_location="cpu")

    base_sd = _pick_state_dict(base_obj if isinstance(base_obj, dict) else {})
    ft_sd = _pick_state_dict(ft_obj if isinstance(ft_obj, dict) else {})
    base_sd = _strip_module_prefix(base_sd)
    ft_sd = _strip_module_prefix(ft_sd)

    has_lora = _has_lora_keys(ft_sd)
    print(f"has LoRA: {has_lora}")
    if has_lora:
        if args.lora_rank is None or args.lora_alpha is None:
            print("错误: 检测到 LoRA 权重，但未提供 --lora_rank 和 --lora_alpha 参数", file=sys.stderr)
            sys.exit(1)
        print(f"检测到 LoRA 权重，执行 LoRA 合并 (rank={args.lora_rank}, alpha={args.lora_alpha})", file=sys.stderr)
        merged = _merge_lora_weights(ft_sd, base_sd, args.lora_rank, args.lora_alpha)
    else:
        # 原始合并逻辑
        merged = dict(base_sd)
        for k, v in ft_sd.items():
            if k in merged and merged[k].shape != v.shape:
                print(f"警告: 键 {k} 形状不一致，基座 {tuple(merged[k].shape)} vs 微调 {tuple(v.shape)}，使用微调张量")
            merged[k] = v

    if isinstance(base_obj, dict) and "state_dict" in base_obj:
        out_obj = dict(base_obj)
        out_obj["state_dict"] = merged
    else:
        out_obj = {"state_dict": merged}

    if args.output_ckpt:
        out_path = os.path.abspath(args.output_ckpt)
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(out_obj, out_path)
        print(out_path)
        print(
            f"基座参数数: {len(base_sd)}, 微调文件中参数数: {len(ft_sd)}, 合并后: {len(merged)}",
            file=sys.stderr,
        )
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for name in os.listdir(args.base_model_dir):
        if name == "model.pt":
            continue
        src = os.path.join(args.base_model_dir, name)
        dst = os.path.join(args.output_dir, name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    out_path = os.path.join(args.output_dir, "model.pt")
    torch.save(out_obj, out_path)
    print(os.path.abspath(out_path))
    print(
        f"基座参数数: {len(base_sd)}, 微调文件中参数数: {len(ft_sd)}, 合并后: {len(merged)}",
        file=sys.stderr,
    )
    print("解码时请设置 model_dir 指向该 output_dir。", file=sys.stderr)


if __name__ == "__main__":
    main()
