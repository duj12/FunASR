#!/usr/bin/env python3
"""
将「基座完整 model.pt」与「微调时因 effective_save_name_excludes 未关闭而缺键的 ckpt」合并，
得到可用于 AutoModel 解码的完整权重。

用法示例（推荐：只写 ckpt，放回训练输出目录）:
  python merge_ckpt_with_base.py \\
    --base_model_dir ~/.cache/huggingface/hub/models--FunAudioLLM--Fun-ASR-Nano-2512/snapshots/<hash> \\
    --finetuned_ckpt /path/to/exp/model.pt \\
    --output_ckpt /path/to/exp/model.pt

或整目录复制（含 configuration 等）:
  python merge_ckpt_with_base.py \\
    --base_model_dir ... \\
    --finetuned_ckpt ... \\
    --output_dir ./merged_model_for_decode

说明:
  - base_model_dir: 与训练时 model_name 一致的 Hub 缓存目录（内含完整 model.pt）。
  - finetuned_ckpt: 微调产出目录下的 model.pt（或某个 ep 的 checkpoint）。
  - 合并规则: 以基座 state_dict 为底，用微调文件里存在的同名张量覆盖。
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


def main():
    p = argparse.ArgumentParser(description="Merge base Fun-ASR-Nano weights with finetuned partial ckpt.")
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
