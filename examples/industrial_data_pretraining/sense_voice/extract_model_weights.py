#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
从训练 checkpoint（含 optimizer / scheduler 等）中只导出模型权重，供推理或作为 init_param 加载。

兼容 FunASR 训练保存的 model.pt（含 state_dict、optimizer 等），以及 SenseVoice 等通过 AutoModel + load_pretrained_model 加载的格式。

用法示例:
  python funasr/bin/extract_model_weights.py \\
    --input /path/to/exp/model.pt \\
    --output /path/to/model_weights_only.pt

  # 保留 DDP 的 module. 前缀（若下游要包一层 DDP 加载）:
  python funasr/bin/extract_model_weights.py -i model.pt -o out.pt --keep-module-prefix

说明:
  - DeepSpeed ZeRO 分片 checkpoint 需先用官方工具合并为完整 state_dict 再使用本脚本。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _merge_nested_client_state(obj: Dict[str, Any]) -> Dict[str, Any]:
    """与 trainer 中类似：部分 checkpoint 把训练元数据放在 client_state 下。"""
    if not isinstance(obj, dict):
        return {}
    out = dict(obj)
    cs = obj.get("client_state")
    if isinstance(cs, dict):
        out.update(cs)
    return out


def _pick_model_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    从已 load 的 dict 中取出模型 state_dict；顺序与 load_pretrained_model 一致。
    """
    ckpt = _merge_nested_client_state(ckpt)

    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        sd = ckpt["model_state_dict"]
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    else:
        # 若整个文件就是纯权重（无 epoch/optimizer 等），常见键均为张量
        skip_keys = {
            "epoch",
            "optimizer",
            "scheduler",
            "scaler_state",
            "saved_ckpts",
            "val_acc_step_or_epoch",
            "val_loss_step_or_epoch",
            "best_step_or_epoch",
            "avg_keep_nbest_models_type",
            "step",
            "step_in_epoch",
            "data_split_i",
            "data_split_num",
            "batch_total",
            "train_loss_avg",
            "train_acc_avg",
            "client_state",
        }
        candidate = {k: v for k, v in ckpt.items() if k not in skip_keys}
        tensor_like = {k: v for k, v in candidate.items() if torch.is_tensor(v)}
        if len(tensor_like) >= max(1, len(candidate) // 2):
            sd = tensor_like
        else:
            raise KeyError(
                "无法在 checkpoint 中找到 state_dict / model_state_dict / model，"
                "且无法从顶层推断为纯权重。请确认输入为 FunASR 训练保存的 model.pt。"
            )

    # 只保留张量（去掉误放入的非张量项）
    out_sd = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    if not out_sd:
        raise RuntimeError("提取到的 state_dict 为空，请检查 checkpoint 格式。")
    return out_sd


def _strip_module_prefix(
    state_dict: Dict[str, torch.Tensor], prefix: str = "module."
) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def extract_model_weights(
    input_path: str,
    output_path: str,
    strip_module_prefix: bool = True,
    map_location: str = "cpu",
) -> None:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    logger.info("Loading: %s", input_path)
    ckpt = torch.load(input_path, map_location=map_location)

    if not isinstance(ckpt, dict):
        raise TypeError(f"期望 checkpoint 为 dict，实际为 {type(ckpt)}")

    state_dict = _pick_model_state_dict(ckpt)
    n_keys = len(state_dict)
    sample_keys = list(state_dict.keys())[:3]
    logger.info("Extracted %d tensors; sample keys: %s", n_keys, sample_keys)

    if strip_module_prefix:
        state_dict = _strip_module_prefix(state_dict)
        logger.info("Stripped 'module.' prefix (use --keep-module-prefix to disable).")

    # 与 load_pretrained_model / 推理侧一致：外层包 state_dict
    payload = {"state_dict": state_dict}

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    torch.save(payload, output_path)
    logger.info("Saved model-only weights: %s", output_path)


def main() -> None:
    p = argparse.ArgumentParser(description="从训练 checkpoint 导出仅含模型权重的 .pt")
    p.add_argument("--input", "-i", required=True, help="输入 checkpoint（如 exp/model.pt）")
    p.add_argument("--output", "-o", required=True, help="输出 .pt，仅含 {\"state_dict\": ...}")
    p.add_argument(
        "--keep-module-prefix",
        action="store_true",
        help="保留参数名中的 module. 前缀（默认去掉，便于裸模型加载）",
    )
    p.add_argument("--map-location", default="cpu", help="torch.load map_location，默认 cpu")
    args = p.parse_args()

    try:
        extract_model_weights(
            args.input,
            args.output,
            strip_module_prefix=not args.keep_module_prefix,
            map_location=args.map_location,
        )
    except Exception as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
