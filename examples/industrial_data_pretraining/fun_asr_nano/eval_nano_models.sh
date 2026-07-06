#!/bin/bash
# ============================================================
# Fun-ASR-Nano 微调 checkpoint 批量评测（合并权重 → 导出 vLLM → 起服务 → 客户端 → WER）
#
# 本地路径映射（Windows）: /data/megastore/Projects/DuJing/code -> D:/work/code
#
# 使用前请修改「路径配置」段，尤其 MERGE_BASE_DIR：
#   必须指向**从未被不完整 ckpt 覆盖**的完整基座目录（Hub 缓存 snapshots 或官方包），
#   其中的 model.pt 为完整权重；合并结果写入 ASR_MODEL_OUTPUT/model.pt。
#   ASR_MODEL_OUTPUT 目录本身需含 configuration.json 等（可从基座整目录复制一份到 deploy checkpoints）。
# ============================================================

set -e

# ---------- 路径配置（按你的机器修改）----------

# 合并用基座：**完整** model.pt（从未被「缺键保存」覆盖）。留空则尝试自动探测 HuggingFace 缓存 snapshots
MERGE_BASE_DIR=/home/dujing/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512
# 手动指定示例:
# MERGE_BASE_DIR="/data/.../Fun-ASR-deploy/checkpoints/FunAudioLLM/Fun-ASR-Nano-2512-full-backup"
# MERGE_BASE_DIR="$HOME/.cache/huggingface/hub/models--FunAudioLLM--Fun-ASR-Nano-2512/snapshots/<hash>"

# 部署用 ASR 模型目录（与 funasr_wss_server.py 中 checkpoints/FunAudioLLM/Fun-ASR-Nano-2512 一致）
ASR_SERVER_DIR="/data/megastore/Projects/DuJing/code/Fun-ASR-deploy"
ASR_MODEL_OUTPUT="${ASR_SERVER_DIR}/checkpoints/FunAudioLLM/Fun-ASR-Nano-2512"

# vLLM 权重输出（与 server 默认 --vllm_model_dir yuekai/Fun-ASR-Nano-2512-vllm 对应）
VLLM_MODEL_OUTPUT="${ASR_SERVER_DIR}/checkpoints/yuekai/Fun-ASR-Nano-2512-vllm"

# FunASR 仓库内脚本（本地 D: 盘对应替换前缀即可）
FUNASR_NANO_DIR="/data/megastore/Projects/DuJing/code/FunASR-main/examples/industrial_data_pretraining/fun_asr_nano"
# 训练产出目录：内含 model.pt、model.pt.ep* 等
ASR_MODEL_DIR="${FUNASR_NANO_DIR}/exp_ft_se_wali3+wild"
#ASR_MODEL_DIR="/data/megastore/Projects/DuJing/code/Fun-ASR/exp_ft_se_wali3+wild"

# LoRA 参数（与 finetune.sh 中一致，留空则不启用 LoRA 合并）
LORA_RANK=8
LORA_ALPHA=16

# 跳过参数合并，直接拷贝原始 ckpt 到目标路径（适用于 ckpt 已是完整模型权重的情况）
# 设为 true 时跳过 merge_ckpt_with_base.py，直接用 cp 拷贝 model.pt
SKIP_MERGE=false

# 导出 vLLM 权重时是否保存为 bf16 格式
# 设为 true 时将 LLM 权重转换为 bfloat16 格式保存，减少显存占用和加载时间
EXPORT_BF16=false

MERGE_SCRIPT="${FUNASR_NANO_DIR}/merge_ckpt_with_base.py"
EXPORT_SCRIPT="${FUNASR_NANO_DIR}/export_nano_llm_for_vllm.py"

# 服务与客户端
SERVER_PORT=10095
SERVER_HOST_LISTEN="0.0.0.0"
CLIENT_HOST="0.0.0.0"

# 评测数据与 WER
TEST_SCP="/data/megastore/Datasets/ASR/Test/WaLi_real/wav.scp"
TEST_OUTPUT_DIR="/data/megastore/Datasets/ASR/Test/WaLi_real/xmov_llmasr_wali2ft_alldata_upreal_llm/asr"
WER_DATA_ROOT="/data/megastore/Datasets/ASR/Test/WaLi_real"

# 结果记录
RESULT_FILE="${ASR_MODEL_DIR}/eval_nano_results.txt"

# 可选 conda（留空则使用当前 PATH 中的 python）
# CONDA_ENV="dj"
CONDA_ENV="vllm"

# ---------- 辅助函数 ----------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

wait_port_listen() {
    local port=$1
    local timeout=${2:-180}
    local elapsed=0
    log "等待端口 ${port} 监听..."
    while true; do
        if command -v ss >/dev/null 2>&1; then
            ss -tlnp 2>/dev/null | grep -q ":${port} " && break
        elif command -v netstat >/dev/null 2>&1; then
            netstat -tlnp 2>/dev/null | grep -q ":${port} " && break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        if [ "$elapsed" -ge "$timeout" ]; then
            log "ERROR: 超时 ${timeout}s，端口 ${port} 未就绪"
            exit 1
        fi
    done
    log "端口 ${port} 已就绪（约 ${elapsed}s）"
}

kill_server() {
    local port=$1
    local pids
    pids=$(lsof -ti "tcp:${port}" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log "关闭占用端口 ${port} 的进程: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

stop_process_group() {
    # 优雅停止一个“进程组”（通常等于服务主进程 PID），确保子进程/worker 一并退出，从而释放显存
    # 用法: stop_process_group <pgid> [timeout_s]
    local pgid=$1
    local timeout=${2:-30}
    local elapsed=0

    if [ -z "$pgid" ]; then
        return 0
    fi

    # 若进程不存在，直接返回
    if ! kill -0 "$pgid" 2>/dev/null; then
        return 0
    fi

    log "优雅停止服务进程组 PGID=${pgid} (SIGTERM, timeout=${timeout}s)"
    kill -TERM "--" "-${pgid}" 2>/dev/null || true

    while kill -0 "$pgid" 2>/dev/null; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ "$elapsed" -ge "$timeout" ]; then
            break
        fi
    done

    if kill -0 "$pgid" 2>/dev/null; then
        log "超时未退出，强制停止进程组 PGID=${pgid} (SIGKILL)"
        kill -KILL "--" "-${pgid}" 2>/dev/null || true
        sleep 2
    fi
}

stop_server() {
    # 优先按进程组停（释放显存更可靠），再按端口兜底
    local port=$1
    local pgid=$2
    stop_process_group "$pgid" 40
    kill_server "$port"
}

activate_conda() {
    if [ -n "$CONDA_ENV" ]; then
        local conda_base
        conda_base=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
        # shellcheck source=/dev/null
        source "${conda_base}/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    fi
}

resolve_merge_base_dir() {
    if [ -n "$MERGE_BASE_DIR" ] && [ -f "${MERGE_BASE_DIR}/model.pt" ]; then
        return 0
    fi
    local hub_root hf_snap
    hub_root="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    hf_snap="${hub_root}/models--FunAudioLLM--Fun-ASR-Nano-2512/snapshots"
    if [ -d "$hf_snap" ]; then
        MERGE_BASE_DIR=$(find "$hf_snap" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -1)
    fi
}

# ---------- 前置检查 ----------
if [ ! -f "$MERGE_SCRIPT" ]; then
    log "ERROR: 未找到 merge 脚本: $MERGE_SCRIPT"
    exit 1
fi
if [ ! -f "$EXPORT_SCRIPT" ]; then
    log "ERROR: 未找到 export 脚本: $EXPORT_SCRIPT"
    exit 1
fi

resolve_merge_base_dir
if [ ! -d "$MERGE_BASE_DIR" ] || [ ! -f "${MERGE_BASE_DIR}/model.pt" ]; then
    log "ERROR: 无法解析 MERGE_BASE_DIR（需要完整基座 model.pt）: ${MERGE_BASE_DIR:-<empty>}"
    log "请设置 MERGE_BASE_DIR 为 Hub snapshots 目录或单独备份的完整 Nano-2512 目录。"
    exit 1
fi
log "使用合并基座目录: $MERGE_BASE_DIR"
if [ ! -d "$ASR_MODEL_OUTPUT" ]; then
    log "ERROR: ASR_MODEL_OUTPUT 不存在: $ASR_MODEL_OUTPUT"
    log "请先将基座模型目录（含 configuration.json、config.yaml、tokenizer 等）拷到 deploy checkpoints 下。"
    exit 1
fi
if [ ! -f "${ASR_MODEL_OUTPUT}/configuration.json" ] && [ ! -f "${ASR_MODEL_OUTPUT}/config.yaml" ]; then
    log "WARN: ASR_MODEL_OUTPUT 下未找到 configuration.json / config.yaml，AutoModel 可能无法加载。"
fi

mkdir -p "$(dirname "$RESULT_FILE")" 2>/dev/null || true

if [ ! -f "$RESULT_FILE" ]; then
    echo "# Fun-ASR-Nano 批量评测" > "$RESULT_FILE"
    echo "# 列: 模型文件名  WER(%)" >> "$RESULT_FILE"
    echo "# 时间: $(date)" >> "$RESULT_FILE"
    echo "# ----------------------------------------" >> "$RESULT_FILE"
fi

activate_conda

log "============================================"
log "ASR_MODEL_DIR=$ASR_MODEL_DIR"
log "MERGE_BASE_DIR=$MERGE_BASE_DIR"
log "ASR_MODEL_OUTPUT=$ASR_MODEL_OUTPUT"
log "VLLM_MODEL_OUTPUT=$VLLM_MODEL_OUTPUT"
log "结果文件: $RESULT_FILE"
log "============================================"

shopt -s nullglob
MODEL_PTS=( "${ASR_MODEL_DIR}"/model.pt* )
if [ ${#MODEL_PTS[@]} -eq 0 ]; then
    log "ERROR: 在 $ASR_MODEL_DIR 下未找到 model.pt*"
    exit 1
fi

# 版本排序：model.pt, model.pt.ep1.100, ...
IFS=$'\n' MODEL_PTS=( $(printf '%s\n' "${MODEL_PTS[@]}" | sort -V) )
unset IFS

for MODEL_PT in "${MODEL_PTS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PT")
    log "---------- 处理: ${MODEL_NAME} ----------"

    if awk -v name="$MODEL_NAME" '
        BEGIN { found=0 }
        /^#/ { next }
        NF == 0 { next }
        $1 == name { found=1; exit }
        END { exit (found ? 0 : 1) }
    ' "$RESULT_FILE" 2>/dev/null; then
        log "已有结果，跳过: ${MODEL_NAME}"
        continue
    fi

    # ---- 1) 获取 ckpt -> ASR_MODEL_OUTPUT/model.pt ----
    if [ "$SKIP_MERGE" = "true" ]; then
        log "[1/5] 跳过合并，直接拷贝: ${MODEL_NAME} -> ${ASR_MODEL_OUTPUT}/model.pt"
        set +e
        cp -f "$MODEL_PT" "${ASR_MODEL_OUTPUT}/model.pt"
        COPY_EC=$?
        set -e
        if [ "$COPY_EC" -ne 0 ]; then
            log "ERROR: 拷贝失败 (exit $COPY_EC): ${MODEL_NAME}"
            echo "${MODEL_NAME}  COPY_FAILED" >> "$RESULT_FILE"
            continue
        fi
        log "拷贝完成: ${ASR_MODEL_OUTPUT}/model.pt"
    else
        log "[1/5] merge_ckpt_with_base: ${MODEL_NAME} -> ${ASR_MODEL_OUTPUT}/model.pt"
        set +e
        # 构建 merge 命令（如果设置了 LoRA 参数则追加 --lora_rank / --lora_alpha）
        MERGE_CMD="python \"$MERGE_SCRIPT\" \
            --base_model_dir \"$MERGE_BASE_DIR\" \
            --finetuned_ckpt \"$MODEL_PT\" \
            --output_ckpt \"${ASR_MODEL_OUTPUT}/model.pt\""
        if [ -n "$LORA_RANK" ] && [ -n "$LORA_ALPHA" ]; then
            MERGE_CMD="$MERGE_CMD --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA}"
        fi
        log "merge 命令: $MERGE_CMD"
        MERGE_LOG="/tmp/merge_ckpt_${MODEL_NAME}.log"
        MERGED=$(eval $MERGE_CMD 2>"$MERGE_LOG")
        MERGE_EC=$?
        # 输出 merge 诊断日志（LoRA 合并信息等）
        if [ -f "$MERGE_LOG" ] && [ -s "$MERGE_LOG" ]; then
            log "merge 日志:"
            while IFS= read -r line; do log "  $line"; done < "$MERGE_LOG"
        fi
        set -e
        if [ "$MERGE_EC" -ne 0 ] || [ -z "$MERGED" ]; then
            log "ERROR: merge 失败 (exit $MERGE_EC): ${MODEL_NAME}"
            echo "${MODEL_NAME}  MERGE_FAILED" >> "$RESULT_FILE"
            continue
        fi
        log "合并完成: $MERGED"
    fi

    # ---- 2) 导出 tokenizer + llm 供 vLLM ----
    log "[2/5] export_nano_llm_for_vllm -> ${VLLM_MODEL_OUTPUT}"
    set +e
    EXPORT_ARGS="--asr_model_dir \"$ASR_MODEL_OUTPUT\" --output_dir \"$VLLM_MODEL_OUTPUT\""
    if [ "$EXPORT_BF16" = "true" ]; then
        EXPORT_ARGS="$EXPORT_ARGS --bf16"
        log "启用 bf16 格式导出"
    fi
    VLLM_OUT=$(eval python "$EXPORT_SCRIPT" $EXPORT_ARGS 2>/dev/null)
    EXPORT_EC=$?
    set -e
    if [ "$EXPORT_EC" -ne 0 ] || [ -z "$VLLM_OUT" ]; then
        log "ERROR: 导出 vLLM 权重失败 (exit $EXPORT_EC): ${MODEL_NAME}"
        echo "${MODEL_NAME}  EXPORT_VLLM_FAILED" >> "$RESULT_FILE"
        continue
    fi
    log "vLLM 目录: $VLLM_OUT"

    # ---- 3) 启动服务 ----
    log "[3/5] 启动 funasr_wss_server (port ${SERVER_PORT})"
    stop_server "$SERVER_PORT" "${SERVER_PID:-}"

    cd "$ASR_SERVER_DIR"
    # setsid: 为服务创建独立 session/process-group，便于后续 kill -TERM -PGID 一次性杀掉所有子进程（vLLM workers 等）
    nohup setsid python -u funasr_wss_server.py \
        --host "$SERVER_HOST_LISTEN" \
        --port "$SERVER_PORT" \
        --device cuda \
        --asr_model "FunAudioLLM/Fun-ASR-Nano-2512" \
        --vllm_model_dir "yuekai/Fun-ASR-Nano-2512-vllm" \
        >> "/tmp/funasr_wss_server_${MODEL_NAME}.log" 2>&1 &
    SERVER_PID=$!
    log "服务 PID: $SERVER_PID, 日志: /tmp/funasr_wss_server_${MODEL_NAME}.log"

    if ! wait_port_listen "$SERVER_PORT" 240; then
        log "ERROR: 服务未启动成功，跳过 ${MODEL_NAME}"
        stop_server "$SERVER_PORT" "$SERVER_PID"
        echo "${MODEL_NAME}  SERVER_FAILED" >> "$RESULT_FILE"
        continue
    fi

    # ---- 4) 客户端 ----
    log "[4/5] funasr_wss_client -> ${TEST_OUTPUT_DIR}"
    rm -rf "$TEST_OUTPUT_DIR"
    mkdir -p "$TEST_OUTPUT_DIR"

    cd "$ASR_SERVER_DIR"
    if ! python -u funasr_wss_client.py \
        --port "$SERVER_PORT" \
        --host "$CLIENT_HOST" \
        --audio_in "$TEST_SCP" \
        --thread_num 4 \
        --mode offline \
        --itn 0 \
        --output_dir "$TEST_OUTPUT_DIR" \
        --svs_lang zh    \
        --vad_energy -100 \
        --vad_tail_sil 800 \
        --vad_max_len 60000; then
        log "ERROR: 客户端失败: ${MODEL_NAME}"
        stop_server "$SERVER_PORT" "$SERVER_PID"
        echo "${MODEL_NAME}  CLIENT_FAILED" >> "$RESULT_FILE"
        continue
    fi

    stop_server "$SERVER_PORT" "$SERVER_PID"

    # ---- 5) WER ----
    log "[5/5] run_wer.sh"
    cd "$WER_DATA_ROOT"
    if [ ! -f "./run_wer.sh" ]; then
        log "ERROR: 未找到 ${WER_DATA_ROOT}/run_wer.sh"
        echo "${MODEL_NAME}  NO_RUN_WER" >> "$RESULT_FILE"
        continue
    fi

    if ! bash run_wer.sh "$TEST_OUTPUT_DIR"; then
        log "ERROR: run_wer.sh 失败: ${MODEL_NAME}"
        echo "${MODEL_NAME}  WER_FAILED" >> "$RESULT_FILE"
        continue
    fi

    WER_LINE=$(tail -n 80 "${WER_DATA_ROOT}/wer.txt" 2>/dev/null | grep "Overall" | tail -1 || true)
    log "WER 行: ${WER_LINE}"
    WER_VALUE=$(echo "$WER_LINE" | grep -oP '(?<=-> )\s*[\d.]+(?=\s*%)' 2>/dev/null | tr -d ' ' || true)
    if [ -z "$WER_VALUE" ]; then
        WER_VALUE=$(echo "$WER_LINE" | grep -oE '[0-9]+\.[0-9]+' | tail -1 || echo "")
    fi
    if [ -z "$WER_VALUE" ]; then
        WER_VALUE="PARSE_FAILED"
    fi

    log ">>> ${MODEL_NAME}  WER: ${WER_VALUE}%"
    printf "%-50s  %s\n" "${MODEL_NAME}" "${WER_VALUE}" >> "$RESULT_FILE"
    log "---------- 完成: ${MODEL_NAME} ----------"
    echo ""
done

log "全部结束，结果: $RESULT_FILE"
cat "$RESULT_FILE"
