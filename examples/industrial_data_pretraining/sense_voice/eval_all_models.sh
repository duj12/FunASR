#!/bin/bash
# ============================================================
# 批量模型评测脚本
# 遍历 exp_83whours 下所有 model.pt* 文件，逐一：
#   1. 复制为 SenseVoiceSmall/model.pt
#   2. 导出 ONNX (conda: dj_py38)
#   3. 部署到 asr-2pass 并启动服务
#   4. 跑 funasr_wss_client.py 测试 (conda: dj)
#   5. 跑 run_wer.sh 计算 WER，记录结果
# ============================================================

set -e  # 遇错即停（除了特别处理的命令）

# ---------- 路径配置 ----------
SENSE_VOICE_DIR="/data/megastore/Projects/DuJing/code/FunASR-main/examples/industrial_data_pretraining/sense_voice"
SMALL_DIR="${SENSE_VOICE_DIR}/SenseVoiceSmall"

ONNX_DST="/data/megastore/Projects/DuJing/code/asr-2pass/websocket/models/iic/SenseVoiceSmall-onnx"
SERVER_DIR="/data/megastore/Projects/DuJing/code/asr-2pass/websocket"
CLIENT_DIR="/data/megastore/Projects/DuJing/code/asr-2pass/clients/python"

MODEL_DIR="${SENSE_VOICE_DIR}/exp_250hours"
TEST_SCP="/data/megastore/Datasets/ASR/Test/Test_xmov/xmov_asr/10db_wav.scp"
TEST_OUT="/data/megastore/Datasets/ASR/Test/Test_xmov/xmov_asr/10db"
WER_DIR="/data/megastore/Datasets/ASR/Test/Test_xmov/xmov_asr"

MODEL_DIR="${SENSE_VOICE_DIR}/exp_ft_se_wali3+wild"
TEST_SCP="/data/megastore/Datasets/ASR/Test/WaLi_real/wav.scp"
TEST_OUT="/data/megastore/Datasets/ASR/Test/WaLi_real/svsori_wali3+wilddata_se/asr"
WER_DIR="/data/megastore/Datasets/ASR/Test/WaLi_real"

RESULT_FILE="${MODEL_DIR}/eval_results.txt"
SERVER_PORT=10096
SERVER_HOST="192.168.89.105"

# ---------- conda 初始化 ----------
# 确保 conda 命令可用
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---------- 辅助函数 ----------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

wait_port_listen() {
    local port=$1
    local timeout=${2:-120}  # 默认最多等 120 秒
    local elapsed=0
    log "等待端口 ${port} 开始监听..."
    while ! ss -tlnp 2>/dev/null | grep -q ":${port} " && \
          ! netstat -tlnp 2>/dev/null | grep -q ":${port} "; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $elapsed -ge $timeout ]; then
            log "ERROR: 超时 ${timeout}s，端口 ${port} 未就绪，退出"
            exit 1
        fi
    done
    log "端口 ${port} 已就绪（等待 ${elapsed}s）"
}

kill_server() {
    # 杀掉占用 SERVER_PORT 的进程
    local pids
    pids=$(lsof -ti tcp:${SERVER_PORT} 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log "关闭旧服务 (pid: $pids)..."
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

# ---------- 初始化结果文件 ----------
if [ ! -f "$RESULT_FILE" ]; then
    echo "# 模型评测结果" > "$RESULT_FILE"
    echo "# 格式: 模型文件名  WER(%)" >> "$RESULT_FILE"
    echo "# 生成时间: $(date)" >> "$RESULT_FILE"
    echo "# ----------------------------------------" >> "$RESULT_FILE"
fi

log "============================================"
log "开始批量评测，结果将写入: $RESULT_FILE"
log "============================================"

# ---------- 主循环 ----------
for MODEL_PT in "${MODEL_DIR}"/model.pt*; do
    MODEL_NAME=$(basename "$MODEL_PT")
    log "---------- 处理模型: ${MODEL_NAME} ----------"

    # 若结果文件中已有该模型记录（第一列为模型文件名，忽略 # 注释行），则跳过，避免重复评测
    # 使用 awk 精确匹配 $1，避免 grep 正则把 model.pt 里的 "." 当通配符导致误判
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

    # ---- Step 1: 复制模型 ----
    log "[1/6] 复制 ${MODEL_NAME} -> ${SMALL_DIR}/model.pt"
    cp -f "$MODEL_PT" "${SMALL_DIR}/model.pt"

    # ---- Step 2: 导出 ONNX ----
    log "[2/6] 导出 ONNX (conda: dj_py38)"
    conda activate dj_py38
    cd "$SENSE_VOICE_DIR"
    if ! python export.py; then
        log "ERROR: export.py 执行失败，跳过 ${MODEL_NAME}"
        conda deactivate
        echo "${MODEL_NAME}  EXPORT_FAILED" >> "$RESULT_FILE"
        continue
    fi
    conda deactivate

    # 检查导出产物
    if [ ! -f "${SMALL_DIR}/model_quant.onnx" ]; then
        log "ERROR: model_quant.onnx 未生成，跳过 ${MODEL_NAME}"
        echo "${MODEL_NAME}  ONNX_NOT_FOUND" >> "$RESULT_FILE"
        continue
    fi

    # ---- Step 3: 部署 ONNX ----
    log "[3/6] 复制 model_quant.onnx -> ${ONNX_DST}"
    cp -f "${SMALL_DIR}/model_quant.onnx" "${ONNX_DST}/model_quant.onnx"

    # ---- Step 4: 启动服务 ----
    log "[4/6] 启动 ASR 服务 (端口 ${SERVER_PORT})"
    kill_server  # 先确保端口干净

    cd "$SERVER_DIR"
    bash run_server_2pass.sh >> /tmp/asr_server_${MODEL_NAME}.log 2>&1 &
    SERVER_PID=$!
    log "服务进程 PID: ${SERVER_PID}"

    # 等待服务在端口监听
    wait_port_listen $SERVER_PORT 180

    # ---- Step 5: 运行客户端测试 ----
    log "[5/6] 运行客户端推理测试 (conda: dj)"
    conda activate dj
    cd "$CLIENT_DIR"

    # 清理上次输出
    rm -rf "$TEST_OUT"

    if ! python -u funasr_wss_client.py \
        --port $SERVER_PORT \
        --host $SERVER_HOST \
        --audio_in "$TEST_SCP" \
        --thread_num 8 \
        --mode offline \
        --itn 0 \
        --output_dir "$TEST_OUT" \
        --vad_energy -100 \
        --vad_max_len 60000; then
        log "ERROR: 客户端推理失败，跳过 ${MODEL_NAME}"
        conda deactivate
        kill_server
        echo "${MODEL_NAME}  CLIENT_FAILED" >> "$RESULT_FILE"
        continue
    fi
    conda deactivate

    # ---- Step 6: 计算 WER ----
    log "[6/6] 计算 WER"
    cd "$WER_DIR"
    if ! bash run_wer.sh $TEST_OUT ; then
        log "ERROR: run_wer.sh 执行失败，跳过 ${MODEL_NAME}"
        kill_server
        echo "${MODEL_NAME}  WER_FAILED" >> "$RESULT_FILE"
        continue
    fi

    # 提取 Overall WER
    WER_LINE=$(tail -n 50 "${WER_DIR}/wer.txt" 2>/dev/null | grep "Overall" | tail -1)
    log "WER 输出行: ${WER_LINE}"

    # 匹配 "Overall -> 4.50 %" 形式，提取百分号前的数字
    WER_VALUE=$(echo "$WER_LINE" | grep -oP '(?<=-> )\s*[\d.]+(?=\s*%)' | tr -d ' ')

    if [ -z "$WER_VALUE" ]; then
        log "WARNING: 未能从 wer.txt 提取到 WER 数值"
        WER_VALUE="PARSE_FAILED"
    fi

    log ">>> 模型: ${MODEL_NAME}  WER: ${WER_VALUE}%"
    printf "%-40s  %s\n" "${MODEL_NAME}" "${WER_VALUE}" >> "$RESULT_FILE"

    # ---- 清理：关闭服务 ----
    kill_server
    log "---------- 完成: ${MODEL_NAME} ----------"
    echo ""
done

log "============================================"
log "全部评测完毕！结果文件: $RESULT_FILE"
log "============================================"
cat "$RESULT_FILE"
