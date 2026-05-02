# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

workspace=`pwd`

# which gpu to train or finetune
export CUDA_VISIBLE_DEVICES="4,5,6,7"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# model_name from model_hub, or model_dir in local path

## option 1, download model automatically
model_name_or_model_dir="iic/SenseVoiceSmall"

## option 2, download model by git
#local_path_root=${workspace}/modelscope_models
#mkdir -p ${local_path_root}/${model_name_or_model_dir}
#git clone https://www.modelscope.cn/${model_name_or_model_dir}.git ${local_path_root}/${model_name_or_model_dir}
#model_name_or_model_dir=${local_path_root}/${model_name_or_model_dir}


# data dir, which contains: train.json, val.json
train_data=${workspace}/data/train_example.jsonl
val_data=${workspace}/data/val_example.jsonl

train_data=/data/megastore/SHARE/TTS/VoiceClone1/250Hours_zh/train/sensevoice_zh_en.jsonl
val_data=/data/megastore/SHARE/TTS/VoiceClone1/250Hours_zh/test/sensevoice.jsonl

train_data=/data/megastore/Datasets/ASR/jsonl/SenseVoice/train.list  # 135 files
train_data=/data/megastore/Datasets/ASR/jsonl/SenseVoice/finetune.list

val_data=/data/megastore/Datasets/ASR/jsonl/SenseVoice/test.list

# exp output dir
output_dir="./exp_ft_se"
log_file="${output_dir}/log.txt"

deepspeed_config=${workspace}/../../deepspeed_conf/ds_stage1.json

mkdir -p ${output_dir}
echo "log_file: ${log_file}"

DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node $gpu_num \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-36668}
"

echo $DISTRIBUTED_ARGS

# whether to enable denoise preprocessing (true/false)
enable_denoise=true
denoise_prob=0.5
# GPU for denoise: auto=one per training GPU via LOCAL_RANK, or set specific id (e.g. 0)
denoise_gpu=auto

# funasr trainer path
# batch_size=32000 for A100 80G, batch_size=16000 for 3090 24G

            # ++dataset_conf.preprocessor_speech=SpeechPreprocessAddNoiseReverb  \
            # ++dataset_conf.preprocessor_speech_conf.reverb_path=/data/megastore/Datasets/AudioData/Noise/RIRS_NOISES/rir.scp \
            # ++dataset_conf.preprocessor_speech_conf.noise_path=/data/megastore/Datasets/AudioData/Noise/WavNoise/noise.scp \

train_tool=../../../funasr/bin/train_ds.py

# denoise args
if [ "$enable_denoise" = true ]; then
    if [ "$denoise_gpu" = "auto" ]; then
        denoise_args="++dataset_conf.preprocessor_speech=SpeechPreprocessDenoise \
                      ++dataset_conf.preprocessor_speech_conf.denoise_prob=${denoise_prob}"
    else
        denoise_args="++dataset_conf.preprocessor_speech=SpeechPreprocessDenoise \
                      ++dataset_conf.preprocessor_speech_conf.denoise_prob=${denoise_prob} \
                      ++dataset_conf.preprocessor_speech_conf.denoise_gpu=${denoise_gpu}"
    fi
else
    denoise_args=""
fi

run_command() {
    torchrun $DISTRIBUTED_ARGS \
        ${train_tool} \
            ++model="${model_name_or_model_dir}" \
            ++train_data_set_list="${train_data}" \
            ++valid_data_set_list="${val_data}" \
            ++dataset_conf.batch_sampler="BatchSampler" \
            ++dataset_conf.batch_size=20000  \
            ++dataset_conf.sort_size=1024 \
            ++dataset_conf.batch_type="token" \
            ++dataset_conf.num_workers=1 \
            ++dataset_conf.max_source_length=4000 \
            ++dataset_conf.min_source_length=20 \
            ++dataset_conf.max_target_length=100 \
            ++dataset_conf.min_target_length=1 \
            ++dataset_conf.max_token_length=4100 \
            ++dataset_conf.data_split_num=1 \
            ++train_conf.max_epoch=60 \
            ++train_conf.log_interval=100 \
            ++train_conf.resume=true \
            ++train_conf.validate_interval=5000 \
            ++train_conf.save_checkpoint_interval=5000 \
            ++train_conf.keep_nbest_models=100 \
            ++train_conf.avg_keep_nbest_models_type="loss" \
            ++train_conf.avg_nbest_model=10 \
            ++train_conf.use_deepspeed=false \
            ++train_conf.deepspeed_config=${deepspeed_config} \
            ++optim_conf.lr=0.0002 \
            ${denoise_args} \
            ++output_dir="${output_dir}" #  2>&1 | tee -a ${log_file}
}

# 循环运行
while true; do
    echo "Starting the command..."
    run_command
    if [ $? -eq 0 ]; then
        echo "Command executed successfully."
        break
    else
        echo "An error occurred. Retrying..."
        sleep 5  # 等待5秒后重试
    fi
done