CUDA_VISIBLE_DEVICES=0,1 \
  python -m torch.distributed.launch --nproc_per_node 2 \
  --master_addr 127.0.0.2 --master_port 29501 \
  finetune.py