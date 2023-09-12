CUDA_VISIBLE_DEVICES=2,3 \
  python -m torch.distributed.launch --nproc_per_node 2 \
  --master_addr 127.0.0.3 --master_port 29502 \
  finetune.py