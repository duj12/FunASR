CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python -m torch.distributed.launch --nproc_per_node 4 \
  --master_addr 127.0.0.5 --master_port 29505 \
  finetune.py