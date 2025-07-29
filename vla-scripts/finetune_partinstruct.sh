export LD_LIBRARY_PATH=/home/pai/envs/openvla/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

GPUS_PER_NODE=1  
NNODES=1
RANK=${RANK:-0}

torchrun --standalone --nnodes ${NNODES} --nproc-per-node ${GPUS_PER_NODE} finetune_partinstruct.py \
  --batch_size 4 \
  --grad_accumulation_steps 2 \
  --max_steps 10000 \
  --save_steps 2500 \
  --window_size 10 \
  --run_root_dir "./part-instruct-log"
