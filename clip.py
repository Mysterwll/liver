import argparse
import os
import torch
if __name__ == "__main__":
    # train self-supervised model 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:100'

    cmd = f"python main.py  --seed 42 --model 8  --loss 3"
    os.system(cmd)