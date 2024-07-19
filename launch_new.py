import argparse
import os

if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="add arguments to start")
    parser.add_argument("--model", default='ROI_vision_only', type=str, help="model")
    parser.add_argument("--seed", default=42, type=int, help="seed given by LinkStart.py, to make sure programs are "
                                                             "working on the same cross val")
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--data_parallel", default=0, type=int)
    args = parser.parse_args()
    for i in range(5):
        cmd = f"python main_rebuild.py --model {args.model} --seed {args.seed} --fold {i} --data_parallel {args.data_parallel}"
        os.system(cmd)
