import argparse
import os

if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="add arguments to start")
    parser.add_argument("--model", default='RadioSA', type=str, help="model")
    parser.add_argument("--seed", default=42, type=int, help="seed given by LinkStart.py, to make sure programs are "
                                                             "working on the same cross val")
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--fold", default=0, type=int, help="0~4")
    args = parser.parse_args()
    for i in range(5):
        cmd = f"python main_rebuild.py --model {args.model} --seed {args.seed} --fold {i}"
        os.system(cmd)
