import argparse
import os

if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="add arguments to start")
    parser.add_argument("--name", default='newversion', type=str, help="title")
    parser.add_argument("--seed", default=42, type=int, help="start seed")
    args = parser.parse_args()
    for i in range(5):
        cmd = f"python train.py --name {args.name} --seed {args.seed + i}"
        os.system(cmd)