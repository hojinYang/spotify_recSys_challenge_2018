import os
import argparse
import pickle
import pandas as pd
# import tensorflow as tf

if __name__ == '__main__':
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--dir', type=str, default='challenge_pkl')

    args = args.parse_args()
    dir = "./"+args.dir

    total_cands = [['team_info', 'main', 'hello world!', 'hojinyang7@skku.edu']]
    for result in  os.listdir(dir):
        result = dir + '/' + result
        with open(result,'rb') as f:
            cand = pickle.load(f)
            total_cands += cand

    print("num_playlist: ", len(total_cands) - 1)
    print("num_rec: ", len(total_cands[0]) - 1)
    df = pd.DataFrame(total_cands)
    df.to_csv('results.csv', index=False, header=False)
