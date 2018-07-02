import os
import argparse
import pickle
import pandas as pd

if __name__ == '__main__':
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--dir', type=str, default='challenge_results')

    args = args.parse_args()
    dir = "./"+args.dir

    total_cands = [['team_info', 'track', 'team_name', 'email@address.com']]
    for result in os.listdir(dir):
        result = dir + '/' + result
        with open(result, 'rb') as f:
            cand = pickle.load(f)
            total_cands += cand

    print("num_playlist: ", len(total_cands) - 1)
    print("num_rec: ", len(total_cands[1]) - 1)
    df = pd.DataFrame(total_cands)
    df.to_csv('results.csv', index=False, header=False)
