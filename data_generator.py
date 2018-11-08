"""
Generate training set, test set and challenge set.
Convert the MPD's data format into our models'.
"""
import sys
import json
import os
import numpy as np
import argparse
from utils.spotify_reader import *


def fullpaths_generator(path):
    filenames = os.listdir(path)
    fullpaths = []
    for filename in filenames:
        fullpath = os.sep.join((path, filename))
        fullpaths.append(fullpath)
    return fullpaths


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--datadir', type=str, default='./data', help="directory where the outputs are stored")
    args.add_argument('--mpd_tr', type=str, default='./mpd_train', help="train mpd path")
    args.add_argument('--mpd_te', type=str, default='./mpd_test', help="test mpd path")
    args.add_argument('--mincount_trk', type=int, default=5, help='minimum count of tracks')
    args.add_argument('--mincount_art', type=int, default=3, help='minimum count of artists')
    args = args.parse_args()

    train_fullpaths = fullpaths_generator(args.mpd_tr)
    train_fold = Spotify_train(train_fullpaths, args.mincount_trk, args.mincount_art, args.datadir)

    if args.mpd_te != 'NULL':
        test_fullpaths = fullpaths_generator(args.mpd_te)

        for test_seed_num in [1, 5, 10, 25, 50, 100]:
            is_shuffle = True
            test_fold = Spotify_test(test_fullpaths, args.datadir+'/train', test_seed_num, args.datadir, is_shuffle)
            del test_fold
