import random
import json
import numpy as np 
import time


class data_reader:
    def __init__(self, data_dir, filename, batch_size):
        
        with open(data_dir+'/'+filename) as data_file:
            data_tr = json.load(data_file)
        self.num_tracks = len(data_tr['track_uri2id'])
        # self.num_items = self.num_tracks + len(data_tr['artist_uri2id'])
        self.playlists = data_tr['playlists']

        del data_tr
        self.batch_size = batch_size
        self.train_idx = 0
    
    def next_batch(self):
        trk_positions = []
        # art_positions = []

        for i in range(self.batch_size):
            train_trk, train_art = self.playlists[self.train_idx]
            tmp = train_trk
            trks = np.array([tmp]).T
            playlist = np.full_like(trks, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, trks), axis=1)
            trk_positions.append(conc)
            
            '''
            tmp = train_art
            arts = np.array([tmp]).T
            playlist = np.full_like(arts, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, arts), axis=1)
            art_positions.append(conc)
            '''

            self.train_idx += 1
            if self.train_idx == len(self.playlists):
                self.train_idx = 0
                random.shuffle(self.playlists)

        trk_positions = np.concatenate(trk_positions)
        '''
        art_positions = np.concatenate(art_positions)
        y_positions = np.concatenate((trk_positions, art_positions), 0)
        trk_val = [1] * len(trk_positions)
        art_val = [1] * len(art_positions)
        '''

        return trk_positions


class data_reader_test:
    def __init__(self, data_dir, filename, batch_size, test_num):
        print("now processing: " + filename)
        with open(data_dir + '/' + filename) as data_file:
            data_te = json.load(data_file)
        self.playlists = data_te['playlists'][:test_num]
        del data_te
        test_num = test_num
        if test_num > len(self.playlists):
            test_num = len(self.playlists)
            print("the number of test will be changed to %d" % test_num)

        self.batch_size = batch_size
        self.test_idx = 0

    def next_batch_test(self):
        trk_positions = []
        art_positions = []

        test_seed = []
        test_answer = []

        # start_time = time.time()
        for i in range(self.batch_size):
            seed, seed_art, title, answer = self.playlists[self.test_idx]

            trks = np.array([seed]).T
            playlist = np.full_like(trks, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, trks), axis=1)
            trk_positions.append(conc)

            test_seed.append(seed)
            test_answer.append(answer)

            '''
            arts = np.array([seed_art]).T
            playlist = np.full_like(arts, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, arts), axis=1)
            art_positions.append(conc)
            '''

            self.test_idx += 1
            if self.test_idx == len(self.playlists):
                self.test_idx = 0
                break
        # print(self.test_idx)
        trk_positions = np.concatenate(trk_positions)
        # art_positions = np.concatenate(art_positions)
        # x_positions = np.concatenate((trk_positions, art_positions), 0)
        # x_ones = [1]*len(trk_positions) + [0.5]*len(art_positions)

        return trk_positions, test_seed, test_answer
