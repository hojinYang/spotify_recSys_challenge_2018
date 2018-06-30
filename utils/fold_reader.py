import random
import json
import numpy as np 
import time


class fold_reader:
    def __init__(self, data_dir, filename, batch_size):
        
        with open(data_dir+'/'+filename) as data_file:
            fold_tr = json.load(data_file)
        self.num_tracks = len(fold_tr['track_uri2id'])
        self.num_items = self.num_tracks + len(fold_tr['artist_uri2id'])
        self.max_title_len = fold_tr['max_title_len']
        self.num_char = fold_tr['num_char']
        self.playlists = fold_tr['playlists']

        del fold_tr
        self.batch_size = batch_size
        self.train_idx = 0
    
    def next_batch(self):
        trk_positions = []
        art_positions = []
        titles = []

        for i in range(self.batch_size):
            train_trk, train_art, train_title = self.playlists[self.train_idx]
            tmp = train_trk
            trks = np.array([tmp]).T
            playlist = np.full_like(trks, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, trks), axis=1)
            trk_positions.append(conc)
            
            tmp = train_art
            arts = np.array([tmp]).T
            playlist = np.full_like(arts, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, arts), axis=1)
            art_positions.append(conc)

            titles.append(train_title)
            self.train_idx += 1
            if self.train_idx == len(self.playlists):
                self.train_idx = 0
                random.shuffle(self.playlists)

        trk_positions = np.concatenate(trk_positions)
        art_positions = np.concatenate(art_positions)
        y_positions = np.concatenate((trk_positions, art_positions), 0)
        trk_val = [1] * len(trk_positions)
        art_val = [1] * len(art_positions)

        return trk_positions, art_positions, y_positions, titles, trk_val, art_val


class fold_reader_firstN:
    def __init__(self, data_dir, filename, batch_size, from_to):

        with open(data_dir + '/' + filename) as data_file:
            fold_tr = json.load(data_file)
        self.num_tracks = len(fold_tr['track_uri2id'])
        self.num_items = self.num_tracks + len(fold_tr['artist_uri2id'])
        self.max_title_len = fold_tr['max_title_len']
        self.num_char = fold_tr['num_char']
        self.playlists = fold_tr['playlists']

        del fold_tr
        self.batch_size = batch_size
        self.train_idx = 0
        self.from_to = from_to

    def next_batch(self):
        trk_positions = []
        art_positions = []

        trk_val = []
        art_val = []
        titles = []

        for i in range(self.batch_size):
            train_trk, train_art, train_title = self.playlists[self.train_idx]

            if len(train_trk) != 0:
                n = self.from_to[0]
                if n !=0:
                    n = int(len(train_trk) / n)
                m = min(len(train_trk), self.from_to[1])
                given_num = random.randrange(n + 1, m + 1)
                tmp = train_trk
                trks = np.array([tmp]).T
                playlist = np.full_like(trks, fill_value=i, dtype=np.int)
                conc = np.concatenate((playlist, trks), axis=1)
                trk_positions.append(conc)
                val = [1] * given_num + [0] * (len(train_trk) - given_num)
                trk_val += val

            if len(train_art) != 0:
                n = self.from_to[0]
                if n != 0:
                    n = int(len(train_art) / n)
                m = min(len(train_art), self.from_to[1])
                given_num = random.randrange(n + 1, m + 1)
                tmp = train_art
                arts = np.array([tmp]).T
                playlist = np.full_like(arts, fill_value=i, dtype=np.int)
                conc = np.concatenate((playlist, arts), axis=1)
                art_positions.append(conc)
                val = [1] * given_num + [0] * (len(train_art) - given_num)
                art_val += val

            titles.append(train_title)
            self.train_idx += 1
            if self.train_idx == len(self.playlists):
                self.train_idx = 0
                random.shuffle(self.playlists)

        trk_positions = np.concatenate(trk_positions)
        art_positions = np.concatenate(art_positions)

        y_positions = np.concatenate((trk_positions, art_positions), 0)
        return trk_positions, art_positions, y_positions, titles, trk_val, art_val


class fold_reader_test:
    def __init__(self, data_dir, filename, batch_size, test_num):
        print("now processing: " + filename)
        with open(data_dir + '/' + filename) as data_file:
            fold_te = json.load(data_file)
        self.playlists = fold_te['playlists'][:test_num]
        del fold_te
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

        test_titles = []

        # start_time = time.time()
        for i in range(self.batch_size):
            seed, seed_art, title, answer = self.playlists[self.test_idx]

            trks = np.array([seed]).T
            playlist = np.full_like(trks, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, trks), axis=1)
            trk_positions.append(conc)

            test_seed.append(seed)
            test_answer.append(answer)

            arts = np.array([seed_art]).T
            playlist = np.full_like(arts, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, arts), axis=1)
            art_positions.append(conc)
            
            test_titles.append(title)
            self.test_idx += 1
            if self.test_idx == len(self.playlists):
                self.test_idx = 0
                break
        # print(self.test_idx)
        trk_positions = np.concatenate(trk_positions)
        art_positions = np.concatenate(art_positions)
        x_positions = np.concatenate((trk_positions, art_positions), 0)
        x_ones = [1]*len(trk_positions) + [0.5]*len(art_positions)

        return x_positions, test_seed, test_answer, test_titles, x_ones
    
    
class fold_reader_challenge:
    def __init__(self, data_dir, filename, batch_size):
        print("now processing: " + filename)
        with open(data_dir + '/' + filename) as data_file:
            fold_ch = json.load(data_file)
        self.playlists = fold_ch['playlists']
        self.id2uri = fold_ch['id2uri']
        self.num_tracks = fold_ch['num_tracks']
        self.num_items = fold_ch['num_items']
        self.is_in_order = fold_ch['in_order']
        self.max_title_len = fold_ch['max_title_len']
        self.num_char = fold_ch['num_char']

        del fold_ch

        self.batch_size = batch_size
        self.ch_idx = 0

    def next_batch(self):
        trk_positions = []
        art_positions = []
        ch_seed = []
        ch_titles = []
        ch_titles_exist = []
        ch_pid = []

        # start_time = time.time()
        for i in range(self.batch_size):
            seed, seed_art, title, title_exist, pid = self.playlists[self.ch_idx]

            trks = np.array([seed]).T
            playlist = np.full_like(trks, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, trks), axis=1)
            trk_positions.append(conc)
            #trk_one_counter += len(seed)

            ch_seed.append(seed)

            arts = np.array([seed_art]).T
            playlist = np.full_like(arts, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, arts), axis=1)
            art_positions.append(conc)
            #art_one_counter+=len(seed_art)
            
            ch_titles.append(title)
            ch_titles_exist.append(title_exist)
            ch_pid.append(pid)
            
            self.ch_idx += 1
            if self.ch_idx == len(self.playlists):
                self.ch_idx = 0
                break

        trk_positions = np.concatenate(trk_positions)
        art_positions = np.concatenate(art_positions)
        x_positions = np.concatenate((trk_positions, art_positions), 0)

        x_ones = [1] * len(trk_positions) + [0.5] * len(art_positions)

        return x_positions, ch_seed, ch_titles, ch_titles_exist, ch_pid, x_ones
