import random
import json
import numpy as np 
import time


class data_reader:
    def __init__(self, data_dir, filename, batch_size):
        
        with open(data_dir+'/'+filename) as data_file:
            data_tr = json.load(data_file)
        self.num_tracks = len(data_tr['track_uri2id'])
        self.num_items = self.num_tracks + len(data_tr['artist_uri2id'])
        self.max_title_len = data_tr['max_title_len']
        self.num_char = data_tr['num_char']
        self.playlists = data_tr['playlists']
        self.class_divpnt = data_tr['class_divpnt']

        del data_tr
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


class data_reader_firstN:
    def __init__(self, data_dir, filename, batch_size, from_to):

        with open(data_dir + '/' + filename) as data_file:
            data_tr = json.load(data_file)
        self.num_tracks = len(data_tr['track_uri2id'])
        self.num_items = self.num_tracks + len(data_tr['artist_uri2id'])
        self.max_title_len = data_tr['max_title_len']
        self.num_char = data_tr['num_char']
        self.playlists = data_tr['playlists']

        del data_tr
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
            len_t = len(train_trk)
            if len_t != 0:
                if self.from_to[0] >= 1:
                    n = int(self.from_to[0])
                    m = int(min(len_t, self.from_to[1]))
                else:
                    n = int(max(len_t * self.from_to[0], 1))
                    m = int(max(len_t * self.from_to[1], 1))
                given_num = random.randrange(n, m + 1)
                tmp = train_trk
                trks = np.array([tmp]).T
                playlist = np.full_like(trks, fill_value=i, dtype=np.int)
                conc = np.concatenate((playlist, trks), axis=1)
                trk_positions.append(conc)
                val = [1] * given_num + [0] * (len_t - given_num)
                trk_val += val

            len_a = len(train_art)
            if len_a != 0:
                if self.from_to[0] >= 1:
                    n = int(self.from_to[0])
                    m = int(min(len_a, self.from_to[1]))
                else:
                    n = int(max(len_a * self.from_to[0], 1))
                    m = int(max(len_a * self.from_to[1], 1))

                given_num = random.randrange(n, m + 1)
                tmp = train_art
                arts = np.array([tmp]).T
                playlist = np.full_like(arts, fill_value=i, dtype=np.int)
                conc = np.concatenate((playlist, arts), axis=1)
                art_positions.append(conc)
                val = [1] * given_num + [0] * (len_a - given_num)
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
        answers_for_grad = []
        # art_positions = []

        test_seed = []
        test_answer = []
        test_answer_cls = []

        # start_time = time.time()
        for i in range(self.batch_size):
            seed, seed_art, answer, seed_cls, answer_cls = self.playlists[self.test_idx]

            trks = np.array([seed], dtype=np.int64).T
            playlist = np.full_like(trks, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, trks), axis=1)
            trk_positions.append(conc)

            test_seed.append(seed)
            test_answer.append(answer)
            test_answer_cls.append(answer_cls)

            answer_for_grad = seed[:]
            for a in answer:
                if a != -1:
                    answer_for_grad.append(i)
            trks = np.array([answer_for_grad], dtype=np.int64).T
            playlist = np.full_like(trks, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, trks), axis=1)
            answers_for_grad.append(conc)

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
        answers_for_grad = np.concatenate(answers_for_grad)
        # art_positions = np.concatenate(art_positions)
        # x_positions = np.concatenate((trk_positions, art_positions), 0)
        # x_ones = [1]*len(trk_positions) + [0.5]*len(art_positions)

        return trk_positions, test_seed, test_answer, test_answer_cls, answers_for_grad

    def next_batch_test_cls(self, cls_list):
        trk_positions = []
        answers_for_grad = []
        # art_positions = []

        test_seed = []
        test_answer = []
        test_answer_cls = []

        test_titles = []

        # start_time = time.time()
        for i in range(self.batch_size):
            seed, seed_art, answer, seed_cls, answer_cls = self.playlists[self.test_idx]

            _seed = []
            for c, s in zip(seed_cls, seed):
                if c in cls_list:
                    _seed.append(s)
            seed = _seed

            trks = np.array([seed], dtype=np.int64).T
            playlist = np.full_like(trks, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, trks), axis=1)
            trk_positions.append(conc)

            test_seed.append(seed)
            test_answer.append(answer)
            test_answer_cls.append(answer_cls)

            answer_for_grad = test_seed[:]
            for i in test_answer:
                if i != -1:
                    answer_for_grad.append(i)
            trks = np.array([answer_for_grad], dtype=np.int64).T
            playlist = np.full_like(trks, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, trks), axis=1)
            answers_for_grad.append(conc)


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
<<<<<<< HEAD
        art_positions = np.concatenate(art_positions)
        x_positions = np.concatenate((trk_positions, art_positions), 0)
        x_ones = [1]*len(trk_positions) + [0.5]*len(art_positions)

        return x_positions, test_seed, test_answer, test_titles, x_ones
    
    
class data_reader_challenge:
    def __init__(self, data_dir, filename, batch_size):
        print("now processing: " + filename)
        with open(data_dir + '/' + filename) as data_file:
            data_ch = json.load(data_file)
        self.playlists = data_ch['playlists']
        self.id2uri = data_ch['id2uri']
        self.num_tracks = data_ch['num_tracks']
        self.num_items = data_ch['num_items']
        self.is_in_order = data_ch['in_order']
        self.max_title_len = data_ch['max_title_len']
        self.num_char = data_ch['num_char']

        del data_ch

        self.batch_size = batch_size
        self.ch_idx = 0

    def next_batch(self):
        trk_positions = []
        trk_ones = []
        art_positions = []
        ch_seed = []
        ch_titles = []
        ch_titles_exist = []
        ch_pid = []

        # start_time = time.time()
        for i in range(self.batch_size):
            seed, seed_art, title, title_exist, pid = self.playlists[self.ch_idx]
            len_s = len(seed)
            if (len_s > 50) and self.is_in_order:
                trk_ones += [0.15]*(len_s-15) + [1.0]*15
            else:
                trk_ones += [1.0] * len_s

            trks = np.array([seed]).T
            playlist = np.full_like(trks, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, trks), axis=1)
            trk_positions.append(conc)
            ch_seed.append(seed)

            arts = np.array([seed_art]).T
            playlist = np.full_like(arts, fill_value=i, dtype=np.int)
            conc = np.concatenate((playlist, arts), axis=1)
            art_positions.append(conc)
            
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

        x_ones = trk_ones + [0.5] * len(art_positions)

        return x_positions, ch_seed, ch_titles, ch_titles_exist, ch_pid, x_ones
=======
        answers_for_grad = np.concatenate(answers_for_grad)
        # art_positions = np.concatenate(art_positions)
        # x_positions = np.concatenate((trk_positions, art_positions), 0)
        # x_ones = [1]*len(trk_positions) + [0.5]*len(art_positions)

        return trk_positions, test_seed, test_answer, test_answer_cls, answers_for_grad
>>>>>>> e699f7b47d3f20b6e90a780eb2af4224ea75800b
