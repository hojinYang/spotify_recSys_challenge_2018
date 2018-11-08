"""
run main_train.py or main_challenge.py

"""

import os
import argparse
import configparser
from main_runner import main_train, main_challenge


class Conf:
    def __init__(self, dir, ini):
        self.dir = dir
        self.ini = ini
<<<<<<< HEAD
        self.data_dir = self.ini.get('BASE','data_dir')
        self.result_dir = self.ini.get('BASE','result_dir')
        self.testsize = int(self.ini.get('BASE', 'testsize'))
        self.verbose = bool(self.ini.get('BASE','verbose'))

    def set_dae_conf(self):
        self.epochs = int(self.ini.get('DAE','epochs'))
        self.batch = int(self.ini.get('DAE','batch'))
        self.lr = float(self.ini.get('DAE', 'lr'))
        self.reg_lambda = float(self.ini.get('DAE','reg_lambda'))
        test_seed = self.ini.get('DAE', 'test_seed')
        self.test_seed = ['test-'+item for item in test_seed.split(',')]
        update_seed = self.ini.get('DAE', 'update_seed')
        self.update_seed = ['test-'+item for item in update_seed.split(',')]
        input_kp = self.ini.get('DAE','input_kp')
        self.input_kp = [float(item) for item in input_kp.split(',')]
        self.kp = float(self.ini.get('DAE', 'keep_prob'))
        firstN = self.ini.get('DAE','firstN_range')
        self.firstN = [float(item) for item in firstN.split(',')]
        if len(self.firstN) == 1:
            assert self.firstN[0] == -1.0
        else:
            assert self.firstN[0] <= self.firstN[1]
            if self.firstN[1] < 1:
                assert self.firstN[0] == 0 or self.firstN[0].is_integer() is False
            else:
                assert self.firstN[0] >= 1
                assert self.firstN[0].is_integer() is True and self.firstN[1].is_integer() is True
        self.initval = os.path.join(self.dir, self.ini.get('DAE', 'initval'))
        self.save = os.path.join(self.dir, self.ini.get('DAE', 'save'))
        self.hidden = int(self.ini.get('DAE', 'hidden'))
        self.mode = 'dae'

    def set_pretrain_conf(self):
        self.epochs = int(self.ini.get('PRETRAIN','epochs'))
        self.batch = int(self.ini.get('PRETRAIN','batch'))
        self.lr = float(self.ini.get('PRETRAIN', 'lr'))
        self.reg_lambda = float(self.ini.get('PRETRAIN','reg_lambda'))
        self.is_pretrain = True
        self.save = os.path.join(self.dir, self.ini.get('PRETRAIN', 'save'))
        self.mode = 'pretrain'

    def set_title_conf(self):
        self.epochs = int(self.ini.get('TITLE', 'epochs'))
        self.batch = int(self.ini.get('TITLE','batch'))
        self.lr = float(self.ini.get('TITLE', 'lr'))
        input_kp = self.ini.get('TITLE','input_kp')
        self.input_kp = [float(item) for item in input_kp.split(',')]
        self.title_kp = self.ini.get('TITLE', 'title_kp')
        test_seed = self.ini.get('TITLE', 'test_seed')
        self.test_seed = ['test-' + item for item in test_seed.split(',')]
        update_seed = self.ini.get('TITLE', 'update_seed')
        self.update_seed = ['test-' + item for item in update_seed.split(',')]
        self.char_emb = int(self.ini.get('TITLE','char_emb'))
        self.char_model = self.ini.get('TITLE','char_model')

        if self.char_model == 'Char_CNN':
            self.filter_num = int(self.ini.get('TITLE', 'filter_num'))
            filter_size = self.ini.get('TITLE','filter_size')
            self.filter_size = [int(item) for item in filter_size.split(',')]

        elif self.char_model == 'Char_LSTM':
            self.rnn_hidden = int(self.ini.get('TITLE', 'rnn_hidden'))
            self.bi = bool(self.ini.get('TITLE', 'bi'))

        self.DAEval = os.path.join(self.dir, self.ini.get('TITLE', 'DAEval'))
        self.save = os.path.join(self.dir, self.ini.get('TITLE', 'save'))
        if not os.path.isdir(self.save):
            os.makedirs(self.save)
            os.rmdir(self.save)
        self.mode = 'title'

    def set_challenge_oonf(self):
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

        self.challenge_data = self.ini.get('CHALLENGE', 'challenge_data')
        self.result = os.path.join(self.result_dir, self.ini.get('CHALLENGE', 'result'))
        self.batch = int(self.ini.get('CHALLENGE', 'batch'))
=======
        self.data_dir = self.ini.get('PARAM','data_dir')
        self.testsize = int(self.ini.get('PARAM', 'testsize'))
        self.verbose = bool(self.ini.get('PARAM','verbose'))

        self.epochs = int(self.ini.get('PARAM','epochs'))
        self.batch = int(self.ini.get('PARAM','batch'))
        self.lr = float(self.ini.get('PARAM', 'lr'))

        test_seed = self.ini.get('PARAM', 'test_seed')
        self.test_seed = ['test-'+item for item in test_seed.split(',')]
        input_kp = self.ini.get('PARAM','input_kp')
        self.input_kp = [float(item) for item in input_kp.split(',')]
        self.hidden_kp = float(self.ini.get('PARAM', 'hidden_kp'))
        self.n_hidden = int(self.ini.get('PARAM', 'n_hidden'))

        self.save = os.path.join(self.dir, self.ini.get('PARAM', 'save'))
>>>>>>> e699f7b47d3f20b6e90a780eb2af4224ea75800b


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--dir', type=str, default='qwerty',help="directory name which contains config file")
    args.add_argument('--pretrain', action='store_true', default=False, help="pretrain mode if Specified")
    args.add_argument('--dae', action='store_true', default=False, help="DAE mode if Specified")
    args.add_argument('--title', action='store_true', default=False, help="title mode if Specified")
    args.add_argument('--challenge', action='store_true', default=False, help="challenge mode if Specified")
    args.add_argument('--testmode', action='store_true', default=False, help="test mode if Specified(just check the result)")

    args = args.parse_args()
    dir = os.path.join(".",args.dir)

    if not os.path.isdir(dir):
        print("ERROR: Cannot find "+dir+" ->Create directory and config.ini file first")
        exit(0)
    if 'config.ini' not in os.listdir(dir):
        print("ERROR: Cannot find config.ini in " +dir+" ->Create config.ini file in the directory first")
        exit(0)

    ini_dir = os.path.join(dir,'config.ini')
    ini = configparser.ConfigParser()
    ini.read(ini_dir)
    conf = Conf(dir, ini)

<<<<<<< HEAD
    conf.set_dae_conf()
    if args.pretrain:
        conf.set_pretrain_conf()
        main_train.run(conf, args.testmode)

    elif args.dae:
        conf.set_dae_conf()
        main_train.run(conf, args.testmode)

    elif args.title:
        conf.set_title_conf()
        main_train.run(conf, args.testmode)

    elif args.challenge:
        conf.set_title_conf()
        conf.set_challenge_oonf()
        main_challenge.run(conf)
=======
    main_train.run(conf, args.testmode)
>>>>>>> e699f7b47d3f20b6e90a780eb2af4224ea75800b
