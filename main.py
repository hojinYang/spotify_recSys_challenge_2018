"""
run main_train.py or main_challenge.py

"""

import os
import argparse
import configparser
from main_runner import main_train

class Conf:
    def __init__(self, dir, ini):
        self.dir = dir
        self.ini = ini
        self.data_dir = self.ini.get('BASE','data_dir')
        self.testsize = int(self.ini.get('BASE', 'testsize'))
        self.verbose = bool(self.ini.get('BASE','verbose'))

        self.epochs = int(self.ini.get('BASE','epochs'))
        self.batch = int(self.ini.get('BASE','batch'))
        self.lr = float(self.ini.get('BASE', 'lr'))

        test_seed = self.ini.get('BASE', 'test_seed')
        self.test_seed = ['test-'+item for item in test_seed.split(',')]
        input_kp = self.ini.get('BASE','input_kp')
        self.input_kp = [float(item) for item in input_kp.split(',')]
        self.hidden_kp = float(self.ini.get('BASE', 'keep_prob'))
        self.n_hidden = int(self.ini.get('BASE', 'hidden'))

        self.save = os.path.join(self.dir, self.ini.get('BASE', 'save'))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--dir', type=str, default='qwerty',help="directory name which contains config file")
    args.add_argument('--testmode', action='store_true', default=False, help="testmode if specified(just check the result)")

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

    conf.set_pretrain_conf()
    main_train.run(conf, args.testmode)