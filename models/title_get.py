"""
hojin yang
for Spotify Recys Challenge
"""

from .title_models.Char_CNN import Char_CNN
from .title_models.Char_LSTM import Char_LSTM


def get_model(conf):
    model = conf.char_model
    assert model in ['Char_CNN', 'Char_LSTM']

    if model == 'Char_CNN':
        conv_layers = []
        # base
        # filter size = 100
        # filter num = 3,5,7,9
        for fs in conf.filter_size:
            conv_layers.append([conf.filter_num, fs, -1])

        return Char_CNN(conf, conv_layers)
    elif model == 'Char_LSTM':
        # bi
        fc_layers = [512, 256]
        return LSTM(conf, fc_layers)
