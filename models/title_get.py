"""
hojin yang
for Spotify Recys Challenge
"""

from .title_models.Char_CNN import Char_CNN
from .title_models.Wide_Deep import WD_CNN
from .title_models.Wide import Wide_CNN
from .title_models.VDCNN2 import VDCNN2
from .title_models.LSTM import LSTM
from .title_models.MULTI_LSTM import MULTI_LSTM


def get_model(conf):
    conv_layers = [
        [conf.filter_num, 7, 3],
        [conf.filter_num, 7, 3],
        [conf.filter_num, 3, -1],
        [conf.filter_num, 3, -1],
        [conf.filter_num, 3, -1],
        [conf.filter_num, 3, 3]
    ]
    # fc_layers = [1024, 1024]
    fc_layers = [512, 256]
    # conv_layers = [
    #     [conf.filter_num, 3, -1],
    #     [conf.filter_num, 3, 3],
    #     [conf.filter_num, 3, -1],
    #     [conf.filter_num, 3, 3]
    # ]
    # For Wide and Deep
    wconv_layers = [
        [conf.filter_num, 3, -1],
        [conf.filter_num, 5, -1],
        [conf.filter_num, 6, -1],
        [conf.filter_num, 7, -1],
        # [conf.filter_num, 7, -1]
    ]
    rnn_layers = [256, 128]

    model = conf.char_model
    assert model in ['WIDE', 'LSTM', 'MULTI_LSTM']

    if model == 'WIDE':
        return Wide_CNN(conf, wconv_layers)
    elif model == 'LSTM':
        return LSTM(conf, fc_layers)
    elif model == 'MULTI_LSTM':
        return MULTI_LSTM(conf, fc_layers, rnn_layers)