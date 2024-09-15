import argparse
from typing import *

class Argument:
    def __init__(self):
        parser = argparse.ArgumentParser()
        self.load_proprecess_args(parser)
        self.load_train_args(parser)
        self.load_model_args(parser)
        self.args = parser.parse_args()

    @staticmethod
    def load_proprecess_args(parser):
        parser.add_argument('--dataset', default='CiteSeer', help='dataset select')
        parser.add_argument('--CiteSeer_file', default='dataset/', type=str, help='CiteSeer dataset dict')
        parser.add_argument('--Cora_file', default='dataset/', type=str, help='Cora dataset dict')
        parser.add_argument('--PubMed_file', default='dataset/', type=str, help='PubMed dict')

        parser.add_argument('--seed', default=2024, type=int)
        parser.add_argument('--reconstruct_dataset', default=True, type=bool)
        
    @staticmethod
    def load_train_args(parser):
        parser.add_argument('--optimizer', default='adam')
        parser.add_argument('--loss', default='mse_loss')
        parser.add_argument('--epoch', default=100, type=int)
        parser.add_argument('--lr', default=0.003, type=float)
        parser.add_argument('--weight_decay', default=5e-4, type=float)

        parser.add_argument('--l_d', default=3.0, type=float)
        parser.add_argument('--l_z', default=0.2, type=float)

    @staticmethod
    def load_model_args(parser):
        parser.add_argument('--low_encode_indim', default=3704, type=int)
        parser.add_argument('--low_encode_outdim', default=64, type=int)

        parser.add_argument('--random_mask_rate', default=0.1, type=float)

        parser.add_argument('--a', default=10, type=int)
        parser.add_argument('--b', default=1, type=int)
        parser.add_argument('--c', default=1, type=int)


    def get_args(self):
        return self.args
