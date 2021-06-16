import argparse


class BaseParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--imgdir', required=True)
        self.parser.add_argument('--target_class', default=6, type=int)
        self.parser.add_argument('--batchSize', type=int, default=1)
        self.parser.add_argument('--dec_factor', type=float, default=0.98)
        self.parser.add_argument('--val_c', type=float, default=3)
        self.parser.add_argument('--val_w1', type=float, default=1e-2)
        self.parser.add_argument('--val_w2', type=float, default=1e-4)
        self.parser.add_argument('--val_gamma', default=0.8, type=float)
        self.parser.add_argument('--max_update', default=200, type=int)
        self.parser.add_argument('--max_epsilon', default=0.05, type=float)
        self.parser.add_argument('--maxiter', default=100, type=int)
        self.parser.add_argument('--name', type=str, default='homotopy')

    def parse(self):
        args = self.parser.parse_args()
        self.args = args
        return self.args

