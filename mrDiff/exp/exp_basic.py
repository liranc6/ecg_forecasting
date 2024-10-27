import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        torch.cuda.empty_cache()

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.hardware.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.hardware.gpu) if not self.args.hardware.use_multi_gpu else self.args.hardware.devices
            device = torch.device('cuda:{}'.format(self.args.hardware.gpu))
            print('Use GPU: cuda:{}'.format(self.args.hardware.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
