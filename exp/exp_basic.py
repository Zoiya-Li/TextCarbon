import torch
import torch.nn as nn
import os
import sys

# 将TimeXer模块导入为一个命名空间
import models.TimeXer as TimeXer

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        # 导入CrossformerWithText模块
        from models.CrossformerWithText import CrossformerWithText
        from models.FEDformerWithText import FEDformerWithText
        
        self.model_dict = {
            'TimeXer': TimeXer,
            'CrossformerWithText': type('CrossformerModule', (), {'Model': CrossformerWithText}),
            'FEDformerWithText': type('FEDformerModule', (), {'Model': FEDformerWithText}),
        }
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.gpu_type == 'cuda':
                if torch.cuda.is_available():
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                    device = torch.device('cuda:{}'.format(self.args.gpu))
                    print(f'使用 CUDA GPU: {self.args.gpu}')
                    print(f'CUDA设备数量: {torch.cuda.device_count()}')
                    print(f'当前CUDA设备: {torch.cuda.get_device_name(self.args.gpu)}')
                else:
                    print('警告: CUDA不可用，回退到CPU')
                    device = torch.device('cpu')
            elif self.args.gpu_type == 'mps':
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = torch.device('mps')
                    print('使用 MPS GPU')
                else:
                    print('警告: MPS不可用，回退到CPU')
                    device = torch.device('cpu')
            else:
                device = torch.device('cpu')
                print('使用 CPU')
        else:
            device = torch.device('cpu')
            print('使用 CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass 