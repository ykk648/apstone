# -- coding: utf-8 --
# @Time : 2022/10/9
# @Author : ykk648
# @Project : https://github.com/ykk648/apstone
import torch


class TJMWrapper:
    def __init__(self, tjm_path, provider='gpu'):
        self.provider = provider
        self.model = torch.jit.load(tjm_path)
        self.model.eval()

        if self.provider == 'cpu':
            pass
        else:
            assert (torch.cuda.is_available())
            if self.provider == 'gpu':
                gpu_ids = [0]
            else:
                gpu_ids = self.provider
            self.model.to(gpu_ids[0])
            self.facenet = torch.nn.DataParallel(self.model, gpu_ids)  # multi-GPUs

    def forward(self, input_):
        if self.provider != 'cpu:':
            input_ = input_.cuda()
        with torch.no_grad():
            return self.model(input_)
