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
            # self.model = torch.nn.DataParallel(self.model, gpu_ids)  # multi-GPUs

    def forward(self, input_list_):
        """
        Args:
            input_list_: input1 or [input1, input2, ...]
        Returns:
        """
        if not isinstance(input_list_, list):
            input_list_ = [input_list_]
        if self.provider != 'cpu:':
            for index, input_ in enumerate(input_list_):
                input_list_[index] = input_list_[index].cuda()
        with torch.no_grad():
            return self.model(*input_list_)
