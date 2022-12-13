# -- coding: utf-8 --
# @Time : 2022/6/24
# @Author : ykk648
# @Project : https://github.com/ykk648/apstone
import torch

torch_model_path = "/hmr_ava.pt"
model = torch.load(torch_model_path)['model']
print(model.keys())
# torch.save(model, 'model.pt')

new_model = model.copy()
for key in model.keys():
    # print(key)
    new_model.pop(key)
    new_model[key.replace('smplx_', '')] = model[key]

print(new_model.keys())

torch.save(new_model, '')
