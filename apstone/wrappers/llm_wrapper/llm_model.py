# -- coding: utf-8 --
# @Time : 2023/5/6
# @Author : ykk648
# @Project : https://github.com/ykk648/apstone

class LLM:
    def __init__(self, model_info, load_in_8bit, device_map):
        self.model_path = model_info['model_path']
        if 'self.config' not in locals():
            self.config = model_info['config'].from_pretrained(self.model_path, return_unused_kwargs=True, trust_remote_code=True)[0]
        self.model = model_info['model'].from_pretrained(self.model_path, trust_remote_code=True,
                                                         load_in_8bit=load_in_8bit, device_map=device_map())
        self.tokenizer = model_info['tokenizer'].from_pretrained(self.model_path, trust_remote_code=True)

    def generate(self):
        pass