import json
import numpy as np
import copy
import torch
from torch import nn

def size_str2tuple(_str):
    out = [int(k) for k in _str.split("x")]
    assert len(out) == 2, "Unknown {}. The size should have been [width]x[height].".format(_str)
    return tuple(out)

class PresetHandler:
    def __init__(self, base_p_dir='./data/base_presets.json'):
        base_settings = self.json_load(base_p_dir)
        self.p_default = base_settings['p_default']
        self.keys = base_settings['keys']
        self.max_bound_np = np.array([base_settings['max'][k] for k in self.keys])
        self.min_bound_np = np.array([base_settings['min'][k] for k in self.keys])

    @staticmethod
    def json_load(_dir):
        with open(_dir) as json_file:
            content = json.load(json_file)
        return content

    def unnorm_preset(self, p_np):
        out = (p_np + 1)/2*(self.max_bound_np-self.min_bound_np) + self.min_bound_np
        return out.round().astype(np.int16).tolist()

    @staticmethod
    def int2bool(_num):
        return True if _num > 0.5 else False

    def save_numpy_preset(self, out_dir, p_np):
        p_list = self.unnorm_preset(p_np)
        p_base = copy.deepcopy(self.p_default)
        p_dict = {k:p_list[i] for i,k in enumerate(self.keys)}
        for k in p_dict:
            p_dict[k] = self.int2bool(p_dict[k]) if type(p_base[k]) == bool else p_dict[k]

        p_base.update(p_dict)
        with open(out_dir, 'w') as outfile:
            json.dump(p_base, outfile)

    def norm_preset(self, p):
        tmp = [p[k] for k in p if k in p_handler.keys()]
        assert len(tmp.keys()) == 69
        for k in tmp:
            tmp[k] = 2*(tmp[k]-p_handler.min_bound_np[k])/(p_handler.max_bound_np[k]-p_handler.min_bound_np[k]) - 1
        return tmp
