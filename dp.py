import torch
import torch.nn as nn

import numpy as np
from PIL import Image
from utils import *
from networks.network import get_model, PRESET_PREDICTION_IMG_SIZE

class ToTensor(object):
    def __call__(self, tmp):
        tmp = tmp / 255.0
        tmp = (tmp - 0.5)/0.5
        tmp = tmp.transpose((2, 0, 1))
        return torch.from_numpy(tmp).unsqueeze(0).float()

class DeepPreset(object):
    def __init__(self, args):
        ckpt = torch.load(args.ckpt)

        # Load model
        self.G = get_model(ckpt['opts'].g_net)(ckpt['opts']).cuda()
        self.G.load_state_dict(ckpt['G'])
        self.G.eval()

        self.totensor = ToTensor()
        self.preset_handler = PresetHandler()

        self.img_size = size_str2tuple(args.size)
        self.p_only = args.p
        if self.p_only:
            self.img_size = PRESET_PREDICTION_IMG_SIZE

    def stylize(self, content_path, style_path, out_path):
        with torch.no_grad():
            pil_cont = Image.open(content_path)
            pil_style = Image.open(style_path)
            if self.img_size[0] != -1:
                pil_cont = pil_cont.resize(self.img_size, resample=Image.BICUBIC)
                pil_style = pil_style.resize(self.img_size, resample=Image.BICUBIC)
            else:
                pil_style = pil_style.resize(pil_cont.size, resample=Image.BICUBIC)
            content = self.totensor(np.array(pil_cont)).cuda()
            style = self.totensor(np.array(pil_style)).cuda()
            img_out, preset_out, preset_emb = self.G.stylize(content, style, self.img_size == PRESET_PREDICTION_IMG_SIZE, preset_only=self.p_only)
            if preset_out is not None:
                preset_out = preset_out[0].cpu().numpy()
                preset_emb = preset_emb[0].cpu().numpy()
                if out_path is not None:
                    self.preset_handler.save_numpy_preset(out_path.replace('.png', '.json'), preset_out)
                if self.p_only:
                    return img_out, preset_out, preset_emb

            # To CPU
            content = content.cpu()
            style   = style.cpu()
            img_out = img_out.cpu()

            # Save results
            img_out = (img_out + 1)/2
            img_out = np.array(img_out[0,:,:,:].clamp(0,1).numpy().transpose(1,2,0) * 255.0, dtype=np.uint8)

            if out_path is not None:
                Image.fromarray(img_out).save(out_path)

        return img_out, preset_out, preset_emb
