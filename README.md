# Deep Preset: Blending and Retouching Photos with Color Style Transfer
#### [[Project Page]](https://minhmanho.github.io/deep_preset/) [[Paper]](https://arxiv.org/abs/2007.10701)
#### [Man M. Ho](https://minhmanho.github.io/), [Jinjia Zhou](https://www.zhou-lab.info/jinjia-zhou)
![Alt Text](https://raw.githubusercontent.com/minhmanho/deep_preset/master/docs/images/intro_1.gif)
(To appear in WACV, 2021)

## Prerequisites
- Ubuntu 16.04
- Pillow
- [PyTorch](https://pytorch.org/) >= 1.1.0
- Numpy
- gdown (for fetching pretrained models)

## Get Started
### 1. Clone this repo
```
git clone https://github.com/minhmanho/deep_preset.git
cd deep_preset
```

### 2. Fetch our trained model
Positive Pair-wise Loss (PPL) could improve Deep Preset in directly stylizing photos; however, it became worse in predicting preset, as described in our paper.
Therefore, depending on your needs, please download [Deep Preset with PPL](https://drive.google.com/uc?id=1GegyHf3OD17k_WID3-vA7S8nRQwPfpTC) for directly stylizing photos

```
./models/fetch_model_wPPL.sh
```

Or [Deep Preset without PPL](https://drive.google.com/uc?id=1cSJpobfUP3hjNv-gGh3Cs9QT4keb9SeV) for preset prediction.

```
./models/fetch_model_woPPL.sh
```

## Blending and Retouching Photos
Run our Deep Preset to stylize photos as:
```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --content ./data/content/ \
    --style ./data/style/ \
    --out ./data/out/ \
    --ckpt ./models/dp_wPPL.pth.tar \
    --size 512x512 
```

Where `--size` is for the photo size _[Width]_x_[Height]_, which should be divisible by 16.
Besides, `--size` set as `352x352` will activate the preset prediction.

In case of only preset prediction needed, please add `--p` as:
```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --content ./data/content/ \
    --style ./data/style/ \
    --out ./data/out/ \
    --ckpt ./models/dp_woPPL.pth.tar \
    --p
```
After processing, the predicted preset will be stored as a JSON file revealing how Lightroom settings are adjusted, as follows:

```
{
    "Highlights2012": -23,
    "Shadows2012": 4,
    "BlueHue": -8, 
    "Sharpness": 19, 
    "Clarity2012": -2
    ...
}
```

## Cosplay Portraits
![Alt Text](https://minhmanho.github.io/deep_preset/images/cp.jpg)
Photos were taken by _Do Khang_ (taking the subject in the top-left one) and the first author (others).

## Citation
If you find this work useful, please consider citing:
```
@article{ho2020deep,
title={Deep Preset: Blending and Retouching Photos with Color Style Transfer},
author={Ho, Man M, and Zhou, Jinjia},
journal={arXiv preprint arXiv:2007.10701},
year={2020}
}
```

## Acknowledgements
We would like to thank:
- _Do Khang_ for the photos of [_Duyen To_](https://twitter.com/Jinnie0159).

- _digantamisra98_ for the [Unofficial PyTorch Implementation of EvoNorm ](https://github.com/digantamisra98/EvoNorm)
```
Liu, Hanxiao, Andrew Brock, Karen Simonyan, and Quoc V. Le. "Evolving Normalization-Activation Layers." 
arXiv preprint arXiv:2004.02967 (2020).
```
- and _Richard Zhang_ for the [BlurPool](https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py).
```
Zhang, Richard. "Making convolutional networks shift-invariant again." 
ICML (2019).
```

## License
Our code and trained models are for non-commercial uses and research purposes only.

## Contact
If you have any questions, feel free to contact me (maintainer) at [manminhho.cs@gmail.com](mailto:manminhho.cs@gmail.com)
