# Deep Preset: Blending and Retouching Photos with Color Style Transfer (WACV'2021)
#### [[Page]](https://minhmanho.github.io/deep_preset/) [[Paper]](https://arxiv.org/abs/2007.10701) [[SupDoc]](https://openaccess.thecvf.com/content/WACV2021/supplemental/Ho_Deep_Preset_Blending_WACV_2021_supplemental.pdf) [[SupVid]](https://drive.google.com/file/d/1hF7clPr6jitjDRBCJCiMwTlYjDEknO8P/view?usp=sharing) [[5-min Presentation]](https://drive.google.com/file/d/1WHt3rPXd-FiUOj_Xnb7tQnQJoXAQY9zz/view?usp=sharing) [[Slides]](https://drive.google.com/file/d/1B4aaP-EWIC5zkd35yw-VXiSlWgkJ87o-/view?usp=sharing)
#### [Man M. Ho](https://minhmanho.github.io/), [Jinjia Zhou](https://www.zhou-lab.info/jinjia-zhou)
![Alt Text](https://raw.githubusercontent.com/minhmanho/deep_preset/master/docs/images/intro_1.gif)

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

## Regarding training data
I planned to share the dataset and Lightroom Add-ons publicly.
However, I don't have much time for it these days.
Uploading the whole dataset is inefficient since it is about **~500GB**.
You can download the training images with natural colors labeled as "0" at this [Google Drive](https://drive.google.com/file/d/1Lj_LIFNGCn5EqpE56peppGH4SKLb3fqL/view?usp=sharing)
Afterward, you will need to script Lightroom (or create a Plugin/Add-ons) to generate other styles.

*(You can refer to these ugly lines of code)*
```
local catalog = LrApplication.activeCatalog()
local photos = catalog:getTargetPhotos()
local catalog_folder = LrPathUtils.parent(catalog:getPath())

pfolder = "<folder containing json presets>"
outfolder = "<out folder>"
for i=1,500,1 do
    pdir = pfolder .. tostring(i) .. ".json"
    local file = io.open(pdir, 'r')
    if file then
        local contents = file:read( "*a" )
        p = json.decode(contents);
        io.close( file )
    else
        p = nil
    end

    for j, photo in ipairs(photos) do
        catalog:withWriteAccessDo ("Apply preset", function()
            local tmp = LrApplication.addDevelopPresetForPlugin( _PLUGIN, "Preset " .. tostring(i), p)
            photo:applyDevelopPreset (tmp, _PLUGIN)
        end)
    end
    local _out = outfolder .. tostring(i)
    LrFileUtils.createDirectory(_out)
    exportImage.exportPhotos(photos, _out)
```

Please check [Adobe Lightroom software development kit (SDK)](https://www.adobe.io/lightroom-classic/) for more details.

## Citation
If you find this work useful, please consider citing:
```
@InProceedings{Ho_2021_WACV,
    author    = {Ho, Man M. and Zhou, Jinjia},
    title     = {Deep Preset: Blending and Retouching Photos With Color Style Transfer},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {2113-2121}
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
