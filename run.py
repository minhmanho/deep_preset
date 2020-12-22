import os
import os.path as osp
import argparse
import glob
from dp import DeepPreset

def main():
    # For parsing commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, default="./data/content/", help='Content folder')
    parser.add_argument("--style", type=str, default="./data/style/", help='Reference folder')
    parser.add_argument("--out", type=str, default="./data/out/", help='Out folder')
    parser.add_argument("--ckpt", type=str, default="./models/dp_saved_weights.pth.tar", help='Checkpoint path')
    parser.add_argument("--size", type=str, default="512x512", help='Image size. Inputting 352x352 will activate preset prediction')
    parser.add_argument("--p", action='store_true', help='Activate it in case of only preset prediction needed. It will save your running time.')
    args = parser.parse_args()

    if not osp.isdir(args.out):
        os.makedirs(args.out)

    deep_preset = DeepPreset(args)
    img_names = [osp.basename(k) for k in glob.glob(args.content + "/*.jpg")]
    img_names.sort()

    for i, img_name in enumerate(img_names):
        print("{}/{}: {}".format(i+1, len(img_names), img_name))
        content_path = osp.join(args.content, img_name)
        style_path = osp.join(args.style, img_name)
        out_path = osp.join(args.out, img_name.replace('.jpg', '.png'))
        deep_preset.stylize(content_path, style_path, out_path)
    print("Done !")

main()
