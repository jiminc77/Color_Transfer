from color_matcher import ColorMatcher
from color_matcher.io_handler import load_img_file, save_img_file, FILE_EXTS
from color_matcher.normalizer import Normalizer
from PIL import Image   
import os
import argparse
import numpy as np


def run_bulk(config, progress_callback = None):

    fname_c = os.listdir(config.style)[0]
    img_ref = load_img_file(os.path.join(config.style, fname_c))

    filenames = [os.path.join(config.content, f) for f in os.listdir(config.content)
                        if f.lower().endswith(FILE_EXTS)]

    cm = ColorMatcher()
    for fname in filenames:
        img_src = load_img_file(fname)
        img_res = cm.transfer(src=img_src, ref=img_ref, method='reinhard') # hm-mvgd-hm or reinhard
        img_res = Normalizer(img_res).uint8_norm()
        print(os.path.join(os.path.dirname(fname)))
        save_img_file(img_res, os.path.join(config.output, 'output.png'))

    for filename in os.listdir(config.content):
        if filename.endswith(".jpg") or filename.endswith(".png"): 
            image_path_A = os.path.join(config.content, filename)
            image_path_B = os.path.join(config.output, 'output.png')
            image_A = Image.open(image_path_A)
            image_B = Image.open(image_path_B)
            image_np_A = np.array(image_A)
            image_np_B = np.array(image_B)
            
            for i in range(config.max_img_num):
                t = config.max_img_num
                x = i/t
                new_image_np = image_np_A * (1-x) + image_np_B * (x)
                new_image_np = np.clip(new_image_np, 0, 255).astype(np.uint8)
                new_image = Image.fromarray(new_image_np)
                output_image_path = os.path.join(config.output_list, f'{i}.png')
                new_image.save(output_image_path)
                progress = (i+1) / t
                progress_callback(progress)

def DeleteAllFiles(filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file.path)
        return "Remove All File"
    else:
        return "Directory Not Found"

def run(progress_callback = None):
    
    if not os.path.exists(os.path.join(config.output)):
        os.makedirs(os.path.join(config.output))

    if not os.path.exists(os.path.join(config.output_list)):
        os.makedirs(os.path.join(config.output_list))
    
    run_bulk(config, progress_callback)
                    

parser = argparse.ArgumentParser()
parser.add_argument('--content', type=str, default='./examples/content')
parser.add_argument('--style', type=str, default='./examples/style')
parser.add_argument('--output', type=str, default='./outputs')
parser.add_argument('--output_list', type=str, default='./outputs_list')
parser.add_argument('--max_img_num', type=int, default=10)
config = parser.parse_args()


if __name__ == '__main__':
    run()