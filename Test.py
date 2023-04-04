# import matplotlib.pyplot as plt
# pic = plt.imread('./examples/content/Test_1.jpg')

# print(pic.shape[2])


#     # The filenames of the content and style pair should match
#     fnames = set(os.listdir(config.content)) & set(os.listdir(config.style))
#     print(set(os.listdir(config.content)))
#     print(set(os.listdir(config.style)))


#     # if config.content_segment and config.style_segment:
#     #     fnames &= set(os.listdir(config.content_segment))
#     #     fnames &= set(os.listdir(config.style_segment))

#     for fname in tqdm.tqdm(fnames):
#         if not is_image_file(fname):
#             print('invalid file (is not image), ', fname)
#             continue
#         _content = os.path.join(config.content, fname)
#         _style = os.path.join(config.style, fname)
#         _content_segment = os.path.join(config.content_segment, "black_.png") if config.content_segment else None
#         _style_segment = os.path.join(config.style_segment, "black_.png") if config.style_segment else None
#         _output = os.path.join(config.output, fname)

#         content = open_image(_content, config.image_size).to(device)
#         style = open_image(_style, config.image_size).to(device)
#         content_segment = load_segment(_content_segment, config.image_size)
#         style_segment = load_segment(_style_segment, config.image_size)     
#         _, ext = os.path.splitext(fname)
        
#         if not config.transfer_all:
#             with Timer('Elapsed time in whole WCT: {}', config.verbose):
#                 postfix = '_'.join(sorted(list(transfer_at)))
#                 fname_output = _output.replace(ext, '_{}_{}.{}'.format(config.option_unpool, postfix, ext))
#                 print('------ transfer:', _output)
#                 wct2 = WCT2(transfer_at=transfer_at, option_unpool=config.option_unpool, device=device, verbose=config.verbose)
#                 with torch.no_grad():
#                     img = wct2.transfer(content, style, content_segment, style_segment, alpha=config.alpha)
#                 save_image(img.clamp_(0, 1), fname_output, padding=0)
#         else:
#             for _transfer_at in get_all_transfer():
#                 with Timer('Elapsed time in whole WCT: {}', config.verbose):
#                     postfix = '_'.join(sorted(list(_transfer_at)))
#                     fname_output = _output.replace(ext, '_{}_{}.{}'.format(config.option_unpool, postfix, ext))
#                     print('------ transfer:', fname)
#                     wct2 = WCT2(transfer_at=_transfer_at, option_unpool=config.option_unpool, device=device, verbose=config.verbose)
#                     with torch.no_grad():
#                         img = wct2.transfer(content, style, content_segment, style_segment, alpha=config.alpha)
#                     save_image(img.clamp_(0, 1), fname_output, padding=0)
#                     print(fname_output)

import os
import tqdm
import argparse

import torch
from torchvision.utils import save_image

from model import WaveEncoder, WaveDecoder

from utils.core import feature_wct
from utils.io import Timer, open_image, load_segment, compute_label_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='./examples/content')
    parser.add_argument('--content_segment', type=str, default='./examples/content_segment')
    parser.add_argument('--style', type=str, default='./examples/style')
    parser.add_argument('--style_segment', type=str, default='./examples/style_segment')
    parser.add_argument('--output', type=str, default='./outputs')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--option_unpool', type=str, default='cat5', choices=['sum', 'cat5'])
    parser.add_argument('-e', '--transfer_at_encoder', action='store_true')
    parser.add_argument('-d', '--transfer_at_decoder', action='store_true')
    parser.add_argument('-s', '--transfer_at_skip', action='store_true')
    parser.add_argument('-a', '--transfer_all', default = True)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    config = parser.parse_args()

IMG_EXTENSIONS = (
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG',
)

def is_image_file(filename):
    if filename.endswith(IMG_EXTENSIONS):
        return "True"

fname_c = os.listdir(config.content)[0]
fname_s = os.listdir(config.style)[0]

if is_image_file(fname_c) and is_image_file(fname_s):
    print("Hellow")