
import os
import tqdm
import argparse
# import psutil
from PIL import Image
from glob import glob

import torch
from torchvision.utils import save_image

from model import WaveEncoder, WaveDecoder

from utils.core import feature_wct
from utils.io import Timer, open_image, load_segment, compute_label_info


IMG_EXTENSIONS = (
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG',
)

def is_image_file(filename):
    if filename.endswith(IMG_EXTENSIONS):
        return "True"

class WCT2:
    def __init__(self, model_path='./model_checkpoints', transfer_at=['encoder', 'skip', 'decoder'], option_unpool='cat5', device='cuda:0', verbose=False):

        self.transfer_at = set(transfer_at)
        assert not(self.transfer_at - set(['encoder', 'decoder', 'skip'])), 'invalid transfer_at: {}'.format(transfer_at)
        assert self.transfer_at, 'empty transfer_at'

        self.device = torch.device(device)
        self.verbose = verbose
        self.encoder = WaveEncoder(option_unpool).to(self.device)
        self.decoder = WaveDecoder(option_unpool).to(self.device)
        self.encoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_encoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_decoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))

    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    def transfer(self, content, style, content_segment, style_segment, alpha=1):
        label_set, label_indicator = compute_label_info(content_segment, style_segment)
        content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style)

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct2_enc_level:
                content_feat = feature_wct(content_feat, style_feats['encoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
                self.print_('transfer at encoder {}'.format(level))
        if 'skip' in self.transfer_at:
            for skip_level in wct2_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    content_skips[skip_level][component] = feature_wct(content_skips[skip_level][component], style_skips[skip_level][component],
                                                                       content_segment, style_segment,
                                                                       label_set, label_indicator,
                                                                       alpha=alpha, device=self.device)
                self.print_('transfer at skip {}'.format(skip_level))

        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
                content_feat = feature_wct(content_feat, style_feats['decoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
                self.print_('transfer at decoder {}'.format(level))
            content_feat = self.decode(content_feat, content_skips, level)
        return content_feat


def get_all_transfer():
    ret = []
    for e in ['encoder', None]:
        for d in ['decoder', None]:
            for s in ['skip', None]:
                _ret = set([e, d, s]) & set(['encoder', 'decoder', 'skip'])
                if _ret:
                    ret.append(_ret)
    return ret


def run_bulk(config, progress_callback = None):
    i = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    device = torch.device(device)

    transfer_at = set()
    if config.transfer_at_encoder:
        transfer_at.add('encoder')
    if config.transfer_at_decoder:
        transfer_at.add('decoder')
    if config.transfer_at_skip:
        transfer_at.add('skip')

    fname_c = os.listdir(config.content)[0]
    fname_s = os.listdir(config.style)[0]

    if is_image_file(fname_c) and is_image_file(fname_s):
        _content = os.path.join(config.content, fname_c)
        _style = os.path.join(config.style, fname_s)
        _content_segment = os.path.join(config.content_segment, "black_.png") if config.content_segment else None
        _style_segment = os.path.join(config.style_segment, "black_.png") if config.style_segment else None
        _output = os.path.join(config.output, fname_c)
        
        # with Image.open(_style) as img_style:
        #     width, height = img_style.size
        #     # fname_s_img = img_style.resize((width//6, height//6), Image.ANTIALIAS)
        #     fname_s_img.convert('RGB')
        #     fname_s_img.save(_style, 'png')
        fname_c_img = Image.open(_content).convert('RGB')
        fname_s_img = Image.open(_style).convert('RGB')

        fname_c_img.save(_content, 'png')
        fname_s_img.save(_style, 'png')

        content = open_image(_content, config.image_size).to(device)
        style = open_image(_style, config.image_size).to(device)
        content_segment = load_segment(_content_segment, config.image_size)
        style_segment = load_segment(_style_segment, config.image_size)     
        _, ext = os.path.splitext(fname_c)
        
        if not config.transfer_all:
            with Timer('Elapsed time in whole WCT: {}', config.verbose):
                postfix = '_'.join(sorted(list(transfer_at)))
                fname_output = _output.replace(ext, '_{}_{}.{}'.format(config.option_unpool, postfix, ext))
                print('------ transfer:', _output)
                wct2 = WCT2(transfer_at=transfer_at, option_unpool=config.option_unpool, device=device, verbose=config.verbose)
                with torch.no_grad():
                    img = wct2.transfer(content, style, content_segment, style_segment, alpha=config.alpha)
                save_image(img.clamp_(0, 1), fname_output, padding=0)
                i = i + 1
                progress = (i+1) / 7
                print(progress)
                progress_callback(progress)
        else:
            for _transfer_at in get_all_transfer():
                with Timer('Elapsed time in whole WCT: {}', config.verbose):    
                    postfix = '_'.join(sorted(list(_transfer_at)))
                    fname_output = _output.replace(ext, '_{}_{}.{}'.format(config.option_unpool, postfix, ext))
                    print('------ transfer:', fname_c)
                    wct2 = WCT2(transfer_at=_transfer_at, option_unpool=config.option_unpool, device=device, verbose=config.verbose)
                    with torch.no_grad():
                        img = wct2.transfer(content, style, content_segment, style_segment, alpha=config.alpha)
                    save_image(img.clamp_(0, 1), fname_output, padding=0)
                    i = i + 1
                    progress = (i+1) / 8
                    progress_callback(progress)
                    print(fname_output)
                    print(progress)

    else:
        print('invalid file (is not image) was included')
    
    return i


def DeleteAllFiles(filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file.path)
        return "Remove All File"
    else:
        return "Directory Not Found"

def SelectOutputFile():
    for file in os.scandir("outputs"):
        if "decoder_encoder_skip" in file.path:
            continue
        os.remove(file.path)


def run(progress_callback = None):
    
    if not os.path.exists(os.path.join(config.output)):
        os.makedirs(os.path.join(config.output))

    '''
    CUDA_VISIBLE_DEVICES=6 python transfer.py --content ./examples/content --style ./examples/style --content_segment ./examples/content_segment --style_segment ./examples/style_segment/ --output ./outputs/ --verbose --image_size 512 -a
    '''
    run_bulk(config, progress_callback)

    print(DeleteAllFiles('./examples/content'))
    print(DeleteAllFiles('./examples/style'))
    SelectOutputFile()


parser = argparse.ArgumentParser()
parser.add_argument('--content', type=str, default='./examples/content')
parser.add_argument('--content_segment', type=str, default='./examples/content_segment')
parser.add_argument('--style', type=str, default='./examples/style')
parser.add_argument('--style_segment', type=str, default='./examples/style_segment')
parser.add_argument('--output', type=str, default='./outputs')
parser.add_argument('--image_size', type=int, default=1024)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--option_unpool', type=str, default='cat5', choices=['sum', 'cat5'])
parser.add_argument('-e', '--transfer_at_encoder', action='store_true')
parser.add_argument('-d', '--transfer_at_decoder', action='store_true')
parser.add_argument('-s', '--transfer_at_skip', action='store_true')
parser.add_argument('-a', '--transfer_all', default = True)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--verbose', action='store_true')
config = parser.parse_args()


if __name__ == '__main__':
    run()