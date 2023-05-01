import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import os
import argparse
from tqdm import tqdm

from models import create_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate images'
    )
    parser.add_argument('--path', type=str, default=".", help='path to generator weights')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--loss', type=str, default='wgan-gp', help='loss function')

    parser.add_argument('--dist', type=str, default='.')
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--batch', default=16, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=2000)
    parser.add_argument('--im_size', type=int, default=64)
    parser.add_argument('--zdim', type=int, default=100)
    args = parser.parse_args()

    #device = torch.device('cuda:%d'%(args.cuda))
    device = torch.device('mps')

    gen = create_eval(args=args)
    gen.to(device)

    checkpoint = torch.load(args.path, map_location=lambda a,b: a)
    gen.load_state_dict(checkpoint)

    gen.to(device)

    del checkpoint

    dist = f'.scratch/eval/{args.path.split("/")[-2]}'
    dist = os.path.join(dist, 'img')
    os.makedirs(dist, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(args.n_sample//args.batch)):
            noise = torch.randn(args.batch, args.zdim, 1, 1).to(device)
            g_imgs = gen(noise)
            for j, g_img in enumerate( g_imgs ):
                vutils.save_image(g_img.add(1).mul(0.5), 
                    os.path.join(dist, '%d.png'%(i*args.batch+j)))#, normalize=True, range=(-1,1))
