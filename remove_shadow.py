import argparse
import math
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms

import lpips
import models
from model import Generator, Discriminator
import utils
from utils import preprocess
from projector import prepare_parser
from utils.projection_utils import *

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(
            spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input

def add_shadow_removal_parser(parser):
    parser.add_argument(
        "--fm_loss", type=str, help="VGG or discriminator", choices=['disc', 'vgg']
    )
    
    parser.add_argument("--w_noise_reg", type=float, default=1e5, help="weight of the noise regularization")
    parser.add_argument("--w_mse", type=float, default=0,
                        help="weight of the mse loss")
    parser.add_argument("--w_percep", type=float, default=0,
                        help="weight of the perceptual loss")
    parser.add_argument("--w_arcface", type=float, default=0,
                        help="weight of the arcface loss")
    parser.add_argument("--w_exclusion", type=float,
                        default=0, help="weight of the exclusion loss")

    parser.add_argument("--stage2", type=int, default=300,
                        help="optimize iterations")
    parser.add_argument("--stage3", type=int, default=450,
                        help="optimize iterations")
    parser.add_argument("--stage4", type=int, default=800,
                        help="optimize iterations")

    parser.add_argument("--detail_refine_loss", action='store_true')
    parser.add_argument("--visualize_detail", action='store_true')
    parser.add_argument("--save_samples", action='store_true')
    parser.add_argument("--save_inter_res", action='store_true')

    return parser


def main(img_path, res_dir, device, args):

    # ----- preprocess img -----
    resize = min(args.size, 256)
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # ----- load shadow images ------
    img_np = np.array(Image.open(img_path).convert('RGB'))
    img = transform(Image.open(img_path).convert("RGB"))  # range: -1~1
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    print(f'target img path: {img_path}')

    # ----- segment face -----
    face_parsing_netp = 'checkpoint/face-seg-BiSeNet-79999_iter.pth'
    binary_mask = preprocess.evaluate(respth=res_dir, dspth=img_path, cp=face_parsing_netp, seg_type='binary')
    binary_mask = binary_mask[::2, ::2]
    utils.save_np_img(binary_mask, img_path=res_dir + '/binarymask.jpg')
    binary_mask = torch.tensor(binary_mask).to(device).reshape(1, 1, 256, 256) / 255.  # 0-1

    # parse facial details for refinement in stage 3 
    parse_map = preprocess.evaluate(respth=res_dir, dspth=img_path, cp=face_parsing_netp, seg_type='five_organs')
    brow = parse_map[0][::2, ::2]
    eye = parse_map[1][::2, ::2]
    nose = parse_map[2][::2, ::2]
    mouse = parse_map[3][::2, ::2]
    glass = parse_map[4][::2, ::2]
    
    if args.visualize_detail:
        utils.save_np_img(brow * 255, img_path=res_dir + '/brow.jpg')
        utils.save_np_img(eye * 255, img_path=res_dir + '/eye.jpg')
        utils.save_np_img(nose * 255, img_path=res_dir + '/nose.jpg')
        utils.save_np_img(mouse * 255, img_path=res_dir + '/mouse.jpg')
        utils.save_np_img(glass * 255, img_path=res_dir + '/glass.jpg')

    brow = torch.tensor(brow).reshape(1, 1, 256, 256).cuda().float()
    eye = torch.tensor(eye).reshape(1, 1, 256, 256).cuda().float()
    nose = torch.tensor(nose).reshape(1, 1, 256, 256).cuda().float()
    mouse = torch.tensor(mouse).reshape(1, 1, 256, 256).cuda().float()
    glass = torch.tensor(glass).reshape(1, 1, 256, 256).cuda().float()

    input_shadow_imgs = img.unsqueeze(dim=0).to(device)


    # ----- define models -----
    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    discriminator = Discriminator(args.size, channel_multiplier=2).to(device)

    ckpt = torch.load(args.ckpt)
    discriminator.load_state_dict(ckpt["d"])

    mask_net = models.get_net(32, 'skip', 'reflection',
                              n_channels=1,
                              skip_n33d=128,
                              skip_n33u=128,
                              skip_n11=4,
                              num_scales=5,
                              upsample_mode='bilinear').type(torch.cuda.FloatTensor).cuda()
    
    # ---- obtain mean and std for noise -----
    with torch.no_grad():
        noise_sample = torch.randn(args.n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() /
                      args.n_mean_latent) ** 0.5

    # ----- define losses -----
    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    # ----- generate noise for stylegan ----
    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(
            input_shadow_imgs.shape[0], 1, 1, 1).normal_())
    for noise in noises:
        noise.requires_grad = True

    # ----- sample 500 initial latents -----
    losses, latents_in = [], []
    with torch.no_grad():
        noise_sample = torch.randn(args.n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)  # 10000, 512
        latent_mean = latent_out.mean(0)        # 1,512
        latent_std = ((latent_out - latent_mean).pow(2).sum() /
                      args.n_mean_latent) ** 0.5

        # ---- obtain mean and std for noise -----
        for i in range(500):
            noise_sample = torch.randn(1, 512, device=device)
            latent_in = g_ema.style(noise_sample).reshape(1, 1, 512)
            latent_in = latent_in.repeat(1, g_ema.n_latent, 1)  # (1, 14, 512)

            img_gen, _ = g_ema(
                [latent_in], input_is_latent=True, noise=noises)  # 1,3,256,256

            if args.save_samples:
                os.makedirs(res_dir + '/sample-latent/', exist_ok=True)
                utils.save_torch_img(img_gen, img_path=res_dir +
                                 '/sample-latent/' + img_name + '-id{}.jpg'.format(i))
            loss = percept(img_gen * binary_mask,
                             input_shadow_imgs * binary_mask)
            losses.append(loss)
            latents_in.append(latent_in)

            if (i + 1) % 100 == 0:
                print('sample {}th samples'.format(i))

    losses = torch.cat(losses)
    idx = torch.argmin(losses)
    latent_in = latents_in[idx]
    latent_in.requires_grad = True

    # if mask has 1 variable
    mask_noise = get_noise(32, 'noise', (256, 256)).cuda()

    shadow_matrix_init = torch.zeros((1, 3)).fill_(0.5).cuda()  # 32,256, 256
    shadow_matrix_init.requires_grad = True

    # --- optimizer for stage 1 ----
    optimizer = optim.Adam([
        {'params': [latent_in] + noises, 'lr': 0.01},
        {'params': mask_net.parameters(), 'lr': 0},
        {'params': [shadow_matrix_init], 'lr': 0},
        {'params': g_ema.parameters(), 'lr': 0}
    ])

    pbar = tqdm(range(args.step))

    # ----- Optimization -----
    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        # optimizer_latent.param_groups[0]["lr"] = lr

        noise_strength = latent_std * args.noise * \
            max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())
        img_gen, _ = g_ema([latent_n], input_is_latent=True,noise=noises)  # 1,3,256,256

        batch, channel, height, width = img_gen.shape

        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])


        shadow_matrix = F.sigmoid(shadow_matrix_init)
        img_gen_shadow = (img_gen + 1) * shadow_matrix.reshape(1, 3, 1, 1) - 1


        mask = mask_net(mask_noise)
        shadow_img = img_gen * mask + img_gen_shadow * (1-mask)

        if args.fm_loss == 'disc':
            disc_loss = utils.DiscriminatorLoss(ftr_num=7)
            D = Discriminator(args.size).to(device)
            p_loss = disc_loss(D, img_gen, input_shadow_imgs)
        elif args.fm_loss == 'vgg':
            p_loss = percept(img_gen, input_shadow_imgs).sum()

        mse_loss = F.mse_loss(shadow_img, input_shadow_imgs)
        n_loss = noise_regularize(noises)
        g_loss = utils.g_loss(discriminator, img_gen)

        # ----- 3-Stage Optimization -----
        if i < args.stage2:  # only optimize latent codes
            optimizer.param_groups[0]['lr'] = 0.01  # latent
            optimizer.param_groups[1]['lr'] = 0  # mask_net
            optimizer.param_groups[2]['lr'] = 0  # colormatrix
            optimizer.param_groups[3]['lr'] = 0  # G
            loss = percept(img_gen * binary_mask,
                           input_shadow_imgs * binary_mask).sum() + n_loss * 1e5
        elif i >= args.stage2 and i < args.stage3:
            optimizer.param_groups[0]['lr'] = 0  # fix gen_img, update mask
            optimizer.param_groups[1]['lr'] = 0.001  # mask net
            optimizer.param_groups[2]['lr'] = 0.01  # color matrix
            loss = F.mse_loss(shadow_img * binary_mask,
                              input_shadow_imgs * binary_mask)
        elif i >= args.stage3 and i < args.stage4:
            loss = percept(shadow_img * binary_mask,
                           input_shadow_imgs * binary_mask).sum() + n_loss * 1e5
            optimizer.param_groups[0]['lr'] = 0.1  # fix gen_img, update mask
            optimizer.param_groups[1]['lr'] = 0  # mask net
            optimizer.param_groups[2]['lr'] = 0.01  # color matrix
        elif i >= args.stage4:
            loss = percept(shadow_img * binary_mask,
                           input_shadow_imgs * binary_mask).sum()
            optimizer.param_groups[0]['lr'] = 0.01  # fix gen_img, update mask
            optimizer.param_groups[1]['lr'] = 0  # mask net
            optimizer.param_groups[2]['lr'] = 0.001  # color matrixs

            if args.detail_refine_loss:
                # parse map guided refine
                loss = percept(shadow_img * brow, input_shadow_imgs * brow).sum() + \
                    2*percept(shadow_img * eye, input_shadow_imgs * eye).sum() + \
                    2*percept(shadow_img * nose, input_shadow_imgs * nose).sum() + \
                    2*percept(shadow_img * mouse, input_shadow_imgs * mouse).sum() + \
                    2*percept(shadow_img * glass, input_shadow_imgs * glass).sum() + \
                    2*percept(shadow_img * binary_mask,
                            input_shadow_imgs * binary_mask).sum()
            else:
                loss = percept(shadow_img * binary_mask,
                            input_shadow_imgs * binary_mask).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f};"
                f"noise regularize: {n_loss.item():.4f};"
                f"mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                f"gen: {g_loss.item():.4f}; lr: {lr:.4f}"
            )
        )

        # save intermidiate results
        if i % 50 == 0:
            if not os.path.exists(res_dir + '/gen-clean'):
                os.mkdir(res_dir + '/gen-clean')
            if not os.path.exists(res_dir + '/rec-shadow'):
                os.mkdir(res_dir + '/rec-shadow')
            if not os.path.exists(res_dir + '/full-shadow'):
                os.mkdir(res_dir + '/full-shadow')
            if not os.path.exists(res_dir + '/mask'):
                os.mkdir(res_dir + '/mask')
            if not os.path.exists(res_dir + '/res'):
                os.mkdir(res_dir + '/res')

            gen_clean_save = img_gen * binary_mask + \
                (1-binary_mask)*input_shadow_imgs
            full_shadow_save = img_gen_shadow * binary_mask
            mask_save = mask * binary_mask
            rec_shadow_save = shadow_img * binary_mask + \
                (1-binary_mask)*input_shadow_imgs

            utils.save_torch_img(gen_clean_save, img_path=res_dir +
                                 '/gen-clean/' + img_name + '-iter{}.jpg'.format(i))

            if i >= args.stage2:
                utils.save_torch_img(full_shadow_save, img_path=res_dir +
                                     '/full-shadow/' + img_name + '-iter{}.jpg'.format(i))
                utils.save_torch_img(
                    mask_save, img_path=res_dir + '/mask/' + img_name + '-iter{}.jpg'.format(i))
                utils.save_torch_img(rec_shadow_save, img_path=res_dir +
                                     '/rec-shadow/' + img_name + '-iter{}.jpg'.format(i))

        if i == args.step - 1:
            utils.save_torch_img(
                gen_clean_save, img_path=res_dir + '/res/' + img_name + '-output.png')
            utils.save_torch_img(
                full_shadow_save, img_path=res_dir + '/res/' + img_name + '-full-shadow.png')
            utils.save_torch_img(
                mask_save, img_path=res_dir + '/res/' + img_name + '-mask.png')
            utils.save_torch_img(
                rec_shadow_save, img_path=res_dir + '/res/' + img_name + '-rec-shadow.png')


if __name__ == "__main__":

    parser = prepare_parser()
    parser = add_shadow_removal_parser(parser)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.set_num_threads(3)

    for img in os.listdir(args.img_dir):
        img_path = os.path.join(args.img_dir, img)
        img_name = os.path.splitext(img)[0]
        res_dir = os.path.join(args.save_dir, img_name)
        os.makedirs(res_dir, exist_ok=True)

        main(img_path, res_dir, args.device, args)
