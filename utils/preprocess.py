#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2

import torch
import torchvision.transforms as transforms

from models import BiSeNet

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8) # [512,512]
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255  # [512,512,3]
    
    num_of_class = np.max(vis_parsing_anno)
    
    for pi in range(1, num_of_class + 1):  # 1-18
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
        # print('class idx:{}, class color: {}, class region:{}'.format(pi, part_colors[pi], index))

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_im

def vis_parsing_maps_binary(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[0,0,0], 
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0], 
                   [0, 0, 0], [0, 0, 0], [0, 0, 0], 
                   [0, 0, 0], [0, 0, 0], [0, 0, 0]
                ]


    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8) # [512,512]
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST) # [512,512]
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(0, num_of_class +1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    if save_im:
        cv2.imwrite(save_path[:-4] + '-binary-color.jpg', vis_parsing_anno_color)   # binary parsing map 

    return vis_parsing_anno_color[:,:,0]

def vis_parsing_maps_five_organs(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts   # 24
    part_colors = [[0,0,0], 
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0], 
                   [0, 0, 0], [0, 0, 0], [0, 0, 0], 
                   [0, 0, 0], [0, 0, 0], [0, 0, 0]
                ]


    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    # initial an empty map
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    
    num_of_class = np.max(vis_parsing_anno)

    brow = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    eye = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    nose = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    mouse = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    glass = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    
    index_brow = np.where((vis_parsing_anno == 2) | (vis_parsing_anno == 3))
    index_eye = np.where((vis_parsing_anno == 4) | (vis_parsing_anno == 5))
    index_nose = np.where(vis_parsing_anno == 10)
    index_mouse = np.where((vis_parsing_anno == 11) | (vis_parsing_anno == 12) | (vis_parsing_anno == 13))
    index_glass = np.where(vis_parsing_anno == 6)

    brow[index_brow[0], index_brow[1]] = 1
    eye[index_eye[0], index_eye[1]] = 1
    nose[index_nose[0], index_nose[1]] = 1
    mouse[index_mouse[0], index_mouse[1]] = 1
    glass[index_glass[0], index_glass[1]] = 1

    return [brow, eye, nose, mouse, glass]

def vis_parsing_maps_binary_hair(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts   # 24
    part_colors = [[0,0,0], 
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [0, 0, 0], [255,255,255], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0], 
                   [0, 0, 0], [0, 0, 0], [0, 0, 0], 
                   [0, 0, 0], [0, 0, 0], [0, 0, 0]
                ]


    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    
    num_of_class = np.max(vis_parsing_anno)  # 17/18

    for pi in range(0, num_of_class +1):  # 1-18
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    if save_im:
        cv2.imwrite(save_path[:-4] + '-binary-color.jpg', vis_parsing_anno_color)   # binary parsing map 

    return vis_parsing_anno_color[:,:,0]




def evaluate(respth='./res/test_res2', dspth='./data', cp='model_final_diss.pth', seg_type='multiple', save_binary_mask=False, stride=1):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        
        if os.path.isdir(dspth):
            for image_path in os.listdir(dspth):
                img = Image.open(osp.join(dspth, image_path))
                image = img.resize((512, 512), Image.BILINEAR)
                img = to_tensor(image)
                img = torch.unsqueeze(img, 0)
                img = img.cuda()

                out = net(img)[0]  # [1, 19, 512, 512])
                parsing = out.squeeze(0).cpu().numpy().argmax(0)  # [512 512] 
                # print(np.unique(parsing))
                if seg_type == 'multiple':
                    mask = vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))
                elif seg_type == 'binary':
                    mask = vis_parsing_maps_binary(image, parsing, stride=1, save_im=save_binary_mask, save_path=osp.join(respth, image_path))
                elif seg_type == 'five_organs':
                    mask = vis_parsing_maps_five_organs(image, parsing, stride=1, save_im=save_binary_mask, save_path=osp.join(respth, image_path))
                elif seg_type == 'binary+hair':
                    mask = vis_parsing_maps_binary_hair(image, parsing, stride=1, save_im=save_binary_mask, save_path=osp.join(respth, image_path))
        elif os.path.isfile(dspth):
            image_path = dspth
            img = Image.open(dspth)
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()

            out = net(img)[0]  # [1, 19, 512, 512])
            parsing = out.squeeze(0).cpu().numpy().argmax(0)  # [512 512] 

            if seg_type == 'multiple':
                mask = vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))
            elif seg_type == 'binary':
                mask = vis_parsing_maps_binary(image, parsing, stride=1, save_im=save_binary_mask, save_path=osp.join(respth, image_path))
            elif seg_type == 'five_organs':
                mask = vis_parsing_maps_five_organs(image, parsing, stride=1, save_im=save_binary_mask, save_path=osp.join(respth, image_path))
            elif seg_type == 'binary+hair':
                    mask = vis_parsing_maps_binary_hair(image, parsing, stride=1, save_im=save_binary_mask, save_path=osp.join(respth, image_path))
    return mask


if __name__ == "__main__":
    binary_mask = evaluate(
            respth='./res/test_res2',
            dspth='/home/yingqing/disk2/code/stylegan2-pytorch/data/test-parsing', 
            cp='/home/yingqing/disk2/code/stylegan2-pytorch/checkpoint/face-seg-BiSeNet-79999_iter.pth',
            seg_type = 'binary')

