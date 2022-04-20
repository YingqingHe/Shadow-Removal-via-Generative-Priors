# ShadowGP


**This repository provides the official codes for our paper: [Unsupervised Portrait Shadow Removal via Generative Priors (ACM MM 2021)](https://arxiv.org/abs/2108.03466).** 
> **Unsupervised Portrait Shadow Removal via Generative Priors** <br>
>  Yingqing He*, Yazhou Xing*, Tianjia Zhang, Qifeng Chen (* indicates joint first authors)<br>
>  HKUST <br>

<!-- [[Paper](https://arxiv.org/abs/2108.03466)]  -->
<!-- [[Project Page (Coming soon)](TBA)]
[[Technical Video (Coming soon)](TBA)] -->


In this repository, we propose an unsupervised method for portrait shadow removal, named as ShadowGP. ShadowGP can recover a shadow-free portrait image via single image optimization, without a large paired training dataset, which is expensive to collect and time-consuming to train. Besides, our method can also be extended to facial tattoo removal and watermark removal tasks.   
![](./figures/teaser.png)
<!-- **Figure:** *Our results* -->
<!-- <br />     -->
ShadowGP can decompose the **single input shadowed portrait image** into **3 parts: a full-shadow portrait, a shadow-free portrait and a shadow mask**. Blending the three parts together can reconstruct the input shadowed portrait. The decomposed shadow-free portrait is the target output.  
![](./figures/result.png)
<!-- **Figure:** *Our unsupervised method takes a single shadow portrait as input and can decompose it into a shadow-free portrait image, a full-shadow portrait image, and a shadow mask* -->


<br />

## Install Environment
To install and activate the environment, run the following commands:
```
conda create -n shadowgp python=3.7
conda activate shadowgp
pip3 install torch==1.3.1+cu100 torchvision==0.4.2+cu100 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install opencv-python tqdm scikit-image==0.15 Pillow==7.1.0 Ipython ninja
```
We use the same system requirements as [StyleGAN2-PyTorch](https://github.com/rosinality/stylegan2-pytorch).

<br />

## Download Checkpoints
Download all checkpoints from [google drive](https://drive.google.com/drive/folders/1Rg5He8XIY8qP4JYPFRRGUIvfZUcqm8zt?usp=sharing), and put them in checkpoint/ folder.
```
cd Shadow-Removal-via-Generative-Priors
mkdir checkpoint
mv ${YOUR_PATH}/550000.pt checkpoint/
mv ${YOUR_PATH}/face-seg-BiSeNet-79999_iter.pth checkpoint/
```
## Run
```
bash run.sh
```

<br />

## Acknowledgement
Our code is built on [StyleGAN2-PyTorch](https://github.com/rosinality/stylegan2-pytorch).


<br />

## Citation

```
@inproceedings{he21unsupervised,
  title     = {Unsupervised Portrait Shadow Removal via Generative Priors},
  author    = {He, Yingqing and Xing, Yazhou and Zhang, Tianjia and Chen, Qifeng},
  booktitle = {ACM International Conference on Multimedia (ACM MM)},
  year      = {2021}
}
```
