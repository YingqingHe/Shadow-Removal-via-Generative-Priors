# Unsupervised Portrait Shadow Removal via Generative Priors


**This repository includes official codes for "[Unsupervised Portrait Shadow Removal via Generative Priors (ACM MM 2021)](https://arxiv.org/abs/)".** 

![](./figures/teaser.png)
**Figure:** *Our results*

Portrait images often suffer from undesirable shadows cast by casual objects or even the face itself. While existing methods for portrait shadow removal require training on a large-scale synthetic dataset, we propose the first unsupervised method for portrait shadow removal without any training data. Our key idea is to leverage the generative facial priors embedded in the off-the-shelf pretrained StyleGAN2. To achieve this, we formulate the shadow removal task as a layer decomposition problem: a shadowed portrait image is constructed by the blending of a shadow image and a shadow-free image. We propose an effective progressive optimization algorithm to learn the decomposition process. Our approach can also be extended to portrait tattoo removal and watermark removal. Qualitative and quantitative experiments on a real-world portrait shadow dataset demonstrate that our approach achieves comparable performance with supervised shadow removal methods. 

> **Unsupervised Portrait Shadow Removal via Generative Priors** <br>
>  Yingqing He*, Yazhou Xing*, Tianjia Zhang, Qifeng Chen (* indicates joint first authors)<br>
>  HKUST <br>

[[Paper](https://arxiv.org/)] 
[[Project Page](TBA)]
[[Technical Video (Coming soon)](TBA)]

![](./figures/method_01.png)
**Figure:** *Our unsupervised method takes a single shadow portrait as input and can decompose it into a shadow-free portrait image, a full-shadow portrait image, and a shadow mask*

## Code will come soon.


## Citation

```
@inproceedings{he21unsupervised,
  title     = {Unsupervised Portrait Shadow Removal via Generative Priors},
  author    = {He, Yingqing and Xing, Yazhou and Zhang Tianjia and Chen, Qifeng},
  booktitle = {ACM International Conference on Multimedia (ACM MM)},
  year      = {2021}
}
```