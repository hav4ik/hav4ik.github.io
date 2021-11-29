---
layout: post
permalink: /articles/:title
type: "article"
title: "Consistency &mdash; a trick for Unsupervised Learning"
image:
  feature: "/articles/images/2021-11-25-consistency/feature.jpg"
  display: false
commits: "#"
tags: [unsupervised learning, deep learning, survey]
excerpt: >-
  One of the ways to train a model on unlabeled data is to use Consistency objectives to exploit the invariances of the model's outputs.
  This post is a curated list of examples of how Consistency Constraints are used in Unsupervised Learning.
  
show_excerpt: true
comments: true
hidden: false
highlighted: true
---

Despite the promising results of Deep Learning methods, most of the existing approaches are based on [supervised learning][supervised-learning] formulation and, as a result, require vast quantities of corresponding ground truth data for training. According to [Richard Sutton's "Bitter Lesson"][bitter-lesson], the best performing AI system are the ones that can make use of more data. While it is relatively easy to collect a large amount of data, labeling that data is an enormously labor-expensive and time-consuming task. The amount of data we can collect grows exponentially faster than our ability to label them. Being able to leverage a vast amount of unlabeled data and learn in an unsupervised fashion is extremely important for AI projects.

That is where **Consistency Constraints** come in. The idea behind this group of methods is quite elegant. Let's say you want to learn a mapping $$f \colon \mathcal{X} \to \mathcal{Y}$$ that minimizes some objective $$\mathcal{L}(f)$$ but you cannot optimize using this objective directly due to lack of labeled data. Instead, you enforce $$f$$ to be invariant under some manipulations of input data, thus reducing the space of possible mapping functions. The assumption here is that if $$f$$ satisfies your invariance constraints, it will also minimize $$\mathcal{L}(f)$$.

In this blog post, I will review how the concept of "Consistency" is used in unsupervised learning methods. Below is the list of examples of the usage of consistency constraints grouped by tasks:


* [Monocular Depth Estimation](#monodepth)
  - [Unsupervised Monocular Depth Estimation with Left-Right Consistency](#left-right-consistency)
  - [Feature-metric Loss for Self-supervised Learning of Depth and Egomotion](#feature-metric-loss)
  - [Unsupervised Monocular Depth Estimation in Highly Complex Environments](#complex-environments)

* [Generative Adversarial Nets (GANs)](#gans)
  - [Image-to-Image Translation using Cycle-Consistent Adversarial Nets](#cycle-gan)

* [Object Localization](#object-localization)
  - [Puzzle-Cam: Improved Localization via matching Partial and Full features](#puzzle-cam)

[supervised-learning]: https://en.wikipedia.org/wiki/Supervised_learning
[bitter-lesson]: http://incompleteideas.net/IncIdeas/BitterLesson.html



----------------------------------------------------------------------------------------------------------------------



<a name="monodepth"></a>
## Monocular Depth Estimation

Let's start with [**Monocular Depth Estimation**][pwc-depth-estimation] because this is one of the most successful examples of consistency-based approaches (they outperform even the [LIDAR][wiki-lidar] ground truth). And also because my first project at my first job at [Samsung Research][srk] was related closely to this topic!

This task is equivalent to estimating the distances to the objects around you with one eye closed. The goal is to predict the depth value of each pixel, given only a single RGB image as input. Monocular Depth Estimation plays a core role for vision-based self-driving vehicles and robotic projects.

<a name="fig-kitti"></a>
{% capture imblock_kitti %}
    {{ site.url }}/articles/images/2021-11-25-consistency/kitti-1.png
    {{ site.url }}/articles/images/2021-11-25-consistency/kitti-2.png
{% endcapture %}
{% capture imcaption_kitti %}
  Example of a depth estimation task. The samples shown here were taken from [Kitti benchmark](https://www.bing.com/search?q=kitti+dataset&cvid=7ae147e68df8445b9c264a4101dac810&aqs=edge..69i57j0l5j69i60l3.1725j0j9&FORM=ANAB01&PC=U531) dataset. Each RGB image in the dataset is paired with a dense depth map ground truth using [LIDAR](https://en.wikipedia.org/wiki/Lidar).
{% endcapture %}
{% include gallery images=imblock_kitti cols=1 caption=imcaption_kitti %}

The depth estimation task was initially treated as a simple regression problem &mdash; given an image $$\bf x$$ and ground truth depth $${\boldsymbol d}_x$$, the objective was simply RMSE (i.e. $$\textstyle \| f({\bf x}) - {\boldsymbol d}_x \|_2^2 \to \min$$) or its variants, averaged across all pixels. Here are some of the most successful supervised methods at that time:

* [Eigen & Fedgus (2015)][depth_eigen2015] designed a multi-scale network to learn 3D geometry information such as depth and surface normals using a scale-invariant version of RMSE. For some time, this work was SOTA on multiple large-scale benchmarks ([NYU v2][pwc-nyu-v2], [Kitti][pwc-kitti]) for monocular depth estimation.

* [Laina et al. (2016)][depth_laina2016] utilized [Huber Loss][huber-loss], which has the advantage of being smoother and more stable than RMSE. It was SOTA on [NYU v2 benchmark][pwc-nyu-v2] for quite a while before unsupervised methods had taken over this research area.



<a name="left-right-consistency"></a>
### Unsupervised Monocular Depth Estimation with Left-Right Consistency

> Paper: [arXiv:1609.03677][depth_godard2017]  
> Consistency type(s): photometric consistency, disparity consistency.


[Godard et al. (2017)][depth_godard2017] proposed an elegant way to learn depth estimation **without ground-truth** data while still achieving SOTA on the [Kitti][pwc-kitti] benchmark, outperforming earlier supervised methods by a large margin. This is, to my knowledge, the first time Consistency Constraints are used for depth estimation. The presentation of this paper on CVPR 2017 is [available on Youtube][depth_godard2017-video].

Given a pair of [calibrated cameras][calibrated-cam] (with similar focal length $$\phi$$ and distance $$\beta$$ between the cameras) the idea is to learn to predict the [disparity map][what-is-disparity] $$\delta$$ between left and right images so that depth $$d$$ can be trivially recovered using the formula $$d = \beta \cdot \phi / \delta$$. With such formulation, the depth estimation problem has turned into an **Image Reconstruction** problem. Here is how it looks like schematically:

<a name="fig-depth_godard2017"></a>
{% capture imblock_depth_godard2017 %}
  {{ site.url }}/articles/images/2021-11-25-consistency/godard2017.png
{% endcapture %}
{% capture imcaption_depth_godard2017 %}
  Architecture of the method proposed by [Godard et al. (2017)](http://visual.cs.ucl.ac.uk/pubs/monoDepth/). The left image is combined with the predicted right disparity map to reconstruct the right image, and vice-versa. (Image source: I drew it 😋)
{% endcapture %}
{% include gallery images=imblock_depth_godard2017 cols=1 caption=imcaption_depth_godard2017 %}

During training, only the left image is fed into a neural network. The network then predicts both left and right disparity maps. Then, left and right input images are reconstructed using these disparity maps. Finally, the network is trained using the following objective:

$$
\begin{equation*}
\mathcal{L} =
\alpha_1 \left( \mathcal{L}_{\text{ph}}^l + \mathcal{L}_{\text{ph}}^r \right) +
\alpha_2 \left( \mathcal{L}_{\text{ds}}^l + \mathcal{L}_{\text{ds}}^r \right) +
\alpha_2 \left( \mathcal{L}_{\text{lr}}^l + \mathcal{L}_{\text{lr}}^r \right) \to \min.
\end{equation*}
$$

- $$\mathcal{L}^l_{\text{ph}}$$ is **photometric consistency** (or photometric image reconstruction) loss. In the paper, it is defined as a combination of per-pixel $$L_1$$ and single scale [SSIM (Structured Similarity Indexing)][ssim]. Given the left image $$\textstyle {\bf x}^l$$ and the reconstructed right image $$\textstyle {\bf \hat{x}}^l$$, the objective $$\mathcal{L}^l_{\text{ph}}$$ is defined as:

$$
\begin{equation}
\label{eqn:photometric} \tag{PHM}
\mathcal{L}^l_{\text{ph}} = \frac{1}{N} \sum_{i,j} \frac{\alpha}{2} \left(1 - \text{SSIM}\left({\bf x}_{ij}^l, {\bf \hat{x}}_{ij}^l\right)\right) + \left(1 - \alpha\right) \|{\bf x}_{ij}^l - {\bf \hat{x}}_{ij}^l\|
\end{equation}
$$

- $$\mathcal{L}^l_{\text{ds}}$$ is disparity smoothness loss. It is used to encourage local smoothness of left disparity maps.
- $$\mathcal{L}^l_{\text{lr}}$$ is **left-right disparity consistency** loss. This cost enforces the left-view disparity map $$\boldsymbol\delta^l$$ be equal to the right-view disparity map $$\textstyle \boldsymbol\delta^r$$ when projected back, defined as: $$\textstyle
\mathcal{L}_{\text{lr}}^l = \frac{1}{N} \sum_{i, j} \| \boldsymbol\delta_{ij}^l - \boldsymbol\delta_{ij + \boldsymbol\delta_{jj}^l}^r \|_1
$$.

- $$\mathcal{L}^r_{\text{rec}}\,$$, $$\mathcal{L}^r_{\text{ds}}\,$$, and $$\mathcal{L}^r_{\text{lr}}$$ are just the right-side counterparts of the below described objectives.


<a name="feature-metric-loss"></a>
### Feature-metric Loss for Self-supervised Learning of Depth and Egomotion

> Paper: [arXiv:2007.10603][depth_shu2020]  
> Consistency type(s): photometric consistency, feature-metric consistency, disparity consistency

[Shu et al. (2020)][depth_shu2020] elevated the use of Consistency Constraints to a whole new level. Following the previous success of unsupervised methods for depth estimation, they proposed a new approach that does not have the limitations of previous methods. More specifically:

* Instead of training on a stereo pair acquired from well-calibrated cameras with a known distance between them, the method trains on a **single-camera footage**.
* The proposed method removes the need for well-calibrated cameras. They trained a neural net to estimate the camera pose and its intrinsic parameters in an unsupervised manner.
* In addition to the enforcement of photometric consistency objectives, they also enforced the consistency of the feature maps, hence the name **feature-metric consistency**.

This method is currently State-of-the-Art in Monocular Depth Estimation (see: [Kitti depth unsupervised][kitti-depth-unsup-benchmark]).


<a name="fig-depth-shu2020"></a>
{% capture imblock_depth_shu2020 %}
  {{ site.url }}/articles/images/2021-11-25-consistency/shu2020.png
{% endcapture %}
{% capture imcaption_depth_shu2020 %}
  The whole thing is trained in a totally unsupervised manner. No ground-truth data is needed, nothing is labeled. **Absolute magic!** (Image source: [Shu et al. 2020](https://arxiv.org/abs/2007.10603v1]))
{% endcapture %}
{% include gallery images=imblock_depth_shu2020 cols=1 caption=imcaption_depth_shu2020 %}

The training pipeline of this approach consists of the following elements:

* $$\boldsymbol{I}_t$$ is the current frame, and $$\boldsymbol{I}_s$$ is the previous camera frame in time (with a short timestamp difference; they don't have to be consecutive frames).
* `FeatureNet` learns meaningful image representations in a self-supervised manner with single-view reconstruction. Its purpose is to remove redundancies and noises.
* `DepthNet` is a monocular depth estimator which takes an image and outputs a depth map.
* `PoseNet` is an egomotion estimator, which takes both frames $$\boldsymbol{I}_s$$ and $$\boldsymbol{I}_t$$ and outputs the [relative camera pose matrix][cam-matrix] $$\textstyle\{ \mathbf{R}, \mathbf{t} \}$$ between them.
* Feature maps $$\boldsymbol \phi_s$$ and $$\boldsymbol \phi_t$$ are computed using `FeatureNet` from $$\boldsymbol{I}_s$$ and $$\boldsymbol{I}_t$$ respectively.
* The reconstruction $$\boldsymbol{\hat{I}}_{s \to t}$$ and $$\boldsymbol{\hat{\phi}}_{s \to t}$$ of the current frame $$\boldsymbol{I}_t$$ and its representation $$\boldsymbol \phi_t$$ is computed by using the depth map generated by `DeepNet` and camera pose estimated by `PoseNet`.

The total loss for the whole architecture is $$\textstyle \mathcal{L}_{\text{total}} = \mathcal{L}_s + \mathcal{L}_{s \to t}$$, where $$\mathcal{L}_s$$ constrains the quality of learned features of `FeatureNet` through single-view reconstruction, whilst $$\mathcal{L}_{s \to t}$$ penalizes the discrepancy from cross-view reconstruction and is defined as:

$$
\begin{equation*}
\mathcal{L}_{s \to t} = \sum_p \
\mathcal{L}_{\text{fm}} \left( {\boldsymbol \phi_t}(p), \boldsymbol{\hat{\phi}}_{s \to t}(p)\right) +
\mathcal{L}_{\text{ph}} \left( \boldsymbol{I}_t(p), \boldsymbol{\hat{I}}_{s \to t}(p)\right)
\end{equation*}
$$

where $$\mathcal{L}_{\text{ph}}$$ is just a photometric objective, defined the same way as in equation ($$\ref{eqn:photometric}$$), and $$\mathcal{L}_{\text{fm}}$$ is the feature-metric consistency objective, defined as $$\mathcal{L}_{\text{fm}} = \| {\boldsymbol \phi_t}(p) - \boldsymbol{\hat{\phi}}_{s \to t}(p) \|_1$$.

<a name="fig-depth-demo"></a>
{% capture imblock_depth_demo %}
  {{ site.url }}/articles/images/2021-11-25-consistency/depthdemo_1.gif
  {{ site.url }}/articles/images/2021-11-25-consistency/depthdemo_3.gif
  {{ site.url }}/articles/images/2021-11-25-consistency/depthdemo_4.gif
{% endcapture %}
{% capture imcaption_depth_demo %}
  Can you believe this network was trained with no supervision at all?
{% endcapture %}
{% include gallery images=imblock_depth_demo cols=3 caption=imcaption_depth_demo %}



<a name="complex-environments"></a>
### Unsupervised Monocular Depth Estimation in Highly Complex Environments

> Paper: [arxiv:2107.13137][depth_zhao2021]  
> Consistency type(s): Representation consistency.

[Zhao et al. (2021)][depth_zhao2021], to improve the performance of the model in highly complex environments, combined previously described unsupervised depth estimation methods using Consistency Constraints with CycleGAN (which by itself is also a consistency-based method and we will go through it later in this post).

<a name="fig-depth-zhao2021"></a>
{% capture imblock_depth_zhao2021 %}
  {{ site.url }}/articles/images/2021-11-25-consistency/zhao2021.png
{% endcapture %}
{% capture imcaption_depth_zhao2021 %}
  The overall architecture of the proposed method. (Image source: [Zhao et al. 2021](https://arxiv.org/abs/2107.13137)).
{% endcapture %}
{% include gallery images=imblock_depth_zhao2021 cols=1 caption=imcaption_depth_zhao2021 %}

The daytime ("easy" environment) encoder $$E_d$$ and depth estimation decoder $$D_d$$ are trained using any of the unsupervised procedures described in previous sections. They additionally train the encoder $$E_x$$ on sequences of highly complex environments (night-time, snowy, rainy, etc.) generated by **CycleGAN**.

In addition to the photometric reconstruction and temporal consistency objectives described in previous subsections, the method also enforces **consistency constraints** on the representation outputted by encoders $$E_d$$ and $$E_x$$. Additionally, the output depth maps of the daytime and generated night-time photos are enforced to be consistent as well.



[srk]: https://research.samsung.com/srk
[wiki-lidar]: https://en.wikipedia.org/wiki/Lidar
[pwc-depth-estimation]: https://paperswithcode.com/task/depth-estimation
[pwc-surface-normal-estimation]: https://paperswithcode.com/task/surface-normals-estimation
[huber-loss]: https://en.wikipedia.org/wiki/Huber_loss
[pwc-nyu-v2]: https://paperswithcode.com/dataset/nyuv2
[pwc-kitti]: https://paperswithcode.com/dataset/kitti
[depth_godard2017-video]: https://www.youtube.com/watch?v=go3H2gU-Zck
[calibrated-cam]: https://www.mathworks.com/help/vision/ug/camera-calibration.html
[what-is-disparity]: https://stackoverflow.com/questions/7337323/what-is-the-definition-of-a-disparity-map
[ssim]: https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
[kitti-depth-unsup-benchmark]: https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen-1
[cam-matrix]: https://en.wikipedia.org/wiki/Camera_matrix



-----------------------------------------------------------------------------



<a name="gans"></a>
## Generative Adversarial Nets (GANs)

GANs are an exciting and active area of ML research, started by [Ian Goodfellow (2014)][goodfellow2014] and probably don't need any introduction. I'm sure that most people reading this blog know what it is and even are familiar with SOTA approaches in GANs. In this section, we present a few papers that utilize consistency constraints to train a GAN.


<a name="cycle-gan"></a>
### Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

> Paper: [arxiv:1703.10593][cyclegan]  
> Consistency type(s): Cycle Consistency.

This work by [Zhu et al. (2017)][cyclegan] is perhaps the most famous use of the idea of consistency constraints, and most of the readers of this post are probably already familiar with it. The paper describes a method for training a GAN that can translate images from one domain to another (e.g. from daytime photos to night-time, or portraits to anime drawings) in the absence of paired examples.

<a name="fig-cyclegan"></a>
{% capture imblock_cyclegan %}
  {{ site.url }}/articles/images/2021-11-25-consistency/cyclegan.png
{% endcapture %}
{% capture imcaption_cyclegan %}
  Training CycleGan in forward direction, learning to convert images from domain $$\mathcal A$$ to $$\mathcal B$$. The backward direction looks the same, with the arrows reversed (from $$\mathcal B$$ to $$\mathcal A$$), and discriminator $$D_{\mathcal B}$$ replaced with $$D_{\mathcal A}$$. The **cycle consistency** between source image and generated one is enforced.
{% endcapture %}
{% include gallery images=imblock_cyclegan cols=1 caption=imcaption_cyclegan %}

The usage of consistency constraints in this work is incredibly elegant and simple. Let's say we want to convert images from domain $$\mathcal A$$ to $$\mathcal B$$ and vice-versa. The idea is to train two generators, $$G_{\mathcal AB}\, \colon \mathcal{A} \to \mathcal{B}$$ and $$G_{\mathcal BA}\, \colon \mathcal{B} \to \mathcal{A}$$, and associated discriminators $$D_{\mathcal A}$$ and $$D_{\mathcal B}$$. $$D_{\mathcal B}$$ encourages $$G_{\mathcal AB}$$ to translate $$\mathcal A$$ images to outputs indistinguishable from domain $$\mathcal B$$, and vice versa for $$D_{\mathcal A}$$ and $$G_{\mathcal BA}$$.

To further regularize the mappings, two **cycle consistency** objectives were introduced. They capture the intuition that if we translate from one domain to the other and back again we should arrive at where we started. For each image $${\bf x} \in \mathcal{A}$$, the image translation cycle should be able to bring $${\bf x}$$ back to the original image, i.e. $${\bf x} \to G_{\mathcal AB}({\bf x}) \to G_{\mathcal BA}(G_{\mathcal AB}({\bf x})) \approx {\bf x}$$. This is called **forward cycle consistency**. Similarly, for each image $${\bf y} \in \mathcal B$$, the generators should also satisfy **backward cycle consistency**: $${\bf y} \to G_{\mathcal BA}({\bf y}) \to G_{\mathcal AB}(G_{\mathcal BA}({\bf y})) \approx {\bf y}$$. The final objective proposed by [Zhu et al. (2017)][cyclegan] is:

$$
\begin{equation*}
\mathcal{L}_\text{total} = \mathcal{L}_{\text{GAN}} \left( G_{\mathcal AB}, D_{\mathcal B} \right) + \mathcal{L}_{\text{GAN}} \left( G_{\mathcal BA}, D_{\mathcal A} \right) + \mathcal{L}_{\text{cycle}} \left( G_{\mathcal AB}, G_{\mathcal BA} \right)
\end{equation*}
$$

where $$\mathcal{L}_{\text{GAN}}$$ is the standard GAN loss ([Goodfellow et al. 2014][goodfellow2014]) and needs no further explanation, $$\mathcal{L}_{\text{cycle}}$$ is the **cycle consistency** objective that is defined as follows:

$$
\begin{align*}
  \mathcal{L}_{\text{cyc}}\left(G_{\mathcal AB}, G_{\mathcal BA}\right)
  = & 
  \mathbb{E}_{ {\bf x} \sim p_\text{data}({\bf x})} \left[ \left\| G_{\mathcal BA}(G_{\mathcal AB}({\bf x})) - {\bf x} \right\|_1 \right]
  \\ + &
  \mathbb{E}_{ {\bf y} \sim p_\text{data}({\bf y})} \left[ \left\| G_{\mathcal AB}(G_{\mathcal BA}({\bf y})) - {\bf y} \right\|_1 \right]
\end{align*}
$$

As in standard GANs, the generators are trained to minimize $$\mathcal{L}_\text{total}$$ while the discriminators are trained to maximize it. A classic example of consistency constraints in GANs.


[goodfellow2014]: https://arxiv.org/abs/1406.2661


-----------------------------------------------------------------------------


Cited as:

```plaintext
@online{hav4ik2021consistency,
  author = "Kha Vu, Chan",
  title = "Consistency — a trick for Unsupervised Learning",
  year = "2021",
  url = "https://hav4ik.github.io/articles/consistency-in-deep-learning",
}
```


<a name="references"></a>
## References

1. David Eigen, Rob Fedgus. ["Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture."][depth_eigen2015] In *ICCV*, 2015.

2. Iro Laina, Christian Rupprecht, Vasileios Belagiannis, Federico Tombari, and Nassir Navab. ["Deeper depth prediction with fully convolutional residual networks."][depth_laina2016] In *3DV*, 2016.

3. Clément Godard, Oisin Mac Aodha, and Gabriel J. Brostow. ["Unsupervised Monocular Depth Estimation with Left-Right Consistency."][depth_godard2017] In *CVPR*, 2017.

4. Chaoqiang Zhao, Yang Tang, Qiyu Sun. ["Unsupervised Monocular Depth Estimation in Highly Complex Environments."][depth_zhao2021] *arXiv:2107.13137*, 2021.

5. Chang Shu, Kun Yu, Zhixiang Duan, Kuiyuan Yang. ["Feature-metric Loss for Self-supervised Learning of Depth and Egomotion."][depth_shu2020] In *ECCV*, 2020.

6. Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros. ["Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks."][cyclegan] In *ICCV* 2017.

7. Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. ["Generative Adversarial Networks."][goodfellow2014] In *ICLR*, 2014. 



<!-- Links to cited papers -->
[depth_eigen2015]: https://cs.nyu.edu/~deigen/dnl/dnl_iccv15.pdf
[depth_laina2016]: https://arxiv.org/abs/1606.00373v2
[depth_godard2017]: http://visual.cs.ucl.ac.uk/pubs/monoDepth/
[depth_zhao2021]: https://arxiv.org/abs/2107.13137
[depth_shu2020]: https://arxiv.org/abs/2007.10603v1
[cyclegan]: https://arxiv.org/abs/1703.10593
[puzzle-can]: https://arxiv.org/abs/2101.11253
