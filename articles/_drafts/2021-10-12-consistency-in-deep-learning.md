---
layout: post
permalink: /articles/:title
type: "article"
title: "Consistency &mdash; a trick for Unsupervised Learning"
image:
  feature: "/articles/images/2021-10-12-consistency/feature.jpg"
  display: false
commits: "#"
tags: [unsupervised learning, deep learning]
excerpt: >-
  The amount of data we can collect grows exponentially faster than our ability to label them.
  This post is a curated collection of examples of how Consistency Constraints are used in Unsupervised Learning.
  I will try to keep this list up-to-date with the latest publications.
  
show_excerpt: true
comments: true
hidden: false
---

Despite the promising results of Deep Learning methods, most of the existing approaches are based on [supervised learning][supervised-learning] formulation and, as a result, require vast quantities of corresponding ground truth data for training. Evidence has shown again and again that [Richard Sutton's "Bitter Lesson"][bitter-lesson] is correct &mdash; usually, the best way to improve your AI system's performance is to feed more data to it. While it is relatively easy to collect a large amount of data, labeling that data is an enormously labor-expensive and time-consuming task. Being able to leverage a vast amount of unlabeled data and learn in an unsupervised fashion is extremely important for AI projects.

That is where **Consistency Constraints** come in. The idea behind this group of methods is quite elegant. Let's say you want to learn a mapping $$f \colon \mathcal{X} \to \mathcal{Y}$$ that minimizes some objective $$\mathcal{L}(f)$$ but you cannot optimize using this objective directly due to lack of labeled data. Instead, you enforce $$f$$ to be invariant under some manipulations of input data, thus reducing the space of possible mapping functions. The assumption here is that if $$f$$ satisfies your invariance constraints, it will also minimize $$\mathcal{L}(f)$$.

In this blog post, I will review how the concept of "Consistency" is used in unsupervised learning methods. Below is the list of examples of the usage of consistency constraints grouped by tasks:


* [Monocular Depth Estimation](#monodepth)
  - [Unsupervised Monocular Depth Estimation with Left-Right Consistency](#left-right-consistency)

* [Generative Adversarial Nets (GANs)](#gans)
  - [Image-to-Image Translation using Cycle-Consistent Adversarial Nets](#cycle-gan)

[supervised-learning]: https://en.wikipedia.org/wiki/Supervised_learning
[bitter-lesson]: http://incompleteideas.net/IncIdeas/BitterLesson.html



----------------------------------------------------------------------------------------------------------------------



<a name="monodepth"></a>
## Monocular Depth Estimation

Let's start with [**Monocular Depth Estimation**][pwc-depth-estimation] because this is one of the most successful examples of consistency-based approaches (they outperform even the [LIDAR][wiki-lidar] ground truth). And also because my first project at my first job at [Samsung Research][srk] was related closely to this topic!

This task is equivalent to estimating the distances to the objects around you with one eye closed. The goal is to predict the depth value of each pixel, given only a single RGB image as input. Monocular Depth Estimation plays a core role for vision-based self-driving vehicles and robotic projects.

<a name="fig-kitti"></a>
{% capture imblock_kitti %}
    {{ site.url }}/articles/images/2021-10-12-consistency/kitti-1.png
    {{ site.url }}/articles/images/2021-10-12-consistency/kitti-2.png
{% endcapture %}
{% capture imcaption_kitti %}
  Example of a depth estimation task. The samples shown here were taken from [Kitti benchmark](https://www.bing.com/search?q=kitti+dataset&cvid=7ae147e68df8445b9c264a4101dac810&aqs=edge..69i57j0l5j69i60l3.1725j0j9&FORM=ANAB01&PC=U531) dataset. Each RGB image in the dataset is paired with a dense depth map ground truth using [LIDAR](https://en.wikipedia.org/wiki/Lidar).
{% endcapture %}
{% include gallery images=imblock_kitti cols=1 caption=imcaption_kitti %}

The depth estimation task was initially treated as a simple regression problem &mdash; given an image $$x$$ and ground truth depth $$d_x$$, the objective was simply RMSE (i.e. $$\textstyle \| f(x) - d_x \|_2^2 \to \min$$) or its variants, averaged across all pixels. Here are some of the most successful supervised methods at that time:

* [Eigen & Fedgus (2015)][eigen2015] designed a multi-scale network to learn 3D geometry information such as depth and surface normals using a scale-invariant version of RMSE. For some time, this work was SOTA on multiple large-scale benchmarks ([NYU v2][pwc-nyu-v2], [Kitti][pwc-kitti]) for monocular depth estimation.

* [Laina et al. (2016)][laina2016] utilized [Huber Loss][huber-loss], which has the advantage of being smoother and more stable than RMSE. It was SOTA on [NYU v2 benchmark][pwc-nyu-v2] for quite a while before unsupervised methods had taken over this research area.


<a name="left-right-consistency"></a>
### Unsupervised Monocular Depth Estimation with Left-Right Consistency

[Godard et al. (2017)][godard2017] proposed an elegant way to learn depth estimation **without ground-truth** data while still achieving SOTA on the [Kitti][pwc-kitti] benchmark, outperforming earlier supervised methods by a large margin. The presentation of this paper on CVPR 2017 is [available on Youtube][godard2017-video].

Given a pair of [calibrated cameras][calibrated-cam] (with similar focal length $$\phi$$ and distance $$\beta$$ between the cameras) the idea is to learn to predict the [disparity map][what-is-disparity] $$\delta$$ between left and right images so that depth $$d$$ can be trivially recovered using the formula $$d = \beta \cdot \phi / \delta$$. With such formulation, the depth estimation problem has turned into an **Image Reconstruction** problem. Here is how it looks like schematically:

<a name="fig-godard2017"></a>
{% capture imblock_godard2017 %}
  {{ site.url }}/articles/images/2021-10-12-consistency/godard2017.png
{% endcapture %}
{% capture imcaption_godard2017 %}
  Description of the method proposed by [Godard et al. (2017)](http://visual.cs.ucl.ac.uk/pubs/monoDepth/). The left image is combined with the predicted right disparity map to reconstruct the right image, and vice-versa.
{% endcapture %}
{% include gallery images=imblock_godard2017 cols=1 caption=imcaption_godard2017 %}

During training, only the left image is fed into a neural network. The network then predicts both left and right disparity maps. Then, left and right input images are reconstructed using these disparity maps. Finally, the network is trained using the following objective:

$$
\begin{equation*}
\mathcal{L} =
\alpha_1 \left( \mathcal{L}_{\text{rec}}^l + \mathcal{L}_{\text{rec}}^r \right) +
\alpha_2 \left( \mathcal{L}_{\text{ds}}^l + \mathcal{L}_{\text{ds}}^r \right) +
\alpha_2 \left( \mathcal{L}_{\text{lr}}^l + \mathcal{L}_{\text{lr}}^r \right) \to \min.
\end{equation*}
$$

- $$\mathcal{L}^l_{\text{rec}}$$ is photometric image reconstruction loss. In the paper, it is defined as a combination of per-pixel $$L_1$$ and single scale [SSIM (Structured Similarity Indexing)][ssim].
- $$\mathcal{L}^l_{\text{ds}}$$ is disparity smoothness loss. It is used to encourage local smoothness of left disparity maps.
- $$\mathcal{L}^l_{\text{lr}}$$ is **left-right disparity consistency** loss. This cost attempts to make the left-view disparity map be equal to the projected right-view disparity map:

$$\begin{equation*} \mathcal{L}_{\text{lr}}^l = \frac{1}{N} \sum_{i, j} \left| \delta_{ij}^l - \delta_{ij + \delta_{jj}^l}^r\right| \end{equation*}$$

- $$\mathcal{L}^r_{\text{rec}}\,$$, $$\mathcal{L}^r_{\text{ds}}\,$$, and $$\mathcal{L}^l_{\text{lr}}$$ are just the right-side counterparts of the above described objectives.

[srk]: https://research.samsung.com/srk
[wiki-lidar]: https://en.wikipedia.org/wiki/Lidar
[pwc-depth-estimation]: https://paperswithcode.com/task/depth-estimation
[pwc-surface-normal-estimation]: https://paperswithcode.com/task/surface-normals-estimation
[huber-loss]: https://en.wikipedia.org/wiki/Huber_loss
[pwc-nyu-v2]: https://paperswithcode.com/dataset/nyuv2
[pwc-kitti]: https://paperswithcode.com/dataset/kitti
[godard2017-video]: https://www.youtube.com/watch?v=go3H2gU-Zck
[calibrated-cam]: https://www.mathworks.com/help/vision/ug/camera-calibration.html
[what-is-disparity]: https://stackoverflow.com/questions/7337323/what-is-the-definition-of-a-disparity-map
[ssim]: https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html


-----------------------------------------------------------------------------


Cite as:

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

[1] Eigen & Fedgus. ["Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture."][eigen2015] ICCV 2015

[2] Laina et al. ["Deeper depth prediction with fully convolutional residual networks."][laina2016] 3DV 2016

[3] Godard et al. ["Unsupervised Monocular Depth Estimation with Left-Right Consistency."][godard2017] CVPR 2017


<!-- Links to cited papers -->
[eigen2015]: https://cs.nyu.edu/~deigen/dnl/dnl_iccv15.pdf
[laina2016]: http://cs231n.stanford.edu/reports/2017/pdfs/203.pdf
[godard2017]: http://visual.cs.ucl.ac.uk/pubs/monoDepth/

