---
layout: post
permalink: /articles/:title
type: "article"
title: "Deep Metric Learning: a (Long) Survey"
image:
  feature: "/articles/images/2020-12-11-deep-metric-learning-survey/metric_losses.png"
  display: false
commits: https://github.com/hav4ik/hav4ik.github.io/commits/master/articles/_posts/2020-12-11-deep-metric-learning-survey.md
tags: [deep-learning, survey]
excerpt: "In this post, I'll briefly go over the common approaches for Deep Metric Learning, as well as the new methods proposed in recent years."
comments: true
hidden: true
---


One of the most amazing aspects of the human's visual system is the ability to recognize similar objects and scenes. We don't need hundreds of photos of the same face to be able to differentiate it among thousands of other faces that we've seen. We don't need thousands of images of the Eiffel Tower to recognize that unique architecture landmark when we visit Paris. Is it possible to design a Deep Neural Network with the similar ability to tell which objects are visually similar and which ones are not? That's essentially what **Deep Metric Learning** attempts to solve.

Advanced readers may immediately recognize by the description that this topic is intimately related to One-Shot Learning. The techniques of Metric Learning are commonly embraced in the field of One-Shot Learning, and sometimes these terms are even used interchangeably (e.g. by [Andrew Ng][andrewng_meme] in his [Deep Learning course][andrewng_one_shot_learning]). However, they are completely different fields &mdash; One-Shot Learning uses more than just Metric Learning techniques, and Metric Learning can be applied in other problems as well.

Although this blog post is mainly about **Supervised Deep Metric Learning** and is self-sufficient by its own, it would be benefitial for you to consider getting familiar with traditional Metric Learning methods (i.e. without Neural Networks) to develop a broader understanding on this topic. I highly recommend the [introductory guides on Metric Learning][sklearn_metric_learning_guide] as a starter. If you want to get into the formal mathematical side of things, I recommend the tutorial by [Diaz et al. (2020)][diaz_tutorial_metric_math]. More advanced Metric Learning methods includes the popular [t-SNE (van der Maaten & Hinton, 2008)][tsne_paper] and the new shiny [UMAP (McInnes et al., 2018)][umap_paper] that everybody uses nowadays for data clustering and visualization.

This article is organized as follows. In the **"Common Approaches"** section, I will quickly glance through the methods which are commonly used for Deep Metric Learning, before the rise of angular margin methods in 2017. In the **"State-of-the-Art Approaches"** section, I will describe in more detail the advances in Metric Learning in recent years. The most useful section for both beginners and more experienced readers will be the **"Getting Practical"** section, in which I will do a case study of how Deep Metric Learning is used to achieve State-of-the-Art results in various practical problems (mostly from Kaggle and large-scale benchmarks), as well as the tricks that were used to make things work.


[sklearn_metric_learning_guide]: http://contrib.scikit-learn.org/metric-learn/introduction.html
[diaz_tutorial_metric_math]: https://arxiv.org/abs/1812.05944
[tsne_paper]: https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf
[umap_paper]: https://arxiv.org/abs/1802.03426
[andrewng_one_shot_learning]: https://www.youtube.com/watch?v=96b_weTZb2w
[andrewng_meme]: https://i.imgur.com/6sQcztc.jpg


---------------------------------------------------------------------------------


- [Problem Setting of Supervised Metric Learning](#problem-setting)
- [Common Approaches](#common-approaches)
    - [Contrastive Loss](#contrastive-loss)
    - [Triplet Loss](#triplet-loss)
    - [Quadruplet Loss](#quadruplet-loss)

- [State-of-the-Art Approaches](#)
    - [CosFace, ArcFace, and SphereFace](#)
    - [AdaCos &mdash; Adaptive $$s$$ Parameter](#)
    - [Sub-Center ArcFace](#)
    - [ArcFace with Dynamic Margin (2020, Unpublished)](#)

- [Getting Practical](#)
  - [Case studies: what works, and what doesn't?](#)
    - [Kaggle: Humpack Whale Challenge](#)
    - [Kaggle: Google Landmarks Challenge](#)
    - [Face Recognition](#)
  - [Tricks to Make Things Work](#)

- [Conclusion](#)
  - [So, which method is State-of-the-Art?](#)
- [References](#)



---------------------------------------------------------------------------------


<a name="problem-setting"></a>
## Problem Setting of Supervised Metric Learning

Generally speaking, Deep Metric Learning is a group of techniques that aims to measure the similarity between data samples. More specifically, for a set of data points $$\mathcal{X}$$ and their corresponding labels $$\mathcal{Y}$$, the goal is to train an embedding neural model $$f_{\theta}(\cdot)\, \colon \mathcal{X} \to \mathbb{R}^n$$ (where $$\theta$$ are learned weights) together with a distance $$\mathcal{D}\, \colon \mathbb{R}^n \to \mathbb{R}$$ (which is usually fixed beforehand), so that the combination $$\mathcal{D}\left(f_{\theta}(x_1), f_{\theta}(x_2)\right)$$ produces small values if the labels $$y_1, y_2 \in \mathcal{Y}$$ of the samples $$x_1, x_2 \in \mathcal{X}$$ are equal, and larger values if they aren't.

Thus, the Deep Metric Learning problem boils down to just choosing the architecture for $$f_{\boldsymbol{\theta}}$$ and choosing the loss function $$\mathcal{L}(\theta)$$ to train it with. One might wonder why we cannot just use the classification objective for the metric learning problem? In fact, the [Softmax loss][softmax_loss] is also a valid objective for metric learning, albeit inferior to other objectives as we will see later in this article.


[softmax_loss]: https://cs231n.github.io/linear-classify/#softmax
[softmax_func_vs_softmax_loss]: https://medium.com/@liangjinzhenggoon/the-difference-between-softmax-and-softmax-loss-173d385120c2


---------------------------------------------------------------------------------


<a name="common-approaches"></a>
## Common Approaches

I will glance throught the most common approaches very quickly for two reasons:

- The methods described here are already covered in other tutorials, videos, and blog posts online in great detail. I highly recommend the great survey by [Kaya & Bilge (2019)][deep_metric_learning_survey].
- The methods that I will describe in the next section outperforms these approaches in most cases, so I have no motivation to delve too deep into the details in this section.

From the bird's eye view, each iteration of the approaches for Deep Metric Learning described in this section consists of 3 steps:

1. <a name="common-approaches-step-1"></a> Run the inference step for $$f_\theta$$ on a few data samples.
2. Apply a loss function on the resulting feature vectors from [step 1](#common-approaches-step-1).
3. Perform [Backpropagation][backpropagation_wiki] and update the network's weights.

Sometimes, there's an additional step of choosing the useful training data samples for the next step using Negative Samples Mining. The distance function for these approaches is usually fixed as $$l_2$$ metric:

$$
\begin{equation*}
\mathcal{D}\left(p, q\right) = \|p - q\|_2^2 = \sqrt{\sum_{i=1}^n \left(p_i - q_i\right)^2}
\end{equation*}
$$


{% capture imblock1 %}
    {{ site.url }}/articles/images/2020-12-11-deep-metric-learning-survey/metric_losses.png
{% endcapture %}
{% capture imcaption1 %}
  Fig 1: A visual overview of different metric loss functions (Image source: [Kaya & Bilge, 2019](https://www.mdpi.com/2073-8994/11/9/1066/htm))
{% endcapture %}
{% include gallery images=imblock1 cols=1 caption=imcaption1 %}


For the ease of notation, let's denote $$\mathcal{D}_{f_\theta}(x_1, x_2)$$ as a shortcut for $$\mathcal{D} \left( f_\theta(x_1), f_\theta(x_2) \right)$$, where $$x_1, x_2 \in \mathcal{X}$$ are samples from the dataset. Also, for some condition $$A$$, let's denote $$\unicode{x1D7D9}_A$$ as the identity function that is equal to $$1$$ if $$A$$ is true, and $$0$$ otherwise.


<a name="contrastive-loss"></a>
### Contrastive Loss

This is a classic loss function for metric learning. **Contrastive Loss** is one of the simplest and most intuitive training objectives. Let $$x_1, x_2$$ be some samples in the dataset, and $$y_1, y_2$$ are their corresponding labels. The loss function is then defined as follows:

$$
\begin{equation*}
\mathcal{L}_\text{contrast} = 
\unicode{x1D7D9}_{y_1 = y_2}
\mathcal{D}_{f_\theta}(x_1, x_2)
+
\unicode{x1D7D9}_{y_1 \ne y_2}
\max\left(0, \alpha - \mathcal{D}_{f_\theta}(x_1, x_2)\right)
\end{equation*}
$$

where $$\alpha$$ is the margin. The reason we need a margin value is because otherwise, our network $$f_\theta$$ will learn to "cheat" by mapping all $$\mathcal{X}$$ to the same point, making distances between any samples to be equal to zero. [Here][contrastive_explained] and [here][contrastive_explained_2] are very great in-depth explanation for this loss function.


<a name="triplet-loss"></a>
### Triplet Loss

**Triplet Loss** [(Schroff et al. 2015)][triplet_loss_paper] is by far the most popular and widely used loss function for metric learning. It is also featured in [Andrew Ng's deep learning course][andrew_ng_triplet_loss].

Let $$x_a, x_p, x_n$$ be some samples from the dataset and $$y_a, y_p, y_n$$ be their corresponding labels, so that $$y_a = y_p$$ and $$y_a \ne y_n$$ ($$x_a$$ is called **anchor** sample, $$x_p$$ is called **positive** sample because it has the same label as $$x_a$$, and $$x_n$$ is called **negative** sample because it has a different label. It is defined as:

$$
\begin{equation*}
\mathcal{L}_\text{triplet} =
\max\left(0,
\mathcal{D}_{f_\theta}(x_a, x_p) -
\mathcal{D}_{f_\theta}(x_a, x_n)
+ \alpha\right)
\end{equation*}
$$

where $$\alpha$$ is the margin to discourage our network $$f_\theta$$ to map the whole dataset $$\mathcal{X}$$ to the same point. The key ingredient to make Triplet Loss work in practice is **Negative Samples Mining** &mdash; on each training step, we sample such triplets that such triplets $$x_a, x_p, x_n$$ that satisfies $$\mathcal{D}_{f_\theta}(x_a, x_n) < \mathcal{D}_{f_\theta}(x_a, x_p) + \alpha$$, i.e. the samples that our network $$f_\theta$$ fails to discriminate or is not able to discriminate with high confidence. You can find in-depth description and analysis of Triplet Loss in [this awesome blog post][triplet_loss_explained].

Triplet Loss is [still being widely used][paperswithcode_tripletloss], despite recent advances in Metric Learning (which we will learn about in the next section), due to its relative effectiveness and the wide availability of code samples online for all deep learning frameworks.


<a name="quadruplet-loss"></a>
### Quadruplet Loss

One of the limitations for Triplet Loss is that it does not care about inter- and intra-class variations of the feature vectors $$f_\theta(x)$$. **Quadruplet Loss** ([Chen et al. 2017][quadruplet_loss_paper]) was developed to address exactly that &mdash; it makes inter-class variation larger and intra-class variation smaller.

Similar to Triplet Loss, Quadruplet Loss also have $$x_a, x_p, x_n$$ (anchor, positive, negative) samples with labels $$y_a, y_p, y_n$$ with $$y_a = y_p$$ and $$y_a \ne y_n$$. However, it has one more sample $$x_s$$ with label $$y_s = y_a$$, which is used for regularizing the class variations. The loss value is defined as follows:

$$
\begin{eqnarray*}
\mathcal{L}_\text{quadruplet}
= &
\max\left(0,
\mathcal{D}_{f_\theta}(x_a, x_p) -
\mathcal{D}_{f_\theta}(x_a, x_s)
+ \alpha_1\right) \\
+ &
\max\left(0,
\mathcal{D}_{f_\theta}(x_a, x_s) -
\mathcal{D}_{f_\theta}(x_a, x_n)
+ \alpha_2\right)
\end{eqnarray*}
$$

where $$\alpha_1$$ is intra-class margin that prevents $$f_\theta$$ to collapse samples from the same class to a single point, and $$\alpha_2$$ is inter-class margin with the same purpose as $$\alpha$$ in triplet and contrastive losses. You can find more about the intricacy of the quadruplet loss in comparison to the triplet loss in [this article][triplet_vs_quadruplet].






[backpropagation_wiki]: https://en.wikipedia.org/wiki/Backpropagation
[deep_metric_learning_survey]: https://www.mdpi.com/2073-8994/11/9/1066/htm
[contrastive_explained]: https://medium.com/@maksym.bekuzarov/losses-explained-contrastive-loss-f8f57fe32246
[contrastive_explained_2]: https://gombru.github.io/2019/04/03/ranking_loss/
[triplet_loss_paper]: https://arxiv.org/abs/1503.03832
[andrew_ng_triplet_loss]: https://www.youtube.com/watch?v=LN3RdUFPYyI
[triplet_loss_explained]: https://omoindrot.github.io/triplet-loss
[paperswithcode_tripletloss]: https://paperswithcode.com/method/triplet-loss
[quadruplet_loss_paper]: https://arxiv.org/abs/1704.01719
[triplet_vs_quadruplet]: https://towardsdatascience.com/how-to-choose-your-loss-when-designing-a-siamese-neural-net-contrastive-triplet-or-quadruplet-ecba11944ec
