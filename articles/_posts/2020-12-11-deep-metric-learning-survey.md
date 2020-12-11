---
layout: post
permalink: /articles/:title
type: "article"
title: "Deep Metric Learning: a (Long) Survey"
image:
  feature: "/articles/images/2020-12-11-deep-metric-learning-survey/metric_losses.png"
  display: false
commits: https://github.com/hav4ik/hav4ik.github.io/commits/master/articles/_posts/2020-02-08-how-to-dominate-on-tech-interviews.md
tags: [deep-learning, survey]
excerpt: "In this post, I'll briefly go over the common approaches for Deep Metric Learning, as well as the new methods proposed in recent years."
comments: true
hidden: true
---

One of the most amazing aspects of the human's visual system is the ability to recognize similar objects and scenes. We don't need hundreds of photos of the same face to be able to differentiate it among thousands of other faces that we've seen. We don't need thousands of images of the Eiffel Tower to recognize that unique architecture landmark when we visit Paris. Is it possible to design a Deep Neural Network with the similar ability to tell which objects are visually similar and which ones are not? That's essentially what **Deep Metric Learning** attempts to solve. Advanced readers may immediately recognize by the description that this topic is intimately related to One-Shot Learning.

Generally speaking, Metric Learning is a group of techniques that aims to measure the similarity between data samples. More specifically, for a set of data points $$\boldsymbol{\mathcal{X}}$$, the goal is to train an embedding model $$f_{\boldsymbol{\theta}}(\cdot)\, \colon \boldsymbol{\mathcal{X}} \to \mathbb{R}^n$$ (where $$\boldsymbol{\theta}$$ are learned weights) together with a distance $$\mathcal{D}\, \colon \mathbb{R}^n \to \mathbb{R}$$, that minimizes the value of $$\mathcal{D}\left(f_{\boldsymbol{\theta}}(\boldsymbol{x}_1), f_{\boldsymbol{\theta}}(\boldsymbol{x}_2)\right)$$ if samples $$\boldsymbol{x}_1, \boldsymbol{x}_2 \in \boldsymbol{\mathcal{X}}$$ are similar in some sense, and maximizes it if they aren't.

Although this blog post is mainly about Deep Metric Learning and is self-sufficient by its own, it would be benefitial for you to consider getting familiar with traditional Metric Learning methods (i.e. without Neural Networks) to develop a broader understanding on this topic. I highly recommend the [introductory guides on Metric Learning][sklearn_metric_learning_guide] as a starter. If you want to get into the formal mathematical side of things, I recommend the tutorial by [Diaz et al. (2020)][diaz_tutorial_metric_math]. More advanced Metric Learning methods includes the popular [t-SNE (van der Maaten & Hinton, 2008)][tsne_paper] and the new shiny [UMAP (McInnes et al., 2018)][umap_paper] that everybody uses nowadays for data clustering and visualization.

The methods described in this blog post are divided into two groups: **Common Methods** that were very popular until the recent takeover of angular margin methods, and **State-of-the-Art Methods** that I recommend using (as of 2020). At the end of this article, I will discuss various Metric Learning problems and which methods have worked in each case. I will try to (hopefully) keep this article up to date with the latest developments.


---------------------------------------------------------------------------------


- [Common Approaches](#)
  - [Metric Loss Functions](#)
    - [Siamese Networks](#)
    - [Contrastive Loss](#)
    - [Triplet Loss](#)
  - [Angular Margin Methods](#)
    - [CosFace, ArcFace, and SphereFace](#)
    - [AdaCos &mdash; Adaptive $$s$$ Parameter](#)
    - [Sub-Center ArcFace](#)
    - [ArcFace with Dynamic Margin (Unpublished)](#)
- [Getting Practical](#)
  - [Case studies: what works, and what doesn't?](#)
    - [Humpack Whale Challenge](#)
    - [Google Landmarks Challenge](#)
    - [Face Recognition](#)
  - [Practical Tricks to Make Things Work](#)
- [Conclusion](#)
  - [So, which method is State-of-the-Art?](#)
- [References](#)


---------------------------------------------------------------------------------


## Common Approaches

{% capture imblock1 %}
    {{ site.url }}/articles/images/2020-12-11-deep-metric-learning-survey/metric_losses.png
{% endcapture %}
{% capture imcaption1 %}
  Fig 1: An overview of different metric loss functions [(Source: Kaya & Bilge, 2019)](https://www.mdpi.com/2073-8994/11/9/1066/htm)
{% endcapture %}
{% include gallery images=imblock1 cols=1 caption=imcaption1 %}



[sklearn_metric_learning_guide]: http://contrib.scikit-learn.org/metric-learn/introduction.html
[diaz_tutorial_metric_math]: https://arxiv.org/abs/1812.05944
[tsne_paper]: https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf
[umap_paper]: https://arxiv.org/abs/1802.03426
