---
layout: post
permalink: /articles/:title
type: "article"
title: 'An "Instant Noodle" survey to Multi-Task Learning'
image:
  feature: "articles/images/2019-01-22-mtl-a-practical-survey/featured.png"
  display: false
tags: [survey, deep learning]
excerpt: "A survey on stuffs that almost always works as-is from the box and are easy to implement &ndash; just like instant noodle!"
comments: true
---



> Industry ML/AI [Engineers][im-an-engineer] are **weirdos**: they hunt the latest papers, they crave for State-of-the-Art, they talk about craziest ideas that might enhance the whole field, yet they are too lazy to implement harder-than-a-resnet idea. This article is written **for** such [Engineers][im-an-engineer].

[im-an-engineer]: https://www.youtube.com/watch?v=rp8hvyjZWHs




## Introduction

If you found yourself in a strange situation, where you want your Neural Network to do several things at once, i.e. detect objects, predict depth, predict surface normals &ndash; you are having a Multi-Task Learning problem.

{% capture imblock1 %}
    {{ site.url }}/articles/images/2019-01-22-mtl-a-practical-survey/teaser.png
{% endcapture %}
{% include gallery images=imblock1 cols=1 %}

Traditionally, the development of Multi-Task Learning was aimed to improve the generalization of multiple task predictors by jointly training them, while allowing some sort of knowledge transfer and between them [(Caruana, 1997)][caruana1997]. If you, for example, train a *surface normal prediction* model and *depth prediction* model together, they will definitely share mutually-benefitial features together [(Eigen et al. 2015)][eigen-dnl]. This motivation is clearly inspired by natural intelligence &mdash; living creatures in an remarkable way can easily learn a task by leveraging the knowledge from other tasks. A broader generalization of this idea is called [Lifelong Learning][lifelong-learning], in which different tasks are not even learned simultaneously.

However, screw these academic stuffs, **we are [engineers][im-an-engineer]**! Why should we care about leveraging diverse features from different tasks when we can just *slap that AI* with more data ([kagglers][kagglers] doesn't count here)? If you are an engineer from *big AF* companies like Google, Samsung, Microsoft, etc. then *Hell Yeah* you've got a *ton* of cash to *splash out* on labellers! Just hire them to get more data. Thus our main motivation for Multi-Task learning will be the following:

-  **To optimize multiple objectives at once.** For instance, in [GANs][gan], it is shown in various tasks that often incorporating additonal loss functions can yield much better results ([Isola et al. 2017][pix2pix]; [Wang et al. 2018][vid2vid]). A [regularization term][regularization] can also be considered as additional objective.

-  **To reduce the cost of running multiple models**. My Korean boss always yells at my team *"we need faster CNNs!"* in his typical asian accent (no offense, I'm also asian). How can we further speed up the 5 models that are already optimized both in size and speed by more than 40 times? Oh, yeah, we can merge all of them into a single Multi-Task model!

In this article, we will only focus on the two motivation above. The *moto* of this article is: *simple as instant noodle, easy to implement, and effective as heck!* For a more comprehensive survey that focuses on the *mutually-benefitial sharing* aspect of Multi-Task Learning, it is recommended to read [Ruder's (2017)][ruder-mtl] paper.

[vid2vid]: https://tcwang0509.github.io/vid2vid/
[pix2pix]: https://phillipi.github.io/pix2pix/
[gan]: https://arxiv.org/abs/1406.2661
[kagglers]: https://www.kaggle.com/umeshnarayanappa/the-world-needs-kaggle
[regularization]: https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/
[eigen-dnl]: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf
[lifelong-learning]: http://lifelongml.org/
[caruana1997]: https://link.springer.com/article/10.1023/A:1007379606734
[im-an-engineer]: https://www.youtube.com/watch?v=rp8hvyjZWHs
[ruder-mtl]: https://arxiv.org/abs/1706.05098




## Too many losses? MOO to the rescue!

The methods of Multi-Objective Optimization (MOO) can help you learn multiple objectives better (here and after we will use the terms *objective* and *task* interchangeably). Consider the input space $$\mathcal{X}$$ and collection of task-specific output spaces $$\{ \mathcal{Y}^t \} _ {t \in [ T ]}$$, where $$T$$ is the number of tasks. Without loss of generality, we consider a Multi-Task Neural Network $$f(x, \theta) = ( f^1(x; \theta^{sh}, \theta^1), \dots, f^T(x; \theta^{sh}, \theta^T) )$$, where $$\theta^{sh}$$ are network parameters shared between tasks, and $$\theta^t$$ are task-specific parameters. Task-specific outputs $$f^t(x; \theta^{sh}, \theta^t)$$ maps the inputs from $$\mathcal{X}$$ to task-specific outputs $$\mathcal{Y}^t$$. In Multi-Task Learning literature, the following formal summation formulation of the problem often yields:

$$
\begin{equation}
\begin{split}
\text{minimize}
\enspace
\sum _ {t=1}^T {\lambda_t \hat{\mathcal{L}} _ t(\theta^{sh}, \theta^{t})}
\quad\quad
\text{w.r.t.}
\enspace
\theta^{sh}, \theta^{1}, \dots, \theta^{T}
\,,
\end{split} \tag{1} \label{eq:mtloss}
\end{equation}
$$

where $$\hat{\mathcal{L}} _ t(\cdot)$$ is an empirical task-specific loss for $$t$$-th task defined as the average loss accross the whole dataset $$\hat{\mathcal{L}} (\theta^{sh}, \theta^{t}) \triangleq \frac{1}{N} \sum_i \mathcal{L} ( f^t(x_i; \theta^{sh}, \theta^{t}), y_i^t )$$, where $$y_i^t$$ is the ground truth of the $$t$$-th task for $$i$$-th sample in the dataset.

**Balancing Problem.** The obvious question from a first glance at \eqref{eq:mtloss} is: how to set the weight coefficient $$\lambda_t$$ for $$t$$-th task? Usually, setting $$\lambda_t$$ to $$1$$ is not a good idea: for different tasks, the magnitude of loss functions, as well as the magnitudes of gradients, might be very different. In an unbalanced setting, the magnitude of the gradients of one task might be so large that it makes the gradients from other tasks insignificant &mdash; i.e. the model will only learn one task while ignoring the other tasks. Even the brute-force approach (e.g. grid search) may not find optimal values of $$\lambda_t$$ since they pre-sets the values at the beginning oftraining, while optimal values may change over time.

Recent works attacks this problem by presenting a heuristic, according to which the coefficients $$\lambda_t$$ are chosen: [Chen et al. (2017)][chen2017] manipulates them in such a way that the gradients are approximately normalized; [Kendall et al. (2018)][kendall2018] models the network output's [homoscedastic][homoscedastic] uncertainty with a probabilistic model. But hey! Heuristic are too unreliable &mdash; there is no guarantee that the chosen weights will be of any good. A true *instant noodle* approach should be reliable. That's where the latest paper [Sener and Koltun (2018)][mtl-as-moo] presented on [NeurIPS][nips2018] this year comes to rescue.

[nips2018]: https://nips.cc/Conferences/2018/Schedule?type=Poster
[mtl-as-moo]: https://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf
[kendall2018]: http://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
[chen2017]: https://arxiv.org/abs/1711.02257
[homoscedastic]: https://en.wikipedia.org/wiki/Homoscedasticity



## Don't have all $$y_i^t$$ for each $$x_i$$? Hallucinate it!




## Need to merge models together? Here is some glue for ya!
