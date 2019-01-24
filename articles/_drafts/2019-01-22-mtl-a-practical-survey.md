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

If you found yourself in a strange situation, where you want your Neural Network to do several things at once, i.e. detect objects, predict depth, predict surface normals &ndash; you are having a Multi-Task Learning (MTL) problem.

{% capture imblock1 %}
    {{ site.url }}/articles/images/2019-01-22-mtl-a-practical-survey/teaser.png
{% endcapture %}
{% include gallery images=imblock1 cols=1 %}

Traditionally, the development of Multi-Task Learning was aimed to improve the generalization of multiple task predictors by jointly training them, while allowing some sort of knowledge transfer and between them [(Caruana, 1997)][caruana1997]. If you, for example, train a *surface normal prediction* model and *depth prediction* model together, they will definitely share mutually-benefitial features together [(Eigen et al. 2015)][eigen-dnl]. This motivation is clearly inspired by natural intelligence &mdash; living creatures in an remarkable way can easily learn a task by leveraging the knowledge from other tasks. A broader generalization of this idea is called [Lifelong Learning][lifelong-learning], in which different tasks are not even learned simultaneously.

However, screw these academic stuffs, **we are [engineers][im-an-engineer]**! Why should we care about leveraging diverse features from different tasks when we can just *slap that AI* with more data ([kagglers][kagglers] doesn't count here)? If you are an engineer from *big AF* companies like Google, Samsung, Microsoft, etc. then *Hell Yeah* you've got a *ton* of cash to *splash out* on labellers! Just hire them to get more data. Thus our main motivation for Multi-Task learning will be the following:

-  **To optimize multiple objectives at once.** For instance, in [GANs][gan], it is shown in various tasks that often incorporating additonal loss functions can yield much better results ([Isola et al. 2017][pix2pix]; [Wang et al. 2018][vid2vid]). A [regularization term][regularization] can also be considered as additional objective.

-  **To reduce the cost of running multiple models**. My Korean boss always yells at my team *"we need faster CNNs!"* in his typical asian accent (no offense, I'm also asian). How can we further speed up the 5 models that are already optimized both in size and speed by more than 40 times? Oh, yeah, we can merge all of them into a single Multi-Task model!

In this article, I will only focus on the two motivation above. The *moto* of this article is: *simple as instant noodle, easy to implement, and effective as heck!* For a more comprehensive survey that focuses on the *mutually-benefitial sharing* aspect of Multi-Task Learning, it is recommended to read [Ruder's (2017)][ruder-mtl] paper.

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






## 1. Too many losses? MOO to the rescue!

*Sorry for bad puns.* The methods of Multi-Objective Optimization (MOO) can help you learn multiple objectives better (here and after we will use the terms *objective* and *task* interchangeably).

### 1.1. Forming the formal formulation

Consider the input space $$\mathcal{X}$$ and collection of task-specific output spaces $$\{ \mathcal{Y}^t \} _ {t \in [ T ]}$$, where $$T$$ is the number of tasks. Without loss of generality, we consider a Multi-Task Neural Network

$$
\begin{equation}
f(x, \theta) = \left( f^1(x; \theta^{sh}, \theta^1), \dots, f^T(x; \theta^{sh}, \theta^T) \right)\,,
\tag{1.1.1} \label{eq:mtnn}
\end{equation}
$$

where $$\theta^{sh}$$ are network parameters shared between tasks, and $$\theta^t$$ are task-specific parameters. Task-specific outputs $$f^t(x; \theta^{sh}, \theta^t)$$ maps the inputs from $$\mathcal{X}$$ to task-specific outputs $$\mathcal{Y}^t$$. In Multi-Task Learning literature, the following formal summation formulation of the problem often yields:

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
\end{split}
\tag{1.1.2} \label{eq:mtloss}
\end{equation}
$$

where $$\hat{\mathcal{L}} _ t(\cdot)$$ is an empirical task-specific loss for $$t$$-th task defined as the average loss accross the whole dataset $$\hat{\mathcal{L}} (\theta^{sh}, \theta^{t}) \triangleq \frac{1}{N} \sum_i \mathcal{L} ( f^t(x_i; \theta^{sh}, \theta^{t}), y_i^t )$$, where $$y_i^t \in \mathcal{Y}^t$$ is the ground truth of the $$t$$-th task for $$i$$-th sample in the dataset.

### 1.2. The $$\lambda_t$$ Balancing Problem

The obvious question from a first glance at \eqref{eq:mtloss} is: how to set the weight coefficient $$\lambda_t$$ for $$t$$-th task? Usually, setting $$\lambda_t$$ to $$1$$ is not a good idea: for different tasks, the magnitude of loss functions, as well as the magnitudes of gradients, might be very different. In an unbalanced setting, the magnitude of the gradients of one task might be so large that it makes the gradients from other tasks insignificant &mdash; i.e. the model will only learn one task while ignoring the other tasks. Even the brute-force approach (e.g. [grid search][grid-search]) may not find optimal values of $$\lambda_t$$ since they pre-sets the values at the beginning of training, while optimal values may change over time.

If we allow $$\lambda_t$$ to change dynamically during training, which is a desirable behaviour, additional challenges occurs. A basic justification is that in this setting, it is not possible to define global optimality for optimization objective \eqref{eq:mtloss}. Consider two sets of solutions $$\theta$$ and $$\bar{\theta}$$ such that

$$
\begin{equation}
\mathcal{L}^{t_1} (\theta^{sh}, \theta^{t_1}) < \mathcal{L}^{t_1} (\bar{\theta}^{sh}, \bar{\theta}^{t_1})
\quad\text{and}\quad
\mathcal{L}^{t_2} (\theta^{sh}, \theta^{t_2}) > \mathcal{L}^{t_2} (\bar{\theta}^{sh}, \bar{\theta}^{t_2})
\tag{1.2.1} \label{eq:mtwtf}
\end{equation}
$$

for some tasks $$t_1$$ and $$t_2$$. In other words, solution $$\theta$$ is better for task $$t_1$$ whereas $$\bar{\theta}$$ is better for $$t_2$$. It is not possible to compare them without explicit pairwise improtance of tasks, which is typically not available.

### 1.3. Instant Noodle in case of Multiple Losses

**Description**. Recent works attacks this problem by presenting a heuristic, according to which the coefficients $$\lambda_t$$ are chosen: [Chen et al. (2017)][chen2017] manipulates them in such a way that the gradients are approximately normalized; [Kendall et al. (2018)][kendall2018] models the network output's homoscedastic uncertainty with a probabilistic model.
However, heuristic are too unreliable &mdash; there is no guarantee that the chosen weights will be of any good. A true *instant noodle* approach should be reliable. That's where the latest paper [Sener and Koltun (2018)][mtl-as-moo] presented on [NeurIPS][nips2018] this year comes to rescue.

The objective \eqref{eq:mtloss} is re-formulated in a sense of [multi-objective optimization][moo]: optimizing a collection of possibly conflicting objectives. The MTL objective is then specified using a vector-valued loss $${L}$$:

$$
\begin{equation}
{L}(\theta^{sh}, \theta^1,\ldots,\theta^T) =
\left( \hat{\mathcal{L}}^1(\theta^{sh},\theta^1), \ldots,  \hat{\mathcal{L}}^T(\theta^{sh},\theta^T) \right)^\intercal
\tag{1.3.1} \label{eq:vecloss}
\end{equation}
$$

The main motivation to re-formulate the objective is the conflict \eqref{eq:mtwtf}. This vector objective will not have a strong order minimum, but we can still talk about a weaker sort of minimality &mdash; the [Pareto optimality][pareto-opt].

> **Definition (Pareto optimality).**
A solution $$\theta$$ dominates a solution $$\bar{\theta}$$ if $$\hat{\mathcal{L}}^t(\theta^{sh},\theta^t)  \leq \hat{\mathcal{L}}^t(\bar{\theta}^{sh},\bar{\theta}^t)$$ for all tasks $$t$$ and $$L(\theta^{sh}, \theta^1,\ldots,\theta^T) \neq L(\bar{\theta}^{sh}, \bar{\theta}^1,\ldots,\bar{\theta}^T)$$;
A solution $$\,\theta^\star$$ is called Pareto optimal if there exists no solution $$\,\theta$$ that dominates $$\,\theta^\star$$.

The multi-objective optimization can be solved to local minimality (in a Pareto sense) via Multiple Gradient Descent algorithm. 

[moo]: https://en.wikipedia.org/wiki/Multi-objective_optimization
[nips2018]: https://nips.cc/Conferences/2018/Schedule?type=Poster
[mtl-as-moo]: https://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf
[homoscedastic]: https://en.wikipedia.org/wiki/Homoscedasticity
[grid-search]: https://scikit-learn.org/0.18/auto_examples/model_selection/grid_search_digits.html
[kendall2018]: http://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
[chen2017]: https://arxiv.org/abs/1711.02257
[pareto-opt]: https://en.wikipedia.org/wiki/Pareto_efficiency






## Appendix A: other stuffs that ain't the best instant noodle

In this section, I will describe the other approaches that I had experience with (i.e. have been used in practice), but won't recommend them for others. On their own, they are not bad, just not the best (comparing to methods described above).

### A.1. GradNorm: Gradients Normalization for Adaptive $$\lambda_t$$ Balancing

**Description.** This approach ([Chen et al. 2017][chen2017]), presented on [ICML 2018][icml2018], attempts to regularize the magnitude of gradients to roughly the same scale during a backward pass. The motivation is simple: the raw value of the loss components of \eqref{eq:mtloss} does not reflect how much your model "cares" about that component; e.g. an $$L_1$$ loss can report arbitrarily large loss values based on loss scale; the gradient's magnitude is what actually matters.

At each training step $$s$$, the average gradient norm $$\bar{g} _ {\theta}(s)$$ is chosen as a common scale for gradients. The relative inverse learning rate of task $$t$$, $$r_t(s)$$ is used to balance the gradients &mdash; the higher the value of it, the higher the gradient magnitudes should be for task $$t$$ in order to encourage the task to train more quickly. The desired gradient magnitude for each task $$t$$ is therefore:

$$
\begin{equation}
g^t _ \theta (s) \mapsto \bar{g} _ \theta (s) \times r_t(s)^\alpha
\tag{A.1.1} \label{eq:gradnorm1}
\end{equation}
$$

where $$\alpha$$ is an additional hyperparameter. It sets the strength of the restoring force which pulls tasks back to a common training rate. If the tasks are very different leading to dramatically different learning dynamics between tasks, the higher value of $$\alpha$$ should be used. At each training step $$s$$, we then encourage the gradients to be closer to the desired magnitude:

$$
\begin{equation}
\mathcal{L} _ {\text{grad}} (s, \lambda_1, \dots, \lambda_T) = \sum _ {t=1}^T {\left\vert g^t _ \theta (s) - \bar{g} _ \theta (s) \times r_t(s)^\alpha \right\vert} _ {1}
\tag{A.1.2} \label{eq:gradnorm2}
\end{equation}
$$

The loss \eqref{eq:gradnorm2} is then differentiated *only w.r.t.* $$\lambda_t$$ and then updated via the standard update in [backpropagation][backprop] algorithm. In other words, the weight coefficients $$\lambda$$ are used to manipulate the gradient's norms and move it towards the desired target \eqref{eq:gradnorm1}.

### A.2. Using uncertainties of losses $$\mathcal{L}^t(\cdot)$$ to balance $$\lambda_t$$

**Description.** On [CVPR 2018][cvpr2018], another approach was proposed by [Kendall et al. (2018)][kendall2018] that models the network output's [homoscedastic][homoscedastic] uncertainty with a probabilistic model. Let $$f^W(x)$$ be the output of the network with weights $$W$$ on input $$x$$. For single-task, we model the network output uncertainty with a density function $$p\left( y \vert f^W(x) \right)$$ (how the true answer is likely to be $$y$$, given network's response). In the case of multiple network outputs $$y_1, \dots y_K$$, we obtain the following multi-task likelihood:

$$
\begin{equation}
\tag{A.2.1} \label{eq:mtlikelihood}
p \left( y_1, \dots y_K \vert f^W(x) \right) = p\left(y_1 \vert f^W(x)\right) \dots p\left(y_K \vert f^W(x)\right)
\end{equation}
$$

[cvpr2018]: http://cvpr2018.thecvf.com/
[icml2018]: https://icml.cc/Conferences/2018/Schedule?type=Poster
[backprop]: https://en.wikipedia.org/wiki/Backpropagation
[kendall2018]: http://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
[chen2017]: https://arxiv.org/abs/1711.02257
