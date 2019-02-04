---
layout: post
permalink: /articles/:title
type: "article"
title: 'Guide to "Instant Noodles" in Multi-Task Learning'
image:
  feature: "articles/images/2019-01-22-mtl-a-practical-survey/featured.png"
  display: false
tags: [survey, lecture notes, deep learning]
excerpt: "An in-depth survey on stuffs that works like a charm as-is right from the box and are easy to implement &ndash; just like instant noodle!"
comments: true
---



> Industry ML/AI [Engineers][im-an-engineer] are [**weirdos**][weirdo]: they hunt the latest papers, they crave for State-of-the-Art, they talk about craziest ideas that might enhance the whole field, yet they are too lazy to implement harder-than-a-resnet idea. This article is written **for** such [Engineers][im-an-engineer].

[weirdo]: https://www.urbandictionary.com/define.php?term=weirdo
[im-an-engineer]: https://www.youtube.com/watch?v=rp8hvyjZWHs






## Introduction

If you found yourself in a strange situation, where you want your Neural Network to do several things at once &mdash; don't worry, you are just having a Multi-Task Learning (MTL) problem. In this article, I will discuss the challenges of MTL, make a survey on effective solutions to them, and propose minor improvements of my own to the readers.

{% capture imblock1 %}
    {{ site.url }}/articles/images/2019-01-22-mtl-a-practical-survey/teaser.png
{% endcapture %}
{% include gallery images=imblock1 cols=1 %}

Traditionally, the development of Multi-Task Learning was aimed to improve the generalization of multiple task predictors by jointly training them, while allowing some sort of knowledge transfer and between them [(Caruana, 1997)][caruana1997]. If you, for example, train a *surface normal prediction* model and *depth prediction* model together, they will definitely share mutually-benefitial features together [(Eigen et al. 2015)][eigen-dnl]. This motivation is clearly inspired by natural intelligence &mdash; living creatures in an remarkable way can easily learn a task by leveraging the knowledge from other tasks. A broader generalization of this idea is called [Lifelong Learning][lifelong-learning], in which different tasks are not even learned simultaneously.

However, screw these academic stuffs, **we are [engineers][im-an-engineer]**! Why should we care about leveraging diverse features from different tasks when we can just *slap that AI* with more data ([kagglers][kagglers] doesn't count here)? If you are an engineer from *big [AF][as-fuck]* companies like Google, Samsung, Microsoft, etc. then *Hell Yeah* you've got a *ton* of cash to *splash out* on labellers! Just hire them to get more data. Thus, our main motivation for Multi-Task learning are less obvious, but even more important from an engineering and consumer standpoint:

-  **To optimize multiple objectives at once.** For instance, in [GANs][gan], it is shown in various tasks that often incorporating additonal loss functions can yield much better results ([Isola et al. 2017][pix2pix]; [Wang et al. 2018][vid2vid]). A [regularization term][regularization] can also be considered as additional objective.

-  **To reduce the cost of running multiple models**. My Korean boss always yells at my team *"we need faster CNNs!"* in his typical asian accent (no offense, I'm also asian). How can we further speed up the 5 models that are already optimized both in size and speed by more than 40 times? Oh, yeah, we can merge all of them into a single Multi-Task model!

In this article, I will only focus on the two motivation above. The *motto* of this article is: *simple as instant noodle*, i.e. easy to implement and effective as heck! This blog post serves me as a lecture note as well, so here you will find more in-depth theoretical stuffs (that normally only the full papers have) than a typical survey will do.

{% comment %}
This article will be structured as following:

-  In [**Section 1**][section-1], I will outline the challenges of optimizing multiple objectives at once and describe a cool paper from [NeurIPS 2018][nips2018] that fits our motto of *instant noodleness.* Then, in [**subsection 1.4**][subsection-1-4], I will make some remarks and propose some modifications of my own to it that generalizes the approach to more complicated architectures that I used in practical applications.

-  In [**Section 2**][section-2], I will discuss the challenges when for each input sample, we don't have ground truth to each of the tasks to it &mdash; a very common situation in MTL. Then, I will describe the *next noodle for ya* &mdash; a simple yet effective solution proposed on [WACV 2018][wacv2018]. In [**subsection 2.3**][subsection-2-3], I will also expand this idea to a more *industrial* setting, and propose minor improvements using my experience in [Knowledge Distillation][knowledge-distillation].

-  In [**Section 3&frac12;**][section-3-and-a-half], I will give a brief survey on different architectures for MTL that might be useful for you. Most of the case, however, the simplest architecture will still do the job.

-  In [**Appendix A**][appendix-a], I will outline other methods that got their way to top conferences such as *CVPR*, *NIPS*, *ICML*, but are not that good in practise to be qualified as *instant noodle*. Most [engineers][im-an-engineer] won't need that, unless being forced by their bosses to increase the accuracy by $$0.01\%$$.
{% endcomment %}

For a more comprehensive survey that gives you a bird-eye-view on a whole field of MTL and focused on the *mutually-benefitial sharing* aspect of Multi-Task Learning, it is recommended to read [Ruder's (2017)][ruder-mtl] paper.

[vid2vid]: https://tcwang0509.github.io/vid2vid/
[pix2pix]: https://phillipi.github.io/pix2pix/
[gan]: https://arxiv.org/abs/1406.2661
[kagglers]: https://www.kaggle.com/umeshnarayanappa/the-world-needs-kaggle
[regularization]: https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/
[3-5-anonymous]: http://lurkmore.to/3,5_%D0%B0%D0%BD%D0%BE%D0%BD%D0%B8%D0%BC%D1%83%D1%81%D0%B0
[eigen-dnl]: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf
[lifelong-learning]: http://lifelongml.org/
[caruana1997]: https://link.springer.com/article/10.1023/A:1007379606734
[im-an-engineer]: https://www.youtube.com/watch?v=rp8hvyjZWHs
[ruder-mtl]: https://arxiv.org/abs/1706.05098
[nips2018]: https://nips.cc/Conferences/2018/Schedule?type=Poster
[wacv2018]: http://wacv18.wacv.net/
[knowledge-distillation]: https://medium.com/neural-machines/knowledge-distillation-dc241d7c2322
[section-1]: #section1
[subsection-1-4]: #subsection14
[section-2]: #section2
[subsection-2-3]: #subsection23
[section-3-and-a-half]: #section3andahalf
[appendix-a]: #appendixa
[as-fuck]: https://www.urbandictionary.com/define.php?term=AF






<br><br>
<a name="section1"></a>
## 1. Too many losses? MOO to the rescue!

*Sorry for bad puns.* The methods of Multi-Objective Optimization (MOO) can help you learn multiple objectives better (here and after we will use the terms *objective* and *task* interchangeably). In this section, I will discuss the challenges of learning multiple objectives, and describe a State-of-the-Art solution to it.

<a name="section11"></a>
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
\sum _ {t=1}^T {\lambda^t \hat{\mathcal{L}}^t(\theta^{sh}, \theta^{t})}
\quad\quad
\text{w.r.t.}
\enspace
\theta^{sh}, \theta^{1}, \dots, \theta^{T}
\,,
\end{split}
\tag{1.1.2} \label{eq:mtloss}
\end{equation}
$$

where $$\hat{\mathcal{L}}^t(\cdot)$$ is an empirical task-specific loss for $$t$$-th task defined as the average loss accross the whole dataset $$\hat{\mathcal{L}} (\theta^{sh}, \theta^{t}) \triangleq \frac{1}{N} \sum_i \mathcal{L} ( f^t(x_i; \theta^{sh}, \theta^{t}), y_i^t )$$, where $$y_i^t \in \mathcal{Y}^t$$ is the ground truth of the $$t$$-th task that corresponds to $$i$$-th sample in the dataset of $$N$$ samples.

### 1.2. The $$\lambda_t$$ Balancing Problem

{% capture imblock11 %}
    {{ site.url }}/articles/images/2019-01-22-mtl-a-practical-survey/sec1_im1.svg
{% endcapture %}
{% include gallery images=imblock11 cols=1 %}

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

### 1.3. Instant Noodle in case of Multiple Losses is MOO

Recent works attacks this problem by presenting a heuristic, according to which the coefficients $$\lambda_t$$ are chosen: [Chen et al. (2017)][chen2017] manipulates them in such a way that the gradients are approximately normalized; [Kendall et al. (2018)][kendall2018] models the network output's [homoscedastic uncertainty][homoscedastic] with a probabilistic model. These approaches are further discussed in [Appendix A.1][appendix-a-1] and [Appendix A.2][appendix-a-2].
However, heuristic are too unreliable &mdash; there is no guarantee that the chosen weights will be of any good. A true *instant noodle* approach should be reliable. That's where the latest paper [Sener and Koltun (2018)][mtl-as-moo] presented on [NeurIPS][nips2018] this year comes to rescue. This paper is very theory-heavy, so I will expose it just enough to give you a glimpse of the core idea without delving too deep into the rigorous theoretical stuffs.

Instead of optimizing the summation objective \eqref{eq:mtloss}, the idea is to look at the MTL problem from the perspective of [multi-objective optimization][moo]: optimizing a collection of possibly conflicting objectives. The MTL objective is then specified using a vector-valued loss $${L}$$:

$$
\begin{equation}
{L}(\theta^{sh}, \theta^1,\ldots,\theta^T) =
\left( \hat{\mathcal{L}}^1(\theta^{sh},\theta^1), \ldots,  \hat{\mathcal{L}}^T(\theta^{sh},\theta^T) \right)^\intercal
\tag{1.3.1} \label{eq:vecloss}
\end{equation}
$$

The main motivation to this formulation is the conflict \eqref{eq:mtwtf}. This vector objective will not have a strong order minimum, but we can still talk about a weaker sort of minimality &mdash; the [Pareto optimality][pareto-opt].

> **Definition (Pareto optimality).**
A solution $$\theta$$ dominates a solution $$\bar{\theta}$$ if $$\hat{\mathcal{L}}^t(\theta^{sh},\theta^t)  \leq \hat{\mathcal{L}}^t(\bar{\theta}^{sh},\bar{\theta}^t)$$ for all tasks $$t$$ and $$L(\theta^{sh}, \theta^1,\ldots,\theta^T) \neq L(\bar{\theta}^{sh}, \bar{\theta}^1,\ldots,\bar{\theta}^T)$$;
A solution $$\,\theta^\star$$ is called Pareto optimal if there exists no solution $$\,\theta$$ that dominates $$\,\theta^\star$$.

{% capture imblock12 %}
    {{ site.url }}/articles/images/2019-01-22-mtl-a-practical-survey/sec1_im2.svg
{% endcapture %}
{% include gallery images=imblock12 cols=1 %}

The multi-objective optimization can be solved to local minimality (in a Pareto sense) via Multiple Gradient Descent Algorithm (MGDA), thoroughly studied by [Désidéri (2012)][mgda]. This algorithm leverages the [Karush&ndash;Kuhn&ndash;Tucker (KKT) conditions][kkt-cond] which are neccessary for optimality. In this case, the KKT conditions for both shared and task-specific parameters are follows:

-  There exists $$\lambda^1 \dots \lambda^T$$ such that $$\sum _ {t=1}^T \lambda^t = 1$$ and the [convex combination][convex-comb] of gradients with respect to shared paramethers $$\sum _ {t=1}^T \lambda^t \nabla _ {\theta^{sh}} \hat{\mathcal{L}}^t(\theta^{sh},\theta^t) = 0$$.
-  For all tasks $$t$$, the gradients with respect to task-specific parameters $$\nabla _ {\theta^t} \hat{\mathcal{L}} (\theta^{sh}, \theta^{t}) = 0$$.

The solutions satisfying these conditions are also called a **Pareto stationary** point. It is worth noting that although every Pareto optimal point is Pareto stationary, the reverse may not be true. Now, we formulate the optimization problem for coefficients $$\lambda^1, \ldots, \lambda^T$$ as follows:

$$
\begin{equation}
\begin{split}
    \text{minimize} \quad &
    \left\| \sum_{t=1}^T {\lambda^t \nabla_{\theta^{sh}} \hat{\mathcal{L}}^t (\theta^{sh},\theta^t)} \right\| _ 2^2
    \\
    \text{subject to} \quad &
    \sum_{t=1}^T \lambda^t = 1, \enspace \lambda^t \ge 0
\end{split}
\tag{1.3.2} \label{eq:lambdaopt}
\end{equation}
$$

Denoting $$p^t = \nabla _ {\theta^{sh}} \hat{\mathcal{L}}^t (\theta^{sh},\theta^t)$$, this optimization problem with respect to $$\lambda^t$$ is equivalent to finding a minimum-norm point in the convex hull of the set of input points $$p^t$$. This problem arises naturally in computational geometry: it is equivalent to finding the closest point within a convex hull to a given query point. Basically, \eqref{eq:lambdaopt} is a convex quadratic problem with linear constraints. If you are like me, chances are you're also sick of the [non-convex][non-convex-rage] optimization problems appearing every day of your career! Having a [convex problem][convex-opt-boyd] popping out of nowhere like this is nothing short of joy. The [Frank&ndash;Wolfe solver][frank-wolfe] was used as a most suitable convex optimization algorithm in this case. The following theorem highlights the nice properties of this optimization problem:

> **Theorem ([Désidéri][mgda]).** If $$\,\lambda^1, \dots, \lambda^T$$ is the solution of \eqref{eq:lambdaopt}, either of the following is true:
> - $$\sum _ {t=1}^T {\lambda^t \nabla _ {\theta^{sh}} \hat{\mathcal{L}}^t (\theta^{sh},\theta^t)} = 0$$ and the resulting $$\,\lambda^1, \ldots, \lambda^T$$ satisfies the KKT conditions.
> - $$\sum _ {t=1}^T {\lambda^t \nabla _ {\theta^{sh}} \hat{\mathcal{L}}^t (\theta^{sh},\theta^t)}$$ is a descent direction that decreases all objectives.

The gist of the approach is clear &mdash; the resulting MTL algorithm is to apply [gradient descent][grad-desc] on the task-specific parameters $$\{ \theta^t \} _ {t=1}^T$$, followed by solving \eqref{eq:lambdaopt} and applying the solution $$\sum_{t=1}^T \lambda^t \nabla_{\theta^{sh}}$$ as a gradient update to shared parameter $$\theta^{sh}$$. This algorithm will work for almost *any* neural network that you can build &mdash; the definition in \eqref{eq:mtnn} is very broad.

{% capture imblock13 %}
    {{ site.url }}/articles/images/2019-01-22-mtl-a-practical-survey/sec1_im3.svg
{% endcapture %}
{% include gallery images=imblock13 cols=1 %}

It is easy to notice that in this case, we need to compute $$\nabla _ {\theta^{sh}}$$ for each task $$t$$, which requires a backward pass over the shared parameters for each task. Hence, the resulting gradient computation would be the forward pass followed by $$T$$ backward passes. This significantly increases our expected training time. To address that, the authors ([Sener and Koltun, 2018][mtl-as-moo]) also provided a clever approximation of $$\nabla _ {\theta^{sh}}$$ that allows us to perform the computations in just one pass, while preserving the nice theorem above under mild assumptions. Also, the [Frank&ndash;Wolfe solver][frank-wolfe] used to optimize \eqref{eq:lambdaopt} requires an efficient algorithm for the [line search][line-search] (a very common subroutine in [convex optimization][convex-opt-boyd] methods). This involves rigorous proofs, so I will omit it here to keep the simplicity (i.e. *noodleness*) of this article.

<a name="subsection14"></a>
### 1.4. Remarks and Modifications

Absolutely brilliant! [ten out of ten][chuck-ten]! The mathematics in this paper is juicy! It outperforms [Chen et al. (2017)][chen2017] and [Kendall et al. (2018)][kendall2018] consistently with a large margin! Heck, it even outperforms the single-task classifier in most of the benchmarks! Absolute insanity! The [second author][vkoltun] is also a beast in modern Machine Learning! This is by far the most tasty *instant noodle* in this survey!

Moreover, the approximation for $$\nabla _ {\theta^{sh}}$$, although it is designed for encoder-decoder architecture, can be generalized to a tree-like structure. It can also be easily modified in case your objectives are not equal in importance: every constraints $$\lambda_i > c\lambda_j$$ is a convex constraint, and a combination of it is also convex. So, we can still use the [Frank&ndash;Wolfe solver][frank-wolfe] here. The exact algorithm is described in [Appendix B.1][appendix-b-1].

[moo]: https://en.wikipedia.org/wiki/Multi-objective_optimization
[nips2018]: https://nips.cc/Conferences/2018/Schedule?type=Poster
[mtl-as-moo]: https://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf
[homoscedastic]: https://en.wikipedia.org/wiki/Homoscedasticity
[grid-search]: https://scikit-learn.org/0.18/auto_examples/model_selection/grid_search_digits.html
[kendall2018]: http://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
[chen2017]: https://arxiv.org/abs/1711.02257
[chuck-ten]: http://m.memegen.com/c1rb75.jpg
[vkoltun]: http://vladlen.info/
[grad-desc]: https://en.wikipedia.org/wiki/Gradient_descent
[pareto-opt]: https://en.wikipedia.org/wiki/Pareto_efficiency
[mgda]: https://hal.inria.fr/inria-00389811v1/document
[convex-comb]: https://en.wikipedia.org/wiki/Convex_combination
[kkt-cond]: http://www.onmyphd.com/?p=kkt.karush.kuhn.tucker
[non-convex-rage]: https://www.cs.ubc.ca/labs/lci/mlrg/slides/non_convex_optimization.pdf
[convex-opt-boyd]: http://web.stanford.edu/~boyd/cvxbook/
[line-search]: https://en.wikipedia.org/wiki/Line_search
[frank-wolfe]: http://fa.bianp.net/blog/2018/notes-on-the-frank-wolfe-algorithm-part-i/
[appendix-a-1]: #appendixa1
[appendix-a-2]: #appendixa2
[appendix-b-1]: #appendixb1






<br><br>
<a name="section2"></a>
## 2. Forgot something? Hallucinate it!

*No, I don't propagandize drugs and weed.* In this section, I will describe the problem of [catastrophic forgetting][catastrophic-forgetting] that occurs when the tasks you are trying to learn are very different so you don't have ground truth labels for each tasks for every input (or, in case of Unsupervised/GANs/Reinforcement &mdash; you can't evaluate the model for all its actions). Then, I will describe the ways to overcome it that were proposed on [WACV 2018][wacv2018] and discuss how it can be improved in the industry setting with abundant resources.

### 2.1. Interference of Tasks and Forgetting Effect

{% capture imblock21 %}
    {{ site.url }}/articles/images/2019-01-22-mtl-a-practical-survey/sec2_im1.svg
{% endcapture %}
{% include gallery images=imblock21 cols=1 %}

Consider a Multi-Task Learning setting as in figure $$(a)$$, where a CNN have to learn the *Action* (presented in orange color) of the image, and the *Caption* (presented in blue) of the image. The data is incomplete &mdash; the *Caption* ground truth is not available for *Action* data, and vice-versa. Thus, the training pipeline as illustrated in figure $$(b)$$ will alternate between *Action* tasks and *Caption* tasks, i.e. the model on each training step will only have data from either of the tasks. Obviously, it makes the training of summation objectives \eqref{eq:mtloss} or vector objectives \eqref{eq:lambdaopt} impossible.

A naive way to get around it is to ignore the losses of other tasks while training on a sample of one task ([Kokkinos, 2016][ubernet]; there is also a [video of the talk][ubernet-talk]). More specifically, on training step $$s$$ where only the inputs $$x^t$$ and ground truths $$y^t$$ of task $$t$$ is available, we will set $$\mathcal{L}^k(\cdot) = 0$$ for all $$k \ne t$$, i.e. zerroing out gradients of the tasks without ground truth. [Kokkinos (2016)][ubernet] also suggests to not use a fixed batch size, but rather accumulate gradients separately for task-specific parameters $$\theta^t$$ and shared parameters $$\theta^{sh}$$, and do the gradient step once the number of samples exceeds certain threshold (individual for each $$\theta^{sh}$$ and $$\theta^t$$).

Unfortunately, there is a well-known issue with this simple method. When we train either branch with a dataset, the knowledge of the network of the other tasks might be forgotten. It is because during training, the optimization path of the $$\theta^{sh}$$ can be different for each task.

### 2.2. Instant Noodle is Your Previous Self!

On [WACV 2018][wacv2018], a very simple approach is proposed ([Kim et al. 2018][disjoint-mtl]). The idea is, if you don't have ground truths for other tasks &mdash; just make sure that the model's output on other branches is the same as previously. On each training step, where you have input samples $$x^t$$ and only ground truths for $$t$$-th task $$y^t$$, instead of setting $$\lambda^k(\cdot) = 0$$ for $$k \ne t$$ as in the naive approach above, you need to enforce the outputs to be similar to your previous outputs on tasks $$k$$ using a [Knowledge Distillation][distill] loss.

{% capture imblock22 %}
    {{ site.url }}/articles/images/2019-01-22-mtl-a-practical-survey/sec2_im2.svg
{% endcapture %}
{% include gallery images=imblock22 cols=1 %}

For example, in the setting in previous subsection, when the model is feeded with *Caption* data, it also tries to be similar to its previous self with respect to its outputs on *Action* branch, as illustrated in figure $$(c)$$; when the model is feeded with *Action* data, it also tries to be similar to its previous self with respect to the outputs on the *Caption* branch, as illustrated in figure $$(d)$$.

Knowledge Distillation is a family of techniques, first proposed by [Hinton (2015)][hinton-distill], to make a model to learn from other model as well while training on a specific dataset. In case of classification, consider a *Teacher Network* (a pre-trained network) and a *Student Network* (the one to be trained) that have $$\text{Softmax}(\cdot)$$ as the output layer, and outputs $$y = (y_1, \ldots, y_n)$$ and $$\hat{y} = (\hat{y} _ 1, \ldots, \hat{y} _ n)$$ respectively, where $$n$$ is number of classes. The [Knowledge Distillation][hinton-distill] loss that applied to the *Student Network* for preserving activation of the *Student Network* is defined as follows:

$$
\begin{equation}
\mathcal{L} _ {\text{distill}}(y, \hat{y}) = -\sum _ {k=1}^n {y' _ k \log \hat{y}' _ k}\,,
\quad
y' _ k = \frac{y_k ^ {1/T}}{ \sum_k y_k ^ {1/T}}
\tag{2.2.1} \label{eq:kd}
\end{equation}
$$

where $$T$$ is called *temperature* &mdash; the parameter that makes $$\text{Softmax}(\cdot)$$ activations softer. In a sense, the *Distillation* loss above is almost the same as the [crossentropy][crossentropy] loss used for classification, but is softer. This makes it ideal for our MTL setting &mdash; we want our outputs to other tasks to be *similar* to a learned model, but not *the same.* Demanding the same outputs might prevent our model to learn. One can construct a *Distillation Loss* for other kind of loss functions as well.

### 2.3. Why should we stop there?

A more general idea is to distill the knowledge from a collection of single-task networks, each already learned on task $$k$$ for all tasks $$k \ne t$$, while training on task $$t$$, as illustrated below. This way, we can pretend that the label is there when we actually don't have it.

{% capture imblock23 %}
    {{ site.url }}/articles/images/2019-01-22-mtl-a-practical-survey/sec2_im3.svg
{% endcapture %}
{% include gallery images=imblock23 cols=1 %}

One can even go a step further &mdash; to ellaborate the more aggressive knowledge transfer techniques that distills hidden representations, such as *FitNets* ([Romero, 2015][fitnets]), to train the Multi-Task model faster. It can be helpful when one needs to perform a Neural Architecture Search for the most efficient MTL architecture. Simple, yet effective. A true *Instant Noodle!*

[catastrophic-forgetting]: https://en.wikipedia.org/wiki/Catastrophic_interference
[ubernet]: https://arxiv.org/abs/1609.02132
[ubernet-talk]: https://www.youtube.com/watch?v=72rUGGw32uY
[wacv2018]: http://wacv18.wacv.net/
[disjoint-mtl]: https://arxiv.org/abs/1802.04962
[distill]: https://medium.com/neural-machines/knowledge-distillation-dc241d7c2322
[hinton-distill]: https://arxiv.org/abs/1503.02531
[crossentropy]: https://en.wikipedia.org/wiki/Cross_entropy
[fitnets]: https://arxiv.org/abs/1412.6550





<br><br>
<a name="section3"></a>
## 3. No Instant Noodle Architecture yet :(

Unfortunately, I can't think of any multi-task architecture that can be used everywhere, i.e. a *Truly Instant Noodle*. In this section, I will instead discuss the **pros** and **cons** of commonly used architectures for Multi-Task Learning (especially in Computer Vision tasks). Since they are not *Instant Noodles*, I will not delve deep into details in this section.

Existing architectures of MTL can be classified according to how they share parameters between tasks, as shown in the figure below [(Meyerson & Miikkulainen, 2018)][beyond-shared]. The common trait between them is that they all have some sort of *shared hierarchies*.

{% capture imblock31 %}
    {{ site.url }}/articles/images/2019-01-22-mtl-a-practical-survey/sec3_im1.svg
{% endcapture %}
{% include gallery images=imblock31 cols=1 %}

<a name="section31"></a>
### 3.1. Shared encoder, task-specific decoders

This is the most straight-forward approach, as shown in Fig. $$(3.a)$$, and is the most natural architecture that one can come up with, dated back from [Caruana (1997)][caruana1997]. In Multi-Task Learning literature, this approach is also referred to as *Hard Parameter Sharing*. Sometimes, this approach is extended to task-specific encoders ([Luong et al., 2016][luong2015]). This is the most widely used architecture as well (sort of an *okay-ish instant noodle*).

**Pros** of this family:
- **Dead simple** &mdash; simple to implement, simple to train, simple to debug. Lots of tutorials are available as well: for TensorFlow ([here][tf-shared-tut]), for Keras ([here][keras-tut1] & [here][keras-tut2]), for PyTorch ([here][torch-tut]).
- **Well-studied** &mdash; a huge body of literature has accumulated ever since [Caruana (1997)][caruana1997], both theoretically ([Kendall et al. 2018][kendall2018]; [Chen et al. 2017][chen2017]; [Sener & Koltun, 2018][mtl-as-moo]) and practically ([Ranjan et al. 2016][ranjan2016]; [Wu et el. 2015][wu2015]; [Jaderberg et al. 2017][jaderberg2017]).
- **The fastest from all** &mdash; it shares everything possible, so the inference time will be not much different than executing a single network.

**Cons** of this family:
- **Not flexible** &mdash; forcing all tasks to share a common encoder is dumb. Some tasks are more *similar* than other, so logically a [*depth prediction*][depth-pred] and [*surface normal prediction*][normal-pred] should share more parameters with each other, than with a [*object detection*][obj-det] task.
- **Pretending to share** &mdash; as highlighted by [Liu & Huang (2018)][meta-mtl-communication], these kind of architectures just collects all the features together into a common layer, instead of learning shared parameters (weights) across different tasks.
- **Fight for resources** &mdash; as a consequence, the tasks often fight with each other for resources (e.g. convolution kernels) within a layer. If the tasks are closely related, it's ok, but otherwise this architecture is very inconvenient. This makes the issue of [*negative transfer*][negative-transfer] (i.e. one task can corrupt useful features of other tasks) more probable.

<a name="section31"></a>
### 3.2. A body for each task

{% capture imblock32 %}
    {{ site.url }}/articles/images/2019-01-22-mtl-a-practical-survey/sec3_im2.svg
{% endcapture %}
{% include gallery images=imblock32 cols=1 %}

*It ain't a communist slogan.* This family of architectures is also referred as *Soft Parameter Sharing* in literature, the core idea is shown in Fig. $$(3.b)$$ &mdash; each task has its own layer of task-specific parameters at each shared depth. They then define a mechanism to share knowledge (i.e. parameters) between tasks at each shared depth (i.e. sharing between columns).

The most *instant noodley* approach is Cross-Stich Networks ([Mirsa et al. 2016][mirsa2016]), illustrated in Fig. $$(a)$$. It allows the model to determine in what way the task-specific columns should combine knowledge from other columns, by learning a linear combination of the output of previous layers. *Use this if you need a noodley.*

A generalization of Cross-Stitch Networks are Sluice Networks ([Ruder et al. 2017][ruder2017]). It combines elements of hard paramenter sharing, cross-stitch networks, as well as other good stuffs to create a task hierarchy, as illustrated in Fig. $$(b)$$. *Use this if you're feeling naughty.*

Another interesting yet extremely simple column-based approach is Progressive Networks ([Rusu et al. 2016][rusu2016]), illustrated in Fig. $$(c)$$. This is arguably another breed &mdash; it is intended to solve a more general problem to MTL, the Learning Without Forgetting (LWF) problem. The tasks are learned gradually, one-by-one. This works best when you have learned a task, and want to learn similar tasks quickly. *This is a very specific noodley.*

A recent work ([He et al. 2018][mtzipping]), presented on [NeurIPS 2018][nips2018], allows one to merge fully-trained networks, by leveraging results from [Network Pruning][pruning].

**Pros** of this family:
- **Explicit sharing mechanism** &mdash; the tasks decides for themselves what to keep and what to share at each pre-defined level, so it will have less problems like *fighting for resources* or *pretending to share*.

**Cons** of this family:
- **Soooo sloooooow, soooo faaat** &mdash; the architecture is very bulky (a whole network for each task), so the approach is impractical. Current trend in tech requires lighter and faster networks for On-Device AI.
- **Huge variety, no silver bullet** &mdash; there are a huge variety in this family of networks. None of them seems much supperior to the others, so choosing the right architecture for specific need might be tricky.
- **Not end-to-end** &mdash; this family of networks usually requires the task-specific columns (at least some of them) to be already pre-trained.

### 3.3. Branching at custom depth

This approach is based on the shared encoder one, discussed in [Section 3.1][section-3-1], with a small modification &mdash; instead of having all task-specific encoders branching from the main body (the shared part) at a fixed layer, each of them now are detaching from different layers, as shown in Fig. $$(3.c)$$.

In my personal experience, I choose the branching place of each task experimentally &mdash; I just run a bunch of experiments over the weekends on a *Huge Ass* cluster with a bunch of GPUs to decide the best performing yet most compact one. Basically, a brute force, which is very inefficient.

{% capture imblock33 %}
    {{ site.url }}/articles/images/2019-01-22-mtl-a-practical-survey/sec3_im3.svg
{% endcapture %}
{% include gallery images=imblock33 cols=1 %}

A more promising way of finding efficient architectures is to dynamically figure out where to branch out from the main body. On [CVPR 2017][cvpr2017], an approach was proposed by [Lu et al. (2017)][lu2017] that starts from a fully-shared architecture, then dynamically splits the layers out greedily according to task affinity heuristics (that should correlate with task similarity). Basically, it is a [Neural Architecture Search (NAS)][nas] algorithm. This approach has many drawbacks as well (very hard to choose hyperparameters, the architecture may not be optimal at all, the affinity metric is questionable, etc. &mdash; just my opinion), but is still an interesting direction of research.

**Pros** of this family:
- **Dead simple** and **well-studied** &mdash; the theoretical and practical stuffs for architectures in [Section 3.1][section-3-1] works here as well, so it has all pros described previously.
- **Still fast [AF][as-fuck]** &mdash; not as fast as the family of architectures in [Section 3.1][section-3-1], but still faster than everything else. In this family of architecture, you still share as much as you can between tasks.
- **Ideal case is ideal** &mdash; different tasks tends to share bottom features and diverge on the deeper layers ([He et al. 2018][mtzipping]). If branching is done ideally, combined with ideas from the family of networks in [Section 3.2][section-3-2], there shouldn't be any *fighting for resources* or *pretending to share* problems as in [Section 3.1][section-3-1].

**Cons** of this family:
- **No one dares to do it** &mdash; not everyone have a luxury of going for a full brute-force as me. Dynamic approaches based on heuristics ([Lu et al. 2017][lu2017]) are very unreliable. If done incorrectly, this family of architectures can inherit **all drawbacks of all families of MTL nets combined!!!**

### 3.4. Beyond sharing, beyond hierarchies, beyond this world

This family, schematically illustrated in Fig. $$(3.d)$$, makes an observation that the tasks can share all parameters in the main body, except batch normalization scaling factors ([Bilen and Vedaldi, 2017][bilen2017]). Basically, the tasks share the whole network, and the only task-specific parameters are [Instance Normalization][instance-norm] parameters. On [ICLR][iclr] last year, [Meyerson & Miikkulainen (2018)][beyond-shared] [quickly escalated][that-escalated-quickly] this idea a step further by allowing the weights themselves to be freely permutted. The idea of changing the orders of layers by itself is not new ([Veit et al. 2016][veit2016]), but learning the best permutation of weights across different tasks is very creative.

**Pros** of this family:
- **Lightweight** &mdash; they share every penny that they can, so the resulting model will have almost as much parameter as one single-task network.
- **Just [WOW][wow-meme]** &mdash; sharing every layer, and even with permutted order, is very counter-intuitive. It makes you wonder "what are features? what is knowledge? what is life?" You can even use it to hook up some girls!

**Cons** of this family:
- **Still slooooooow** &mdash; as in [Section 3.2][section-3-2], you still have to propagate through the whole network for each task. If you don't intend to execute all tasks, just want to save some space, this is not a cons at all.
- **Still vulnerable** &mdash; this family can still be vulnerable to *fights for resources* or *pretending to share* problems as in [Section 3.1][section-3-1].

### 3.5. Remarks

I just want to make a quick note that the *Instant Noodles* in [Section 1][section-1] and [Section 2][section-2] can be applied to any of the architectures above, with a limitation that the upper-bound approximation of \eqref{eq:lambdaopt} may not apply to architectures with no explicit enconder/decoder. A true *Instant Noodle Architecture* in the future might utilize Neural Architecture Search (NAS) for MTL might be very promising in the future, as the industry is moving towards smaller and faster models.

[resnet]: https://arxiv.org/abs/1512.03385
[beyond-shared]: https://openreview.net/forum?id=BkXmYfbAZ
[ruder-mtl]: https://arxiv.org/abs/1706.05098
[caruana1997]: https://link.springer.com/article/10.1023/A:1007379606734
[luong2015]: https://arxiv.org/abs/1511.06114
[tf-shared-tut]: https://jg8610.github.io/Multi-Task/
[keras-tut1]: https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/
[keras-tut2]: https://blog.manash.me/multi-task-learning-in-keras-implementation-of-multi-task-classification-loss-f1d42da5c3f6
[torch-tut]: https://medium.com/@zhang_yang/multi-task-deep-learning-experiment-using-fastai-pytorch-2b5e9d078069
[kendall2018]: http://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
[chen2017]: https://arxiv.org/abs/1711.02257
[mtl-as-moo]: https://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf
[ranjan2016]: https://arxiv.org/abs/1603.01249
[wu2015]: https://ieeexplore.ieee.org/document/7178814
[jaderberg2017]: https://arxiv.org/abs/1611.05397
[depth-pred]: http://www.cs.cornell.edu/projects/megadepth/
[normal-pred]: http://www.cs.cmu.edu/~aayushb/marrRevisited/
[obj-det]: https://en.wikipedia.org/wiki/Object_detection
[meta-mtl-communication]: https://arxiv.org/abs/1810.09988
[negative-transfer]: https://arxiv.org/pdf/1708.00260.pdf
[mirsa2016]: https://arxiv.org/abs/1604.03539
[ruder2017]: https://arxiv.org/abs/1705.08142
[rusu2016]: https://arxiv.org/abs/1606.04671
[section-3-1]: #section31
[section-3-2]: #section32
[lu2017]: https://arxiv.org/abs/1611.05377
[cvpr2017]: http://openaccess.thecvf.com/CVPR2017.py
[nas]: https://en.wikipedia.org/wiki/Neural_architecture_search
[as-fuck]: https://www.urbandictionary.com/define.php?term=AF
[mtzipping]: https://arxiv.org/abs/1805.09791
[bilen2017]: https://arxiv.org/abs/1701.07275
[instance-norm]: https://arxiv.org/abs/1607.08022
[iclr]: https://en.wikipedia.org/wiki/International_Conference_on_Learning_Representations
[veit2016]: https://arxiv.org/abs/1605.06431
[that-escalated-quickly]: https://www.dictionary.com/e/slang/that-escalated-quickly/
[wow-meme]: https://www.youtube.com/watch?v=jUy9_0M3bVk
[section-1]: #section1
[section-2]: #section2
[nips2018]: https://nips.cc/Conferences/2018/Schedule?type=Poster
[pruning]: https://jacobgil.github.io/deeplearning/pruning-deep-learning






<br><br>
## Appendix A: other noodles that ain't the yummiest noodle

In this section, I will describe the other approaches that I had experience with, but won't recommend them for others. On their own, they are quite good and convenient, just not the best out there (comparing to methods described above).

<a name="appendixa1"></a>
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

**Comments.** This approach requires choosing an additional hyperparameter $$\alpha$$ that you have to guess. This pisses me off because I don't have the insane intuition on hyperparameter tuning as the [Kagglers][kagglers] do. Furthermore, it introduces another loss function $$\mathcal{L} _ {\text{grad}}$$ that regularizes the gradient magnitudes. Lol, [good job bro][good-job-meme] &mdash; you've just introduced another hyperparameter to optimize your existing hyperparameters, and introduced another loss to the sum to simplify the bunch of losses that you have *(sarcasm)*. Somehow it works, but not as good as the instant noodle above.

<a name="appendixa2"></a>
### A.2. Using uncertainties of losses $$\mathcal{L}^t(\cdot)$$ to balance $$\lambda_t$$

**Description.** On [CVPR 2018][cvpr2018], another approach was proposed by [Kendall et al. (2018)][kendall2018] that models the network output's [homoscedastic][homoscedastic] uncertainty with a probabilistic model. We will use the notion \eqref{eq:mtnn} in [Section 1.1][section-11]. For single-task, we model the network output uncertainty with a density function $$p\left( y \vert f(x, \theta) \right)$$ (how the true answer is likely to be $$y$$, given network's response). In the case of multiple network outputs $$y^1, \dots y^T$$, we obtain the following multi-task likelihood:

$$
\begin{equation}
\tag{A.2.1} \label{eq:mtlikelihood}
p \left( y^1, \ldots, y^T \vert f(x, \theta) \right) = p\left(y^1 \vert f^1(x, \theta)\right) \ldots p\left(y^T \vert f^T(x, \theta)\right) \to \max
\end{equation}
$$

Instead of balancing the weights of loss functions as in \eqref{eq:mtloss}, we can now require the likelihood \eqref{eq:mtlikelihood} to be maximal, i.e. we have a [maximal likelihood][max-likelihood] inference problem, when the objective is to minimize $$-\log p(y^1, \ldots, y^T \vert f(x, \theta))$$ with respect to $$\theta$$. The trick now is to construct such a likelihood $$p(y^t \vert f^t(x,\theta))$$ for each task, so that it will contain a loss $$\mathcal{L}^t(\cdot)$$ term. This way, we will be able to create a bridge between the maximum likelihood \eqref{eq:mtlikelihood} and the summation loss \eqref{eq:mtloss}. The $$\log(\cdot)$$ will also convert multiplications to summation, which will basically bring the maximum likelihood to the summation form.

As an example to this dark magic approach, consider a multi-regression regression where your objective is to optimize the loss $$\mathcal{L}^t(\theta) = \| y^t - f^t(x, \theta) \|^2$$ for all tasks $$t \in \{1 \ldots T\}$$. The likelihood is defined artificially as a Gaussian with mean given by model's output and deviation given by a noise factor $$\sigma$$:

$$
\begin{equation}
p\left(y \vert f(x, \theta)\right) = \mathcal{N}(f(x, \theta), \sigma^2)
\tag{A.2.2} \label{eq:gausslike}
\end{equation}
$$

The noise scalar $$\sigma$$ is observed during training, i.e. it is a *trainable* parameter. In essense, it is the parameter that captures the uncertainty. So, our objective now is to maximize \eqref{eq:mtlikelihood} with respect to $$\sigma$$ sa well. After careful computations, our *log likelihood* will take the following form:

$$
\begin{equation}
-\log p \left( y^1, \ldots, y^T \vert f(x, \theta) \right)
\propto 
\underbrace{\sum_{t=1}^T {\frac{1}{2\sigma _ 1^2} 
\| y^t - f^t(x, \theta) \|^2}} _
{\text{the same as}\, \sum_{t=1}^T \lambda^t\mathcal{L}^t(\theta)}
+ \underbrace{\log \prod_{t=1}^T \sigma_t} _ {\text{regularization}}
\tag{A.3.3} \label{eq:l2ll}
\end{equation}
$$

which is the same as the summation loss \eqref{eq:mtloss}, where we assign $$\lambda_t = \frac{1}{2}\sigma_t^{-2}$$, plus the [regularization term][regularization] that discourages $$\sigma_t$$ to increase too much (effectively ignoring the data).

**Comments.** [What kind of black magic is this?][black-magic] Basically, to optimize a loss function, you will need to construct a whole distribution, the logarithm of which will give you the loss function multiplied by a learnable scalar $$\sigma$$, and make sure that this distribution is physically meaningful! Or, get rid of the notion of "loss fuction" at all and just make hypothesis about the form of $$\mathcal{L}(\cdot)$$ uncertainty. This is too much pain in the ass for a lazy engineer. There is no guarantee that the density you constructed is correct either.

[cvpr2018]: http://cvpr2018.thecvf.com/
[icml2018]: https://icml.cc/Conferences/2018/Schedule?type=Poster
[backprop]: https://en.wikipedia.org/wiki/Backpropagation
[kendall2018]: http://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
[chen2017]: https://arxiv.org/abs/1711.02257
[good-job-meme]: https://i.kym-cdn.com/photos/images/newsfeed/000/514/589/66b.jpg
[kagglers]: https://www.kaggle.com/umeshnarayanappa/the-world-needs-kaggle
[section-11]: #section11
[max-likelihood]: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
[regularization]: https://en.wikipedia.org/wiki/Regularization_(mathematics)
[black-magic]: http://memecrunch.com/meme/HXNV/black-magic/image.jpg
