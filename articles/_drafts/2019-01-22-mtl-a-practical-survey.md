---
layout: post
permalink: /articles/:title
type: "article"
title: 'An "Instant Noodle" guide to Multi-Task Learning'
image:
  feature: "articles/images/2019-01-22-mtl-a-practical-survey/featured.png"
  display: false
tags: [survey, lecture notes, deep learning]
excerpt: "A survey on stuffs that works like a charm as-is right from the box and are easy to implement &ndash; just like instant noodle!"
comments: true
---



> Industry ML/AI [Engineers][im-an-engineer] are [**weirdos**][weirdo]: they hunt the latest papers, they crave for State-of-the-Art, they talk about craziest ideas that might enhance the whole field, yet they are too lazy to implement harder-than-a-resnet idea. This article is written **for** such [Engineers][im-an-engineer].

[weirdo]: https://www.urbandictionary.com/define.php?term=weirdo
[im-an-engineer]: https://www.youtube.com/watch?v=rp8hvyjZWHs






## Introduction

If you found yourself in a strange situation, where you want your Neural Network to do several things at once, i.e. detect objects, predict depth, predict surface normals &ndash; don't worry, you are just having a Multi-Task Learning (MTL) problem.

{% capture imblock1 %}
    {{ site.url }}/articles/images/2019-01-22-mtl-a-practical-survey/teaser.png
{% endcapture %}
{% include gallery images=imblock1 cols=1 %}

Traditionally, the development of Multi-Task Learning was aimed to improve the generalization of multiple task predictors by jointly training them, while allowing some sort of knowledge transfer and between them [(Caruana, 1997)][caruana1997]. If you, for example, train a *surface normal prediction* model and *depth prediction* model together, they will definitely share mutually-benefitial features together [(Eigen et al. 2015)][eigen-dnl]. This motivation is clearly inspired by natural intelligence &mdash; living creatures in an remarkable way can easily learn a task by leveraging the knowledge from other tasks. A broader generalization of this idea is called [Lifelong Learning][lifelong-learning], in which different tasks are not even learned simultaneously.

However, screw these academic stuffs, **we are [engineers][im-an-engineer]**! Why should we care about leveraging diverse features from different tasks when we can just *slap that AI* with more data ([kagglers][kagglers] doesn't count here)? If you are an engineer from *big AF* companies like Google, Samsung, Microsoft, etc. then *Hell Yeah* you've got a *ton* of cash to *splash out* on labellers! Just hire them to get more data. Thus our main motivation for Multi-Task learning will be the following:

-  **To optimize multiple objectives at once.** For instance, in [GANs][gan], it is shown in various tasks that often incorporating additonal loss functions can yield much better results ([Isola et al. 2017][pix2pix]; [Wang et al. 2018][vid2vid]). A [regularization term][regularization] can also be considered as additional objective.

-  **To reduce the cost of running multiple models**. My Korean boss always yells at my team *"we need faster CNNs!"* in his typical asian accent (no offense, I'm also asian). How can we further speed up the 5 models that are already optimized both in size and speed by more than 40 times? Oh, yeah, we can merge all of them into a single Multi-Task model!

In this article, I will only focus on the two motivation above. The *motto* of this article is: *simple as instant noodle, easy to implement, and effective as heck!* For a more comprehensive survey that focuses on the *mutually-benefitial sharing* aspect of Multi-Task Learning, it is recommended to read [Ruder's (2017)][ruder-mtl] paper. As I continue to write this article, it became a convenient note for my lecture as well, so here you will find more in-depth theoretical stuffs (that normally only the full papers have) than a typical survey will do.

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

*Sorry for bad puns.* The methods of Multi-Objective Optimization (MOO) can help you learn multiple objectives better (here and after we will use the terms *objective* and *task* interchangeably). In this section, I will discuss the challenges of learning multiple objectives, and describe a State-of-the-Art solution to it.

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

Recent works attacks this problem by presenting a heuristic, according to which the coefficients $$\lambda_t$$ are chosen: [Chen et al. (2017)][chen2017] manipulates them in such a way that the gradients are approximately normalized; [Kendall et al. (2018)][kendall2018] models the network output's [homoscedastic uncertainty][homoscedastic] with a probabilistic model.
However, heuristic are too unreliable &mdash; there is no guarantee that the chosen weights will be of any good. A true *instant noodle* approach should be reliable. That's where the latest paper [Sener and Koltun (2018)][mtl-as-moo] presented on [NeurIPS][nips2018] this year comes to rescue. This paper is very theory-heavy, so I will expose it just enough to give you a glimpse of the core idea without delving too deep into the rigorous theoretical stuffs.

**Description.** Instead of optimizing the summation objective \eqref{eq:mtloss}, the idea is to look at the MTL problem from the perspective of [multi-objective optimization][moo]: optimizing a collection of possibly conflicting objectives. The MTL objective is then specified using a vector-valued loss $${L}$$:

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

The multi-objective optimization can be solved to local minimality (in a Pareto sense) via Multiple Gradient Descent Algorithm (MGDA), thoroughly studied by [Désidéri (2012)][mgda]. This algorithm leverages the [Karush&ndash;Kuhn&ndash;Tucker (KKT) conditions][kkt-cond] which are neccessary for optimality. In this case, the KKT conditions for both shared and task-specific parameters are follows:

-  There exists $$\lambda^1 \dots \lambda^T$$ such that $$\sum_{t=1}^T \lambda^t = 1$$ and the [convex combination][convex-comb] of gradients with respect to shared paramethers $$\sum_{t=1}^T \lambda^t \nabla_{\theta^{sh}} \hat{\mathcal{L}}^t(\theta^{sh},\theta^t) = 0$$.
-  For all tasks $$t$$, the gradients with respect to task-specific parameters $$\nabla_{\theta^t} \hat{\mathcal{L}} (\theta^{sh}, \theta^{t}) = 0$$.

The solutions satisfying these conditions are also called a *Pareto stationary* point. It is worth noting that although every Pareto optimal point is Pareto stationary, the reverse may not be true. Now, we formulate the optimization problem for coefficients $$\lambda^1, \ldots, \lambda^T$$ as follows:

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

Denoting $$p^t = \nabla_{\theta^{sh}} \hat{\mathcal{L}}^t (\theta^{sh},\theta^t)$$, this optimization problem with respect to $$\lambda^t$$ is equivalent to finding a minimum-norm point in the convex hull of the set of input points $$p^t$$. This problem arises naturally in computational geometry: it is equivalent to finding the closest point within a convex hull to a given query point. Basically, \eqref{eq:lambdaopt} is a convex quadratic problem with linear constraints. If you are like me, chances are you're also sick of the [non-convex][non-convex-rage] optimization problems appearing every day of your career! Having a [convex problem][convex-opt-boyd] popping out of nowhere like this is nothing short of joy. The [Frank&ndash;Wolfe solver][frank-wolfe] was used as a most suitable convex optimization algorithm in this case. The following theorem highlights the nice properties of this optimization problem:

> **Theorem ([Désidéri][mgda]).** If $$\,\lambda^1, \dots, \lambda^T$$ is the solution of \eqref{eq:lambdaopt}, either of the following is true:
> - $$\sum_{t=1}^T {\lambda^t \nabla_{\theta^{sh}} \hat{\mathcal{L}}^t (\theta^{sh},\theta^t)} = 0$$ and the resulting $$\,\lambda^1, \ldots, \lambda^T$$ satisfies the KKT conditions.
> - $$\sum_{t=1}^T {\lambda^t \nabla_{\theta^{sh}} \hat{\mathcal{L}}^t (\theta^{sh},\theta^t)}$$ is a descent direction that decreases all objectives.

The gist of the approach is clear &mdash; the resulting MTL algorithm is to apply [gradient descent][grad-desc] on the task-specific parameters $$\{ \theta^t \} _ {t=1}^T$$, followed by solving \eqref{eq:lambdaopt} and applying the solution $$\sum_{t=1}^T \lambda^t \nabla_{\theta^{sh}}$$ as a gradient update to shared parameter $$\theta^{sh}$$. This algorithm will work for almost *any* neural network that you can build &mdash; the definition in \eqref{eq:mtnn} is very broad.

It is easy to notice that in this case, we need to compute $$\nabla_{\theta^{sh}}$$ for each task $$t$$, which requires a backward pass over the shared parameters for each task. Hence, the resulting gradient computation would be the forward pass followed by $$T$$ backward passes. This significantly increases our expected training time. To address that, the authors ([Sener and Koltun, 2018][mtl-as-moo]) also provided a clever approximation that allows us to perform the computations in just one pass, while preserving the nice theorem above under mild assumptions. Also, the [Frank&ndash;Wolfe solver][frank-wolfe] used to optimize \eqref{eq:lambdaopt} requires an efficient algorithm for the [line search][line-search] (a very common subroutine in [convex optimization][convex-opt-boyd] methods). This involves rigorous proofs, so I will omit it here to keep the simplicity (i.e. *noodleness*) of this article.

[moo]: https://en.wikipedia.org/wiki/Multi-objective_optimization
[nips2018]: https://nips.cc/Conferences/2018/Schedule?type=Poster
[mtl-as-moo]: https://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf
[homoscedastic]: https://en.wikipedia.org/wiki/Homoscedasticity
[grid-search]: https://scikit-learn.org/0.18/auto_examples/model_selection/grid_search_digits.html
[kendall2018]: http://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
[chen2017]: https://arxiv.org/abs/1711.02257
[grad-desc]: https://en.wikipedia.org/wiki/Gradient_descent
[pareto-opt]: https://en.wikipedia.org/wiki/Pareto_efficiency
[mgda]: https://hal.inria.fr/inria-00389811v1/document
[convex-comb]: https://en.wikipedia.org/wiki/Convex_combination
[kkt-cond]: http://www.onmyphd.com/?p=kkt.karush.kuhn.tucker
[non-convex-rage]: https://www.cs.ubc.ca/labs/lci/mlrg/slides/non_convex_optimization.pdf
[convex-opt-boyd]: http://web.stanford.edu/~boyd/cvxbook/
[line-search]: https://en.wikipedia.org/wiki/Line_search
[frank-wolfe]: http://fa.bianp.net/blog/2018/notes-on-the-frank-wolfe-algorithm-part-i/






## 2. Forgot something? Hallucinate it!

*No, I don't propagandize drugs and weed.* In this section, I will describe the problem of [catastrophic forgetting][catastrophic-forgetting] that occurs when the tasks you are trying to learn are very different so you don't have ground truth labels for each tasks for every input (or, in case of Unsupervised/GANs/Reinforcement &mdash; you can't evaluate the model for all its actions).

[catastrophic-forgetting]: https://en.wikipedia.org/wiki/Catastrophic_interference






## Appendix A: other noodles that are yummy but ain't the best

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

**Comments.** This approach requires choosing an additional hyperparameter $$\alpha$$ that you have to guess. This pisses me off because I don't have the insane intuition on hyperparameter tuning as the [Kagglers][kagglers] do. Furthermore, it introduces another loss function $$\mathcal{L} _ {\text{grad}}$$ that regularizes the gradient magnitudes. Lol, [good job bro][good-job-meme] &mdash; you've just introduced another hyperparameter to optimize your existing hyperparameters, and introduced another loss to the sum to simplify the bunch of losses that you have *(sarcasm)*. Somehow it works, but not as good as the instant noodle above.

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
[good-job-meme]: https://i.kym-cdn.com/photos/images/newsfeed/000/514/589/66b.jpg
[kagglers]: https://www.kaggle.com/umeshnarayanappa/the-world-needs-kaggle