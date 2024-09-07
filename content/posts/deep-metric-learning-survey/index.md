---
title: "Deep Metric Learning: a (Long) Survey"
url: "/articles/deep-metric-learning-survey"
date: 2021-06-14T00:00:00+00:00
# weight: 1
# aliases: ["/first"]
tags: ["Contrastive Learning", "RecSys", "Survey"]
author: "Kha Vu Chan"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: true
summary: "In this post, I'll briefly go over the common approaches for Deep Metric Learning, as well as the new methods proposed in recent years."
canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
hideSummary: false
hideAuthor: true
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: false
ShowPostNavLinks: true
ShowWordCount: false
ShowRssButtonInSectionTermList: true
UseHugoToc: false
cover:
    image: "feature.png" # image path/url
    alt: "Deep Metric Learning" # alt text
    caption: "Deep Metric Learning" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide in list view
editPost:
    URL: "https://github.com/hav4ik.github.io/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

One of the most amazing aspects of the human visual system is the ability to recognize similar objects and scenes. We don't need hundreds of photos of the same face to be able to differentiate it among thousands of other faces that we've seen. We don't need thousands of images of the Eiffel Tower to recognize that unique architectureal landmark when we visit Paris. Is it possible to design a Deep Neural Network with a similar ability to tell which objects are visually similar and which ones are not? That's essentially what **Deep Metric Learning** attempts to solve.

<!--
Advanced readers may immediately recognize by the description that this topic is intimately related to One-Shot Learning. The techniques of Metric Learning are commonly embraced in the field of One-Shot Learning, and sometimes these terms are even used interchangeably (e.g. by [Andrew Ng][andrewng_meme] in his [Deep Learning course][andrewng_one_shot_learning]). However, they are completely different fields &mdash; One-Shot Learning uses more than just Metric Learning techniques, and Metric Learning can be applied in other problems as well.
-->

Although this blog post is mainly about **Supervised Deep Metric Learning** and is self-sufficient on its own, it would be beneficial for you to consider getting familiar with traditional Metric Learning methods (i.e. without Neural Networks) to develop a broader understanding of this topic. I highly recommend the [introductory guides on Metric Learning][sklearn_metric_learning_guide] as a starter. If you want to get into the formal mathematical side of things, I recommend the tutorial by [Diaz et al. (2020)][diaz_tutorial_metric_math]. Popular Metric Learning methods include the popular [t-SNE (van der Maaten & Hinton, 2008)][tsne_paper] and the new shiny [UMAP (McInnes et al., 2018)][umap_paper] that everybody uses nowadays for data clustering and visualization.

This article is organized as follows. In the **"Contrastive Approaches"** section, I will quickly glance through the methods that were commonly used for Deep Metric Learning, before the rise of angular margin methods in 2017. Then, in **Moving Away from Contrastive Approaches**, I will describe the transitioning to current angular margin SOTA models and the reasons why we ditch the direct approaches. Then, in the **"State-of-the-Art Approaches"** section, I will describe in more detail the advances in Metric Learning in recent years.

The most useful section for both beginners and more experienced readers will be the **"Getting Practical"** section, in which I will do a case study of how Deep Metric Learning is used to achieve State-of-the-Art results in various practical problems (mostly from Kaggle and large-scale benchmarks), as well as the tricks that were used to make things work.


[sklearn_metric_learning_guide]: http://contrib.scikit-learn.org/metric-learn/introduction.html
[diaz_tutorial_metric_math]: https://arxiv.org/abs/1812.05944
[tsne_paper]: https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf
[umap_paper]: https://arxiv.org/abs/1802.03426
[andrewng_one_shot_learning]: https://www.youtube.com/watch?v=96b_weTZb2w
[andrewng_meme]: https://i.imgur.com/6sQcztc.jpg


---------------------------------------------------------------------------------


<a name="problem-setting"></a>
## Problem Setting of Supervised Metric Learning

Generally speaking, Deep Metric Learning is a group of techniques that aims to measure the similarity between data samples. More specifically, for a set of data points \(\mathcal{X}\) and their corresponding labels \(\mathcal{Y}\) (a discrete finite set), the goal is to train an [embedding][embedding_nlab] neural model (also referred to as feature extractor) \(f_{\theta}(\cdot)\, \colon \mathcal{X} \to \mathbb{R}^n\) (where \(\theta\) are learned weights) together with a distance \(\mathcal{D}\, \colon \mathbb{R}^n \to \mathbb{R}\) (which is usually fixed beforehand), so that for two data samples \(x_1, x_2 \in \mathcal{X}\) together with their labels \(y_1, y_2 \in \mathcal{Y}\), the combination \(\mathcal{D}\left(f_{\theta}(x_1), f_{\theta}(x_2)\right)\) produces small values if the labels \(y_1, y_2 \in \mathcal{Y}\) are equal, and larger values if they aren't.

Thus, the Deep Metric Learning problem boils down to just choosing the architecture for \(f_{\boldsymbol{\theta}}\) and choosing the loss function \(\mathcal{L}(\theta)\) to train it with. One might wonder why we cannot just use the classification objective for the metric learning problem? In fact, the [Softmax loss][softmax_loss] is also a valid objective for metric learning, albeit inferior to other objectives as we will see later in this article.


[softmax_loss]: https://cs231n.github.io/linear-classify/#softmax
[softmax_func_vs_softmax_loss]: https://medium.com/@liangjinzhenggoon/the-difference-between-softmax-and-softmax-loss-173d385120c2
[embedding_nlab]: https://ncatlab.org/nlab/show/embedding+of+topological+spaces


---------------------------------------------------------------------------------


<a name="contrastive-approaches"></a>
## Contrastive Approaches

The main idea of Contrastive Approaches is to design a loss function that directly pulls together the embeddings of samples with the same label (i.e. "similar" samples) and pushes away the embeddings of dissimilar samples, hence the name "Contrastive". These methods are sometimes regarded as "Direct" in other surveys because they directly apply the definition of metric learning. The distance function in the embedding space for these approaches is usually fixed as \( l_2 \) metric:

$$
\begin{equation*}
\mathcal{D}\left(p, q\right) = \|p - q\|_2 = \left(\sum_{i=1}^n \left(p_i - q_i\right)^2\right)^{1/2}
\end{equation*}
$$

For the ease of notation, let's denote \(\mathcal{D}_{f_\theta}(x_1, x_2)\) as a shortcut for \(\mathcal{D} \left( f_\theta(x_1), f_\theta(x_2) \right)\), where \(x_1, x_2 \in \mathcal{X}\) are samples from the dataset (more formally, \(\mathcal{D}_{f_\theta}\) is a [pullback][pullback] of \(\mathcal{D}\)).
I will just glance through the most common approaches in this section very quickly without getting too much into details for two reasons:

- The methods described here are already covered in the context of deep metric learning in other tutorials and blog posts. I highly recommend the great survey by [Kaya & Bilge (2019)][deep_metric_learning_survey].
- The methods that I will describe in the section ["State-of-the-Art Approaches"](#state-of-the-art-approaches) outperforms these approaches in most cases of **supervised** deep metric learning anyways.

To learn more about Contrastive Learning approaches in more general setting (i.e. not only in Metric Learning), I highly recommend the overview blog post by [Lilian Weng (2021)][lillog_contrastive]. This branch of research is still in active development, usually for Representation Learning or Manifold Learning purposes.


<a name="contrastive-loss"></a>
### Contrastive Loss

This is a classic loss function for metric learning. **Contrastive Loss** ([Chopra et al. 2005][contrastive_loss_paper]) is one of the simplest and most intuitive training objectives. Let \( x_1, x_2 \) be some samples in the dataset, and \( y_1, y_2 \) be their corresponding labels. Also, for some condition \( A \), let's denote \( \mathbb{1}_A \) as the identity function that is equal to \( 1 \) if \( A \) is true, and \( 0 \) otherwise. The loss function is then defined as follows:

$$
\begin{equation*}
\mathcal{L}_\text{contrast} = 
\mathbb{1}_{y_1 = y_2}
\mathcal{D}^2_{f_\theta}(x_1, x_2)
+
\mathbb{1}_{y_1 \ne y_2}
\max\left(0, \alpha - \mathcal{D}^2_{f_\theta}(x_1, x_2)\right)
\end{equation*}
$$

where \( \alpha \) is the margin. The reason we need a margin value is that otherwise, our network \( f_\theta \) will learn to "cheat" by mapping all \( \mathcal{X} \) to the same point, making distances between any samples equal to zero. [Here][contrastive_explained] and [here][contrastive_explained_2] are very great in-depth explanations for this loss function.

<a name="triplet-loss"></a>
### Triplet Loss

{{< figure src="triplet_loss.png" caption="The core idea of Triplet Loss (Image source: [Schroff et al. 2015](https://arxiv.org/abs/1503.03832))" invertible="true" >}}

**Triplet Loss** [(Schroff et al. 2015)][triplet_loss_paper] is by far the most popular and widely used loss function for metric learning. It is also featured in [Andrew Ng's deep learning course][andrew_ng_triplet_loss].

Let \( x_a, x_p, x_n \) be some samples from the dataset and \( y_a, y_p, y_n \) be their corresponding labels, so that \( y_a = y_p \) and \( y_a \ne y_n \). Usually, \( x_a \) is called the **anchor** sample, \( x_p \) is called the **positive** sample because it has the same label as \( x_a \), and \( x_n \) is called the **negative** sample because it has a different label. It is defined as:

$$
\begin{equation*}
\mathcal{L}_\text{triplet} =
\max\left(
    0, \mathcal{D}^2_{f_\theta}(x_a, x_p) - \mathcal{D}^2_{f_\theta}(x_a, x_n) + \alpha
\right)
\end{equation*}
$$

where \(\alpha\) is the margin to discourage our network \(f_\theta\) to map the whole dataset \(\mathcal{X}\) to the same point. The key ingredient to make Triplet Loss work in practice is **Negative Samples Mining** — on each training step, we sample such triplets that such triplets \(x_a, x_p, x_n\) that satisfies \(\mathcal{D}_{f_\theta}(x_a, x_n) < \mathcal{D}_{f_\theta}(x_a, x_p) + \alpha\), i.e. the samples that our network \(f_\theta\) fails to discriminate or is not able to discriminate with high confidence. You can find in-depth description and analysis of Triplet Loss in [this awesome blog post][triplet_loss_explained].

Triplet Loss is [still being widely used][paperswithcode_tripletloss] despite being inferior to the recent advances in Metric Learning (which we will learn about in the next section) due to its relative effectiveness, simplicity, and the wide availability of code samples online for all deep learning frameworks.



<a name="improving-triplet-loss"></a>
### Improving the Triplet Loss

Despite its popularity, Triplet Loss has a lot of limitations. Over the past years, there have been a lot of efforts to improve the Triplet Loss objective, building on the same idea of sampling a bunch of data points, then pulling together similar samples and pushing away dissimilar ones in \( l_2 \) metric space.

{{< figure src="metric_losses.png" caption="A visual overview of different deep metric learning approaches that are based on the same idea as the Triplet Loss objective (Image source: [Kaya & Bilge, 2019](https://www.mdpi.com/2073-8994/11/9/1066/htm))" invertible="true" >}}

<a name="quadruplet-loss"></a>

**Quadruplet Loss** ([Chen et al. 2017][quadruplet_loss_paper]) is an attempt to make inter-class variation of the features \(f_\theta(x)\) larger and intra-class variation smaller, contrary to the Triplet Loss that doesn't care about class variation of the features. For all quadruplets \((x_i, x_j, x_k, x_l)\) where \(y_i = y_j\), \(y_i \ne y_k\), \(y_i \ne y_l\), and \(y_k \ne y_l\) (basically \((i, j)\) is a positive pair, and samples \(k\) and \(l\) are from completely different categories), the Quadruplet Loss is defined as:

$$
\begin{eqnarray*}
\mathcal{L}_\text{quadruplet}
= &
\sum_{i,j,k} {
\max\left(0,
\mathcal{D}^2_{f_\theta}(x_i, x_j) -
\mathcal{D}^2_{f_\theta}(x_i, x_k) + \alpha_1\right)}
\\ + &
\sum_{i,j,k} {
\max\left(0,
\mathcal{D}^2_{f_\theta}(x_i, x_j) -
\mathcal{D}^2_{f_\theta}(x_k, x_l) + \alpha_2\right)}
\end{eqnarray*}
$$  

With the help of this constraint, the minimum inter-class distance is required to be larger than the maximum intra-class distance regardless of whether pairs contain the same probe.

In a similar paper ([Ni et al. 2017][quadruplet_loss_2_paper]) with the same core idea, for samples \(x_a, x_p, x_n, x_s\) and their corresponding labels \(y_a = y_p = y_s\), \(y_a \ne y_n\), the Quadruplet Loss is defined as:

$$
\begin{eqnarray*}
\mathcal{L}_\text{quadruplet}
= &
\max\left(0,
\mathcal{D}^2_{f_\theta}(x_a, x_p) -
\mathcal{D}^2_{f_\theta}(x_a, x_s) + \alpha_1\right)
\\ + &
\max\left(0,
\mathcal{D}^2_{f_\theta}(x_a, x_s) -
\mathcal{D}^2_{f_\theta}(x_a, x_n) + \alpha_2\right)
\end{eqnarray*}
$$

<a name="structured-loss"></a>
**Structured Loss** ([Song et al. 2016][structured_loss_paper]) was proposed to improve the sampling effectiveness of Triplet Loss and make full use of the samples in each batch of training data. Here, I will describe the generalized version of it by [Hermans et al. (2017)][generalized_structured_loss_paper].

Let \(\mathcal{B} = (x_1, \ldots, x_b)\) be one batch of data, \(\mathcal{P}\) be the set of all positive pairs in the batch (\(x_i, x_j \in \mathcal{P}\) if their corresponding labels satisfy \(y_i = y_j\)) and \(\mathcal{N}\) is the set of all negative pairs (\(x_i, x_j \in \mathcal{N}\) if corresponding labels satisfy \(y_i \ne y_j\)). The Structured Loss is then defined as:

$$
\begin{eqnarray*}
\widehat{\mathcal{J}}_{i,j}
=&&
\max\left(
\max_{(i,k) \in \mathcal{N}} \left\{\alpha - \mathcal{D}_{f_\theta}(x_i, x_k)\right\},
\max_{(l,j) \in \mathcal{N}} \left\{\alpha - \mathcal{D}_{f_\theta}(x_l, x_j)\right\}
\right) + \mathcal{D}_{f_\theta}(x_i, x_j)
\\
\widehat{\mathcal{L}}_\text{structured}
=&&
\frac{1}{2|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}}
\max\left( 0, \widehat{\mathcal{J}}_{i,j} \right)^2
\end{eqnarray*}
$$

Intuitively, the formula above means that for each pair of positive samples, we compute the distance to the closes negative sample to that pair, and we try to maximize it for every positive pair in the batch. To make it differentiable, the authords proposed to optimize an upper bound instead:

$$
\begin{eqnarray*}
\mathcal{J}_{i,j}
=&&
\log\left(
\sum_{(i,k) \in \mathcal{N}} \exp\left\{\alpha - \mathcal{D}_{f_\theta}(x_i, x_k)\right\} +
\sum_{(l,j) \in \mathcal{N}} \exp\left\{\alpha - \mathcal{D}_{f_\theta}(x_l, x_j)\right\}
\right) + \mathcal{D}_{f_\theta}(x_i, x_j)
\\
\mathcal{L}_\text{structured}
=&&
\frac{1}{2|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}}
\max\left( 0, \mathcal{J}_{i,j} \right)^2
\end{eqnarray*}
$$

The **N-Pair Loss** [(Sohn, 2016)][n_pair_loss_paper] paper discusses in great detail one of the main limitations of the Triplet Loss while proposing a similar idea to [Structured Loss](#structured-loss) of using positive and negative pairs:

> During one update, the triplet loss only compares a sample with one negative sample while ignoring negative samples from the rest of the classes.
As consequence, the embedding vector for a sample is only guaranteed to be far from the selected negative class but not necessarily the others. Thus we can end up only differentiating an example from a limited selection of negative classes yet still maintain a small distance from many other classes.
>
> In practice, the hope is that, after looping over sufficiently many randomly sampled triplets, the final distance metric can be balanced correctly; but the individual update can still be unstable and the convergence would be slow.  Specifically, towards the end of the training, most randomly selected negative examples can no longer yield non-zero triplet loss error.

Other attempts to design a better metric learning objective based on the core idea of the Triplet Loss objective include **Magnet Loss** ([Rippel et al. 2015][magnet_loss_paper]) and **Clustering Loss** ([Song et al. 2017][clustering_loss_paper]). Both objectives are defined on the dataset distribution as a whole, not only on single elements. However, they didn't receive much traction due to the scaling difficulties, and simply because of their complexity. There has been some attempt to compare these approaches, notably by [Horiguchi et al. (2017)][comparing_classification_and_metric], but they performed experiments on very small datasets and were unable to achieve meaningful results.

The methods described in this section are part of a wider family of machine learning methods, called "Contrastive Representation Learning". I highly recommend [this blog post][lillog_contrastive] for more details.


[contrastive_loss_paper]: http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf
[backpropagation_wiki]: https://en.wikipedia.org/wiki/Backpropagation
[deep_metric_learning_survey]: https://www.mdpi.com/2073-8994/11/9/1066/htm
[contrastive_explained]: https://medium.com/@maksym.bekuzarov/losses-explained-contrastive-loss-f8f57fe32246
[contrastive_explained_2]: https://gombru.github.io/2019/04/03/ranking_loss/
[triplet_loss_paper]: https://arxiv.org/abs/1503.03832
[andrew_ng_triplet_loss]: https://www.youtube.com/watch?v=LN3RdUFPYyI
[triplet_loss_explained]: https://omoindrot.github.io/triplet-loss
[paperswithcode_tripletloss]: https://paperswithcode.com/method/triplet-loss
[quadruplet_loss_paper]: https://arxiv.org/abs/1704.01719
[quadruplet_loss_2_paper]: https://dl.acm.org/doi/10.1145/3132847.3133022
[triplet_vs_quadruplet]: https://towardsdatascience.com/how-to-choose-your-loss-when-designing-a-siamese-neural-net-contrastive-triplet-or-quadruplet-ecba11944ec
[structured_loss_paper]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Song_Deep_Metric_Learning_CVPR_2016_paper.pdf
[generalized_structured_loss_paper]: https://arxiv.org/abs/1703.07737
[n_pair_loss_paper]: https://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
[magnet_loss_paper]: https://arxiv.org/abs/1511.05939
[clustering_loss_paper]: https://openaccess.thecvf.com/content_cvpr_2017/papers/Song_Deep_Metric_Learning_CVPR_2017_paper.pdf
[pytorch_metric_learning]: https://github.com/KevinMusgrave/pytorch-metric-learning
[comparing_classification_and_metric]: https://openreview.net/forum?id=HyQWFOVge
[lillog_contrastive]: https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html


---------------------------------------------------------------------------------


<a name="moving-away-from-contrastive-approaches"></a>
## Moving Away from Contrastive Approaches

After countless research papers attempting to solve the problems and limitations of [Triplet Loss](#triplet-loss) in the context of Supervised Deep Metric Learning, it became clear that learning to directly minimize/maximize euclidean distance between samples with the same/different labels may not be the way to go. There are two main issues of such approaches:

- **Expansion Issue** &mdash; it is very hard to ensure that samples with a similar label will be pulled together to a common region in space as noted by [Sohn (2016)][n_pair_loss_paper] (mentioned in the previous section). [Quadruplet Loss](#quadruplet-loss) only improves the variability, and [Structured Loss](#structured-loss) can only enforce the structure locally for the samples in the batch, not globally. Attempts to solve this problem directly with a global objective ([Magnet Loss, Rippel et al. 2015][magnet_loss_paper] and [Clustering Loss, Song et al. 2017][clustering_loss_paper]) were not successful in gaining much traction due to scalability issues.

- **Sampling Issue** &mdash; all of the Deep Metric Learning approaches that try to directly minimize/maximize \(l_2\) distance between samples rely heavily on sophisticated sample mining techniques that choose the "most useful" samples for learning for each training batch. This is inconvenient enough in the local setting (think about GPU utilization), and can become quite problematic in a distributed training setting (e.g. when you train on 10s of [cloud TPUs][cloud_tpu] and pull the samples from a remote GCS bucket).


<a name="center-loss"></a>
### Center Loss

**Center Loss** ([Wen et al. 2016][center_loss_paper]) is one of the first successful attempts to solve both of the above-mentioned issues. Before getting into the details of it, let's talk about Softmax Loss.

Let \(z = f_\theta(x)\) be the feature vector of the sample \(x\) after propagating through the neural network \(f_\theta\). In the classification setting of \(m\) classes, on top of the backbone neural network \(f_\theta\) we usually have a linear classification layer \(\hat{y} = W^\intercal z + b\), where \(W \in \mathbb{R}^{n \times m}\) and \(b \in \mathbb{R}^m\). The Softmax Loss (that we're all familiar with and know by heart) for a batch of \(N\) samples is then presented as follows:

$$
\begin{equation*}
\mathcal{L}_\text{softmax} = - \frac{1}{N} \sum_{i=1}^{N}{
\log \frac{
\exp\left\{W^\intercal_{y_i} z_i + b_{y_i}\right\}
}{
\sum_{j=1}^{m} \exp\left\{W^\intercal_{j} z_i + b_{j}\right\}
}}
\end{equation*}
$$

Let's have a look at the training dynamics of the Softmax objective and how the resulting feature vectors are distributed relative to each other:

{{< figure src="softmax_traintest.gif" caption="The training dynamics of Softmax Loss on MNIST. The feature maps were projected onto 2D space to produce this visualization. On the left is the dynamics on train set, and on the right is the dynamics on test set. (Image source: [KaiYang Zhou](https://github.com/KaiyangZhou/pytorch-center-loss))" invertible="true" >}}

As illustrated above, the Softmax objective is not discriminative enough, there's still a significant intra-class variation even on such a simple dataset as MNIST. So, the idea of Center Loss is to add a new regularization term to the Softmax Loss to pull the features to corresponding class centers:

$$
\begin{equation*}
\mathcal{L}_\text{center} = \mathcal{L}_\text{softmax} +
\frac{\lambda}{2} \sum_{i=1}^N \| z_i - c_{y_i} \|_2^2
\end{equation*}
$$

where \(c_j\) is also updated using gradient descent with \(\mathcal{L}_\text{center}\) and can be thought of as moving mean vector of the set of feature vectors of class \(j\). If we now visualize the training dynamics and the resulting distribution of feature vectors of Center Loss on MNIST, we will see that it is much more discriminative compared to Softmax Loss.

{{< figure src="center_traintest.gif" caption="The training dynamics of Center Loss on MNIST. The feature maps were projected onto 2D space to produce this visualization. On the left is the dynamics on train set, and on the right is the dynamics on test set. (Image source: [KaiYang Zhou](https://github.com/KaiyangZhou/pytorch-center-loss))" invertible="true" >}}

The Center Loss solves the Expansion Issue by providing the **class centers** \( c_j \), thus forcing the samples to cluster together to the corresponding class center; it also solves the Sampling issue because we don't need to perform hard sample mining anymore. Despite having its own problems and limitations (which I will describe in the next sub-section), Center Loss is still a pioneering work that helped to steer the direction of Deep Metric Learning to its current form.


<!--
<a name="large-margin-softmax-loss"></a>
### Large-Margin Softmax Loss

Another idea to improve the discriminativeness of the Softmax Objective is to consider the angular distance between the feature vectors of the samples (after the last layer of the backbone neural network) instead of euclidean $$l_2$$ distance, and enforce a certain **margin** between them. **Large-Margin Softmax Loss** ([Liu et al. 2017][large_margin_paper]) is designed to do just that.

> TODO: Finish the description for L-Softmax Loss
-->


<a name="sphereface"></a>
### SphereFace

The obvious problem of the formulation of [Center Loss](#center-loss) is, ironically, the choice of centers. First, there's still no guarantee that you will have a large inter-class variability since the clusters closer to zero will benefit less from the regularization term. To make it "fair" for each class, why don't we just enforce the class centers to be at the same distance from the center? Let's map it to a hypersphere!

That's the main idea behind **SphereFace** ([Liu et al. 2017][sphereface_paper]). The setting of SphereFace is very simple. We start from the Softmax loss with the following modifications:

- Fix the bias vector \(b = 0\) to make the future analysis easier (the whole heavy-lifting is performed by our neural network anyways).
- Normalize the weights so that \(\smash{\| W_j \| = 1}\). This way, when we rewrite the product \(\smash{W_j^\intercal z}\) as \(\smash{\| W_j \| \| z \| \cos\theta_j}\), where \(\smash{\theta_j}\) is the angle between feature vector \(z\) and the row vector \(\smash{W_j}\), it becomes just \(\smash{\| z \| \cos\theta_j}\). So, the final classification output for class \(j\) can be thought about as **projecting** the feature vector \(z\) onto vector \(\smash{W_j}\), which in this case, geometrically, is the **class center**.

Let's denote \(\smash{\theta_{j,i}}\) (\(\smash{0 \le \theta_{j,i} \le \pi}\)) as the angle between the feature vector \(z_i\) and class center vector \(\smash{W_j}\). The **Modified Softmax** objective is thus:

$$
\begin{eqnarray*}
\mathcal{L}_\text{mod. softmax} = && - \frac{1}{N} \sum_{i=1}^{N}{
\log \frac{
\exp\left\{W^\intercal_{y_i} z_i + b_{y_i}\right\}
}{
\sum_{j=1}^{m} \exp\left\{W^\intercal_{j} z_i + b_{j}\right\}
}}
\\ = && - \frac{1}{N} \sum_{i=1}^{N}{
\log \frac{
\exp\left\{\| W_{y_i} \| \|z_i\| \cos (\theta_{y_i, i}) + b_{y_i}\right\}
}{
\sum_{j=1}^{m} \exp\left\{\|z_i\| \cos (\theta_{j,i}) + b_{j}\right\}
}}
\\ = && - \frac{1}{N} \sum_{i=1}^{N}{
\log \frac{
\exp\left\{\|z_i\| \cos (\theta_{y_i, i}) \right\}
}{
\sum_{j=1}^{m} \exp\left\{\|z_i\| \cos (\theta_{j,i})\right\}
}}
\end{eqnarray*}
$$

Geometrically, it means that we assign the sample to class \(j\) if the projection of the logits vector \(z\) to the class center vector \(\smash{W_j}\) is the largest, i.e. if the angle between \(\smash{W_j}\) and \(z\) is the smallest among all class center vectors.

It is important to always keep in mind the **decision boundary**. At which point you will consider a sample as belonging to a certain class?

For Modified Softmax, the decision boundary between classes \(i\) and \(j\) is actually the bisector between two class center vectors \(\smash{W_i}\) and \(\smash{W_j}\). Having such a thin decision boundary will not make our features discriminative enough &mdash; the inter-class variation is too small. Hence the second part of SphereFace &mdash; introducing the **margins**.

The idea is, instead of requiring \(\smash{\cos(\theta_i) > \cos(\theta_j)}\) for all \(\smash{j = 1, \ldots, m\, (j \ne i)}\) to classify a sample as belonging to \(i\)-th class as in Modified Softmax, we additionally enforce a margin \(\mu\), so that a sample will only be classified as belonging to \(i\)-th class if \(\smash{\cos(\mu \theta_i) > \cos(\theta_j)}\) for all \(\smash{j = 1, \ldots, m\, (j \ne i)}\), with the requirement that \(\smash{\theta_i \in [0, \frac{\pi}{\mu}]}\). The SphereFace objective can be then expressed as:

$$
\begin{equation*}
\mathcal{L}_\text{SphereFace} =
    - \frac{1}{N} \sum_{i=1}^{N}{
    \log \frac{
    \exp\left\{\|z_i\| \cos (\mu \theta_{y_i, i}) \right\}
    }{
    \exp\left\{\|z_i\| \cos (\mu \theta_{y_i,i})\right\} +
    \sum_{j \ne y_i} \exp\left\{\|z_i\| \cos (\theta_{j,i})\right\}
    }}
\end{equation*}
$$

The limitations on the value of \(\smash{\theta_i \in [0, \frac{\pi}{\mu}]}\) is really annoying. We don't normally optimize neural networks with such restrictions on weights values. We can get rid of it by replacing \(\smash{\cos(\theta)}\) with a monotonically decreasing angle function \(\smash{\psi(\theta)}\), which we define as \(\smash{\psi(\theta) = (-1)^k \cos(\mu \theta) - 2k}\) for \(\smash{\theta \in [k\pi/\mu, (k+1)\pi/\mu]}\) and \(k \in [0, \mu - 1]\). Thus the final form of **SphereFace** is:

$$
\begin{equation*}
\mathcal{L}_\text{SphereFace} =
    - \frac{1}{N} \sum_{i=1}^{N}{
    \log \frac{
    \exp\left\{\|z_i\| \psi (\mu \theta_{y_i, i}) \right\}
    }{
    \exp\left\{\|z_i\| \psi (\mu \theta_{y_i,i})\right\} +
    \sum_{j \ne y_i} \exp\left\{\|z_i\| \psi (\theta_{j,i})\right\}
    }}
\end{equation*}
$$

The differences between Softmax, Modified Softmax, and SphereFace is schematically shown below.

{{< figure src="sphereface_1.svg" caption="Difference between Softmax, Modified Softmax, and SphereFace. A 2D features model was trained on CASIA data to produce this visualization. One can see that features learned by the original softmax loss can not be" invertible="true" >}}

[hold_on_to_your_papers]: https://www.youtube.com/channel/UCbfYPyITQ-7l4upoX8nvctg
[n_pair_loss_paper]: https://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
[magnet_loss_paper]: https://arxiv.org/abs/1511.05939
[clustering_loss_paper]: https://openaccess.thecvf.com/content_cvpr_2017/papers/Song_Deep_Metric_Learning_CVPR_2017_paper.pdf
[cloud_tpu]: https://cloud.google.com/tpu
[center_loss_paper]: https://link.springer.com/chapter/10.1007/978-3-319-46478-7_31
[large_margin_paper]: https://arxiv.org/abs/1612.02295
[sphereface_paper]: https://arxiv.org/abs/1704.08063
[pullback]: https://en.wikipedia.org/wiki/Pullback_(differential_geometry)



---------------------------------------------------------------------------------


<a name="state-of-the-art-approaches"></a>
## State-of-the-Art Approaches


The success of [SphereFace](#sphereface) resulted in an avalanche of new methods that are based on the idea of employing angular distance with angular margin. 

Please note that these methods are only applicable to Supervised Deep Metric Learning setting. In an Unsupervised setting, or in case when we have a lot of out-of-distribution samples during test time, Contrastive Learning approaches are still amongst the most decent choices.


<a name="cosface"></a>
### CosFace

[Wang et al. (2018)][cosface_paper] discussed in great details the limitations of [SphereFace](#sphereface):

> The decision boundary of the [SphereFace](#sphereface) is defined over the angular space by \(\,\smash{\cos(\mu \theta_1) = \cos(\theta_2)}\), which has a difficulty in optimization due to the nonmonotonicity of the cosine function. To overcome such a difficulty, one has to employ an extra trick with an ad-hoc piecewise function for [SphereFace](#sphereface). More importantly, the decision margin of [SphereFace](#sphereface) depends on \(\,\smash{\theta}\), which leads to different margins for different classes. As a result, in the decision space, some inter-class features have a larger margin while others have a smaller margin, which reduces the discriminating power.

**CosFace** ([Wang et al. 2018][cosface_paper]) proposes a simpler yet more effective way to define the margin. The setting is similar to [SphereFace](#sphereface) with normalizing the rows of weight matrix \(\smash{W}\), i.e., \(\| \smash{W_j} \| = 1\), and zeroing the biases \(b = 0\). Additionally, we normalize the features \(z\) (extracted by a neural network) as well, so \(\| z \| = 1\). The CosFace objective is then defined as:


$$
\begin{equation*}
\mathcal{L}_\text{CosFace} = - \frac{1}{N} \sum_{i=1}^{N}{
\log \frac{
\exp\left\{s \left(\cos (\theta_{y_i, i}) - m\right) \right\}
}{
\exp\left\{s \left(\cos (\theta_{y_i, i}) - m\right) \right\} +
\sum_{j \ne y_i} \exp\left\{s \cos (\theta_{j,i})\right\}
}}
\end{equation*}
$$

where \(s\) is referred to as the **scaling** parameter, and \(m\) is referred to as the **margin** parameter. As in [SphereFace](#sphereface), \(\smash{\theta_{j,i}}\) denotes the angle between \(i\)-th feature vector \(z_i\) and \(\smash{W_j}\), and \(\smash{W_j^\intercal z_i = \cos \theta_{j,i}}\), because \(\smash{\| W_j \| = \| z_i \| = 1}\). Visually, it looks like follows:

<a name="fig-cosface-1"></a>
{{< figure src="cosface_1.svg" caption="Geometrical interpretation of CosFace from a feature perspective. Different colors represent feature space from different classes. CosFace has a relatively compact feature region compared with Modified Softmax (Image source: [Wang et al. 2018](https://arxiv.org/abs/1801.09414))" invertible="true" >}}

<!--
From [Fig 5.](#fig-cosface-1), considering classes $$\smash{C_1}$$ and $$\smash{C_2}$$, we suppose that the normalized feature vector $$x$$ is given, $$\smash{W_i}$$ denotes the normalized weight vector, and $$\smash{\theta_i}$$ denotes the angle between $$z$$ and $$\smash{W_i}$$. For CosFace, the inter-class variance is enlarged while the intra-class variation shrinks, comparing to Modified Softmax.
-->

Choosing the right scale value \(s\) and margin value \(m\) is very important. In the CosFace paper ([Wang et al. 2018][cosface_paper]), it is shown that \(s\) should have a lower bound to at least obtain the expected classification performance. Let \(\smash{C}\) be the number of classes. Suppose that the learned feature vectors separately lie on the surface of the hypersphere and center around the corresponding weight vector. Let \(\smash{P_{W}}\) denote the expected minimum posterior probability of class center (i.e., \(\smash{W}\)). The lower bound of \(s\) is given by:

$$
\begin{equation*}
s \ge \frac{C-1}{C} \log \frac{\left(C-1\right) P_W}{1 - P_W}
\end{equation*}
$$

Supposing that all features are well-separated, the theoretical variable scope of \( m \) is supposed to be:

$$
\begin{equation*}
0 \le m \le \left( 1 - \max\left( W_i^\intercal W_j \right) \right)
\end{equation*}
$$

where \(i, j \le n\) and \(i \ne j\). Assuming that the optimal solution for the Modified Softmax loss should uniformly distribute the weight vectors on a unit hypersphere, the variable scope of margin \(m\) can be inferred as follows:

$$
\begin{align*}
0 \le m \le & \,1 - \cos\frac{2\pi}{C}\,, & (K=2) \\
0 \le m \le & \,\frac{C}{C-1}\,, & (C \le K + 1) \\
0 \le m \ll & \,\frac{C}{C-1}\,, & (C > K + 1)
\end{align*}
$$

where \( K \) is the dimension of learned features. The inequalities indicate that
as the number of classes increases, the upper bound of the cosine margin between classes are decreased correspondingly. Especially, if the number of classes is much larger than the feature dimension, the upper bound of the cosine margin will get even smaller.

<a name="fig-cosface-2"></a>
{{< figure src="cosface_2.svg" caption="Decision boundaries of different loss functions in cosine space (Image source: [Wang et al. 2018](https://arxiv.org/abs/1801.09414))" invertible="true" >}}

<a name="arcface"></a>
### ArcFace


**ArcFace** [(Deng et al. 2019)][arcface_paper] is very similar to [CosFace](#cosface) and addresses the same limitations of [SphereFace](#sphereface) as mentioned in the [CosFace](#cosface) description. However, instead of defining the margin in the cosine space, it defines the margin directly in the angle space.

The setting is identical to [CosFace](#cosface), with the requirements that the last layer weights and feature vector should be normalized, i.e., \(\smash{\| W_j \| = 1}\) and \(\smash{\| z \| = 1}\), and last layer biases should be equal to zero (\(b = 0\)). The ArcFace objective is then defined as:

$$
\begin{equation*}
\mathcal{L}_\text{ArcFace} = - \frac{1}{N} \sum_{i=1}^{N}{
\log \frac{
\exp\left\{s \cos \left(\theta_{y_i, i} + m\right) \right\}
}{
\exp\left\{s \cos \left(\theta_{y_i, i} + m\right) \right\} +
\sum_{j \ne y_i} \exp\left\{s \cos (\theta_{j,i})\right\}
}}
\end{equation*}
$$

where \(s\) is the scaling parameter and \(m\) is referred to as the margin parameter. While the differences with [CosFace](#cosface) is very minor, the results on various benchmarks show that ArcFace is still slightly better than CosFace in most cases. Below is the illustration of the decision boundaries of different loss functions we've reviewed so far:

<a name="fig-arcface"></a>
{{< figure src="arcface.svg" caption="Decision boundaries of different loss functions in the angle space. ArcFace has a constant linear angular margin throughout the whole interval. (Image source: [Deng et al. 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf))" invertible="true" >}}

Face recognition datasets are used as standard benchmark for [CosFace](#cosface), [ArcFace](#arcface), and other angular margin methods, because it is the most popular application of deep metric learning. Here are some ablation study and comparison of loss functions:

{{< figure src="arcface_ablation_merged.png" caption="Ablation study of loss functions on different benchmarks. In the first table, verification results (in \%) of different loss functions are reported for LFW, CFP-FP, and AgeDB-30 benchmarks. On the second table, verification on Megaface is reported, where \"Id\" refers to the rank-1 face identification accuracy with 1M distractors, and \"Ver\" refers to the face verification TAR at \(10^{-6}\) FAR. \"R\" refers to data refinement on both probe set and 1M distractors. (Image source: [Deng et al. 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf))" invertible="true" >}}


### AdaCos &mdash; how to choose \( s \) and \( m \)?


For both [CosFace](#cosface) and [ArcFace](#arcface), the choice of scaling parameter \(s\) and margin \(m\) is crucial. Both papers did very little analysis on the effect of these parameters. To answer the question on how to choose the optimal values of \(s\) and \(m\), [Zhang et al. (2019)][adacos_paper] performed an awesome analysis on the hyperparameters of cosine-based losses.

We denote the pre-softmax activation vector of our network as \(f\), and the number of classes as \(C\). Let's consider the classification probability \(P_{i,j}\) of \(i\)-th sample for \(j\)-th class, which is defined as:

$$
\begin{equation*}
P_{i,j} = \frac{\exp{f_{i,j}}}{\sum_{k=1}^C \exp{f_{i,k}}}.
\end{equation*}
$$

When scaling and margin parameters \(s\) and \(m\) are introduced, the pre-softmax activations \(f_{i,j}\) for \(i\)-th sample and \(j\)-th class with ground-truth \(y_i\) in the case of [ArcFace](#arcface) are defined as follows:

$$
\begin{equation*}
f_{i,j} = \begin{cases}
s \cdot \cos\theta_{i,j}, & \mbox{if } j\ne y_i \\
s \cdot \cos(\theta_{i,j} + m), & \mbox{if } j = y_i
\end{cases}
\end{equation*}
$$

Now, let's plot the value of \(P_{i,j}\) against the angle \(\theta_{i,y_i}\) between the feature vector of \(i\)-th data sample and class center of \(y_i\)-th class, for different values of \(s\):

<a name="fig-adacos-s-anal"></a>
{{< figure src="adacos_s_analysis.svg" caption="curves of \(P_{i,j}\) by choosing different ArcFace scale parameters (Image source: [Zhang et al. 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_AdaCos_Adaptively_Scaling_Cosine_Logits_for_Effectively_Learning_Deep_Face_CVPR_2019_paper.pdf))" invertible="true" >}}

As we can see, when the value of the scaling parameter \(s\) is small (e.g., \(s = 10\)), the maximum value of \(P_{i,j}\) couldn't reach \(1\). This is undesirable because even when the network is very confident on a sample, the loss function would still penalize the correct results. On the other hand, when \(s\) is too large (e.g., \(s = 64\)), it would produce very high probability even when \(\theta_{i,y_i}\) is large, which means the loss function would fail to penalize the mistakes.

Let's take a look at the \(P_{i, y_i}\) curves of different values of the margin parameter \(m\):

<a name="fig-adacos-m-anal"></a>
{{< figure src="adacos_m_analysis.svg" caption="curves of \(P_{i,j}\) by choosing different ArcFace margin parameters (Image source: [Zhang et al. 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_AdaCos_Adaptively_Scaling_Cosine_Logits_for_Effectively_Learning_Deep_Face_CVPR_2019_paper.pdf))" invertible="true" >}}

Increasing the margin parameter shifts the probability curve of \(P_{i,y_i}\) to the left. A larger margin leads to lower probabilities \(P_{i,y_i}\), and thus larger loss even with small angles \(\theta_{i,y_i}\). This also explains why margin-based losses provide stronger supervision for the same \(\theta_{i,y_i}\) than cosine-based losses with no margin.

You might have noticed the dashed "Fixed AdaCos" curve. In the [AdaCos paper (Zhang et al. 2019)][adacos_paper], the following fixed value of the scaling parameter \(s\) is proposed:

$$
\begin{equation*}
\tilde{s} \approx \sqrt{2} \cdot \log \left( C - 1 \right)
\end{equation*}
$$

where \(C\) is the number of classes. The reasoning behind this choice of scaling parameter is outside the scope of this blog post and can be found in paragraph 4.1 of the [AdaCos paper][adacos_paper]. As for the other method proposed in that paper, Adaptive AdaCos &mdash; I've never seen it being successfully deployed in real-world problems.


<a name="subcenter-arcface"></a>
### Sub-Center ArcFace

Having only one center for each class (as in [ArcFace](#arcface), [CosFace](#cosface), etc.) causes the following issues:
- If the intra-class sample variance is high, then it doesn't make sense to enforce compression into a single cluster in the embedding space.
- For large and noisy datasets, the noisy/bad samples can wrongly generate a large loss value, which impairs the model training.

**Sub-Center ArcFace** [(Deng et al. 2020)][subcenter_arcface_paper] solves that by introducing **sub-centers**. The idea is that each class would have multiple class centers. The majority of samples would be contracted to dominant centers, and noisy or hard samples would be pulled to other centers. The formula for Sub-Center ArcFace looks almost the same as [ArcFace](#arcface):

$$
\begin{equation*}
\mathcal{L}_\text{SCAF} = - \frac{1}{N} \sum_{i=1}^{N}{
\log \frac{
\exp\left\{s \cos \left(\tilde{\theta}_{y_i, i} + m\right) \right\}
}{
\exp\left\{s \cos \left(\tilde{\theta}_{y_i, i} + m\right) \right\} +
\sum_{j \ne y_i} \exp\left\{s \cos (\tilde{\theta}_{j,i})\right\}
}}
\end{equation*}
$$

with the exception of the angle \(\tilde{\theta}_{y_i, i}\), which is defined as the angle to the closest sub-center among \(K\) sub-centers \(W_{j, 1} \ldots W_{j, K}\) (as opposed to being just the angle to class center as in [ArcFace](#arcface)):

$$
\begin{equation*}
\tilde{\theta}{i,j} = \arccos \left( \max_k\left(W^\intercal_{j, k} z_i \right) \right)\,, \quad k \in \{ 1, \ldots ,K \}
\end{equation*}
$$


<a name="afdynmargin"></a>
### ArcFace with Dynamic Margin

**ArcFace with Dynamic Margin** [(Ha et al. 2020)][glc2020_3rd_place_paper] is a simple modification of [ArcFace](#arcface) proposed by the 3rd place winners of [Google Landmarks Challenge 2020][kaggle_glr2020]. The main motivation for having different margin values for different classes is the extreme imbalance of the dataset &mdash; some classes can have tens of thousands of samples, while other classes may have only 10 samples.

For models to converge better in the presence of heavy imbalance, smaller classes need to have bigger margins as they are harder to learn. The proposed formula for margin value \(m_i\) of \(i\)-th class is simple:

$$
\begin{equation*}
m_i = a \cdot n_i ^ {-\lambda} + b
\end{equation*}
$$

where \(n_i\) is the number of samples in training data for \(i\)-th class, \(a\) and \(b\) are parameters that control the upper and lower bound of the margin, and \(\lambda > 0\) determines the shape of the margin function. Together with [Sub-Center ArcFace](#subcenter-arcface), this method turns out to be much more effective than [ArcFace](#arcface).

[cosface_paper]: https://arxiv.org/abs/1801.09414
[arcface_paper]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf
[adacos_paper]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_AdaCos_Adaptively_Scaling_Cosine_Logits_for_Effectively_Learning_Deep_Face_CVPR_2019_paper.pdf
[subcenter_arcface_paper]: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf
[glc2020_3rd_place_paper]: https://arxiv.org/pdf/2010.05350.pdf
[kaggle_glr2020]: https://www.kaggle.com/c/landmark-recognition-2020


---------------------------------------------------------------------------------


<a name="getting-practical"></a>
## Getting Practical: a Case Study of Real-World Problems

Sadly, it's a [well-known fact][reproducibility_crisis] that the reported SOTA results on academic benchmarks of cool and shiny new methods might not reflect its performance in real-world problems. That's why in this section, we will take a look at how Deep Metric Learning is being used in real-world problems, and which methods were used to achieve the best results.

[reproducibility_crisis]: https://www.wired.com/story/artificial-intelligence-confronts-reproducibility-crisis/


<a name="humpback-whale-challenge"></a>
### Kaggle: Humpback Whale Challenge (2019)

The main goal of the [Kaggle Humpback Whale Challenge][humpback_whale_challenge] was to identify, whether the given photo of the whale fluke belongs to one of the 5004 known individuals of whales, or it is a new_whale, never observed before.

<a name="fig-humpback-intro"></a>
{{< figure src="humpback_whale_intro.png" caption="Example of 9 photos of the same whale from the training data" >}}

The puzzling aspect of this competition was a huge class imbalance. For more than 2000 classes there was only one training sample. What’s more, the important part of the competition is to classify whether the given whale is new or not.

While a lot of methods tricks were used by top performers in this competition, I will focus only on Deep Metric Learning methods. A short survey of the methods used by top teams (i.e. Gold medalists):
- [ArcFace](#arcface) is used by [2nd place][humpback_2nd_place], [3rd place][humpback_3rd_place], 6th place, and [9th place][humpback_9th_place] medalists.
- [CosFace](#cosface) is used as part of the [9th place's][humpback_9th_place] solution as well.
- [Triplet Loss](#triplet-loss) is used by [2nd place][humpback_2nd_place] (in combination with [ArcFace](#arcface) and Focal losses) and [9th place][humpback_9th_place] solutions (in combination with local keypoints matching).
- The [2nd place][humpback_2nd_place] also used Focal Loss to combat class imbalances.
- [4th place][humpback_4th_place] used a good old Siamese network which, given two images, will tell if they are from the same whale or not. They also used keypoint matching with SIFT and ROOTSIFT.
- Interestingly, the [1st place][humpback_1st_place] team used only classification model together with some grandmaster's wizardry to come up on top.

At the time of this competition, [ArcFace](#arcface) and [CosFace](#cosface) were still pretty new and untested techniques, so they were not widely adopted by top teams. We will see that in the next case studies, [ArcFace](#arcface) and [CosFace](#cosface) will be the most popular methods used by top performers.

[humpback_whale_challenge]: https://www.kaggle.com/c/humpback-whale-identification
[humpback_1st_place]: https://www.kaggle.com/c/humpback-whale-identification/discussion/82366
[humpback_2nd_place]: https://github.com/SeuTao/Humpback-Whale-Identification-Challenge-2019_2nd_palce_solution
[humpback_3rd_place]: https://www.kaggle.com/c/humpback-whale-identification/discussion/82484
[humpback_4th_place]: https://www.kaggle.com/c/humpback-whale-identification/discussion/82356
[humpback_9th_place]: https://medium.com/@anastasiya.mishchuk/thanks-radek-7th-place-solution-to-hwi-2019-competition-metric-learning-story-c94b74a3eaa2
[humpback_9th_place]: https://www.kaggle.com/c/humpback-whale-identification/discussion/82427


<a name="google-landmarks-challenge"></a>
### Kaggle: Google Landmarks Challenge (2020)

The [Google Landmarks Challenge][kaggle_glc2020] is a large-scale competition, with more than 5'000'000 images of more than 200'000 human-made or natural landmarks. The competition is divided into 2 tracks:
- **Retrieval Track** &mdash; matching a specific object in an input image to all other instances of that object in a catalog of reference images. Think of it as the Reverse Image Search feature of Google.
- **Recognition Track** &mdash; recognizing specific instances of objects, e.g. distinguishing Niagara Falls from just any waterfall.

<a name="fig-google-landmarks-intro"></a>
{{< figure src="google_landmarks_intro.png" caption="Some beautiful samples from Google Landmarks Dataset V2" >}}

Aside from the gigantic size of this dataset, there are a lot of challenges that make it an ideal testbed for Deep Metric Learning techniques:
- **Class Imbalance.** For popular landmarks such as the Eiffel Tower, there can be thousands of images, while a less-known boathouse in Seattle will have less than 10 images per class.
- **Real-World Conditions.** To mimic a realistic setting, only 1% of the test images are within the target domain of landmarks, while 99% are out-of-domain images.
- **Noisy Data.** The data is collected in a crowdsourced setting, so incorrect labels and low-quality out-of-domain samples are expected to be present in the training set.
- **Intra-Class Variability.** Since images of the same class can include indoor and outdoor views, as well as images of indirect relevance to a class, such as paintings in a museum.

It is hard to evaluate the importance of this challenge. The methods developed in this competition are being used in all following metric learning competitions. Large companies with image search services were closely monitoring this competition as well.

As usual, winning solutions contain a lot of tricks and wizardry. However, in this blog post, I will only provide a short recap on the Metric Learning methods used by top-performing teams:
- [ArcFace](#arcface), [CosFace](#cosface), or some variants of these methods were used by ALL competitors in the top 100. It's a huge contrast to previous Google Landmarks competitions, where [Triplet Loss](#triplet-loss) and variants of it were the most popular ones.
- [1st place winners (recognition)][glrec2020_1st_place], in contrast to all other teams, were able to solve the out-of-domain test images problem. They designed an inference process that takes into account the all-pair similarity between test images, landmark images, and non-landmark images.
- [3rd place winners (recognition)][glrec2020_3rd_place] used [Sub-Center ArcFace](#subcenter-arcface) with [Dynamic Margin](#afdynmargin). It is quite amazing that they were able to achieve such results (with a huge margin with 4th place team) with only raw Metric Learning, without relying on sophisticated post-processing or local features matching as other top-performing teams.
- The combinations of the methods developed by [1st place (recognition)][glrec2020_1st_place] and [3rd place (recognition)][glrec2020_3rd_place] teams can be combined into a powerful image recognition and retrieval algorithm.
- [GeM][gem_pooling] (with \(p=3\)) and [GAP][gap_pooling] pooling methods, followed by a linear layer, are used by all competitors in the top 100. There's no visible difference between those pooling methods. Some teams tried to train \(p\) in [GeM][gem_pooling] as well, but the performance gain was minimal.
- The general network architecture looks pretty similar to the following schema:

<a name="fig-net-gem-af"></a>
{{< figure src="arcface_network_arch.png" caption="The variants of this architecture were used by all teams in the top 100." invertible="true" >}}

- Class imbalance was resolved by [1st place (retrieval)][glret2020_1st_place] using a weighted cross-entropy on top of ArcFace. [5th place (retrieval)][glret2020_5th_place] proposed another approach, using Focal Loss with Label Smoothing on top of ArcFace.
- It seems like, without sophisticated post-processing, vanilla [ArcFace](#arcface) alone was not enough to achieve high leaderboard standing. Top teams in the Recognition track (i.e. [2nd place][glrec2020_2nd_place] and [7th place][glrec2020_7th_place]) additionally relied on local feature matching with [SuperPoint][superpoint] + [SuperGlue][superglue], which is a State-of-the-Art combo in feature matching.
- Funny, but everyone tried Dynamic AdaCos [(Zhang et al. 2019)][adacos_paper] because the results in the published paper were extremely promising. But nobody managed to make that work.

It seems to me that after this competition, the list of methods above became a standard for Metric Learning competitions on Kaggle platform.


[kaggle_glc2020]: https://www.kaggle.com/c/landmark-recognition-2020
[glrec2020_1st_place]: https://www.kaggle.com/c/landmark-recognition-2020/discussion/187821
[glrec2020_2nd_place]: https://www.kaggle.com/c/landmark-recognition-2020/discussion/188299
[glrec2020_3rd_place]: https://www.kaggle.com/c/landmark-recognition-2020/discussion/187757
[glrec2020_7th_place]: https://www.kaggle.com/c/landmark-recognition-2020/discussion/187894
[glret2020_1st_place]: https://www.kaggle.com/c/landmark-retrieval-2020/discussion/176037
[glret2020_5th_place]: https://www.kaggle.com/c/landmark-retrieval-2020/discussion/176151
[gem_pooling]: https://amaarora.github.io/2020/08/30/gempool.html
[gap_pooling]: https://paperswithcode.com/method/global-average-pooling
[superpoint]: https://arxiv.org/abs/1712.07629
[superglue]: https://arxiv.org/abs/1911.11763


<a name="shopee-price-match"></a>
### Kaggle: Shopee Price Match Guarantee (2021)

This competition is quite different compared to the two above. It involves Metric Learning for text data. Given an image of some product and its description, the task is to find similar products in the test set (not seen during training) by their images and text descriptions.

<a name="fig-shopee-intro"></a>
{{< figure src="shopee_intro.png" >}}

This competition is quite challenging because the train set is very small (34k images), but the test set is quite large and much more diverse than the training set (70k images). Moreover, a lot of product categories in the test set are not presented in the training set.

These 2 challenges mean that top competitors are forced to design very sophisticated post-processing pipelines, query expansion, and rely heavily on local features and information from text. However, it is still useful to take a look at how Deep Metric Learning is used in this competition:

- [ArcFace](#arcface) and [CosFace](#cosface) is being used by a lot of top teams, including [1st place][shopee_1st_place], [4th place][shopee_4th_place], and [8th place][shopee_8th_place] winning solutions. Interestingly, it is being used to learn **both image and text embeddings!** That's soo cool &mdash; you're not limited only to image data!
- [2nd place][shopee_2nd_place] team used a variant of ArcFace called [Curriculum Face (Huang et al. 2020)][curricular_face]. According to them, this loss function is much better than ArcFace and its variants.
- Witnessing the return of [Triplet Loss](#triplet-loss) used by the [3rd place team][shopee_3rd_place] was quite interesting. They tried ArcFace also, but strangely it did not work out for their team.
- Another interesting thing that top teams have noticed is that the optimal scaling and margin parameters for ArcFace is different for image and text models. So they had to tune them individually.

[shopee_1st_place]: https://www.kaggle.com/c/shopee-product-matching/discussion/238136
[shopee_2nd_place]: https://www.kaggle.com/c/shopee-product-matching/discussion/238022
[shopee_3rd_place]: https://www.kaggle.com/c/shopee-product-matching/discussion/238515
[shopee_4th_place]: https://www.kaggle.com/c/shopee-product-matching/discussion/238295
[shopee_5th_place]: https://www.kaggle.com/c/shopee-product-matching/discussion/238078
[shopee_8th_place]: https://www.kaggle.com/c/shopee-product-matching/discussion/238125
[curricular_face]: https://arxiv.org/pdf/2004.00288.pdf


-----------------------------------------------------------------------------


<a name="what-is-sota"></a>
## So, which method is State-of-the-Art?

It really depends on your specific task and your data. As of now, I would recommend the following:
- If you don't have much data, or in an unsupervised setting, [Triplet Loss](#triplet-loss) might still be a solid option. The 3rd place on [Shopee Price Match Guarantee Challenge (2020)](#shopee-price-match) clearly demonstrated that.
- Otherwise, in a strictly supervised setting, where you don't expect wildly out-of-distribution samples in test set, [ArcFace](#arcface) and its variants are definitely the way to go, as demonstrated by top performers of recent competitions on Kaggle. If your data has a large intra-class variability and a long tail of rare classes, [Sub-Center ArcFace](#subcenter-arcface) + [Dynamic Margin](#afdynmargin) is probably the method you need to consider. Don't forget to use [GeM][gem_pooling] and follow the architecture as shown in [Fig. 14](#fig-net-gem-af).
- Depending on the task, Metric Learning alone would often not be enough. In image retrieval tasks, it is often paired with [Query Expansion][wiki_qe], [Feature Matching][wiki_pfm], and other post-processing and verification methods.

I will try to keep this blog post updated with the latest State-of-the-Art methods.


[wiki_qe]: https://en.wikipedia.org/wiki/Query_expansion
[wiki_pfm]: https://en.wikipedia.org/wiki/Point_feature_matching



<a name="references"></a>
## References

1. Chopra, Hadsell, and LeCun. ["Learning a Similarity Metric Discriminatively, with Application to Face Verification."](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf) CVPR 2005.

2. Schroff, Kalenichenko, and Philbin. ["FaceNet: A Unified Embedding for Face Recognition and Clustering."](https://arxiv.org/abs/1503.03832) CVPR 2015.

3. Weihua Chen et al. ["Beyond triplet loss: a deep quadruplet network for person re-identification."](https://arxiv.org/abs/1704.01719)
 CVPR 2017.

4. Hyun Oh Song et al. ["Deep Metric Learning via Lifted Structured Feature Embedding."](https://arxiv.org/abs/1511.06452) CVPR 2016.

5. Herman, Beyer, and Leibe. ["In Defense of the Triplet Loss for Person Re-Identification."](https://arxiv.org/abs/1703.07737) CVPR 2017.

6. Kihyuk Sohn. ["Improved Deep Metric Learning with Multi-class N-pair Loss Objective."](https://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf) NIPS 2016.

7. Oren Rippel et al. ["Metric Learning with Adaptive Density Discrimination."](https://arxiv.org/abs/1511.05939) ICLR 2016.

8. Hyun Oh Song et al. ["Deep Metric Learning via Facility Location."](https://openaccess.thecvf.com/content_cvpr_2017/papers/Song_Deep_Metric_Learning_CVPR_2017_paper.pdf) CVPR 2017.

9. Horiguchi, Ikami, Aizawa. ["Significance of Softmax-based Features in Comparison to Distance Metric Learning-based Features."](https://arxiv.org/abs/1712.10151). Arxiv:1712.10151

10. Kaya and Bilge. ["Deep Metric Learning: A Survey."](https://www.mdpi.com/2073-8994/11/9/1066/htm) Symmetry 2019.

11. Wen, Zhang, and Li. ["A Discriminative Feature Learning Approach for Deep Face Recognition."](https://link.springer.com/chapter/10.1007/978-3-319-46478-7_31) ECCV 2016.

12. Weiyang Liu et al. ["SphereFace: Deep Hypersphere Embedding for Face Recognition."](https://arxiv.org/abs/1704.08063) CVPR 2017.

13. Hao Wang et al. ["CosFace: Large Margin Cosine Loss for Deep Face Recognition."](https://arxiv.org/abs/1801.09414) CVPR 2018.

14. Jiankang Deng et al. ["ArcFace: Additive Angular Margin Loss for Deep Face Recognition."](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf) CVPR 2019.

15. Xiao Zhang et al. ["AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations."](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_AdaCos_Adaptively_Scaling_Cosine_Logits_for_Effectively_Learning_Deep_Face_CVPR_2019_paper.pdf) CVPR 2019.

16. Jiankang Deng et al. ["Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces."](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf) ECCV 2020.

17. Quishen Ha et al. ["Google Landmark Recognition 2020 Competition Third Place Solution"](https://arxiv.org/pdf/2010.05350.pdf) Arxiv:2010.05350

18. Tobias Weyand et al. ["Google Landmarks Dataset v2 &ndash; A Large-Scale Benchmark for Instance-Level Recognition and Retrieval."](https://arxiv.org/abs/2004.01804) CVPR 2020.
