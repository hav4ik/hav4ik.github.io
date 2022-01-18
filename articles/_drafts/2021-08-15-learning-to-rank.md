---
layout: post
permalink: /articles/:title
type: "article"
title: "Learning to Rank: at the core of a Search Engine"
image:
  feature: "/articles/images/2021-08-15-learning-to-rank/feature.png"
  display: false
commits: "#"
tags: [machine learning, tutorial, deep dive]
excerpt: "Search relevance ranking is one of the most important part of any search and recommendation system. This post is just my personal study notes, where I delve deeper into Learning-to-Rank (LTR) approaches and try to make sense for myself."
show_excerpt: true
comments: true
hidden: false
highlighted: true
---

I still remember being fascinated by Google Search when I saw it the first time. As an 8th-grade kid getting his first computer, the ability to search for any information I want among billions of web pages looked like magic to me. As Arthur C. Clarke famously said, ["any sufficiently advanced technology is indistinguishable from magic."][tech_is_magic] By that definition, the search engines that allow us to access thousands of years of humanity's accumulated knowledge at our fingertip, are the modern version of magic!

{% comment %}

Back then, even in my wildest dreams, I couldn't have imagined that 25 years old me will have the privilege to move across the globe to work on a search engine called [Microsoft Bing][bing] &mdash; an ambitious project with enough guts to compete with Google in the search market! Now that I can see how it works from the inside, the "magic" behind that little search box became even more impressive to me. The search engine is a truly gigantic marvel of modern technology, built and supported by tens of thousands of hardware engineers, software developers, and machine learning scientists.

There is a lot for me to learn about and there is a lot of things that I don't know, so in this blog post, I'll take you together with me on my study journey about [Learning to Rank (LTR)][ltr] algorithms. I'm by no means an expert in this field so this post is likely to be filled with a lot of inaccuracies. If you spotted any mistakes in this post or if I'm completely wrong in some sections, please let me know.

> **Disclaimer:** all information in this blog post is taken from published research papers or publically available online articles. No [NDA][nda]s were violated. You won't find any details specific to the inner working of [Bing][bing] or other search engines here :)

{% endcomment %}


[tech_is_magic]: https://en.wikipedia.org/wiki/Clarke%27s_three_laws
[bing]: https://www.bing.com/
[nda]: https://en.wikipedia.org/wiki/Non-disclosure_agreement
[ltr]: https://en.wikipedia.org/wiki/Learning_to_rank


- [How do search engines work?](#how-search-engines-work)
- [Search Relevance](#search-relevance)
- [Relevance Ranking Metrics](#metrics)
- [Classical LTR methods](#classical-ltr)
  - [RankNet](#ranknet)
  - [LambdaRank and LambdaMART](#lambdarank-and-lambdamart)
  - [Theoretical justification of LambdaMART](#)
  - [Train LambdaMART using LightGBM](#train-lambdamart-using-lgbm)
- [Click signal biases](#)
- [References](#)


---------------------------------------------------------------------------------


<a name="how-search-engines-work"></a>
## How do search engines work?

Not all search engines are built with the ambitious goal of "searching the whole internet." Tech giants like Quora, Netflix, Amazon, and Facebook have in-house search engines as well, created to recommend the best products, content, and movies that match the user’s search queries. Big online retail companies, for example, also have their own search engines. That's how they recommend you the products that you are more likely to be interested in, given your prior purchases.

In information retrieval, the items that are being searched for (e.g. videos, books, web pages, etc.) are regarded as **documents.** All modern search engines, on the most abstract schematic level, have a similar underlying mechanism of searching for the most relevant documents for a given query:

{% capture imblock_search_engine %}
    {{ site.url }}/articles/images/2021-08-15-learning-to-rank/search_engine.png
{% endcapture %}
{% capture imcaption_search_engine %}
  Over-simplified general schema of search engines. Features extracted from all documents using the indexer are stored in the index database. For a search given query, top k documents are retrieved from the index database and then sorted by their relevance to the given query.
{% endcapture %}
{% include gallery images=imblock_search_engine cols=1 caption=imcaption_search_engine %}

**Indexing** is performed continuously offline. At this step, meaningful features and signals from all crawled documents are extracted and stored in the Index database. For retail companies, these features can be as primitive as raw description or [TF-IDF][tfidf] of the product description together with its popularity and user rating. For web-scale search engines like Google and [Bing][bing], the index is constructed from thousands of different signals and compressed embeddings from state-of-the-art neural networks. Needless to say, feature engineering is extremely important, so the choice of what signals and features to extract is kept secret by each search engine to maintain the competitive edge on the market.

**Top-k Retrieval** (sometimes also called *"Level-0 Ranking"* or *"Matching"*) is performed on each user's query to retrieve the potentially relevant documents for the given query. For small search engines, simple text matching is usually enough at this stage. For web-scale search engines, a hybrid of keyword (entity) matching and [Embedding][embedding_in_ml]-based Retrieval is used. In Embedding-based Retrieval, an embedding vector is calculated for the given query, and then k nearest embedding vectors (by euclidean or cosine similarity) of all documents stored in the Index database are retrieved.

[Huang et al. (2020)][fbsearch_embedding] described in detail how Facebook Search is using Embedding-based Retrieval in their search engine. [Bing Search][bing], according to their [2018 blog post][bing_img_search_2018], calculates image embeddings in addition to text embeddings for their retrieval stage. Google's blog post ["Building a real-time embeddings similarity matching system"][google_building_retrieval] gives us a glimpse of how Embedding-based Retrieval is likely to be performed inside Google, although their inner system is for sure much more sophisticated than that, and is probably combined Rule-based Retrieval as well.

Algorithmic nerds out there might find it interesting that metric trees (like [k-d tree][kdtree]) is not used in large-scale search engines due to their slow $$O(\log n)$$ complexity and large memory consumption. Instead, [Approximate Nearest Neighbors (ANN)][ann_methods] search (like [LHS][lhs_hashing] or [PCA hashing][pca_hashing]) is used to achieve close to $$O(1)$$ retrieval complexity. If you want to learn more about these algorithms, I highly recommend [this Medium post][ann_methods] about ANN search.

**Ranking** is the step that actually makes search engines work. Retrieved documents from the previous step are then ranked by their relevance to the given query and (optionally) the user's preferences. While hand-crafted heuristics and rule-based methods for relevance ranking are often more than enough for small and even mid-sized search engines, all big names in the industry right now are using Machine-Learning (i.e. [Learning-to-Rank][ltr]) techniques for search results ranking.

There was a time when [PageRank][pagerank] was a sole ranking factor for Google, but they quickly moved to more sophisticated ranking algorithms as more diverse features are extracted from web pages. As of 2020, [PageRank][pagerank] score is still a small part of Google's index, as [confirmed multiple times][pagerank_alive] by googlers. Interestingly, for a long time Google has resisted using machine learning for their core search ranking algorithm, as explained in [this Quora answer][google_hates_ml] from 2011 by a former Google engineer. For more information about Google's algorithm changes over years, [this blog post][google_algo_changes] is an excellent tracker of their recent publically known major changes.


> **Note:** Despite the deceptive simplicity of the above described schema, for web-scale search engines everything is a million times more complicated. Only few companies have enough infrastructure, computing resources, and manpower to develop and deploy search engines at such scale.


[tfidf]: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
[bing]: https://www.bing.com/
[pagerank]: https://en.wikipedia.org/wiki/PageRank
[pagerank_alive]: https://www.seroundtable.com/google-still-uses-pagerank-29056.html
[bert]: https://arxiv.org/abs/1810.04805
[ocr]: https://en.wikipedia.org/wiki/Optical_character_recognition
[fbsearch_embedding]: https://arxiv.org/pdf/2006.11632.pdf
[kdtree]: https://en.wikipedia.org/wiki/K-d_tree
[knn]: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
[google_building_retrieval]: https://cloud.google.com/architecture/building-real-time-embeddings-similarity-matching-system
[lhs_hashing]: https://en.wikipedia.org/wiki/Locality-sensitive_hashing
[pca_hashing]: https://ieeexplore.ieee.org/document/7926439
[ltr]: https://en.wikipedia.org/wiki/Learning_to_rank
[google_algo_changes]: https://www.searchenginejournal.com/google-algorithm-history/
[google_hates_ml]: https://www.quora.com/Why-is-machine-learning-used-heavily-for-Googles-ad-ranking-and-less-for-their-search-ranking-What-led-to-this-difference/answer/Edmond-Lau
[bing_img_search_2018]: https://blogs.bing.com/search-quality-insights/May-2018/Internet-Scale-Deep-Learning-for-Bing-Image-Search
[embedding_in_ml]: https://datascience.stackexchange.com/questions/53995/what-does-embedding-mean-in-machine-learning
[ann_methods]: https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6


---------------------------------------------------------------------------------


<a name="search-relevance"></a>
## Search Relevance

Before talking about ranking search results, we first need to understand how to decide which result is relevant to the given query and which one is not, so that we can measure the ranking quality. There are many ways to estimate the relevance of search results, in both online and offline settings. In most cases, the relevance is defined as a combination of the following 3 factors:

**Human labeled.** Most search engines have an offline labor-intensive process to accurately measure search relevance and test their models. Google, for example, have a [long and notoriously defined guideline][google_sqe_guidelines] for human labelers to evaluate search quality. A similar evaluation process happens in [Bing][bing] and other companies that have search engines (Facebook, Amazon, etc.) as well. Given a query and a list of potential matching results, human labelers can assign a relevance score (i.e. from 1 to 5) to each result based on some standartized guideline. Alternatively, human labelers can also be asked which one of two given results is more relevant to the query.

**Click-through rate (CTR).** A cheap way to estimate search relevance is to count the number of times a user clicks on a given result on the page. There are a lot of issues with this method though. Usually, users tends to click on first results even when there are more relevant results below (also known as *position bias* which I will cover further below). The biggest issue is that users will rarely go to the next page of results, so it is hard to use click signals to estimate the relevance of a large number of results.

**Conversion Rate.** Depending on the purpose of the search/recommendation system, conversion can be defined as: buys, sales, profit, or any goal that defines the success of the business. For e-commerce, conversion rate is usually defined as number of buys divided by number of searches.


[google_sqe_guidelines]: https://static.googleusercontent.com/media/guidelines.raterhub.com/en//searchqualityevaluatorguidelines.pdf
[bing]: https://www.bing.com/


---------------------------------------------------------------------------------


<a name="metrics"></a>
## Relevance Ranking Metrics


Information retrieval researchers use ranking quality metrics such as [Mean Average Precision (**MAP**)][map-explained] which I'm sure many of you are familiar with, [Mean Reciprocal Rank (**MRR**)][wiki-mrr], Expected Reciprocal Rank (**ERR**), and Normalized Discounted Cumulative Gain (**NDCG**) to evaluate the quality of search results. The former two (MAP and MRR) are widely used for documents retrieval but not for search results ranking. Let's take a closer look at the latter two.

> TODO: ADD REASON


<a name="metrics-ndcg"></a>
To define **NDCG (Normalized Discounted Cumulative Gain)**, first we need to define the DCG (Discounted Cumulative Gain) metric. For a given query, the DCG of a list of search results is defined as:

$$
\begin{equation*}
DCG@T = \sum_{i=1}^T \frac{2^{l_i} - 1}{\log (1 + i)}
\end{equation*}
$$

where $$T$$ is the truncation level (for example, if we only care about the first page of results, we might take $$T=10$$), and $$l_i$$ is the relevance score label of the $$i$$-th result (for example, if $$5$$ levels of relevance is used, then $$l_i \in {1, 2, \ldots, 5}$$). The **NDCG** is just normalized version of DCG:

$$
\begin{equation}\label{eq:ndcg}\tag{NDCG}
NDCG@T = \frac{DCG@T}{\max DCG@T}
\end{equation}
$$

where the denominator $$\max DCG@T$$ is the maximum possible DCG for the given query and list of results, so that $$NDCG@T \in [0, 1]$$. This is the most commonly used metric.


<a name="metrics-err"></a>
**ERR (Expected Reciprocal Rank)** is used when a user is assumed to read down the list of returned search results until they find the one that they like. ERR is defined as:

$$
\begin{equation}\tag{ERR}
ERR = \sum_{r=1}^n \frac{1}{r} R_{r} \prod_{i=1}^{r-1} \left( 1 - R_i \right),
\quad \text{where} \enspace
R_i = \frac{2^{l_i} - 1}{2^{l_m}}
\end{equation}
$$

where $$R_{i}$$ models the probability that the user finds the document at $$i$$-th position relevant and $$l_m$$ is the maximum possible label value.



[map-explained]: https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
[wiki-mrr]: https://en.wikipedia.org/wiki/Mean_reciprocal_rank


---------------------------------------------------------------------------------


<a name="classical-ltr"></a>
## Classical LTR methods

From 2005 to 2006, a series of incredibly important papers in Learning to Rank research were published by [Christopher Burges][burges-website], a researcher at [Microsoft][microsoft-research]. With **RankNet** [(Burges et al. 2005)][burges-ranknet], the LTR problem is re-defined as an optimization problem that can be solved using gradient descent. In **LambdaRank** and **LambdaMART** [(Burges et al. 2006)][burges-lambdarank], a method for directly optimizing NDCG was proposed. At the time of writing this blog post, LamdaMART is still being used as a strong baseline model, and can even out-perform newer methods on various benchmarks. If you want to know more about the story behind these methods, I highly recomment [this blog post by Microsoft][ranknet-retrospect].

In this section, I will closely follow the survey by [Burges (2010)][burges-ranknet-to-lambdamart] and describe these methods in the context of web search (but it is generic enough to be generalized to other domains, e.g. e-commerce). Given fixed query $$\mathcal{Q}$$, we define the following notations for the documents that matches the given query:

<table class="notations-table">
<tr><th>Notation</th><th class="notations-desc">Description</th></tr>
<tr>
<td>\( {\bf U}_{i} \)</td>
<td>The \(i\)-th document (or web page URL) that matches the given query.</td>
</tr>
<tr>
<td>\( {\bf x}_{i} \)</td>
<td>Feature vector for the \(i\)-th result, computed from the query-document pair \((\mathcal{Q}, {\bf U}_{i})\).</td>
</tr>
<tr>
<td>\( f_\theta (\cdot) \)</td>
<td>The model (either a Neural Net or GBDT) with weights \(\theta\) that we want to train.</td>
</tr>
<tr>
<td>\( s_i = f_\theta ({\bf x}_{i}) \)</td>
<td>Score for the \(i\)-th result, computed using a model \(f\) with weights \(\theta\).</td>
</tr>
<tr>
<td>\( {\bf U}_{i} \rhd {\bf U}_{j} \)</td>
<td>Denotes the event that \( {\bf U}_{i} \) should rank higher than \( {\bf U}_{j} \).</td>
</tr>
<tr>
<td>\( \boldsymbol{\mathcal I} \)</td>
<td>Set of pairs \(\{ i, j\}\) such that \({\bf U}_{i} \rhd {\bf U}_{j}\). Note that it is not symmetrical.</td>
</tr>
</table>



<a name="ranknet"></a>
### RankNet

[Burges et al. (2005)][burges-ranknet] proposed an optimization objective for the Learning-to-Rank problem so that a model can be trained using gradient descent. Let's model the learned probability $$P_{ij}$$ that $$i$$-th document should rank higher than $$j$$-th document as a sigmoid function:

$$
\begin{equation*}
P_{ij} = P\left({\bf U}_{i} \rhd {\bf U}_{j}\right) = \frac{1}{1 + e^{-\sigma (s_i - s_j)} }
\end{equation*}
$$

where the choice of the parameter $$\sigma$$ determines the shape of the sigmoid. Let $$\widetilde{P}_{ij}$$ be the known probability that $${\bf U}_{i} \rhd {\bf U}_{j}$$ (it could be estimated by asking several judges to compare the pair and average out their answers). The **RankNet**'s cost function for a given query-document pair is defined as the [Cross Entropy][crossentropy]:

$$
\begin{equation}\tag{RankNet}
\mathcal{C}(s_i, s_j) = - \widetilde{P}_{ij} \log P_{ij} - (1 - \widetilde{P}_{ij}) \log (1 - P_{ij})
\end{equation}
$$

Obviously, this cost function is symmetric: swapping $$i$$ and $$j$$ and flipping the probabilities $$P_{ij}$$ and $$\widetilde{P}_{ij}$$ should not change the cost. At each minibatch descent step, the RankNet's cost is summarized from all document pairs in $$\boldsymbol{\mathcal I}$$.

RankNet opened a new direction in LTR research, and is the precursor to LambdaRank and LambdaMART. In 2015, the RankNet paper won the ICML Test of Time Best Paper Award, which honors “a paper from ICML ten years ago that has had substantial impact on the field of machine learning, including both research and practice.”


<a name="lambdarank-and-lambdamart"></a>
### LambdaRank and LambdaMART

The objective of [RankNet](#ranknet) is optimizing for (a smooth, convex approximation to) the number
of pairwise errors, which is fine if that is the desired cost. However, it does not produce desired gradients for minimizing position-sensitive objectives like [NDCG](#metrics-ndcg) or [ERR](#metrics-ERR) (as shown in figure below).

> NOTE: add figure illustrating gradients

Ideally, we would want to be able to optimize the position-sensitive objectives in a more direct way (just a reminder &mdash; both [NDCG](#metrics-ndcg) and [ERR](#metrics-ERR) are not differentiable objectives). That is exactly what the **LambdaRank** framework, developed by [Burges et al. (2006)][burges-lambdarank], allows us to do.

Let's start from the RankNet's cost $$\mathcal{C}$$. It is easy to see that $$\partial \mathcal{C} / \partial s_i = - \partial \mathcal{C} / \partial s_j$$ due to the symmetry of the RankNet's cost function. More specifically:

$$
\begin{equation*}
\frac{\partial \mathcal{C}}{\partial s_i} = \sigma \left( 1 - \widetilde{P}_{ij} - \frac{1}{1 + e^{\sigma (s_i - s_j)}} \right) = - \frac{\partial \mathcal{C}}{\partial s_j}
\end{equation*}
$$

Let's denote $${\boldsymbol \lambda}_{ij} = \partial \mathcal{C} / \partial s_i$$. The gradients $$\partial \mathcal{C} / \partial \theta$$ of our model's weights averaged over the set $$\boldsymbol{\mathcal I}$$ of document pairs can be written as:

$$
\begin{align*}
  \frac{\partial \mathcal{C}}{\partial \theta}
  &=
  \sum_{\{i, j\} \in \boldsymbol{\mathcal I}}
  \frac{\partial \mathcal{C}}{\partial s_i} \frac{\partial s_i}{\partial \theta} +
  \frac{\partial \mathcal{C}}{\partial s_j} \frac{\partial s_j}{\partial \theta}
  &
  \textcolor{gray}{
    \text{Apply the chain rule}
  }

  \\ &=
  \sum_{\{i, j\} \in \boldsymbol{\mathcal I}}
  {\boldsymbol \lambda}_{ij} \left(
    \frac{\partial s_i}{\partial \theta} - \frac{\partial s_j}{\partial \theta}
  \right)
  &
  \textcolor{gray}{
    \text{Use}\enspace
    {\boldsymbol \lambda}_{ij} = \frac{\partial \mathcal{C}}{\partial s_i} = -\frac{\partial \mathcal{C}}{\partial s_j}
  }

  \\ &=
  \sum_{k} \left[
    \sum_{\{k, j\} \in \boldsymbol{\mathcal I}} {\boldsymbol \lambda}_{kj} \frac{\partial s_k}{\partial \theta} -
    \sum_{\{i, k\} \in \boldsymbol{\mathcal I}} {\boldsymbol \lambda}_{ik} \frac{\partial s_k}{\partial \theta}
  \right]
  &
  \textcolor{gray}{
    \text{Re-group the summation}
  }

  \\ &=
  \sum_{k} {\boldsymbol \lambda}_{k} \frac{\partial s_k}{\partial \theta}
  &
  \textcolor{gray}{
    {\boldsymbol \lambda}_{k} =
    \sum_{\{k, j\} \in \boldsymbol{\mathcal I}} {\boldsymbol \lambda}_{kj} -
    \sum_{\{i, k\} \in \boldsymbol{\mathcal I}} {\boldsymbol \lambda}_{ik}
  }
\end{align*}
$$

where we introduced one $${\boldsymbol \lambda}_{k}$$ for each document. You can think of the $${\boldsymbol \lambda}$$'s as little arrows (or forces), one attached to each (sorted) document, the direction of which indicates the direction we'd like the document to move (to increase relevance), the length of which indicates by how much, and where the $${\boldsymbol \lambda}$$ for a given document is computed from all the pairs in which that document is a member.

So far, each individual $${\boldsymbol \lambda}_{ij}$$ contributes equally to the magnitude of $${\boldsymbol \lambda}_{i}$$. That means re-ranking the documents below, let's say, 30th position is given equal improtance to re-ranking the top documents. This is not what we want: we should prioritize having relevant documents at the top much more than having correct ranking below 30th position.




[burges-website]: https://chrisburges.net/
[microsoft-research]: https://www.microsoft.com/en-us/research/
[crossentropy]: https://en.wikipedia.org/wiki/Cross_entropy
[ranknet-retrospect]: https://www.microsoft.com/en-us/research/blog/ranknet-a-ranking-retrospective/


---------------------------------------------------------------------------------


<a name="train-lambdamart-using-lgbm"></a>
## Hands-on Tutorial



```python
model = lightgbm.LGBMRanker(
  objective="lambdarank",
  metric="ndcg",

  # Almost the same parameters as described in the benchmark:
  # https://lightgbm.readthedocs.io/en/latest/Experiments.html
  learning_rate=0.1,
  num_leaves=255,
  n_estimators=200,
  num_threads=multiprocessing.cpu_count(),
  min_data_in_leaf=0,
  min_sum_hessian_in_leaf=100,
  bagging_fraction=0.7,
)
```


---------------------------------------------------------------------------------


<a name="references"></a>
## References

1. Huang et al. ["Embedding-based Retrieval in Facebook Search."][fb-search-engine] In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2020.

2. C.J.C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton, G. Hullender. ["Learning to Rank using Gradient Descent."][burges-ranknet] In *ICML*, 2005.

3. Christopher J.C. Burges, Robert Ragno, Quoc V. Le. ["Learning to Rank with Nonsmooth Cost Functions."][burges-lambdarank] In *NIPS*, 2006.

4. Christopher J.C. Burges. ["From RankNet to LambdaRank to LambdaMART: An Overview."][burges-ranknet-to-lambdamart] *Microsoft Research Technical Report MSR-TR-2010-82*, 2010.


[burges-ranknet]: https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/
[burges-lambdarank]: https://papers.nips.cc/paper/2006/hash/af44c4c56f385c43f2529f9b1b018f6a-Abstract.html
[burges-ranknet-to-lambdamart]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
[fb-search-engine]: https://arxiv.org/abs/2006.11632