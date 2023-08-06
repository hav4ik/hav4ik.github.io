---
layout: post
permalink: /:title
type: "article"
title: "At the core of a Search Engine: Learning to Rank"
image:
  feature: "/articles/images/2021-08-15-learning-to-rank/feature.png"
  display: false
commits: "#"
tags: [information retrieval, tutorial, deep dive]
excerpt: "Search relevance ranking is one of the most important part of any search and recommendation system. This post is just my personal study notes, where I delve deeper into Learning-to-Rank (LTR) approaches and try to make sense of the current body of literature for myself."
show_excerpt: true
comments: true
hidden: false
highlighted: true
---

I still remember being fascinated by Google Search when I saw it the first time. As an 8th-grade kid getting his first computer, the 
ability to search for any information I want among billions of web pages looked like magic to me. As Arthur C. Clarke famously said, 
["any sufficiently advanced technology is indistinguishable from magic."][tech_is_magic] Search engines that 
allow us to access thousands of years of humanity's accumulated knowledge are truly the modern version of magic!

Back then, even in my wildest dreams, I couldn't have imagined that 25 years old me will have the privilege to move across the globe 
to work on a search engine called [Bing][bing] &mdash; an ambitious project with enough guts to compete with Google in the 
search market! Now that I can see how it works from the inside, the "magic" behind that little search box became even more 
impressive to me. The search engine is a truly gigantic marvel of modern technology, built and maintained by thousands of engineers.

{% comment %}

There is a lot for me to learn about and there is a lot of things that I don't know, so in this blog post, I'll take you together 
with me on my study journey about [Learning to Rank (LTR)][ltr] algorithms. I'm by no means an expert in this field so this post is 
likely to be filled with a lot of inaccuracies. If you spotted any mistakes in this post or if I'm completely wrong in some 
sections, please let me know.

{% endcomment %}

***Disclaimer:** all information in this blog post is taken from published research papers or publically available online articles. 
No [NDA][nda]s were violated. Only general knowledge is presented. You won't find any details specific to the inner working of [Bing]
[bing] or other search engines here :)*



[tech_is_magic]: https://en.wikipedia.org/wiki/Clarke%27s_three_laws
[bing]: https://www.bing.com/
[nda]: https://en.wikipedia.org/wiki/Non-disclosure_agreement
[ltr]: https://en.wikipedia.org/wiki/Learning_to_rank


- [1. How do search engines work?](#how-search-engines-work)
- [2. Introduction to Learning to Rank](#ltr-intro)
  - [2.1. Search Relevance](#search-relevance)
  - [2.2. Flavors of LTR methods](#ltr-flavors)
  - [2.3. Relevance Ranking Metrics](#ltr-metrics)
- [3. Supervised Learning to Rank methods](#supervised-ltr)
  - [3.1. RankNet](#ranknet)
  - [3.2. LambdaRank and LambdaMART](#lambdarank-and-lambdamart)
    - [3.2.1. Train $\lambda$MART using LightGBM](#train-lambdamart-using-lgbm)
    - [3.2.2. Theoretical justification of $\lambda$Rank](#theoretical-justification-of-lambrank)
  - [3.3. LambdaLoss Framework](#lambdaloss)
- [4. Unbiased Learning to Rank (from User Behavior)](#)
  - [4.1. Click signal biases](#)
  - [4.2. Counterfactual Learning to Rank](#)
    - [4.2.1. Full information Learning to Rank](#)
    - [4.2.2. Partial information Learning to Rank](#)
    - [4.2.3. What's wrong with naive estimator?](#)
    - [4.2.4. Inverse Propensity Weighting (IPW)](#)
    - [4.2.5. Dual Learning Algorithm (DLA)](#)
  - [Online Learning to Rank](#)
- [References](#)


---------------------------------------------------------------------------------


<a name="how-search-engines-work"></a>
## 1. How do search engines work?

Not all search engines are built with the ambitious goal of "searching the whole internet." Tech giants like Quora, Netflix, Amazon, 
and Facebook have in-house search engines as well, created to recommend the best products, content, and movies that match the user’s 
search queries. Big online retail companies, for example, also have their own search engines. That's how they recommend you the 
products that you are more likely to be interested in, given your prior purchases.

In information retrieval, the items that are being searched for (e.g. videos, books, web pages, etc.) are regarded as **documents.** 
All modern search engines, on the most abstract schematic level, have a similar underlying mechanism of searching for the most 
relevant documents for a given query:

{% capture imblock_search_engine %}
    {{ site.url }}/articles/images/2021-08-15-learning-to-rank/search_engine.png
{% endcapture %}
{% capture imcaption_search_engine %}
  Over-simplified general schema of search engines. Features extracted from all documents using the indexer are stored in the index database. For a search given query, top k documents are retrieved from the index database and then sorted by their relevance to the given query. *(Source: I drew it 😜)*
{% endcapture %}
{% include gallery images=imblock_search_engine cols=1 caption=imcaption_search_engine %}

**Indexing** is performed continuously offline. At this step, meaningful features and signals from all crawled documents are 
extracted and stored in the Index database. For retail companies, these features can be as primitive as raw description or [TF-IDF]
[tfidf] of the product description together with its popularity and user rating. For web-scale search engines like Google and [Bing]
[bing], the index is constructed from thousands of different signals and compressed embeddings from state-of-the-art neural 
networks. Needless to say, feature engineering is extremely important, so the choice of what signals and features to extract is kept 
secret by each search engine to maintain the competitive edge on the market.

**Top-k Retrieval** (sometimes also called *"Level-0 Ranking"* or *"Matching"*) is performed on each user's query to retrieve the 
potentially relevant documents for the given query. For small search engines, simple text matching is usually enough at this stage. 
For web-scale search engines, a hybrid of keyword (entity) matching and [Embedding][embedding_in_ml]-based Retrieval is used. In 
Embedding-based Retrieval, an embedding vector is calculated for the given query, and then k nearest embedding vectors (by euclidean 
or cosine similarity) of all documents stored in the Index database are retrieved.

[Huang et al. (2020)][fbsearch_embedding] described in detail how Facebook Search is using Embedding-based Retrieval in their search 
engine. [Bing Search][bing], according to their [2018 blog post][bing_img_search_2018], calculates image embeddings in addition to 
text embeddings for their retrieval stage. Google's blog post ["Building a real-time embeddings similarity matching system"][google_building_retrieval] 
gives us a glimpse of how Embedding-based Retrieval is likely to be performed inside Google, although their inner system is for sure 
much more sophisticated than that, and is probably combined Rule-based Retrieval as well.

Algorithmic nerds out there might find it interesting that metric trees (like [k-d tree][kdtree]) is not used in large-scale search 
engines due to their slow $O(\log n)$ complexity and large memory consumption. Instead, [Approximate Nearest Neighbors (ANN)][ann_methods] 
search (like [LHS][lhs_hashing] or [PCA hashing][pca_hashing]) is used to achieve close to $O(1)$ retrieval complexity. If you want 
to learn more about these algorithms, I highly recommend [this Medium post][ann_methods] about ANN search.

**Ranking** is the step that actually makes search engines work. Retrieved documents from the previous step are then ranked by their 
relevance to the given query and (optionally) the user's preferences. While hand-crafted heuristics and rule-based methods for 
relevance ranking are often more than enough for small and even mid-sized search engines, all big names in the industry right now 
are using Machine-Learning (i.e. [Learning-to-Rank][ltr]) techniques for search results ranking.

There was a time when [PageRank][pagerank] was a sole ranking factor for Google, but they quickly moved to more sophisticated 
ranking algorithms as more diverse features are extracted from web pages. As of 2020, [PageRank][pagerank] score is still a small 
part of Google's index, as [confirmed multiple times][pagerank_alive] by googlers. Interestingly, for a long time Google has 
resisted using machine learning for their core search ranking algorithm, as explained in [this Quora answer][google_hates_ml] from 
2011 by a former Google engineer. For more information about Google's algorithm changes over years, [this blog post][google_algo_changes] 
is an excellent tracker of their recent publically known major changes.


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



<a name="ltr-intro"></a>
## 2. Introduction to Learning to Rank

Given a query $q$ and a set of $n$ retrieved documents $\mathcal{D} = \{ d_1, d_2, \ldots, d_n \}$, we'd like to learn a function $f(\mathcal{Q}, \mathcal{D})$ that will return a correct ordering of the documents, such that the first documents would be the most relevant to the user. Usually, $f$ predicts a score for each document, and then the ranking order is determined by the scores.

{% capture imblock_ltrtask %}
    {{ site.url }}/articles/images/2021-08-15-learning-to-rank/ltr_task.png
{% endcapture %}
{% capture imcaption_ltrtask %}
  Given a query and a list of documents, the Learning-to-Rank task is to predict the relevance ranking of the documents, i.e. which document is the most relevant to the query. *(Source: I drew it 😜)*
{% endcapture %}
{% include gallery images=imblock_ltrtask cols=1 caption=imcaption_ltrtask %}


<a name="search-relevance"></a>
### 2.1. Search Relevance

Before talking about ranking search results, we first need to understand how to decide which result is relevant to the given query and which one is not, so that we can measure the ranking quality. There are many ways to estimate the relevance of search results, in both online and offline settings. In most cases, the relevance is defined as a combination of the following 3 factors:

**Human labeled.** Most search engines have an offline labor-intensive process to accurately measure search relevance and test their models. Google, for example, have a [long and notoriously defined guideline][google_sqe_guidelines] for human labelers to evaluate search quality. A similar evaluation process happens in [Bing][bing] and other companies that have search engines (Facebook, Amazon, etc.) as well. Given a query and a list of potential matching results, human labelers can assign a relevance score (i.e. from 1 to 5) to each result based on some standartized guideline. Alternatively, human labelers can also be asked which one of two given results is more relevant to the query.

**Click-through rate (CTR).** A cheap way to estimate search relevance is to count the number of times a user clicks on a given result on the page. There are a lot of issues with this method though. Usually, users tends to click on first results even when there are more relevant results below (also known as *position bias* which I will cover further below). The biggest issue is that users will rarely go to the next page of results, so it is hard to use click signals to estimate the relevance of a large number of results.

**Conversion Rate.** Depending on the purpose of the search/recommendation system, conversion can be defined as: buys, sales, profit, or any goal that defines the success of the business. For e-commerce, conversion rate is usually defined as number of buys divided by number of searches.


[google_sqe_guidelines]: https://static.googleusercontent.com/media/guidelines.raterhub.com/en//searchqualityevaluatorguidelines.pdf
[bing]: https://www.bing.com/



<a name="ltr-flavors"></a>
### 2.2. Flavors of LTR methods

Learning to Rank methods are divided into **Offline** and **Online** methods. In offline LTR, a model is trained in an offline setting with a fixed dataset. Online methods learns from the interactions with the user in real-time, and the model is updated after each interaction.

Offline methods can be further divided into **Supervised** methods, where for each document in the dataset its relevance to a query is judged by a human labeler, and **Counterfactual** methods, where the model is trained on historical data of user interactions (i.e. click-through rate) and/or document's conversion rate.

Supervised methods, depending on how the optimization objective is constructed, can be divided into **Pointwise** (look at a single document at a time in the loss function), **Pairwise** (look at a pair of documents at a time in the loss function), and **Listwise** (directly look at the entire list of documents) methods.

Online and Counterfactual LTR are extremely important classes of LTR methods and are currently active areas of research. They are much trickier to train than supervised approaches since both Online and Counterfactual methods learns from biased signals. Approaches to counter this bias are commonly called **Unbiased Learning-to-Rank**.


<a name="ltr-metrics"></a>
### 2.3. Relevance Ranking Metrics

Information retrieval researchers use ranking quality metrics such as [Mean Average Precision (**MAP**)][map-explained] which I'm sure many of you are familiar with, [Mean Reciprocal Rank (**MRR**)][wiki-mrr], Expected Reciprocal Rank (**ERR**), and Normalized Discounted Cumulative Gain (**NDCG**) to evaluate the quality of search results ranking. The former two (MAP and MRR) are widely used for documents retrieval but not for search results ranking because they don't take into account the relevance score for each document.

<a name="metrics-ndcg"></a>
To define **NDCG (Normalized Discounted Cumulative Gain)**, first we need to define the DCG (Discounted Cumulative Gain) metric. For a given query, the DCG of a list of search results is defined as:

$$
\begin{equation*}
DCG@T = \sum_{i=1}^T \frac{2^{l_i} - 1}{\log (1 + i)}
\end{equation*}
$$

where $T$ is the truncation level (for example, if we only care about the first page of results, we might take $T=10$), and $l_i$ is the relevance score label of the $i$-th result (for example, if $5$ levels of relevance is used, then $l_i \in {1, 2, \ldots, 5}$). The **NDCG** is just normalized version of DCG:

$$
\begin{equation}\label{eq:ndcg}\tag{NDCG}
NDCG@T = \frac{DCG@T}{\max DCG@T}
\end{equation}
$$

where the denominator $\max DCG@T$ is the maximum possible DCG for the given query and list of results, so that $NDCG@T \in [0, 1]$. This is the most commonly used metric.

> TODO: add NDCG visualization


<a name="metrics-err"></a>
**ERR (Expected Reciprocal Rank)** is used when a user is assumed to read down the list of returned search results until they find the one that they like. ERR is defined as:

$$
\begin{equation}\tag{ERR}
ERR = \sum_{r=1}^n \frac{1}{r} R_{r} \prod_{i=1}^{r-1} \left( 1 - R_i \right),
\quad \text{where} \enspace
R_i = \frac{2^{l_i} - 1}{2^{l_m}}
\end{equation}
$$

where $R_{i}$ models the probability that the user finds the document at $i$-th position relevant and $l_m$ is the maximum possible label value.


[map-explained]: https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
[wiki-mrr]: https://en.wikipedia.org/wiki/Mean_reciprocal_rank



---------------------------------------------------------------------------------



<a name="supervised-ltr"></a>
## 3. Supervised Learning to Rank methods

From 2005 to 2006, a series of incredibly important papers in Learning to Rank research were published by [Christopher Burges][burges-website], a researcher at [Microsoft][microsoft-research]. With **RankNet** [(Burges et al. 2005)][burges-ranknet], the LTR problem is re-defined as an optimization problem that can be solved using gradient descent. In **LambdaRank** and **LambdaMART** [(Burges et al. 2006)][burges-lambdarank], a method for directly optimizing NDCG was proposed. At the time of writing this blog post, LamdaMART is still being used as a strong baseline model, and can even out-perform newer methods on various benchmarks. If you want to know more about the story behind these methods, I highly recomment [this blog post by Microsoft][ranknet-retrospect].

In this section, I will closely follow the survey by [Burges (2010)][burges-ranknet-to-lambdamart] and describe these methods in the context of web search (but it is generic enough to be generalized to other domains, e.g. e-commerce). Given fixed query $\mathcal{Q}$, we define the following notations for the documents that matches the given query:

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
### 3.1. RankNet

[Burges et al. (2005)][burges-ranknet] proposed an optimization objective for the Learning-to-Rank problem so that a model can be trained using gradient descent. Let's model the learned probability $P_{ij}$ that $i$-th document should rank higher than $j$-th document as a sigmoid function:

$$
\begin{equation*}
P_{ij} = P\left({\bf U}_{i} \rhd {\bf U}_{j}\right) = \frac{1}{1 + e^{-\sigma (s_i - s_j)} }
\end{equation*}
$$

where the choice of the parameter $\sigma$ determines the shape of the sigmoid. Let $$\widetilde{P}_ {ij}$$ be the known probability that $${\bf U}_ {i} \rhd {\bf U}_ {j}$$ (it could be estimated by asking several judges to compare the pair and average out their answers). The **RankNet**'s cost function for a given query-document pair is defined as the [Cross Entropy][crossentropy]:

$$
\begin{equation}\label{eq:ranknet}
\mathcal{C}(s_i, s_j) = - \widetilde{P}_{ij} \log P_{ij} - (1 - \widetilde{P}_{ij}) \log (1 - P_{ij})
\end{equation}
$$

Obviously, this cost function is symmetric: swapping $i$ and $j$ and flipping the probabilities $P_{ij}$ and $\widetilde{P}_{ij}$ should not change the cost. At each minibatch descent step, the RankNet's cost is summarized from all document pairs in $\boldsymbol{\mathcal I}$.

RankNet opened a new direction in LTR research, and is the precursor to LambdaRank and LambdaMART. In 2015, the RankNet paper won the ICML Test of Time Best Paper Award, which honors “a paper from ICML ten years ago that has had substantial impact on the field of machine learning, including both research and practice.”


<a name="lambdarank-and-lambdamart"></a>
### 3.2. LambdaRank and LambdaMART

The objective of [RankNet](#ranknet) is optimizing for (a smooth, convex approximation to) the number
of pairwise errors, which is fine if that is the desired cost. However, it does not produce desired gradients for minimizing position-sensitive objectives like [NDCG](#metrics-ndcg) or [ERR](#metrics-ERR), as we will illustrate in figure below.

Ideally, we would want to be able to optimize the position-sensitive objectives in a more direct way. However, ranking metrics such as [NDCG](#metrics-ndcg) and [ERR](#metrics-ERR) are not differentiable objectives since sorting is needed
to obtain ranks based scores. This makes the ranking metrics either discontinuous or flat everywhere and can not be directly used as learning objectives. The **LambdaRank** framework, developed by [Burges et al. (2006)][burges-lambdarank], solves this problem by modifying the gradients during training.

Let's start from the RankNet's cost $\mathcal{C}$ defined in $(\ref{eq:ranknet})$. It is easy to see that $\partial \mathcal{C} / \partial s_i = - \partial \mathcal{C} / \partial s_j$ due to the symmetry of the RankNet's cost function. More specifically:

$$
\begin{equation*}
\frac{\partial \mathcal{C}}{\partial s_i} = \sigma \left( 1 - \widetilde{P}_{ij} - \frac{1}{1 + e^{\sigma (s_i - s_j)}} \right) = - \frac{\partial \mathcal{C}}{\partial s_j}
\end{equation*}
$$

Let's denote ${\boldsymbol \lambda}_{ij} = \partial \mathcal{C} / \partial s_i$. The gradients $\partial \mathcal{C} / \partial \theta$ of our model's weights averaged over the set $\boldsymbol{\mathcal I}$ of document pairs can be written as:

$$
\begin{align*}
  \frac{\partial \mathcal{C}}{\partial \theta}
  &=
  \sum_{\{i, j\} \in \boldsymbol{\mathcal I}}
  \frac{\partial \mathcal{C}}{\partial s_i} \frac{\partial s_i}{\partial \theta} +
  \frac{\partial \mathcal{C}}{\partial s_j} \frac{\partial s_j}{\partial \theta}
  &
  \textcolor{gray}{
    
  }

  \\ &=
  \sum_{\{i, j\} \in \boldsymbol{\mathcal I}}
  {\boldsymbol \lambda}_{ij} \left(
    \frac{\partial s_i}{\partial \theta} - \frac{\partial s_j}{\partial \theta}
  \right)
  &
  \textcolor{gray}{
    \text{Use}\enspace
    {\boldsymbol \lambda}_{ij} \triangleq \frac{\partial \mathcal{C}}{\partial s_i} = -\frac{\partial \mathcal{C}}{\partial s_j}
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
    {\boldsymbol \lambda}_{k} \triangleq  
    \sum_{\{k, j\} \in \boldsymbol{\mathcal I}} {\boldsymbol \lambda}_{kj} -
    \sum_{\{i, k\} \in \boldsymbol{\mathcal I}} {\boldsymbol \lambda}_{ik}
  }
\end{align*}
$$

where we introduced one $\boldsymbol{\lambda}_ {k}$ for each document. You can think of the $\boldsymbol{\lambda}$'s as little arrows (or forces), one attached to each (sorted) document, the direction of which indicates the direction we'd like the document to move (to increase relevance), the length of which indicates by how much, and where the $\boldsymbol{\lambda}$ for a given document is computed from all the pairs in which that document is a member.

> **NOTE:** add figure illustrating per-document forces

So far, each individual $\boldsymbol{\lambda}_ {ij}$ contributes equally to the magnitude of $\boldsymbol{\lambda}_ {i}$. That means the rankings of documents below, let's say, 100th position is given equal improtance to the rankings of the top documents. This is not what we want if the chosen metric is position-sensitive (like [NDCG](#metrics-ndcg) or [ERR](#metric-err)): we should prioritize having relevant documents at the very top much more than having correct ranking below 100th position. 

[Burges et al. (2006)][burges-lambdarank] proposed an elegant framework to this problem, called **LambdaRank**. Let's multiply each individual $ \boldsymbol{\lambda}_ {ij} $ by the amount of change in our chosen metric if we swap the positions of $i$-th and $j$-th documents. For example, if the chosen metric is [NDCG](#metrics-ndcg), then we adjust $\boldsymbol{\lambda}_ {ij}$ as follows:

$$
\begin{equation}
  {\boldsymbol \lambda}_{ij} \equiv \frac{\partial \mathcal{C}}{\partial s_i} \cdot \left| \Delta NDCG_{ij} \right|
\end{equation}
$$

where $\Delta NDCG_{ij}$ is the change in NDCG when the position of $i$-th and $j$-th documents are swapped and can be calculated as follows:

$$
\begin{equation*}
  \Delta NDCG_{ij} =
  \frac{2^{l_j} - 2^{l_i}}{\max DCG@T} \left( \frac{1}{\log(1 + i)} - \frac{1}{\log(1 + j)} \right)
\end{equation*}
$$

which takes $O(n^2)$ time to compute for all pair of documents. From first glance, $\Delta ERR_{ij}$ is harder to compute: the naive implementation would require $O(n^3)$ time to compute for all pair of documents. However, we can use the trick described by [Burges (2010)][burges-ranknet-to-lambdamart] to bring the cost down to $O(n^2)$.

**LambdaMART** is basically the same, but uses [Gradient Boosted Decision Trees](#gbdt) instead of Neural Networks (MART stands for Multiple Additive Regression Trees). Basically we perform gradient ascent in the models (function) space instead of the model's weights space.


[burges-website]: https://chrisburges.net/
[microsoft-research]: https://www.microsoft.com/en-us/research/
[crossentropy]: https://en.wikipedia.org/wiki/Cross_entropy
[ranknet-retrospect]: https://www.microsoft.com/en-us/research/blog/ranknet-a-ranking-retrospective/
[gbdt]: https://towardsdatascience.com/gradient-boosted-decision-trees-explained-9259bd8205af



<a name="train-lambdamart-using-lgbm"></a>
#### 3.2.1. Train $\lambda$MART using LightGBM

<a href="#"><img src="https://img.shields.io/badge/open_in_colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open in Colab"></a>
<a href="#"><img src="https://img.shields.io/badge/github-000000?style=for-the-badge&logo=github&logoColor=white" alt="Github"></a>

There a various implementations of LambdaMART, the most popular ones are [RankLib][ranklib]'s implementation and [LightGBM][lgbm] developed by Microsoft Research. Various benchmarks (i.e. [Qin et al. 2021][neural-rankers-vs-gbdt]) have shown that the [LightGBM][lgbm] implementation provides better performance.

We will use [MSLR-WEB30K][mslr-web30k] dataset, published in 2010 as an example. It is a retired commercial labeling set of [Microsoft Bing][bing], which contains more than 3'700'000 documents grouped into 30'000 queries. Each document is represented by a 136-dimensional feature vector and labeled with a relevance score ranging from 0 (irrelevant) to 4 (perfectly relevant). The dataset's format is similar to [LibSVM][libsvm] format. Let's download the dataset and make it consumable by LightGBM:

```bash
# Download the dataset
URL="https://api.onedrive.com/v1.0/shares/s!AtsMfWUz5l8nbXGPBlwD1rnFdBY/root/content"
wget $URL -O MSLR-WEB30K.zip

# Unzip only the 1st fold into "data/" folder
mkdir data; unzip -j "MSLR-WEB30K.zip" "Fold1/*" -d "data/"

# Process the dataset into proper LibSVM-like format
git clone https://github.com/guolinke/boosting_tree_benchmarks
cd data; python ../boosting_tree_benchmarks/data/msltr2libsvm.py
```

Loading the dataset into LightGBM is incredibly simple. Since our dataset is relatively small (only 3GB), we can load it entirely to the memory. For larger datasets, one of the options is to use the [HDF5][h5py] format. For more information about supported data formats, please refer to the [documentation][lgbm-docs].

```python
import lightgbm

# Load the dataset
train_data = lightgbm.Dataset('data/msltr.train')
valid_data = lightgbm.Dataset('data/msltr.test')

# LightGBM needs to know the sizes of query groups, which is already
# provided by `msltr2libsvm.py`.
train_group_size = [int(l.strip("\n")) for l in open('data/msltr.train.query')]
valid_group_size = [int(l.strip("\n")) for l in open('data/msltr.test.query')]
train_data.set_group(train_group_size)
valid_data.set_group(valid_group_size)
```

Now, we can easily train the model using the [`lightgbm.train`][lgbm-train] API:

```python
# LightGBM parameters. We use the same parameters as in
# https://lightgbm.readthedocs.io/en/latest/Experiments.html
param = {
  "task": "train",
  "num_leaves": 255,
  "min_data_in_leaf": 1,
  "min_sum_hessian_in_leaf": 100,
  "learning_rate": 0.1,
  "objective": "lambdarank",       # LambdaRank
  "metric": "ndcg",                # You can define your own metric, e.g. ERR
  "ndcg_eval_at": [1, 3, 5, 10],   # NDCG at ranks 1, 3, 5, 10
  "num_threads": mp.cpu_count(),   # Use all available CPUs
}

res = {}
bst = lightgbm.train(param, train_data, valid_sets=[valid_data],
                     num_boost_round=250, evals_result=res, verbose_eval=50)
```

At each interval, the script will output the NDCG scores at different ranks. As we can see, after 250 boosting rounds, the NDCG scores of our model already outperforms the [benchmark][lgbm-benchmark] by LightGBM.

```
[50]    ndcg@1: 0.497597   ndcg@3: 0.479561   ndcg@5: 0.483374   ndcg@10: 0.502566
[100]   ndcg@1: 0.513941   ndcg@3: 0.493917   ndcg@5: 0.498266   ndcg@10: 0.515446
[150]   ndcg@1: 0.516273   ndcg@3: 0.498433   ndcg@5: 0.502623   ndcg@10: 0.520829
[200]   ndcg@1: 0.51923    ndcg@3: 0.500929   ndcg@5: 0.506352   ndcg@10: 0.523464
[250]   ndcg@1: 0.522536   ndcg@3: 0.503643   ndcg@5: 0.508457   ndcg@10: 0.525354
```

It's always interesting to peek inside the model and see what features contributes the most to its performance. For this, we can use the [`lightgbm.plot_importance`][lgbm-plot-importance] API:

```python
from matplotlib import pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(14, 8))
lightgbm.plot_importance(bst, importance_type='split', ax=ax[0], max_num_features=20)
lightgbm.plot_importance(bst, importance_type='gain', ax=ax[1], max_num_features=20)
```

<a name="fig-lambdamart-fi"></a>
{% capture imblock_lambdamart_fi %}
  {{ site.url }}/articles/images/2021-08-15-learning-to-rank/feat_importance.svg
{% endcapture %}
{% capture imcaption_lambdamart_fin %}
  Top 20 most important features by split (left plot) and by gain (right plot) for the LambdaMART model trained on the MSLR-WEB30K dataset.
{% endcapture %}
{% include gallery images=imblock_lambdamart_fi cols=1 caption=imcaption_lambdamart_fin %}

From the plots, we can see that for `feature_importance='split'`, which sorts the features by numbers of times the feature is used in a model:

* The most important features  are **#131** (Site-level [PageRank][pagerank]) and **#130** ([PageRank][pagerank]). clearly, in the early days of [Bing][bing] prior to the development of deep learning features, PageRank was a strong indicator of the quality of the site (note that this dataset was published in 2010, and Bing was launched in 2009).
* Surprisingly, **#127** (Length of URL) and **#126** (Number of slashes in URL) were strong indicator back then. Likely because high-quality sites tend to have shorter URLs with fewer slashes.
* Then follows **#133** (QualityScore2) and **#132** (QualityScore), which are the quality score of a web page outputted by a web page quality classifier.
* **#108** (Title [BM25][bm25]) and **#110** (Whole document [BM25][bm25]) features are quite important as well. BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document, regardless of their proximity within the document.

If we instead sort the features by their gains (i.e. `feature_importance='gain'`), then interestingly **#134** (Query-url click count) came up on top. Clearly, the click count of a query-url pair at a search engine in a period, which is a strong indicator that the query-url pair is relevant to the user.


[lgbm]: https://lightgbm.readthedocs.io/en/latest/
[lgbm-docs]: https://lightgbm.readthedocs.io/en/latest/
[lgbm-train]: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
[lgbm-benchmark]: https://lightgbm.readthedocs.io/en/latest/Experiments.html
[ranklib]: https://sourceforge.net/p/lemur/wiki/RankLib/
[mslr-web30k]: https://www.microsoft.com/en-us/research/project/mslr/
[bing]: https://www.bing.com/
[libsvm]: https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#/Q3:_Data_preparation
[h5py]: https://docs.h5py.org/en/stable/
[lgbm-plot-importance]: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_importance.html
[pagerank]: https://en.wikipedia.org/wiki/PageRank
[bm25]: https://en.wikipedia.org/wiki/Okapi_BM25#:~:text=In%20information%20retrieval%2C%20Okapi%20BM25%20%28%20BM%20is,Stephen%20E.%20Robertson%2C%20Karen%20Sp%C3%A4rck%20Jones%2C%20and%20others.



<a name="theoretical-justification-of-lambrank"></a>
#### 3.2.2. Theoretical justification of $\lambda$Rank

Despite experimental success and promising results of $\lambda$Rank and $\lambda$MART in optimizing the ranking metrics like NDCG and ERR, there are few questions that has bothered researcher for a long time:

* From the theoretical perspective, does the iterative procedure employed by $\lambda$Rank converge?
* Is there an underlying global loss function for $\lambda$Rank. If so, what is it? How it relates to the metric?

[Donmez et al. (2009)][donmez-lambdatheory] empirically showed the local optimality of $\lambda$Rank by applying an one-sided Monte-Carlo test. They sample $n$ random directions uniformly from a unit sphere, then move the model's weights $\theta$ along each direction by a small $\eta > 0$ and check that the chosen metric $M$ always decreases. Their experimental results indicates that the iterative procedure employed by $\lambda$Rank converges to a local minimum with a high statistical significance.

[Burges et al. (2006)][burges-lambdarank] attempted to show that there is an underlying global loss function for $\lambda$Rank by using [Poincaré lemma][poincare-lemma] in differential geometry:

> **Poincaré lemma**. For a set of functions $f_1, \ldots, f_n \colon \mathbb{R}^n \to \mathbb{R}$, the existence of a function $F \colon \mathbb{R}^n \to \mathbb{R}$ such that $\partial F / \partial x_i = f_i$ is equivalent to $\textstyle \sum_{i} f_i dx^i$ being an exact form.

On $\mathbb{R}^n$ being exact means that the form is closed. More specifically:

$$
\begin{align*}
  d\left(\sum_{i} f_i dx_i \right) = 0
  \iff &
  \sum_{j < i} \left( \frac{df_i}{dx_j} - \frac{df_j}{dx_i} \right) dx_j \land dx_i = 0 \\
  \implies &
  \frac{df_i}{dx_j} = \frac{df_j}{dx_i} \enspace \text{for all pairs} \enspace i, j
\end{align*}
$$

For a given pair, for any particular values for the model weights, it is easy to verify that the requirement is satisfied due to the symmetries of $\lambda$Rank gradients. However, the existance of a global loss function remains unknown across iterations, since the model with updated weights generates the score by which the urls are sorted, and since the $\boldsymbol{\lambda}$'s are computed after the sort.

Finally, [Wang et al. (2019)][lambdaloss] developed a probabilistic framework, within which $\lambda$Rank optimizes for an upper bound of a well-defined cost function, which we will review closely in the next section.


[poincare-lemma]: http://nlab-pages.s3.us-east-2.amazonaws.com/nlab/show/Poincar%C3%A9%20lemma


<a name="lambdaloss"></a>
### 3.3. LambdaLoss Framework

> **TODO:** recap the main results of lambdaloss paper

---------------------------------------------------------------------------------


<a name="unbiased-ltr"></a>
## 4. Unbiased Learning to Rank (from User Clicks)

In the previous section, we have learned how to train a ranker on labeled data, where each document-query pair is annotated with a score (from 1 to 5) that shows how relevant that document is to the given query. This process is very expensive: to ensure the objectivity of labeled score, the human labeler would have to go through a strict checklist with multiple questions, then the document's relevance score will be calculated from the given answers. [Google's guidelines for search quality rating][google_sqe_guidelines] is a clear example of how complicated that process is (167 pages of guideline).

One might wonder **why we can't just use user's clicks as relevance labels?** Which is quite a natural idea: clicks are the main way users interacts with our web page, and the more relevant the document is to the given query, the more likely it is going to be clicked on. In this section, we will learn about approaches that allows us to learn directly from click data.

The structure of this section will closely follow the structure of the amazing lecture by [Oosterhuis et. al. (youtube, 2020)][ltr-lectures-harrie-youtube], which I personally used when I first started to learn about LTR.

Since we're learning from user clicks only, it is hereby natural to make following **assumptions:**
- By drawing the training signal directly from the user, it is more appropriate to talk about query instances $\boldsymbol{\mathcal{q}}$ that include contextual information about the user, instead of just a query string, since each user acts upon their own relevance judgement subject to their specific context and intention.
- Relevance label to each document is binary ($1$ if relevant to given query and specific user, $0$ if not).


[google_sqe_guidelines]: https://static.googleusercontent.com/media/guidelines.raterhub.com/en//searchqualityevaluatorguidelines.pdf
[ltr-lectures-harrie-youtube]: https://www.youtube.com/watch?v=BEEfMrn9T9c


<a name="click-biases"></a>
### 4.1. Click Signal Biases

User click should be a good indicator that a document is relevant to the user, right? Well, not that easy &mdash; not every relevant document is given an equal chance to be clicked by the user. Implicit user signals typically include multiple biases, the most common types are: position bias, selection bias, and trust bias. It is important to identify the types of biases, because for Unbiased Learning to Rank we will need to build models to estimate such biases.

**Position bias** occurs because users usually clicks on an item only after examining it, and users are more likely to examine the items displayed at the beginning of search results page, i.e. with higher ranks by the ranking model [(Craswell & Taylor, 2008)][experimental_comparison_of_click_models]. The best way to illustrate the effects of position bias is by tracking user's eyes while looking at returned search results:

<a name="fig-click-google"></a>
{% capture imblock_click_google %}
  {{ site.url }}/articles/images/2021-08-15-learning-to-rank/eyetrack_heatmap_google.jpg
{% endcapture %}
{% capture imcaption_click_google %}
  Eye-tracking heatmaps for Google Search page in 2004 (left) and 2014 (right).
{% endcapture %}
{% include gallery images=imblock_click_google cols=1 caption=imcaption_click_google %}

As we can clearly see, top results gets way more attention than bottom results. Also, notice how in 2004 the eye-tracking heatmap was concentrated on the top few results, but in 2014 the heatmap gets more "flattened" accross the whole results page. What have changed? Design.

> **Takeaway:** the design of a search results page can greatly affect the biases of the user's behavior. Make sure to adjust your position bias estimators after each major web page design change.

**Selection bias** occurs when some items have zero probability of being examined. Let's take Google search results page as an example. How often do you go to second search results page? To the third? Have you ever reached 10th page of Google's search results? The user rarely goes further than top few results. The distiction between position bias and selection bias is important because some methods can only correct for the former if the later is not present.

**Trust bias** occurs because the users trust the ranking system, so they are more likely to click on top ranked items even when they are not relevant. This may sound similar to position bias that we described above, because ultimately both of these biases amplifies the top items ranked by the relevance ranking model, but it's actually important to have this distinction if we want to build a model for such biases.


<!---------------------------------------------------------------------------------------------------------------
COUNTERFACTUAL LEARNING TO RANK
----------------------------------------------------------------------------------------------------------------->

<a name="counterfactual-ltr">
### 4.2. Counterfactual Learning to Rank

Counterfactual Learning to Rank is a family of LTR methods that learns from historical interactions between users and returned results (e.g. in form of clicks, content engagement, or other discrete or non-discrete factors). At the core of these methods lies the following idea:

> **Counterfactual Evaluation:** evaluate a new ranking function $f_\theta$ using historical interaction data collected from a previously deployed ranking function $f_\text{deploy}$.

In Unbiased Learning-to-Rank literature, the ranking function $f_\text{deploy}$ is often referred to as **"policy"** (or **"behavior policy"**), and the new ranking function $f_\theta$ (parametrized by $$\theta$$) is referred to as **"evaluation policy"**. The goal of counterfactual evaluation is to estimate the performance of the new ranking function $f_\theta$ without deploying it in production, which is often expensive and risky. The evaluation policy $f_\theta$ is trained on a dataset of historical interactions between users and the deployed ranking function $f_\text{deploy}$, and the performance of the evaluation policy $f_\theta$ is estimated by comparing the ranking of the evaluation policy $f_\theta$ to the ranking of the deployed ranking function $f_\text{deploy}$.


<a name="fullinfo-ltr">
#### 4.2.1. Full Information LTR

Before talking about approaches for Learning-to-Rank from biased implicit feedback (e.g. user clicks), let's review what we know so far about LTR from a curated & notoriously annotated dataset, where true relevance labels are known for each query-document pairs (i.e. we have full information about the data we're evaluating the model $$f_\theta$$ on). Given a sample $$\boldsymbol{\mathcal{Q}}$$ of queries $$\boldsymbol{\mathcal{q}} \sim P(\boldsymbol{\mathcal{q}})$$ for which we assume the user-specific binary relevances $$\boldsymbol{\mathcal{y}}^{\boldsymbol{\mathcal{q}}} = \{y^{\boldsymbol{\mathcal{q}}}_d\}_{d \in \boldsymbol{\mathcal{D}}}$$ of all documents $$d \in \boldsymbol{\mathcal{D}}$$ are known (assuming $\boldsymbol{\mathcal{q}}$ already captures user context), we can define overall empirical risk of a ranking system $$f_\theta$$ as follows:

$$
\begin{equation*}
  \hat{R}(f_\theta) = \sum_{\boldsymbol{\mathcal{q}} \in \boldsymbol{\mathcal{Q}}} {
    \frac{\mathcal{w}^\boldsymbol{\mathcal{q}}}{|\boldsymbol{\mathcal{Q}}|} \cdot
    \Delta \left( \boldsymbol{\mathcal{y}}^{\boldsymbol{\mathcal{q}}}, \boldsymbol{\pi}_\theta^{\boldsymbol{\mathcal{q}}} \right)
  }\,.
\end{equation*}
$$

There's quite a bit of shortcut notations here:
- $$\mathcal{w}^\boldsymbol{\mathcal{q}}$$ is the weight for each query $$\boldsymbol{\mathcal{q}}$$ (depending on its frequency, importance, or other criterias that are important to your business).
- $$\boldsymbol{\pi}^{\boldsymbol{\mathcal{q}}}_\theta$$ denotes the ranking of documents $$\boldsymbol{\mathcal{D}}$$ for query $$\boldsymbol{\mathcal{q}}$$ by the ranking system $$f_\theta$$ (parametrized by $$\theta$$). The individual rank of document $$d$$ is denoted as $$\boldsymbol{\pi}^{\boldsymbol{\mathcal{q}}}_\theta(d)$$, in other words it is the rank of document $$d$$ after calculating the document's score $$f_\theta(\boldsymbol{\mathcal{q}}, d)$$ and sorting by descending order;
- $$\Delta$$ denotes any additive linearly composable IR metric that measures ranking quality.

For query $$\boldsymbol{\mathcal{q}}$$, $$\Delta (\cdot, \cdot)$$ can be computed as follows:

$$
\begin{align*}
  \Delta \left( \boldsymbol{\mathcal{y}}^{\boldsymbol{\mathcal{q}}}, \boldsymbol{\pi}_\theta^{\boldsymbol{\mathcal{q}}} \right)
  & =
  \sum_{d \in \boldsymbol{\mathcal{D}}} {
    \mu \big[
      \boldsymbol{\pi}^{\boldsymbol{\mathcal{q}}}_\theta(d)
    \big] \cdot
    \boldsymbol{\mathcal{y}}^{\boldsymbol{\mathcal{q}}}_d
  }
  \\
  & =
  \sum_{d \in \boldsymbol{\mathcal{D}} \colon \boldsymbol{\mathcal{y}}^{\boldsymbol{\mathcal{q}}}_d = 1} {
    \mu \big[
      \boldsymbol{\pi}^{\boldsymbol{\mathcal{q}}}_\theta(d)
    \big]
  }
\end{align*}
$$

where $\mu$ is a rank weighting function, some of which were mentioned in the [Relevance Ranking Metrics section](#ltr-metrics). For example:
- For ARR (Average Relevant Position), $\mu(r) = r$.
- For DCG@T (Discounted Cumulative Gain at T), $\mu(r) = 1 / \log_2 (1 + r)$ if $r < T$ else $\mu(r) = 0$. For NDCG@T, just divide the whole thing to $\max \text{DCG@T}$.
- For precision at $k$, $\mu(r) = \boldsymbol{1} [r \le k] / k$.

Having so many variables and functions to keep in mind can be confusing and makes tracking core ideas harder, so let's simplify the notation a bit:

> Since we treat each query similarly (up to a weighting factor), from now on we can omit the query $\boldsymbol{\mathcal{q}}$ altogether in our notations when we're working with a single query.


<a name="partialinfo-ltr">
#### 4.2.2. Partial Information LTR

Since we don't know the true relevance $\boldsymbol{\mathcal{y}}_d$ of each document and rely on user clicks, we need to understand how the click biases plays out in practice. Let's take a look at toy example of a typical user session (also called "impression" in search and recommendation sysmtes) illustrated below:

<a name="fig-fullinfo-vs-clickinfo"></a>
{% capture imblock_fullinfo_vs_clickinfo %}
  {{ site.url }}/articles/images/2021-08-15-learning-to-rank/fullinfo_vs_clickinfo.png
{% endcapture %}
{% capture imcaption_fullinfo_vs_clickinfo %}
  Toy example of a typical user session (also called "impression"). Left: full information setting when you know true relevance $$y(d)$$ of each document. Right: partial information setting when you only have user click information and the true relevances $$y(d)$$ are not known. If the document is relevant and is observed by the user, then we might observe a click (i.e. $$d_1$$). Non-relevant documents can still get user clicks due to noise or trust bias (i.e. $$d_3$$). Un-observed documents are not getting any clicks at all even if they're relevant (i.e. $$d_4$$). *(Source: [Oosterhuis et al.](https://ilps.github.io/webconf2020-tutorial-unbiased-ltr/WWW2020handout.pdf))*
{% endcapture %}
{% include gallery images=imblock_fullinfo_vs_clickinfo cols=1 caption=imcaption_fullinfo_vs_clickinfo %}

A few obvious observations that is worth pointing out from the toy example above:
- A click $c_i$ on document $d$ is a **biased and noisy** indicator of its relevance. Sometimes, user click on an non-relevant item because they trust the search algorithm or simply noise.
- A missing click does not necessarily indicate non-relevance. The user might not click on a relevant document for various reasons.
- If a document was not examined by the user (i.e. the user did not scroll down to that document, or did not go to 2nd search page), we can't tell anything about its relevance.

The above observation is very primitive and does not include other kinds of deeper biases that might affect the user's behavior. However, it still captures the most common factors that we need to take into account: *position bias*, *selection bias*, and *click noise*.


<a name="naiveestimator">
#### 4.2.3. What's wrong with Naive Estimator?

Let $$\boldsymbol{o} = \{ o_d \}_{d \in \boldsymbol{\mathcal{D}}}$$ denote the indicator of which relevance values are being revealed (think of it as "the user saw the document and decided that it is relevant enough"). For each document $$d$$, denote $$P(o_d = 1 \vert \boldsymbol{\mathcal{\pi}})$$ as the marginal probability of observing the relevance $$y_d$$ of result $$d$$ for the given user query $$\boldsymbol{\mathcal{q}}$$, if the user was presented a ranking $$\boldsymbol{\mathcal{\pi}}$$. This probability value is called *propensity* of the observation. Later, we will discuss how this propensity can be estimated from different click models.

For the sake of demonstration simplicity, let's only consider *examination* and *relevance*: a user clicks on the document [if and only if][iff_wiki] the user had a chance to observe the document and the document is perceived as relevant to the given query.

Within this simplified framework, the user-specific relevance value observation probability depends only on the position $$i$$ at which the document $$d$$ is being presented. In our simplified model, this probability encapsulates both *position* and *selection* biases because ultimately they depend only on the presentation order.

A naive way to estimate $$\Delta$$ is to assume that clicks are unbiased indicators of document's relevance, i.e. $$c_d = 1 \iff y_d = 1$$:

$$
\begin{align*}
\Delta_{\text{naive}} \left( \boldsymbol{\mathcal{c}}, \boldsymbol{\pi} \right)
& =
\sum_{d \in \boldsymbol{\mathcal{D}}} {
  \mu \big[
    \boldsymbol{\pi}(d)
  \big]
  \cdot c_d
}
\\ & =
\sum_{d \in \boldsymbol{\mathcal{D}} \colon c_d = 1} {
  \mu \big[
    \boldsymbol{\pi}(d)
  \big]
}\,.
\end{align*}
$$

One can easily show that, even when no click noise or other biases is present, this estimator is biased by the examination probabilities:

$$
\begin{align*}
\mathbb{E}_o\big[ \Delta_{\text{naive}} \left( \boldsymbol{\mathcal{c}}, \boldsymbol{\pi} \right) \big]
& =
\mathbb{E}_o \left[ 
  \sum_{d \colon o_d = 1 \\ \land y_d = 1} {
    \mu \big[
      \boldsymbol{\pi}(d)
    \big]
  }
\right]
\\
& =
\sum_{d \colon y_d = 1} {
  P\left( o_d = 1 \vert \boldsymbol{\pi} \right) \cdot
  \mu \big[
    \boldsymbol{\pi}(d)
  \big]
}
\\
& =
\sum_{d \in \boldsymbol{\mathcal{D}}} {
  \underbrace{
    P\left( o_d = 1 \vert \boldsymbol{\pi} \right)
  }_{\text{bias}}
  \cdot
  \underbrace{
    \mu \big[
      \boldsymbol{\pi}(d)
    \big]
    \cdot y(d)
  }_{\text{estimates }\Delta}
} \,.
\end{align*}
$$

The biased estimator $$\Delta_{\text{naive}}$$ weights documents according to their examination probabilities in the ranking displayed during logging.


[iff_wiki]: https://en.wikipedia.org/wiki/If_and_only_if


<a name="inverse-propensity-weighting">
#### 4.2.4. Inverse Propensity Weighting

The naive estimator above can be easily de-biased by dividing each term by its bias factor. That's the basic idea of **Inverse Propensity Weighting** Estimator, first applied to the Learning to Rank problem in the works of [Joachims et al. (2016)][joachims_2016] and [Wang et al. (2016)][wang_2016]. For any new ranking $$\boldsymbol{\pi}_{\phi}$$ (different from the ranking $$\boldsymbol{\mathcal{\pi}}_{\theta}$$ presented to the user), the IPS estimator is defined as:

$$
\begin{equation*} \tag{IPW} \label{eq:ipw}
  \Delta_{\text{IPW}} \left(
    \boldsymbol{\mathcal{\pi}}_\phi, \boldsymbol{\mathcal{y}} \vert
    \boldsymbol{\pi}_\theta
  \right)
  =
  \sum_{d \colon o_d = 1} {
    \frac{
      \mu \big[
        \boldsymbol{\pi}_\phi(d)
      \big]
      \cdot y_d
    }{
      P\left(o_d = 1 \vert \boldsymbol{\pi}_\theta \right)
    }
  }
\end{equation*}
$$

This is an unbiased estimate of $$\Delta \left( \boldsymbol{\mathcal{\pi}}_\phi, \boldsymbol{\mathcal{y}} \right)$$ for any ranking $$\boldsymbol{\mathcal{\pi}}$$ and relevance observation indicator $$\boldsymbol{o}$$, if $$P\left(o_d = 1 \vert \boldsymbol{\mathcal{\pi}}\right) > 0$$ for all documents that are relevant for the specific user (again, we assume that the query $$\boldsymbol{q}$$ captures user context as well), i.e. $$y_d = 1$$ (but not necessarily for the irrelevant documents).

$$
\begin{equation}
\begin{aligned}
\mathbb{E}_{\boldsymbol{o}^{\boldsymbol{q}}}
\big[
  \Delta_{\text{IPW}} \left(
    \boldsymbol{\mathcal{\pi}}_\phi, \boldsymbol{\mathcal{y}} \vert
    \boldsymbol{\pi}_\theta
  \right)
\big]
& =
\mathbb{E}_{\boldsymbol{o}^{\boldsymbol{q}}}
\left[
\sum_{d \colon o_d = 1} {
  \frac{
    \mu \big[
      \boldsymbol{\pi}_\phi(d)
    \big]
    \cdot y_d
  }{
    P\left(o_d = 1 \vert \boldsymbol{\mathcal{\pi}}_{\theta}\right)
  }
}
\right]
\\ &=
\sum_{d \in \boldsymbol{\mathcal{D}}} {
  \mathbb{E}_{\boldsymbol{o}^{\boldsymbol{q}}}
  \left[
  \frac{
    o_d \cdot
    \mu \big[
      \boldsymbol{\pi}_\phi(d)
    \big]
    \cdot y_d
  }{
    P\left(o_d = 1 \vert \boldsymbol{\mathcal{\pi}}_{\theta}\right)
  }
  \right]
}
\\ &=
\sum_{d \in \boldsymbol{\mathcal{D}}} {
  \frac{
    P\left(o_d = 1 \vert \boldsymbol{\mathcal{\pi}}_\theta \right) \cdot
    \mu \big[
      \boldsymbol{\pi}_\phi(d)
    \big]
    \cdot y_d
  }{
    P\left(o_d = 1 \vert \boldsymbol{\mathcal{\pi}}_{\theta}\right)
  }
}
\\ &=
\sum_{d \in \boldsymbol{\mathcal{D}}} {
  \mu \big[
    \boldsymbol{\pi}_\phi(d)
  \big]
  \cdot y_d
}
\\ &=
\Delta \left( \boldsymbol{\mathcal{\pi}}_\phi, \boldsymbol{y}\right)
\,.
\end{aligned}
\label{eq:ipw_unbiased}
\end{equation}
$$

Note that this estimator sums only over the results where the relevance feedback is observed (i.e. $$o(d) = 1$$) and positive (i.e. $$y_d = 1$$). Therefore, we only need the propensities $$P\left(o_d = 1 \vert \boldsymbol{\mathcal{\pi}}\right)$$ for the relevant documents, which means that we do not have to disambiguate whether lack of positive feedback (e.g., the lack of a click) is due to a lack of relevance or due to missing the observation (e.g., result not relevant vs. not viewed). An additional requirement for making $$\Delta_\text{IPS}$$ computable while remaining unbiased is that the propensities only depends on observable information.



<a name="dual-learning-algorithm">
#### 4.2.5. Dual Learning Algorithm (DLA)

The most crucial part of Inverse Propensity Weighting (IPW) is to accurately model the click propensities. Most of such click bias estimation methods (that were described in [Section 4.1](#click-biases)) either conduct randomization of the search results ordering in online setting (which negatively affects user experience) or offline estimation which often has special assumptions and requirements for click data and is optimized for objectives that are not directly related to the main ranking metric.

Let's denote $$\boldsymbol{c}^{\boldsymbol{q}} = \{ c_d^\boldsymbol{q}\}_{d \in \boldsymbol{\mathcal{D}}}$$ be a Bernoulli variable that represent whether the documents in the ranked list $$\boldsymbol{\mathcal{\pi}}^{\boldsymbol{q}}$$ presented to the user got clicked on. Let's assume that a click on a document happens only when that document was observed and it is perceived as relevant by the user. If we model the probability of a click (conditioned by a ranking $$\boldsymbol{\mathcal{\pi}}$$):

$$
\begin{equation} \label{eq:symmetric_click}
P\left( c_d = 1 \vert \boldsymbol{\mathcal{\pi}} \right) = 
P\left( o_d = 1 \vert \boldsymbol{\mathcal{\pi}} \right) \cdot
P\left( y_d = 1 \vert \boldsymbol{\mathcal{\pi}} \right)
\end{equation}
$$

We cannot directly infer the relevance of a document without knowing whether it was examined. Symmetrically, we also cannot estimate the propensity of observation without knowing whether the documents are relevant or not. The key observation here is that $$o_d$$ and $$y_d$$ are interchangeable in $$\eqref{eq:symmetric_click}$$, so we can treat the *Unbiased Propensity Estimation* and *Unbiased Learning to Rank* problems as dual to each other, so they can be solved jointly in a symmetric way. That's the core idea of **Dual Learning Algorithm** proposed by [Ai et al. (2018)][ai_2018].

In Learning to Rank problem we need to train a ranking model $$f_\theta$$ that minimizes the global loss function $$\mathcal{L}_\text{R}$$ accross all the queries $$\boldsymbol{\mathcal{Q}}$$. Similarly, in Propensity Estimation problem we need to train a propensity model $$g_\psi$$ that minimizes $$\mathcal{L}_\text{P}$$. The global loss functions are defined as:

$$
\begin{align*}
\mathcal{L}_\text{R} \left( f_\theta \right) & =
\mathbb{E}_{\boldsymbol{q} \in \boldsymbol{\mathcal{Q}}} \left[
\mathcal{l}_{\text{R}} \left( f_\theta, \boldsymbol{q} \right)
\right]
\\
\mathcal{L}_\text{P} \left( g_\psi \right) & =
\mathbb{E}_{\boldsymbol{q} \in \boldsymbol{\mathcal{Q}}} \left[
\mathcal{l}_{\text{P}} \left( g_\psi, \boldsymbol{q} \right)
\right]
\end{align*}
$$

where $$\mathcal{l}_{\text{R}}$$ and $$\mathcal{l}_{\text{P}}$$ are the local loss functions for ranking and propensity models respectively. The key idea of the Dual Learning Algorithm is to jointly train both models:

<div class="algorithm">
  <div class="algo-name">Dual Learning  Algorithm (DLA)</div>
  <div class="algo-input">
    training data $\boldsymbol{\mathcal{Q}} = \left\{ \boldsymbol{q}, \boldsymbol{\pi}^\boldsymbol{q}, \boldsymbol{c}^\boldsymbol{q} \right\}$ &mdash; set of user sessions (query, presented ranking, user clicks log), initial Ranking model $f_\theta$ and Propensity model $g_\psi$ (parametrized by $\theta$ and $\psi$).
  </div>
  <div class="algo-output">
    Ranking model $f_{\theta^*}$ and Propensity model $g_{\psi^*}$.
  </div>
  <div class="algo-body">
    <div>Initialize $\theta$ and $\psi$ (either randomly or pre-train ranking and propensity models separately).</div>
    <!-- Repeat block -->
    <div><b>repeat:</b></div>
    <div class="algo-indent-block">
      <div>Randomly sample a batch $\boldsymbol{\mathcal{Q}}' \subset \boldsymbol{\mathcal{Q}}\,$;</div>
      <!-- For block (session in batch) -->
      <div><b>for each</b> user session $\left(\boldsymbol{q}, \boldsymbol{\pi}^\boldsymbol{q}, \boldsymbol{c}^\boldsymbol{q}\right) \in \boldsymbol{\mathcal{Q}}'$ <b>do:</b></div>
      <div class="algo-indent-block">
        <!-- For block (docs in session)-->
        <div><b>for each</b> document $d \in \boldsymbol{\pi}^\boldsymbol{q}$ <b>do:</b></div>
        <div class="algo-indent-block">
          <div>Compute propensities $P\left( o_d = 1 \vert \boldsymbol{\pi}^\boldsymbol{q} \right)$ and relevances $P\left( y_d = 1 \vert \boldsymbol{\pi}^\boldsymbol{q} \right)$ using $g_\psi$ and $f_\theta\,$;</div>
        </div>
        <div><b>end</b></div>
        <!-- for doc in session -->
      </div><div><b>end</b></div>
      <!-- for session in batch -->
      <div>Compute loss values $\mathcal{l}_{\text{R}}$ and $\mathcal{l}_{\text{P}}$ using $\boldsymbol{\pi}^\boldsymbol{q}$, $\boldsymbol{c}^\boldsymbol{q}$, $P\left( o_d = 1 \vert \boldsymbol{\pi}^\boldsymbol{q} \right)$ and $P\left( y_d = 1 \vert \boldsymbol{\pi}^\boldsymbol{q} \right)\,$;</div>
      <div>Update $\theta$ and $\psi$ using gradients $\nabla_\theta \mathcal{l}_{\text{R}}$ and $\nabla_\psi \mathcal{l}_{\text{P}}\,$;</div>
    </div>
    <div><b>until:</b> Convergence</div>
    <!-- Repeat -->
  </div> <!-- algo-body -->
  <div class="algo-end"></div>
</div>

Similarly to IPW, we can define an Inverted Relevance Weighting (IRW) loss function to be additive over the documents in the dataset:

$$
\begin{equation*} \tag{IRW} \label{eq:irw}
  \Delta_{\text{IRW}} \left(
    \boldsymbol{\mathcal{\pi}}_\phi, \boldsymbol{\mathcal{o}} \vert
    \boldsymbol{\pi}_\theta
  \right)
  =
  \sum_{d \colon y_d = 1} {
    \frac{
      \mu \big[
        \boldsymbol{\pi}_\phi(d)
      \big]
      \cdot o_d
    }{
      P\left(y_d = 1 \vert \boldsymbol{\pi}_\theta \right)
    }
  }
\end{equation*}
$$

Notice that it looks exactly like ($$\ref{eq:ipw}$$), just with the observance indicator $$\boldsymbol{o}$$ and relevance indicator $$\boldsymbol{y}$$ swapped. We can show that is is an unbiased estimator of $$\Delta \left( \boldsymbol{\mathcal{\pi}}_\phi, \boldsymbol{\mathcal{o}} \right)$$ by following the same steps as in $$\eqref{eq:ipw_unbiased}$$:

$$
\begin{equation}
\mathbb{E}_{\boldsymbol{y}^{\boldsymbol{q}}}
\big[
  \Delta_{\text{IRW}} \left(
    \boldsymbol{\mathcal{\pi}}_\phi, \boldsymbol{\mathcal{o}} \vert
    \boldsymbol{\pi}_\theta
  \right)
\big]
=
\Delta \left( \boldsymbol{\mathcal{\pi}}_\phi, \boldsymbol{y}\right)
\,.
\label{eq:irw_unbiased}
\end{equation}
$$

In this paper, [Ai et al. (2018)][ai_2018] also provided a rigorous proof that the Dual Learning Algorithm (DLA) will converge, under assumption that only position bias is considered.



---------------------------------------------------------------------------------


<a name="references"></a>
## References

1. Huang et al. ["Embedding-based Retrieval in Facebook Search."][fb-search-engine] In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2020.

2. C.J.C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton, G. Hullender. ["Learning to Rank using Gradient Descent."][burges-ranknet] In *ICML*, 2005.

3. Christopher J.C. Burges, Robert Ragno, Quoc V. Le. ["Learning to Rank with Nonsmooth Cost Functions."][burges-lambdarank] In *NIPS*, 2006.

4. Christopher J.C. Burges. ["From RankNet to LambdaRank to LambdaMART: An Overview."][burges-ranknet-to-lambdamart] *Microsoft Research Technical Report MSR-TR-2010-82*, 2010.

5. Qin Z., Yan L., Zhuang H., Tay Y., Pasumarthi K. R., Wang X., Bendersky M., Najork M. ["Are Neural Rankers still Outperformed by Gradient Boosted Decision Trees?"][neural-rankers-vs-gbdt] In *ICLR*, 2021.

6. Tao Qin, Tie-Yan Liu. ["Introducing LETOR 4.0 Datasets."][letor4] In *Arxiv:1306.2597*, 2013.

7. Pinar Donmez, Krysta M. Svore, Chris J.C. Burges. ["On the Local Optimality of LambdaRank."][donmez-lambdatheory] In *SIGIR*, Pages 460–467, 2009.

8. Xuanhui Wang Cheng Li Nadav Golbandi Mike Bendersky Marc Najork. ["The LambdaLoss Framework for Ranking Metric Optimization."][lambdaloss] In *CIKM*, 2018.

9. Nick Craswell, Mike Taylor. ["An experimental comparison of click position-bias models."][experimental_comparison_of_click_models] In *Proceedings of the international conference on Web search and web data mining (WSDM)*, 2008.

10. Thorsten Joachims, Laura Granka, Bing Pan, Helene Hembrooke, and Geri Gay. ["Accurately
interpreting clickthrough data as implicit feedback."][joachims_2005] In SIGIR, 2005.

11. Thorsten Joachims, Adith Swaminathan, Tobias Schnabel. ["Unbiased Learning-to-Rank with Biased Feedback."][joachims_2016] In IJCAI, 2018.

12. Xuanhui Wang, Michael Bendersky, Donald Metzler, Marc Najork. ["Learning to Rank with Selection Bias in Personal Search."][wang_2016] In SIGIR, 2016.

13. Ai, Qingyao, Keping Bi, Cheng Luo, Jiafeng Guo, and W. Bruce Croft. ["Unbiased learning to rank with unbiased propensity estimation."][ai_2018] In The 41st international ACM SIGIR conference on research & development in information retrieval, pp. 385-394. 2018.

14. Ai, Qingyao, Tao Yang, Huazheng Wang, and Jiaxin Mao. ["Unbiased learning to rank: online or offline?."][ai_2021] ACM Transactions on Information Systems (TOIS) 39, no. 2 (2021): 1-29.


[burges-ranknet]: https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/
[burges-lambdarank]: https://papers.nips.cc/paper/2006/hash/af44c4c56f385c43f2529f9b1b018f6a-Abstract.html
[burges-ranknet-to-lambdamart]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
[fb-search-engine]: https://arxiv.org/abs/2006.11632
[neural-rankers-vs-gbdt]: https://openreview.net/pdf?id=Ut1vF_q_vC
[letor4]: https://arxiv.org/abs/1306.2597
[donmez-lambdatheory]: https://www.microsoft.com/en-us/research/publication/on-the-local-optimality-of-lambdarank/
[lambdaloss]: https://research.google/pubs/pub47258/
[google_sqe_guidelines]: https://static.googleusercontent.com/media/guidelines.raterhub.com/en//searchqualityevaluatorguidelines.pdf
[experimental_comparison_of_click_models]: https://www.microsoft.com/en-us/research/publication/an-experimental-comparison-of-click-position-bias-models/
[joachims_2005]: https://www.cs.cornell.edu/people/tj/publications/joachims_etal_17a.pdf
[joachims_2016]: https://arxiv.org/abs/1608.04468
[wang_2016]: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45286.pdf
[ai_2018]: https://ciir-publications.cs.umass.edu/getpdf.php?id=1297
[ai_2021]: https://arxiv.org/abs/2004.13574