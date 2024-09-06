---
title: "At the core of a Search Engine: Learning to Rank"
url: "/learning-to-rank"
date: 2024-09-01T00:00:00+00:00
# weight: 1
# aliases: ["/first"]
tags: ["RecSys"]
author: "Kha Vu Chan"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: true
hidemeta: false
comments: false
summary: "Learning to Rank is a core component of any recommendation system. It is the algorithm that forms the final list of items to be shown to the user. This blog post is a comprehensive introduction to the landscape of LTR algorithms. Hopefully, it will give you enough context to start reading recent LTR research papers."
canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
hideAuthor: true
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: false
ShowPostNavLinks: true
ShowWordCount: false
ShowRssButtonInSectionTermList: true
UseHugoToc: false
strikethrough: true
cover:
    image: "feature-transformed.png" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide in list view
editPost:
    URL: "https://github.com/hav4ik.github.io/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---


I still remember being fascinated by Google Search when I saw it the first time. As an 8th-grade kid getting his first computer back in 2010, the 
ability to search for any information I want among billions of web pages looked like magic to me. As Arthur C. Clarke famously said, 
["any sufficiently advanced technology is indistinguishable from magic."][tech_is_magic] Search engines that 
allow us to access thousands of years of humanity's accumulated knowledge are truly the modern version of magic!

Back then, even in my wildest dreams, I couldn't have imagined that 25 years old me will have the privilege to move across the globe to work on a search engine called [Bing][bing] &mdash; an ambitious project with enough guts to compete with Google in the search market! Now that I can see how it works from the inside, the "magic" behind that little search box became even more  impressive to me. The search engine is a truly gigantic marvel of modern technology, built and maintained by hundreds of engineers.

In this blog post, I'll walk you through the basic body of literature of [Learning to Rank (LTR)][ltr] algorithms, starting from the core metrics and supervised methods to the more recent paradigm of learning from user behavior. The list of papers that I'm going to cover is based on my learning notes and by no means exhaustive, but I hope it will give you a good starting point to dive deeper into the field.

I'm far from an expert in this field (literally all my team members are more knowledgeable than me), so this post likely contains inaccuracies. If you spotted any mistakes in this post or if I'm completely wrong somewhere, please let me know.

***Disclaimer:** all information in this blog post is taken from published research papers or publically available online articles. 
No [NDA][nda]s were violated. Only general knowledge is presented. You won't find any details specific to the inner working of [Bing]
[bing] or other search engines here :)*

[tech_is_magic]: https://en.wikipedia.org/wiki/Clarke%27s_three_laws
[bing]: https://www.bing.com/
[nda]: https://en.wikipedia.org/wiki/Non-disclosure_agreement
[ltr]: https://en.wikipedia.org/wiki/Learning_to_rank


------------------------------------------------------------


# 1. How do search engines work?

Not all search engines are built with the ambitious goal of "searching the whole internet." Tech giants like Quora, Netflix, Amazon, 
and Facebook have in-house search engines as well, created to recommend the best products, content, and movies that match the user’s 
search queries. Big online retail companies, for example, also have their own search engines. That's how they recommend you the 
products that you are more likely to be interested in, given your prior purchases.

In information retrieval, the items that are being searched for (e.g. videos, books, web pages, etc.) are regarded as **documents.** 
All modern search engines, on the most abstract schematic level, have a similar underlying mechanism of searching for the most 
relevant documents for a given query:


{{< figure src="search_engine.png" caption="Over-simplified general schema of search engines. Features extracted from all documents using the indexer are stored in the index database. For a search given query, top k documents are retrieved from the index database and then sorted by their relevance to the given query." invertible="false" >}}


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
engines due to their slow \( O(\log n) \) complexity and large memory consumption. Instead, [Approximate Nearest Neighbors (ANN)][ann_methods] 
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


# 2. Introduction to Learning to Rank

Given a query \( q \) and a set of \( n \) retrieved documents \( \mathcal{D} = \{ d_1, d_2, \ldots, d_n \} \), we'd like to learn a function \( f(\mathcal{Q}, \mathcal{D}) \) that will return a correct ordering of the documents, such that the first documents would be the most relevant to the user. Usually, \( f \) predicts a score for each document, and then the ranking order is determined by the scores.


{{< figure src="ltr_task.png" caption="Given a query and a list of documents, the Learning-to-Rank task is to predict the relevance ranking of the documents, i.e. which document is the most relevant to the query." invertible="false" >}}


## 2.1. Search Relevance

Before talking about ranking search results, we first need to understand how to decide which result is relevant to the given query and which one is not, so that we can measure the ranking quality. There are many ways to estimate the relevance of search results, in both online and offline settings. In most cases, the relevance is defined as a combination of the following 3 factors:

**Human labeled.** Most search engines have an offline labor-intensive process to accurately measure search relevance and test their models. Google, for example, have a [long and notoriously defined guideline][google_sqe_guidelines] for human labelers to evaluate search quality. A similar evaluation process happens in [Bing][bing] and other companies that have search engines (Facebook, Amazon, etc.) as well. Given a query and a list of potential matching results, human labelers can assign a relevance score (i.e. from 1 to 5) to each result based on some standartized guideline. Alternatively, human labelers can also be asked which one of two given results is more relevant to the query.

**Click-through rate (CTR).** A cheap way to estimate search relevance is to count the number of times a user clicks on a given result on the page. There are a lot of issues with this method though. Usually, users tends to click on first results even when there are more relevant results below (also known as *position bias* which I will cover further below). The biggest issue is that users will rarely go to the next page of results, so it is hard to use click signals to estimate the relevance of a large number of results.

**Conversion Rate.** Depending on the purpose of the search/recommendation system, conversion can be defined as: buys, sales, profit, or any goal that defines the success of the business. For e-commerce, conversion rate is usually defined as number of buys divided by number of searches.


[google_sqe_guidelines]: https://static.googleusercontent.com/media/guidelines.raterhub.com/en//searchqualityevaluatorguidelines.pdf
[bing]: https://www.bing.com/



<a name="ltr-flavors"></a>
## 2.2. Flavors of LTR methods

Learning to Rank methods are divided into **Offline** and **Online** methods. In offline LTR, a model is trained in an offline setting with a fixed dataset. Online methods learns from the interactions with the user in real-time, and the model is updated after each interaction.

Offline methods can be further divided into **Supervised** methods, where for each document in the dataset its relevance to a query is judged by a human labeler, and **Counterfactual** methods, where the model is trained on historical data of user interactions (i.e. click-through rate) and/or document's conversion rate.

Supervised methods, depending on how the optimization objective is constructed, can be divided into **Pointwise** (look at a single document at a time in the loss function), **Pairwise** (look at a pair of documents at a time in the loss function), and **Listwise** (directly look at the entire list of documents) methods.

Online and Counterfactual LTR are extremely important classes of LTR methods and are currently active areas of research. They are much trickier to train than supervised approaches since both Online and Counterfactual methods learns from biased signals. Approaches to counter this bias are commonly called **Unbiased Learning-to-Rank**.


<a name="ltr-metrics"></a>
## 2.3. Relevance Ranking Metrics

Information retrieval researchers use ranking quality metrics such as [Mean Average Precision (**MAP**)][map-explained] which I'm sure many of you are familiar with, [Mean Reciprocal Rank (**MRR**)][wiki-mrr], Expected Reciprocal Rank (**ERR**), and Normalized Discounted Cumulative Gain (**NDCG**) to evaluate the quality of search results ranking. The former two (MAP and MRR) are widely used for documents retrieval but not for search results ranking because they don't take into account the relevance score for each document.

<a name="metrics-ndcg"></a>
To define **NDCG (Normalized Discounted Cumulative Gain)**, first we need to define the DCG (Discounted Cumulative Gain) metric. For a given query, the DCG of a list of search results is defined as:

$$
\begin{equation*}
DCG@T = \sum_{i=1}^T \frac{2^{l_i} - 1}{\log (1 + i)}
\end{equation*}
$$

where \( T \) is the truncation level (for example, if we only care about the first page of results, we might take \( T=10 \)), and \( l_i \) is the relevance score label of the \( i \)-th result (for example, if \( 5 \) levels of relevance is used, then \( l_i \in {1, 2, \ldots, 5} \)). The **NDCG** is just normalized version of DCG:

$$
\begin{equation}\label{eq:ndcg}\tag{NDCG}
NDCG@T = \frac{DCG@T}{\max DCG@T}
\end{equation}
$$

where the denominator \( \max DCG@T \) is the maximum possible DCG for the given query and list of results, so that \( NDCG@T \in [0, 1] \). This is the most commonly used metric.

<a name="fig-ndcg-at-10"></a>
{{< figure src="ndcg_vis.png" caption="Visualization of NDCG@5 metric for different rankings of a collection of retrieved documents, with relevance (to a hypothetical query) scores \( \left[0, 0, 0, 0, 1, 2, 3, 4, 4, 5\right] \). Left: the best ranking (highest NDCG@5). Right: the worst ranking (lowest NDCG@5). Middle: a random ranking. Notice that the best ranking has the highest possible NDCG@5 of \( 1.00 \), while the worst ranking of the retrieved documents has a non-zero score, because it still has some relevant documents in the list. The only way to get zero NDCG@10 is to have no relevant documents in the list." >}}

In [Figure 3](#fig-ndcg-at-10), you can see how the NDCG@5 metric behaves for different rankings. To get a better intuition on the NDCG metric and see how it behaves under different permutations, feel free to play around with the visualization script: [https://gist.github.com/hav4ik/100aa247eff4d3075db4f8314461f4c2](gist.github.com/hav4ik/100aa247eff4d3075db4f8314461f4c2).

<a name="metrics-err"></a>
**ERR (Expected Reciprocal Rank)** is used when a user is assumed to read down the list of returned search results until they find the one that they like. ERR is defined as:

$$
\begin{equation}\tag{ERR}
ERR = \sum_{r=1}^n \frac{1}{r} R_{r} \prod_{i=1}^{r-1} \left( 1 - R_i \right),
\quad \text{where} \enspace
R_i = \frac{2^{l_i} - 1}{2^{l_m}}
\end{equation}
$$

where \( R_{i} \) models the probability that the user finds the document at \( i \)-th position relevant and \( l_m \) is the maximum possible label value.


[map-explained]: https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
[wiki-mrr]: https://en.wikipedia.org/wiki/Mean_reciprocal_rank



---------------------------------------------------------------------------------



<a name="supervised-ltr"></a>
# 3. Supervised Learning to Rank

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
## 3.1. RankNet objective

[Burges et al. (2005)][burges-ranknet] proposed an optimization objective for the Learning-to-Rank problem so that a model can be trained using gradient descent. Let's model the learned probability \( P_{ij} \) that \( i \)-th document should rank higher than \( j \)-th document as a sigmoid function:

$$
\begin{equation*}
P_{ij} = P\left({\bf U}_{i} \rhd {\bf U}_{j}\right) = \frac{1}{1 + e^{-\sigma (s_i - s_j)} }
\end{equation*}
$$

where the choice of the parameter \( \sigma \) determines the shape of the sigmoid. Let \( \widetilde{P}_ {ij} \) be the known probability that \( {\bf U}_ {i} \rhd {\bf U}_ {j} \) (it could be estimated by asking several judges to compare the pair and average out their answers). The **RankNet**'s cost function for a given query-document pair is defined as the [Cross Entropy][crossentropy]:

$$
\begin{equation}\label{eq:ranknet}
\mathcal{C}(s_i, s_j) = - \widetilde{P}_{ij} \log P_{ij} - (1 - \widetilde{P}_{ij}) \log (1 - P_{ij})
\end{equation}
$$

Obviously, this cost function is symmetric: swapping $i$ and $j$ and flipping the probabilities $P_{ij}$ and $\widetilde{P}_{ij}$ should not change the cost. At each minibatch descent step, the RankNet's cost is summarized from all document pairs in $\boldsymbol{\mathcal I}$.

RankNet opened a new direction in LTR research, and is the precursor to LambdaRank and LambdaMART. In 2015, the RankNet paper won the ICML Test of Time Best Paper Award, which honors “a paper from ICML ten years ago that has had substantial impact on the field of machine learning, including both research and practice.”


<a name="listnet"></a>
## 3.2. ListNet objective

The **ListNet** algorithm [(Cao et al. 2007)][listnet] is another important paper in the LTR literature. It is often referred to as the listwise approach to Learning To Rank, as the loss function is defined over the entire list of documents.

Given query $$\boldsymbol{q} \in \boldsymbol{\mathcal{Q}}$$ and a list of documents $$\boldsymbol{\mathcal{D}}^\boldsymbol{q} = \{d_1^\boldsymbol{q}, d_2^\boldsymbol{q}, \ldots, d_n^\boldsymbol{q}\}$$ retrieved for that query, together with their labeled relevance scores $$\boldsymbol{y} = \{y_1^\boldsymbol{q}, y_2^\boldsymbol{q}, \ldots, y_n^\boldsymbol{q}\}$$ (the score can be binary, or it can have relevance gradation). Let's say that the model $$f_{\theta}(\cdot)$$ outputs a score for each document, and let's denote the score for the $$i$$-th document as $$\boldsymbol{s}^{\boldsymbol{q}} = \{ f_{\theta}(d_i) \}_{i = 1\ldots n}$$. In the Plackett-Luce model, the probability of a document $$d_i$$ being ranked at the top of the list is defined as:

$$
\begin{equation*}
P_{\theta}\left( d_i^\boldsymbol{q} \vert \boldsymbol{\mathcal{D}}^\boldsymbol{q} \right) =
\frac{\exp[f_\theta(d_i^\boldsymbol{q})]}{\sum_{j=1}^n \exp[f_\theta(d_j^\boldsymbol{q})]}
\end{equation*}
$$

In this setting, the probability of sampling a specific ranked list of documents $$\boldsymbol{\mathcal{\pi}}^\boldsymbol{q} = \{d_{\pi_1}^\boldsymbol{q}, d_{\pi_2}^\boldsymbol{q}, \ldots, d_{\pi_n}^\boldsymbol{q}\}$$ (with replacement) is defined as the product of the probabilities of each document being ranked at the top of the remaining list:

$$
\begin{equation*}
P_{\theta}\left( \boldsymbol{\mathcal{\pi}}^\boldsymbol{q} \vert \boldsymbol{\mathcal{D}}^\boldsymbol{q} \right) =
\prod_{i=1}^n P_{\theta}\left( d_{\pi_i}^\boldsymbol{q} \vert \boldsymbol{\mathcal{D}}^\boldsymbol{q} \setminus \{d_{\pi_1}^\boldsymbol{q}, \ldots, d_{\pi_{i-1}}^\boldsymbol{q}\} \right)
\end{equation*}
$$

We then can use the all familiar Cross Entropy loss to define the ListNet's cost function:

$$
\begin{align*}
\mathcal{L}_{\text{ListNet}}(\boldsymbol{s}^\boldsymbol{q}, \boldsymbol{y}^\boldsymbol{q})
&=
- \sum_{i=1}^n P_{\boldsymbol{y}^\boldsymbol{q}}(d_i^\boldsymbol{q} \vert \boldsymbol{\mathcal{D}}^\boldsymbol{q}) \log P_{\theta}(d_i^\boldsymbol{q} \vert \boldsymbol{\mathcal{D}}^\boldsymbol{q})
\\ &=
- \sum_{i=1}^n
  \frac{\exp[y_i^\boldsymbol{q}]}{\sum_{j=1}^n \exp[y_j^\boldsymbol{q}]}
  \log \left[
    \frac{\exp[f_\theta(d_i^\boldsymbol{q})]}{\sum_{j=1}^n \exp[f_\theta(d_j^\boldsymbol{q})]}
  \right]\,.
\end{align*}
$$


<a name="lambdarank-and-lambdamart"></a>
## 3.3. LambdaRank and LambdaMART

The objective of [RankNet](#ranknet) is optimizing for (a smooth, convex approximation to) the number
of pairwise errors, which is fine if that is the desired cost. However, it does not produce desired gradients for minimizing position-sensitive objectives like [NDCG](#metrics-ndcg) or [ERR](#metrics-ERR), as we will illustrate in figure below.

Ideally, we would want to be able to optimize the position-sensitive objectives in a more direct way. However, ranking metrics such as [NDCG](#metrics-ndcg) and [ERR](#metrics-ERR) are not differentiable objectives since sorting is needed
to obtain ranks based scores. This makes the ranking metrics either discontinuous or flat everywhere and can not be directly used as learning objectives. The **LambdaRank** framework, developed by [Burges, Ragno, and Quoc (2006)][burges-lambdarank] during their time at Microsoft Research, solves this problem by modifying the gradients during training.

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
### 3.3.1. Train LambdaMART using LightGBM

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
### 3.3.2. Theoretical justification of LambdaRank

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

Finally, [Wang et al. (2019)][lambdaloss] developed a probabilistic framework, within which $\lambda$Rank optimizes for an upper bound of a well-defined cost function, which we will review closely next.


[poincare-lemma]: http://nlab-pages.s3.us-east-2.amazonaws.com/nlab/show/Poincar%C3%A9%20lemma


<a name="lambdaloss"></a>
## 3.4. LambdaLoss Framework

[Wang et al. (2019)][lambdaloss] proposed a general probabilistic framework called **LambdaLoss**, in which Information Retrieval metrics (like NDCG and ARR) can be optimized directly using gradient descend.

Let $$\boldsymbol{s}$$ denote the relevance scores for the documents produced by some function $$f$$, and $$\pi$$ denote a ranked list of documents. We use $$\left\{ P(\pi \vert \boldsymbol{s}) \colon \pi \in \Pi \right\}$$ do denote a distribution over all possible ranked list or permutations, determined by the scores $$\boldsymbol{s}$$. Within this framework, the ranked list $$\pi$$ is treated as a hidden variable, and the generic form of the loss function is defined based on the likelihood of observing relevance labels $$\boldsymbol{y}$$ given the scores $$\boldsymbol{s}$$ using a mixture model over all possible ranked lists $$\Pi$$:

$$
\begin{equation*}
P\left(\boldsymbol{y} \vert \boldsymbol{s}\right) = \sum_{\pi \in \Pi} P\left(\boldsymbol{y} \vert \boldsymbol{s}, \pi\right) P\left(\pi \vert \boldsymbol{s}\right)
\end{equation*}
$$

We can then define the loss function $$\mathcal{L} \left( \boldsymbol{y}, \boldsymbol{s} \right)$$ as the negative log-likelihood based on the maximum likelihood principle:

$$
\begin{equation} \label{eq:generic_lambdaloss}
\mathcal{L} \left( \boldsymbol{y}, \boldsymbol{s} \right) =
- \log P\left(\boldsymbol{y} \vert \boldsymbol{s}\right) =
- \log \sum_{\pi \in \Pi} P\left(\boldsymbol{y} \vert \boldsymbol{s}, \pi\right) P\left(\pi \vert \boldsymbol{s}\right)
\end{equation}
$$

This loss function can be optimized using an Expectation-Maximization (EM) process: at the E-step, we compute the distribution $$P(\pi \vert \boldsymbol{s})$$ given the scores $$\boldsymbol{s}$$ generated by the model $$f^t$$, and at the M-step, we update the model's weights by minimizing the negative likelihood.

Such generic framework might seem like an overkill &mdash; why do we need to take into consideration **all possible rankings** $$\Pi$$ if we can just sort the documents by their scores $$f(d)$$? The way I like to think about it is to draw an analogy with the classification task: the end metric that we care about is accuracy, but the loss function is defined as the negative log-likelihood of observing the true label $$y$$ given the scores $$f(x)$$. Why? Simply because accuracy is not differentiable, but the log likelihood formulation is.

By playing around with different formulations of the likelihood $$P\left(\boldsymbol{y} \vert \boldsymbol{s}, \pi\right)$$ and the ranked list distribution $$P\left(\pi \vert \boldsymbol{s}\right)$$, we can derive different loss functions that optimize for different metrics.

For example, if we treat the relevance labels $$\boldsymbol{y}$$ as a set of pairwise preference observations and formulate $$P\left(\boldsymbol{y} \vert \boldsymbol{s}, \pi\right)$$ with the basic Bradley-Terry model, where the probability that document $$i$$ is more relevant than $$j$$ (i.e. $$y_i > y_j$$) is only based on their scores $$s_i$$ and $$s_j$$ regardless of $$\pi$$, the loss function ($$\ref{eq:generic_lambdaloss}$$) turns into **RankNet** objective and can be optimized directly since it does not depend on $$P(\pi \vert \boldsymbol{s})$$:

$$
\begin{equation*} \label{eq:ranknet_lambdaloss}
\mathcal{L} \left( \boldsymbol{y}, \boldsymbol{s} \right) =
- \log P\left(y_i > y_j \vert s_i, s_j \right) =
- \sum_{y_i > y_j} \log \left( 1 + e^{-\sigma \left(s_i - s_j\right)}\right)
\end{equation*}
$$

The Bradley-Terry model can be turned into a rank-sensitive one by making $$P(y_i > y_j)$$ sensitive to the document's ranks in $$\pi$$. Assuming that $$\pi_i$$ and $$\pi_j$$ are ranks in $$\pi$$ of documents $$i$$ and $$j$$, let's define the relevance gain $$G(\cdot)$$ and position discount $$D(\cdot)$$ functions as follows:

$$
\begin{equation*}
G(i) = \frac{2^{y_i} - 1}{\max \text{DCG}}\,,
\quad
D(i) = \log(1 + \pi_i)\,.
\end{equation*}
$$

As you might have noticed, $$G(\cdot)$$ and $$D(\cdot)$$ are the same gain and discount functions used in NDCG objective. To construct the likelihood estimation for LambdaRank, we then modify $$P(y_i > y_j)$$ as follows:

$$
\begin{equation*} \label{eq:lambdarank_likelihood}
P\left( y_i > y_j \vert s_i, s_j, \pi_i, \pi_j \right) =
\left(
  \frac{1}{1 + e^{-\sigma (s_i - s_j)}}
\right)^{
  \left| G(i) - G(j) \right| \cdot \left| \frac{1}{D(\pi_i)} - \frac{1}{D(\pi_j)} \right|
}
\end{equation*}
$$

The only thing left is to define the distribution $$P(\pi \vert \boldsymbol{s})$$ over all possible ranked lists. Let's model the uncertainties of the score $$s_i = f(d_i)$$ of document $$d$$ using a normal distribution $$\mathcal{N}\left( s_i, \epsilon \right)$$ with Gaussian noise $$\epsilon$$ &mdash; this was proposed in an earlier paper by [Taylor et al. (2008)][softrank_2008]. Ranked lists $$\pi$$ will now form a distribution $$\mathcal{N}\left( \pi \vert \boldsymbol{s} \right)$$ with soft assignment over $$\Pi$$. In the LambdaLoss paper, the authors considered the hard assignment distribution $$\mathcal{H}\left(\pi \vert \boldsymbol{s}\right) = \lim_{\epsilon \to 0} \mathcal{N}\left( \pi \vert \boldsymbol{s} \right)$$ to reduce the computational cost. The loss function ($$\ref{eq:generic_lambdaloss}$$) thus becomes:

$$
\begin{equation*} \label{eq:lambdarank_lambdaloss} \tag{LambdaLoss}
\mathcal{L} \left( \boldsymbol{y}, \boldsymbol{s} \right) =
- \sum_{y_i > y_j} \log \sum_{\pi \in \Pi} P\left( y_i > y_j \vert s_i, s_j, \pi_i, \pi_j \right) \mathcal{H}\left(\pi \vert \boldsymbol{s}\right)
\end{equation*}
$$

[Wang et al. (2019)][lambdaloss] proved that **LambdaRank is an EM procedure that optimizes this loss!** The paper contains a lot more theoretical analysis. A more intriguing aspect of the LambdaLoss framework is that the likelihood $$P\left(\boldsymbol{y} \vert \boldsymbol{s}, \pi\right)$$ allows us to incorporate both scores and ranks into our objective definitions, thus opening the doors to a new family of metric-driven loss functions. They also propose a few new loss functions that offers tighter bounds when optimizing for NDCG.


---------------------------------------------------------------------------------


<a name="unbiased-ltr"></a>
# 4. Unbiased Learning to Rank

In the previous section, we have learned how to train a ranker on labeled data, where each document-query pair is annotated with a score (from 1 to 5) that shows how relevant that document is to the given query. This process is very expensive: to ensure the objectivity of labeled score, the human labeler would have to go through a strict checklist with multiple questions, then the document's relevance score will be calculated from the given answers. [Google's guidelines for search quality rating][google_sqe_guidelines] is a clear example of how complicated that process is (167 pages of guideline).

One might wonder **why we can't just use user's clicks as relevance labels?** Which is quite a natural idea: clicks are the main way users interacts with our web page, and the more relevant the document is to the given query, the more likely it is going to be clicked on. In this section, we will learn about approaches that allows us to learn directly from click data.

The structure of this section will closely follow the structure of the amazing lecture by [Oosterhuis et. al. (youtube, 2020)][ltr-lectures-harrie-youtube], which I personally used when I first started to learn about LTR.

Since we're learning from user clicks only, it is hereby natural to make following **assumptions:**
- By drawing the training signal directly from the user, it is more appropriate to talk about query instances $\boldsymbol{\mathcal{q}}$ that include contextual information about the user, instead of just a query string, since each user acts upon their own relevance judgement subject to their specific context and intention.
- Relevance label to each document is binary ($1$ if relevant to given query and specific user, $0$ if not).


[google_sqe_guidelines]: https://static.googleusercontent.com/media/guidelines.raterhub.com/en//searchqualityevaluatorguidelines.pdf
[ltr-lectures-harrie-youtube]: https://www.youtube.com/watch?v=BEEfMrn9T9c


Some frequently used notations for this section:

<table class="notations-table">
<tr><th>Notation</th><th class="notations-desc">Description</th></tr>
<tr>
<td>\( \boldsymbol{\mathcal{q}} \)</td>
<td>User query, generally assumed to contain some user information and context as well.</td>
</tr>
<tr>
<td>\( d \in \boldsymbol{\mathcal{D}} \)</td>
<td>Documents retrieved for the query \( \boldsymbol{\mathcal{q}}\, \).</td>
</tr>
<tr>
<td>\( \boldsymbol{\pi}_\theta^{\boldsymbol{\mathcal{q}}} \)</td>
<td>The ranking of documents for query \( \boldsymbol{\mathcal{q}} \), generated by model \( f \) with parameters \( \theta \).</td>
</tr>
<tr>
<td>\( \boldsymbol{y}^{\boldsymbol{\mathcal{q}}} = \{ y_d^{\boldsymbol{\mathcal{q}}} \} \)</td>
<td>The true relevance labels for documents \( d \in \boldsymbol{\mathcal{D}} \) retrieved for query \( \boldsymbol{\mathcal{q}}\, \). </td>
</tr>
<tr>
<td>\( \boldsymbol{o}^{\boldsymbol{\mathcal{q}}} = \{ o_d^{\boldsymbol{\mathcal{q}}} \} \)</td>
<td>The indicators whether relevance labels for documents \( d \in \boldsymbol{\mathcal{D}} \) were observed.</td>
</tr>
<tr>
<td>\( \boldsymbol{c}^{\boldsymbol{\mathcal{q}}} = \{ c_d^{\boldsymbol{\mathcal{q}}} \} \)</td>
<td>The click indicators. \( c_d^{\boldsymbol{\mathcal{q}}} = 1\) if the user has clicked on document \(d\).</td>
</tr>
<tr>
<td>\( \Delta \left( \boldsymbol{\mathcal{y}}^{\boldsymbol{\mathcal{q}}}, \boldsymbol{\pi}_\theta^{\boldsymbol{\mathcal{q}}} \right) \)</td>
<td>Any linearly decomposable Information Retrieval metric (NDCG, MRR, MAP, etc.)</td>
</tr>
<tr>
<td>\( \mu (r) \)</td>
<td>Rank weighting function ("how important are the document at rank \(r\)"), part of \( \Delta \).</td>
</tr>
</table>

Please be aware that the notation used in this section are slightly different from the notation used in the original publications. This is because different authors prefer different notations styles and the bias models are slightly different from each other. This set of notations is my attempt to unify and simplify the notations used in different papers while preserving as much of the semantics as possible, at the cost of losing some of the flexibility and granularity of the original notations.

<a name="click-biases"></a>
## 4.1. Click Signal Biases

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



## 4.2. Counterfactual Learning to Rank

Counterfactual Learning to Rank is a family of LTR methods that learns from historical interactions between users and returned results (e.g. in form of clicks, content engagement, or other discrete or non-discrete factors). At the core of these methods lies the following idea:

> **Counterfactual Evaluation:** evaluate a new ranking function $f_\theta$ using historical interaction data collected from a previously deployed ranking function $f_\text{deploy}$.

In Unbiased Learning-to-Rank literature, the ranking function $f_\text{deploy}$ is often referred to as **"policy"** (or **"behavior policy"**), and the new ranking function $f_\theta$ (parametrized by $$\theta$$) is referred to as **"evaluation policy"**. The goal of counterfactual evaluation is to estimate the performance of the new ranking function $f_\theta$ without deploying it in production, which is often expensive and risky. The evaluation policy $f_\theta$ is trained on a dataset of historical interactions between users and the deployed ranking function $f_\text{deploy}$, and the performance of the evaluation policy $f_\theta$ is estimated by comparing the ranking of the evaluation policy $f_\theta$ to the ranking of the deployed ranking function $f_\text{deploy}$.


### 4.2.1. Full vs Partial Information LTR

**Full Information Learning to Rank.** Before talking about approaches for Learning-to-Rank from biased implicit feedback (e.g. user clicks), let's review what we know so far about LTR from a curated & notoriously annotated dataset, where true relevance labels are known for each query-document pairs (i.e. we have **full information** about the data we're evaluating the model $$f_\theta$$ on). Given a sample $$\boldsymbol{\mathcal{Q}}$$ of queries $$\boldsymbol{\mathcal{q}} \sim P(\boldsymbol{\mathcal{q}})$$ for which we assume the user-specific binary relevances $$\boldsymbol{\mathcal{y}}^{\boldsymbol{\mathcal{q}}} = \{y^{\boldsymbol{\mathcal{q}}}_d\}_{d \in \boldsymbol{\mathcal{D}}}$$ of all documents $$d \in \boldsymbol{\mathcal{D}}$$ are known (assuming $\boldsymbol{\mathcal{q}}$ already captures user context), we can define overall empirical risk of a ranking system $$f_\theta$$ as follows:

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


**Partial Information Learning to Rank.** In this setup, we don't know the true relevance $\boldsymbol{\mathcal{y}}_d$ of each document and have to rely on user clicks, so we need to understand how the click biases plays out in practice. Let's take a look at toy example of a typical user session (also called "impression" in search and recommendation sysmtes) illustrated below:

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


### 4.2.2. What's wrong with Naive Estimator?

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


### 4.2.3. Inverse Propensity Weighting

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


<a name="bias-estimation-randomization">
### 4.2.4. Estimating Position Bias by Randomization

Now that we have a way to estimate unbiased relevances through Inverse Propensity Weighting, the next step is to address the challenge of accurately estimating propensities $$P\left(o_d = 1 \vert \boldsymbol{\mathcal{\pi}}\right)$$. Here, we will consider only the position bias, which is the most common bias in search and recommendation systems. According to the Position-based Model, or **PBM** ([Craswell & Taylor, 2008][experimental_comparison_of_click_models]):

$$
P \left(c_d = 1 \vert \boldsymbol{\mathcal{\pi}} \right) =
P \left(o_d = 1 \vert \boldsymbol{\mathcal{\pi}} \right) \cdot P \left(y_d = 1 \vert \boldsymbol{\mathcal{\pi}} \right)
$$

In this PBM model, the following assumpttions are implicitly made:
- The user clicks on an item only after examining (observing) it and perceiving it as relevant.
- The act of examining an item only depends on its position in the presented ranking. In other words, observing an item is independent of its relevance or other items in the ranking.

The PBM model implies that the only effect on clicks when showing documents at different positions is due to the position bias. This is a strong assumption, but it is often used in practice because it is simple and effective.

The simplest way to estimate $$P\left(o_d = 1 \vert \boldsymbol{\mathcal{\pi}}\right)$$ is to conduct **randomization** experiments (first proposed by [Wang et al. 2016][wang_2016]). The idea is to randomly shuffle the top-K results and see how the click-through rate (CTR) changes at each rank.

{% capture imblock_randomization_propensity_estimate %}
    {{ site.url }}/articles/images/2021-08-15-learning-to-rank/randomization_propensity_estimate.png
{% endcapture %}
{% capture imcaption_randomization_propensity_estimate %}
  The expected Click Through Rate (CTR) after randomization is proportional to the **position bias**. For the same query but during different impressions (e.g. when different users searched for the same query), we shuffle top-K results randomly to see how CTR changes at each rank. *(<a href="https://sites.google.com/view/sigir-2023-tutorial-ultr">Image: Gupta et al.</a>)*
{% endcapture %}
{% include gallery images=imblock_randomization_propensity_estimate cols=1 caption=imcaption_randomization_propensity_estimate %}

The expected CTR after randomization is proportional to the **position bias**. For the same query but during different impressions (e.g. when different users searched for the same query), we shuffle top-K results randomly to see how CTR changes at each rank. The position bias can be estimated by the ratio of the expected CTR after randomization to the CTR before randomization.

The major drawback is that it requires randomizing the top search results ordering in online setting, which negatively affects user experience. To mitigate this issue, [Joachims et al. (2017)][joachims_2016] proposed a softer approach that swaps only one item at a time to a fixed pivot rank $$k$$, which is shown to be a good approximation of the full randomization approach. As illustrated below, the impact of randomization on user experience is significantly lower.

{% capture imblock_randomization_top_k %}
    {{ site.url }}/articles/images/2021-08-15-learning-to-rank/randomization_top_k.png
{% endcapture %}
{% capture imcaption_randomization_top_k %}
  The impact of randomization on user experience can be lowered by swapping only one item at a time to a fixed pivot rank $$k$$. *(<a href="https://sites.google.com/view/sigir-2023-tutorial-ultr">Image: Gupta et al.</a>)*
{% endcapture %}
{% include gallery images=imblock_randomization_top_k cols=1 caption=imcaption_randomization_top_k %}

The CTR ratio between the original ranking and the swapped ranking of an item is proportional to the **position bias ratio** between the two ranks:

$$
\begin{equation*}
\frac{P(C = 1 | \text{no swap})}{P(C = 1 | \text{swap } d \text{ to rank } k)} = \frac{P(C = 1 | d, q, \text{rank}(d | y))}{P(C = 1 | d, q, k)} \propto \frac{P(O = 1 | \text{rank}(d | y))}{P(O = 1 | k)}
\end{equation*}
$$

where $$O$$ is the event that the user examines the item, and $$C$$ is the event that the user clicks on the item. $$d$$ is the document to be swapped, $$q$$ is the query, and $$y$$ is the relevance of the document.

Despite their simplicity, online search page randomization approaches are considered the gold standard of position bias estimation. However, they are not suitable for training data collection because they require randomizing the top search results ordering in online setting, which may have a negative impact on user experience.

[Agarwal et al. (2019)][agarwal_2019] made an observation that if you have enough historical data, you can estimate the position bias without randomizing the search results ordering in online setting. The idea is to use the log data of previously deployed rankers as an implicit source of "randomized" data. Some observations:
- Previous rankers probably ranked the documents differently from each other.
- Different models in previous A/B tests probably made different ranking decisions.

With enough variety in the historical data, we can use it to estimate the position bias. This is the idea behind **Intervention Harvesting** proposed by [Agarwal et al. (2019)][agarwal_2019], illustrated below:

{% capture imblock_intervention_harvesting %}
    {{ site.url }}/articles/images/2021-08-15-learning-to-rank/intervention_harvesting.png
{% endcapture %}
{% capture imcaption_intervention_harvesting %}
  Estimating position bias using **Intervention Harvesting**. We collect the log data of previously deployed rankers, weight the clicks by their exposure at a given rank (since some documents appear in certain ranks more often than others), then we can infer the position bias. *(<a href="https://sites.google.com/view/sigir-2023-tutorial-ultr">Image: Gupta et al.</a>)*
{% endcapture %}
{% include gallery images=imblock_intervention_harvesting cols=1 caption=imcaption_intervention_harvesting %}

The downside of Intervention Harvesting is that it requires a lot of historical data, which is not always available. Also, it requires minimal shift in the query distribution between rankers and over time, and no shift of the document relevance over time. Moreover, if there is not enough variety in the historical data (i.e. between rankers), the estimated position bias may be inaccurate.

{% comment %}
TODO: write more detailed explanation of Intervention Harvesting, as this is a very important paper published by Google and tested on both ArXiv and Google Drive search data.
{% endcomment %}


<a name="dual-learning-algorithm">
### 4.2.5. Dual Learning Algorithm (DLA)

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

In Learning to Rank problem we need to train a ranking model $$f_\theta$$ that minimizes the global ranking loss function $$\mathcal{L}_\text{R}$$ accross all the queries $$\boldsymbol{\mathcal{Q}}$$. Similarly, in Propensity Estimation problem we need to train a propensity model $$g_\psi$$ that minimizes propensity loss $$\mathcal{L}_\text{P}$$. The empirical loss functions are defined as:

$$
\begin{align*}
\hat{\mathcal{L}}_\text{R} \left( f_\theta \right) & =
\mathbb{E}_{\boldsymbol{q} \in \boldsymbol{\mathcal{Q}}} \left[
\mathcal{l}_{\text{R}} \left( f_\theta, \boldsymbol{q} \right)
\right]
\\
\hat{\mathcal{L}}_\text{P} \left( g_\psi \right) & =
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
      <div>Compute empirical loss values $\hat{\mathcal{L}}_{\text{R}}$ and $\hat{\mathcal{L}}_{\text{P}}$ on the batch $\boldsymbol{\mathcal{Q}}'\,$;</div>
      <div>Update $\theta$ and $\psi$ using gradients $\nabla_\theta \hat{\mathcal{L}}_{\text{R}}$ and $\nabla_\psi \hat{\mathcal{L}}_{\text{P}}\,$;</div>
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
    \boldsymbol{\mathcal{o}}\, \vert
    \boldsymbol{\pi}_\theta
  \right)
  =
  \sum_{d \colon y_d = 1, o_d = 1} {
    \frac{
      \mu \big[
        \boldsymbol{\pi}_\theta(d)
      \big]
    }{
      P\left(y_d = 1 \vert \boldsymbol{\pi}_\theta \right)
    }
  }
\end{equation*}
$$

Notice that it looks exactly like ($$\ref{eq:ipw}$$), just with the observance indicator $$\boldsymbol{o}$$ and relevance indicator $$\boldsymbol{y}$$ swapped. We can show that is is an unbiased estimator of the local (per-query) loss $$l_{\text{P}}$$ by following the same steps as in $$\eqref{eq:ipw_unbiased}$$:

$$
\begin{equation}
\mathbb{E}_{\boldsymbol{y}^{\boldsymbol{q}}}
\big[
  \Delta_{\text{IRW}} \left(
    \boldsymbol{\mathcal{o}}\, \vert
    \boldsymbol{\pi}_\theta
  \right)
\big] =
\Delta \left( \boldsymbol{\mathcal{\pi}}_\phi, \boldsymbol{o}\right) =
l_{\text{P}} \left( g_\psi, \boldsymbol{q} \right)
\,.
\label{eq:irw_unbiased}
\end{equation}
$$

> **Exercise for the reader:** Prove that the IRW estimator is unbiased by following the same steps as in $$\eqref{eq:ipw_unbiased}$$.

In this paper, [Ai et al. (2018)][ai_2018] also provided a rigorous proof that the Dual Learning Algorithm (DLA) will converge, under assumption that only position bias is considered and the loss functions are chosen as cross-entropy of softmax probabilities.



<a name="regression-em"></a>
### 4.2.6. Regression EM

[Wang et al. (2018)][regressionem] proposed a different approach called **Regression EM** to estimate the position bias without having to randomize the search results ordering in online setting. The idea is to use the Expectation-Maximization (EM) algorithm to estimate the position bias. The key idea is to treat the position bias as a latent variable and estimate it using the EM algorithm.

Why do we need this? Is it more effective? Does it converge faster than DLA? No, but turns out that this algorithm can estimate not only position bias but some other types of biases as well, which will be handy in the next section where we will discuss trust bias and other types of biases.

Let's define some shorthand notations for the examination $$P(e = 1)$$ and relevance $$P(y = 1)$$ probabilities for document $$d$$ at position $$k = \mathrm{rank}(d \vert \boldsymbol{\mathcal{\pi}})$$:

$$
\begin{equation*}
\alpha_k = P\left( e = 1 \vert k \right) \,, \quad
\beta_{q,d} = P\left( y = 1 \vert \boldsymbol{q}, d \right)
\end{equation*}
$$

Note that, in the PBM model, the examination probability $$\alpha_k$$ only depends on the rank $$k$$, and the relevance probability $$\beta_{\boldsymbol{q},d}$$ only depends on the query $$q$$ and the document $$d$$. We are using the terms *examination* and *observation* interchangeably here. This is because in the PBM model, these two events are the same. Given a regular click log $$\boldsymbol{\mathcal{L}} = \{ c, \boldsymbol{\mathcal{q}}, d, k \}$$, the likelihood of getting this log data is:

$$
\begin{equation*}
\log P\left( \boldsymbol{\mathcal{L}} \right) = \sum_{c, \boldsymbol{q}, d, k} {
  c \cdot \log \left( \alpha_k \cdot \beta_{\boldsymbol{q},d} \right)
  + (1 - c) \cdot \log \left( 1 - \alpha_k \cdot \beta_{\boldsymbol{q},d} \right)
}
\end{equation*}
$$

The **Regression EM** then finds the parameters $$\alpha_k$$ and $$\beta_{\boldsymbol{q},d}$$ that maximize the likelihood of the click log data. Similar to the standard EM process, the *Regression EM* process consists of two steps: the *Expectation* (E) step and the *Maximization* (M) step. In the *Expectation* step, we estimate the probabilities of examination and relevance for each document at each rank using the latent variables $$\alpha_k$$ and $$\beta_{\boldsymbol{q},d}$$. In the *Maximization* step, we update these latent variables using a regression model.

Although the PBM model is not widely used in practice anymore (other than as a baseline to compare against), for educational purposes let's take a closer look at the Regression-EM process designed for PBM model, as originally proposed by [Wang et al. (2018)][regressionem]. During the *Expectation* (E) step of iteration $$t + 1$$, we estimate the distribution of hidden events $$y$$ (relevance) and $$e$$ (examination) given the current model parameters $$\alpha_k^{(t)}$$ and $$\beta_{q,d}^{(t)}$$ and the observed data log $$\boldsymbol{\mathcal{L}}$$ as follows:

$$
\begin{align*}
P(e = 1, y = 1 \mid c = 1, \boldsymbol{q}, d, k) &= 1 \\
P(e = 1, y = 0 \mid c = 0, \boldsymbol{q}, d, k) &= \frac{\alpha_{k}^{(t)} \left( 1 - \beta_{\boldsymbol{q},d}^{(t)} \right)}{1 - \alpha_{k}^{(t)} \beta_{\boldsymbol{q},d}^{(t)}} \\
P(e = 0, y = 1 \mid c = 0, \boldsymbol{q}, d, k) &= \frac{\left( 1 - \alpha_{k}^{(t)} \right) \beta_{\boldsymbol{q},d}^{(t)}}{1 - \alpha_{k}^{(t)} \beta_{\boldsymbol{q},d}^{(t)}} \\
P(e = 0, y = 0 \mid c = 0, \boldsymbol{q}, d, k) &= \frac{\left( 1 - \alpha_{k}^{(t)} \right) \left( 1 - \beta_{\boldsymbol{q},d}^{(t)} \right)}{1 - \alpha_{k}^{(t)} \beta_{\boldsymbol{q},d}^{(t)}}
\end{align*}
$$

With these relations, we can calculate marginals $$P(e = 1 \mid c, \boldsymbol{q}, d, k)$$ and $$P(y = 1 \mid c, \boldsymbol{q}, d, k)$$ from the log data $$\boldsymbol{\mathcal{L}}$$ and the current model parameters $$\alpha_k^{(t)}$$ and $$\beta_{\boldsymbol{q},d}^{(t)}$$. This can be seen as complete data where hidden variables are estimated. Then, in the *Maximization* (M) step, we update the model parameters using the estimated marginals:

$$
\begin{equation*}
\alpha_{k}^{(t+1)} = \frac{\sum_{c,\boldsymbol{q},d,k'} \mathbb{I}_{k' = k} \cdot \left( c + (1 - c) P(e = 1 \mid c, \boldsymbol{q}, d, k) \right)}{\sum_{c,\boldsymbol{q},d,k'} \mathbb{I}_{k' = k}}
\end{equation*}
$$

Although it is possible to express $$\beta_{\boldsymbol{q},d}^{(t+1)}$$ in a similar way, it is more complex because it depends on the query $$\boldsymbol{q}$$ and the document $$d$$. We simply might not have enough samples to estimate it accurately. Therefore, [Wang et al. (2018)][regressionem] proposed to learn it from features &mdash; this way, we can generalize the model to unseen queries and documents, as well as discover the underlying patterns in the data. By sampling $$\hat{y} \sim P(y = 1 \mid c, \boldsymbol{q}, d, k)$$ and having the features $$\boldsymbol{x}_{\boldsymbol{q},d}$$ for the query-document pair, we can treat it as a classification problem and optimize the log likelihood of the observed data:

$$
\begin{equation*}
\sum_{\boldsymbol{x}, \hat{y}} \hat{y} \log \left( f_\theta \left( \boldsymbol{x} \right) \right) + \left( 1 - \hat{y} \right) \log \left( 1 - f_\theta \left( \boldsymbol{x} \right) \right)
\end{equation*}
$$

After learning the model $$f_\theta$$, we can use it to estimate the relevance probability $$\beta_{\boldsymbol{q},d}^{(t+1)} = f_\theta \left( \boldsymbol{x}_{\boldsymbol{q},d} \right)$$ for any query-document pair. The (E) and (M) steps are repeated until convergence.



<a name="online-ltr"></a>
## 4.3. Online Learning to Rank

Similar to Counterfactual Learning to Rank (CLTR), Online Learning to Rank (OLTR) also uses implicit user feedback (e.g. clicks) to train a ranking model. However, unlike CLTR which uses historical data to estimate the click biases and train the ranking model, OLTR uses the click feedback in real-time to update the ranking model, by directly interacting with the users. The main advantage is flexibility &mdash; the online setting allows the learning algorithm to control data acquisition and handle biases and noise through online interventions, by choosing which documents to present to the user and observing the user's feedback. The main disadvantage is that it requires a lot of traffic to be effective, and the infrastructure to support online learning is, in general, more complex than offline learning.


<a name="interleaving"></a>
### 4.3.1. Comparing rankers by Interleaving

Before we dive into the details of Online Learning to Rank, it is important to first understand how to compare two rankers using **interleaving**. Interleaving is a technique to compare two or more rankers by presenting the results of the rankers in an interleaved list (i.e. a mix of results from the rankers) and observing the user's feedback.

Compared to traditional A/B testing, interleaving is more efficient because it requires fewer user interactions to compare rankers. However, it also comes with a drawback: it is more difficult to estimate the statistical significance of the results and to control for biases.

{% capture imblock_interleaving_intro %}
    {{ site.url }}/articles/images/2021-08-15-learning-to-rank/interleaving_intro.png
{% endcapture %}
{% capture imcaption_interleaving_intro %}
  The difference between A/B testing and interleaving. In A/B testing, the users are split into two groups and each group is presented with results from one ranker. In interleaving, the user is presented with an interleaved list of results from both rankers.
{% endcapture %}
{% include gallery images=imblock_interleaving_intro cols=1 caption=imcaption_interleaving_intro %}

**Team-Draft Interleaving** ([Radlinski et al. 2008][radlinski_2008]) is one of the simplest and the most widely used form of interleaving. Let's say we have two rankers (considered as "teams"), each produced a list of results $$\boldsymbol{\mathcal{\pi}}_A$$ and $$\boldsymbol{\mathcal{\pi}}_B$$, and we want to compare them. I personally find it easiest to understand how Team-Draft Interleaving works through a simple Python implementation, as shown below:

```python
def team_draft_interleaving(pi_A, pi_B):
  """Inputs:
    pi_A, pi_B: list of documents ordered by rankers A and B
  Output:
    interleaved: documents shown to the user, interleaved from A and B
    team_A, team_B: which documents can be attributed to rankers A and B
  """
  interleaved, team_A, team_B = [], [], []
  while A or B:  # Repeat while there are still documents in A or B
    if (len(team_A) < len(team_B)
        or (len(team_A) == len(team_B) and random.choice([True, False]))):
      doc = A.pop(0)
      if doc not in interleaved:
        interleaved.append(doc), team_A.append(doc)
    else:
      doc = B.pop(0)
      if doc not in interleaved:
        interleaved.append(doc), team_B.append(doc)
  return interleaved, team_A, team_B
```

The `interleaved` list is then shown to the user, and the user's feedback is used to determine which ranker is better. The lists `team_A` and `team_B` are then used for attribution, i.e. which document's interactions (like clicks) can be attributed to which ranker. Note that, according to this algorithm, if there's a similar document in both lists, it is shown to the user only once and will be attributed to the ranker that ranked it higher.

Here are some cool blog posts and write-ups of how variations of **interleaving** is being used in large-scale recommendation systems:
* **Netflix**: ["Innovating Faster on Personalization Algorithms at Netflix Using Interleaving." (2017)][netflix_interleaving] Netflix uses a variation of *Team-Draft Interleaving* in a 2-stage process to compare different recommendation algorithms. The best algorithms are then compared in an A/B test, and the winner is deployed to production.
* **Airbnb**: ["Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving." (2022)][airbnb_interleaving] Airbnb also uses a variation of *Team-Draft Interleaving* in their 2-stage testing process (first interleaving, then A/B testing). However, they modified the attribution logic to account for their unique search scenario in which a user can issue multiple search requests before booking.
* **Thumbtack**: ["Accelerating Ranking Experimentation at Thumbtack with Interleaving." (2023)][thumbtack_interleaving] Interestingly, they are using *Team-Draft Interleaving* as well.
* **Etsy**: ["Faster ML Experimentation at Etsy with Interleaving."][etsy_interleaving] This article is a gem! They described engineering challenges and how they tested their interleaving system to ensure that it is working as expected.

The most fascinating property of interleaving is that, if done properly, it can result in a significant reduction in the number of users needed to compare two rankers.

{% capture imblock_interleaving_100x %}
    {{ site.url }}/articles/images/2021-08-15-learning-to-rank/interleaving_100x.png
{% endcapture %}
{% capture imcaption_interleaving_100x %}
  Sensitivity of interleaving vs traditional A/B metrics for two rankers of known relative quality. Bootstrap subsampling was used to measure the sensitivity of interleaving compared to traditional engagement metrics. Both <b>Netflix (left chart)</b> and <b>Thumbtack (right chart)</b> found that interleaving can require >100x fewer subscribers to correctly determine ranker preference even compared to the most sensitive A/B metric.
{% endcapture %}
{% include gallery images=imblock_interleaving_100x cols=1 caption=imcaption_interleaving_100x %}

[netflix_interleaving]: https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55
[airbnb_interleaving]: https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e
[thumbtack_interleaving]: https://medium.com/thumbtack-engineering/accelerating-ranking-experimentation-at-thumbtack-with-interleaving-20cbe7837edf
[etsy_interleaving]: https://www.etsy.com/codeascraft/faster-ml-experimentation-at-etsy-with-interleaving



<a name="dbgd">
#### 4.3.2. Dueling Bandit Gradient Descent (DBGD)

Having a way to efficiently compare two rankers using interleaving, we can formulate the Online Learning to Rank problem as a **Dueling Bandit** problem and perform weights updates to our ranker. Let's say we have a ranker $$f_\theta$$. We can slightly update its weights and create a new ranker $$f_{\theta + \epsilon}$$. We can then use interleaving to compare the two rankers and determine which one is better, and update the weights $$\theta$$ accordingly afterwards. This is the core idea behind **Dueling Bandit Gradient Descent** (DBGD) proposed by [Yue & Joachims (2009)][yue_dbgd]. Formally, the algorithm looks like this:

<div class="algorithm">
  <div class="algo-name">Dueling Bandit Gradient Descent</div>
  <div class="algo-input">
    Ranker $f_{\theta_0}$ with parameters (weights) $\theta_0 \in \mathbb{R}^d$, learning rate $\eta$, exploration magnitude $\delta$.
  </div>
  <div class="algo-output">
    Updated ranking model $f_{\theta_T}$.
  </div>
  <div class="algo-body">
    <!-- For block -->
    <div><b>for</b> query $q_t \, (t = 1\ldots T)$ <b>do</b></div>
    <div class="algo-indent-block">
      <div>Randomly sample a unit vector $u_t \in \mathbb{R}^d$ from the unit sphere uniformly.</div>
      <div>Assign $\theta_t' \leftarrow \theta_t + \delta u_t$</div>
      <div>Compare rankers $f_{\theta_t}$ and $f_{\theta_t'}$ using interleaving</div>
      <div><b>if</b> $f_{\theta_t'}$ wins <b>then</b></div>
      <div class="algo-indent-block">
        <div>Update weights $\theta_{t+1} \leftarrow \theta_t + \eta u_t$</div>
      </div>
      <div><b>else</b></div>
      <div class="algo-indent-block">
        <div>$\theta_{t+1} \leftarrow \theta_t\,$ (leave the weights the same if there's no improvements)</div>
      </div>
      <div><b>end if</b></div>
    </div>
    <div><b>end for</b></div>
  </div> <!-- algo-body -->
  <div class="algo-end"></div>
</div>

where the initial weights $\theta_0$ can be trained using any offline learning to rank algorithm (either supervised or counterfactual). The exploration magnitude $\delta$ controls the magnitude of the perturbation, and the learning rate $\eta$ controls the step size of the weight updates.

{% capture imblock_dbgd %}
    {{ site.url }}/articles/images/2021-08-15-learning-to-rank/dbgd.png
{% endcapture %}
{% capture imcaption_dbgd %}
  Illustration of one iteration of the Dueling Bandits Gradient Descent (DBGD) algorithm. *(Image source: [Oosterhuis et al.](https://ilps.github.io/webconf2020-tutorial-unbiased-ltr/WWW2020handout.pdf))*
{% endcapture %}
{% include gallery images=imblock_dbgd cols=1 caption=imcaption_dbgd %}

To speed up the chance of finding the improving weight updates, [Schuth et al. (2016)][multileave_dbgd] proposed a **Multi-leave Dueling Bandit Gradient Descent** algorithm, which samples multiple directions per iteration and uses multiple interleaving lists to compare the rankers. This allows us to explore more diverse perturbations of the weights at every step and find the improving direction faster. Other works have proposed to leverage historical data to reject unpromising directions and focus on the most promising ones, such as [Hofmann et al. (2013)][hofmann_2013], [Zhao and King (2016)][zhao_king_2016], and [Wang et al. (2018)][wang_dbgd_2018].

For this family of algorithms to work, the utility space with respect to $\theta$ should be "smooth" enough &mdash; that is, small perturbations in the weights should result in small changes in the ranking quality and user experience. More formally, if we measure the **regret** of using $f_{\theta'}$ over $f_\theta$ as the fraction of users who would prefer the former over the latter over possible queries:

$$
\begin{equation*}
\epsilon(\theta, \theta') \propto \mathbb{E}_{\boldsymbol{q}} \left[ f_{\theta}(\boldsymbol{q}) \succ f_{\theta'}(\boldsymbol{q}) \right]
\end{equation*}
$$

Then such regret $\epsilon(\cdot, \cdot)$ should be Lipschitz continuous with respect to the weights $\theta$, i.e. $\exists L > 0$ such that $\forall \theta, \theta'\,\colon \left\| \epsilon(\theta, \theta') \right\| \leq L \left\Vert \theta - \theta' \right\Vert$.

Readers with a good mathematical background will call me out on this, because this is not exactly what Lipschitz continuity means. In the original paper ([Yue & Joachims, 2009][yue_dbgd]), the authors used a more elaborated formulation with intermediate value and link functions to apply the standard definition of Lipschitz continuity. The formulation in this blog post is essentially the same as the rigorous one &mdash; it just simplifies the notation and makes it more intuitive.

The authors also proved that, if we additionally assume that there is a unique optimal weights $$\theta^*$$, then the algorithm will achieve sublinear global regret in $$T$$. In practice, however, none of these assumptions holds for more complex ranking models (like neural networks, gradient boosted trees, etc.), as noted in the paper by [Oosterhuis and Rijke (2019)][oosterhuis_2019]:

- Optimization space of neural networks is highly non-convex, and there can be multiple blobs local minima. There has been plenty of works analyzing this in Machine Learning Literature, such as [(Haeffele & Vidal, 2017)][global_optimality_nn]. A simple example is this: just multiple the last linear projection by $$\alpha$$. The results will be the same, but the weights are different.
- The Lipschitz continuity assumption can't be true. Take a linear model, for example. Its regret is scale-invariant, i.e. $$\epsilon(\alpha \theta_1, \alpha \theta_2) = \epsilon(\theta_1, \theta_2)$$ for any $$\alpha \neq 0$$ because the ranking of the documents are going to be the same. But the Lipschitz constant is not scale-invariant, so we will have a contradiction statement that $$\epsilon(\theta_1, \theta_2) \leq L \Vert \alpha \theta_1 - \alpha \theta_2 \Vert = \alpha L \Vert \theta_1 - \theta_2 \Vert$$ for any $$\alpha \neq 0$$.




<a href="#pdgd"></a>
#### 4.3.3. Pairwise Differentiable Gradient Descent (PDGD)

Unlike the DBDG family of algorithms, the **Pairwise Differentiable Gradient Descent** (PDGD) algorithm proposed by [Oosterhuis et al. (2019)][oosterhuis_2019] does not require the Lipschitz continuity assumption, nor relying on any online evaluation methods. Similar to ListNet, let's look at the ranking function $$f_\theta(\cdot)$$ as a probability distribution over documents $$d \in \boldsymbol{\mathcal{D}}$$ by applying a Plackett-Luce (PL) model (essentially a softmax function) over the scores:

$$
\begin{equation*}
P(d \vert \boldsymbol{\mathcal{D}}, \theta) = \frac{\exp(f_\theta(d))}{\sum_{d' \in \boldsymbol{\mathcal{D}}} \exp(f_\theta(d'))}
\end{equation*}
$$

A ranking $$\boldsymbol{\mathcal{\pi}} = \{ \boldsymbol{\mathcal{\pi}}_1, \ldots, \boldsymbol{\mathcal{\pi}}_k \}$$, where $$\boldsymbol{\mathcal{\pi}}_i$$ is the $$i$$-th document in the ranking, is then presented to the user by sampling from this distribution $$k$$ times, where after each sample the document is removed from the set of available documents before the next sample to prevent duplicates. Formally, the probability of ranking $$\boldsymbol{\mathcal{\pi}}$$ is then the product of the probabilities of the documents in the ranking:

$$
\begin{equation*}
P\left(\boldsymbol{\mathcal{\pi}} \vert \boldsymbol{\mathcal{D}}, \theta\right) = \prod_{i=1}^{k} P\left(\boldsymbol{\mathcal{\pi}}_i \vert \boldsymbol{\mathcal{D}} \setminus \{ \boldsymbol{\mathcal{\pi}}_1, \ldots, \boldsymbol{\mathcal{\pi}}_{i-1} \}, \theta\right)\,.
\end{equation*}
$$

After the ranking is presented to the user, the user's feedback (in the form of clicks) is then used to infer preferences between the documents in the ranking. However, we can't know which documents were examined (considered) by the user during the session, so a reasonable assumption is that the user has examined all documents before the last clicked one, and one document after that, as illustrated in the figure below (documents considered by PDGD are colored).

Let's denote the user preference between documents $$d_k$$ and $$d_l$$ inferred from clicks as $$d_k >_c d_l$$ (i.e. $$d_k$$ is preferred over $$d_l$$). The objective on each step of PDGD is to increase the probability of the preferred document being ranked higher than the non-preferred one, which looks like follows in the PL model:

$$
\begin{align*}
P\left( d_k \succ d_l \vert \boldsymbol{\mathcal{D}}, \theta\right) &=
\frac{P(d_k \vert \boldsymbol{\mathcal{D}})}{P\left(d_k \vert \boldsymbol{\mathcal{D}}, \theta\right) + P\left(d_l \vert \boldsymbol{\mathcal{D}}, \theta\right)} \\ &=
\frac{\exp(f_\theta(d_k))}{\exp(f_\theta(d_k)) + \exp(f_\theta(d_l))}\,.
\end{align*}
$$

A naive approach would be to directly maximize the above probability for all observed preferences $$d_k >_c d_l$$ in the training data by gradient ascent:

$$
\begin{align*}
\nabla_\theta
&\approx
\sum_{d_k >_c d_l} \nabla_\theta P\left( d_k \succ d_l \vert \boldsymbol{\mathcal{D}}, \theta\right)
\\ &=
\sum_{d_k >_c d_l}
\frac{
  \exp[f_\theta(d_k)]\exp[f_\theta(d_l)]
}{
  \left( \exp[f_\theta(d_k)] + \exp[f_\theta(d_l)] \right)^2
} \left( f'_\theta(d_k) - f'_\theta(d_l) \right)\,.
\end{align*}
$$

However, we can't directly maximize this probability because it is **biased** &mdash; some preferences are more likely to be observed than others due to position and selection biases. More specifically, if documents $$d_k$$ and $$d_l$$ are equally relevant, but $$d_k$$ is ranked higher than $$d_l$$, then the probability of observing user preference $$d_k >_c d_l$$ is higher than the probability of observing user preference $$d_l >_c d_k$$ because of the position bias. PDGD resolves this issue by re-weighting the preferences as follows:

- Let $$\boldsymbol{\mathcal{\pi}}^*(d_k, d_l, \boldsymbol{\mathcal{\pi}})$$ be the same ranking as $$\boldsymbol{\mathcal{\pi}}$$, but the positions of documents $$d_k$$ and $$d_l$$ are swapped (as illustrated in the figure below).
- If $$d_k$$ and $$d_l$$ are equally relevant and a preference $$d_k >_c d_l$$ is observed in the ranking $$\boldsymbol{\mathcal{\pi}}$$, then the reverse preference $$d_l >_c d_k$$ is equally likely to be observed in the ranking $$\boldsymbol{\mathcal{\pi}}^*(d_k, d_l, \boldsymbol{\mathcal{\pi}})$$.
- Then, scoring **as if** the rankings $$\boldsymbol{\mathcal{\pi}}$$ and $$\boldsymbol{\mathcal{\pi}}^*(d_k, d_l, \boldsymbol{\mathcal{\pi}})$$ are **equally likely to occur** will result in an unbiased estimator of the gradient.

{% capture imblock_pdgd_selection %}
    {{ site.url }}/articles/images/2021-08-15-learning-to-rank/pdgd.png
{% endcapture %}
{% capture imcaption_pdgd_selection %}
  **Left:** All documents preceeding the clicked document, and one document after the last clicked is considered. Arrows shows inferred preferences of $$d_3$$ over $$\{d_1, d_2, d_4\}$$. $$d_5$$ is not considered in the algorithm. **Right:** Original ranking $$\boldsymbol{\mathcal{\pi}}$$ and the ranking $$\boldsymbol{\mathcal{\pi}}^*(d_3, d_1, \boldsymbol{\mathcal{\pi}})$$ where the positions of $$d_3$$ and $$d_1$$ are swapped.
  *(Image source: [Oosterhuis et al.](https://ilps.github.io/webconf2020-tutorial-unbiased-ltr/WWW2020handout.pdf))*
{% endcapture %}
{% include gallery images=imblock_pdgd_selection cols=1 caption=imcaption_pdgd_selection %}

The ratio between the probability of ranking $$\boldsymbol{\mathcal{\pi}}$$ and the reversed ranking $$\boldsymbol{\mathcal{\pi}}^*(d_k, d_l, \boldsymbol{\mathcal{\pi}})$$ indicates the bias between two directions:

$$
\begin{equation*}
\rho \left( d_k, d_l, \boldsymbol{\mathcal{\pi}} \vert \boldsymbol{\mathcal{D}}, \theta \right) =
\frac{
  P\left( \boldsymbol{\mathcal{\pi}}^*(d_k, d_l, \boldsymbol{\mathcal{\pi}}) \vert \boldsymbol{\mathcal{D}}, \theta \right)
}{
  P\left( \boldsymbol{\mathcal{\pi}}) \vert \boldsymbol{\mathcal{D}}, \theta \right) +
  P\left( \boldsymbol{\mathcal{\pi}}^*(d_k, d_l, \boldsymbol{\mathcal{\pi}}) \vert \boldsymbol{\mathcal{D}}, \theta \right)
}
\end{equation*}
$$

This ratio is then used to unbias the gradient estimator:

$$
\begin{equation}\tag{PDGD}
\nabla_\theta \approx
\sum_{d_k >_c d_l} \rho \left( d_k, d_l, \boldsymbol{\mathcal{\pi}} \vert \boldsymbol{\mathcal{D}}, \theta \right) \nabla_\theta P\left( d_k \succ d_l \vert \boldsymbol{\mathcal{D}}, \theta \right)
\end{equation}
$$

In the paper, the authors also provided a rigorous proof that the PDGD gradient is unbiased. More specifically, they proved that the PDGD gradient can be expressed in the weighted sum form:

$$
\begin{equation*}
\mathbb{E}_{\boldsymbol{\mathcal{D}}} \left[ \nabla_\theta \right] =
\sum_{d_i, d_j \in \boldsymbol{\mathcal{D}}}
\alpha_{ij} \left( f'_\theta (d_i) - f'_\theta(d_j)\right)
\end{equation*}
$$

where the signs of the weights $$\alpha_{ij}$$ adhere to the user preferences $$d_i >_c d_j$$ between documents:
- If documents are equally relevant $$d_k =_{\text{rel}} d_l$$, then $$\alpha_{kl} = \alpha_{lk} = 0$$.
- If the user prefers $$d_k >_c d_l$$, i.e. $$d_k$$ is more relevant, then $$\alpha_{kl} > 0$$.
- If the user prefers $$d_l >_c d_k$$, i.e. $$d_k$$ is less relevant, then $$\alpha_{kl} < 0$$.

The full **Pairwise Differentiable Gradient Descent** algorithm looks like this:

<div class="algorithm">
  <div class="algo-name">Dueling Bandit Gradient Descent</div>
  <div class="algo-input">
    Ranker $f_{\theta_0}$ with parameters (weights) $\theta_0 \in \mathbb{R}^d$, learning rate $\eta$.
  </div>
  <div class="algo-output">
    Updated ranking model $f_{\theta_T}$.
  </div>
  <div class="algo-body">
    <!-- For block -->
    <div><b>for</b> query $q_t \, (t = 1\ldots T)$ <b>do</b></div>
    <div class="algo-indent-block">
      <div>Retrieve list of documents $\boldsymbol{\mathcal{D}}$ for query $q_t$</div>
      <div>Sample ranked list $\boldsymbol{\mathcal{\pi}}$ from documents $\boldsymbol{\mathcal{D}}$ using $f_{\theta_{t-1}}$</div>
      <div>Receive clicks $\boldsymbol{c}^{\boldsymbol{q}}$ from the user</div>
      <div>Initialize gradient $\nabla_\theta \leftarrow 0$</div>
      <div><b>for all</b> preference pairs $d_k >_c d_l$ in $\boldsymbol{c}^{\boldsymbol{q}}$ <b>do</b></div>
      <div class="algo-indent-block">
        <div>Accumulate gradient $\nabla_\theta \leftarrow \nabla_\theta + \rho \left( d_k, d_l, \boldsymbol{\mathcal{\pi}} \vert \boldsymbol{\mathcal{D}}, \theta_{t-1} \right) \nabla_\theta P\left( d_k \succ d_l \vert \boldsymbol{\mathcal{D}}, \theta_{t-1} \right)$</div>
      </div>
      <div><b>end for</b></div>
      <div>Update weights $\theta_t \leftarrow \theta_{t-1} + \eta \nabla_\theta$</div>
    </div>
    <div><b>end for</b></div>
  </div> <!-- algo-body -->
  <div class="algo-end"></div>
</div>



<a name="advanced-click-models"></a>
### 4.5. Advanced Click Models

So far, we've only discussed the simplest bias &mdash; the position bias. A kind reminder for the fellow scholars (in the voice of [Karoly Zsolnai-Feher](https://www.youtube.com/channel/UCbfYPyITQ-7l4upoX8nvctg)) who already forgot: the position bias is the bias that the user is more likely to click on the documents that are ranked higher. This bias can be modelled using Position-based Model, or PBM ([Craswell & Taylor, 2008][experimental_comparison_of_click_models]):

$$
\underbrace{
  P \left(c_d = 1 \vert \boldsymbol{\mathcal{\pi}} \right)
}_{
  \substack{
    \text{Probability of click} \\
    \text{on document } d
  }
}
=
\underbrace{
  P \left(o_d = 1 \vert \boldsymbol{\mathcal{\pi}} \right)
}_{
  \substack{
    \text{Probability of document } d \\
    \text{being observed}
  }
}
\cdot
\underbrace{
  P \left(y_d = 1 \vert \boldsymbol{\mathcal{\pi}} \right)
}_{
  \substack{
    \text{Probability of document } d \\
    \text{being relevant to query}
  }
}
$$

which basically says "user clicks on a document if the user observes it and perceives it as relevant". The core assumption of the PBM is that the act of observing an item is independent of its relevance or other items in the ranking.

This assumption does not always hold true. Moreover, there are many other biases that can affect the user's click behavior. Again, $$c_d$$ denotes the click on document $$d$$, $$o_d$$ denotes the observation of document $$d$$, $$y_d$$ denotes the relevance of document $$d$$, and $$\boldsymbol{\mathcal{\pi}}$$ denotes the displayed ranking of documents. Correcting these biases can be more tricky than just re-weighting the clicks. [Chulkin et al. (2015)][clickmodelsforwebsearch] and [Grotov et al. (2015)][clickmodelsforwebsearch2] have quite a comprehensive overview of the most common biases and how to model them.


<a name="cascading-position-bias"></a>
#### 4.5.1. Cascading Position Bias

Unlike PBM, the Cascading Model, first analyzed in the works of [Craswell & Taylor (2008)][experimental_comparison_of_click_models], does not assume that the act of observing an item is independent from other items. Instead, it assumes that the user will scan documents on the SERP page from top to bottom **until they find a relevant document.** Hence, observation (or examination) depends not only on the position of the document but also on the relevance of previously seen items.

Given the displayed ranking $$\boldsymbol{\mathcal{\pi}} = \{ \boldsymbol{\mathcal{\pi}}_1, \ldots, \boldsymbol{\mathcal{\pi}}_k \}$$ for the user query $$\boldsymbol{\mathcal{q}}$$, the probability of the user clicking on a document $$d$$ at position $$\mathrm{rank}(d \vert \boldsymbol{\mathcal{\pi}})$$ can be expressed as:

$$
\begin{equation*}
\underbrace{
  P \left(c_d = 1 \vert \boldsymbol{\mathcal{\pi}} \right)
}_{
  \substack{
    \text{Probability of click} \\
    \text{on document } d
  }
}
=
{\underbrace{
  P \left(y_d = 1 \vert \boldsymbol{\mathcal{\pi}} \right)
}_{
  \substack{
    \text{Probability of document } d \\
    \text{being relevant to query}
  }
}}
\cdot
{\underbrace{
  \prod_{i=1}^{\mathrm{rank}(d\vert \boldsymbol{\mathcal{\pi}})-1}
  \left( 1 - P \left(y_{\boldsymbol{\mathcal{\pi}}_i} = 1 \vert \boldsymbol{\mathcal{\pi}} \right) \right)
}_{
  \substack{
    \text{Probability of documents displayed} \\
    \text{before document} d \text{ being irrelevant}
  }
}}
\end{equation*}
$$

Have you noticed a problem? The examination term depends on the **relevance of other items,** but relevance is unknown and yet to be learned.

One way to estimate that is to treat is as a black-box propensities and use RegressionEM or DLA ([Ai et al. 2018][ai_2018]), but they still contain a major flaw &mdash; the estimated examination propensities will be session-independent. [Vardasbi et al. (2020)][vardasbi2020] proposed a more efficient method &mdash; treat the examination term as a **session-dependent** probability term! The cascade model becomes:

$$
\begin{equation*}
P \left(c_d = 1 \vert \boldsymbol{\mathcal{\pi}} \right)
=
P \left(y_d = 1 \vert \boldsymbol{\mathcal{\pi}} \right)
\cdot
{\underbrace{
  P \left(e_d = 1 \vert c_1, \ldots, c_{\mathrm{rank}(d\vert \boldsymbol{\mathcal{\pi}})-1} \right)
}_{
  \substack{
    \text{Probability examination of document } d \\
    \text{given clicks on previous ranks}
  }
}}
\end{equation*}
$$

Instead of global propensities like in PBM, this formulation of cascade model requires different propensities per query! [Vardasbi et al. (2020)][vardasbi2020] leverages clicks in the current user session to estimate per-query propensities. In particular, two extensions of the cascade model are analyzed:
- Dependent Click Model (DCM) &mdash; a session can have multiple clicks, i.e. if a user clicks on a document, they may still examine other documents. In this case, the examination probability of document $$r$$ is conditioned on the clicks on previous rank $$r - 1$$:

  $$
  \begin{equation*}
  P \left(e_r = 1 \vert c_{r - 1} = 1, \boldsymbol{\mathcal{\pi}} \right) = \lambda_r
  \end{equation*}
  $$

  where $$\lambda_r$$ is the continuation parameter, which depends only on the rank of the document. I tend to think of it as "fatigue" parameter &mdash; the lower the rank, the less likely the user is to continue examining the documents. Therefore:

  $$
  \begin{equation*}
  P_{\text{DCM}} \left(e_d = 1 \vert \{ c_i \}_{i < \mathrm{rank}(d\vert \boldsymbol{\mathcal{\pi}})-1} \right)
  =
  \prod_{i=1}^{\mathrm{rank}(d\vert \boldsymbol{\mathcal{\pi}})-1}
  \left( c_i \cdot \lambda_i \right) + \left( 1 - c_i \right)
  \end{equation*}
  $$

- Dynamic Bayesian Network (DBN) &mdash; a more complex model where an event $$s_i = 1$$ that represents satisfaction is introduced. A satisfied user abandons the session. An unsatisfied user might also abandon the session with a constant probability $$\gamma$$. After a click, the satisfaction probability depends on the document: $$P(s_i = 1 \vert c_i = 1) = s_{\boldsymbol{\mathcal{\pi}}_i}$$. Therefore:

  $$
  \begin{equation*}
  P_{\text{DBN}} \left(e_d = 1 \vert \{ c_i \}_{i < \mathrm{rank}(d\vert \boldsymbol{\mathcal{\pi}})-1} \right)
  =
  \prod_{i=1}^{\mathrm{rank}(d\vert \boldsymbol{\mathcal{\pi}})-1}
  \gamma \cdot \left (1 - c_i \cdot s_{\boldsymbol{\mathcal{\pi}}_i} \right)
  \end{equation*}
  $$

These examination probabilities can be easily plugged back into the IPS framework as follows (proof of unbiasedness is left as an exercise to the reader):

$$
\begin{equation*} \tag{IPS-CM}
  \Delta_{\text{IPS-CM}} \left(
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
      P \left(e_d = 1 \vert \{ c_i \}_{i < \mathrm{rank}(d\vert \boldsymbol{\mathcal{\pi}}_\phi)-1} \right)
    }
  }
\end{equation*}
$$

Some other extensions of the Cascading Model are described in [Chulkin et al. (2015)][clickmodelsforwebsearch], including multiple clicks per session (impression) and abandoning a session without clicking. Deriving a bias term for them should not pose any difficulties for the reader after understanding the DCM and DBN models.


<a name="trust-bias"></a>
#### 4.5.2. Trust Bias


When your search engine become good enough, users will start to trust it. They are more likely to perceive the top documents on the SERP page to be relevant, even when the displayed information about the item suggests otherwise. This is called the **trust bias**.

The trust bias can be modelled by distinguishing **real relevance** $$y_d$$ of a document $$d$$ (as proposed by [Agarwal et al. 2019][agarwal_trust_2019]) and **perceived relevance** $$\hat{y}_d$$. Trust bias occurs because users are more likely to perceive top items as relevant $$\hat{y}_d = 1$$. In this model, a click happens when the user observes and perceives an item to be relevant, thus the click probability of document $$d$$ at position $$k = \mathrm{rank}(d \vert \boldsymbol{\mathcal{\pi}})$$ can be expressed as:

$$
\begin{equation*}
P \left(c_d = 1 \vert \boldsymbol{\mathcal{\pi}} \right)
=
P \left(e_d = 1 \vert k \right)
\cdot
P \left(\hat{y}_d = 1 \vert e_d = 1, y_d, k \right)
\end{equation*}
$$

Notice the symmetry with PBM and CM models. In CM, the static observation probability term of PBM is replaced with a dynamic term. In Trust Bias, the relevance term is replaced with a perceived relevance term. Since these terms are independent from each other, the Trust Bias model can be easily combined with the Cascade Model.

[Vardasbi et al. (2020)][affine_correction] provided a rigorous proof that **no IPS estimator is unbiased w.r.t. trust bias.** Instead, they proposed to model the combination of the trust bias and position bias as an affine transformation between the relevance probabilities and click probabilities. As in *Regression EM*, we denote $$\theta_k$$ as the position bias (which only depends on document's position $$k$$ in the absense of other biases) and $$\gamma_{\boldsymbol{q}, d}$$ as the probability of actual relevance:

$$
\begin{equation*}
\theta_k = P(e = 1 \vert k)\,,\quad \gamma_{\boldsymbol{q}, d} = P(y = 1 \vert \boldsymbol{q}, d)
\end{equation*}
$$

Additionally, we explicitly consider the probability of perceived relevance of an examined item displayed at rank $$k$$, conditioned on the actual relevance of the item:

$$
\begin{align*}
P(\hat{y} = 1 \vert e = 1, y = 1, k) &= \epsilon_k^+ \\
P(\hat{y} = 1 \vert e = 1, y = 0, k) &= \epsilon_k^- \\
\end{align*}
$$

Having these shorthand notations, we can express the click probability as:

$$
\begin{equation*}
P(c = 1 \vert \boldsymbol{q}, d, k) = \theta_k \cdot \left(
  \epsilon_k^+ \cdot \gamma_{\boldsymbol{q}, d} +
  \epsilon_k^- \cdot (1 - \gamma_{\boldsymbol{q}, d})
\right)
\end{equation*}
$$

With a change of variables $$\alpha_k = \theta_k \cdot (\epsilon_k^+ - \epsilon_k^-)$$ and $$\beta_k = \theta_k \cdot \epsilon_k^-$$, we can express the click probability as an affine transformation:

$$
\begin{equation*}
P(c = 1 \vert \boldsymbol{q}, d, k) = \alpha_k \cdot P(y = 1 \vert \boldsymbol{q}, d) + \beta_k
\end{equation*}
$$

With this, we can construct the affine estimator for the Trust Bias:

$$
\begin{equation*} \tag{Affine}
  \Delta_{\text{Affine}} \left(
    \boldsymbol{\mathcal{\pi}}_\phi, \boldsymbol{\mathcal{y}} \vert
    \boldsymbol{\pi}_\theta
  \right)
  =
  \sum_{d \in \boldsymbol{\mathcal{\pi}}_\phi} {
    \frac{
      \mu \big[
        \boldsymbol{\pi}_\phi(d)
      \big]
      \cdot y_d
    }{
      \alpha_{\mathrm{rank}(d\vert \boldsymbol{\mathcal{\pi}})} \cdot P(y = 1 \vert \boldsymbol{q}, d) + \beta_{\mathrm{rank}(d\vert \boldsymbol{\mathcal{\pi}})}
    }
  }


---------------------------------------------------------------------------------



<a name="practical-considerations"></a>
## 5. Practical Considerations

In this section, I’ll share some of the lessons I’ve learned from hands-on experience. These are practical tips that usually are not mentioned in the published literature, things that make data science as much an art as it is a science. I hope these insights will help you navigate the complexities while building your own Learning to Rank projects in a real-world setting.


<a name="design-ndcg-labels"></a>
### 5.1. Design your NDCG labels carefully.

The first version of your ranking model, if you ever need one, will likely be trained in a supervised manner. It is just so much simpler than all the fancy Unbiased LTR stuff. Depending on your business case and user interface (UX), it is likely that you will find NDCG to be a good metric to optimize. Users care most about top results. Even if you are building a RAG pipeline, your LLM will likely appreciate having more relevant results in top-K positions as well. In this case, if NDCG is your choice of quality metric, you will likely use LambdaMART or LambdaRank as your learning algorithm, as they are widely implemented in libraries like LightGBM and XGBoost.

When it comes to designing labels for NDCG measurement, the obvious choise is to just ask labelers to directly provide the relevance scores (e.g. from 0 to 5) for query-document pairs. However, this is **the worst** way to do it. The reason is that the relevance scores are subjective and can vary greatly between labelers. Here are some better ways to do it:

- **Use checklist with points.** Instead of asking for relevance scores, ask labelers to check a list of points that make a document relevant. This way, you can ensure that the labels are consistent across labelers. Google has a huge check list for search quality evaluation, just check out their guidelines: [Search Quality Evaluator Guidelines][google_sqe_guidelines]. The final list of question depends on your business case. A few examples of points that can be included in the checklist (+1 point if the document satisfies the point, 0 otherwise):
  - Is the document relevant to the query?
  - Does the document provide a direct answer to the query?
  - Is the document from a reputable source?
  - Is the document up-to-date?
  - Does the document match the user's intent, based on user's search history (if available)?
  - ... and so on.

- **Use pairwise comparison.** Instead of asking for relevance scores, ask labelers to compare two documents and choose the one that is more relevant to the query. This way, you can ensure that the labels are more consistent across labelers. The final relevance score can be calculated as the fraction of times a document is chosen over another document. This is the approach used in many of the LETOR datasets. This approach does **not** guarantee the lack of biases, but it is cheaper and faster than the checklist approach. You can actually combine the scores from both checklist and pairwise comparison approaches.

You will need a few iterations to get the labels and metrics design right. [Bing][bing], for example, has iterated on their labels design for years.



<a name="when-to-interleave"></a>
### 5.2. Know when can you use Interleaving.

Interleaving is a powerful technique to compare two rankers, but it is not always applicable. It is very important to understand **when** you can use interleaving and you can't. I've seen data scientists more experienced than me accidentally using interleaving instead of A/B testing in cases where it is not applicable, and then they spent weeks trying to figure out why they did not get the expected improvements.

Generally, interleaving works well when the quantity you're evaluating directly depends on per-document interactions, thus can be attributed to the specific ranker. More on attribution can be found in this paper by [Radlinski and Craswell (2013)][optimized_interleaving]. A good rule of thumb is, if the quality metric you're trying to infer is an accumulated sum of per-document scores (like NDCG, or relevance in general), then interleaving is likely applicable. If the quality metric is a more global, per-session metric (like session time, ads revenue, or conversion rate), then interleaving is likely not applicable. A few examples to build up your intuition:

- Products sales in an e-commerce platform: interleaving is likely **applicable**, as the sales ultimately depend on the relevance of the products shown to the user, i.e. can be attributed to specific items, hence to the specific ranker's results.
- Ads revenue of a SERP (Search Engine Results Page), it the ads is displayed on the side: interleaving is likely **not applicable**, as the revenue depends on the ads shown to the user, not the relevance of the ranker's results. We can't attribute the revenue to the results of a specific ranker, but rather to the combined ranking of all rankers (i.e. to overall user experience).
- User retention in a news recommendation system: interleaving is likely **not applicable**, as the retention depends on the user's overall experience, not the relevance of the ranker's results.
- Dwell time per session in a news recommendation system: interleaving is likely **applicable**, as the dwell time can be attributed to the relevance of the documents shown to the user.

When in doubt of whether interleaving is applicable to the thing you want to evaluate, it is always a good idea to run a small-scale experiment to see if the interleaving results align with A/B testing results. If they don't, then interleaving is likely not applicable to your case. AirBnB has a great write-up on how they used interleaving to compare search ranking algorithms and made sure that it is well-aligned with A/B testing: ["Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving." (2022)][airbnb_interleaving].

{% capture imblock_interleaving_airbnb %}
    {{ site.url }}/articles/images/2021-08-15-learning-to-rank/interleaving_airbnb.png
{% endcapture %}
{% capture imcaption_interleaving_airbnb %}
  Interleaving and A/B consistency at AirBnB. They tracked eligible interleaving and A/B ranker pairs and the results demonstrate that the two are consistent with each other 82% of the time. *(Image source: [AirBnB Engineering](https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e))*
{% endcapture %}
{% include gallery images=imblock_interleaving_airbnb cols=1 caption=imcaption_interleaving_airbnb %}



<a name="modelling-tips"></a>
### 5.3. Modelling tips for LTR

It is very tempting to jump to latest and shiniest methods, but the first version of your ranking model will should be a supervised one. There's just too much complexity in the unbiased learning to rank algorithms to start with them. Moreover, if you don't have a large enough user base and a good spam/bots filter, you will likely have a lot of noise in your click data.


{% comment %}
<a name="always-ablate"></a>
### 5.4. Always ablate, don't trust academic benchmarks
{% endcomment %}



---------------------------------------------------------------------------------

<a name="conclusion"></a>
## 6. Conclusion

If you are interested in Recommendation Systems (RecSys), I highly recommend the following incredible blogs (I personally learned a lot from them):

- **[https://eugeneyan.com](https://eugeneyan.com)** &mdash; Eugene Yan has one of the most amazing data science blogs out there. He has a lot of great posts on recommendation systems, machine learning, and data science in general. Although he lately pivoted to LLMs, his old posts on RecSys are still very relevant.
- **[https://blog.reachsumit.com](https://blog.reachsumit.com)** &mdash; Sumit is an incredibly knowledgeable machine learning engineer who worked at some of the most amazing RecSys teams in the industry (Amazon, TikTok, Meta). You need to be fluent in various ML areas to fully understand his posts though.





---------------------------------------------------------------------------------



<a name="references"></a>
## References

1. Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin, Janani Padmanabhan, Giuseppe Ottaviano, Linjun Yang. ["Embedding-based Retrieval in Facebook Search."][fb-search-engine] In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2020.

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

15. Taylor M., Guiver J., Robertson S., Minka T. ["SoftRank: optimizing non-smooth rank metrics."][softrank_2008] In *International Conference on Web Search and Data Mining (WSDM)*, 2008.

16. A. Agarwal, I. Zaitsev, X. Wang, C. Li, M. Najork, T. Joachims. ["Estimating Position Bias without Intrusive Interventions."][agarwal_2019] In *Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining* (WSDM), 2019.

17. Radlinski, F., Kurup, M., & Joachims, T. ["How does clickthrough data reflect retrieval quality?."][radlinski_2008] In *Proceedings of the ACM conference on Information and knowledge management (CIKM)*, 2008.

18. Yue Yisong, Joachims Thorsten. ["Interactively optimizing information retrieval systems as a dueling bandits problem."][yue_dbgd] In *Proceedings of the 26th Annual International Conference on Machine Learning* (ICML), 2009.

19. Benjamin D. Haeffele and Rene Vidal. ["Global Optimality in Neural Network Training."][global_optimality_nn] In *2017 IEEE Conference on Computer Vision and Pattern Recognition* (CVPR), 2017.

20. Schuth A., Oosterhuis H., Whiteson S., de Rijke M. ["Multileave Gradient Descent for Fast Online Learning to Rank."][multileave_dbgd] In *WSDM '16: Proceedings of the Ninth ACM International Conference on Web Search and Data Mining*, 2016.

21. Zhao T., King I. ["Constructing Reliable Gradient Exploration for Online Learning to Rank."][zhao_king_2016] In *Proceedings of the 25th ACM International on Conference on Information and Knowledge Management* (CIKM), 2016.

22. Hofmann K., Whiteson S., de Rijke M. ["Reusing historical interaction data for faster online learning to rank for IR."][hofmann_2013] In *Proceedings of the sixth ACM international conference on Web search and data minin* (WSDM), 2013.

23. Huazheng Wang, Ramsey Langley, Sonwoo Kim, Eric McCord-Snook, Hongning Wang. ["Efficient Exploration of Gradient Space for Online Learning to Rank."][wang_dbgd_2018] In *The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval* (SIGIR), 2018.

24. Harrie Oosterhuis, Maarten de Rijke. ["Optimizing Ranking Models in an Online Setting."][oosterhuis_2019] In *European Conference on Information Retrieval* (ECIR), 2019.

25. Harrie Oosterhuis, Maarten de Rijke. ["Differentiable Unbiased Online Learning to Rank."][oosterhuis_2019] In *CIKM '18: Proceedings of the 27th ACM International Conference on Information and Knowledge Management* (SIGIR), 2019.

26. Filip Radlinski, Nick Craswell. ["Optimized Interleaving for Online Retrieval Evaluation."][optimized_interleaving] In 
*Proceedings of the sixth ACM international conference on Web search and data mining* (WSDM), 2013.

27. Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, Hang Li. ["Learning to Rank: From Pairwise Approach to Listwise Approach."][listnet] In *Proceedings of the 24th international conference on Machine learning* (ICML), 2007.

28. Alexandr Chulkin, Ilya Markov, Maarten de Rijke. ["Click Models for Web Search."][clickmodelsforwebsearch] In *Synthesis Lectures on Information Concepts, Retrieval, and Services*, 2015.

29. Grotov A., Chulkin A., Markov I., Stout L., Xumara F., Rijke M. ["A Comparative Study of Click Models for Web Search."][clickmodelsforwebsearch2] In *Proceedings of the 6th International Conference on Experimental IR Meets Multilinguality, Multimodality, and Interaction* (CLEF), 2015.

30. Ali Vardasbi, Maarten de Rijke, Ilya Markov. ["Cascade Model-based Propensity Estimation for Counterfactual Learning to Rank."][vardasbi2020] In *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval* (SIGIR), 2020.

31. Xuanhui Wang, Nadav Golbandi, Michael Bendersky, Donald Metzler, Marc Najork. ["Position Bias Estimation for Unbiased Learning to Rank in Personal Search."][regressionem] In *Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining* (WSDM), 2018.

32. Aman Agarwal, Xuanhui Wang, Cheng Li, Mike Bendersky, Marc Najork. ["Addressing Trust Bias for Unbiased Learning-to-Rank."][agarwal_trust_2019] In *Proceedings of the 2019 World Wide Web Conference* (WWW'19).

33. Ali Vardasbi, Harrie Oosterhuis, and Maarten de Rijke. ["When Inverse Propensity Scoring does not Work: Affine Corrections for Unbiased Learning to Rank."][affine_correction] In *Proceedings of the 29th ACM International Conference on Information and Knowledge Management* (CIKM), 2020.



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
[softrank_2008]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/SoftRankWsdm08Submitted.pdf
[agarwal_2019]: https://arxiv.org/abs/1812.05161
[radlinski_2008]: https://www.cs.cornell.edu/people/tj/publications/radlinski_etal_08b.pdf
[yue_dbgd]: https://www.cs.cornell.edu/people/tj/publications/yue_joachims_09a.pdf
[global_optimality_nn]: https://openaccess.thecvf.com/content_cvpr_2017/papers/Haeffele_Global_Optimality_in_CVPR_2017_paper.pdf
[multileave_dbgd]: https://irlab.science.uva.nl/wp-content/papercite-data/pdf/schuth-multileave-2016.pdf
[zhao_king_2016]: https://dl.acm.org/doi/abs/10.1145/2983323.2983774
[hofmann_2013]: https://dl.acm.org/doi/abs/10.1145/2433396.2433419
[wang_dbgd_2018]: https://arxiv.org/abs/1805.07317
[oosterhuis_2019]: https://arxiv.org/abs/1901.10262
[optimized_interleaving]: https://www.microsoft.com/en-us/research/wp-content/uploads/2013/02/Radlinski_Optimized_WSDM2013.pdf.pdf
[listnet]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf
[clickmodelsforwebsearch]: https://clickmodels.weebly.com/uploads/5/2/2/5/52257029/mc2015-clickmodels.pdf
[clickmodelsforwebsearch2]: https://irlab.science.uva.nl/wp-content/papercite-data/pdf/grotov-comparative-2015.pdf
[vardasbi2020]: https://arxiv.org/abs/2005.11938
[regressionem]: https://dl.acm.org/doi/10.1145/3159652.3159732
[agarwal_trust_2019]: https://research.google/pubs/addressing-trust-bias-for-unbiased-learning-to-rank/
[affine_correction]: https://irlab.science.uva.nl/wp-content/papercite-data/pdf/vardasbi-2020-inverse.pdf