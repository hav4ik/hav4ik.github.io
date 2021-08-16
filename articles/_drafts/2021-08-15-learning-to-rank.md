---
layout: post
permalink: /articles/:title
type: "article"
title: "Learning to Rank: at the core of a Search Engine"
image:
  feature: "/articles/images/2021-08-15-learning-to-rank/feature.png"
  display: false
commits: "#"
tags: [machine-learning, long-read]
excerpt: "Search relevance ranking is one of the most important part of any search and recommendation system. This post is just my personal study notes, where I delve deeper into Learning-to-Rank (LTR) approaches and try to make sense for myself."
show_excerpt: true
comments: true
hidden: false
---

I still remember being fascinated by Google Search when I saw it the first time. As an 8th-grade kid getting his first computer, the ability to search for any information I want among billions of web pages looked like magic to me. As Arthur C. Clarke famously said, ["any sufficiently advanced technology is indistinguishable from magic."][tech_is_magic] By that definition, the search engines that allow us to access thousands of years of humanity's accumulated knowledge at our fingertip, are the modern version of magic!

Back then, even in my wildest dreams, I couldn't have imagined that 25 years old me will have the privilege to move across the globe to work on a search engine called [Microsoft Bing][bing] &mdash; an ambitious project with enough guts to compete with Google in the search market! The "magic" behind that little search box is even more impressive than what I have assumed. The search engine is a truly gigantic marvel of modern technology, built and supported by tens of thousands of hardware and software engineers, data scientists, and machine learning scientists.

There is a lot for me to learn about and there is a lot of things that I don't know, so in this blog post, I'll take you together with me on my study journey about [Learning to Rank (LTR)][ltr] algorithms. If you spotted any mistakes in this post or if I'm completely wrong in some sections, please let me know.

**Disclaimer:** all information in this blog post is taken from published research papers or publically available online articles. No [NDA][nda]s were violated. You won't find any details specific to the inner working of [Bing][bing] or other search engines here :)


[tech_is_magic]: https://en.wikipedia.org/wiki/Clarke%27s_three_laws
[bing]: https://www.bing.com/
[nda]: https://en.wikipedia.org/wiki/Non-disclosure_agreement
[ltr]: https://en.wikipedia.org/wiki/Learning_to_rank


------------------------------------------------------------------------------------


- [How do search engines work?](#how-search-engines-work)
- [About Search Relevance](#about-search-relevance)
- [Measuring ranking quality](#)
  - [NDCG](#)
- [Learning to Rank](#)
  - [RankNet](#)
  - [LambdaRank](#)
  - [LambdaMART](#)
- [Introducing Ranking Biases](#)
- [Overcoming biases in Learning to Rank](#)
- [References](#)


------------------------------------------------------------------------------------


<a name="how-search-engines-work"></a>
## How do search engines work?

Not all search engines are built with the ambitious goal of "searching the whole internet." Tech giants like Quora, Netflix, Amazon, and Facebook have in-house search engines as well, created to recommend the best products, content, and movies that match the user’s search queries. Big online retail companies, for example, also have their own search engines. That's how they recommend you the products that you are more likely to be interested in, given your prior purchases.

In information retrieval, the items that are being searched for (e.g. videos, books, web pages, etc.) are regarded as **documents.** All modern search engines, on the most abstract schematic level, have a similar underlying mechanism of searching for the most relevant documents for a given query:

{% capture imblock_search_engine %}
    {{ site.url }}/articles/images/2021-08-15-learning-to-rank/search_engine.png
{% endcapture %}
{% capture imcaption_search_engine %}
  Simplified general schema of search engines. Features extracted from all documents using the indexer are stored in the index database. For a search given query, top k documents are retrieved from the index database and then sorted by their relevance to the given query.
{% endcapture %}
{% include gallery images=imblock_search_engine cols=1 caption=imcaption_search_engine %}

**Indexing** is performed continuously offline. At this step, meaningful features and signals from all documents are extracted and stored in the Index database. For retail companies, these features can be as primitive as raw description or [TF-IDF][tfidf] of the product description together with its popularity and user rating. For web-scale search engines like Google and [Bing][bing], the index is constructed from thousands of different signals and compressed embeddings from state-of-the-art neural networks like [BERT][bert]. Needless to say, feature engineering is extremely important, so the choice of what signals and features to extract is kept secret by each search engine to them the competitive edge on the market.

**Top-k Retrieval** (sometimes also called **"Level-0 Ranking"** or **"Matching"**) is performed on each user's query to retrieve the potentially relevant documents for the given query. For small search engines, simple text matching is usually enough at this stage. For web-scale search engines, an embedding vector is calculated for the given query, and then k nearest embedding vectors (by euclidean or cosine similarity) of all documents stored in the Index database are retrieved.

[Huang et al. (2020)][fbsearch_embedding] described in detail how Facebook Search is using Embedding-based Retrieval in their search engine. [Bing Search][bing], according to their [2018 blog post][bing_img_search_2018], calculates image embeddings in addition to text embeddings for their retrieval stage. Google's blog post ["Building a real-time embeddings similarity matching system"][google_building_retrieval] gives us a glimpse of how Embedding-based Retrieval is likely to be performed inside Google, although their inner system is for sure much more sophisticated than that, and is probably combined Rule-based Retrieval as well. The most interesting part for me is that metric trees (like [k-d tree][kdtree]) is not used in large-scale search engines due to their slow $$O(\log n)$$ complexity and large memory consumption. Instead, hashing methods (like [LHS][lhs_hashing] or [PCA hashing][pca_hashing]) are used to achieve close to $$O(1)$$ retrieval complexity.

**Ranking** (the main focus of this blog post) is the step that actually makes search engines work. Retrieved documents from the previous step are then ranked by their relevance to the given query and (optionally) the user's preferences. While handcrafted heuristic rule-based methods for relevance ranking are often more than enough for small and even mid-sized search engines, all big names in the industry right now are using Machine-Learning (i.e. [Learning-to-Rank][ltr]) techniques for search results ranking.

There was a time when [PageRank][pagerank] was a sole ranking factor for Google, but they quickly moved to more sophisticated ranking algorithms as more diverse features are extracted from web pages. As of 2020, [PageRank][pagerank] score is still a small part of Google's index, as [confirmed multiple times][pagerank_alive] by googlers. Interestingly, for a long time Google has resisted using machine learning for their core search ranking algorithm, as explained in [this Quora answer][google_hates_ml] from 2011 by a former Google engineer. For more information about Google's algorithm changes over years, [this blog post][google_algo_changes] is an excellent tracker of their recent publically known major changes.


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


------------------------------------------------------------------------------------


<a name="about-search-relevance"></a>
## About Search Relevance

