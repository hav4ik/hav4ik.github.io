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
excerpt: "Search relevance ranking is one of the most important part of any search and recommendation system. This post is just my personal study notes, where I delve deeper into approaches for ranking search results and try to make sense for myself."
show_excerpt: true
comments: true
hidden: false
---

I still remember being fascinated by Google Search when I saw it the first time. As an 8th-grade kid getting his first computer, the ability to search for any information I want among billions of web pages looked like magic to me. As Arthur C. Clarke famously said, ["any sufficiently advanced technology is indistinguishable from magic."][tech_is_magic] By that definition, the search engines that allow us to access thousands of years of humanity's accumulated knowledge at our fingertip, are the modern version of magic!

Back then, even in my wildest dreams, I couldn't have imagined that 25 years old me will have the privilege to move across the globe to work on a search engine called [Microsoft Bing][bing] &mdash; an ambitious project with the guts to compete with Google in the search market! The "magic" behind that little search box is even more impressive than what I have assumed. The search engine is a truly gigantic marvel of technology, built and supported by thousands of engineers, data scientists, and machine learning scientists. There is a lot for me to learn about and there is a lot of things that I don't know, so in this blog post, I'll take you together with me on my study journey about Learning to Rank algorithms.

**Disclaimer:** all information in this blog post is taken from published research papers or publically available online articles. No [NDA][nda]s were violated. You won't find any details specific to the inner working of [Bing][bing] or other search engines here :)


[tech_is_magic]: https://en.wikipedia.org/wiki/Clarke%27s_three_laws
[bing]: https://www.bing.com/
[nda]: https://en.wikipedia.org/wiki/Non-disclosure_agreement


------------------------------------------------------------------------------------


- [How do search engines work?](#how-search-engines-work)
- [Measuring ranking quality with NDCG](#)
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

In information retrieval, the items that are being searched (e.g. videos, books, clothes, etc.) are regarded as **documents.** All search engines, on the most abstract schematic level, have a similar underlying mechanism of searching for the most relevant documents for a given query, as shown below:

{% capture imblock_search_engine %}
    {{ site.url }}/articles/images/2021-08-15-learning-to-rank/search_engine.png
{% endcapture %}
{% capture imcaption_search_engine %}
  Simplified general schema of search engines. Features extracted from all documents are stored in the index database. For a search given query, top k documents are retrieved from the index database and then sorted by their relevance to the given query.
{% endcapture %}
{% include gallery images=imblock_search_engine cols=1 caption=imcaption_search_engine %}