---
layout: post
type: "article"
title: "Hello World! (again)"
image:
  feature: "articles/images/2018-11-25-hello-world/feature.png"
  display: false
tags: [stories of my life, tutorial]
excerpt: "This blog was in pretty crappy condition before, then I decided to make it great again. Here is the story..."
comments: true
---

Just not so long ago, this blog was in a pretty crappy condition: the articles were empty (I was too lazy to write anything), the design was messy, and the homepage was just a placeholder. I always wanted to make a proper online blog/portfolio since many sources (e.g. [here][why-portfolio-1], [here][why-portfolio-2], and [here][why-portfolio-3]) suggests that it is very helpful in job seeking and is good generally, but I was too lazy to finish it.

The old code was a mess -- it is impossible to modify it, so I did the following with my old site:
{% capture imblock1 %}
    {{ site.url }}/articles/images/2018-11-25-hello-world/delete-repo.png
{% endcapture %}
{% include gallery images=imblock1 cols=1 %}

The purpose of this post is mainly to remind myself of how I made it and from where I copied the parts of the code (more on that in the next section), so that I can troubleshoot or upgrade the design or make modifications in the future.

## How did I overcome my laziness and fixed this blog?

Well, long story short, this person is the reason:
{% capture imblock2 %}
    {{ site.url }}/articles/images/2018-11-25-hello-world/reason-1.jpg
    {{ site.url }}/articles/images/2018-11-25-hello-world/reason-2-3.jpg
{% endcapture %}
{% include gallery images=imblock2 cols=2 %}

And this one on the other day was the tipping point...
{% capture imblock3 %}
    {{ site.url }}/articles/images/2018-11-25-hello-world/reason-4.png
{% endcapture %}
{% include gallery images=imblock3 %}

So, giving her my Github link was a **really really bad idea**... Glad I didn't give her the link to my other Github account with crappier projects!


## Where I stole the code?

Do you really expected me, a **Machine Learning engineer** and a **Mathematician**, to know designing and `HTML` and `CSS` and `Javascript` and other stuffs related to web designing? *Of course not!* I've worked in *gamedev* before, I had some time in *backend* before, but definitely *never* did anything seriously with frontend! So I *stole* the snippets and design elements from other sites and blogs, and modified them so that they comes together into a nice and sleek design.

*  From [here][article-list] I stole the design of list of articles.
*  From [here][navbar] I stole the navigation bar (and added cool elements of my own).
*  I got the inspiration (the code) for post design from [here][post-design].
*  The sleek portfolio design and buttons I got from [here][portfolio].
*  From [here][liam-fedus] I stole the timeline! (that guy from MILA is really cool)

I don't remember where I got the grid layout for [travelog][travelog] though. The awesome map I got from [mapael][mapael]. Well, basically, I just **copy-pasted** my way to this blog.


[mapael]: https://www.vincentbroute.fr/mapael/
[travelog]: /travelog
[portfolio]: https://sproogen.github.io/modern-resume-theme/
[article-list]: https://nathanrooy.github.io/
[navbar]: http://jekyllthemes.org/themes/voyager/
[post-design]: http://jekyllthemes.org/themes/moon/
[liam-fedus]: http://acsweb.ucsd.edu/~wfedus/
[why-portfolio-1]: https://towardsdatascience.com/how-to-build-a-data-science-portfolio-5f566517c79c
[why-portfolio-2]: http://varianceexplained.org/r/start-blog/
[why-portfolio-3]: https://www.superdatascience.com/why-ds-should-write-more/
