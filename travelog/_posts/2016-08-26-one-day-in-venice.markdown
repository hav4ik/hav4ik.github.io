---
layout: post
title: "Morning in Venice"
date: 2016-08-26
permalink: /travelog/:title

mapinfo:
  city: "Venice"
  latitude: 45.4408
  longtitude: 12.3155

image:
  feature: travelog/images/2016-08-26-venice/IMG_6450.jpg
  display: true

excerpt: ""
comments: true
---

{% capture imblock1 %}
	{{ site.url }}/img/2016-08-26-venice/IMG_6412.jpg
	{{ site.url }}/img/2016-08-26-venice/IMG_6450.jpg
	{{ site.url }}/img/2016-08-26-venice/IMG_6480.jpg

{% endcapture %}

{% capture imblock2 %}
	{{ site.url }}/img/2016-08-26-venice/IMG_6540.jpg
{% endcapture %}

{% capture imblock3 %}
	{{ site.url }}/img/2016-08-26-venice/IMG_6572.jpg
	{{ site.url }}/img/2016-08-26-venice/IMG_6550.jpg
	{{ site.url }}/img/2016-08-26-venice/IMG_6646.jpg
	{{ site.url }}/img/2016-08-26-venice/IMG_6599.jpg

{% endcapture %}


{% capture imblock4 %}
	{{ site.url }}/img/2016-08-26-venice/IMG_6616.jpg
	{{ site.url }}/img/2016-08-26-venice/IMG_6650.jpg
	{{ site.url }}/img/2016-08-26-venice/IMG_6662.jpg
	{{ site.url }}/img/2016-08-26-venice/IMG_6661.jpg
	{{ site.url }}/img/2016-08-26-venice/IMG_6653.jpg
	{{ site.url }}/img/2016-08-26-venice/IMG_6693.jpg
{% endcapture %}

{% capture imblock5 %}
	{{ site.url }}/img/2016-08-26-venice/IMG_6603.jpg
	{{ site.url }}/img/2016-08-26-venice/IMG_6645.jpg
	{{ site.url }}/img/2016-08-26-venice/IMG_6665.jpg
	{{ site.url }}/img/2016-08-26-venice/IMG_6680.jpg
	{{ site.url }}/img/2016-08-26-venice/IMG_6681.jpg
{% endcapture %}


Travelling is always a great experience. However, it's not cheap, and sometimes your money is only enough to spend only one day in the place you desired for years to visit. How should I spend my 18 hours in Venice? 


{% include gallery images=imblock1 cols=1 %}
{% include gallery images=imblock2 cols=1 caption="The view at night from Rialto bridge is gorgeous" %}

{% include gallery images=imblock3 cols=2 %}
{% include gallery images=imblock4 cols=2 %}

{% include gallery images=imblock5 cols=1 %}
