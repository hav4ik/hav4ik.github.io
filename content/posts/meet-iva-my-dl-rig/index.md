---
title: Meet “Iva”, my mini Deep Learning rig
url: "/meet-iva-my-dl-rig"
date: 2023-11-24T00:00:00+00:00
# weight: 1
# aliases: ["/first"]
tags: ["projects"]
author: "Kha Vu Chan"
# author: ["Me", "You"] # multiple authors
showToc: false
TocOpen: false
draft: false
hidemeta: false
comments: true
disqus_identifier: hav4ik/meet-iva-my-dl-rig
canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
summary: >-
    I built a cheap-ish 2x3090 RTX Deep Learning rig for my personal projects and experiments. I called it “Iva” — in honor of Oleksii Ivakhnenko from Ukraine, the Godfather of Deep Learning, who first developed an algorithm to train multi-layer perceptrons back in 1965.
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
    image: "iva_1.webp" # image path/url
    alt: "Deep learning rig with 2x3090 RTX" # alt text
    caption: "Deep learning rig with 2x3090 RTX" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide in list view
editPost:
    URL: "https://github.com/hav4ik.github.io/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

> This article was cross-posted on Medium: https://medium.com/@chankhavu/meet-iva-my-mini-deep-learning-rig-f5588588ca8a).

I’ve always relied on cloud GPU resources throughout my entire Machine Learning career. As a frugal person, I never saw the need to invest thousands of dollars in building my own rig. During university, my Intel i5 + 1070ti gaming computer and Google Colab sufficed for my Bachelor’s and Master’s theses. Starting my professional Machine Learning career in the fourth year of my Bachelor’s, I often borrowed available GPUs at work (thankfully, Samsung Research had a lot of GPUs). Not owning hardware was inconvenient but not essential.

{{< figure src="iva_1.webp" caption="Meet “Iva”, a small Deep Learning rig on my desk at home." >}}

As my skills improved, so did my need for more control over my pipelines and experiments, something beyond what the cloud setup offered. Additionally, transferring data to and from borrowed hardware at work makes me uncomfortable as well. That’s why I’ve decided to finally build my own Deep Learning rig.

I called it “Iva” — in honor of [Oleksii Ivakhnenko](https://en.wikipedia.org/wiki/Alexey_Ivakhnenko) from Ukraine, the Godfather of Deep Learning, who first developed an algorithm to train multi-layer perceptrons back in 1965. Jürgen Schmidhuber also mentioned him in his [blog post about the history of Deep Learning](https://people.idsia.ch/~juergen/deep-learning-history.html).


## Some more photos
A few more pictures of my 2x3090 mini Deep Learning rig. In my personal opinion, it’s absolutely gorgeous :)

{{< figure src="iva_21.webp,iva_22.webp,iva_23.webp,iva_24.webp" caption="More photos of Iva from different angles!" >}}

This build is still a work in progress. I will fine-tune the thermals by adding more case fans to regulate the air flow in the future.

## Inspirations
[Aleksa Gordić’s Youtube series](https://www.youtube.com/watch?v=OWvy-fCWTBQ) about building your own machine for Deep Learning was a big inspiration for me. He discussed the nuances of building a multi-GPU rig in great detail.

Tim Dettmer’s blog posts, [“A Full Hardware Guide to Deep Learning”](https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/) and [“Which GPU(s) to Get for Deep Learning,”](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) are incredibly informative and helped me understand the intricacies of GPU architectures. I think these blog posts are a must-read for anyone who plans to build their own rig.

The [Non-Interactive blog post about 8-GPU DL rig](https://nonint.com/2022/05/30/my-deep-learning-rig/) was the most incredible writeup on consumer-grade Deep Learning hardware I’ve seen on the entire internet! I wouldn’t dare to try out such a build, but it’s cool to see what a DL rig looks like at the absolute limits of consumer hardware.

My build design philosophy is heavily inspired by the PC builder [80ishplus](https://www.instagram.com/80ishplus/?hl=en). To quote him from one of [my favourite builds of his](https://pcpartpicker.com/b/8rcTwP):

> I think PC builders get caught fighting the false dichotomy of ascetic minimalism versus RGB maximalism. On the one hand, we have the Small Form Factor (SFF) community who love their aluminium shoeboxes, and on the other, the more vibrant PC builders creating EATX RGB behemoths. But the shoebox and the RGB behemoth are at their core the same in that they both seek to obscure their purpose from view. Functional exposition is subsumed either by perforated aluminium panels or distracted with flashing RGB lights and coloured coolant.


This is why I have no fancy RGB components, no RAM with LEDs, no flashy colorful parts. I let the form of the exposed components speak for themselves. Non-RGB components are also cheaper :)

## Components
Here is the [full list of my components from PC Part Picker](https://pcpartpicker.com/b/TQYgXL). I chose the cheapest components that fit my needs (that’s the advantage of knowing exactly which type of algorithms I plan to run on my machine). GPU prices are not listed on the page. However, I purchased both of them for $800 each, bringing the total cost of the build to $2954.

{{< figure src="iva_3.webp" caption="The full list of components on PC Part Picker: https://pcpartpicker.com/b/TQYgXL" >}}

### GPU
The choice was between the RTX 4090 and the RTX 3090. It was tempting to go for the RTX 4090 — its main advantage being the number of tensor cores and true FP8 capabilities, making it ideal for LLM applications and low-bit fine-tuning (which is still an active area of research, but I’m sure we will see more refined techniques it next year).

I opted for the RTX 3090s because they are cheaper (two of them are cheaper than one RTX 4090 and provide more performance). The RTX 3090 still allows me to run larger models with FP16 and 8-bit quantization, albeit significantly slower (akin to the speed of full 32-bit precision) due to hardware locks and a lack of tensor cores. That’s fine by me — once I’m certain my code works, I can always move my experiments to the cloud for some true FP8 performance. Besides, smaller models are becoming more and more capable thanks to better instruction tuning and pre-training data curation, so perhaps I won’t fine-tune any (multi-modal) LLMs larger than 13B in the near future.

The two RTX 3090 GPUs are just enough for my personal projects and experiments. I was looking specifically for the Founders Edition, because they look beautiful, and their cooling system, which exhausts to the rear, is well-suited for a multi-GPU setting.

### Motherboard
I was initially looking for motherboards with 4-slot spacing between two PCIe x16 slots so that my GPUs can have some breathing room, but they are all quite expensive. For example, the Asus ROG Crosshair X670 Hero that [Aleksa Gordić got for his build](https://www.youtube.com/watch?v=OWvy-fCWTBQ) is like… $670. The MSI Meg X570 Godlike, priced around $800, costs as much as a used RTX 3090! I didn’t want to pay so much for a motherboard just to get an extra 1-slot space between my GPUs, so I quickly ditched this approach.

An open-air case with a mining-like configuration with PCIe risers made more sense to me. So, I opted for the most affordable motherboard I could find with two PCIe x16 4.0 slots capable of operating in x8/x8 mode when two GPUs are plugged in. The MSI X570 Unify, at $199, fit the bill perfectly, despite not supporting RGB fans and accessories — a non-issue for me.

### CPU
I don’t plan to run CPU-heavy learning algorithms on this rig, so something like a Threadripper would be overkill. Moreover, with only 2 GPUs, I don’t need more than 20 PCI lanes for GPU connectivity. The previous-generation AMD Ryzen 9 5900X processor with 12 cores and 24 threads hits that sweet spot between price and performance. If I ever need more CPU power, I can always rent cloud computing resources temporarily.

### RAM and Storage
The more the better, and I don’t need top-tier speed for them. I’ve opted for 2x32GB DDR4–3600 RAM sticks and plan to upgrade to 4x32GB in the future. I’ve also chosen a couple of 2TB SSDs (NVME and SATA) and intend to add more SATA SSDs for training data storage as I find affordable options. Although these components aren’t the fastest on the market, they suffice for my needs and don’t bottleneck any training runs.

I’ve also considered going RAID 5 on the SATA SSDs, but at the end decided not to. I don’t think the added stability is justified, since the training data can be re-downloaded, more important data and models will be stored in the cloud anyways.

### PSU
A little-known fact about the RTX 3090 GPUs is that, despite their 350W maximum rating, their power draw sometimes spikes up to 600W for brief periods. Therefore, I chose an EVGA 1600W PSU that’s 80+ Gold certified.

The PSU is often overlooked in many builds and tutorials that I’ve seen on the internet. In my opinion, it’s one of the most important parts of any build. A good PSU not only saves money by converting power more effectively and producing less heat, but it also protects expensive equipment from overcurrent, short circuits, overvoltage, and more. I always look at [reviews of internal components](https://www.techpowerup.com/review/evga-supernova-g2-1600/4.html) and check if the PSU components come from trusted high-quality manufacturers. The EVGA PSU I chose features Nippon Chemi-Con capacitors from Japan. My go-to brands are Seasonic and EVGA.

I don’t think that Platinum/Titanium PSUs are justified — they are couple of hundred $$$ more expensive, and given $0.1/kWh in Washington state, I would need to run the rig constantly for a few years to make up the difference.

### Case Design and Cooling
Back in the days of the 10xx series, stacking multiple blower-style GPUs closely together was absolutely fine. However, with the 30xx and 40xx series, this isn’t feasible — they easily suffocate each other, leading to overheating and throttling. The lower GPU suffocates the upper one by either blowing hot air into it or leaving no room for fresh air intake. On top of that, putting the 3090s close together would make it impossible to cool the VRAMs on the backplate.

Watercooling is a viable solution for modern multi-GPU rigs, though it voids hardware warranties. However, the results are often worth it. For a multi-GPU setting, a fully-customized water loop is needed, which requires a lot of skills. Here is a perfect example of a multi-GPU water-cooling setup fromPC Part Picker: https://pcpartpicker.com/b/zrGqqs

I’m not skilled enough to build similar water loops (and, frankly, I don’t really want to maintain water loops). So, my next best option is to design an open-air system (like mining rigs) with a good amount of space between the GPUs. I just ordered a test bench made from aluminum profile rods and some spare parts, creating a fully-customized case. It brings me back to my childhood days of playing with Lego. Sadly, I forgot to take photos of the building process. Here are the only ones that I have:

{{< figure src="iva_41.webp,iva_42.webp" caption="Putting every nuts and bolts together. Was struggling to find the right configuration for GPU support bracket." >}}

The gigantic Noctua NH-D15 CPU cooler might have been a bit overkill, but just look at how beautiful it is… An all-black custom-made case, an all-black motherboard, with some bare metallic accents on the GPU, and white caps on the NH-D15 cooler to create an interesting contrast and drama. Ideally, I would like some white LEDs on the Chromax caps or on the RAM sticks (but then I will have to move the Noctua fans).

I’m a bit worried about the GPUs dumping heat on the motherboard. I plan to add some 120mm fans to force some more air flow through the whole system. For now, I’m pretty happy with the results. Here are some more photos of cable management between the GPUs and my robot cat alongside my Deep Learning rig:

{{< figure src="iva_51.webp,iva_52.webp" >}}

The robot cat’s name is “Chappity”. I will definitely write a blog post about putting Multi-Modal LLMs into this bad boy in the future, so stay tuned :)

## Experience of buying old GPUs
Now that mining is over and 40xx series have released, the prices of 30xx series just dropped by half. I grabbed both of my GPUs for ~$800 each. I could have got them for even less (I saw Facebook Marketplace listings for $650), but it wasn’t in a condition that I liked and I don’t want to spend too much time cleaning and fixing the thermals.

It is a known fact by now that 3090 RTX have really bad thermals, especially on the VRAM side.

## I could have built a more powerful rig, but…
At the time of building this machine, my home country **Ukraine is still fighting to defend itself from the russian invasion.** Although I could afford a much more powerful machine with more and better GPUs with, I would rather send those couple of thousand dollars to those in need.

If you find this blog post helpful, please consider donating to [Come Back Alive fund](https://savelife.in.ua/en/). Just a few dollars might help them save a life.

