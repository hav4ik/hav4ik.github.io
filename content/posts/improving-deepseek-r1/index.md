---
title: "Improving DeepSeek R1 in Math"
url: "/improving-deepseek"
date: 2025-04-18T00:00:00+00:00
# weight: 1
# aliases: ["/first"]
tags: ["LLM", "Reasoning", "Reinforcement Learning"]
keywords: ["Reinforcement Learning", "RLHF", "Machine Learning", "RLVR", "AIME", "AIME 2025", "AIME 2024", "DeepSeek", "R1", "DeepSeek-R1", "DeepSeek R1", "GRPO", "REINFORCE++", "Light-R1", "NuminaMath", "AIMO", "AIMO2", "Kaggle", "Reasoning", "LLM", "Math Olympiads"]
author: "Kha Vu Chan"
# author: ["Kha Vu Chan", "Geremie Yeo", "Kelvin Soh", "Raja Biswas", "Udbhav Bamba"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: true
disqus_identifier: hav4ik/improving-deepseek-r1
summary: "I joined a team and we trained 7B and 14B math reasoning models based on DeepSeek-R1-Distill using SFT and GRPO. Our 14B model achieved **75.8%** Maj@32 on AIME’25 (**+8.7%** improvement), and our 7B model reached **65.8%** Maj@32 (**+7.5%**). Here is what I've learned."
# canonicalURL: "https://canonical.url/to/page"
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
    image: "featured.png" # image path/url
    alt: "We trained 14B and 7B reasoning model surpassing DeepSeek R1 models twice their size in math olympiads" # alt text
    caption: "We trained 14B and 7B reasoning model" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide in list view
# editPost:
#     URL: "https://github.com/hav4ik/hav4ik.github.io/content"
#     Text: "Suggest Changes" # edit text
#     appendFilePath: true # to append file path to Edit link
---

> Model contributors: [**Geremie Yeo**](https://www.linkedin.com/in/geremie-yeo/), [**Kelvin Soh**](https://www.linkedin.com/in/kelvin-soh/), **[Raja Biswas](https://www.linkedin.com/in/raja-biswas/)**, **[Chan Kha Vu](https://hav4ik.github.io/about/)**, [**Udbhav Bamba**](https://ubamba98.github.io).  
> *Our team are just enthusiasts doing things outside of our full-time jobs (unrelated to LLMs or Reasoning) and other commitments in life. Cloud compute costs are entirely self-funded.*

{{< figure src="cover.png" caption="Majority voting accuracy (Maj@32) on the uncontaminated AIME 2025 Math Olympiad, comparing our models to the DeepSeek-R1-Distill baselines. See the 'Evaluations' section for details on how this click-baity chart was produced." invertible="false" >}}

This blog post is a more personal and extended version of [our team’s short write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573496), with a stronger focus on what I’ve learned through the process. Here, you’ll find the kinds of details that rarely make it into polished papers or tech reports &mdash; system design, failed experiments, engineering details, and other discussions. You’ll also notice some awkward switches between “we” and “I” as I alternate between describing team efforts and my own experiments and thoughts.


-------------------


## Preface

Mathematical Olympiads hold a special place in my heart. In high school, I competed in math and programming contests, formed friendships that have lasted decades, and eventually moved to the U.S., building a career through the connections I made. A decade later, it still feels surreal that I now improve LLMs to solve olympiad-level problems &mdash; as a side hobby project.

I’m deeply grateful to the best teammates I could’ve asked for &mdash; especially [Geremie Yeo](https://www.linkedin.com/in/geremie-yeo/), [Kelvin Soh](https://www.linkedin.com/in/kelvin-soh/), and [Raja Biswas](https://www.linkedin.com/in/raja-biswas/) who invited me to the team. They did most of the heavy lifting work. I feel incredibly lucky to have learned so much from them. This blog post is an extended and more personal version of [our team's short writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573496), with more focus on technical details and analysis of our GRPO experiments.


## Introduction

Abstract mathematical reasoning has long been seen as a hallmark of human intelligence. Top performers in prestigious competitions like the International Math Olympiad (IMO) are celebrated as geniuses &mdash; and rightfully so. Solving olympiad-level problems demands not just sharp analytical thinking, but also creativity, deep intuition, and imagination.

### AIME Problems

Take the [American Invitational Mathematics Examination (AIME)](https://en.wikipedia.org/wiki/American_Invitational_Mathematics_Examination), for example. It's a selective contest for students who place in the top 2.5% on the [AMC 12](https://en.wikipedia.org/wiki/American_Mathematics_Contest), and top AIME scorers are invited to the [USAMO](https://en.wikipedia.org/wiki/United_States_of_America_Mathematical_Olympiad). Here’s one of the problems:

> **AIME 2025 I, problem 13.** Alex divides a disk into four quadrants with two perpendicular diameters intersecting at the center of the disk. He draws \( 25 \) more lines segments through the disk, drawing each segment by selecting two points at random on the perimeter of the disk in different quadrants and connecting those two points. Find the expected number of regions into which these \( 27 \) line segments divide the disk.

*Please pause here for at least 5 minutes. Grab a pen and paper &mdash; try solving it! This fun little problem has more subtlety than it seems 😜. Answer is provided at the end of this blog post.*

Due to their complexity, AIME problems have become a standard hard benchmark for LLM reasoning. Just a year ago, any improvement on AIME 2024 performance would make headlines. The best open-weight models, like [DeepSeek Math](https://arxiv.org/abs/2402.03300), could only score 3% &mdash; or 10% with external tools. High-quality human-written solutions are scarce online, and for a while, it felt like real progress in mathematical reasoning would require hundreds of thousands (if not millions) of dollars to produce a usable dataset.


### Reasoning models

The release of [DeepSeek R1](https://arxiv.org/abs/2501.12948) &mdash; a model comparable to OpenAI’s o1 &mdash; alongside a paper detailing its full algorithmic approach, sparked a leap in the reasoning abilities of open-source models. Interestingly, the authors hinted that the distilled R1 models could be further improved through Reinforcement Learning.

{{< figure src="deepseek_paper_hint.png" caption="Section 3, page 14 of the [DeepSeek R1](https://arxiv.org/abs/2501.12948) paper. The authors basically left the further model improvement “as an excercise to the reader.” This paragraph also confirm effectiveness of SFT from reasoning traces of larger model." invertible="true" >}}

Early reproduction efforts like [DeepScaleR](https://www.notion.so/19681902c1468005bed8ca303013a4e2?pvs=21) by the [Agentica](https://agentica-project.com/) team suggest that fine-tuning models like the Distilled DeepSeek R1 might be far more affordable than expected. With even better data and a even smarter training curriculum, could we improve the distilled R1 models on a budget? That question led me to enter the [Kaggle's AIMO2 competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/overview).

### AIMO Prize

[AIMO](https://aimoprize.com) is a prestigious competition hosted on Kaggle with a total prize pool of $2’000’000, designed to push the frontier of open-source reasoning models. The difficulty of the problems are around the National Olympiad level. The problems have also been designed to be 'AI hard' in terms of the mathematical reasoning required, which was tested against open LLMs' capabilities (as of October 2024). Here is one of the [10 publicly available reference problems](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/data?select=AIMO_Progress_Prize_2_Reference_Problems_Solutions.pdf):

> **AIMO2 reference problem 2.** Let \( ABC \) be a triangle with \( BC=108 \), \( CA=126 \), and \( AB=39 \). Point \( X \) lies on segment \( AC \) such that \( BX \) bisects \( \angle CBA \). Let \( \omega \) be the circumcircle of triangle \( ABX \). Let \( Y \) be a point on \( \omega \) different from \( X \) such that \( CX=CY \). Line \( XY \) meets \( BC \) at \( E \). The length of the segment \( BE \) can be written as \( \frac{m}{n} \), where \( m \) and \( n \) are coprime positive integers. Find \( m+n \).

*Again, I highly encourage you to pause here for at least 5 minutes, grab a pen and a piece of paper, and try to solve it. Answer is provided at the end of this blog post.*


### What makes a problem 'AI hard'?

By analyzing failure cases on medium-level math problems from AIME, we noticed that reasoning models often struggle with problems that have multiple corner cases and Geometry problems (due to the lack of visual understanding). My teammate [Geremie Yeo](https://www.linkedin.com/in/geremie-yeo/) wrote a great analysis on this: [AI outputs Wrong Answers due to Corner Cases for Medium-Level Math Problems](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/571769).

I find that many of the model’s mistakes mirror the kinds of errors humans make too. Let’s look at the following problem as an example:

> **AIME 2025 II, problem 7.** Let \( A \) be the set of positive integer divisors of \( 2025 \). Let \( B \) be a randomly selected subset of \( A \). The probability that \( B \) is a nonempty set with the property that the least common multiple of its elements is \( 2025 \) is \( \frac{m}{n} \), where \( m \) and \( n \) are relatively prime positive integers. Find \( m + n \).

*Let's pause here and try to solve it. Did you make the same mistake that the AI made? Answer is provided at the end of this blog post.*

The `DeepSeek-R1-Distill-14B` model often misinterprets the problem and assumes that \( A \) does not contain the empty set, resulting in the answer of \(m = 27904 \) and \( n = 32767 \) (60671 modulo 1000 = 671). I find this mistake to be quite cute.


--------------------------------------------


## A few things about GRPO

For our main RL method, we followed the DeepSeek R1 paper and used GRPO &mdash; a lightweight variant of policy gradient optimization introduced in [DeepSeek Math](https://arxiv.org/abs/2402.03300). While arguably less effective than PPO, GRPO removes the critic network, cutting memory usage and training time by roughly half. If you're new to RL or GRPO, I highly recommend this explainer: [*A Vision Researcher's Guide to Some RL Stuff: PPO & GRPO*](https://yugeten.github.io/posts/2025/01/ppogrpo/).

{{< figure src="grpo.png" caption="Comparison between PPO and GRPO, taken from [DeepSeek Math](https://arxiv.org/abs/2402.03300) paper. The Value model (also called the Critic) is omitted, and the advantage is calculated from group statistics rather than per-sample scores." invertible="true" >}}

While I was writing this blog post, Nathan Lambert published an excellent post analyzing GRPO, its main pitfalls, and ways to address them: [*Recent reasoning research: GRPO tweaks, base model RL, and data curation*](https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo). During the model training period, our team was already aware of these issues and we applied techniques from several papers to mitigate them.


### Length Bias

Since our goal was to train an effective reasoning model, we quickly identified generation length as a major issue with SFT models distilled from DeepSeek-R1 (671B). The longer we trained, the longer the model's generated solutions became.

But why are reasoning chains so long? Is verbosity an emergent property of reasoning models? As it turns out, not necessarily. The [REINFORCE++](https://arxiv.org/html/2501.03262v1) paper showed that "length hacking" &mdash; as they called it &mdash; is a GRPO-specific issue. When comparing PPO, GRPO, RLOO, and REINFORCE++, they found that other algorithms achieved higher reward gains per unit of KL divergence without the same explosion in output length. That said, GRPO did show faster convergence.

Fortunately, right just as we began our last GRPO runs, two new papers dropped that tackled the length problem head-on: [DAPO](https://dapo-sia.github.io/) and [Dr. GRPO](https://arxiv.org/abs/2503.20783).

{{< figure src="dr_grpo.png" caption="[DAPO](https://dapo-sia.github.io/) and [Dr. GRPO](https://arxiv.org/abs/2503.20783) papers independently found that there is an implicit length bias in the [original GRPO formulation](https://arxiv.org/abs/2402.03300): longer incorrect solutions gets penalized less (token-wise) than shorter incorrect ones. The solution? Get rid of the per-sample loss normalization term completely!" invertible="true" >}}

It turns out GRPO’s length bias is largely an implementation issue. If you follow the [original GRPO formulation](https://arxiv.org/abs/2402.03300) and average losses *within* each sample before aggregating across the group, you will end up in a situation where tokens within longer sequences in a group may have a disproportionately lower contribution to the overall loss. The fix is simple: remove per-sample normalization and compute the loss across all tokens uniformly.


### Difficulty Bias

The \( 1 / {\text{std}(\{R(q, o_1), \ldots, R(q, o_G)\})} \) term in the [original GRPO formulation](https://arxiv.org/abs/2402.03300) which is used to normalize advantages, introduces a bias toward very easy or very hard questions, with the outcome rewards being almost all \(1\) or \(0\). [Dr. GRPO](https://arxiv.org/abs/2503.20783) addresses this by removing reward scaling entirely, as illustrated above.

Interestingly, I learned about this *difficulty bias* even before Dr. GRPO, thanks to a great blog post by [@leroykun](https://x.com/leloykun), published just two weeks after DeepSeek R1’s release: [*GRPO's Main Flaw*](https://leloykun.github.io/ponder/grpo-flaw/).


### Do we even need KL?

Unlike PPO, GRPO uses the *forward* KL divergence \( D_{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) \), rather than the *reverse* KL \( D_{KL}(\pi_{\text{ref}} \,\|\, \pi_\theta) \) used in PPO. It also adopts [Schulman’s unbiased KL approximation](http://joschu.net/blog/kl-approx.html) instead of the exact KL from [OpenAI’s original RLHF paper](https://arxiv.org/pdf/2203.02155).

Twitter user [@kalomaze](https://x.com/kalomaze) reported worse results with forward KL, but in my experiments with `DeepSeek-R1-Distill-1.5B`, the difference was negligible &mdash; matching findings from [@danielhanchen](https://x.com/danielhanchen) ([source](https://x.com/danielhanchen/status/1892643424538595611)). The [DAPO](https://dapo-sia.github.io/) paper goes a step further and removes the KL term entirely, arguing that during long-CoT reasoning, the model diverges enough from its initial state that regularization is no longer helpful.


--------------------------------------------


## How we cooked our models

Given our budget, full reliance on reinforcement learning wasn’t feasible. A single [DeepScaleR-1.5B](https://www.notion.so/19681902c1468005bed8ca303013a4e2?pvs=21) training run costs around $5,000 for just a 10% gain over the base `DeepSeek-R1-Distill-1.5B`. Scaling that up to 7B or 14B models &mdash; 5× and 10× larger &mdash; would cost nearly a whole bitcoin.

For this reason, we decided to shift our focus on collecting high-quality data and performing SFT first, and later do RL on top to further enhance its reasoning abilities and steer its behavior. The [Light-R1](https://arxiv.org/abs/2503.10460) paper, published just 2 weeks before our last experiment, showed potential for SFT on high quality data, which confirms that our direction was correct.


### Dataset Curation

Our team focused on collecting solution traces under 16K tokens. The reasoning was simple: most correct DeepSeek R1 outputs are under 6K, and 16K struck a good balance between accuracy and compute cost for our 7B and 14B models according to our teseting.

- **Initial pool.** First, we filtered math word problems from [**NuminaMath-1.5**](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5), sourcing problems with topics in Algebra, Geometry, Number Theory, and Combinatorics from Olympiads, AoPS forums, AMC and AIME of previous years, olympiad references, and number theory sources. Our goal was to get harder problems in topics that DeepSeek models would likely struggle with.
- **Joining R1 traces.** We then joined with correct R1 reasoning traces from [**OpenR1-Math-220k**](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k). Together with previous step, this filtered 800K problems down to 27K.
- **Difficulty filtering.** To filter further, we sampled 8 solutions per problem with `max_len` of 8K tokens using [deepseek-r1-distill-qwen-7b-awq](https://huggingface.co/casperhansen/deepseek-r1-distill-qwen-7b-awq) and removed easy problems. We kept only problems with 7 or fewer correct solutions, leaving 8K problems. To get harder problems for GRPO, we used the 14B AWQ model for similar filtering.
- **Light R1.** Later, we added a subset of from [**Light-R1**](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData) stage 2 data and removed duplicates with our dataset (after removing CoT with more than 16K tokens). We then filter them further by difficulty, resulting in around 2K samples.

While sampling our dataset, we purposefuly avoided the following data sources:

- We purposefully avoided the `cn_k12` data source as it has much lower difficulty. Our team member [Raja Biswas](https://www.linkedin.com/in/raja-biswas/) found that these examples harms performance on harder problems.
- We avoided synthetic math datasets like [Orca-Math](https://arxiv.org/abs/2402.14830), because such datasets are usually created by weaker LLMs with weaker solution correctness validators. Such datasets are only useful for training a new reasoning model from scratch (i.e. from a non-reasoning one), not for finetuning an already strong reasoning model.

Our final dataset for SFT consists of over 10K samples (8K that we filtered from [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) and 2K from [Light-R1](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData)). The harder half of the dataset is then used for the GRPO stage. Dataset curation was led by [Raja Biswas](https://www.linkedin.com/in/raja-biswas/) and [Kelvin Soh](https://www.linkedin.com/in/kelvin-soh/), with some additional help from [Udbhav Bamba](https://www.linkedin.com/in/ubamba98/) in initial difficulty filtering.


### Stage 1: Supervised Fine Tuning

Our SFT models were fine-tuned courtesy of [Kelvin Soh](https://www.linkedin.com/in/kelvin-soh/), [Raja Biswas](https://www.linkedin.com/in/raja-biswas/), and [Geremie Yeo](https://www.linkedin.com/in/geremie-yeo/), who trained them on an 8xH100 node for 6 epochs at 16K context length. We used [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) and [7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) as base models. While [some teams](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/571252) found that longer fine-tuning can further improve accuracy, it often leads to unnecessarily long CoTs.


### Stage 2: Reinforcement Learning

During the Reinforcement Learning process, we tried to steer our model's behavior toward shorter reasoning CoTs. We spent considerable time on 7B models, hoping our findings would translate to 14B, only to relearn [the bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) repeatedly &mdash; good tricks at smaller scales don't always work at larger ones.

- For the 7B model, we performed GRPO in two stages: first on 8K context, then on 16K context. We found LoRA converged faster than FFT while being more VRAM-efficient. We used DAPO's clipping and online sample filtering, and length penalty worked well for 7B. We used outcome reward and DAPO's overlong penalty, which we found to be gentler than Cosine length penalty.
- For 14B, length penalty severely hurts accuracy, so we removed it, leaving only outcome reward. Also, training on much shorter contexts significantly reduced accuracy at intended inference lengths. Model merging helped regain some accuracy, so our final submission is a merged SFT and GRPO on 6K context. We had another 14B GRPO on 16K context trained on the last day of the AIMO2 competition. Our 14B GRPO models were all trained on a single 8xH200 node.

Our GRPO models were trained courtesy of [Geremie Yeo](https://www.linkedin.com/in/geremie-yeo/), [Kelvin Soh](https://www.linkedin.com/in/kelvin-soh/), [Raja Biswas](https://www.linkedin.com/in/raja-biswas/), and [Chan Kha Vu](https://hav4ik.github.io/about/). During the final week of the AIMO2 competition, each of us ran GRPO experiments independently. Every day, we checked in on each other’s progress, compared results, shared what worked, and built on each other’s findings. It was super fun &mdash; and rewarding (pun intended)!

Interestingly, while my teammates used [Open-R1](https://github.com/huggingface/open-r1) with a [faster version of trl’s GRPOTrainer](https://github.com/nhannguyen2709/open-r1) by Kaggle user [@andy2709](https://www.kaggle.com/andy2709), I maintained an [active fork of veRL](https://github.com/hav4ik/verl/tree/dapo-lora) with DAPO, FSDP-LoRA, and Dr. GRPO integrated. Both frameworks supported long-context RL with sequence packing, Ulysses, hybrid trainers with model colocation for memory savings, vLLM rollouts, and more.

### Model Merging

Our team used [MergeKit](https://github.com/arcee-ai/mergekit) to merge our final SFT and GRPO checkpoints using [TIES](https://arxiv.org/abs/2306.01708) method. We did not play around with any of the hyperparameters and just set all weights to 1 and density to 1. 

We found that for 7B models, merging increases the overall performance: the merged model surpasses both the SFT and all GRPO checkpoints by accuracy and token economy. However, when we moved to 14B, merging becomes more of a compromise.


## Evaluation

For evaluation, we used AIME 2025 (released in March 2025) as an uncontaminated benchmark &mdash; it was published after DeepSeek R1 was trained, and our only data source, [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5), was collected beforehand. Since our main motivation was AIMO2 competition, we also included the [10 “AI-hard” reference problems](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/data?select=AIMO_Progress_Prize_2_Reference_Problems_Solutions.pdf) in our local validation set. Our local benchmark, referred to as **CV**, consists of 40 problems and was used to track progress after each development stage. Evaluations below were conducted by [Chan Kha Vu](https://hav4ik.github.io/about/).


### Setting

Reasoning traces are sampled with `BFloat16` precision, `temperature=0.75`, and `top_p=0.95`. We collect 64 traces per question with `max_len=32768`. Length and `Pass@1` metrics were averaged across 64 rollouts per question. For aggregated metrics like `Maj@K` (majority voting accuracy), we sampled `K` traces per problem from our pool of 64 traces. We repeated this process 16 times and reported the average to reduce noise.

We used `stop=['</think>']` for our models, as they were trained with more restricted generation lengths and often produced correct answers before completing the full reasoning chain. Applying the same stopping condition to DeepSeek-R1 models led to noticeable accuracy drops, so their results below are reported using the default (stop at EOS) setting.

You can reproduce our evals using our evaluation code [aime25-aimo2-evals](https://www.kaggle.com/code/chankhavu/aime25-aimo2-evals) and our dataset of collected reasoning traces [reasoning-traces-aime25-aimo25](https://www.kaggle.com/datasets/chankhavu/reasoning-traces-aime25-aimo25).

### 14B models

- **Merged-14B:** the model we submitted to the Private LB of the AIMO2 competition for both of our final submissions. It's a merge of several SFT and GRPO checkpoints. We uploaded the AWQ of this model on Huggingface: [bogoconic1/aimo2-final-merged-model-14b](https://huggingface.co/bogoconic1/aimo2-final-merged-model-14b).
- **Last-GRPO-14B:** trained in the final days of the AIMO2 competition. It showed worse results than the merged model on our local End-to-End whole pipeline validation, despite being better on academic-style benchmarking settings, so we never submitted it to LB.

Below are majority voting metrics with generation `max_len` set at 12800 and 32768 tokens:

| Token budget | Model Name | CV Pass@1 | CV Maj@32 | AIME'25 Pass@1 | AIME'25 Maj@32 | Average length |
| --- | --- | --- | --- | --- | --- | --- |
| 12800 | DeepSeek-R1-Distill-14B | 0.412 | 0.613 | 0.41 | 0.648 | 9473 |
|  | Light-R1-14B-DS | 0.442 | 0.664 | 0.45 | 0.671 | 9787 |
|  | **Our Merged 14B** | 0.477 | **0.747** | 0.455 | **0.731** | **8921** |
|  | **Our Last GRPO 14B** | **0.482** | 0.736 | **0.468** | **0.738** | **8981** |
| 16384 | DeepSeek-R1-Distill-14B | 0.449 | 0.664 | 0.447 | 0.671 | 10910 |
|  | Light-R1-14B-DS | 0.498 | 0.713 | 0.502 | 0.731 | 11432 |
|  | **Our Merged 14B** | 0.525 | 0.759 | 0.504 | 0.746 | **10125** |
|  | **Our Last GRPO 14B** | **0.541** | **0.762** | **0.521** | **0.758** | **10071** |

We observe an interesting phenomenon: at 32K tokens budget, all models perform slightly worse on Maj@32 than with 16K tokens budget. The reason is, when given more thinking time the model tends to self-doubt, sometimes leading to wrong answers.

It's easier to visualize the test-time scaling economy of our models with the following plots:
 
{{< figure src="tts_14b_aime25.png" caption="Test-time scaling economy of our last 14B GRPO model. Each “dot” represents accuracy results on AIME 2025 when total token budget is set to one of the following: 8192, 9000, 12800, 16384, 20480. Our model reaches the peak Maj@32 of the base DeepSeek-R1-14B 33% faster, and have comparable peak Maj@32 to Light-R1." invertible="true" >}}

{{< figure src="tts_14b_cv.png" caption="Test-time scaling economy of our last 14B GRPO model. Each “dot” represents accuracy results on our 40 questions **CV** set (AIME'25 + AIMO-2) when total token budget is set to one of the following: 8192, 9000, 12800, 16384, 20480. Our model reaches the peak Maj@32 of the base DeepSeek-R1-14B 33% faster, and have better peak Maj@32 to Light-R1." invertible="true" >}}

It should be noted that we built on top of [Light-R1](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData) work by using a subset of their stage 2 data, so this comparison is done only to show how much we were able to improve upon our baselines. Light-R1 models performs slightly better than our models on AIME 2025 at much longer token budgets (24K and 32K) because it was trained on longer reasoning traces. However, our models are much more efficient and more accurate at shorter token budgets (up to 16K).


### 7B models

* **Final SFT 7B:** trained for 6 epochs on our final dataset mixture. Interestingly, it was much harder to make improvements on the 7B model than the 14B one.
* **Merged 7B:** a merge of our SFT and several GRPO checkpoints. We never submitted it to the LB as we moved on to 14B models. 7B GRPO was trained in 2 stages: with 8K then with 16K context.

| Token budget | Model name | CV Pass@1 | CV Maj@32 | AIME'25 Pass@1 | AIME'25 Maj@32 | Average length |
| --- | --- | --- | --- | --- | --- | --- |
| 12800 | DeepSeek-R1-Distill-7B | 0.345 | 0.55 | 0.353 | 0.562 | **9553** |
|  | Light-R1-7B-DS | 0.36 | 0.606 | 0.371 | 0.606 | 9831 |
|  | **Our Final SFT 7B** | 0.379 | 0.627 | 0.374 | 0.604 | 9482 |
|  | **Our Merged 7B** | **0.383** | **0.658** | **0.38** | **0.637** | **9303** |
| 16384 | DeepSeek-R1-Distill-7B | 0.368 | 0.58 | 0.377 | 0.583 | 11104 |
|  | Light-R1-7B-DS | 0.391 | 0.611 | 0.401 | 0.631 | 11511 |
|  | **Our Final SFT 7B** | 0.412 | 0.678 | 0.398 | **0.658** | 11104 |
|  | **Our Merged 7B** | **0.422** | **0.686** | **0.409** | **0.654** | **10778** |

It's easier to visualize the test-time scaling economy of our models with the following plot:

{{< figure src="tts_7b_aime25.png" caption="Test-time scaling economy of our final merged 7B model. Each “dot” represents accuracy results on AIME 2025 when total token budget is set to one of the following: 8192, 9000, 12800, 16384. Our model reaches the peak Maj@32 of the base DeepSeek-R1-7B 33% faster, and have comparable peak Maj@32 to Light-R1." invertible="true" >}}

{{< figure src="tts_7b_cv.png" caption="Test-time scaling economy of our final merged 7B model. Each “dot” represents accuracy results on our 40 questions **CV** set (AIME'25 + AIMO-2) when total token budget is set to one of the following: 8192, 9000, 12800, 16384. Our model reaches the peak Maj@32 of the base DeepSeek-R1-7B 33% faster, and have better peak Maj@32 to Light-R1." invertible="true" >}}


### Evals summary

To produce the chart at the beginning of this post, we used same sampling settings as described above (same prompts, temperature, and top_p). **Pass@1** and **Avg. Length** were measured with `max_len=32768` and averaged over 64 rollouts, while **Maj@32** was calculated with `max_len=16384` across 16 simulated runs.

Why use different `max_len` for these metrics? All models &mdash; especially the distilled DeepSeek-R1 &mdash; show worse Maj@32 at longer token budgets. The most likely reason is that with more room to think, models tend to hallucinate default or overconfident answers.  

**Bottom line:** better Pass@1 does *not* mean better Maj@32!

| Params | Model name | AIME'25 Pass@1 | AIME’25 Maj@32 | Avg. Length |
|:---:| :--- |:---:|:---:|:---:|
| 7B | DeepSeek-R1-Distill-Qwen-7B | 0.389 | 0.583 | 14069 |
| | **Ours 7B** *(Merged)* | **0.422** | **0.654** | **12211** |
| 14B+ | DeepSeek-R1-Distill-Qwen-14B | 0.485 | 0.671 | 13071 |
| | DeepSeek-R1-Distill-Qwen-32B | 0.51 | 0.723 | 12440 |
| | **Ours 14B** *(Final GRPO)* | **0.542** | **0.758** | **10731** |


I wasn’t able to reproduce the AIME’25 Pass@1 results for `DeepSeek-R1-Distill-Qwen-32B` reported by [MathArena](https://matharena.ai) (0.51 vs 0.59), though my results for the 14B and 1.5B models match theirs closely. The main difference between our settings is that they used a different prompt &mdash; one that hints the answer should be an integer &mdash; which could boost Pass@1. This prompting difference might have a bigger impact on the 32B model, which follows the system prompt better than smaller models.


--------------------------------------------


## GRPO Experiments

Now is the part that you came here for, and the main reason I wrote a separate blog post from [our team's short writeup on Kaggle](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573496). I will try to describe some interesting experiments — mostly failed ones, ugly ones, and sometimes relatively good ones. I will even discuss experiments that I don’t even know how to interpret due to the lack of ablation studies (we don’t have budget for ablations, every experiment is a YOLO experiment).

Now for the part you came for—and the main reason I wrote this blog post instead of just linking to [our team’s short write-up on Kaggle](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573496). I’m going to walk through some of our most interesting experiments &mdash; mostly failed, occasionally ugly, and sometimes surprisingly decent. I’ll even include a few that I still don’t know how to interpret. Every experiment was a YOLO experiment since we couldn’t afford proper ablations.

{{< figure src="grpo_is_hard_tweet.png" caption="Self-funding long context GRPO runs is the most dumb, but also rewarding things I’ve ever done as a hobby." invertible="false" >}}

### Engineering bits 1: which framework?

These days, every kid and their mom can run GRPO using libraries like [Unsloth](https://unsloth.ai/blog/r1-reasoning) or [TRL](https://huggingface.co/docs/trl/en/index) and show gains on GSM8K or MATH &mdash; where generations are under 1K tokens. But doing RL on 8K, 16K, or even 24K token reasoning traces? That’s a whole different beast.

After the DeepSeek-R1 paper dropped, research teams around the world scrambled to reproduce the results &mdash; most notably [Open-R1](https://github.com/huggingface/open-r1) using TRL. At the time, TRL’s `GRPOTrainer` had serious scaling issues: only one device per node could run actor rollouts, and there was no support for model colocation or offloading.

[DeepScaleR-1.5B-Preview](https://www.notion.so/19681902c1468005bed8ca303013a4e2?pvs=21) was the first open-source effort to successfully run RL on long-form reasoning (up to 24K tokens). It used [veRL](https://github.com/volcengine/verl), a battle-tested RL framework for LLMs. veRL supports everything needed for long-context RL: memory efficiency via FSDP and CPU offloading, long-sequence handling with Sequence Packing and Ulysses, and zero-redundancy training with full model collocation.

{{< figure src="frameworks_comparison.png" caption="Comparison table between HybridFlow (aka veRL framework) and other popular RLHF frameworks. This chart is now outdated &mdash; OpenRLHF now supports full collocation of all models, and TRL now supports most of the veRL features for long-context and zero-redundancy." invertible="true" >}}

Many of the latest research ideas &mdash; like [DAPO](https://dapo-sia.github.io/) &mdash; are built on top of veRL. That’s why I maintained an active fork during AIMO2: [github.com/hav4ik/verl/tree/dapo-lora](https://github.com/hav4ik/verl/tree/dapo-lora). I merged DAPO into the latest veRL release, added [LoRA support](https://github.com/volcengine/verl/pull/1127), fixed FSDP wrapping/offloading issues (planning to upstream these), and continuously pulled in new features from the main branch to stay on the cutting edge.

Meanwhile, my teammates [Geremie Yeo](https://www.linkedin.com/in/geremie-yeo/), [Kelvin Soh](https://www.linkedin.com/in/kelvin-soh/), and [Raja Biswas](https://www.linkedin.com/in/raja-biswas/) used [Open-R1](https://github.com/huggingface/open-r1) with a [faster `GRPOTrainer`](https://github.com/nhannguyen2709/open-r1) by [@andy2709](https://www.kaggle.com/andy2709), which addressed many of TRL’s earlier limitations. Around that time, TRL also added Ulysses support and Dr. GRPO techniques, making it a solid option for long-context GRPO on reasoning tasks.


### Engineering bits 2: training bottlenecks

Solving VRAM pressure with Hybrid Engine &mdash; collocating the Actor, Ref, and vLLM in the same memory space &mdash; is just the first step. The real pain of long-context GRPO shows up in GPU utilization. Below is a chart from one of my latest GRPO runs: a 14B model with 16K context, trained on an 8×H200 node. For each global step, we generate 256 samples (32 problems × 8 rollouts) and perform 4 optimization steps (with 96 rollouts per minibatch).

{{< figure src="vram_usage.png" caption="GPU memory allocation chart of a 14B model’s GRPO run with 16K tokens on a single 8×H200 node. Each global step includes a 'vLLM rollout' phase to collect trajectories, an 'offload' phase to release memory and prep the trainer, and a 'training' phase for optimization. Rollouts take the longest, and the idle gap &mdash; caused by waiting for the longest sequence &mdash; can be as long as the training phase itself." invertible="true" >}}

One way to reduce the “Idle GPUs” and “Offload” gap is to increase the number of problems and rollouts per step. To boost training time percentage-wise, we also reuse rollouts by splitting the global batch into smaller minibatches and performing multiple optimization steps—similar to [Data Echoing](https://arxiv.org/abs/1907.05550). In my setup, with 2 minibatches per step and each rollout reused twice (lagging the policy by up to 4 steps), I saw no performance drop compared to the standard setting.

There’s also a cleaner solution: asynchronous RL, like in the [DeepCodeR](https://www.notion.so/1cf81902c14680b3bee5eb349a512a51?pvs=21) project. But our team is GPU poor and can’t afford extra nodes 😭. All this Hybrid Engine and collocation magic? It’s the hustle of the GPU poor.


### Engineering bits 3: LoRA is surprisingly hard

- [**LoRA**](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) by itself is easy &mdash; it’s the most widely used fine-tuning method and usually the first thing people learn when diving into LLMs.
- **LoRA with FSDP1** is a different story. You have to carefully wrap layers and adapters to avoid `dtype` mismatches between `DTensor` objects and gradients. I highly recommend studying [Answer.AI’s FSDP-QLoRA](https://github.com/AnswerDotAI/fsdp_qlora) to see what it really takes to get this working.
- Now, throw **vLLM offloading** into the mix. To run RL efficiently, you need to train the Actor model with LoRA, while also retaining a Ref model for reward computation (so the model should be wrapped in such a way to allow both base model and LoRA model forward pass). Then, you have to merge the sharded `DTensor` weights and send them to a vLLM engine, which uses its own tensor parallelism and wrapping system. This complexity is exactly why the [GRPO with LoRA PR in veRL](https://github.com/volcengine/verl/pull/205) still isn’t merged.

I managed to resolve all the integration issues with LoRA, FSDP, and veRL, but the vLLM offloading challenge remains tricky. There's no clean way to avoid sudden VRAM spikes caused by asynchronous model merging overlapping with vLLM startup. Adding more `torch.distributed.barrier()` and `torch.empty_cache()` calls (with waits) might help, but at the cost of even more performance drop and I didn’t have time to debug that.

{{< figure src="lora_vram_spikes.png" caption="Offloading a 14B model with LoRA adapters to vLLM causes random VRAM spikes when model merging overlaps with vLLM startup." invertible="true" >}}

In the end, I just reduced memory usage on both the trainer and vLLM rollout sides. It's not elegant, but it worked &mdash; training was as fast (or faster) than full fine-tuning, and with lower memory usage, I could fit more samples per minibatch.


### Training hyperparameters

Given a fixed token-per-gradient-update budget, is it better to increase the number of prompts per batch (with fewer rollouts per prompt), or increase rollouts per prompt (with fewer prompts)? I still don’t have a definitive answer.

More rollouts per problem help with harder questions &mdash; you’re more likely to sample a good reasoning trace and get positive rewards. But fewer prompts per batch risks making the gradient biased to just a few problems, which hurts generalization.

In our setup, we kept the number of rollouts per problem between 8 and 16. I tried lowering it to 6 once, but saw a clear drop in performance. Prompts per batch ranged from 12 to 24, depending on the budget. Dropping below 10 consistently led to worse reward curves.

As for learning rate, we mostly stayed between `1e-6` and `4e-6`. I did try `5e-6` and higher on 1.5B models with 8K context, but the results were unstable. Since we reused batches twice (like [Data Echoing](https://arxiv.org/abs/1907.05550)), any bias in a small batch gets amplified &mdash; especially at higher learning rates &mdash; leading to immediate performance degradation.


### Language mixing problem

There is this meme by Andrej Karpathy, saying that a properly trained model through Reinforcement Learning will stop speaking English at some point because it has developed an internal strategy somewhere in the latent space, detached from normal words.

{{< figure src="karpathy_tweet.png" caption="Everyone gangsta until the math reasoning model starts speaking Chinese!" invertible="false" >}}

Well... we accidentally did the meme 😭. During the final week, some of my teammates who were using Open-R1 with TRL started seeing his 14B model generating wild chain-of-thoughts in the middle of their GRPO training run:

{{< figure src="langmix_1.png" caption="Turns out language mixing doesn’t always mean your model has unlocked a higher strategy. Most likely, it’s just as confused by the problem as you are 😅" invertible="true" >}}

We debugged the run together and quickly spotted a few things:

- KL divergence *and* gradient norms exploded right when language mixing began. Although we had gradient clipping in place, this pointed to something deeper going wrong in the training process.
- Interestingly, the issue only appeared in Open-R1 runs. I never saw it in my veRL runs &mdash; though I always had DAPO enabled and never tested without it.

Eventually, we traced the root cause to *hard problems* where all rollouts received a reward of 0. Without diverse learning signal, the model spirals into producing garbage or mixing languages. The fix? Add more easy and medium-level problems to stabilize training.

DAPO saved me from this issue entirely. Since it dynamically filters out too-easy and too-hard problems at each step—oversampling prompts and selecting a subset for training &mdash; it acts like an online curriculum. We basically have online adaptive difficulty tuning for free.

Bonus image: well, maybe we just had to let it cook… 🤷

{{< figure src="langmix_2.png" caption="Well, maybe “it” does have some sort of hidden math-solving strategy in its latents LMAO" invertible="false" >}}


### Is LoRA any good?

During AIMO2, I had a hypothesis I was eager to test. While exploring [DeepScaleR-1.5B-Preview](https://www.notion.so/19681902c1468005bed8ca303013a4e2?pvs=21), I noticed that its training mostly introduced low-rank changes to the linear layer weights &mdash; a pattern I documented in this post: [*Does reasoning exist in low rank?*](https://www.notion.so/Does-reasoning-exists-in-low-rank-Part-1-exploratory-analysis-19d105a2faeb801785bcddc5778a5d81?pvs=21).

That led me to an assumption: maybe a LoRA adapter could act as a form of regularization—guiding the model to discover low-rank "steering vectors" that promote shorter chains of thought. Sounds too good to be true? Definitely. But it was something I was trying to test out.

{{< figure src="low_rank_diff.png" caption="Analysis of a strong GRPO run &mdash; low-rank changes dominate even after 5K steps." invertible="true" >}}

Since we separated *skill acquisition* (SFT) from *behavior steering* (GRPO), I figured LoRA was worth a shot &mdash; it’s more memory-efficient than full fine-tuning (FFT), letting us train on longer CoTs with the same compute.

In my 14B GRPO experiments with 8K context (40 global steps = 160 optimization steps), LoRA showed much faster convergence than FFT:

{{< figure src="lora_vs_fft_14b.png" caption="WandB dashboard comparing FFT and LoRA GRPO runs on a 14B model with 8K-token CoTs. LoRA converges faster in this setting." invertible="true" >}}

That said, our best 14B GRPO model was still trained with FFT. There were more differences between the runs than just flipping the “enable LoRA” flag, and I didn’t have the budget for proper ablations &mdash; each 100–200 step GRPO run costs ~$200. So I can’t definitively say LoRA outperforms FFT. And of course, I can’t rule out bugs in my code either.


### The Bitter Lesson

Early on, I had high hopes for **iterative context lengthening**, a technique used in [DeepScaleR-1.5B](https://www.notion.so/19681902c1468005bed8ca303013a4e2?pvs=21), where models are trained on progressively longer sequences &mdash; 8K, then 16K, then 24K. I burned through a lot of cloud credits running 8K &mdash; context experiments, hoping for cheap accuracy gains &mdash; only to find that training on shorter contexts significantly hurt performance at longer inference lengths. Turns out I wasn’t alone. The [DeepCodeR](https://www.notion.so/1cf81902c14680b3bee5eb349a512a51?pvs=21) team ran into the exact same issue:

{{< figure src="deepcoder_bitter.png" caption="The DeepCodeR team discovered the same issue when applying iterative context lengthening to their 14B model. Sadly, this trick seems to work only on 1.5B models &mdash; not on 7B or 14B. Wish they had published this earlier 😭." invertible="true" >}}

To make the lesson even more bitter: training directly at 16K context yields the same solution-length shortening effect &mdash; but with better accuracy (both Pass@1 and Maj@32). So unless you’re planning to run inference at shorter lengths (e.g. 8K), there’s little value in starting with shorter contexts for GRPO training.

Here’s a fun &mdash; but completely useless and cash-burning &mdash; experiment. In the top-left chart below, the runs *approaching from below* were first trained at 8K, then extended to 16K. The run *descending from above* was trained at 16K from the beginning &mdash; and outperformed them all.

{{< figure src="bitter_14b.png" caption="Completely useless and cash-burning experiment. At least it shattered all of my hopes about iterative context lengthening on 14B models." invertible="true" >}}

The 16K-from-the-start run had a pretty aggressive length penalty &mdash; so aggressive that the model seemed to prioritize shorter solutions over actual correctness. Sounds useless, right? But here’s the twist: once the length reward balances out during training, the 16K model still ends up with better validation accuracy. (Ignore the overall reward scores &mdash; they’re not directly comparable across runs.)

The takeaway? Starting GRPO at 16K context doesn’t just match the length-saving effect &mdash; it does so *while preserving more accuracy*. If your target is long-context inference anyway, you might as well train for it from the start.


-------------------------------------------------


## Conclusion

We trained a family of math reasoning models, 7B and 14B, finetuned with SFT and GRPO from `DeepSeek-Distill-R1` models. Our 14B model achieves **75.8%** Maj@32 accuracy on AIME’25 (**+8.7%** improvement), surpassing twice larger `DeepSeek-R1-Distill-32B`. Our 7B model achieves **65.8%** Maj@32 (**+7.5%** improvement), comparable to `DeepSeek-R1-Distill-14B` &mdash; a model twice its size. A single end-to-end training run (SFT + GRPO) of our final 14B model costs less than $800.

Our main motivation was the [Kaggle AIMO2 competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/overview). The leaderboard was extremely noisy &mdash; teams placing 3rd to 5th (prize money range) used the base DeepSeek-R1-14B model with no fine-tuning, just clever inference tricks and lucky shots. One major challenge we faced was quantization, which erased a sizeable part of our training gains. Unlike other teams, we didn’t use early stopping on reasoning rollouts and other tricks in our final submission, though we did experiment with it earlier on and found no difference in local testing.

Although luck was not on our side and we didn't achieve the leaderboard results we hoped for this time due to reasons both outside and within our control (not related to model training), we will definitely get it on the next AIMO progress prize! See you at the bleeding edge of open-source reasoning models at AIMO3.


-------------------------------------------------


Answers to some of the math problems mentioned in this post:
* **AIME 2025 I, problem 13**: 204
* **AIMO2 reference problem 2**: 751
* **AIME 2025 II, problem 7**: 237
