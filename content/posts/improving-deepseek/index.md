---
title: "Improving DeepSeek R1 in Math Olympiads"
url: "/improving-deepseek"
date: 2025-04-18T00:00:00+00:00
# weight: 1
# aliases: ["/first"]
tags: ["Projects", "LLM", "Reinforcement Learning"]
keywords: ["Reinforcement Learning", "RLHF", "Machine Learning", "RLVR", "AIME", "AIME 2025", "AIME 2024", "DeepSeek", "R1", "DeepSeek-R1", "DeepSeek R1", "GRPO", "REINFORCE++", "Light-R1", "NuminaMath", "AIMO", "AIMO2", "Kaggle", "Reasoning", "LLM", "Math Olympiads"]
author: "Kha Vu Chan"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: true
hidemeta: false
comments: true
disqus_identifier: hav4ik/learning-to-rank
summary: "We trained a family of math reasoning models, 7B and 14B, finetuned with SFT and GRPO from DeepSeek-Distill-R1 models. Our 14B model achieves **75.8%** Maj@32 accuracy on AIME’25 (**+8.7%** improvement), surpassing twice larger DeepSeek-R1-Distill-32B. Our 7B model achieves **65.8%** Maj@32 (**+7.5%** improvement), comparable to DeepSeek-R1-Distill-14B. A single end-to-end training run (SFT + GRPO) of our 14B model costs less than $800."
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
    alt: "Web Search" # alt text
    caption: "Web Search" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide in list view
# editPost:
#     URL: "https://github.com/hav4ik/hav4ik.github.io/content"
#     Text: "Suggest Changes" # edit text
#     appendFilePath: true # to append file path to Edit link
---

> Contributors: [**Geremie Yeo**](https://www.linkedin.com/in/geremie-yeo/), [**Kelvin Soh**](https://www.linkedin.com/in/kelvin-soh/), **[Raja Biswas](https://www.linkedin.com/in/raja-biswas/)**, **[Chan Kha Vu](https://hav4ik.github.io/about/)**, [**Udbhav Bamba**](https://ubamba98.github.io).  
> *Our team are just enthusiasts doing things outside of our full-time jobs (unrelated to LLMs or Reasoning) and other commitments in life. Cloud compute costs are entirely self-funded.*

{{< figure src="featured.png" caption="Majority voting accuracy (Maj@32) on the uncontaminated AIME 2025 math olympiad of our models compared to DeepSeek R1 Distill models." invertible="false" >}}

This blog post is an extended and more personal version of [our team's short writeup on Kaggle](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573496). Here, you will find details that are normally never included in polished papers or tech reports &mdash; system design, failed experiments, and engineering trics.

-------------------

## Preface

Mathematical Olympiads hold a special place in my heart. In high school, I competed in math and programming contests, formed friendships that have lasted decades, and eventually moved to the U.S., building a career through the connections I made. A decade later, it still feels surreal that I now improve LLMs to solve olympiad-level problems &mdash; as a side hobby project.

I’m deeply grateful to the best teammates I could’ve asked for &mdash; especially [Geremie Yeo](https://www.linkedin.com/in/geremie-yeo/), [Kelvin Soh](https://www.linkedin.com/in/kelvin-soh/), and [Raja Biswas](https://www.linkedin.com/in/raja-biswas/). They did the heavy lifting, and I feel incredibly lucky to have learned so much from them. This blog post is an extended and more personal version of [our team's short writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573496), with more focus on technical details and additional analysis of successful and failed experiments.


## Introduction

Abstract mathematical reasoning has long been seen as a hallmark of human intelligence. Top performers in prestigious competitions like the International Math Olympiad (IMO) are celebrated as geniuses &mdash; and rightfully so. Solving olympiad-level problems demands not just sharp analytical thinking, but also creativity, deep intuition, and imagination.

### AIME Problems

Take the [American Invitational Mathematics Examination (AIME)](https://en.wikipedia.org/wiki/American_Invitational_Mathematics_Examination), for example. It's a selective contest for students who place in the top 2.5% on the [AMC 12](https://en.wikipedia.org/wiki/American_Mathematics_Contest), and top AIME scorers are invited to the [USAMO](https://en.wikipedia.org/wiki/United_States_of_America_Mathematical_Olympiad). Here’s one of the problems:

> **AIME 2025 I, problem 13.** Alex divides a disk into four quadrants with two perpendicular diameters intersecting at the center of the disk. He draws \( 25 \) more lines segments through the disk, drawing each segment by selecting two points at random on the perimeter of the disk in different quadrants and connecting those two points. Find the expected number of regions into which these \( 27 \) line segments divide the disk.

*Please pause here for at least 5 minutes. Grab a pen and paper &mdash; try solving it! This fun little problem has more subtlety than it seems 😜. The correct human and AI solutions are included at the end of this post.*

Due to their complexity, AIME problems have become a standard hard benchmark for LLM reasoning. Just a year ago, any improvement on AIME 2024 performance would make headlines. The best open-weight models, like [DeepSeek Math](https://arxiv.org/abs/2402.03300), could only score 3% &mdash; or 10% with external tools. High-quality human-written solutions are scarce online, and for a while, it felt like real progress in mathematical reasoning would require hundreds of thousands (if not millions) of dollars to produce a usable dataset.


### Reasoning models

The release of DeepSeek R1 &mdash; a model comparable to OpenAI’s o1 &mdash; alongside a paper detailing its full algorithmic approach, sparked a leap in the reasoning abilities of open-source models. Interestingly, the authors hinted that the distilled R1 models could be further improved through Reinforcement Learning.

{{< figure src="deepseek_paper_hint.png" caption="Section 3, page 14 of the DeepSeek R1 paper. The authors basically left the further model improvement “as an excercise to the reader.” This paragraph also confirm effectiveness of SFT from reasoning traces of larger model." invertible="true" >}}

Early reproduction efforts like [DeepScaleR](https://www.notion.so/19681902c1468005bed8ca303013a4e2?pvs=21) by the [Agentica](https://agentica-project.com/) team suggest that fine-tuning models like DeepSeek R1 might be far more affordable than expected. With even better data and a even smarter training curriculum, could we improve the distilled R1 models on a budget? That question led me to enter the [Kaggle's AIMO2 competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/overview).

### AIMO Prize

[AIMO](https://aimoprize.com) is a prestigious competition hosted on Kaggle with a total prize pool of $2’000’000, designed to push the frontier of open-source reasoning models. The difficulty of the problems are around the National Olympiad level. The problems have also been designed to be 'AI hard' in terms of the mathematical reasoning required, which was tested against open LLMs' capabilities (as of October 2024). Here is one of the [10 publicly available reference problems](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/data?select=AIMO_Progress_Prize_2_Reference_Problems_Solutions.pdf):

> **AIMO2 reference problem.** Let \( ABC \) be a triangle with \( BC=108 \), \( CA=126 \), and \( AB=39 \). Point \( X \) lies on segment \( AC \) such that \( BX \) bisects \( \angle CBA \). Let \( \omega \) be the circumcircle of triangle \( ABX \). Let \( Y \) be a point on \( \omega \) different from \( X \) such that \( CX=CY \). Line \( XY \) meets \( BC \) at \( E \). The length of the segment \( BE \) can be written as \( \frac{m}{n} \), where \( m \) and \( n \) are coprime positive integers. Find \( m+n \).

*Again, I highly encourage you to pause here for at least 5 minutes, grab a pen and a piece of paper, and try to solve it. Correct human solution and AI solution will be provided at the end of this post.*


--------------------------------------------


## A few things about GRPO

For our main RL method, we followed the DeepSeek R1 paper and used GRPO &mdash; a lightweight variant of policy gradient optimization introduced in [DeepSeek Math](https://arxiv.org/abs/2402.03300). While arguably less effective than PPO, GRPO removes the critic network, cutting memory usage and training time by roughly half. If you're new to RL or GRPO, I highly recommend this explainer: [*A Vision Researcher's Guide to Some RL Stuff: PPO & GRPO*](https://yugeten.github.io/posts/2025/01/ppogrpo/).

{{< figure src="grpo.png" caption="Comparison between PPO and GRPO, taken from [DeepSeek Math](https://arxiv.org/abs/2402.03300) paper. The Value model (also called the Critic) is omitted, and the advantage is calculated from group statistics rather than per-sample scores." invertible="true" >}}

While I was writing this blog post, Nathan Lambert published an excellent post analyzing GRPO, its main pitfalls, and ways to address them: [*Recent reasoning research: GRPO tweaks, base model RL, and data curation*](https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo). During AIMO2, our team was already aware of these issues and we applied techniques from several papers to mitigate them.


### Length Bias

Since our goal was to train an effective reasoning model, we quickly identified generation length as a major issue with SFT models distilled from DeepSeek-R1 (671B). The longer we trained, the longer the model's generated solutions became.

But why are reasoning chains so long? Is verbosity an emergent property of reasoning models? As it turns out, not necessarily. The [REINFORCE++](https://arxiv.org/html/2501.03262v1) paper showed that "length hacking" &mdash; as they called it &mdash; is a GRPO-specific issue. When comparing PPO, GRPO, RLOO, and REINFORCE++, they found that other algorithms achieved higher reward gains per unit of KL divergence without the same explosion in output length. That said, GRPO did show faster convergence.

Fortunately, right before the final week of AIMO2 and just as we began our last GRPO runs, two new papers dropped that tackled the length problem head-on: [DAPO](https://dapo-sia.github.io/) and [Dr. GRPO](https://arxiv.org/abs/2503.20783).

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
- **Light R1.** Later, we added a subset of from [**Light-R1**](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData) stage 2 data and removed duplicates with our dataset (after removing CoT with more than 16K tokens). We filter them further filter by difficulty, resulting in around 2K samples.

While sampling our dataset, we purposefuly avoided the following data sources:

- We purposefully avoided the `cn_k12` data source as it has much lower difficulty. Our team member [Raja Biswas](https://www.linkedin.com/in/raja-biswas/) found that these examples harms performance on harder problems.
- We avoided synthetic math datasets like [Orca-Math](https://arxiv.org/abs/2402.14830), because such datasets are usually created by weaker LLMs with weaker solution correctness validators. Such datasets are only useful for training a new reasoning model from scratch (i.e. from a non-reasoning one), not for finetuning an already strong reasoning model.

Our final dataset for SFT consists of over 10K samples (8K that we filtered from [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) and 2K from [Light-R1](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData)). The harder half of the dataset is then used for the GRPO stage. Dataset curation was led by [Raja Biswas](https://www.linkedin.com/in/raja-biswas/) and [Kelvin Soh](https://www.linkedin.com/in/kelvin-soh/), with some additional help from [Udbhav Bamba](https://www.linkedin.com/in/ubamba98/) in initial difficulty filtering.


### Stage 1: Supervised Fine Tuning

Our SFT models were fine-tuned courtesy of [Kelvin Soh](https://www.linkedin.com/in/kelvin-soh/), [Raja Biswas](https://www.linkedin.com/in/raja-biswas/), and [Geremie Yeo](https://www.linkedin.com/in/geremie-yeo/), who trained them on an 8xH100 node for 6 epochs at 16K context length. We used [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) and [7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) as base models. While [some teams](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/571252) found that longer fine-tuning can further improve accuracy, it often leads to unnecessarily long CoTs.


### Stage 2: Reinforcement Learning

During the Reinforcement Learning process, we tried to steer our model's behavior toward shorter reasoning CoTs. We spent considerable time on 7B models, hoping our findings would translate to 14B, only to relearn [the bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) repeatedly &mdash; good tricks at smaller scales don't always work at larger ones.

- For the 7B model, we performed GRPO in two stages: first on 8K context, then on 16K context. We found LoRA converged faster than FFT while being more VRAM-efficient. We used DAPO's clipping and online sample filtering, and length penalty worked well for 7B.
- For 14B, length penalty severely hurts accuracy, so we removed it. Also, training on much shorter contexts significantly reduced accuracy at intended inference lengths. Model merging helped regain some accuracy, so our final submission is a merged SFT and GRPO on 6K context. We had another 14B GRPO on 16K context trained on the last day of the AIMO2 competition. Our 14B GRPO models were all trained on a single 8xH200 node.

Our GRPO models were trained courtesy of [Geremie Yeo](https://www.linkedin.com/in/geremie-yeo/), [Kelvin Soh](https://www.linkedin.com/in/kelvin-soh/), [Raja Biswas](https://www.linkedin.com/in/raja-biswas/), and [Chan Kha Vu](https://hav4ik.github.io/about/). During the final week of the AIMO2 competition, each of us ran GRPO experiments independently. Every day, we checked in on each other’s progress, compared results, shared what worked, and built on each other’s findings. It was super fun &mdash; and rewarding (pun intended)!

Interestingly, while my teammates used [Open-R1](https://github.com/huggingface/open-r1) with a [faster version of trl’s GRPOTrainer](https://github.com/nhannguyen2709/open-r1) by Kaggle user [@andy2709](https://www.kaggle.com/andy2709), I maintained an [active fork of veRL](https://github.com/hav4ik/verl/tree/dapo-lora) with DAPO, FSDP-LoRA, and Dr. GRPO integrated. Both frameworks supported long-context RL with sequence packing, Ulysses, hybrid trainers with model colocation for memory savings, vLLM rollouts, and more.

### Model Merging

We found that for 7B models, merging increases the overall performance: the merged model surpasses both the SFT and all GRPO checkpoints by accuracy and token economy. However, when we moved to 14B, merging becomes more of a compromise.


## Evaluation

For evaluation, we used AIME 2025 (released in March 2025) as an uncontaminated benchmark &mdash; it was published after DeepSeek R1 was trained, and our only data source, [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5), was collected beforehand. Since our main motivation was AIMO2 competition, we also included the [10 “AI-hard” reference problems](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/data?select=AIMO_Progress_Prize_2_Reference_Problems_Solutions.pdf) in our local validation set. Our local benchmark, referred to as **CV**, consists of 40 problems and was used to track progress after each development stage. Evaluations below were conducted by [Chan Kha Vu](https://hav4ik.github.io/about/).


### Setting

Reasoning traces are sampled with `BFloat16` precision, `temperature=0.75`, and `top_p=0.95`. We collect 64 traces per question with `max_len=32768`. Length and `Pass@1` metrics were averaged across 64 rollouts per question. For aggregated metrics like `Maj@K` (majority voting accuracy), we sampled `K` traces per problem from our pool of 64 traces. We repeated this process 16 times and reported the average to reduce noise.

You can reproduce our evals using our evaluation code [aime25-aimo2-evals](https://www.kaggle.com/code/chankhavu/aime25-aimo2-evals) and our dataset of collected reasoning traces [reasoning-traces-aime25-aimo25](https://www.kaggle.com/datasets/chankhavu/reasoning-traces-aime25-aimo25).

### 14B models

- **Merged-14B:** the model we submitted to the Private LB of the AIMO2 competition for both of our final submissions. It's a merge of several SFT and GRPO with 6K context checkpoints.
- **Last-GRPO-14B:** trained in the final days of the AIMO2 competition. It showed worse results on our local End-to-End whole pipeline validation, despite being better on academic-style benchmarking settings, so we never submitted it to LB.

Below are majority voting metrics with generation `max_len` set at 12800 and 32768 tokens:

| Token budget | Model Name | CV Pass@1 | CV Maj@32 | AIME'25 Pass@1 | AIME'25 Maj@32 | Average length |
| --- | --- | --- | --- | --- | --- | --- |
| 12800 | DeepSeek-R1-Distill-14B | 0.412 | 0.613 | 0.41 | 0.648 | 9473.66 |
|  | Light-R1-14B-DS | 0.442 | 0.664 | 0.45 | 0.671 | 9787.01 |
|  | **Our Merged 14B** | 0.477 | **0.747** | 0.455 | **0.731** | **9251.56** |
|  | **Our Last GRPO 14B** | **0.482** | 0.736 | **0.468** | **0.738** | 9312.03 |
| 16384 | DeepSeek-R1-Distill-14B | 0.449 | 0.664 | 0.447 | 0.671 | 10910.2 |
|  | Light-R1-14B-DS | 0.498 | 0.713 | 0.502 | 0.731 | 11432.1 |
|  | **Our Merged 14B** | 0.525 | 0.759 | 0.504 | 0.746 | **10552.1** |
|  | **Our Last GRPO 14B** | **0.541** | **0.762** | **0.521** | **0.758** | **10511.6** |

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
| 12800 | DeepSeek-R1-Distill-7B | 0.345 | 0.55 | 0.353 | 0.562 | **9553.74** |
|  | Light-R1-7B-DS | 0.36 | 0.606 | 0.371 | 0.606 | 9831.07 |
|  | **Our Final SFT 7B** | 0.379 | 0.627 | 0.374 | 0.604 | 9751 |
|  | **Our Merged 7B** | **0.383** | **0.658** | **0.38** | **0.637** | **9583.42** |
| 16384 | DeepSeek-R1-Distill-7B | 0.368 | 0.58 | 0.377 | 0.583 | **11104.1** |
|  | Light-R1-7B-DS | 0.391 | 0.611 | 0.401 | 0.631 | 11511.7 |
|  | **Our Final SFT 7B** | 0.412 | 0.678 | 0.398 | **0.658** | 11449.6 |
|  | **Our Merged 7B** | **0.422** | **0.686** | **0.409** | **0.654** | **11145.9** |

It's easier to visualize the test-time scaling economy of our models with the following plot:

{{< figure src="tts_7b_aime25.png" caption="Test-time scaling economy of our final merged 7B model. Each “dot” represents accuracy results on AIME 2025 when total token budget is set to one of the following: 8192, 9000, 12800, 16384. Our model reaches the peak Maj@32 of the base DeepSeek-R1-7B 33% faster, and have comparable peak Maj@32 to Light-R1." invertible="true" >}}

{{< figure src="tts_7b_cv.png" caption="Test-time scaling economy of our final merged 7B model. Each “dot” represents accuracy results on our 40 questions **CV** set (AIME'25 + AIMO-2) when total token budget is set to one of the following: 8192, 9000, 12800, 16384. Our model reaches the peak Maj@32 of the base DeepSeek-R1-7B 33% faster, and have better peak Maj@32 to Light-R1." invertible="true" >}}


--------------------------------------------


