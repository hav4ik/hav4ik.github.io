---
title: "Improving DeepSeek R1 in Math"
url: "/improving-deepseek"
date: 2025-04-18T00:00:00+00:00
# weight: 1
# aliases: ["/first"]
tags: ["Projects", "LLM", "Reinforcement Learning"]
keywords: ["Learning to Rank", "LTR", "Machine Learning", "Information Retrieval", "RecSys", "Unbiased Learning to Rank", "Unbiased LTR", "Counterfactual Learning to Rank", "Counterfactual LTR", "Online Learning to Rank", "Online LTR", "Search Engine", "Web Search", "Ranking"]
author: "Kha Vu Chan"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: true
hidemeta: false
comments: true
disqus_identifier: hav4ik/learning-to-rank
summary: "We trained a family of math reasoning models, 7B and 14B, finetuned with SFT and GRPO from DeepSeek-Distill-R1 models. Our 14B model achieves **75.8%** Maj@32 accuracy on AIME’25 (**+8.7%** improvement), surpassing twice larger DeepSeek-R1-Distill-32B. Our 7B model achieves **65.8%** Maj@32 (**+7.5%** improvement), comparable to DeepSeek-R1-Distill-14B. A single end-to-end training run (SFT + GRPO) of our 14B model costs less than $1000."
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

Mathematical Olympiads hold a special place in my heart. I spent my high school years competing in math and programming olympiads, forming friendships that have lasted decades, and eventually moving to the US where I built a career thanks to the connections I made. It still feels unbelievable that a decade later, I would be improving LLMs to solve olympiad-level math problems as a side hobby project.

I'd like to express my deepest gratitude to the best teammates I could have ever asked for, especially [Geremie Yeo](https://www.linkedin.com/in/geremie-yeo/), [Kelvin Soh](https://www.linkedin.com/in/kelvin-soh/), and [Raja Biswas](https://www.linkedin.com/in/raja-biswas/). They did most of the heavy lifting, and I feel lucky to have learned so much from them.

This blog post is an extended and more personal version of [our team's short writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573496), with more focus on technical details and additional analysis of our experiments, both successful and failed.

## Introduction

Abstract mathematical reasoning has always been attributed to human’s intelligence. Top performers in prestigious competitions such as IMO (International Math Olympiad) are celebrated as geniuses by our society, and rightfully so &mdash; being able to solve olympiad-level problems requires not only lightning fast analytical thinking, but also creativity, deep intuition, and imagination.

### AIME Problems

Take [American Invitational Mathematics Examination (AIME)](https://en.wikipedia.org/wiki/American_Invitational_Mathematics_Examination) for example. It is a selective test given to those who rank in the top 2.5% on the [AMC 12](https://en.wikipedia.org/wiki/American_Mathematics_Contest) high school math exams, and top performers on AIME are invited to [USAMO](https://en.wikipedia.org/wiki/United_States_of_America_Mathematical_Olympiad). Here is one of the problems:

> **AIME 2025 I, problem 13.** Alex divides a disk into four quadrants with two perpendicular diameters intersecting at the center of the disk. He draws \( 25 \) more lines segments through the disk, drawing each segment by selecting two points at random on the perimeter of the disk in different quadrants and connecting those two points. Find the expected number of regions into which these \( 27 \) line segments divide the disk.

*Please pause here for at least 5 minutes, grab a pen and a piece of paper, and try to solve it. There are so many nuances in this fun little problem 😜. Correct human solution and AI solution will be provided at the end of this blog post.*

Due to its complexity, AIME problems has become a standard hard benchmark for LLM reasoning. Just a year ago, any improvements on AIME 2024 would make headlines in the news. The best open weight models like [DeepSeek Math](https://arxiv.org/abs/2402.03300) could only pull 3% on AIME 2024 (or 10% if we allow it to use tools). There are simply not enought human-written solutions in the Internet. It felt like progress in mathematical reasoning can only be achieved if you hundreds of thousands, or even millions of dollars to create a high quality dataset of solutions.

### Reasoning models
The release of DeepSeek R1 (a model comparable to OpenAI’s o1) together with the paper with full algorithmic details inspired a significant leap in reasoning abilities of open models. Interestingly, the authors of DeepSeek R1 included a small hint that one can improve the distilled R1 models further by performing Reinforcement Learning on these models.

{{< figure src="deepseek_paper_hint.png" caption="Section 3, page 14 of the DeepSeek R1 paper. The authors basically left the further model improvement “as an excercise to the reader.” This paragraph also confirm effectiveness of SFT from reasoning traces of larger model." invertible="true" >}}

Early R1 reproduction works like [DeepScaleR](https://www.notion.so/19681902c1468005bed8ca303013a4e2?pvs=21) by the [Agentica](https://agentica-project.com/) team suggests that finetuning such models might be much cheaper than expected. Perhaps, with better data and training curriculum, it is possible to make improvements to DeepSeek R1 distilled models on a budget? With that in mind, I entered the AIMO2 competition.

### AIMO competition
AIMO is a prestigious competition hosted on Kaggle with a total prize pool of $2’000’000, designed to push the frontier of open-source reasoning models. The difficulty of the problems are around the National Olympiad level. The problems have also been designed to be 'AI hard' in terms of the mathematical reasoning required, which was tested against open LLMs' capabilities (as of October 2024). Here is one of the [10 publicly available reference problems](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/data?select=AIMO_Progress_Prize_2_Reference_Problems_Solutions.pdf):

> **AIMO2 reference problem.** Let \( ABC \) be a triangle with \( BC=108 \), \( CA=126 \), and \( AB=39 \). Point \( X \) lies on segment \( AC \) such that \( BX \) bisects \( \angle CBA \). Let \( \omega \) be the circumcircle of triangle \( ABX \). Let \( Y \) be a point on \( \omega \) different from \( X \) such that \( CX=CY \). Line \( XY \) meets \( BC \) at \( E \). The length of the segment \( BE \) can be written as \( \frac{m}{n} \), where \( m \) and \( n \) are coprime positive integers. Find \( m+n \).

*Again, I highly encourage you to pause here for at least 5 minutes, grab a pen and a piece of paper, and try to solve it. Correct human solution and AI solution will be provided at the end of this post.*

## A few things about GRPO

Following the DeepSeek R1 paper, as our main RL method we chose GRPO — a variant of Policy Gradient Optimization first introduced in the [DeepSeek Math](https://arxiv.org/abs/2402.03300) paper. Though arguably less effective than PPO, GRPO's lack of a critic network cuts down memory usage and training time by half. For those new to RL and GRPO, I highly recommend reading [A vision researcher's guide to some RL stuff: PPO & GRPO](https://yugeten.github.io/posts/2025/01/ppogrpo/).

{{< figure src="grpo.png" caption="Comparison between PPO and GRPO, taken from [DeepSeek Math](https://arxiv.org/abs/2402.03300) paper. The Value model (also called the Critic) is omitted, and the advantage is calculated from group statistics rather than per-sample scores." invertible="true" >}}

While I was writing this blog post, Nathan Lambert published an excellent blog post discussing GRPO, its main problems, and how to mitigate them: [Recent reasoning research: GRPO tweaks, base model RL, and data curation](https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo). Our team knew about all of those problems (and more) during the AIMO2 competition and applied techniques from various papers to fix them.

### Length Bias

Since our end goal is to achieve a fast reasoning model, we identified generation length as the main problem with SFT models distilled from DeepSeek-R1 (671B) — we noticed that the longer we train our models, the longer the generated solutions gets.

But why are reasoning chains of thought so long? Is this an emergent property of reasoning models? As it turns out, it doesn't have to be this way. For example, in the [REINFORCE++](https://arxiv.org/html/2501.03262v1) paper, the authors found that "length hacking," as they called it, is an issue unique to GRPO. When they compared PPO, GRPO, RLOO, and REINFORCE++, they discovered that under the same unit of KL consumption, other algorithms achieved better reward increases while the average length of generated solutions didn't explode like it did with GRPO. Conversely, their experiments showed that GRPO achieved faster convergence.

Luckily, just before the final week of AIMO2 competition and when we just started our final GRPO experiments, two papers came out that solves the length problem: [DAPO](https://dapo-sia.github.io/) and [Dr. GRPO](https://arxiv.org/abs/2503.20783).

{{< figure src="dr_grpo.png" caption="[DAPO](https://dapo-sia.github.io/) and [Dr. GRPO](https://arxiv.org/abs/2503.20783) papers independently found that there is an implicit length bias in the [original GRPO formulation](https://arxiv.org/abs/2402.03300): longer incorrect solutions gets penalized less (token-wise) than shorter incorrect solutions. The solution? Get rid of the per-sample loss normalization term completely!" invertible="true" >}}

Turns out, GRPO’s length bias is just a matter of implementation — if you follow the original formulation and average the losses of each sample independently before aggregating the losses across samples, you will end up in a situation where tokens within longer sequences in a group may have a disproportionately lower contribution to the overall loss. The solution is to simply get rid of per-sample normalization.

### Difficulty Bias

The \( 1 / {\text{std}\left(\{R(q, o_1), \ldots R(q, o_G)\}\right)} \) term in the [original GRPO formulation](https://arxiv.org/abs/2402.03300), by which advantages are normalized, causes bias towards too easy or too hard questions over average questions, with the outcome rewards being almost all \(1\) or \(0\). The solution, proposed by [Dr. GRPO](https://arxiv.org/abs/2503.20783) as illustrated above, is to get rid of the rewards scaling it completely.

Interestingly, I first learned about the difficulty bias long before the [Dr. GRPO](https://arxiv.org/abs/2503.20783) paper, thanks to the following blog post by Twitter user [@leroykun](https://x.com/leloykun) posted just 2 weeks after DeepSeek R1’s release: [GRPO's Main Flaw](https://leloykun.github.io/ponder/grpo-flaw/).

### Do we even need KL?

Unlike PPO, the GRPO algorithm uses Forward KL Divergence formulation \( D_{KL}\left( \pi_\theta \vert| \pi_\text{ref} \right) \) instead of the Reverse KL \( D_{KL}\left( \pi_\text{ref} \vert| \pi_\theta \right) \) used in PPO. By the way, unlike the [original OpenAI’s RLHF paper](https://arxiv.org/pdf/2203.02155), GRPO uses the [Unbiased Approximator of KL bu Schulman](http://joschu.net/blog/kl-approx.html) rather than true KL). Twitter user [@kalomaze](https://x.com/kalomaze) found that [using Forward KL yield worse results](https://x.com/kalomaze/status/1891621285894995971), but in my own experiments with `DeepSeek-R1-Distill-1.5B` model they are roughly the same, which [agrees with the experiments](https://x.com/danielhanchen/status/1892643424538595611) by Unsloth's creator [@danielhanchen](https://x.com/danielhanchen).

The [DAPO](https://dapo-sia.github.io/) paper proposed to remove the term completely, because during training the long-CoT reasoning model, the model distribution can diverge significantly from the initial model, thus this restriction is not necessary.

## How we cooked our models

It is clear that, given our budget, we won’t be able to rely only on Reinforcement Learning to train our models as it is too expensive. A single [DeepScaleR-1.5B](https://www.notion.so/19681902c1468005bed8ca303013a4e2?pvs=21) experiment costs $5000 for a 10% increase in accuracy over the base `DeepSeek-R1-Distill-1.5B` model. To improve the 7B and 14B models, which are 5 and 10 times larger, it would cost a whole bitcoin to train these models.

For this reason, we decided to shift our focus on collecting high quality data and performing SFT first, and later do RL on top to enhance its reasoning abilities further and steer its behavior. The [Light-R1](https://arxiv.org/abs/2503.10460) paper, published just 2 weeks before our last experiment, showed potential for SFT on high quality data, which confirms that our direction was correct.

### Dataset Curation

Our team focused on collecting solutions that are less than 16K tokens. The reason is simple: most correct solutions of DeepSeek R1 are less than 6K tokens anyways, and after benchmarking the base 7B and 14B models we decided that 16K offers a nice balance between accuracy and compute cost.

- First, we filtered math word problems from [**NuminaMath-1.5**](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5), sourcing problems with topics in Algebra, Geometry, Number Theory, or Combinatorics from Olympiads, AoPS forums, AMC and AIME of previous years, olympiad references, and number theory sources. Our goal was to get harder problems in topics that DeepSeek models would likely struggle with.
- We then joined with correct R1 reasoning traces from [**OpenR1-Math-220k**](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k). Together with previous step, this filtered 800K problems down to 27K.
- To filter further, we sampled 8 solutions per problem with `max_len` of 8K tokens using [deepseek-r1-distill-qwen-7b-awq](https://huggingface.co/casperhansen/deepseek-r1-distill-qwen-7b-awq) and removed easy problems. We kept only problems with 7 or fewer correct solutions, leaving 8K problems. To get harder problems for GRPO, we used the 14B AWQ model for similar filtering.
- Later, we added a subset of from [**Light-R1**](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData) stage 2 data and removed duplicates with our dataset (after removing CoT with more than 16K tokens). We filter them further filter by difficulty, resulting in around 2K samples.

While sampling our dataset, we purposefuly avoided the following data sources:

- We purposefully avoided the `cn_k12` data source as it has much lower difficulty. Our team member [Raja Biswas](https://www.linkedin.com/in/raja-biswas/) found that these examples tend to deteriorate difficult math reasoning capability.
- We avoided synthetic math datasets like [Orca-Math](https://arxiv.org/abs/2402.14830), because such datasets are usually created by weaker LLMs with weaker solution correctness validators. Such datasets are only useful for training a new reasoning model from scratch (i.e. from a non-reasoning one), not for finetuning an already strong reasoning model.

Our final dataset for SFT consists of over 10K samples (8K that we filtered from [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) and 2K from [Light-R1](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData)). The harder half of the dataset is then used for the GRPO stage. Dataset curation is courtesy to [Raja Biswas](https://www.linkedin.com/in/raja-biswas/) and [Kelvin Soh](https://www.linkedin.com/in/kelvin-soh/), with some help from [Udbhav Bamba](https://www.linkedin.com/in/ubamba98/) in filtering.

### Stage 1: Supervised Fine Tuning

We used [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) and [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) as our base models and fine-tuned them for 6 epochs at 16K context length on a 8xH100 node. Some [other teams](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/571252) has found that fine tuning for much longer will further enhance accuracy, but at the expense of longer CoTs. Our SFT models are trained courtesy to [Kelvin Soh](https://www.linkedin.com/in/kelvin-soh/), [Raja Biswas](https://www.linkedin.com/in/raja-biswas/), and [Geremie Yeo](https://www.linkedin.com/in/geremie-yeo/).

### Stage 2: Reinforcement Learning

During the Reinforcement Learning process, we tried to steer our model's behavior toward shorter reasoning CoTs. We spent considerable time on 7B models, hoping our findings would translate to 14B, only to relearn [the bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) repeatedly — good tricks at smaller scales don't always work at larger ones.

- For the 7B model, we performed GRPO in two stages: first on 8K context, then on 16K context. We found LoRA converged faster than FFT while being more VRAM-efficient. We used DAPO's clipping and online sample filtering, and length penalty worked well for 7B.
- For 14B, length penalty severely hurts accuracy, so we removed it. Also, training on much shorter contexts significantly reduced accuracy at intended inference lengths. Model merging helped regain some accuracy, so our final submission is a merged SFT and GRPO on 6K context. We had another 14B GRPO on 16K context trained on the last day of the AIMO2 competition.

Our GRPO models are trained courtesy to [Geremie Yeo](https://www.linkedin.com/in/geremie-yeo/), [Kelvin Soh](https://www.linkedin.com/in/kelvin-soh/), [Raja Biswas](https://www.linkedin.com/in/raja-biswas/) and [Chan Kha Vu](https://hav4ik.github.io/about/). Each of these team members were running GRPO experiments independently during the last week of the AIMO2 competition. Every day we would check on each other’s experiments, compare them, share what worked with each other, and continue building on each other’s learnings and results. It was super fun and rewarding (pun intended)!

Interestingly, my teammates were using [Open-R1](https://github.com/huggingface/open-r1) with the [faster version of trl GRPOTrainer](https://github.com/nhannguyen2709/open-r1) created by user [@andy2709](https://www.kaggle.com/andy2709) on Kaggle, and I maintained an [active fork of veRL](https://github.com/hav4ik/verl/tree/dapo-lora) with DAPO, FSDP-LoRA, and Dr. GRPO integrated. Both frameworks offered long context RL with Sequence Packing and Ulysses, Hybrid Trainer with models collocation to save memory, vLLM rollouts, and more.

## Evaluation

For evaluation, we use AIME 2025 (published in March 2025) was used as an uncontaminated benchmark — it was published after the base DeepSeek R1 model was trained, and our only data source ([NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)) was collected before AIME 2025 was published. Since the main goal of our team is to train a model for AIMO2, we added the [10 “AI hard” reference problems](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/data?select=AIMO_Progress_Prize_2_Reference_Problems_Solutions.pdf) to our local validation set as well. In total, our local benchmark (denoted **CV**) that we track after each stage of development consists of 40 problems. All evaluations below were done by [Chan Kha Vu](https://hav4ik.github.io/about/).

### Setting

Reasoning traces are sampled with `BFloat16` precision, `temperature=0.75`, and `top_p=0.95`. We collect 64 traces per question with `max_len=32768`.

Length and `Pass@1` metrics were averaged across 64 rollouts per question. For aggregated metrics like `Maj@K` (majority voting accuracy), we sampled `K` traces per problem from our pool of 64 traces. We repeated this process 16 times and reported the average to reduce noise.

You can reproduce our evals using our evaluation code [aime25-aimo2-evals](https://www.kaggle.com/code/chankhavu/aime25-aimo2-evals) and our dataset of collected reasoning traces [reasoning-traces-aime25-aimo25](https://www.kaggle.com/datasets/chankhavu/reasoning-traces-aime25-aimo25).

### 14B models

- **Merged-14B:** the model we submitted to the Private LB of the AIMO2 competition for both of our final submissions. It's a merge of several SFT and GRPO with 6K context checkpoints.
- **Last-GRPO-14B:** trained in the final days of the AIMO2 competition. It showed worse results on our local End-to-End whole pipeline validation, despite being better on academic-style benchmarking settings, so we never submitted it to LB.

Below are majority voting metrics with generation `max_len` set at 12800 and 32768 tokens:

| Token budget | Model Name | CV Pass@1 | CV Maj@32 | AIME'25 Pass@1 | AIME'25 Maj@32 | Avg. length |
| --- | --- | --- | --- | --- | --- | --- |
| 12800 | DeepSeek-R1-Distill-14B | 0.412 | 0.613 | 0.41 | 0.648 | 9473.66 |
|  | Light-R1-14B-DS | 0.442 | 0.664 | 0.45 | 0.671 | 9787.01 |
|  | **Our Merged 14B** | 0.477 | **0.747** | 0.455 | **0.731** | **9251.56** |
|  | **Our Last GRPO 14B** | **0.482** | 0.736 | **0.468** | **0.738** | 9312.03 |
| 16384 | DeepSeek-R1-Distill-14B | 0.449 | 0.664 | 0.447 | 0.671 | 10910.2 |
|  | Light-R1-14B-DS | 0.498 | 0.713 | 0.502 | 0.731 | 11432.1 |
|  | **Our Merged 14B** | 0.525 | 0.759 | 0.504 | 0.746 | **10552.1** |
|  | **Our Last GRPO 14B** | **0.541** | **0.762** | **0.521** | **0.758** | **10511.6** |

It's easier to visualize the test-time scaling economy of our models with the following plot:
 
{{< figure src="test_time_scaling_14b.png" caption="Test-time scaling economy of our last 14B GRPO model. Each “dot” represents accuracy results on AIME 2025 when total token budget is set to one of the following: 8192, 9000, 12800, 16384, 20480. Our model reaches the peak Maj@32 of the base DeepSeek-R1-14B 33% faster, and have comparable peak Maj@32 to Light-R1." invertible="true" >}}

It should be noted that we built on top of [Light-R1](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData) work by using a subset of their stage 2 data, so this comparison is done only to show how much we were able to improve upon our baselines.

### 7B models

The SFT model was trained for 6 epochs. We trained our 7B GRPO in 2 stages: first with 8K context, then with 16K context. Our best 7B model is a merge of the SFT and GRPO models, but we never submitted it to the LB as we moved on to 14B models.

| Token budget | Model name | CV Pass@1 | CV Maj@32 | AIME'25 Pass@1 | AIME'25 Maj@32 | Avg. length |
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

{{< figure src="test_time_scaling_14b.png" caption="Test-time scaling economy of our final merged 7B model. Each “dot” represents accuracy results on AIME 2025 when total token budget is set to one of the following: 8192, 9000, 12800, 16384, 20480. Our model reaches the peak Maj@32 of the base DeepSeek-R1-7B 33% faster, and have comparable peak Maj@32 to Light-R1." invertible="true" >}}