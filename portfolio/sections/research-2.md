
## Adaptive Multi-Head Network Branching using Feature Similarity (2019)

This work eliminates the limitations of previous Multi-Task Branching Scheme Search methods: it doesn't require each sample to have labels for all tasks, is not limited only to classification, and doesn't require training for all configurations on each level. The core idea is to enforce Network Slimming regularization on Batch Norm after each block, then estimate branch affinity as sum of Jensen&ndash;Shannon Divergence between activations, weighted by pruning importance. This is part of my Master's thesis. **Work in progress.**
