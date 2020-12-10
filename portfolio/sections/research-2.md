
## Adaptive Multi-Head Network Branching using Feature Similarity (2019)

This work eliminates the limitations of previous Multi-Task Architecture Search methods. More specifically, it doesn't require that each sample should have labels for all tasks, is not limited only to classification, doesn't require O(N^2) training on all possible pairwise configurations on each level. The core idea is to enforce Network Slimming regularization on Batch Norm on top of each block, then estimate branch affinity as Kullback&ndash;Leibner Divergence between activations. This is part of my Master's thesis. Work in progress.
