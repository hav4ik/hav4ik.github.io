---
layout: post
permalink: /articles/:title
type: "article"
title: "How to dominate on technical interviews"
image:
  feature: "articles/images/2020-02-08-how-to-dominate-on-tech-interviews/trollfacaaaae.jpg"
  display: false
commits: https://github.com/hav4ik/hav4ik.github.io/commits/master/articles/_posts/2019-01-07-lorenz-attractor.md
tags: [math]
excerpt: "Do you want to dominate your interviewer? Wanna bring him to your knees? Make him think that you're superior? I'll tell you how!"
comments: true
hidden: true
---

We all know the frustration of technical interviews, when other candidates gets an offer not because they are better than you, but just because they were able to convince the interviewer that they are smarter. Basically, **the technical interview is a battle of persuasion**, not a test of one's engineering proficiency.

Let's put that to an end. You are not going to an interview to have the interviewer judge your skills. You are coming to the interview to **consolidate your dominance** and **bring the interviewer to your knees** with your knowledge. Show that you are **superior** than other candidates by mentioning other methods listed in this article that requires deep theoretical background. Here we go, topic-by-topic:

--------------------------------------------------

## Graph Algorithms

#### Q: How to find the number of connected components of an undirected graph?

*  **Virgin:** <s>Well, let's take a random node and run DFS from it, then mark the nodes that we visited &mdash; they belong to the same component. Increase the counter by 1 and repeat the procedure for the un-marked nodes. At the end, the value of the counter is what we need.</s>

*  **Chad:** It's simply $$\dim(\ker(L))$$, where $$L$$ is the [graph's (combinatorial) Laplacian][wiki-graph-laplacian]. Oh, you don't know what [kernel][nlab-kernel] is? It's just the [pullback][nlab-pullback] along $$L$$ of the unique morphism $$0 \to \text{im}(L)$$.


#### Q: How to check if a directed graph is acyclic?

* **Virgin:** <s>Let's perform a topological sorting of the graph. We can use Kahn's algorithm for that, it works by choosing vertices in the same order as the eventual topological sort. First, find a list of "start nodes" which have no incoming edges and insert them into a set S; at least one such node must exist in a non-empty acyclic graph. If at some point we fail to continue topological sorting, then the graph has cycles.</s>

* **Chad:** Just check if the [first homology][nlab-homology] of the [chain complex][nlab-chain-complex] $$\textstyle H_1(\ldots \to 0 \to Z[E] \xrightarrow{d} Z[V]) = 0$$, where $$d$$ is the [graph's incidence matrix][wiki-incidence-matrix], $$E$$ and $$V$$ are set of edges and vertices.


#### Q: How check if a directed graph is strongly connected?

* **Virgin:** <s>We can use Kosaraju’s algorithm. First, create an empty stack ‘S’ and do DFS traversal of a graph. In DFS traversal, after calling recursive DFS for adjacent vertices of a vertex, push the vertex to stack. Then, reverse directions of all arcs to obtain the transpose graph. One by one pop a vertex ‘v’ from S while S is not empty. Take v as source and do DFS. The DFS starting from v prints strongly connected component of v.</s>

* **Chad:** Simply check if its [adjacency matrix][wiki-adjacency-matrix] is [irreducible][wolfram-irreducible-matrix]. 


#### Q: How to check if a graph is bipartite?

* **Virgin:** <s>The idea is to assign to each vertex the color that differs from the color of its parent in the DFS tree, assigning colors in a preorder traversal of the DFS tree. If there exists an edge connecting current vertex to a previously-colored vertex with the same color, then the graph is not bipartite.</s>

* **Chad:** Just check if the [spectrum][wiki-spectrum-of-a-matrix] of the [graph's normalized Laplacian][wiki-graph-laplacian] $$L^{\text{sym}}$$ is symmetric about $$1$$.


#### Q: How to calculate the diameter of a unweighted graph?

* **Virgin:** <s> From every node of the graph, we can run a DFS to calculate the shortest distance from all nodes to the current node. Remember the distance to the furthest node. The diameter of our graph is the maximum from rememberred distances. Complexity is O(V * (E+V))</s>

* **Chad:** It's simply $$\inf_{k \in \mathbb{N}} \{ A^k_{i,j} \ne 0 \}$$, where $$A$$ is the [adjacency matrix][wiki-adjacency-matrix]. 



----------------------------------------------

## Backtracking algorithms

#### Q: How to find the solutions for a Sudoku puzzle?

* **Virgin:** <s>Like all other Backtracking problems, we can solve Sudoku by one by one assigning numbers to empty cells. Before assigning a number, we check whether it is safe to assign. We basically check that the same number is not present in the current row, current column and current 3X3 subgrid. After checking for safety, we assign the number, and recursively check whether this assignment leads to a solution or not. If the assignment doesn’t lead to a solution, then we try the next number for the current empty cell. And if none of the number (1 to 9) leads to a solution, we return false.</s>

* **Chad:** It's just points in the [vanishing locus][nlab-vanishing-locus] of [ideal][nlab-ideal] $$I$$ generated by [uni-][wolfram-univariate-polynomial] and [bi-nomials][wolfram-bivariate-polynomial] imposed by puzzle rules. Simply calculate the shape of the [reduced Gröbner basis][wiki-groebner-basis] of $$I + \langle \{x_i - a_i\}_{i \in L}\rangle$$, where $$\{a_i\}_{i\in L}$$ are pre-assigned values &mdash; the solution can be read off from it. [*(source)*][sol-sudoku]


#### Q: How to check if a graph is Hamiltonian (has a path that visits each vertex exactly once)?

* **Virgin:** <s>Create an empty path array and add vertex 0 to it. Add other vertices, starting from the vertex 1. Before adding a vertex, check for whether it is adjacent to the previously added vertex and not already added. If we find such a vertex, we add the vertex as part of the solution. If we do not find a vertex then we return false.</s>

* **Chad:** Compute it's [first homology group][nlab-homology] of it. If it's not [torsion-free][wiki-torsion-free], then it's not Hamiltonian. [*(source)*][sol-hamilton]


[nlab-chain-complex]: https://ncatlab.org/nlab/show/chain+complex
[nlab-homology]: https://ncatlab.org/nlab/show/homology
[nlab-ideal]: https://ncatlab.org/nlab/show/ideal
[nlab-kernel]: https://ncatlab.org/nlab/show/kernel
[nlab-pullback]: https://ncatlab.org/nlab/show/pullback
[nlab-vanishing-locus]: https://ncatlab.org/nlab/show/zero+locus

[wiki-adjacency-matrix]: https://en.wikipedia.org/wiki/Adjacency_matrix
[wiki-graph-laplacian]: https://en.wikipedia.org/wiki/Laplacian_matrix
[wiki-groebner-basis]: https://en.wikipedia.org/wiki/Gr%C3%B6bner_basis
[wiki-incidence-matrix]: https://en.wikipedia.org/wiki/Incidence_matrix
[wiki-spectrum-of-a-matrix]: https://en.wikipedia.org/wiki/Spectrum_of_a_matrix
[wiki-torsion-free]: https://en.wikipedia.org/wiki/Torsion_(algebra)#torsion-free_group

[wolfram-irreducible-matrix]: http://mathworld.wolfram.com/IrreducibleMatrix.html
[wolfram-univariate-polynomial]: http://mathworld.wolfram.com/UnivariatePolynomial.html
[wolfram-bivariate-polynomial]: http://mathworld.wolfram.com/BivariatePolynomial.html


[sol-sudoku]: https://www.cambridge.org/core/books/a-first-course-in-computational-algebraic-geometry/sudoku/29C4710314733FB36D793D70F914B7A5
[sol-hamilton]: https://arxiv.org/pdf/1912.06603.pdf