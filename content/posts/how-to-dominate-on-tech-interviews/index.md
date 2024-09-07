---
title: "How to Dominate on Tech Interviews"
url: /articles/how-to-dominate-on-tech-interviews
date: 2020-02-08T00:00:00+00:00
# weight: 1
# aliases: ["/first"]
tags: ["Math"]
author: "Kha Vu Chan"
# author: ["Me", "You"] # multiple authors
showToc: false
TocOpen: false
draft: false
hidemeta: false
comments: true
# Migrated from the old blog
disqus_identifier: "08 Feb 2020/articles/how-to-dominate-on-tech-interviews"
summary: "Do you want to dominate your interviewer? Wanna bring him to his knees? Make him think that you're superior? I'll tell you how!"
canonicalURL: "https://canonical.url/to/page"
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
    image: "virgin-vs-chad-interview.png" # image path/url
    alt: "Virgin coder vs Chad mathematician meme" # alt text
    caption: "Virgin coder vs Chad mathematician meme" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide in list view
editPost:
    URL: "https://github.com/hav4ik.github.io/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---


> **NOTE**: this is a **Troll Post**, and I used strong and offensive language tone for satirical purposes. Interviewers are very good and extremely smart people, usually with many years of experience of building amazing products. Please treat them with respect and just show what you're capable for :)


We all know the frustration of technical interviews when other candidates get an offer not because they are better than you, but just because they were able to convince the interviewer that they are smarter. Basically, **the technical interview is a battle of persuasion**, not a test of one's engineering proficiency.

Let's put that to an end. You are NOT going to an interview to have the interviewer judge your skills. You are coming to the interview to **consolidate your dominance** and **bring the interviewer to his knees** with your knowledge. Show that you are **superior** to other candidates by using methods listed in this article.

I've interviewed for top tech companies, including Google, Facebook, Samsung, Microsoft, and got offers **more often than random chance!** I know what I'm talking about! Trust me!


{{< figure src="virgin-vs-chad-interview.png" alt="Virgin vs Chad Interview" >}}


Below are the most common interview questions, taken from [Geek4Geeks: top interview questions][geek-for-geeks-questions], and examples of answers that will help you achieve **total domination.** Here we go, topic-by-topic:

* [Graph Algorithms](#graph-algorithms)
* [Backtracking Algorithms](#backtracking-algorithms)
* [Combinatorics](#combinatorics)

--------------------------------------------------

<a name="graph-algorithms"></a>
## Graph Algorithms

#### Q: How to find the number of connected components of an undirected graph?

*  **Virgin:** ~~Well, let's take a random node and run DFS from it, then mark the nodes that we visited &mdash; they belong to the same component. Increase the counter by 1 and repeat the procedure for the un-marked nodes. At the end, the value of the counter is what we need.~~

*  **Chad:** It's simply \( \dim(\ker(L)) \), where \( L \) is the [graph's (combinatorial) Laplacian][wiki-graph-laplacian]. Oh, you don't know what [kernel][nlab-kernel] is? It's just the [pullback][nlab-pullback] along \( L \) of the unique morphism \( 0 \to \text{im}(L) \).


#### Q: How to check if a graph is connected?

* **Virgin:** ~~Take two bool arrays vis1 and vis2 of size N (number of nodes of a graph) and keep false in all indexes. Start at a random vertex v of the graph G, and run a DFS(G, v). Make all visited vertices v as vis1[v] = true. Now reverse the direction of all the edges. Start DFS at the same vertex v. Make all visited vertices v as vis2[v] = true. If any vertex v has vis1[v] = false and vis2[v] = false then the graph is not connected.~~

* **Chad:** Simply check if its first cohomology group vanishes, i.e. \( \textstyle H^0(G) = \{ 0 \} \). [*(source)*][utube-cohomomoly-for-cs-02-51]


#### Q: How to check if a directed graph is acyclic?

* **Virgin:** ~~Let's perform a topological sorting of the graph. We can use Kahn's algorithm for that, it works by choosing vertices in the same order as the eventual topological sort. First, find a list of "start nodes" which have no incoming edges and insert them into a set S; at least one such node must exist in a non-empty acyclic graph. If at some point we fail to continue topological sorting, then the graph has cycles.~~

* **Chad:** Just check if the [first homology][nlab-homology] of the [chain complex][nlab-chain-complex] \( \textstyle H_1(\ldots \to 0 \to Z[E] \xrightarrow{d} Z[V]) = 0 \), where \( d \) is the [graph's incidence matrix][wiki-incidence-matrix], \( E \) and \( V \) are set of edges and vertices.


#### Q: How check if a directed graph is strongly connected?

* **Virgin:** <s><font color="#777">We can use Kosaraju’s algorithm. First, create an empty stack ‘S’ and do DFS traversal of a graph. In DFS traversal, after calling recursive DFS for adjacent vertices of a vertex, push the vertex to stack. Then, reverse directions of all arcs to obtain the transpose graph. One by one pop a vertex ‘v’ from S while S is not empty. Take v as source and do DFS. The DFS starting from v prints strongly connected component of v.</font></s>

* **Chad:** Simply check if its [adjacency matrix][wiki-adjacency-matrix] is [irreducible][wolfram-irreducible-matrix].


#### Q: How to check if a graph is bipartite?

* **Virgin:** ~~The idea is to assign to each vertex the color that differs from the color of its parent in the DFS tree, assigning colors in a preorder traversal of the DFS tree. If there exists an edge connecting current vertex to a previously-colored vertex with the same color, then the graph is not bipartite.~~

* **Chad:** Just check if the [spectrum][wiki-spectrum-of-a-matrix] of the [graph's normalized Laplacian][wiki-graph-laplacian] \(L^{\text{sym}}\) is symmetric about \(1\).


#### Q: How to calculate the diameter of a unweighted graph?

* **Virgin:** ~~ From every node of the graph, we can run a DFS to calculate the shortest distance from all nodes to the current node. Remember the distance to the furthest node. The diameter of our graph is the maximum from rememberred distances. Complexity is O(V * (E+V))~~

* **Chad:** It's simply \(\inf_{k \in \mathbb{N}} \{ A^k_{i,j} \ne 0 \}\), where \(A\) is the [adjacency matrix][wiki-adjacency-matrix].


----------------------------------------------

<a name="backtracking-algorithms"></a>
## Backtracking algorithms

#### Q: How to find the solutions for a Sudoku puzzle?

* **Virgin:** ~~Like all other Backtracking problems, we can solve Sudoku by one by one assigning numbers to empty cells. Before assigning a number, we check whether it is safe to assign. We basically check that the same number is not present in the current row, current column and current 3X3 subgrid. After checking for safety, we assign the number, and recursively check whether this assignment leads to a solution or not. If the assignment doesn’t lead to a solution, then we try the next number for the current empty cell. And if none of the number (1 to 9) leads to a solution, we return false.~~

* **Chad:** It's just points in the [vanishing locus][nlab-vanishing-locus] of [ideal][nlab-ideal] \(I\) generated by [uni-][wolfram-univariate-polynomial] and [bi-nomials][wolfram-bivariate-polynomial] imposed by puzzle rules. Simply calculate the shape of the [reduced Gröbner basis][wiki-groebner-basis] of \(I + \langle \{x_i - a_i\}_{i \in L}\rangle\), where \(\{a_i\}_{i\in L}\) are pre-assigned values — the solution can be read off from it. [*(source)*][sol-sudoku]

#### Q: How to check if a graph is Hamiltonian (has a path that visits each vertex exactly once)?

* **Virgin:** ~~Create an empty path array and add vertex 0 to it. Add other vertices, starting from the vertex 1. Before adding a vertex, check for whether it is adjacent to the previously added vertex and not already added. If we find such a vertex, we add the vertex as part of the solution. If we do not find a vertex then we return false.~~

* **Chad:** Compute it's [first homology group][nlab-homology] of it. If it's not [torsion-free][wiki-torsion-free], then it's not Hamiltonian. [*(source)*][sol-hamilton]


#### Q: Place \( n \) queens on an \(n \times n\) chessboard such that no two queens attack each other.

* **Virgin:** ~~(1) Start in the leftmost column. (2) If all queens are placed then return true. (3) Try all rows in the current column. Do following for every tried row. (3a) If the queen can be placed safely in this row then mark this [row, column] as part of the solution and recursively check if placing queen here leads to a solution. (3b) If placing the queen in [row, column] leads to a solution then return true. (3c) If placing queen doesn't lead to a solution then unmark this [row, column] and go to step (3a) to try other rows. If all rows have been tried and nothing worked, return false to trigger backtracking.~~

* **Chad:** Construct an \(n \times n\) [magic square][wiki-magic-square] (e.g., using a combination of [de la Loubère][wiki-siamese-method] and [LUX][wolfram-magic-square] methods). Replace each entry \(a_{ij}\) with \(b_{ij} \equiv a_{ij} \mod n + 1\). Every row, column, and modular diagonal yields a superimposable solution for \(n\)-queens. [*(source)*][sol-nqueens]

--------------------------------------------------

<a name="combinatorics"></a>
## Combinatorics

#### Q: In how many ways one can choose \(k\) items from \(n\) items?

* **Virgin:** ~~You take the total number of permutations. Then, you divide it by the number of ways to arange the first k elements (the chosen elements), and divide that to the number of ways to arange the last n-k elements, the ones that we don't choose. So, the formula is n! / (k! (n-k)!).~~

* **Chad:** Just compute the [residue][wiki-residue] \(\displaystyle \frac{1}{2\pi i} \oint \frac{(1 + z)^n\,dz}{z^{k+1}}\) around \(0\) (e.g., using [Gauss-Legendre][wiki-gauss-legendre-integration]). [*(source)*][wiki-egorychev-method]


#### Q: What is the number of expressions containing \(n\) pairs of parentheses which are correctly matched?

* **Virgin:** ~~This is the n-th Catalan Number, which can be calculated recursively. The leftmost opening parenthesis l corresponds to certain closing bracket r, which divides the sequence into 2 parts which in turn should be a correct sequence of brackets. Thus formula is also divided into 2 parts. If we denote k=r−l−1, then for fixed r, there will be exactly C_k * C_{n−1−k} such bracket sequences. Summing this over all admissible k′s, we get the recurrence relation on C_n.~~

* **Chad:** Calculate the [generating function][nlab-generating-function] of it using the [continued fraction][nlab-continued-fraction] \(\textstyle \cfrac{1}{1-\cfrac{z^2}{1-\cfrac{z^2}{\dots}}}\,.\) [*(source)*][sol-catalan]


#### Q: What is the number of permutation of \(n\) elements such that no element appears in its original position?

* **Virgin:** ~~There are n – 1 ways for element 0. Let 0 be placed at index i. There are now two possibilities, depending on whether or not element i is placed at 0 in return. If i is placed at 0, it is equivalent to solving the problem for n-2 elements as two elements have just swapped their positions. Otherwise, it is equivalent to solving the problem for n-1 elements as now there are n-1 elements, n-1 positions and every element has n-2 choices. The recursive relation is D(n) = (n - 1) * [D(n - 1) + D(n - 2)].~~

* **Chad:** Just compute the [generating function][nlab-generating-function] of it, which is simply \(\textstyle \cfrac{1}{1-\cfrac{1^2 z^2}{1-2z-\cfrac{2^2 z^2}{\dots}}}\,.\) [*(source)*][sol-derangements]



------------------------------------------------------

## Conclusion

This is just a trolling article that I wrote for fun, so please don't use this in your technical interviews (especially if you REALLY want to get the job). Be nice and respectful to your interviewers &mdash; they are very nice and smart people, just trying to find a good fit for their team.

Special thanks to **@Quantum** and **Pochekay Nikolai** for suggesting some of the stuffs above. By the way, I am looking for more unexpected answers to common technical interviews. So, if you have any &mdash; please suggest in the comments below!



[geek-for-geeks-questions]: https://www.geeksforgeeks.org/top-10-algorithms-in-interview-questions/

[nlab-chain-complex]: https://ncatlab.org/nlab/show/chain+complex
[nlab-continued-fraction]: https://ncatlab.org/nlab/show/continued+fraction
[nlab-generating-function]: https://ncatlab.org/nlab/show/generating+function
[nlab-homology]: https://ncatlab.org/nlab/show/homology
[nlab-ideal]: https://ncatlab.org/nlab/show/ideal
[nlab-kernel]: https://ncatlab.org/nlab/show/kernel
[nlab-pullback]: https://ncatlab.org/nlab/show/pullback
[nlab-vanishing-locus]: https://ncatlab.org/nlab/show/zero+locus

[wiki-adjacency-matrix]: https://en.wikipedia.org/wiki/Adjacency_matrix
[wiki-egorychev-method]: https://en.wikipedia.org/wiki/Egorychev_method
[wiki-gauss-legendre-integration]: https://en.wikipedia.org/wiki/Gaussian_quadrature
[wiki-graph-laplacian]: https://en.wikipedia.org/wiki/Laplacian_matrix
[wiki-groebner-basis]: https://en.wikipedia.org/wiki/Gr%C3%B6bner_basis
[wiki-incidence-matrix]: https://en.wikipedia.org/wiki/Incidence_matrix
[wiki-magic-square]: https://en.wikipedia.org/wiki/Magic_square
[wiki-residue]: https://en.wikipedia.org/wiki/Residue_(complex_analysis)
[wiki-siamese-method]: https://en.wikipedia.org/wiki/Siamese_method
[wiki-spectrum-of-a-matrix]: https://en.wikipedia.org/wiki/Spectrum_of_a_matrix
[wiki-torsion-free]: https://en.wikipedia.org/wiki/Torsion_(algebra)#torsion-free_group

[wolfram-irreducible-matrix]: http://mathworld.wolfram.com/IrreducibleMatrix.html
[wolfram-magic-square]: http://mathworld.wolfram.com/MagicSquare.html
[wolfram-univariate-polynomial]: http://mathworld.wolfram.com/UnivariatePolynomial.html
[wolfram-bivariate-polynomial]: http://mathworld.wolfram.com/BivariatePolynomial.html

[sol-catalan]: http://algo.inria.fr/flajolet/Publications/Flajolet80b.pdf
[sol-derangements]: http://algo.inria.fr/flajolet/Publications/Flajolet80b.pdf
[sol-hamilton]: https://arxiv.org/pdf/1912.06603.pdf
[sol-nqueens]: https://www.sciencedirect.com/science/article/pii/S0012365X07010394
[sol-sudoku]: https://www.cambridge.org/core/books/a-first-course-in-computational-algebraic-geometry/sudoku/29C4710314733FB36D793D70F914B7A5

[utube-cohomomoly-for-cs-02-51]: https://youtu.be/1wtq5A7VMsA?t=1761