---
layout: post
permalink: /articles/:title
type: "article"
title: "Remarks on 14th Smale's Problem"
image:
  feature: "articles/images/2019-01-07-lorenz-attractor/featured.png"
  display: false
tags: [math, overview]
excerpt: "The connection between the proof of the existance of the Lorenz Attractor, its Geometric Flow Model, and the Knot Theory Model."
comments: true
---

> This is a small diary of my journey of understanding the solution and some non-trivial topological implications of one of the [Smale's Problems for the Next Century][smales-problems] &mdash; in particular, the **14th Problem**. Here, I will outline an intuitive overview that hides away most of the complicated mathematical parts.




## The Lorenz System

{% capture imblock1 %}
    {{ site.url }}/articles/images/2019-01-07-lorenz-attractor/test.svg
{% endcapture %}
{% include gallery images=imblock1 cols=1 %}

The [Lorenz System][lorenz-attractor] with specific parameter values studied by [Edward Lorenz (1963)][lorenz-paper] is a classical textbook example of a nonlinear system of ODE that behaves [chaotically][chaos-theory] (arbitrary close particles will diverge exponentially fast through time) and [attracting][attractor] (particles will end up in a bounded set, called *Lorenz Attractor*). The system is a simplified model of atmospheric dynamics:

$$
\begin{equation}
\begin{split}
    \dot{x_1} & = -\sigma x_1 + \sigma x_2 \\
    \dot{x_2} & = \rho x_1 - x_2 - x_1 x_3 \\
    \dot{x_3} & = -\beta x_3 + x_1 x_2
\end{split}
\tag{1}
\label{eq:lorenzsys}
\end{equation}
$$

and should be familiar for majority of students taking advanced Differential Equations courses.
The derivation of this simplified form from the full equations describing two-dimensional fluid can be found [here][lorenz-system-derivation].
Under classical parameters $$\rho = 28, \sigma = 10, \beta = 8/3,\,$$ the phase space trajectories will have a gorgeous behaviour as in the illustration above.

[smales-problems]: https://en.wikipedia.org/wiki/Smale%27s_problems
[attractor]: https://en.wikipedia.org/wiki/Attractor
[chaos-theory]: https://en.wikipedia.org/wiki/Chaos_theory
[lorenz-attractor]: https://en.wikipedia.org/wiki/Lorenz_system
[lorenz-paper]: https://journals.ametsoc.org/doi/10.1175/1520-0469%281963%29020%3C0130%3ADNF%3E2.0.CO%3B2
[lorenz-system-derivation]: http://mathworld.wolfram.com/LorenzAttractor.html




## The claim that captured my curiosity

About a year ago, during a *senseless wandering through the web on the sleepless night before university finals,* I've stumbled upon a [very interesting post][original-post] with a gorgeously animated [video][video] that states the following claim without providing any further sources:

> **Claim 1.** In 2001 mathematician Warwick Tucker proved that the paper model accurately describes the motion on the Lorenz attractor. For every trajectory on the attractor, there is a trajectory on the paper model that behaves exactly the same way (illustration below: paper model on the left and trajectory on Lorenz Attractor on the right).

<center>
<img src="{{ site.url }}/articles/images/2019-01-07-lorenz-attractor/trajectories.png"/>
</center>

The trajectories described in [this video][video] were so simple and elegant that I thought "It is too good to be true"!
How can such a chaotic system contain periodic trajectories with such elegant dynamics? And, more curiously, how could one possibly prove such a bold claim?
I was immediately captured.

The first thing that I did was to search for the [2001 paper][2001paper] by Dr. Warwick Tucker.
To say that it was hard to read was a huge understatement &mdash; I didn't understand a thing!
More importantly, there is no explicit mention about the topology of phase space trajectories, as stated above.

Naturally, I posted a [question on StackExchange][stackquestion] to ask for help. Although I did not received an answer for my question, the comment by [Giuseppe Negro][giuseppe-negro] suggested that the precise description of the *attracting set* and the dynamics of it might be burried in the proofs. 
The comment by [Evgeny][evgeny] has shed some light to my question &mdash; he pointed out that the claim above corresponds to [Birman&ndash;Guckenheimer&ndash;Williams Model][bgw-model] of the Lorenz Attractor, while Tucker's work is more about the [Geometric Model][geom-model], studied by Guckenheimer&ndash;Williams and Afraimovich&ndash;Bykov&ndash;Shilnikov. More detailed on these models in the next section.

This answers the question why I was unable to find any explicit hint about the topology of trajectories. However, this raises even more question on how I can use this proof to get to the claim above.
A more thoughtful skim (that lasted a few hours!) through the paper, I realised that the paper was about something completely different. But I needed to justify my concerns.
So, I emailed [Dr. Warwick Tucker][warwick-tucker] directly and quickly received the following answer:

> I did not prove that there is a 1-1 correspondence between individual
trajectories of the geometric model and the flow of the Lorenz
equations. I proved that the flow has a strange (singular hyperbolic)
attractor, and that the underlying dynamics is similar to that of the
geometric model.

This has cleared my doubts and now I'm confident that I got the gist of the paper correctly. However, this still leaves the question about the dynamics of trajectories, on which he replied:

> Proving a 1-1 correspondence as the movie claims would be impossible;
the attractor is extremely unstable in this sense: an arbitrarily small
perturbation of the parameters will radically change the set of periodic
orbits within the attractor.

Now that's a bummer! The claim was so beautiful, yet it's not correct. However, I was surprised that such a simple-looking system as the *Lorenz System* is in the list of important unsolved math problems of current century. So I decided to dig deeper into this problem (yup, I'm attracted to cool names and cool visuals).

[original-post]: http://www.chaos-math.org/en/chaos-vii-strange-attractors
[video]: https://www.youtube.com/watch?v=Rz2yEMeKZuE&feature=youtu.be&t=9m18s
[2001paper]: https://www.researchgate.net/profile/Warwick_Tucker/publication/220104401_A_rigorous_ODE_Solver_and_Smale%27s_14th_problem/links/54ba43650cf29e0cb049da3f/A-rigorous-ODE-Solver-and-Smales-14th-problem.pdf
[stackquestion]: https://math.stackexchange.com/questions/2798600/lorenz-attractor-its-geometric-model-and-14th-smales-problem
[giuseppe-negro]: https://math.stackexchange.com/users/8157/giuseppe-negro
[evgeny]: https://math.stackexchange.com/users/87697/evgeny
[bgw-model]: http://www.scholarpedia.org/article/Chaos_topology#Templates
[geom-model]: http://w3.impa.br/~viana/out/lsa.pdf
[warwick-tucker]: http://www2.math.uu.se/~warwick/CAPA/warwick/warwick.html




## The 14th Smale's Problem

At the end of previous century, a list of eighteen unsolved mathematical problems was proposed ([Steven Smale, 1998][smales-problems-original]) in reply to the request by Vladimir Arnold, who asked leading mathematicians to construct a list of important problems for the 21st century &mdash; similarly to the list of [Hilbert's Problems][hilbert-problems] for the 20th century. Famous unsolved problems, such as the [P versus NP problem][p-vs-np], the [Riemann Hypothesis][RH], and [Smoothness of Navier-Stokes equations][navier-stokes] has also made it to the list.

While the problems as [Riemann Hypothesis][RH] or [P vs NP][p-vs-np] are utterly important by their own due to numerous practical consequences that can reshape whole fields of mathematics, other problems might not be as important as the development of techniques and theories to solve them. The solution of the [Poincare Conjecture][poincare-conj], for example, has led to the proof of a more general theorem, the [Thurson's Geometrization Conjecture][thurson-conj], by adding Ricci Flow with surgery to the toolbox. The only solved problems in the list are the [Poincare Conjecture][poincare-conj] and the *14th Problem* that we are going to study. Its formulation is very simple:

> **Problem Number 14:**  Is the dynamics of the ordinary differential equations of Lorenz that of the geometric Lorenz attractor of Williams, Guckenheimer, and Yorke?

I will describe the geometric model more detailed in the next section. Now, one might ask: "why study a specific ODE? Isn't it too narrow?" Indeed, this specific formulation can be interpreted as a call for developing new techniques to analyze such kind of non-linear chaotic systems, rather than creating a whole new theory in which these problems become trivial &mdash; an unbearable problem, given our current understanding of mathematics.

[p-vs-np]: https://en.wikipedia.org/wiki/P_versus_NP_problem
[poincare-conj]: https://en.wikipedia.org/wiki/Poincar%C3%A9_conjecture
[thurson-conj]: https://en.wikipedia.org/wiki/Geometrization_conjecture
[RH]: https://en.wikipedia.org/wiki/Riemann_hypothesis
[navier-stokes]: https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations
[smales-problems-original]: https://www6.cityu.edu.hk/ma/doc/people/smales/pap104.pdf
[hilbert-problems]: https://en.wikipedia.org/wiki/Hilbert%27s_problems




## The Geometric Model, and the Knot Model

What exactly is the *Geometric Lorenz Attractor* in the *Problem 14 formulation* in previous section?
As it was untractable to extract rigorous information of the dynamics of system \eqref{eq:lorenzsys}, a geometric model of the Lorenz Flow was proposed by [John Guckenheimer (1976)][geometric-model]. The model was extensively studied, but the original equations remained a puzzle.

[geometric-model]: https://authors.library.caltech.edu/25053/25/Hopfch12-references-index.pdf
