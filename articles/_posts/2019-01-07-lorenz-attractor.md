---
layout: post
permalink: /articles/:title
type: "article"
title: "Remarks on 14th Smale's Problem"
image:
  feature: "articles/images/2019-01-07-lorenz-attractor/featured.png"
  display: false
commits: https://github.com/hav4ik/hav4ik.github.io/commits/master/articles/_posts/2019-01-07-lorenz-attractor.md
tags: [math, overview]
excerpt: "The connection between the proof of the existance of the Lorenz Attractor, its Geometric Flow Model, and the Knot Theory Model."
comments: true
highlighted: true
---

> This is a small diary of my journey of understanding the solution and some non-trivial topological implications of one of the [Smale's Problems for the Next Century][smales-problems] &mdash; in particular, the **14th Problem**. Here, I will outline the gist of the solution that hides away most of the complicated computational rigors.






## 1. The Lorenz System

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






## 2. The claim that captured my curiosity

About a year ago, during a *senseless wander through the web on the sleepless night before university finals,* I've stumbled upon a [very interesting post][original-post] with a gorgeously animated [video][video] that states the following claim without providing any further sources:

> **Claim 1.** In 2001 mathematician Warwick Tucker proved that the paper model accurately describes the motion on the Lorenz attractor. For every trajectory on the attractor, there is a trajectory on the paper model that behaves exactly the same way (illustration below: paper model on the left and trajectory on Lorenz Attractor on the right).

<center>
<div style="width: 70%;">
	<img src="{{ site.url }}/articles/images/2019-01-07-lorenz-attractor/trajectories.png" width="100%"/>
</div>
</center>

The trajectories described in [this video][video] were so simple and elegant that I thought **"It is too good to be true"**!
How can such a chaotic system contain periodic trajectories with such elegant dynamics? And, more curiously, how could one possibly prove such a bold claim?
I was immediately captured.

The first thing that I did was to search for the [2001 paper][2001paper] by Dr. Warwick Tucker.
To say that it was hard to read was a huge understatement &mdash; I didn't understand a thing!
More importantly, there is no explicit mention about the topology of phase space trajectories, as stated above.

Naturally, I posted a [question on StackExchange][stackquestion] to ask for help. Although I did not received an answer for my question, the comment by [Giuseppe Negro][giuseppe-negro] suggested that the precise description of the *attracting set* and the dynamics of it might be burried in the proofs. 
The comment by [Evgeny][evgeny] has shed some light to my question &mdash; he pointed out that the claim above corresponds to [Birman&ndash;Guckenheimer&ndash;Williams Model][bgw-model] of the Lorenz Attractor, while Tucker's work is more about the [Geometric Model][geom-model], studied by Guckenheimer&ndash;Williams and Afraimovich&ndash;Bykov&ndash;Shilnikov. More detailed on these models in the next section.

This answers the question why I was unable to find any explicit hint about the topology of trajectories. However, this raises even more question on how I can use this proof to get to the claim above.
A more thoughtful skim (that lasted a few hours!) through the paper, I realised that the paper was about something completely different. But I needed to justify my concerns.
So, I emailed [**Dr. Warwick Tucker**][warwick-tucker] directly and quickly received the following answer:

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






## 3. The 14th Smale's Problem

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






## 4. The Geometric Model

What exactly is the *Geometric Lorenz Attractor* in the *Problem 14 formulation* in previous section?
As it was untractable to extract rigorous information of the dynamics of system \eqref{eq:lorenzsys}, a geometric model of the Lorenz Flow was proposed by [John Guckenheimer (1976)][geometric-model]. The model was extensively studied, but the original equations remained a puzzle. It is important to note that while the geometric model is not a single model but a *family* of models that implies a [vector field][vector-field] with certain features, the [original proof (W. Tucker, 2002)][orig-proof] only considers a specific model.

{% capture imblock2 %}
    {{ site.url }}/articles/images/2019-01-07-lorenz-attractor/geometric_model.svg
{% endcapture %}
{% include gallery images=imblock2 cols=1 %}

Let's look closer at the properties of the global flow, illustrated in the figure $$(a)$$ above. As the parameter $$\rho > 1$$ in \eqref{eq:lorenzsys}, the flow has three fixed points: the origin and two "twin points"

$$
Q_\pm = \left(\pm\sqrt{\beta(\rho-1)}, \pm\sqrt{\beta(\rho-1)}, \rho-1\right)
$$

For the classical parameters, the origin is a saddle point around which are the two-dimensional [stable manifold][stable-unstable-manifold] $$W^s(0)$$ and one-dimensional [unstable manifold][stable-unstable-manifold] $$W^u(0)$$, as illustrated in figure $$(a)$$ above. We also consider the cross-section rectangle $$\Sigma \in \{ (x_1, x_2, x_3) \colon x_3 = \rho - 1\}$$ filled in gray in $$(a)$$ that is [transversal][transversality] to the flow, two opposite sides of which are parallel to $$x_2$$-axis and pass through the [equilibrium points][equilibrium-points] $$Q_-$$ and $$Q_+$$. The manifold $$W^u(0)$$ intersects $$\Sigma$$ at points $$\rho_-$$ and $$\rho_+$$. Let $$D$$ be the intersection of $$W^s(0)$$ with $$\Sigma$$. Obviously, trajectories starting from $$D$$ tends to origin as time moves forward and never returns to $$\Sigma$$.

As the flow is supposed to go down from $$\Sigma$$ and eventually return to $$\Sigma$$ from above, we can consider a [Poincaré Map][poincare-map] $$F \colon \Sigma_- \cup \Sigma_+ \rightarrow \Sigma\,$$, where $$\Sigma_-$$ and $$\Sigma_+$$ are parts of $$\Sigma$$ separated by $$D\,$$, as illustrated in figure $$(b)$$ above. Notice that some of the trajectories starting from $$\Sigma_+$$ will return to itself, while others will eventually end up in $$\Sigma_-\,$$, and vice versa.

We can now decomposite the return map: $$F = G \circ P\,$$, where $$G$$ is the [diffeomorphism][diffeomorphism] corresponding to the flow outside a unit cube centered at the origin, and $$P$$ describes the flow inside the cube. In other words, we divide the geometric model into two pieces: one piece dealing with all trajectories passing near origin, and one piece taking care of the global aspect of the flow, as illustrated below.

{% capture imblock3 %}
    {{ site.url }}/articles/images/2019-01-07-lorenz-attractor/global_local.svg
{% endcapture %}
{% include gallery images=imblock3 cols=1 %}

In the model, we also assume that the global flow $$G$$ preserves the $$x_2$$ direction, i.e. $$G$$ takes the horizontal lines $$\mathcal{l}(t) = (\pm 1, t, c)$$ into lines $$\tilde{\mathcal{l}}(t) = (\tilde{c}, t, 1)$$ &mdash; this ensures that the contracting direction is preserved. The return map now has a [hyperbolic][hyperbolic-set] splitting $$\mathbb{E}_ x^{s} \oplus \mathbb{E}_ x^u\,$$ with $$\mathbb{E}_ 0^s = D$$, and the *stable leaves* $$\tilde{\mathcal{l}}(t)$$ [foliate][foliation] $$\Sigma$$. Note that in this case, the map is an [Anosov diffeomorphism][anosov-diffeomorphism]. While the geometric model above have a lot of similar properties to the Lorenz flow, the real dynamics of the system is much chaotic and complicated.

[anosov-diffeomorphism]: https://en.wikipedia.org/wiki/Anosov_diffeomorphism
[foliation]: https://en.wikipedia.org/wiki/Foliation
[hyperbolic-set]: https://en.wikipedia.org/wiki/Hyperbolic_set
[vector-field]: https://en.wikipedia.org/wiki/Vector_field
[diffeomorphism]: https://en.wikipedia.org/wiki/Diffeomorphism
[poincare-map]: https://en.wikipedia.org/wiki/Poincar%C3%A9_map
[transversality]: https://en.wikipedia.org/wiki/Transversality_(mathematics)
[equilibrium-points]: https://en.wikipedia.org/wiki/Equilibrium_point
[stable-unstable-manifold]: https://en.wikipedia.org/wiki/Stable_manifold
[geometric-model]: https://authors.library.caltech.edu/25053/25/Hopfch12-references-index.pdf
[orig-proof]: http://www2.math.uu.se/~warwick/main/thesis_2.1.html






## 5. Solution to the 14-th Problem

In this section, I will briefly describe the beautiful idea of the proof for the 14-th Smale's Problem (basically highlighting the key details of [Warwick's paper](orig-proof)). First, we need to understand the notion of *SRB measure* and how it relates to dynamical systems.


### 5.1. The notion of Sinai&ndash;Ruelle&ndash;Bowen Measure

> **Definition (Axiom A Attractor):** a compact [$$f$$-invariant][invariant-set] set $$\Lambda$$ is called an *attractor* if there is a neighborhood $$B(\mathcal{A})$$ of $$\Lambda$$ called its *basin* such that $$f(x, t) \to \Lambda \enspace \forall x \in B(\mathcal{A})$$. $$\Lambda$$ is called an Axiom A Attractor if the [tangent bundle][tangent-bundle] over $$\Lambda$$ is split into $$\mathbb{E}^{s} \oplus \mathbb{E}^s\,$$, where $$\mathbb{E}^u$$ and $$\mathbb{E}^{s}$$ are [$$Df$$-invariant][invariant-set] subspaces &mdash; $$Df \vert _ {\mathbb{E}^u}$$ is uniformly expanding and $$Df \vert _ {\mathbb{E}^s}$$ is uniformly contracting.

In other words, *Axiom A Attractors* are [hyperbolic][hyperbolic-dynamics] attractors, i.e. expands in some direction and contracts in the other directions.
To be precise, by *uniformly expanding* property of $$Df \vert _ {\mathbb{E}^s}$$ we mean there is a constant $$\lambda > 1$$ such that $$\| Df(v) \| > \lambda \vert v \vert \enspace \forall v \in \mathbb{E}^u$$. We also assume that the attractor is [irreducible][irreducibility].
For the notion of *Axiom A Attractors* above, an invariant measure is proposed &mdash; the [Sinai&ndash;Ruelle&ndash;Bowen measure][srb-measure-1975], or *SRB measure*. Roughly speaking, it is the invariant measures that are most compatible with volume when volume is not preserved, and they provide a mechanism for describing the coherent statistics for unstable orbits of the attractor, starting from large sets in the basin. A formal definition of SRB measures is given by the following theorem.

> **Theorem (SRB measure)**. Let $$f$$ be a [$$C^2$$-diffeomorphism][c2-diffeomorphism] with an Axiom A Attractor $$\Lambda$$. Then there is a unique [$$f$$-invariant][invariant-measure] [Borel probability measure][borel-measure] $$\mu$$ on $$\Lambda$$, for which there is a set $$V \in B(\mathcal{A})$$ having full [Lebesque measure][lebesque-measure] such that for every continuous observable $$\varphi \colon B(\mathcal{A}) \to \mathbb{R}$$, we have, for every point $$x \in V$$:
>
$$
\lim _ {T \to \infty} \frac{1}{T} \int_0^T {\varphi\left(f(x, t)\right) dt} =
\int {\varphi(x) d\mu}
$$
>
> i.e. the time and space averages coincide. The invariant measure $$\mu$$ above is called the **Sinai&ndash;Ruelle&ndash;Bowen measure**, or **SRB measure** of $$f$$.

From the definition above, we can see that *SRB measure* is a notion that allows us to "escape" the sensitivity to initial conditions by time by observing the density of the flow itself.
[Lorenz][edward-lorenz] is one of the people who have most clearly expressed this idea: "over the years minuscule disturbances neither increase nor decrease the frequency of occurrence of various weather events such as tornados; the most they may do is to modify the sequence in which these events occur." i.e. the *frequency* of the event does not depends on initial conditions.
More details on SRB measures and the systems that have them was described by [L.-S. Young (2002)][young-srb]. Although they only describes the definition for discrete flow, most of it can be generalized to continuous case of dynamical systems as well.

Obviously, the definition of an *attractor* above is very broad &mdash; it also includes "boring" cases such as isolated attracting points, uniform circular motion, etc., i.e. cases with nothing "chaotic". For the purposes of distinguishing more "interesting" attractors, we introduce the notion of *strange attractors* &mdash; for almost all pairs (in the Lebesque mean) of different points in $$B(\mathcal{A})$$, their forward orbits eventually separates by a constant $$\delta$$ (only depending on $$\mathcal{A}$$). So, no matter how accurately we measure the initial conditions, we will eventually accumulate an error of size $$\delta$$.


### 5.2. Main result of Warwick Tucker

The main result of [W. Tucker (2002)][orig-proof] addresses the general behaviour of the flow, rather than behaviour of single particles, and reads as follows:

> **Main Theorem (Tucker).** For the classical parameter values, the Lorenz equations support a robust strange attractor $$\,\mathcal{A}\,$$. Furthermore, the flow admits a unique SRB measure $$\,\mu_X$$ with $$\,\text{supp}(\mu_X)  = \mathcal{A}\,$$.

By *robust* we mean that a strange attractor exists in an open neighbourhood of the classical parameter values. Here, the relation between the dynamics of Lorenz System with the Geometric Model is buried in the proof. Basically the theorem above is saying that not only the trajectories os Lorenz Flow \eqref{eq:lorenzsys} tends to an attractor $$\mathcal{A}$$, the flow also smears the whole space $$\mathbb{R}^3$$ across $$\mathcal{A}$$, like one would smear jelly on bread.


### 5.3. Proof Strategy

The general outline of the proof separately deals with the global part and the local part of the flow:

- First, we analytically inspect the flow $$P$$ inside the cube near origin, illustrated in figure $$(c)$$. More specifically, we inspect the flow from the cross-section $$\Sigma$$ to the point of leaving the cube.
- The global flow $$G$$ outside the cube, as illustrated in figure $$(d)$$, is hard to analyze, so we will use an [algorithm][algorithm] (a program) which, if successfully executed, proves the existance of strange attractor together with its dynamics.

The reason we have to explicitly analyze the cube near origin and the local flow is because inspecting the chaotic global flow analytically is an intractable task, and the algorithmic proof fails near origin due to difficulties dealing with stable manifolds.


### 5.4. Inspecting the Local Flow near the Origin

{% capture imblock4 %}
    {{ site.url }}/articles/images/2019-01-07-lorenz-attractor/varchange2.svg
{% endcapture %}
{% include gallery images=imblock4 cols=1 %}

To inspect the local flow near origin, a change of coordinates is introduced that deforms the incoming flow slightly, but in a controllable way. In the new coordinates, the vector field assumes a carefully designed [normal form][normal-forms], which is virtually linear (but the crucial part is that it shouldn't be completely linear). We choose to work with the [normal form][normal-forms] since finding a change of variables that completely linearizes the Lorenz Equations is impossible.

The following pair of propositions allows us to go back and forth from the original Lorenz form to a normal form, allowing us to perform further analysis and calculations:

> **Lemma 1 (change of coordinates).** There exists a close to identity change of variables in the small neighborhood of the origin
>
$$
\begin{equation}
\underbrace{\dot{\xi} = A\xi + F(\xi)} _ {\text{original Lorenz}}
\quad \xrightarrow[\xi = \zeta + \phi(\zeta)]{} \quad
\underbrace{\dot{\zeta} = A\zeta + G(\zeta)} _ {\text{normal form}}
\tag{2} \label{eq:coordchange}
\end{equation}
$$
>
> with $$\| \phi \| _ r \le \frac{r^2}{2}$$ for $$r \le 1$$, where $$A$$ is a linear operator, $$\,G(\zeta) \in \mathcal{O}^{10}(\zeta_1) \cap \mathcal{O}^{10}(\zeta_2, \zeta_3)$$ (flatness of order $$10$$) and satisfies $$\| G \| _ r \le 7 \cdot 10^{-9} \frac{r^{20}}{1 - 3r}$$ for $$r \le \frac{1}{3}$$ ($$\,G$$ is almost linear in that neigborhood).

In these new coordinates, the unstable manifold coincides with the $$\zeta_1$$-axis, and the stable manifold coincides with the $$\zeta_2\zeta_3$$ plane. The norm $$\| \cdot \| _ r$$ is defined as follows: let the modulo $$\vert \zeta \vert = \max\{\vert\zeta_1\vert, \vert\zeta_2\vert, \vert\zeta_3\vert\}$$, and the norm is defined in a $$r$$-neighborhood $$\| f \| _ r = \sup\{ \vert f \vert \colon \vert \zeta \vert \le r\}$$. This permits us to estimate the evolution of the flow analytically. When changing back to the original coordinates, the out-going flow is once again deformed, but still in a controllable fashion, as shown in the following theorem:

> **Lemma 2 (inverse coordinate change).** For $$\vert \zeta \vert \le r \le \frac{1}{2}$$, the change of variables $$\xi = \zeta + \phi(\zeta)$$ in the previous theorem has a well defined inverse $$\,\zeta = \xi + \psi(\xi)$$ in the ball $$\,\vert \xi \vert \le \tilde{r} = r - \| \phi \| _ r$$ satisfying
>
$$
\| \psi \| _ {\tilde{r}} \le \| \phi \| _ {r}
\quad\quad
\| D\psi \| _ {\tilde{r}} \le \frac{\| D\phi \| _ {r}}{1 - \| D\phi \| _ {r}}
$$

Combining the theorems *1* and *2* on the change of coordinates and inverse change, we get the cyclic scheme as in figure $$(e)$$ and the ability to enter and leave the cube around the origin without much iterruption to the flow. This allows us to analyze the normal flow instead.

It is also proven that the normal form in \eqref{eq:coordchange} of the real Lorenz flow and the linear flow introduced by Geometric Model have very similar behaviour. The following shows the similarity of their $$C_1$$ properties:

> **Lemma 3 (linearity of normal flow).** Let $$\psi(\zeta, t)$$ be the normal flow, i.e. denotes the solution to the equations $$\dot{\zeta} = A\zeta + G(\zeta)$$. Let $$\phi(\zeta, t)$$ denote the flow of the linearized Lorenz equations $$\zeta = A\zeta$$. For all trajectories starting from the lid of the cube $$\{\zeta \colon \vert \zeta \vert \le r \}$$, where $$r \le \frac{1}{4}$$, we have:
>
$$
\left\vert
\frac{\partial \psi_i}{\partial \zeta_j} \left(\zeta, t\right) -
\frac{\partial \phi_i}{\partial \zeta_j} \left(\zeta, t\right)
\right\vert
\le \kappa e^{[9(\lambda_3 + \kappa) + \lambda_j]t}
\quad
(i, j = 1, 2, 3)
$$
>
> where $$\kappa = 2 \cdot 10^{-19}$$ and $$\lambda_i\$$ are eigenvalues of $$\phi$$. These estimates holds throughout the cube.

This theorem says that in our small cube illustrated in $$(c)$$, the normal flow expands and contracts [tangent vectors][tangent-vector] at almost the same rate as the linear flow does.
It is important to point out that $$9(\lambda_3 + \kappa) + \lambda_j$$ is negative, which means that the error decreases as the exit-time increases, i.e. as we take $$\vert \zeta_1 \vert$$ small.

<blockquote markdown="1">
**Lemma 4 (normal flow tracing).** The next table shows the similarity of the $$C_0$$ properties of the coordinate variables $$\zeta_1, \zeta_2, \zeta_3$$ and the exit time $$\tau_e(\zeta)$$ for all trajectories starting from the lid of the cube $$\{\zeta \colon \vert \zeta \vert \le r \}$$:

<table>
<thread>
  <tr>
    <th>Linear Flow</th>
    <th>Normal Flow</th>
  </tr>
</thread>
<tbody>
  <tr>
    <td rowspan="2">$$\textstyle \phi_i (\zeta, t) = e^{\lambda_i t} \zeta_i\,,$$ $$\textstyle i = 1, 2, 3$$</td>
    <td>$$\textstyle \vert \zeta_i \vert e^{(\lambda_i - \kappa)t} \le \vert \psi_i(\zeta, t)\vert \le \vert \zeta_i \vert e^{(\lambda_i + \kappa)t}\,, i = 1, 3$$</td>
  </tr>
  <tr>
    <td>$$\textstyle \left(\zeta_2 - \kappa r(1 - e^{-3t})\right) e^{\lambda_2 t} \le \psi_2(\zeta, t) \le \left(\zeta_2 + \kappa r(1 - e^{-3t})\right) e^{\lambda_2 t}$$</td>
  </tr>
  <tr>
  	<td>$$\textstyle \tau_e(\zeta) = \frac{1}{\lambda_1} \log \frac{r}{\vert \zeta_1 \vert}$$</td>
  	<td>$$\textstyle \frac{1}{\lambda_1 + \kappa} \log \frac{r}{\vert \zeta_1 \vert} \le \tau_e(\zeta) \le \frac{1}{\lambda_1 - \kappa} \log \frac{r}{\vert \zeta_1 \vert}$$</td>
  </tr>
</tbody>
</table>

where $$r \le \frac{1}{4}$$ and $$\kappa = 2 \cdot 10^{-19}$$
</blockquote>


So, with some care, we can treat them similarly. I will not further provide the comparison of properties of the normal flow and linear flow here, since the objective of this article is just to provide the reader with a general vision on the topic.

The stages of propagating a rectangle on the cross-section $$\Sigma$$ through the cube illustrated in figure $$(c)$$ are follows: first, we distort it by performing the change of coordinates \eqref{eq:coordchange}; if the rectangle intersects with $$D$$ &mdash; cross-section of stable manifold $$W^s(0)$$, we split it in two parts and propagate each part separately, as shown in figure $$(f_1)$$, else we propagate it as a whole, as shown in figure $$(f_2)$$; once we reach the exit boundary of the box, we distort the region back by performing the inverse change of coordinates; then, we hand the region to the program that deals with the global flow.


### 5.5. Investigating the Global Flow

The flow around the origin is hard to analyze numerically because it is highly unstable, so we did an analytic trick. For the global flow, however, we cannot use such approach. Here is where computer modelling kicks in &mdash; if we can model the flow and prove that it is bounded, then it automatically means that the attractor exists.





<!-- References for the general solution outline part -->
[c2-diffeomorphism]: https://en.wikipedia.org/wiki/Diffeomorphism
[srb-measure-1975]: https://www.cpht.polytechnique.fr/sites/default/files/Bowen_LN_Math_470_second_ed_v2013.pdf
[young-srb]: https://www.researchgate.net/publication/225834113_What_Are_SRB_Measures_and_Which_Dynamical_Systems_Have_Them
[hyperbolic-dynamics]: http://www.scholarpedia.org/article/Hyperbolic_dynamics
[irreducibility]: https://www.encyclopediaofmath.org/index.php/Irreducible_topological_space
[tangent-bundle]: https://en.wikipedia.org/wiki/Tangent_bundle
[invariant-set]: https://en.wikipedia.org/wiki/Positive_invariant_set
[orig-proof]: http://www2.math.uu.se/~warwick/main/thesis_2.1.html
[invariant-measure]: https://en.wikipedia.org/wiki/Invariant_measure
[borel-measure]: https://en.wikipedia.org/wiki/Borel_measure
[lebesque-measure]: https://en.wikipedia.org/wiki/Lebesgue_measure
[edward-lorenz]: https://en.wikipedia.org/wiki/Edward_Norton_Lorenz
[algorithm]: https://en.wikipedia.org/wiki/Algorithm

<!-- References for the Normal Flow part -->
[normal-forms]: http://www.scholarpedia.org/article/Normal_forms
[tangent-vector]: https://en.wikipedia.org/wiki/Tangent_vector