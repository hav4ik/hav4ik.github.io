---
layout: post
type: "article"
title: "Network Surgery in TensorFlow & Keras"
image:
  feature: "articles/images/2018-11-26-network-surgery/featured.jpg"
  display: false
tags: [tensorflow, tutorial]
excerpt: "A practical tutorial on Network Surgery for Keras and Tensorflow using tf.contrib.graph_editor toolbox."
comments: true
---

> TL;DR: Don't do Network Surgery in TensorFlow! Do it in PyTorch! But, if you have to, this tutorial will save ya!

**Network Surgery** is the process of modifying the computation graph and/or injecting values to variables, either through the source code or directly making changes to the `GraphDef`.
There are several practical reasons why one might want to modify the computational graph in `TensorFlow`:

* **Deployment**. The process of deployment the final model for production either on a server or on a mobile device requires modifications and tweaks to the computational graph itself.

* **Model porting**. To port the trained model on a different framework, *op replacement* might be needed if the framework you are porting to does not support similar ops as `TensorFlow`. This happens a lot when converting the model to `TF-Lite` (a new hip framework by TF team) for mobile inference.

* **Architecture Search**. Some of the *Neural Architecture Search* algorithms greedily modifies the graph during the training loop. This also includes the *Live Pruning* problems.

In general, the ability to directly make graph modifications is very useful.
It might enable the engineers and researchers to implement new and exciting ideas without sacrificing the maturity of `TensorFlow` deployment pipeline (which is the main reason why one might want to do Network Surgery in `TensorFlow` instead of `PyTorch`).

This tutorial is organized as a collection of hands-on examples that I encountered and used myself during my career:


0. [Limited functionality of Keras: replacing a layer](#limited-keras)
1. [TensorFlow: Graphs vs GraphDefs, Variables vs Tensors](#theory)
2. [Wise words from a Wise Man!](#wise-words)

These examples will cover the majority of use-cases and provide the reader with a comprehensive understanding of the graph editing mechanism.



## Limited functionality of Keras: replace a layer
<a name="limited-keras"></a>
If you just need to replace a layer in Keras, then *Congrats*! You will just need to read this section and throw the rest away! Don't need to dig deep into internal mechanisms of `TensorFlow`! Good for ya.



## TensorFlow: Graphs vs GraphDefs, Variables vs Tensors
<a name="theory"></a>

To perform delicate operations and modifications on a `TensorFlow` computation graph, you will need to understand the concepts for `Graph`, `GraphDef`, `Variable`, and `Tensor`, how they are related as well as the difference between them.



## Wise words from a Wise Man!
<a name="wise-words"></a>

Messing with a computation graph with hundreds of operations and tons of connections is hard! You will quickly get confused if you were not careful.
Here are some enlightment words from a Wise Man for ya:

> **Always visualize! Print out everything and visualize everything after each action!**

Here are some snippets to help you do that:

{% highlight python %}
# We use default graph in this example, but you can use any graph
graph = tf.get_default_graph()

# Iterate through all nodes in the GraphDef of the given graph
for n in graph.as_graph_def().node:
    print(n.name)   # node's name
    print(n.op)     # node's op type
{% endhighlight %}


