---
theme: presentation
marp: true
math: katex
---

<!-- _class: titlepage-->

<div>

![width:160](images/logo/cheese.png)

# Artificial Intelligence and Machine Learning for Geosciences

`Barcelona` `November 2024`

Léonard Seydoux, Hugo Frezat, Geneviève Moguilny & Alexandre Fournier

[`cheese-ai-for-geosciences`](https://github.com/leonard-seydoux/cheese-ai-for-geosciences)

<br>

![width:420px](images/logo/europe.png)

</div>

---

<!-- paginate: true -->
## Goal: Learn about statistical inference and machine learning
1. __Identify__ data-related scientific problems
1. __Define__ the problem and design a solution
1. __Learn__ from examples in the litterature
1. __Criticize__ the litterature
1. __Train__ on real geoscience problems 

![drop-shadow width:400px](images/papers/bergen2019machine.png)
<!-- _footer: Bergen et al. (2019) -->

---

## Goal: Keep up with the ongoing pace
![width:750px](images/papers/mousavi2022papers.jpg)
<!-- _footer: Mousavi et al. (2022) -->

---

## Contents of this class make use of the scikit-learn library
![width:900](https://scikit-learn.org/stable/_downloads/b82bf6cd7438a351f19fac60fbc0d927/ml_map.svg)
<!-- _footer:  [www.scikit-learn.org](https://scikit-learn.org/stable/) -->

---

## Contents of this class make use of the scikit-learn library
<iframe src="https://scikit-learn.org" width="1280px" height="500px" style="border:none; box-shadow: 0px 0px 20px #ccc;border-radius:15px;"></iframe>

<!-- _footer:  [www.scikit-learn.org](https://scikit-learn.org/stable/) -->

---

## Lots of resources are also taken from the _Deep learning_ book
- Very complete introduction 
- Historical aspects
- Starts from scratch (linear algebra)
- Covers machine/deep learning
- Illustrates with examples
- Online free access

![drop-shadow width:350px](https://m.media-amazon.com/images/I/A1GbblX7rOL._AC_UF1000,1000_QL80_.jpg)
<!-- _footer: www.deeplearningbook.org -->

---

<!-- _class: titlepage-->

<div>

![width:160](images/logo/cheese.png)

# 1. Introduction

__Machine learning__ for Earth science: 
Why, what, and how? Are any of those methods useful for your research? How to read papers that use machine learning?

</div>

---
## How much time do you need to describe the following images?
![width:900px](images/papers/karpathy2015deep-nocap.png)
<!-- _footer: Karpathy & Fei-Fei (2015) -->
---
## How accurate are those descriptions?
![width:900px](images/papers/karpathy2015deep.png)
<!-- _footer: Karpathy & Fei-Fei (2015) -->
---

## How do we extract such high-level knowledge from data?

- Identify objects within image
- Recognize objects category
- Understand the link between objects
- Sort links by priority
- Generate text out of it

![width:530px](images/papers/karpathy2015motivation.png)
<!-- _footer: Karpathy & Fei-Fei (2015) -->

---

## Ingredients: hierarchical knowledge extraction
![width:600px](images/deep-learning-book/figure-1-2.png)
<!-- _footer: Goodfellow et al. (2016) -->

---

## Can you spot the seismogram?
![width:500px](images/papers/valentine2012spot.png)

<!-- _footer: Valentine & Trampert (2012).<br>Top to bottom: UK stock exchange; Temperature in Central England; Gaussian noise; Long-period seismogram.-->

---

## Detection and classification of events from seismograms

<div style="flex-basis: 15%">

Most humans can pinpoint events. 

</div>
<div style="flex-basis: 40%">

![width:520](images/papers/moran2008helens-nolabels.png)

</div>
<div style="flex-basis: 7%">
<br>
</div>

<!-- _footer: Moran et al. (2008) -->


---

## Detection and classification of events from seismograms

<div style="flex-basis: 15%">

Most humans can pinpoint events. 

</div>
<div style="flex-basis: 40%">

![width:520](images/papers/moran2008helens.png)

</div>
<div style="flex-basis: 7%">

Experts can __classify__ them.

</div>

<!-- _footer: Moran et al. (2008) -->

---

## Diving into previously unseed data

Expert-detected marsquake within continuous insight data <br>
![width:1000px](images/papers/clinton2021marsquake.jpg)

<!-- _footer: Clinton et al. (2021) -->
---

## Target tasks of machine learning

- Time-consuming tasks
- Hard-to-describe tasks
- Exploration of new data

![width:500](images/datasets/zebra.jpeg)

---

<!-- _class: titlepage-->

<div>


![width:160](images/logo/cheese.png)

# 2. Definitions

__Machine learning__ is a field of study in artificial intelligence of statistical algorithms that can effectively generalize and thus perform tasks without explicit instructions. 

</div>

---

## General definition of machine learning

<div>

An algorithm learns from <span style="color:var(--color-lava)">experience</span> with respect to a <span style="color:var(--color-terra)">task</span> and <span style="color:var(--color-olivine)">performance</span>, if its <span style="color:var(--color-olivine)">performance</span> at solving the <span style="color:var(--color-terra)">task</span> improves with <span style="color:var(--color-lava)">experience</span>.

__All three elements are required.__

</div>
<div>

<svg viewBox="0 0 370 350" font-size="18px" text-anchor="middle" style="padding: 30px" font-family=var(--font)>
<circle cx="100" cy="100" r="100" fill=var(--color-magma) opacity="0.1"/>
<circle cx="260" cy="100" r="100" fill=var(--color-terra) opacity="0.1"/>
<circle cx="180" cy="240" r="100" fill=var(--color-olivine) opacity="0.1"/>
<text x="100" y="100" alignment-baseline="middle" fill=var(--color-magma)>Experience</text>
<text x="260" y="100" alignment-baseline="middle" fill=var(--color-terra)>Task</text>
<text x="180" y="240" alignment-baseline="middle" fill=var(--color-olivine)>Performance</text>
</svg>

</div>

---

## The data, the model, and the loss

<div style="flex-basis: 25%;background-color: var(--color-sable); border-radius: 40px; padding: 20px;" align=center data-marpit-fragment="0">

🙊

__the data__ 

A set of samples $\mathbf{x}_i$ and labels $\mathbf{y}_i$ to learn from:

$$\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^N$$

</div>
<div style="flex-basis: 25%;background-color: var(--color-sable); border-radius: 40px; padding: 20px;" align=center  data-marpit-fragment="0"

🙉 

__the model__ 

A parametric function $f_\theta$ that maps data $\mathbf{x}$ to  $\hat{\mathbf{y}}$ 

$$f_\theta : \mathbf{x} \mapsto \hat{\mathbf{y}}$$

</div>
<div style="flex-basis: 25%;background-color: var(--color-sable); border-radius: 40px; padding: 20px;" align=center  data-marpit-fragment="0"

🙈 

__the loss__

A measurement of the  model performance

$$\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y})$$

</div>
<div align=center  data-marpit-fragment="0"

__Learning__ = find the optimal parameters $\theta^*$ that minimize the loss $\mathcal{L}$ function

$$\theta^* = \underset{\theta}{\arg\!\min}\, \mathcal{L}\Big(f_\theta(\mathbf{x}), \mathbf{y}\Big)$$

</div>

---

## Useful vocabulary and symbols

| Symbol | Name |
|:-|:-|
|$\left\{ \mathbf{x}_i \in \mathbb{X} \right\}_{i =  1\ldots N}$| Collection of __samples__|
|$\left\{ \mathbf{y}_i \in \mathbb{Y} \right\}_{i =  1\ldots N}$| Collection of __labels__|
|$\mathbf{x}=(x_1, \ldots, x_F)$| Set of sample __features__|
|$\mathbf{y}=(y_1, \ldots, y_T)$| Set of label __targets__|
|$N$| Dataset size|
|$F$| Feature space dimensions|
|$T$| Target space dimension|
|$\mathbb{X}$| Data space|
|$\mathbb{Y}$| Label space|

<div>

An image is a sample $\bf x$ with 
$$\mathbf{x} \in \mathbb{X} = \mathbb{R}^{H \times W \times C}$$
$H$ is the height, $W$ the width, and $C$ the channels. The labels are a category $y$ with
$$y \in \mathbb{Y} = \{0, 1, \ldots, K\}$$
with $K$ the number of categories. 
Note that $y$ is scalar in this case.

</div>

---

## Main types of learning

<div style="flex-basis: 25%; font-size:smaller;" align=center data-marpit-fragment="0">

![width:265px](images/diagrams/mathworks-supervised.png)

Predict $\mathbf{y}$ from $\mathbf{x}$ (regression, classification).

</div>
<div style="flex-basis: 25%; font-size: smaller" align=center data-marpit-fragment="1">

![width:250px](images/diagrams/mathworks-unsupervised.png)

Learn some distribution $p(\mathbf{x})$ (clustering, reduction).

</div>
<div style="flex-basis: 25%; font-size:smaller; opacity: 0.25" data-marpit-fragment="2" align=center>

![width:265px](images/diagrams/mathworks-reinforcement.png)

Learn a policy to maximize a reward (gaming, robotics).

</div>

<!-- _footer: from [mathworks.com](https://nl.mathworks.com/discovery/reinforcement-learning.html) -->

---

<!-- _class: titlepage-->


<div>

![width:160](images/logo/cheese.png)

# 3. Regression

How to solve a regression or classification task with machine learning?

</div>

---

## The two main tasks of supervised learning

<div align=center>

__Regression__
$x$ and $y$ are continuous

![width:300px](images/supervised/linear_regression.svg)

</div>
<div align=center>

__Classification__
$x$ is continuous and $y$ is descrete 

![width:300px](images/supervised/linear_classification.svg)

</div>

---

## The two main tasks of supervised learning

<div align=center>

__Regression__
$x$ and $y$ are continuous

![width:300px](images/supervised/linear_regression.svg)

</div>
<div align=center style="opacity: 0.3;">

__Classification__
$x$ is continuous and $y$ is descrete 

![width:300px](images/supervised/linear_classification.svg)

</div>

--- 

## The regression task

<div>

__Dataset:__ set of $N$ samples $x_i$ and corresponding labels $y_i$ such as 

$$\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$$

__Regression:__ optimize the parameters $\theta$ of a function $f_\theta$ to predict $y$ from $x$. Find the optimal parameters $\theta^*$ that minimize $\mathcal{L}$, such as
$$\theta^* = \underset{\theta}{\arg\!\min }\mathcal{L}\Big(f_\theta(x), y\Big).$$

</div>

![width:400px](images/supervised/linear_regression_math.svg)

---

## The linear regression

<div>

__Linear model:__ coefficients $\theta = (a, b) \in \mathbb{R}^2$ that map $x$ to $y$ with 

$$f_\theta : x \mapsto y= ax + b.$$

__Loss function:__ mean squared error (example), given by 

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \left( f_\theta(x_i) - y_i \right)^2.$$

__How do we minimize the loss?__

</div>

![width:400px](images/supervised/linear_regression_math.svg)


---

## Naive attempt: grid search

<div>

__Grid search:__ find $\theta^*$ among tested values of $\theta$. 

__Pros:__ easy to implement, exhaustive search, uncertainty estimation.

__Cons:__ unscalable. If 0.1s / evaluation, then 2 parameters with 100 values each takes 1/4 hour. _For 5 parameters it takes more than 30 years!_

__Any smarter idea?__

</div>

![width:400px](images/supervised/linear_regression_brute_force.svg)

---

## Random search

<div>

__Random search__ to find $\theta^*$. 

__Pros:__ easy to implement, scalable, uncertainty estimation, can include prior knowledge.

__Cons:__ not exhaustive, can be slow to converge.

</div>

![width:400px](images/supervised/linear_regression_random.svg)

---

## Gradient descent

<div>

__Idea:__ estimate the gradient of $\mathcal{L}$ w.r.t. the parameters $\theta$, update the parameters towards gradient descent.

__Pros:__ converges faster than random search.

__Cons:__ gets stuck in local minima, slow to converge, needs differentiability.

</div>

![width:400px](images/supervised/linear_regression_gradient_descent.svg)


---

## Gradient descent

<div>

__Recipe__

1. Initial model $\theta = (a_0, b_0)$
1. Compute the gradient $\nabla \mathcal{L}(\theta)$
1. Update the model $\theta \leftarrow \theta - \eta \nabla \mathcal{L}(\theta)$
1. Repeat until convergence

__Hyperparameters__

The __learning rate__ $\eta$ is the update step.

</div>

![width:500px](images/supervised/gradient_descent_3d.svg)


---

## How to deal with learning rate?
![width:950px](images/supervised/learning_rate.svg)
That's part of the __hyperparameters tuning__.
More about that in the deep learning lectures.

---

## The problem of overfitting
![width:900px](images/supervised/overfitting.svg)
Having a loss close to 0 does not mean that the model __generalizes__ well.

---

## Key concepts to prevent overfitting: split the dataset
![width:900px](images/supervised/splitting.svg)
By splitting the dataset into a __training__ and a __testing__ set, 
we evaluate the performance on unseen (but __similar__) data. 

--- 

## Key concepts to prevent overfitting: regularization
Add a penalty term $\mathcal{R}$ to the loss $\mathcal{L_R} = \mathcal{L} + \lambda \mathcal{R}$, with $\lambda$ the regularization strength<br>
![width:900px](images/supervised/regularization.svg)
The regularization penalizes the model's complexity. 

---

## Why so many regression algorithms?

<div>

Because of combination of models, losses, and regularizations. The [scikit-learn.org](https://scikit-learn.org/stable/) website provides a unified interface in a `greybox style`. <br><br>The model selection is made by experience or __trial and error__.

</div>

<iframe src="https://scikit-learn.org/stable/supervised_learning.html#supervised-learning" width="500px" height="500px" style="border: none; box-shadow: 0px 0px 20px #ccc; border-radius: 10px;"></iframe>

<!-- _footer:  [www.scikit-learn.org](https://scikit-learn.org/stable/) -->

---

## Guidelines for exploring relevant models

![width:900](https://scikit-learn.org/stable/_downloads/b82bf6cd7438a351f19fac60fbc0d927/ml_map.svg)

<!-- _footer:  [www.scikit-learn.org](https://scikit-learn.org/stable/) -->

--- 

<!-- _footer: Jupyter $\\times$ Obsera -->

## Notebook: expansive data from cheap sensors

![width:530px](images/notebooks/lab_1_sensor_calibration.svg) 

<img src="images/logo/obsera.png" width=110px style="position:absolute; right:90px; bottom:40px;">

![width:500px](images/notebooks/lab_1_picture_of_river.png)<br><br>


---

<!-- _footer: Jupyter $\\times$ PhaseNet -->

## Find out _P_ and _S_ waves within continuous seismograms

How do you addess this regression problem?<br>
![width:600px](images/papers/zhu2018phasenet.png)

---

<!-- _class: titlepage-->

<div>

![width:160](images/logo/cheese.png)

# 4. Supervised machine learning: classification

How to solve a regression or classification task with machine learning?

</div>

---

## The two main tasks of supervised machine learning

<div align=center style="opacity: 0.3;">

__Regression__
$x$ and $y$ are continuous

![width:300px](images/supervised/linear_regression.svg)

</div>
<div align=center>

__Classification__
$x$ is continuous and $y$ is descrete 

![width:300px](images/supervised/linear_classification.svg)

</div>

---

<!-- _footer: www.scikit-learn.org -->

## The classification task

<div align=center>

![](images/supervised/classification.svg)

Here again, we have many possibilities.

</div>

---

## The classification task

<div>

__Experience__: labels $y \in \{0, 1\}$ for two features $\mathbf{x} \in \mathbb{R}^2$.

__Task__: predict $\hat{y}$ of each sample $\mathbf{x}$.

__Performance__: how should we measure the performance of a classifier?

</div>

![width:600px](images/supervised/svc.svg)

---

<!-- _footer: https://en.wikipedia.org/wiki/Support_vector_machine -->

## The classification task with support vector machines (SVM)

<div>

Support vector machines search the hyperplane of normal vector $\mathbf{w}$ and bias $b$ that split the classes.

> Note: in 2D, a hyperplane is a line.

The support vectors are the samples that are closest to the other class.


</div>

![width:400px](https://upload.wikimedia.org/wikipedia/commons/7/72/SVM_margin.png)

---

## The classification task with support vector machines (SVM)

<div>

The decision function $f(\mathbf{x})$ dependson  the sign of the linear combination of the normal vector and the sample:

$$f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$$

The quantity to minimize is the __Hinge loss__:

$$\mathcal{L}(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^N \max\left(0, 1 - y_i \left(\mathbf{w} \cdot \mathbf{x}_i + b\right)\right)$$

<br>

</div>

![width:500px](images/supervised/svc.svg)

---

## The classification task with support vector machines (SVM)

<div>

The decision function $f(\mathbf{x})$ depends on the sign of the linear combination of the normal vector and the sample:

$$f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$$

The quantity to minimize is the __hinge loss__:

$$\mathcal{L}(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^N \max\left(0, 1 - y_i \left(\mathbf{w} \cdot \mathbf{x}_i + b\right)\right)$$

__What about non linear problems?__

</div>

![width:500px](images/supervised/svc.svg)

--- 

<!-- _footer: www.medium.com -->

## The kernel trick for non linear classification problems


The kernel trick allows to map the data to a __higher dimensional__ space
made from the input features where the problem is __linearly separable__. <br>
![width:700px](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*mCwnu5kXot6buL7jeIafqQ.png)
The __Radial Basis Functions__ (RBF) is an infinite kernel $K(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\sigma^2}\right)$

---

## Generalization of the SVM: the support vector classifier (SVC)


The SVC is a generalization of the SVM that digests more than two classes. <br> 
![width:1000px](images/supervised/svc_multiclass.svg)
The decision function is linear in the kernel space only. 
We can project it back to the data space to inspect it.

</div>

---

## Various classifcation metrics from the confusion matrix
![width:800px](images/supervised/svc_multiclass.svg)
![width:810px](images/supervised/svc_confusion.svg)

---

## Various classifcation metrics: accuracy, precision, recall


![width:900px](https://www.researchgate.net/publication/336402347/figure/fig3/AS:812472659349505@1570719985505/Calculation-of-Precision-Recall-and-Accuracy-in-the-confusion-matrix_W640.jpg)

<!-- _footer: Ma et al. (2019) -->

---

## Decision trees and random forests

<div>

__Decision trees__ learn to predict $y$ with feature splitting.

__Random forests__ are ensembles of decision trees that vote for $y$.

__These algorithms are extremely powerful.__

</div>

![width:600px](https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_dtc_002.png)

<!-- _footer: from [scikit-learn.org](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html) -->

---

## Representation matters!

![width:700px](images/deep-learning-book/figure-1-1.png)
There is no need for a complex model if you have a good __representation__ of the data.

---

## Learning strategies depending on the task complexity

<div>

Hand-designed or learned features?

</div>

![width:750px](images/deep-learning-book/figure-1-5.png)

<!-- _footer: Goodfellow et al. (2016) -->

---

<!-- _footer: still from Valentine & Trampert (2012) -->

## Why would waveforms not be the best representation of ground motion?

<div>

We can see waveforms $\mathbf{x}\in\mathbb{R}^N$ as points of a $N$-dimensional space 

<img src="images/waveforms/waveform_0.png" width=700/>

Yet, seismic waveform do not occupy this space fully, likely very sparse.

__Dimension > Information__

</div>

![width:520px](images/papers/valentine2012spot.png)

---

## Actually, natural data spaces may be often not fully occupied

<div>

Random sampling of the pixels of a face. What is the likelihood that the reshuffled image _is_ a face?

<br>

![width:120](images/deep-learning-book/figure-x-x-1.jpg)

<br>

Like waveforms, __images are living on a manifold.__

</div>

![width:550](images/deep-learning-book/figure-x-x-2.png)

<!-- _footer: modified from Goodfellow et al. (2016) -->

---

## Supervised learning for sismo-volcanic signal classification


__Supervised learning__ experiences a set of examples containing features $\mathbf{x}_i \in \mathbb{X}$ associated with labels $\mathbf{y} \in \mathbb{Y}$ to be predicted from the features (here, classification). <br>
<img src="images/examples/malfante_2018.png" width=900/>

<!-- _footer: Malfante et al. (2018) -->
---

## Supervised learning for sismo-volcanic signal classification

In this case, $\mathbf{x}$ lies in $\mathbb{R}^{3 \times N}$, and $\mathbf{y}$ in $[0, \ldots, 5]$. 
Which __representation__ of $\mathbf{x}$ works best?
<img src="images/examples/malfante_2018.png" width=900/>
<!-- _footer: Malfante et al. (2018) -->

---

## Handcrafted features for classical machine learning

<div>

We need to find relevant descriptors of our data, used as features $\mathbf{x}$.
<br>

<img src="images/examples/features_signal.png" width=500/>

</div>

![width:600](images/examples/features.png)

<!-- _footer: Jasperson et al. (2022) -->

---

## Performance measure, and what can we learn from it?
Accuracy of the predictions measures the model's performance (= confusion matrix) <br><br> <img src="images/examples/malfante_accuracy.png" width=800/><br>
What is the guarantee that the features we choose are the best ones?

<!-- _footer: Malafante et al. (2018) -->

---

## Notebook: Lidar point cloud classification

<div>

__Problem:__ automate the identification of objects in a lidar cloud from labeled subset.

__Objectives:__ supervised learning, classification, non-linear models, multi-scale features.

</div>
<div>

![width:700px](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTomIqR3vtw9M5K2TyhvAtLKsSb8gGOGmxQcn0lHdoWO4uGjl3MxQ00TJgi2pvO7jRq2rs&usqp=CAU)

</div>

---

<!-- _class: titlepage-->

<div>

![width:160](images/logo/cheese.png)

# 5. Deep learning: multi-layer perceptrons

How deep learning works? What is a neural network? How to train it, and what for?

</div>

---

## A general form of an artificial neuron

<div>

A __neuron__, or unit, takes a set of inputs $\bf x$ and outputs an activation value $h$, as
$$
h = \varphi\left(\sum_{i=1}^{N} w_i x_i + b \right)
$$
with $w_i$ the weights, $b$ the bias, $\varphi$ is the activation function, and $N$ is the number of inputs.

</div>

![](images/models/neuron.png)

---

## A famous neuron: the sigmoid unit

<div>

A __neuron__, or unit, transforms a set of inputs $\bf x$ into an output $h$, as
$$
h = \varphi\left(\sum_{i=1}^{N} w_i x_i + b \right)
$$
with $w_i$ the weights, $b$ the bias, $\varphi$ is the activation function, and $N$ is the number of inputs. Common activation functions include the __sigmoid__ function, defined as
$$
\varphi(z) = \frac{1}{1 + e^{-z}}
$$

</div>

![width:500](images/models/sigmoid.png)

---

## An even more famous one: the rectified linear unit

<div>

A __neuron__, or unit, transforms a set of inputs $\bf x$ into an output $h$, as

$$
h = \varphi\left(\sum_{i=1}^{N} w_i x_i + b \right)
$$

with $w_i$ the weights, $b$ the bias, $\varphi$ is the activation function, and $N$ is the number of inputs. Common activation functions include the __rectified linear unit__ (ReLU), defined as

$$
\varphi(z) = \max(0, z)
$$

<!-- _footer: ReLU are empirically preferred to sigmoid units for  computational efficiency no saturation when $x$ is large.-->

</div>

![width:500](images/models/relu.png)

---

## The multilayer perceptron (MLP)

<div>

A __multilayer perceptron__ is a neural network with multiple hidden layers:
$$
\begin{align*}
h_i^{(1)} &= \varphi^{(1)}\left(\sum_j w_{ij}^{(1)}x_j + b_i^{(1)}\right)\\
h_i^{(2)} &= \varphi^{(2)}\left(\sum_j w_{ij}^{(2)}h_j^{(1)} + b_i^{(2)}\right)\\
y_i &= \varphi^{(3)}\left(\sum_j w_{ij}^{(3)}h_j^{(2)} + b_i^{(3)}\right)
\end{align*}
$$

</div>

![width:500](images/models/mlp_annotated.png)

---

## The multilayer perceptron (MLP)

<div>

A __multilayer perceptron__ is a neural network with multiple hidden layers. Generally speaking (omitting the biases):
$$
y = \varphi^{(\ell)}\left(\mathbf{W}^{(\ell)}\varphi^{(\ell - 1)}\left(\mathbf{W}^{(\ell - 1)} \ldots \varphi^{(1)}\left(\mathbf{W}^{(1)}\mathbf{x}\right) \ldots \right)\right)
$$
</div>

![width:500](images/models/mlp_annotated.png)

---

## Quick example for solving the XOR problem

<div>

Multi-layer perceptrons that solves the XOR problem with binary activations:

![width:300](images/models/xor.png)

</div>

![width:500](images/datasets/xor.png)


<!-- _footer: See Section 6.1 of Goodfellow et al. (2016) -->

---

## Gradient descent for neural networks

<div>

We note $f_\theta(x): x \mapsto y$ the model, where $\theta$ are the parameters of the model (including biases and weights).

1. __Learning__ is the process of finding the parameters $\theta^*$ that minimize the loss $\mathcal{L}$.

2. The __backpropagation__ computes the loss function gradient with respect to $\theta$.

3. The __gradient descent__ updates $\theta$ in the direction of the steepest descent.

</div>

![width:450](images/models/gd.gif)

---

## Gradient computation with backpropagation

1. __Initialization__: the weights are initialized randomly, the biases to zero
2. __Feed forward__: the input is propagated through the network to compute the output
3. __Loss__: the loss is computed between the output and the target
4. __Back propagation__: computation of the gradient from the loss to the input
5. __Gradient descent__: update the parameters in the direction of the steepest descent

---

## Gradient-based optimization

<div>

Once the gradient is computed, the parameters are updated using the __gradient descent__ algorithm:

$$
\begin{align*}\\
\theta &\leftarrow \theta - \eta \frac{\partial \mathcal L}{\partial \theta}
\end{align*}
$$

where is $\eta$ the __learning rate__ that controls the size of the update.

</div>

![width:450](images/models/gd.gif)

---

## Gradient descent common issues

<div>

- __Local minima__: getting stuck in a local minimum.

- __Sattling points__: behaves as a local minimum but is not.

- __Plateau__: flat loss function, vanishing gradient, slow convergence.

</div>

<img src="images/models/gd_issues.png">

---

## Gradient descent common issues with plateau

<img src="images/models/relu.png" height=350 align="left"/>
<img src="images/models/sigmoid.png" height=350 align="left"/>

__Plateau__ are flat regions of the loss function where the gradient is zero. This can happen with activation functions such as the sigmoid function with saturation. It can also happen with the ReLU function for inputs with negative values.

---

## Gradient-descent tricks to avoid issue

- __Learning rate__: set up, and maybe adapt it. 
- __Momentum__: use the gradient of the previous iteration to update the parameters.
- __Normalization__: normalize the inputs of each layer.
- __Stochastic gradient descent__: use a mini-batch of samples to compute the gradient.
- __Dropout__: randomly drop some neurons during training.

---

## Gradient descent and learning rate

<div>

The __learning rate__ is a hyperparameter that controls the size of the update of the parameters:

$$
\theta \leftarrow \theta - \eta \cfrac{\partial \mathcal L}{\partial \theta}
$$

We must look for a learning rate to avoid local minima while still converging fast enough, without diverging.

> We can also __adapt__ the learning rate.

</div>

![width:480](images/models/lr.png)


---

## Gradient descent and momentum

<div>

The __momentum__ is a technique to accelerate the gradient descent by adding a fraction of the gradient of the previous iteration:
$$
\begin{align*}
p &\leftarrow \alpha p - \eta \frac{\partial \mathcal L}{\partial \theta}\\
\theta &\leftarrow \theta + p
\end{align*}
$$
where $\alpha$ is the a damping parameter, and $v_i$ is the __velocity__. Lower values of $\alpha$ give more weight to the current gradient, higher values give more weight to the previous gradients.

</div>

![width:500](images/models/mom.webp)



<!-- _footer: From Zhang et al. (2021) -->

---

## Data normalization

<div

To avoid getting in the saturation of sigmoidal activation functions, it is important to normalize the data. This can be done by __normalizing the input and the features__:

$$
\hat x_i = \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}
$$

where $\mu_i$ is the mean of the input, $\sigma_i$ is the standard deviation of the input, and $\epsilon$ is a small constant to avoid division by zero. You can also apply the normalization after the activation function.

</div>

---

## Monitor the training curves

<div>

The __training curves__ are a good way to monitor the training of a model.

- Slow: increase the learning rate.
- Growing: decrease the learning rate.
- Cross-validation: within 0.0001 to 0.1

</div>

![width:450](images/models/learning_curve.png)

---

## Stochastic gradient descent

<div>

The gradient of the loss function with respect to the parameters $\theta$ is computed using the __full-batch gradient descent__ equal to:

$$
\frac{\partial \mathcal L}{\partial \theta} = \frac{1}{N} \sum_{i=1}^N \frac{\partial \mathcal L^{(i)}}{\partial \theta}\\
$$

The __stochastic gradient descent__ is a technique to compute the loss gradient from every sample in the dataset at each iteration.

</div>

![](images/models/sgd.png)

---

## Mini-batch gradient descent

<div>

The __mini-batch gradient descent__ is a technique to compute the gradient of the loss function with respect to a subset of the dataset. It is a compromise between the full-batch gradient descent and the stochastic gradient descent.

<br>

![](images/models/mb.png)

---

## Overfitting and underfitting

<div>

__Overfitting__: too complex model, does not generalize to new data.

__Underfitting__: too simple model, does not capture the data structure.

</div>

![width:750](images/models/fitting_mod.png)

---

## Splitting the dataset into train and test sets

<div>

The __training set__ is used to train the model. The __test set__ is used to evaluate the model generalization error on unseen data.

> The typical split is 80% for the training set and 20% for the test set.

</div>

![width:500](images/models/train_test_split.png)

---

## Training and test learning curves

<div>

We must ensure that both the training and test losses decrease. If the training loss is much lower than the test loss, the model __overfits__ the training set.

![width:500](images/models/train_test_split.png)

</div>

![width:450](images/models/tt_loss.png)

---

## Targetting the right model complexity

<div>

The __model complexity__ is roughly the number of parameters of the model. The __model generalization error__ is the error on the test set.

<img src="images/models/complexity.png" width=70%/>

</div>

---

## Regularization

<div>

__Regularization__ is a technique to control overfitting by adding a penalty term $\mathcal{R}$ to the loss function. The __regularization parameter__ $\lambda$ controls the strength of the regularization.

$$
\mathcal{L}_\mathrm{reg} = \mathcal{L} + \lambda \mathcal{R} = \mathcal{L} + \lambda \|\mathbf{\theta}\|^2_2
$$

</div>

![width:650](images/models/wd.png)

<!-- _footer: From Goodfellow et al. (2016) -->

---

## A fully connected network for solving the MNIST classification

<!-- _footer: LeCun et al. (1998) -->

Handwritten digits set of grayscale images $x \in \mathbb{R}^{28 \times 28}$ and classes $y \in \{0, \dots, 9\}$.<br>
![width:700](images/datasets/mnist.png)
__Goal__: predict the number encoded in the pixels.


---

<!-- _class: dark -->
<!-- _footer: adamharley.com (Harley, 2015) -->

## A fully connected network for solving the MNIST classification

<iframe src="https://adamharley.com/nn_vis/mlp/3d.html" width="1280" height="630" frameborder="0" allowfullscreen style="position:fixed;bottom:0;zoom:100%;"></iframe>

---

## Example in seismology: fully-connected autoencoder

<!-- _footer: from Valentine and Trampert (2012) -->

<div>

Training of a fully-connected autoencoder on real seismic data. 

This is an __unsupervised__ learning task: the input and output are the same.

![width:400](images/examples/valentine_2.png)

</div>

![width:600](images/examples/valentine_ae.png)

---

## Example in seismology: fully-connected autoencoder

<!-- _footer: from Valentine and Trampert (2012) -->

<div>

We __learned__ a low-dimensional representation for the seismic data.

</div>

![width:500](images/examples/valentine_1.png)

<div>

These are the __latent variables__ of the autoencoder.

</div>

---

## Example in seismology: fully-connected autoencoder

<!-- _footer: from Valentine and Trampert (2012) -->

<div>

Example applications:
- quality assessment
- compression

</div>

![width:700](images/examples/valentine_3.png)

---

<!-- _class: titlepage-->

<div>

![width:160](images/logo/cheese.png)

# 6. Deep learning: convolutional neural networks

How deep learning works? What is a neural network? How to train it, and what for?

</div>

---

## Limitations of fully connected networks

<div>

Vision is robust to a lot:

- Translation
- Rotation
- Scaling
- Shearing
- Illumination
- Occlusion

We need invariance to these transformations.

</div>

![](images/models/invariance.jpg)

---

## Example: the handwritten digits

<!-- _footer: LeCun et _al._ (1998) -->

Handwritten digits set of grayscale images $x \in \mathbb{R}^{28 \times 28}$ and classes $y \in \{0, \dots, 9\}$.<br> 
<img src="images/datasets/mnist.png" style="width: 70%;">

---

## Limitations of fully connected networks


<div>

An image may be of $200 \times 200$ pixels $\times 3$ color channels. With a __fully connected network__ with $1000$ hidden units, we would have $N = 200 \times 200 \times 3 \times 1000 = 120$M parameters. <br><br>__This clearly does not scale to large images.__

</div>

![width:500](images/models/densely.png)

---

## Convolutional neural networks

<div>

__Convolutional layers__ are a type of layer that are used in convolutional neural networks. They are composed of a set of learnable filters.

![width:900](images/models/convlay.png)

Each hidden unit look a local content from the input image, althought the weights are shared across the entire image.

</div>

---

## Convolutional neural networks

<div>

Discrete image convolution:

$$ (A * B)_{ij} = \sum_n \sum_m A_{nm}B_{i-n, j-m} $$

where $A$ is a input image, and $B$ is a convolutional kernel (weights) to learn.

> Convolutional layers extract local features from the input image ≠ fully connected layers that extract global features.

</div>

![](images/models/no_padding_no_strides.gif)

<!-- _footer: From Vincent Dumoulin, Francesco Visin (2016) -->

---

## Convolution operation

<div>

$$(A * B)_{ij} = \sum_n \sum_m A_{nm}B_{i-n, j-m}$$
![](images/datasets/zebra_filtered.jpeg)

</div>

---

## Convolution operation

<div>

$$(A * B)_{ij} = \sum_n \sum_m A_{nm}B_{i-n, j-m}$$
![](images/datasets/zebra_edges.jpeg)

</div>

---

## Convolution unit

![](images/models/zebra_conv.png)

---

## Convolutional neural network: example with VGG16


Now, we can understand this winning architecture for image classification.<br>
![width:850](images/models/vgg16.png)
Note the last three layers are __fully connected__.
When extracting low-dimensional data from images, this is often needed.

</div>

---

## Convolutional neural network: example with VGG16

<!-- _footer: from Zeiler and Fergus (2013) -->

<div>

Here are the __filters from the first layer__ of VGG16 after training on 100k+ images. These filters collect various shapes, scales, colors, etc.

![](images/models/vgg16.png)

</div>

![width:500](images/models/vgg_layer_1.png)

---

<!-- _class: dark -->
<!-- _footer: adamharley.com (Harley, 2015) -->

## A convolutional network for solving the MNIST classification

<iframe src="https://adamharley.com/nn_vis/cnn/3d.html" width="1280" height="630" frameborder="0" allowfullscreen style="position:fixed;bottom:0;zoom:100%;"></iframe>

---

<!-- _class: titlepage-->

<div>

![width:160](images/logo/cheese.png)

# 7. Applications

The illustration of the previous concepts with examples from seismology. And then you will be ready to apply these concepts to your own problems!

</div>

---

## Deep-learning applications in seismology


- Signal detection, pattern recognition
- Classification
- Source localization from sparse or evolving datasets
- Denoising and compression

![](images/references/seismic_signal_class.png)

---

## Earthquake detection and location with ConvNetQuake

<div>

__Features__: 3-comp. waveform $x \in \mathbb{R}^{N \times 3}$
__Target__: prob. of event in cell $1$ to $6$
__Loss__: cross-entropy with regularization $\mathcal{L} = - \sum_c q_c \log p_c + \lambda \| \mathbf{w}\|^2_2$ 
<br>

![width:400](images/models/perol_2.png)

</div>

![width:500](images/models/perol_1.png)


<!-- _footer: From Perol et al. (2016) -->

---

## Seismic phase picking with PhaseNet

<div>

__Features__: 3-component seismic signal $x \in \mathbb{R}^{3000 \times 3}$. 

<br> 

__Targets__: probabilities $p_i(x)$ of $P$, $S$, and $N$oise over time $= y \in \mathbb{R}^{3000 \times 3}$

</div>

![width:700](images/models/beroza_example.png)


<!-- _footer: From Zhu et al. (2016) -->

---

## Seismic phase picking with PhaseNet

__Features__: 3-component seismic signal $x \in \mathbb{R}^{3000 \times 3}$
__Targets__: probabilities $p_i(x)$ of $P$, $S$, and $N$oise over time $= y \in \mathbb{R}^{3000 \times 3}$
__Loss__: cross-entropy $\mathcal{L} = -\sum_i\sum_x p(x)\log(q(x))$<br>
![width:800](images/models/unet_phasnet.jpg)

</div>

<!-- _footer: From Zhu et al. (2016) -->

---

## Seismic phase picking with PhaseNet

__Features__: 3-component seismic signal $x \in \mathbb{R}^{3000 \times 3}$
__Predictions__: likelihood $q_i(x)$ of $P$, $S$, and $N$oise over time<br>
![](images/models/beroza_2.png)

<!-- _footer: From Zhu et al. (2016) -->

---

## Seismic phase picking with PhaseNet

__Features__: 3-component seismic signal $x \in \mathbb{R}^{3000 \times 3}$
__Predictions__: likelihood $q_i(x)$ of $P$, $S$, and $N$oise over time<br>
![](images/models/beroza_3.png)

<!-- _footer: From Zhu et al. (2016) -->

---

## Transfer learning and fine-tuning

<div>

__Transfer learning__ is the use of a pre-trained model $f_\alpha = f_{\theta^*}$ on a new task as a initial point for training a new model $f_\alpha \rightarrow f_{\alpha^*}$.

__Fine-tuning__ is the partial re-training of a pre-trained model on a new task, while keeping the weights of the pre-trained layers fixed.

</div>

![width:550](images/examples/scedc_mapplot.png)

---

## Deep-learning libraries in Julia and Python

<div style="flex-basis: 25%;">

### Warning

Libraries are constantly evolving, and the documentation is often incomplete.

</div>
<div style="flex-basis: 50%; columns:2;">

<img src="images/examples/logo_sklearn.png" width=200/>
<br>
<img src="images/examples/logo_tf2.png" width=300/>
<br>
<img src="images/examples/logo_keras.png" width=250/>
<br>
<img src="images/examples/logo_pytorch.png" width=250/>
<br>
<img src="images/examples/logo_julia.png" width=150/>
<br>
<img src="images/examples/logo_seisbench.svg" width=400/>

</div>


---

## Dive into the scikit-learn toolbox documentation

<!-- _footer: Online documentation at [scikit-learn.org](https://scikit-learn.org/stable) -->

<div style="flex-basis: 30%;">

### Scikit-Learn toolbox documentation

- Machine learning in Python
- Online examples
- Explanation of algorithms
- Grey-box models

</div>
<div style="flex-basis: 50%;">

<iframe src="https://scikit-learn.org/stable/" width="700px" height="520px" style="border: none; box-shadow: 0px 0px 20px #ccc; border-radius: 5px;zoom:1"></iframe>

</div>

---

## Learning visually with the TensorFlow playground

<div>

<iframe src="https://playground.tensorflow.org/" width="1280px" height="849px" style="zoom: 1; border: none; top: -130px; left: 0px; position: absolute; z-index: -10;"></iframe>

</div>

<!-- _footer: https://playground.tensorflow.org -->

---

## Interesting online projects

<div>

<iframe src="https://quickdraw.withgoogle.com/data" width="1280px" height="650px" style="zoom: 1; border: none; margin-top: -130px !important;"></iframe>

</div>

<!-- _footer: https://quickdraw.withgoogle.com/data -->

---

<!-- _class: titlepage -->

<div>

![width:160](images/logo/cheese.png)

# 8. Unsupervised learning

</div>

---

## Main types of machine learning

<div style="flex-basis: 25%; font-size:smaller;" align=center>

![width:265px](images/diagrams/mathworks-supervised.png)

Predict some output $\mathbf{y}$ from input $\mathbf{x}$ (regression, classification).

</div>
<div style="flex-basis: 25%; font-size: smaller" align=center>

![width:250px](images/diagrams/mathworks-unsupervised.png)

Learn data distribution $p(\mathbf{x})$ or structure (clustering, reduction).

</div>
<div style="flex-basis: 25%; font-size:smaller; opacity: 0.5">

![width:265px](images/diagrams/mathworks-reinforcement.png)

Learns a policy to maximize the reward (game playing, robotics).

</div>

<!-- _footer: from [mathworks.com](https://nl.mathworks.com/discovery/reinforcement-learning.html) -->

---

## Contents of this class make use of the scikit-learn library

![width:850px](https://scikit-learn.org/stable/_downloads/b82bf6cd7438a351f19fac60fbc0d927/ml_map.svg)

<!-- _footer:  [www.scikit-learn.org](https://scikit-learn.org/stable/) -->

---

<!-- _footer: from [researchgate.net](https://www.researchgate.net/publication/351953193_Supervised_and_unsupervised_machine_learning_a_Schematic_representation_of_an_artificial_neural_network) -->

## Unsupervised learning: learning the structure of the data without labels

<div align=center>

__Clustering__

<img src="images/unsupervised/clustering_scheme.png" width=68%/>

Group similar data points together based on some similarity measure.

</div>
<div align=center>

__Dimensionality reduction__

<img src="images/unsupervised/reduction_scheme.png" width=80%/>

Find a low-dimensional representation of the data.

</div>

--- 

## Clustering – class-membership identification without labels

![width:800px](images/unsupervised/cluster_comparison.svg)

<!-- _footer: from [scikit-learn.org](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html) -->

---

## Definitions of clustering

<div style="flex-basis: 40%;">

![width:300px](images/unsupervised/blobs.svg)

</div>
<div style="flex-basis: 50%;">

Two main definitions of clustering:

- Top-bottom: partition the heterogeneous data into __homogeneous__ subsets.

- Bottom-up: group the data samples based on some criterion of __similarity__.

`->` We need to provide a definition of __similarity__ or __homogeneity__.

</div>

---

## Example of clustering with _k_-means

<div style="flex-basis: 40%;">

![width:300px](images/unsupervised/blobs_kmeans.svg)

</div>
<div style="flex-basis: 50%;">

$k$-means is a clustering algorithm that partitions the data into $k$ clusters by minimizing the __inertia__:

$$
\arg\min_{\mathbf{C}} \sum_{i=1}^{k} \sum_{\mathbf{x}_j \in C_i} \|\mathbf{x}_j - \mu_i\|^2
$$

where $\mu_i$ is the centroid of cluster $C_i$.

</div>

--- 

## Example of clustering with _k_-means

<div style="flex-basis: 50%;">

In practice, we need to provide the number of clusters $k$, and the algorithm will find the best partition:

1. Initialize the centroids $\mu_i$ randomly.
2. Assign each sample to the closest centroid.
3. Update the centroids with the new samples.
4. Repeat until convergence.

</div>
<div style="flex-basis: 40%;">

![width:450px](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/K-means_convergence.gif/617px-K-means_convergence.gif)

</div>

<!-- _footer: Wikipedia -->

---

## Example of clustering with spectral clustering

<div style="flex-basis: 40%;">

Spectral clustering uses the eigenvectors of the similarity matrix to find the clusters. 

1. Compute the adjacency matrix.
2. Compute the Laplacian matrix. 
3. Get the first $m$ eigenvectors.
4. Cluster the data using $k$-means.

</div>
<div style="flex-basis: 37%;">

![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs00332-022-09863-0/MediaObjects/332_2022_9863_Figa_HTML.png?as=webp)

</div>

---

<!-- _footer: from [researchgate.net](https://www.researchgate.net/publication/351953193_Supervised_and_unsupervised_machine_learning_a_Schematic_representation_of_an_artificial_neural_network) -->

## Unsupervised learning: learning the structure of the data without labels

<div align=center>

__Clustering__

<img src="images/unsupervised/clustering_scheme.png" width=68%/>

Group similar data points together based on some similarity measure.

</div>
<div align=center>

__Dimensionality reduction__

<img src="images/unsupervised/reduction_scheme.png" width=80%/>

Find a low-dimensional representation of the data.

</div>

---

## Eigenvectors of a matrix

<div style="flex-basis: 40%;">

The eigenvectors of a matrix are obtained by solving the eigenvalue problem:

$$
\mathbf{A}\mathbf{x} = \lambda \mathbf{x}
$$

where $\mathbf{A}$ is a square matrix, $\mathbf{x}$ is the eigenvector, and $\lambda$ is the eigenvalue.

</div>
<div style="flex-basis: 37%;">

![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Eigenvalue_equation.svg/1200px-Eigenvalue_equation.svg.png)

</div>

---

## Singular value decomposition

<div style="flex-basis: 40%;" align=center>

The singular value decomposition (SVD) of a matrix $\mathbf{A}$ is defined as:
$$
\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T
$$
where $\mathbf{U}$ and $\mathbf{V}$ are orthogonal matrices, and $\mathbf{\Sigma}$ is a diagonal matrix.
![width:700px](https://www.askpython.com/wp-content/uploads/2020/11/SVD-1.jpg.webp)

</div>

---

## Example of dimensionality reduction with principal component analysis

<div style="flex-basis: 47%;">

Principal component analysis (PCA) is a dimensionality reduction technique that uses eigendecomposition:

1. Compute the data covariance matrix.
2. Compute the eigenvectors.
3. Project the data onto the eigenvectors.

</div>
<div style="flex-basis: 37%;">

![](https://bookdown.org/andreabellavia/mixtures/images/pca2.png)

</div>

<!-- _footer: Wikipedia -->

---

## Example of dimensionality reduction with independent component analysis

<div style="flex-basis: 47%;">

Independent component analysis (ICA) is a dimensionality reduction technique that maximizes the independence of the components.

</div>
<div style="flex-basis: 50%;">

![](https://scikit-learn.org/stable/_images/sphx_glr_plot_ica_vs_pca_001.png)

</div>

---

## Principal vs. independent component analysis

<div align=center>

For blind source separation, ICA is preferred over PCA.

![](https://scikit-learn.org/stable/_images/sphx_glr_plot_ica_blind_source_separation_001.png)

</div>

---

## Kernel PCA, of course

<div align=center>

Kernel PCA is a non-linear dimensionality reduction technique that uses the kernel trick to project the data onto a higher-dimensional space.

![width:1300px](https://scikit-learn.org/stable/_images/sphx_glr_plot_kernel_pca_002.png)

</div>

---

## Deep unsupervised learning

<div align=center>

Deep neural networks can be used for unsupervised learning:
__Autoencoders__: learn a low-dimensional representation of the data.
__Generative adversarial networks__: learn a generative model of the data.

![width:900px](https://fr.mathworks.com/discovery/autoencoder/_jcr_content/mainParsys/image.adapt.480.medium.svg/1665035671723.svg)

</div>

---

## Autoencoders

<div style="flex-basis: 40%;">

Autoencoders are neural networks that learn a low-dimensional representation of the data. They are composed of an __encoder__ and a __decoder__.

The input $\mathbf{x}$ is encoded into a latent representation $\mathbf{z}$, and decoded into $\mathbf{x}'$.

The loss function is the difference between the input and the reconstruction: 

$$
\mathcal{L} = \|\mathbf{x} - \mathbf{x}'\|^2
$$

</div>
<div style="flex-basis: 40%;">

![](images/examples/valentine_ae.png)

</div>

---

## Convolutional autoencoders

<div align=center>

Convolutional autoencoders use convolutional layers instead of fully connected layers.
They are used for image denoising, compression, and quality assessment.

![width:1000px](https://miro.medium.com/v2/resize:fit:1400/1*gzJAJDLDavH_W7Zv2M2J7w.png)

</div>

---

<!-- _class: titlepage-->

<div>

![width:160](images/logo/cheese.png)

# _The end!_

</div>