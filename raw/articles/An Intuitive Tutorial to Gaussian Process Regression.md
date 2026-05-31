---
title: "An Intuitive Tutorial to Gaussian Process Regression"
source: "https://ar5iv.labs.arxiv.org/html/2009.10862"
author:
published:
created: 2026-05-31
description: "This tutorial aims to provide an intuitive introduction to Gaussian process regression (GPR). GPR models have been widely used in machine learning applications due to their representation flexibility and inherent capab…"
tags:
  - "clippings"
---
XX XX 8 December An Intuitive Tutorial to Gaussian Process Regression

Feature Article: Introduction to Gaussian Processes

Jie Wang University of Waterloo, Waterloo, ON, N2L 3G1, Canada

(2023)

###### Abstract

This tutorial aims to provide an intuitive introduction to Gaussian process regression (GPR). GPR models have been widely used in machine learning applications due to their representation flexibility and inherent capability to quantify uncertainty over predictions. The tutorial starts with explaining the basic concepts that a Gaussian process is built on, including multivariate normal distribution, kernels, non-parametric models, and joint and conditional probability. It then provides a concise description of GPR and an implementation of a standard GPR algorithm. In addition, the tutorial reviews packages for implementing state-of-the-art Gaussian process algorithms. This tutorial is accessible to a broad audience, including those new to machine learning, ensuring a clear understanding of GPR fundamentals.

Gaussian Process is a key model in probabilistic supervised machine learning, widely applied in regression and classification tasks. It makes predictions incorporating prior knowledge (kernels) and provides uncertainty measures over its predictions [^1]. Despite its broad application, understanding GPR can be challenging, especially for professionals outside computer science, due to its reliance on complex concepts like multivariate normal distribution, kernels, and non-parametric models.

This tutorial aims to explain GPR in a clear, accessible way, starting from fundamental mathematical concepts including multivariate normal distribution, kernels, non-parametric models, and joint and conditional probability. To facilitate an intuitive understanding, the tutorial extensively utilizes plots and provides practical code examples, available at [https://github.com/jwangjie/Gaussian-Process-Regression-Tutorial](https://github.com/jwangjie/Gaussian-Process-Regression-Tutorial). This tutorial is designed to make GPR accessible to a diverse audience, ensuring that even those new to the field can grasp its core principles.

## MATHEMATICAL BASICS

This section explains the foundational concepts essential for understanding Gaussian process regression (GPR). We start with the Gaussian (normal) distribution, followed by an explanation of multivariate normal distribution (MVN) theories, kernels, non-parametric models, and the principles of joint and conditional probability. The objective of regression is to formulate a function that accurately represents observed data points and then utilize this function for predicting new data points. Considering a set of observed data points depicted in Fig. 1(a), an infinite array of potential functions can be fitted to these data points. Fig. 1(b) illustrates five such sample functions. In GPR, Gaussian processes perform regression by defining a distribution over this infinite number of functions [^2].

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2009.10862/assets/x1.png)

(a) Data point observations

### Gaussian Distribution

A random variable $X$ is Gaussian or normally distributed with mean $\mu$ and variance $\sigma^{2}$ if its probability density function (PDF) is [^3]:

$$
\displaystyle P_{X}(x)=\frac{1}{\sqrt{2\pi}\sigma}exp{\left(-\frac{{\left(x-\mu\right)}^{2}}{2\sigma^{2}}\right)}\ .
$$

Here, $X$ represents random variables and $x$ is the real argument. This normal distribution of $X$ is usually represented by $P_{X}(x)~{}\sim\mathcal{N}(\mu,\sigma^{2})$. The PDF of a uni-variate normal (or Gaussian) distribution was plotted in Fig. 2, where 1000 points from a uni-variate normal distribution were randomly generated and plotted along the $X$ axis.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2009.10862/assets/x3.png)

Figure 2: Visualization of 1000 normally distributed data points as red vertical bars on the X 𝑋 -axis, alongside their PDF plotted as a two-dimensional bell curve.

These randomly generated data points can be expressed as a vector $x_{1}=[x_{1}^{1},x_{1}^{2},\ldots,x_{1}^{n}]$. By plotting the vector $x_{1}$ on a new $Y$ axis at $Y=0$, we projected the points $[x_{1}^{1},x_{1}^{2},\ldots,x_{1}^{n}]$ into a different space shown in Fig. 3. We did nothing but vertically plot points of the vector $x_{1}$ in a new $Y,x$ coordinates space. Similarly, another independent Gaussian vector $x_{2}=[x_{2}^{1},x_{2}^{2},\ldots,x_{2}^{n}]$ can be plotted at $Y=1$ within the same coordinate framework, as demonstrated in Fig. 3. It’s crucial to remember that both $x_{1}$ and $x_{2}$ are a uni-variate normal distribution depicted in Fig. 2.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2009.10862/assets/x4.png)

Figure 3: Two independent uni-variate Gaussian vector points plotted vertically within the Y, x 𝑌 𝑥 Y,x coordinates space.

Next, we selected 10 points randomly in vector $x_{1}$ and $x_{2}$ respectively and connected these points in order with lines as shown in Fig. 4(a). These connected lines look like linear functions spanning within the $[0,1]$ domain. We can use these functions to make predictions for regression tasks if the new data points are on (or proximate to) these linear lines. However, the assumption that new data points will consistently lie on these linear functions often does not hold. If we plot more random generated uni-variate Gaussian vectors, say 20 vectors like $x_{1},x_{2},\ldots,x_{20}$ within $[0,1]$ interval, and connecting 10 randomly selected sample points of each vector as lines, we get 10 lines that look more like functions within $[0,1]$ shown in Fig. 4(b). Yet, we still cannot use these lines to make predictions for regression tasks because they are too noisy. These functions must be smoother, meaning input points that are close to each other should have similar output values. These “functions” generated by connecting points from independent Gaussian vectors lack the required smoothness for regression tasks. Therefore, it is necessary to correlate these independent Gaussians, forming a joint Gaussian distribution, as described by the theory of multivariate normal distribution.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2009.10862/assets/x5.png)

(a) Two Gaussian vectors

### Multivariate Normal Distribution

It is quite usual and often necessary for a system to be described by more than one feature variable $(x_{1},x_{2},\ldots,x_{D})$ that are correlated to each other. If we would like to model these variables all together as one Gaussian model, we need to use a multivariate Gaussian/normal (MVN) distribution model [^3]. The PDF of a D-dimensional MVN is defined as [^3]:

$$
\mathcal{N}(x|\mu,\Sigma)=\dfrac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\exp\left[-\dfrac{1}{2}(x-\mu)^{\mathsf{T}}\Sigma^{-1}(x-\mu)\right],
$$

Here, $D$ represents the number of the dimensionality, $x$ denotes the variable, $\mu=\mathbb{E}[x]\in\mathbb{R}^{D}$ is the mean vector, and $\Sigma=\text{cov}[x]$ is the $D\times D$ covariance matrix. The $\Sigma$ is a symmetric matrix that stores the pairwise covariance of all jointly modeled random variables, with $\Sigma_{ij}=\text{cov}(y_{i},y_{j})$ as its $(i,j)$ element.

A bi-variate normal (BVN) distribution offers a simpler example to understand the MVN concept. A BVN distribution can be visualized as a three-dimensional (3-D) bell curve, where the vertical axis (height) represents the probability density, as shown in Fig. 5(a). The ellipse contours on the $x_{1},x_{2}$ plane, illustrated in Fig. 5(a) and 5(b), are the projections of this 3-D curve. The shape of ellipses shows the correlation degree between $x_{1}$ and $x_{2}$ points, i.e. how one variable of $x_{1}$ relates to another variable of $x_{2}$. The function $P(x_{1},x_{2})$ denotes the joint probability density of $x_{1}$ and $x_{2}$.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2009.10862/assets/x7.png)

(a) 3-D bell curve

For a BVN, the mean vector $\mu$ is a two-dimensional vector $\begin{bmatrix}\mu_{1}\\
\mu_{2}\end{bmatrix}$, where $\mu_{1}$ and $\mu_{2}$ represent the independent means of $x_{1}$ and $x_{2}$, respectively. The covariance matrix is $\begin{bmatrix}\sigma_{11}&\sigma_{12}\\
\sigma_{21}&\sigma_{22}\end{bmatrix}$, with the diagonal terms $\sigma_{11}$ and $\sigma_{22}$ being the independent variance of $x_{1}$ and $x_{2}$, respectively. The off-diagonal terms, $\sigma_{12}$ and $\sigma_{21}$ represent correlations between $x_{1}$ and $x_{2}$. The BVN is expressed as:

$$
\displaystyle\begin{bmatrix}x_{1}\\
x_{2}\end{bmatrix}\sim\mathcal{N}\left(\begin{bmatrix}\mu_{1}\\
\mu_{2}\end{bmatrix},\begin{bmatrix}\sigma_{11}&\sigma_{12}\\
\sigma_{21}&\sigma_{22}\end{bmatrix}\right)=\mathcal{N}(\mu,\Sigma)\ .
$$

It is intuitively understandable that we need conditional probability rather than joint probability for regression tasks. If slicing the 3-D bell curve of a BVN at a certain constant point, as shown in Fig. 5(a), we can obtain the conditional probability distribution $P(x_{1}|\,x_{2})$, with $x=x_{2}=\text{constant}$, shown in Fig. 6. This conditional distribution is also Gaussian [^1].

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2009.10862/assets/x9.png)

Figure 6: The conditional probability distribution P ( x 1 | 2 ) 𝑃 conditional subscript 𝑥 P(x\_{1}|\\,x\_{2}) obtained by cutting a slice on the PDF 3-D bell curve of a BVN.

### Kernels

Having introduced the MVN distribution, we want to smooth the functions in Fig. 4(b) for regression tasks. The kernel, or covariance function, plays a pivotal role in this smoothing process, encapsulating our prior knowledge about the functions we aim to model. In regression, we desire the predictions to be smooth and logical: similar inputs should yield similar outputs. For example, consider two houses, A and B, with comparable size, location, and features; we expect their market prices to be similar. A natural measure of ‘similarity’ between two inputs is the dot product $A\cdot B=\lVert A\rVert\lVert B\rVert\text{cos}\theta$, where $\theta$ is the angle between two input vectors. Smaller angles, indicating high similarity, correspond to larger dot products, and vice versa.

Imagine a scenario in which we could lift our house into a ‘magical’ space, where doing this dot product becomes more powerful and tells us even more about how similar our houses are. This magical space is called “feature space”. The function that helps us do this lift and enhanced compression in the feature space is named as “kernel function”, denoted as $k(x,\ x^{\prime})$. We do not actually move our data into this new high-dimensional “feature space” (that could be computationally expensive); instead, the kernel function facilitates the comparison of data though providing us the same dot product result as if we had done so. This is known as the famous “kernel trick”. Formally, the kernel function $k(x,\ x^{\prime})$ computes the similarity between data points in a high-dimensional feature space without explicitly transforming the inputs [^1]. Instead of directly computing the dot product of transformed inputs, $\langle\phi(x),\phi(x^{\prime})\rangle$, with $\phi$ being the feature mapping function, the kernel function accomplishes the same result in a computationally efficient manner.

The squared exponential (SE) kernel, also known as the Gaussian or Radial Basis Function (RBF) kernel, is widely used in Gaussian processes due to its exceptional properties [^4]. It is recognized for its adaptability across various functions. Additionally, every function in its prior is smooth and infinitely differentiable, leading to naturally smooth and differentiable model predictions. The SE kernel function is defined as <sup>1</sup>:

$$
\displaystyle\text{cov}(x_{i},x_{j})=\exp\left(-~{}\frac{(x_{i}-x_{j})^{2}}{2}\right)\ .
$$

In Fig. 4(b), we plotted 20 independent Gaussian vectors by connecting 10 randomly selected sample points from each vector in order by lines. Instead of plotting 20 independent Gaussian, we can generate 10 twenty-variate normal (20-VN) distributions with an identity covariance function as shown in 7(a). It is the same as Fig. 4(b) due to the absence of correlations among points by using identity as its kernel function. Employing an RBF kernel as the covariance function, on the other hand, we got smooth lines observed in 7(b).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2009.10862/assets/x10.png)

(a) Ten samples of the 20-VN prior with an identity kernel

By integrating covariance functions, we obtain smoother lines, and they start to look like functions. It is natural to consider continuing to increase the dimension of MVN. Here, dimension refers to the number of variables in the MVN. When the dimension of MVN becomes larger, the region of interest will be filled up with more points. When the dimension reaches infinity, there will be a point to represent every possible input point. Utilizing MVN with infinite dimensions allows us to fit functions with infinite parameters for regression tasks, thus enabling predictions throughout the region of interest. In Fig. 8, We illustrate 200 samples from a two hundred-variate normal (200-VN) distribution to conceptualize functions with infinite parameters. We call these functions “kernelized prior functions”, because there are no observed data points yet. All functions are randomly generated by the MVN model with kernel functions as prior knowledge before having any observed data points.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2009.10862/assets/figs/200d_gaussian_kernel_prior.png)

Figure 8: Two hundred kernelized prior functions from a two hundred-variate normal distribution.

### Non-parametric Model

This section explains the distinction between parametric and non-parametric models [^3]. Parametric models assume that the data distribution can be modeled in terms of a set of finite numbers of parameters. In regression, given some data points, we would like to predict the function value $y=f(x)$ for a new specific $x$. If we assume a linear regression model, $y=\theta_{1}+\theta_{2}x$, we need to identify the parameters $\theta_{1}$ and $\theta_{2}$ to define the function. often, a linear model is insufficient, and a polynomial model with more parameters, like $y=\theta_{1}+\theta_{2}x+\theta_{3}x^{2}$ is needed. We use the training dataset $D$ comprising $n$ observed points, $D=[(x_{i},y_{i})\,|\,i=1,\ldots,n]$ to train the model, i.e. establish a mapping $x$ to $y$ through basis functions $f(x)$. After the training process, all information in the dataset is assumed to be encapsulated by the feature parameters $\mathbf{\theta}$, thus predictions are independent of the training dataset $D$. This can be expressed as $P(f_{*}\,|\,X_{*},\mathbf{\theta},D)=P(f_{*}\,\,|\,X_{*},\mathbf{\theta})$, in which $f_{*}$ are predictions made at unobserved data points $X_{*}$. Thus, when conducting regressions using parametric models, the complexity or flexibility of models is inherently limited by the number of parameters. Conversely, if the parameter number of a model grows with the size of the observed dataset, it’s a non-parametric model. Non-parametric models do not imply that there are no parameters; but rather they entail an infinite number of parameters.

## GAUSSIAN PROCESSES

Before delving into the Gaussian processes, we first do a quick review of the foundational concepts we have covered. In regression, our objective is to model a function $\mathbf{f}$ based on observed data points $D$ (the training dataset) from the unknown function $\mathbf{f}$. Traditional nonlinear regression methods often give a single function that is considered to best fit the dataset. However, there could be more than one function that fits the observed data points equally well. We observed that when the dimension of MVN was infinite, we could make predictions at any point using these infinite numbers of functions. These functions are MVN because it is our (prior) assumption. More formally, the prior distribution of these infinite functions is MVN, representing the expected outputs of $\mathbf{f}$ over inputs $\mathbf{x}$ before observing any data. When we start to have observations, instead of infinite numbers of functions, we only keep functions that fit the observed data points, forming the posterior distribution. This posterior is the prior updated with observed data. When we have new observations, we use the current posterior as prior, and new observed data points to obtain a fresh posterior.

Definition of Gaussian processes: A Gaussian process model describes a probability distribution over possible functions that fit a set of points. Because we have the probability distribution over all possible functions, we can compute the means to represent the maximum likelihood estimate of the function, and the variances as an indicator of prediction confidence. Key points include: i) the function prior is updated with new observations; ii) a Gaussian process model is a probability distribution over possible functions, with any finite samples of functions being jointly Gaussian distributed; iii) the mean function derived from the posterior distribution of possible functions is the function used for regression predictions.

Now, it is time to explore the standard Gaussian process model. All the parameter definitions align the classic textbook by Rasmussen (2006) [^1]. Besides the covered basic concepts, Appendix A.1 and A.2 of [^1] are also recommended reading. The regression function modeled by a multivariate Gaussian is given by:

$$
\displaystyle P(\mathbf{f}\,\lvert\,\mathbf{X})=\mathcal{N}(\mathbf{f}\,\lvert\,\boldsymbol{\mu},\mathbf{K})\ ,
$$

where $\mathbf{X}=[\mathbf{x}_{1},\ldots,\mathbf{x}_{n}]$ represents the observed data points, $\mathbf{f}=\left[f(\mathbf{x}_{1}),\ldots,f(\mathbf{x}_{n})\right]$ the function values, $\boldsymbol{\mu}=\left[m(\mathbf{x}_{1}),\ldots,m(\mathbf{x}_{n})\right]$ the mean function, and $K_{ij}=k(\mathbf{x}_{i},\mathbf{x}_{j})$ the kernel function, which is a positive definite. With no observation, we default the mean function to $m(\mathbf{X})=0$, assuming the data is normalized to zero mean. The Gaussian process model is thus a distribution over functions whose shapes (smoothness) are defined by $\mathbf{K}$. If points $\mathbf{x}_{i}$ and $\mathbf{x}_{j}$ are considered similar by the kernel, their respective function outputs, $f(\mathbf{x}_{i})$ and $f(\mathbf{x}_{j})$, are expected to be similar too. The regression process using Gaussian processes is illustrated in Fig. 9: given observed data (red points) and a mean function $\mathbf{f}$ (blue line) estimated from these observed data points, we predict at new points $\mathbf{X}_{*}$ as $\mathbf{f}(\mathbf{X}_{*})$.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2009.10862/assets/figs/predictions.png)

Figure 9: A illustrative process of conducting regressions by Gaussian processes. The red points are observed data, the blue line represents the mean function estimated by the observed data points, and predictions will be made at new blue points.

The joint distribution of $\mathbf{f}$ and $\mathbf{f}_{*}$ is expressed as:

$$
\displaystyle\begin{bmatrix}\mathbf{f}\\
\mathbf{f}_{*}\end{bmatrix}\sim\mathcal{N}\left(\begin{bmatrix}m(\mathbf{X})\\
m(\mathbf{X}_{*})\end{bmatrix},\begin{bmatrix}\mathbf{K}&\mathbf{K}_{*}\\
\mathbf{K}_{*}^{\mathsf{T}}&\mathbf{K}_{**}\end{bmatrix}\right)\ ,
$$

where $\mathbf{K}=K(\mathbf{X},\mathbf{X})$, $\mathbf{K}_{*}=K(\mathbf{X},\mathbf{X}_{*})$ and $\mathbf{K}_{**}=K(\mathbf{X}_{*},\mathbf{X}_{*})$. The mean is assumed to be $\begin{pmatrix}m(\mathbf{X}),m(\mathbf{X}_{*})\end{pmatrix}=\mathbf{0}$.

While this equation describes the joint probability distribution $P(\mathbf{f},\mathbf{f}_{*}\,|\,\mathbf{X},\mathbf{X}_{*})$ over $\mathbf{f}$ and $\mathbf{f}_{*}$, in regressions, we need the conditional distribution $P(\mathbf{f}_{*}\,|\,\mathbf{f},\mathbf{X},\mathbf{X}_{*})$ over $\mathbf{f}_{*}$ only. The derivation of the conditional distribution $P(\mathbf{f}_{*}\,|\,\mathbf{f},\mathbf{X},\mathbf{X}_{*})$ from the joint distribution $P(\mathbf{f},\mathbf{f}_{*}\,|\,\mathbf{X},\mathbf{X}_{*})$ is achieved by using the Marginal and conditional distributions of MVN theorem [^5]. The result is:

$$
\displaystyle\mathbf{f}_{*}\,|\,\mathbf{f},\mathbf{X},\mathbf{X}_{*}\sim\mathcal{N}\left(\mathbf{K}_{*}^{\mathsf{T}}\,\mathbf{K}^{-1}\,\mathbf{f},\>\mathbf{K}_{**}-\mathbf{K}_{*}^{\mathsf{T}}\,\mathbf{K}^{-1}\,\mathbf{K}_{*}\right)\ .
$$

In realistic scenarios, we typically have access only to noisy versions of true function values, $y=f(x)+\epsilon$, where $\epsilon$ represents additive independent and identically distributed (i.i.d.) Gaussian noise with variance $\sigma_{n}^{2}$. The prior on these noisy observations then becomes $\text{cov}(y)=\mathbf{K}+\sigma_{n}^{2}\mathbf{I}$. The joint distribution of the observed values and the function values at new testing points is:

$$
\displaystyle\begin{pmatrix}\mathbf{y}\\
\mathbf{f}_{*}\end{pmatrix}\sim\mathcal{N}\left(\mathbf{0},\begin{bmatrix}\mathbf{K}+\sigma_{n}^{2}\mathbf{I}&\mathbf{K}_{*}\\
\mathbf{K}_{*}^{\mathsf{T}}&\mathbf{K}_{**}\end{bmatrix}\right)\ .
$$

By deriving the conditional distribution, we get the predictive equations for Gaussian process regression:

$$
\displaystyle\mathbf{\bar{f}_{*}}\,|\,\mathbf{X},\mathbf{y},\mathbf{X}_{*}\sim\mathcal{N}\left(\mathbf{\bar{f}_{*}},\text{cov}(\mathbf{f}_{*})\right)\ ,
$$

where

$$
\displaystyle\mathbf{\bar{f}_{*}}
$$
 
$$
\displaystyle\overset{\Delta}{=}\mathbb{E}[\mathbf{\bar{f}_{*}}\,|\,\mathbf{X},\mathbf{y},\mathbf{X}_{*}]
$$
 
$$
\displaystyle=\mathbf{K}_{*}^{\mathsf{T}}[\mathbf{K}+\sigma_{n}^{2}\mathbf{I}]^{-1}\mathbf{y}\ ,
$$
$$
\displaystyle\text{cov}(\mathbf{f}_{*})
$$
 
$$
\displaystyle=\mathbf{K}_{**}-\mathbf{K}_{*}^{\mathsf{T}}[\mathbf{K}+\sigma_{n}^{2}\mathbf{I}]^{-1}\mathbf{K}_{*}\ .
$$

In this expression, the variance function $\text{cov}(\mathbf{f}_{*})$ reveals that the uncertainty in predictions depends solely on the input values $\mathbf{X}$ and $\mathbf{X}_{*}$, not on the observed outputs $\mathbf{y}$. This characteristic is a distinctive property of Gaussian distributions [^1].

## ILLUSTRATIVE EXAMPLE

This section demonstrates an implementation of the standard GPR, adhering to the algorithm outlined in Rasmussen (2006) [^1].

$$
\boxed{\begin{split}L&=\text{cholesky}(\mathbf{K}+\sigma^{2}_{n}\mathbf{I})\\
\boldsymbol{\alpha}&=L^{\top}\setminus(L\setminus\mathbf{y})\\
\mathbf{\bar{f}_{*}}&=\mathbf{K}_{*}^{\top}\boldsymbol{\alpha}\\
\mathbf{v}&=L\setminus\mathbf{K}_{*}\\
\mathbb{V}[\mathbf{\bar{f}_{*}}]&=K(\mathbf{X}_{*},\mathbf{X}_{*})-\mathbf{v}^{\top}\mathbf{v}.\\
\log p(\mathbf{y}\mid\mathbf{X})&=-\frac{1}{2}\mathbf{y}^{\top}(\mathbf{K}+\sigma_{n}^{2}\mathbf{I})^{-1}\mathbf{y}-\frac{1}{2}\log\det(\mathbf{K}+\sigma_{n}^{2}\mathbf{I})\\
&\qquad\qquad\qquad\qquad\quad\,\,\,\,\!-\frac{n}{2}\log 2\pi\end{split}}
$$

The inputs of this algorithm are $\mathbf{X}$ (inputs), $\mathbf{y}$ (targets), $K$ (covariance function), $\sigma^{2}_{n}$ (noise level), and $\mathbf{X_{*}}$ (test input). The outputs include $\mathbf{\bar{f}_{*}}$ (mean), $\mathbb{V}[\mathbf{\bar{f}_{*}}]$ (variance), and $\log p(\mathbf{y}\mid\mathbf{X})$ (log marginal likelihood).

An example result is illustrated in Fig. 10. We conducted regression within the \[-5, 5\] interval. Observed data points (training dataset) were generated from a uniform distribution between -5 and 5. The functions were evaluated at evenly spaced points between -5 and 5. The regression function is composed of mean values estimated by a GPR model. Twenty samples of posterior mean functions, along with 3 times variances, were also plotted.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2009.10862/assets/x12.png)

Figure 10: An illustrative example of standard GPR. Black crosses represent observed data points generated by the blue dotted line (true function). Given these data points, infinite possible posterior functions were obtained, with 20 samples plotted in different colors. The mean function, derived from the probability distribution of these functions, is plotted as a red solid line. The blue shaded area around the mean function indicates 3 times prediction variances.

### Hyperparameters Optimization

We have covered the basics of GPR and provided a straightforward example. However, practical GPR models often present more complexity. The selection of the kernel function is critical, as it greatly affects the model’s ability to generalize [^6]. Kernel functions range from well-established options like the RBF to custom designs tailored to specific needs based on model requirements such as smoothness, sparsity, drastic changes, and differentiability [^4]. Selecting an appropriate kernel function for a specific GPR task is detailed in Duvenaud (2014) [^4]. Additionally, hyperparameter optimization plays an essential role in kernel-based methods. For example, consider the widely used RBF kernel:

$$
\displaystyle k(\mathbf{x}_{i},\mathbf{x}_{j})=\sigma_{f}^{2}\exp\Big{(}-\frac{1}{2l}(\mathbf{x}_{i}-\mathbf{x}_{j})^{\mathsf{T}}(\mathbf{x}_{i}-\mathbf{x}_{j})\Big{)}\ ,
$$

In this kernel, $\sigma_{f}$ (vertical scale) and $l$ (horizontal scale) are hyperparameters. The parameter $\sigma_{f}$ determines the vertical span of the function, while $l$ indicates the rate at which the correlation between two points decreases with increasing distance. The influence of the hyperparameter $l$ on the smoothness of the function is demonstrated in Fig. 11. Increasing the value of $l$ results in a smoother function, while a smaller $l$ value leads to a function with more fluctuations or ‘wiggles’.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2009.10862/assets/x13.png)

(a) l 𝑙 = small

The optimal hyperparameters $\boldsymbol{\Theta^{*}}$ are determined by maximizing the log marginal likelihood [^1]:

$$
\displaystyle\boldsymbol{\Theta^{*}}=\arg\max\limits_{\Theta}\log p(\mathbf{y}\,|\,\mathbf{X},\boldsymbol{\Theta})\ .
$$

Thus, considering hyperparameters, a more generalized prediction equation at new testing points is [^7]:

$$
\displaystyle\mathbf{\bar{f}_{*}}\,|\,\mathbf{X},\mathbf{y},\mathbf{X}_{*},\boldsymbol{\Theta}\sim\mathcal{N}\left(\mathbf{\bar{f}_{*}},\text{cov}(\mathbf{f}_{*})\right)\ .
$$

Note that after learning/optimizing the hyperparameters, the predictive variance $\text{cov}(\mathbf{f}_{*})$ depends on not only the inputs $\mathbf{X}$ and $\mathbf{X}_{*}$ but also the outputs $\mathbf{y}$ [^8]. With the optimized hyperparameters, $\sigma_{f}=0.0067$ and $l=0.0967$, the regression result of the observed data points shown in Fig. 11 is depicted in Fig. 12. Here, the hyperparameters optimization was conducted by the GPy package, which will be introduced in the next section.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2009.10862/assets/x16.png)

Figure 12: Regression result with the optimized hyperparameters σ f subscript 𝜎 𝑓 \\sigma\_{f} and l 𝑙.

### Gaussian Processes Packages

This section reviews three Python packages for implementing Gaussian processes. GPy is a mature and well-documented package in development since 2012 [^9]. It utilizes NumPy for computations, offering sufficient stability for tasks that are not computationally intensive. However, GPR is computationally expensive in high dimensional spaces (beyond a few dozen). For complex and computationally intense tasks, packages incorporating advanced algorithms and GPU acceleration are especially preferable. GPflow [^9] originates from GPy with a similar interface. It leverages TensorFlow as its computational backend. GPyTorch [^10] is a more recent package that provides GPU acceleration through PyTorch. Like GPflow, GPyTorch supports automatic gradients, which simplifies the development of complex models, such as those embedding deep neural networks within GP frameworks.

## CONCLUSION

A Gaussian process is a probability distribution over possible functions that fit a set of points [^1]. A Gaussian process regression model provides prediction values together with uncertainty estimates. The model incorporates prior knowledge about the nature of the functions through the use of kernel functions.

The GPR model discussed in this tutorial is the standard or “vanilla” approach to Gaussian processes [^11]. There are two primary limitations with it: 1) The computational complexity is $O(N^{3})$, where $N$ represents the dimension of the covariance matrix $K$. 2) The memory consumption increases quadratically with data size. Due to these constraints, standard GPR models become impractical for large datasets. In such cases, sparse Gaussian Processes are employed to alleviate computational complexity [^12].

## ACKNOWLEDGMENTS

The author would like to express sincere gratitude to Prof. Krzysztof Czarnecki from the University of Waterloo. His insightful and constructive feedback have significantly contributed to the progression and enhancement of the quality of this tutorial. The author is deeply thankful for his invaluable guidance and support.

## REFERENCES

Jie Wang is a postdoctoral research associate at the University of Waterloo. He earned his Ph.D. degree in Mechanical Engineering from the University of Calgary. His research bridges machine learning and traditional robotics, primarily focusing on enhancing the performance of mobile robots in real-world settings while ensuring safety and efficiency. For further information or collaboration inquiries, he can be reached at jwangjie@outlook.com.

[^1]: C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*. The MIT Press, 2006.

[^2]: Z. Ghahramani, “A Tutorial on Gaussian Processes (or why I don’t use SVMs),” in *Machine Learning Summer School (MLSS)*, 2011.

[^3]: K. P. Murphy, *Machine Learning: A Probabilistic Perspective*. The MIT Press, 2012.

[^4]: D. Duvenaud, “Automatic model construction with Gaussian processes,” Ph.D. dissertation, University of Cambridge, 2014.

[^5]: C. M. Bishop and N. M. Nasrabadi, *Pattern recognition and machine learning*. Springer, 2006.

[^6]: D. Duvenaud, “The Kernel Cookbook,” Available at [https://www.cs.toronto.edu/~duvenaud/cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook), 2016.

[^7]: Z. Dai, “Computationally efficient GPs,” Available at [https://www.youtube.com/watch?v=7mCfkIuNHYw](https://www.youtube.com/watch?v=7mCfkIuNHYw), 2019.

[^8]: Z. Chen and B. Wang, “How priors of initial hyperparameters affect Gaussian process regression models,” *Neurocomputing*, vol. 275, pp. 1702–1710, 2018.

[^9]: A. G. De G. Matthews, M. Van Der Wilk, T. Nickson, K. Fujii, A. Boukouvalas, P. León-Villagrá, Z. Ghahramani, and J. Hensman, “GPflow: A Gaussian process library using TensorFlow,” *The Journal of Machine Learning Research*, vol. 18, no. 1, pp. 1299–1304, 2017.

[^10]: J. R. Gardner, G. Pleiss, D. Bindel, K. Q. Weinberger, and A. G. Wilson, “GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration,” in *Advances in Neural Information Processing Systems*, 2018.

[^11]: R. Frigola, F. Lindsten, T. B. Schön, and C. E. Rasmussen, “Bayesian Inference and Learning in Gaussian Process State-Space Models with Particle MCMC,” in *Advances in Neural Information Processing Systems*, 2013, pp. 3156–3164.

[^12]: H. Liu, Y.-S. Ong, X. Shen, and J. Cai, “When Gaussian process meets big data: A review of scalable GPs,” *IEEE Transactions on Neural Networks and Learning Systems*, 2020.