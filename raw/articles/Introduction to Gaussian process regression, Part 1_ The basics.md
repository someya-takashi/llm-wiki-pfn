---
title: "Introduction to Gaussian process regression, Part 1: The basics"
source: "https://medium.com/data-science-at-microsoft/introduction-to-gaussian-process-regression-part-1-the-basics-3cb79d9f155f"
author:
  - "[[Kaixin Wang]]"
published: 2022-10-04
created: 2026-05-31
description: "Gaussian process (GP) is a supervised learning method used to solve regression and probabilistic classification problems.¹ It has the term “"
tags:
  - "clippings"
---
![](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*oVW9VFXgIVGgiRZV)

Photo by Garrett Sears on Unsplash.

Gaussian process (GP) is a supervised learning method used to solve regression and probabilistic classification problems.[¹](#0bfc) It has the term “Gaussian” in its name as each Gaussian process can be seen as an infinite-dimensional generalization of multivariate Gaussian distributions. In this article, I focus on Gaussian processes for regression purposes, known as Gaussian process regression (GPR). GPR has been applied to solve several different types of real-world problems, including ones in materials science, chemistry, physics, and biology.

## Methodology

GPR is a non-parametric Bayesian approach for inference. Instead of inferring a distribution over the parameters of a parametric function, ==Gaussian processes can be used to infer a distribution over the function of interest directly==. A Gaussian process defines a prior function, which is converted to a posterior function after having observed some values from the prior distribution.[²](#9474)

A Gaussian process is a random process where any point *x* in the real domain is assigned a random variable *f(x)* and where the joint distribution of a finite number of these variables *p(f(x₁), …, f(xₙ))* follows a Gaussian distribution:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ww-JNN5feQBqR6myVIR0oQ.png)

In Equation (1)*, m* is the *mean* function, where people typically use *m(x) = 0* as GPs are flexible enough to model the mean even if it’s set to an arbitrary value at the beginning. **k** is a positive definite function referred to as the *kernel* function or *covariance* function. Therefore, a Gaussian process is a distribution that is defined by **K**, the covariance matrix. If two points are similar in the kernel space, the function values at these points will also be of similar value.

Suppose we are given the values of the noise-free function *f* at some inputs *x*. In this case, a GP prior can be converted into a GP posterior, which can be used to make predictions at new inputs. By the definition of a GP, the joint distribution of observed values and predictions is Gaussian, and can be partitioned into the following:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*m81CGpWRhi5ngNAZmjY8zQ.png)

If there are *m* training data points and *n* new observations (i.e., test data points), **K** is an *m × m* matrix, **K\*** is an *m× n* matrix, and **K\*\*** is an *n × n* matrix.

Based on properties of Gaussian distributions, the predictive distribution (i.e., posterior) is given by:

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*qyeWDxnMcsfT7q41PEc15A.png)

Now suppose we have the objective function with noise, *y = f +* ε, where the noise

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*8jNZ--WP5gsEmc28J9I5NA.png)

is independently and identically distributed (i.i.d.). The posterior can then be represented as

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*LBQE38e9zxSKWfvQZRrJLA.png)

Finally, to include noise σ into predictions, we need to add it to the diagonal of the covariance matrix:

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*t8Lvk-nGOuNJZ8dDxTbW1A.png)

## Python implementation and example

We saw in the previous section that the kernel function plays a key role in making the predictions in GPR. Instead of building the kernels and GPR architecture from scratch, we can leverage the existing Python packages that have implemented GPR. The frequently used ones are scikit-learn/sklearn,[¹](#0bfc) GPyTorch,[³](#5f32) and GPflow.[⁴](#b578)

The sklearn version is implemented mainly upon NumPy, which is simple and easy to use, but has limited hyperparameter tuning options. GPyTorch is built via PyTorch — it’s highly flexible but requires some prior knowledge of PyTorch to build the architecture. GPflow is built upon TensorFlow, which is flexible in terms of hyperparameter optimization, and it is straightforward to construct the model. Weighing the pros and cons of different implementations, the GPR models demonstrated in the following sections are implemented using GPflow.

## Toy dataset creation

To illustrate how kernels work in GPR, we will look at a simple toy dataset curated intentionally. Figure 1 shows the true distribution and the observations collected. The goal is to build a model to find the real signal based on the data observed, but the challenge is that real-world observations always come with noise that perturbs the underlying patterns. Thus, selecting the proper kernels and tuning the hyperparameters of the kernels is critical for ensuring the model is neither *overfitted* nor *underfitted.*

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*kkr_EP3PU_RNpAt7iSHe2A.png)

***Figure 1*:** Example dataset. The blue line represents the true signal (i.e., *f*), the orange dots represent the observations (i.e., *y = f + σ*).

## Kernel selection

There are an infinite number of kernels that we can choose when fitting the GPR model. For simplicity, we will look only at the two most used functions—the linear kernel and the Radial basis function (RBF) kernel.

## Get Kaixin Wang’s stories in your inbox

Join Medium for free to get updates from this writer.

Linear kernel is one of the simplest kernel functions, parameterized by a variance (σ²) parameter. The construction of the kernel is as follows:

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*_95Gz09lZ3MT8448qp-ymw.png)

Note that the kernel is called a homogeneous linear kernel when σ² = 0.

The RBF kernel is a stationary kernel. It is also known as the “squared exponential” kernel. It is parameterized by a lengthscale (*l*) parameter, which can either be a scalar or a vector with the same number of dimensions as the inputs, and a variance (σ²) parameter, which controls the spread of the distribution. The kernel is given by:

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*jsOLx7lAJCCP9xP_TPUTpA.png)

The RBF kernel is infinitely differentiable, which implies that GPs with this kernel have mean square derivatives of all orders and are thus smooth in shape.

Using the data observed, let’s look at what happens when we fit a GPR model using different kernel functions. Figure 2 illustrates the effect of the kernel. It’s clear that the linear kernel predicts a purely linear relation between the input and the target, which gives an underfitted model. The RBF kernel interpolates the data points quite well, but isn’t very good at extrapolation (i.e., predicting unseen data points). As we can see, by trying to pass through as many data points as possible, the RBF kernel is fitted to the noise, creating an overfitted model. The combination of linear and RBF kernel, in comparison, has the best balance between interpolation and extrapolation—it interpolates the data points well enough, while at the same time extrapolating unseen data points at the two ends reasonably.

![Figure 2: Kernel selection. The green, red, and purple lines demonstrate the predictions made by GPR when using the Linear, RBF, and summation of the linear and RBF kernels, respectively. The shaded region around each curve represents the 95 percent confidence interval associated with the point estimates.](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*MKs9HBgmYeSbd8AA3n--Wg.png)

Figure 2: Kernel selection. The green, red, and purple lines demonstrate the predictions made by GPR when using the Linear, RBF, and summation of the linear and RBF kernels, respectively. The shaded region around each curve represents the 95 percent confidence interval associated with the point estimates.

***Figure 2:*** Kernel selection. The green, red, and purple lines demonstrate the predictions made by GPR when using the Linear, RBF, and summation of the linear and RBF kernels, respectively. The shaded region around each curve represents the 95-percent confidence interval associated with the point estimates.

Notice that each of the three prediction curves has its associated empirical confidence intervals. Because GPR is a probabilistic model, we can not only get the point estimate, but also compute the level of confidence in each prediction. The confidence intervals (i.e., the shaded region around each curve) shown in Figure 2 are the 95-percent confidence intervals of each kernel, where we see the intervals of non-linear kernels (RBF kernel, and the combination of linear and RBF kernel) touch the true signal, meaning their predictions are close enough to the ground truth with good confidence.

## Hyperparameter optimization

In addition to determining the kernel to use, hyperparameter tuning is another important step to ensure a well-fitted model. As shown in Equations (6) and (7), the RBF kernel depends on two parameters—the lengthscale (*l)*, which controls the smoothness of the distribution, and variance (σ²), which determines the spread of the curve.

Figure 3 shows the optimization of the two key parameters. From the plot on the left, we see that as the lengthscale decreases, the fitted curve becomes less smooth and more overfitted to the noise, while increasing the lengthscale results in a smoother shape. From the plot, the lengthscale chosen is 2.5, a value at which the model has a good balance between overfitting and underfitting. The figure on the right shows the effect of the variance parameter by fixing the lengthscale to 2.5. A smaller variance results in a smoother curve, whereas a larger variance results in a more overfitted model. From the plot, we observe that changing the value of variance has a relatively smaller impact on the shape of the curve compared to results from tuning the lengthscales. As before, the variance parameter is chosen in a way such that the balance between overfitting and underfitting is retained. Hence, using a lengthscale of 2.5, the optimized variance is chosen as 0.1.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*EYo_kURz4bOcPPqO81ayuQ.png)

***Figure 3:*** Hyperparameter optimization of lengthscales (left) and variance (right) hyperparameter.

## Discussion and conclusion

In this article, I have reviewed the rationale behind the GPR model and provided a simple example that illustrates the effect of choosing different kernel functions and the associated hyperparameters. Figure 4 below shows the optimal model found after the kernel selection and hyperparameter optimization step. It’s clear that the combination of the linear and RBF kernel captures the true signal quite accurately, and its 95-percent confidence interval aligns with the degree of noise in the data distribution quite well.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*J-Vvc5nWEPDOhEUctBkyzQ.png)

***Figure 4:*** Predictions from the optimized GPR model, with the associated 95-percent confidence interval.

To summarize, there are three advantages that GPR has over many other ML models: (1) interpolating—the prediction from GPR interpolates the observations for most types of kernel functions; (2) probabilistic—because the prediction is probabilistic, we can compute its empirical confidence intervals; and (3) versatile—different types and combinations of kernel functions can be used to fit the model.

In my [next article](https://medium.com/data-science-at-microsoft/introduction-to-gaussian-process-regression-part-2-application-to-predicting-concrete-strength-3facef04f639?sk=784ea7f747251e71e33998cbcf6c3de9), I look at how GPR can be applied to solve real-world problems and derive insights.

[*Kaixin Wang*](https://www.linkedin.com/in/kaixinwang/) *is on LinkedIn*.

## References

1\. Pedregosa, F. *et al.* Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research* **12**, 2825–2830 (2011).

2\. Krasser, M. [Gaussian processes](http://krasserm.github.io/2018/03/19/gaussian-processes/). (2018).

3\. Gardner, J. R., Pleiss, G., Bindel, D., Weinberger, K. Q. & Wilson, A. G. [GPyTorch: Blackbox matrix-matrix gaussian process inference with GPU acceleration](http://arxiv.org/abs/1809.11165). *CoRR* **abs/1809.11165**, (2018).

4\. Matthews, A. G. de G. *et al.* [GPflow: A Gaussian process library using TensorFlow](http://jmlr.org/papers/v18/16-537.html). *Journal of Machine Learning Research* **18**, 1–6 (2017).