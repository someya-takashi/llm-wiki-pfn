---
title: "The EM Algorithm Explained"
source: "https://medium.com/@chloebee/the-em-algorithm-explained-52182dbb19d9"
author:
  - "[[Chloe Bi]]"
published: 2019-02-08
created: 2026-06-17
description: "The Expectation-Maximization algorithm (or EM, for short) is probably one of the most influential and widely used machine learning algorithm"
tags:
  - "clippings"
---
The Expectation-Maximization algorithm (or EM, for short) is probably one of the most influential and widely used machine learning algorithms in the field. When I first came to learn about the EM algorithm, it is surprisingly difficult to find a tutorial that offers an intuitive explanation about what it attempts to achieve as well as a thorough analysis of why it works at theoretical level. It is, of course, a daunting task: the algorithm seems so simple at surface with only two steps involved, but it is also extremely complicated due to the heavy mathematical underpinning. As a sidebar, this actually reminds me of a question I got asked today: we always take for granted that the marginal of a Gaussian distribution is also Gaussian, but how much algebra does it take to prove it? Well, at least two pages is what I was told.

To explain what EM is, let us first consider this example. Say that we are in a college, and interested to learn the height distribution of male and female students in this college. The most sensible thing to do, as you probably would agree with me, is to randomly take a sample of N students of both genders, collect their height information and estimate the mean and standard deviation for boys and girls separately by way of the maximum likelihood method.

Now say that we are not able to know the gender of the student while we collect their height information, and so there are two things we have to guess/estimate: (1) whether the individual sample of height information belongs to a boy or a girl and (2) the parameters (μ, θ) for each gender which is now unobservable. This is tricky because only with the knowledge of who belongs to which group, can we make reasonable estimates of the group parameters separately. Similarly, only if we know the parameters that defines the groups, can we assign a subject properly. This literally has just become the chicken and egg problem. How do we break out of this infinite loop? Well, the EM algorithm just says to start with initial random guesses.

Let’s go back with our height example, say now we randomly assign the first half of the samples to one group, and second to another, we can get an estimate for the parameters of the two groups. With the identified group, we can review the membership assignment and reassign as appropriate. Iteratively doing this is guaranteed to be eventually at a stage where no further update is needed to be made either to the member assignment or group parameters. If that reminds you of the famous K-means algorithm, you are exactly right. As a matter of fact, ==K-means is special variant of the EM algorithm with the assumption that the clusters are spherical.==

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*OyXgise21a23D5JCss8Tlg.gif)

A nice visualization of EM algorithm from Wikipedia

If you need to convince yourself why the iterative process can guarantee to reach the local optima, then let me try explaining again via a more mathematical way.

Let us start again by formally presenting the problem using mathematical notations. In a probabilistic model, there are visible variables (y), latent variables (z) and associating parameter (θ). The likelihood p(y|θ), which we aim to maximize, measures the probability of the observables (height) given the parameters (group characteristics, such as mean and variance). Because θ depends on z, but z is hidden, we cannot directly apply maximum likelihood estimation to solve the argmax problem. Let us also define q(z) as any arbitrary distribution of the latent variable z. As such, we have:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*x-LTa3GrugvdR1TJ1Y2tHw.png)

It should be clear that Eq 1 is derived from Bayes’ rule. Rearranging terms from the left hand side and right hand side of Eq 1, we can arrive at Eq 2. Then, artificially multiply and divide by q(z), we get Eq3.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Sz0z2YWp73wKFemjrKYw_Q.png)

Taking logarithms on both sides of Eq 3 yields Eq 4. Computing the expectation of both sides with respect to q(z), we get:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*SgCrXGc9cb-DWFfZyFRiiQ.png)

Now, we need the help of Jensen’s inequality. Recall that for any strictly convex function, we have E(f(x)) ≥ f(E(x)). The inequality reverses for strictly concave functions. Going back to Eq 5, since the log function is indeed a concave function (i.e. 2nd order derivative is -1/x², which < 0), now we can derive that log likelihood of the y given θ has a lower bound as outlined in Eq 5. Moreover, the second term of Equation 5 is essentially the KL divergence between q(z)and p(z|y, θ), and it is always non-negative. Therefore, the lower bound can be reduced to just the first term, which we denote as F(q(z), θ).

You may ask, to maximize likelihood, why not just take the first-order derivative, set it to 0 and solve for the parameter in any of the equation 1–4? If you think about it, the derivative part is actually not an easy task. However, F(q(z), θ) in Eq 5 is more manageable to work with, since it is sum of logs.

## Get Chloe Bi’s stories in your inbox

Join Medium for free to get updates from this writer.

Now, the question may arise: the maximum of the lower bound is not the maximum of the actual log likelihood which we are after, right? So, what do we do? Below is a really nice visualization of EM algorithm’s convergence from the computational statistics course by Duke University.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*HDz1wzLb8w4j8VtLiwmEFw.png)

Computational Statistics in Python, Duke University

The E step starts with a fixed θ(t), and attempts to maximize the lower bound(LB) function F(q(z), θ) with respect to q(z). Intuitively, this happens when the LB function meets the objective likelihood function. Mathematically, this is the case because the likelihood function is independent of q(z), and so maximizing lower bound is equivalent to minimizing the KL divergence of q(z) and p(z|y, θ(t)). Therefore, the E step gives q(z) = p(z|y, θ(t)).

On the other hand, the M step tries to maximize the lower bound function with respect to θ(t) based on the fixed q(z). Recall from Eq 5 that

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*zkgYR5QraUrzBEnz5gEHvQ.png)

We can ignore q(z) in the denominator since it is independent of θ. Therefore, the M step becomes the following argmax problem at time t:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*UerGgLWIX_h5JdtsC-0Byg.png)

Repeatedly executing the E step and the M step, it’s expected to reach local maximum of the likelihood function, as indicated by the above chart.

The other way to think about the EM algorithm would be coordinate ascent. Below is a great visualization from Wikipedia.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*NmO-U3iXDbUBT1nR6HEFkw.png)

This is a powerful optimization algorithm when the function is not directly differentiable (and as so gradient descent/ascent method cannot be used). As you can see from the chart, the optimizer is working towards one direction at a time and gradually reaches the optimal value.

How does EM algorithm compare to (stochastic)gradient descent? Well, first of all, in order to apply SGD, it is necessary for the underling objective function to be differentiable, which may not always be the case. Also, it is easy for EM to be parallelized using map-reduce (mappers take care of the E step while reducers deal with the M step). To run SGD on large data, you will likely to be relying on expensive GPUs.

To conclude, I just want to speak a little bit about applications of EM algorithm. It is commonly used in Mixture of Gaussian, clustering (i.e. K-means) and Hidden Markov Model. It works well thanks to its capability of modeling the latent variables which are not presented in the input data. I’ve also put the class note from Carl Rasmussen as well as Andrew Ng below.

I hope you will find it helpful:)

**Useful Resources:**

Carl Rasmussen’s class note: [http://mlg.eng.cam.ac.uk/teaching/4f13/1819/expectation%20maximization.pdf](http://mlg.eng.cam.ac.uk/teaching/4f13/1819/expectation%20maximization.pdf)

Andrew Ng’s class note: [http://cs229.stanford.edu/notes/cs229-notes8.pdf](http://cs229.stanford.edu/notes/cs229-notes8.pdf)