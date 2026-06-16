---
title: "Bayesian Inference A step-by-step guide"
source: "https://rahuldhrh.medium.com/bayesian-inference-a-step-by-step-guide-f9db93109fa6"
author:
  - "[[Rahuldhrh]]"
published: 2024-06-05
created: 2026-06-16
description: "Let’s dive into the fascinating world of Bayesian Inference. I’ll walk you through its practical application with easy-tofollow examples."
tags:
  - "clippings"
---
![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*RGHVwtUe3-Kst3RWHIRrCw.jpeg)

Let’s dive into the fascinating world of Bayesian Inference. I’ll walk you through its practical application with easy-tofollow examples.

## Bayesian Inference:

For a quick refresher on MLE you can check my another blog on [MLE](https://medium.com/@rahuldhrh/maximum-likelihood-estimation-a-step-by-step-guide-25af44b6fa23).

Before discussing Bayesian Inference let’s discuss what do we have and why do we need anything new — We have already discussed about Maximum Likelihood Estimation to estimate unknown quantity (θ) from some known data (𝑋).  
So, what does MLE lacks –

- MLE treats the estimated quantity is constant. It seeks to find the parameter (𝜃) that maximizes the likelihood of given or observed data (𝑋).
![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*unfaJopigzplCBzRcKazBQ.png)

What if 𝜃 comes from a distribution of its own then how to incorporate it?

- As MLE finds the parameters it gives us the point estimates does not quantify any uncertainty associated with it
- MLE tends to overfit the data with a complex model specially when no estimated parameters are high.

For the problem of estimating 𝜃 from 𝑋 we discussed a certain approach where we assumed that the unknown quantity 𝜃 is fixed. This approach is called frequentist approach. To overcome the short comings of MLE we needed a different framework for inference, namely Bayesian approach. In this framework, we treat parameter θ as a random variable which comes from a distribution 𝑃(θ). This distribution 𝑃(θ) is known as Prior distribution. As we have observed data 𝑋 from that we update the Prior distribution to Posterior distribution, we do this by using Bayes Rule —

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*zirk4uSkKyQt7Ul8cL3DIw.png)

**Intuition:  
**To get an intuitive idea about Bayesian Inference Let’s investigate a simple problem  
Q: As you enter your drawing room one evening, you’re puzzled to find your sofa wet. You must play detective and solve the mystery of how this happened.  
Scenario 1: Perhaps your younger brother, engrossed in his favourite TV show, accidentally spilled his water while watching.  
Scenario 2: A mischievous Shark, stealthily infiltrating your home, leaving the sofa damp in its wake. Just as mysteriously as it appeared, the Shark vanishes upon your return.  
So, what scenario do you think caused the wet sofa?  
You can easily understand that Scenario 3 farfetched from reality and your little brother is the main culprit. But let’s analyse the Scenarios with the help of probability concepts:

![](https://miro.medium.com/v2/resize:fit:1274/format:webp/1*XsXwpGhThBVi39BGFwEkkA.png)

Wait a second the Scenario 2 is most suitable answer according to MLE? But that does not make any sense.  
If we use prior knowledge which says probability of a shark coming to your room too far fetched.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*wOrux3hWRDZXrcepFkoTuw.png)

If we use this prior knowledge then

![](https://miro.medium.com/v2/resize:fit:1210/format:webp/1*XlIvmqhIORZdUXPOUjkeTA.png)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*z82WEJ1zP7vcOxrBDbWTzg.png)

From this simple analysis, we observe that while Maximum Likelihood Estimation (MLE) suggests Scenario 3 as the most likely explanation, incorporating prior beliefs changes the solution to Scenario 2. This revised solution aligns better with the initial intuition derived rather than the MLE solution.  
This framework, known as **Bayesian inference**, involves updating the Likelihood with **Prior information** to derive revised probabilities, termed as the **Posterior probabilities**.

## Review of Parametric Statistical Inference:

Let’s review the main topics of a statistical inference problem –

- We have observed data 𝑋.
- We do not know about the probability distribution that generate 𝑋.
- We define a statistical model, that is a probability distribution that could have generated the data.
- We parameterize the proposed model with parameters 𝜃.
- We use the data 𝑋 and the model to estimate the parameters 𝜃.
- We make a statement about the data generating distribution.

Bayesian inference expands on the parametric approach by incorporating prior knowledge through probability models. We then update our beliefs using Bayes’ theorem, which helps us combine our prior knowledge with new evidence from observed data. The result is a set of posterior distributions that we can use for making decisions and drawing conclusions. This approach gives us a flexible and thorough way to handle uncertainty when estimating parameters and making decisions.

Let us discuss the building blocks of Bayesian inference one by one —

**Likelihood:  
**The initial step of parametric Bayesian inference is the likelihood, it is a function that simply says given parameter 𝜃 what is the probability of seeing the data 𝑋.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*IjvPelBsUIzebui7C9q73g.png)

The likelihood is equals to the pdf of 𝑋 when the data generating distribution’s parameter is 𝜃.  
Example –

Suppose from 𝑁 number of coin tosses generated sample is

𝑋 = \[𝑥1, 𝑥2, ⋯, 𝑥𝑁\] where 𝑥𝑖 = {0,1}.

As the data is Independent and identically distributed (IID) and follows a Bernoulli distribution. Bernoulli distribution has only one parameter 𝜇 Pdf for 𝑥𝑖 sample is

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*kKD7qBck81cAJUSXwLG4Dw.png)

We can write the likelihood as:

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*80lWgbmj02EuF-ZUVs_fNA.png)

**Prior distribution:  
**The prior distribution is a probability distribution assigned to the parameter 𝜃. For easy interpretation of Bayesian update we use conjugate priors.  
If the likelihood function 𝑃(𝑋|θ) and the prior probability distribution 𝑃(𝜃) belong to the same probability distribution family, then the resulting posterior distribution 𝑃(𝜃|𝑋) shares the same family. In such cases, we refer to the prior and posterior distributions as conjugate distributions with respect to that likelihood function.  
Example — For the previous example we can use a Beta distribution as prior.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*qZtL1Lzel5NGzKfw08mELQ.png)

Where 𝛼 and 𝛽 are the parameters of prior. Where 𝛼 indicates successes and 𝛽 failures.

## Get Rahuldhrh’s stories in your inbox

Join Medium for free to get updates from this writer.

**Posterior distribution:  
**We use information from data 𝑋 to update the prior by using Bayes Rule:

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*f5ud25YK1Hz3HcvsaNOy5Q.png)

Example —  
To continue the previous example  
The posterior becomes:

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*Lryg47RUshKV_e2PgwfAIw.png)

We’ll leave this scary looking equation as it is for now without simplifying it, as we can estimate 𝜇 from it using MAP estimation. However, by examining it, we can grasp the key concepts essential to parametric Bayesian inference.

## General Idea in Bayesian inference:

The objective is to infer information about unknown variables (parameters) θ by observing the given random variable (data) 𝑋. These unknown variables 𝜃 are connected with a prior distribution,

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*s7fRUJ06nD7zYNv0sytYng.png)

After observing the value of 𝑋, we find the posterior distribution of 𝜃. Which is the conditional pdf (or pmf) of 𝜃 given 𝑋 = x.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*UD_XgwufstQbvczBmc3tkg.png)

The posterior distribution can be found using Bayes rule.

## Example:

Let’s understand all the concepts with some examples:

**Example 1**  
A coin toss data 𝑋 is given as \[1,1,1,1,1,1,1,0,0,0\]. We need to find the parameter θ = 𝑃(𝑋 = 1)  
Solution:  
**Parameter: 𝛉 = 𝑷(𝑿 = 𝟏)**  
**Data: 𝑋 = \[1,1,1,1,1,1,1,0,0,0\]** where 1 denotes head 0 denotes tail.  
**Prior:** As we know nothing about θ we can assume θ comes from a uniform distribution.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*YYs7rPGxa84L8UDsULnyZg.png)

Prior distribution

**Likelihood:** Each samples follow a Bernoulli distribution and holds the IID assumption.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*64n_HzMETveG3eKBca80IQ.png)

Likelihood

**Posterior distribution:** By using Bayes rule we get the Posterior distribution,

![](https://miro.medium.com/v2/resize:fit:1362/format:webp/1*gC9Yov1FKTYmolJ_H7Tf8A.png)

Posterior distribution

![](https://miro.medium.com/v2/resize:fit:1360/format:webp/1*rDCbiPvavOFlEBT5zuevOQ.png)

Posterior distribution of 𝜃: 𝑓(𝜃|𝑋)

**Example 2:**  
Real value data 𝑋 is given as — **\[66.75,70.24,67.19,67.09,63.65,64.64,69.81,69.79,73.52,71.74\]  
**and that the population standard deviation is known and has value 3, We need to find the parameter μ = Ε(𝑋).  
**Solution:  
****Parameter: 𝜇 = Ε(𝑋)  
****Data: 𝑋 = \[66.75,70.24,67.19,67.09,63.65,64.64,69.81,69.79,73.52,71.74\]** **Prior:** Suppose we believe that the mean of 𝜃 is 60 with 5 as standard deviation.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*cGuhQt8-CtS966fEjL_UUg.png)

Prior distribution

**Likelihood:** Each samples follow a Normal distribution and holds the IID assumption

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*U63EvzyYGcvIf-0OEsqQ0g.png)

Likelihood

**Posterior distribution:** By using Bayes rule we get the Posterior distribution,

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*twyUD1hBETOnFrseKWZOKQ.png)

Posterior distribution

After some manipulation we can get that, (It is too long)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*zyk0748udJhzlxmezjB5vQ.png)

Posterior distribution

![](https://miro.medium.com/v2/resize:fit:1368/format:webp/1*9sMr37BVYQiIPSvVVq2jZA.png)

Posterior distribution of 𝜃: 𝑓(𝜃|𝑋)

## Iterative learning:

By using the Bayesian framework, we can develop an iterative learning system. Let’s see how we can do that:

- Start with a prior knowledge about parameters 𝜃, that 𝜃~𝑃(𝜃)
- Update that prior 𝑃(𝜃) to posterior 𝑃(𝜃|𝑋) by incorporating observed data 𝑋 using Bayes rule.
![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*u6y5pYIA5aqbmTtql-k2NA.png)

- Then set the posterior to prior and update it with new observed data 𝑌 and continue. This known as Sequential Bayesian Inference.

## Conclusion:

In conclusion, Bayesian inference offers a powerful and flexible framework for statistical analysis, particularly in the presence of uncertainty and prior knowledge. By incorporating prior distributions and using Bayes’ theorem to update these beliefs in light of new evidence, Bayesian methods allow us to make more informed and nuanced inferences about unknown parameters. This approach not only addresses the limitations of traditional methods like MLE but also provides a comprehensive probabilistic understanding that is crucial for making robust decisions in the face of uncertainty. As we continue to advance in our computational capabilities, the application and relevance of Bayesian inference are likely to grow, providing us with even deeper insights across various fields of study.