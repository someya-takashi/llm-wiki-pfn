---
title: "CausalPFN: Amortized Causal Effect Estimation via In-Context Learning"
source: "https://ar5iv.labs.arxiv.org/html/2506.07918"
author:
published:
created: 2026-06-03
description: "Causal effect estimation from observational data is fundamental across various applications. However, selecting an appropriate estimator from dozens of specialized methods demands substantial manual effort and domain e…"
tags:
  - "clippings"
---
Vahid Balazadeh   <sup>1</sup> <sup>2</sup> Hamidreza Kamkari <sup>∗</sup> <sup>3</sup> Valentin Thomas <sup>3</sup> Benson Li <sup>1</sup> <sup>2</sup> Junwei Ma <sup>3</sup>  
Jesse C. Cresswell <sup>3</sup> Rahul G. Krishnan <sup>1</sup> <sup>2</sup>  
vahid@cs.toronto.edu, hamid@layer6.ai  
<sup>1</sup> University of Toronto  <sup>2</sup> Vector Institute   <sup>3</sup> Layer 6 AI, Toronto, Canada Equal Contribution

###### Abstract

Causal effect estimation from observational data is fundamental across various applications. However, selecting an appropriate estimator from dozens of specialized methods demands substantial manual effort and domain expertise. We present CausalPFN, a single transformer that *amortizes* this workflow: trained once on a large library of simulated data-generating processes that satisfy ignorability, it infers causal effects for new observational datasets out-of-the-box. CausalPFN combines ideas from Bayesian causal inference with the large-scale training protocol of prior-fitted networks (PFNs), learning to map raw observations directly to causal effects without any task-specific adjustment. Our approach achieves superior average performance on heterogeneous and average treatment effect estimation benchmarks (IHDP, Lalonde, ACIC). Moreover, it shows competitive performance for real-world policy making on uplift modeling tasks. CausalPFN provides calibrated uncertainty estimates to support reliable decision-making based on Bayesian principles. This ready-to-use model does not require any further training or tuning and takes a step toward automated causal inference ([https://github.com/vdblm/CausalPFN](https://github.com/vdblm/CausalPFN/)).

## 1 Introduction

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.07918/assets/x1.png)

Figure 1: Time vs. Performance. Comparison across 130 causal inference tasks from IHDP, ACIC, and Lalonde. CausalPFN achieves the best average rank (by precision in estimation of heterogeneous effect) while being much faster than other baselines.

Causal inference—estimating the effects of interventions from data—is fundamental across numerous domains, including public policy, economics, and healthcare [^74] [^6] [^46]. The central challenge lies in estimating causal quantities from observational data: records collected without explicit interventions, where confounding factors can obscure true causal effects. Various causal identification settings have emerged to address this challenge [^5] [^7] [^11] [^70]. Perhaps the most common one is to assume no unobserved confounding (ignorability or backdoor) [^95] [^86].

Even within the conceptually straightforward ignorability framework, researchers have developed dozens of specialized causal estimators over the past four decades. Prominent examples include Meta-Learners [^56], doubly robust methods [^31] [^51], double machine learning (DML) [^17] [^30], and neural network approaches [^100] [^102] [^20] [^21] [^22] [^23], among others [^94] [^55] [^69] [^64]. This large number of estimators creates practical challenges as domain expertise is required to select, tune, or design the most appropriate estimator for each application [^103] [^99] [^28] [^2] [^80] [^72].

The Bayesian paradigm offers an elegant framework to address these challenges [^96] [^45] [^46] [^41] [^10]; rather than manually designing or selecting the best estimator, one can: (1) parameterize an appropriate prior distribution over plausible underlying causal mechanisms, i.e., the data-generating processes (DGPs), (2) define the causal estimand as a functional of the DGP parameters, (3) compute a posterior distribution over DGPs conditioned on observed data, and (4) derive the posterior-predictive distribution (PPD) of the causal estimand. However, the practical adoption of Bayesian methods remains limited. Computing posterior distributions typically requires expensive sampling methods [^83] [^41], which often leads researchers to make specific assumptions about the DGPs or priors that are not necessarily reflective of the complexity of the downstream tasks [^37] [^61].

Meanwhile, an emerging area in deep in-context learning suggests using large models that can approximate PPDs by taking the entire list of observations as context and amortize the expensive process of posterior inference [^33] [^32] [^53]. A successful example is the prior-fitted network (PFN) [^79] that achieved remarkable performance in tabular prediction tasks [^43] [^68] [^38] [^44] [^112] [^75] [^65]. PFNs employ transformer architectures trained on large-scale simulated DGPs, representing a rich prior, to perform posterior-predictive inference via in-context learning; given a dataset of input-output examples as context, they can predict outputs for new inputs. PFNs shift the computational burden from inference time to (pre-)training, producing a single set of model parameters that can make fast and accurate predictions on unseen datasets. However, they are only designed for regression and classification, not for causal inference.

We propose to bridge the large-scale training of amortized models with Bayesian causal inference and introduce CausalPFN, a transformer model for causal effect estimation via in-context learning. Our framework leverages a general-purpose prior, based on the *ignorability* assumption, to generate a vast collection of simulated DGPs. By training on these diverse DGPs, our method learns to infer the causal estimands directly from observational data. While our approach requires expensive pre-training taking up to seven days, once done it is ready for inference on new datasets with no further training, fine-tuning, or hyperparameter optimization. Hence, CausalPFN is easy-to-use, efficient for inference, and shows remarkably strong performance as an estimator. Figure˜1 illustrates the relative performance and efficiency of our method compared to standard baselines. For inference on an unseen dataset, CausalPFN requires only forward passes, whereas baseline methods have additional costs including hyperparameter tuning or cross-validation. We therefore report the computational time for all of these stages for the baselines to reflect the total costs of predicting on a new dataset.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.07918/assets/x2.png)

Figure 2: Traditional Causal Inference vs. CausalPFN. (Left): A domain expert manually builds or selects an estimator for a DGP that they deem appropriate for the given data. (Right): The domain expert simulates diverse DGPs for pre-training, and a transformer learns to amortize causal inference automatically.

We show CausalPFN’s workflow compared to traditional causal inference in Figure˜2. Our key contributions are: (i) To our knowledge, for the first time, we demonstrate that a single transformer-based model trained on a diverse library of simulated DGPs can match or surpass specialized estimators across multiple datasets without task-specific tuning. Specifically, CausalPFN achieves superior average performance on the IHDP, ACIC, and Lalonde benchmarks. (ii) We highlight CausalPFN’s competitive out-of-the-box performance for real-world policy making on various uplift modeling tasks. (iii) We theoretically characterize the assumptions under which CausalPFN’s estimates are asymptotically consistent. (iv) We develop a principled uncertainty quantification framework for CausalPFN to produce finite-sample calibrated credible intervals for the estimates. (v) Finally, we release our model’s weights with a user-friendly API, streamlining the adoption of CausalPFN as a capable estimator. CausalPFN is fast, ready-to-use, and does not require any further training or hyperparameter tuning.

## 2 Background

Causal Effect Estimation. We adopt the potential–outcomes framework for causal inference [^97]. Let $T$ denote treatment, $\boldsymbol{{\mathrm{X}}}$ the observed covariates, and $\mathcal{T}$ a finite treatment set. For every $t\in\mathcal{T}$, $Y_{t}$ is the potential outcome under treatment $t$, while the observed (factual) outcome is $Y\coloneqq Y_{T}$. We call the joint distribution $\operatorname{P}(\boldsymbol{{\mathrm{X}}},T,\{Y_{t}\}_{t\in\mathcal{T}},Y)$ the *data-generating process* (DGP), and its marginal distribution of the observed samples $(\boldsymbol{{\mathrm{X}}},T,Y)$, the observational distribution or $\operatorname{P}_{\mathrm{\text{obs}}}$. Given samples from $\operatorname{P}_{\mathrm{\text{obs}}}$, a central goal is to recover the *conditional expected potential outcomes* (CEPOs):

$$
\mu_{t}(\boldsymbol{{\mathrm{X}}})\coloneqq\operatorname{\mathbb{E}}[Y_{t}\mid\boldsymbol{{\mathrm{X}}}],\qquad\forall t\in\mathcal{T},\qquad\boldsymbol{{\mathrm{X}}}\sim\operatorname{P}.
$$

For binary treatments, two common estimands, average treatment effect (ATE), and conditional average treatment effect (CATE) follow directly from the CEPOs. We refer to CEPOs, CATE, and ATE collectively as *causal effects*.

$$
\displaystyle\text{ATE}:\quad
$$
 
$$
\displaystyle\lambda\coloneqq\operatorname{\mathbb{E}}[Y_{1}-Y_{0}]=\operatorname{\mathbb{E}}[\mu_{1}(\boldsymbol{{\mathrm{X}}})-\mu_{0}(\boldsymbol{{\mathrm{X}}})],
$$
$$
\displaystyle\text{CATE}:\quad
$$
 
$$
\displaystyle\tau(\boldsymbol{{\mathrm{X}}})\coloneqq\operatorname{\mathbb{E}}[Y_{1}-Y_{0}\mid\boldsymbol{{\mathrm{X}}}]=\mu_{1}(\boldsymbol{{\mathrm{X}}})-\mu_{0}(\boldsymbol{{\mathrm{X}}}).
$$

Estimating causal effects from observational data is impossible without further assumptions: different DGPs can induce the same $\operatorname{P}_{\mathrm{\text{obs}}}$ but have different causal effects [^86] [^40] [^46]. We thus define:

###### Definition 1 (Identifiability of CEPOs).

For each DGP $\operatorname{P}(\boldsymbol{{\mathrm{X}}},T,\{Y_{t}\}_{t\in\mathcal{T}},Y)$ and $t\in\mathcal{T}$, the CEPO $\mu_{t}$ is identifiable if it can be written as a functional of the observational distribution $\operatorname{P}_{\mathrm{\text{obs}}}$.

Throughout, we assume *strong ignorability*, a standard *sufficient* assumption that makes CEPOs identifiable. Strong ignorability posits that, conditional on observed covariates, treatment assignment has positive probability for all $t\in\mathcal{T}$ and is independent of all potential outcomes [^95] [^94] [^88]:

###### Assumption 1 (Strong Ignorability).

(i) $Y_{t}\perp\!\!\!\!\perp T\mid\boldsymbol{{\mathrm{X}}}$ for all $t\in\mathcal{T}$ (Unconfoundedness), and (ii) $\operatorname{P}(T=t\mid\boldsymbol{{\mathrm{X}}})>0$ a.e. for all $t\in\mathcal{T}$ (Positivity).

Bayesian Causal Inference. A Bayesian formulation of causal inference considers an explicit likelihood model for the DGP [^96] [^83] [^61]. Let $\psi$ be the parameter that indexes the DGPs $\operatorname{P}^{\psi}\left(\boldsymbol{{\mathrm{X}}},T,\{Y_{t}\}_{t\in\mathcal{T}},Y\right)$. A prior $\pi(\psi)$ encodes domain knowledge on parameters $\psi$. Given i.i.d. observations $\mathcal{D}_{\mathrm{\text{obs}}}=\left\{(\boldsymbol{{\mathrm{x}}}^{(n)},t^{(n)},y^{(n)})\right\}_{n=1}^{N}$ coming from the observational distribution $\operatorname{P}^{\psi}_{\mathrm{\text{obs}}}$, Bayes’ rule yields the posterior $\pi\left(\psi\mid\mathcal{D}_{\mathrm{\text{obs}}}\right)$. For any functional $g(\psi)$ —for example $g(\psi)=\operatorname{\mathbb{E}}^{\psi}[Y_{1}-Y_{0}]$ for ATE—the posterior predictive distribution (PPD)

$$
{\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}\pi^{g}\left(\cdot\mid\mathcal{D}_{\mathrm{\text{obs}}}\right)}\coloneqq B\;\mapsto\;\int\mathbb{I}\left(g(\psi)\in B\right){\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}\pi(\psi\mid\mathcal{D}_{\mathrm{\text{obs}}})}\mathop{}\!\mathrm{d}\psi,\qquad\forall B\in\mathcal{B},
$$

is induced by the posterior distribution ${\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}\pi\left(\psi\mid\mathcal{D}_{\mathrm{\text{obs}}}\right)}$ ($\mathcal{B}$ denotes the Borel $\sigma\text{-algebra}$ over $\mathbb{R}$). Point estimates (posterior means) and credible intervals therefore arise automatically from these induced posteriors. Because the posterior is rarely available in closed form, one resorts to approximate inference such as Markov-chain Monte-Carlo (MCMC) [^41] or variational inference [^67] [^47]. Such techniques have been applied with flexible priors including nonparametric BART models [^41] [^37], Dirichlet processes [^63] and Gaussian processes [^3]. In summary, the Bayesian paradigm offers a unified framework for inference on causal estimands and provides automatic uncertainty quantification.

Amortizing Posterior-Predictive Inference with Prior-Fitted Networks. Running a new posterior inference for every dataset is computationally demanding, especially with high-dimensional covariates [^37] [^61]. Recent work shows that in-context transformers can *amortize* Bayesian prediction: instead of sampling from the posterior at test time, a single network is trained to map a context set directly to the PPD [^32] [^33] [^53] [^79] [^38]. PFNs instantiate this idea for supervised learning [^43].

Consider a supervised dataset $\mathcal{D}^{\text{SL}}=\{(\boldsymbol{{\mathrm{x}}}^{(n)},y^{(n)})\}_{n=1}^{N}$ and a prior $\pi^{\text{SL}}$ on parameters $\phi$ indexing $\operatorname{P}^{\phi}\left(\boldsymbol{{\mathrm{X}}},Y\right)$. The Bayesian approach to predict the output for a new input $\boldsymbol{{\mathrm{x}}}$ is to use the PPD

$$
{\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}\operatorname{P}\left(Y\mid\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}},\mathcal{D}^{\text{SL}}\right)}:=\int\operatorname{P}^{\phi}\left(Y\mid\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}\right){\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}\pi^{\text{SL}}\left(\phi\mid\mathcal{D}^{\text{SL}}\right)}\mathop{}\!\mathrm{d}\phi.
$$

Rather than approximating the posterior distribution ${\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}\pi^{\text{SL}}\left(\phi\mid\mathcal{D}^{\text{SL}}\right)}$ with MCMC or variational inference [^48] [^4] [^81], PFNs parameterize the PPD directly using a single transformer model $q_{\theta}\left(Y\mid\boldsymbol{{\mathrm{X}}},\mathcal{D}^{\text{SL}}\right)$ by minimizing the *data-prior loss*

$$
\ell_{\theta}:=\operatorname{\mathbb{E}}_{\phi\sim\pi^{\text{SL}},\ \mathcal{D}^{\text{SL}}\cup\{\boldsymbol{{\mathrm{X}}},Y\}\sim\operatorname{P}^{\phi}}\left[-\log q_{\theta}\!\left(Y\mid\boldsymbol{{\mathrm{X}}},\mathcal{D}^{\text{SL}}\right)\right].
$$

Crucially, training requires only *prior* samples $(\phi,\mathcal{D}^{\text{SL}})$; no posterior sampling is needed. With a suitably rich prior, a single PFN can be applied *off-the-shelf* to diverse regression or classification problems and often surpasses bespoke models [^68] [^44].

## 3 The Mathematical Framework of CausalPFN

Our primary estimands of interest are the CEPOs from ˜1. As shown in ˜2 and ˜3, CEPOs directly enable estimation of both ATE and CATEs. Therefore, we focus on developing an estimator that can accurately infer these quantities from the observational data. Specifically, we follow the Bayesian paradigm for causal inference, as introduced in Section˜2, and parameterize CEPOs as $\mu_{t}\left(\boldsymbol{{\mathrm{X}}}\,;\,\psi\right)$. Given a suitably rich prior distribution $\pi$ over the DGPs, which we will explicitly design in Section˜4, we define our target as the posterior-predictive distribution of CEPOs:

###### Definition 2 (CEPO-PPD).

For each $t\in\mathcal{T}$ and covariate vector $\boldsymbol{{\mathrm{x}}}$, the *CEPO-PPD* is

$$
{\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}\pi^{\mu_{t}}(\cdot\mid\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}},\mathcal{D}_{\mathrm{\text{obs}}})}\;\coloneqq\;B\;\mapsto\;\int\mathbb{I}\left(\mu_{t}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}\,;\,\psi)\in B\right){\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}\pi(\psi\mid\mathcal{D}_{\mathrm{\text{obs}}})}\mathop{}\!\mathrm{d}\psi,\qquad\forall B\in\mathcal{B}.
$$

Consistent Estimation of CEPOs. The CEPO-PPD captures the epistemic uncertainty about the CEPO encoded in the posterior. A contentrated distribution $\pi^{\mu_{t}}$ indicates that the observations $\mathcal{D}_{\mathrm{\text{obs}}}$ are informative and enough samples are available to accurately pin down the true CEPO, whereas a high-variance distribution implies that the data is not sufficient for estimation. With that in mind, we now study under which conditions, increasing the size of the observations $\mathcal{D}_{\mathrm{\text{obs}}}$ allows us to accurately recover the true CEPO from the CEPO-PPD. This is given through the following informal result (re-stated and proven formally in Appendix˜B) which provides necessary and sufficient conditions on the prior $\pi$ under which the CEPO-PPDs enable consistent estimation of the CEPOs:

###### Proposition 1 (Informal).

Under mild regularity assumptions, for almost all $\psi^{\star}\sim\pi$ and any set of i.i.d. samples $\mathcal{D}_{\mathrm{\text{obs}}}\sim\operatorname{P}^{\psi^{\star}}_{\mathrm{\text{obs}}}$, we have that as $|\mathcal{D}_{\mathrm{\text{obs}}}|\to\infty$,

$$
\operatorname{\mathbb{E}}^{\pi^{\mu_{t}}}\bigl{[}\mu\mid\boldsymbol{{\mathrm{X}}},\mathcal{D}_{\mathrm{\text{obs}}}\bigr{]}\overset{a.s.}{\longrightarrow}\mu_{t}(\boldsymbol{{\mathrm{X}}}\,;\,\psi^{\star}),\quad\forall t\in\mathcal{T},\quad\boldsymbol{{\mathrm{X}}}\sim\operatorname{P}^{\psi^{\star}},
$$

if and only if the support of the prior $\pi$ only contains $\psi$ with identifiable CEPOs almost everywhere.

(Proof sketch) We group all DGPs $\psi$ that share the same observational distribution $\operatorname{P}_{\mathrm{\text{obs}}}^{\psi}$ into an equivalence class and induce a new prior obtained from $\pi$ on the resulting quotient space. By Doob’s theorem [^27] –a classical result from Bayesian consistency theory–the posterior on this new prior almost surely concentrates on the true equivalence class once asymptotically many observations are given. Consequently, for any functional of the observations that is constant within each equivalence class, its posterior-predictive converges almost surely to its true value. Importantly, the causal functional of interest, $\mu_{t}$, can be written as a functional of the observations if and only if the corresponding DGP has identifiable CEPOs. Thus, identifiability is both necessary and sufficient to ensure that $\mu_{t}$ is constant throughout the equivalence class, and for the consistency result to hold.

(Remark 1) While the algorithms in our paper use *strong ignorability*, Proposition˜1 itself is an entirely general result and can be extended to DGPs that are not necessarily ignorable, but whose CEPOs satisfy identifiability in Definition˜1. Importantly for our practical setting, when the prior $\pi$ enforces strong ignorability, Proposition˜1 suggests that the CEPO-PPDs consistently recover the true CEPO.

(Remark 2) Proposition˜1 highlights two key design principles for the prior $\pi$: $(i)$ $\pi$ must rule out non-identifiable cases, and, once identifiability is secured, $(ii)$ broadening $\pi$ increases the chance that a particular $\psi^{\star}$ lies within its support, thus enabling consistent recovery of the true CEPO for that $\psi^{\star}$.

Learning the CEPO-PPD. Having shown that CEPO-PPDs are useful for estimating the true CEPOs, we now describe how to learn them. Inspired by PFNs, we train a single transformer $q_{\theta}$ to approximate the full predictive distribution $\pi^{\mu_{t}}$. To fit this model, we introduce the following loss:

###### Definition 3 (Causal Data-Prior Loss).

For any $t\in\mathcal{T}$, we define the causal data-prior loss as

$$
\displaystyle\mathcal{L}_{t}(\theta)\;\coloneqq\;\operatorname{\mathbb{E}}_{\psi\sim\pi,\ \mathcal{D}_{\mathrm{\text{obs}}}\cup\{\boldsymbol{{\mathrm{X}}}\}\ \sim\ \operatorname{P}^{\psi}_{\mathrm{\text{obs}}}}\left[-\log{q_{\theta}(\mu_{t}(\boldsymbol{{\mathrm{X}}}\,;\,\psi)\mid\boldsymbol{{\mathrm{X}}},t,\mathcal{D}_{\mathrm{\text{obs}}})}\right].
$$

In Appendix˜C, we show that minimizing $\mathcal{L}_{t}(\theta)$ also minimizes the KL-divergence between the true CEPO-PPD and $q_{\theta}$, leading to $q_{\theta}\bigl{(}\cdot\mid\boldsymbol{{\mathrm{X}}},t,\mathcal{D}_{\mathrm{\text{obs}}}\bigr{)}\approx\pi^{\mu_{t}}\bigl{(}\cdot\mid\boldsymbol{{\mathrm{X}}},\mathcal{D}_{\mathrm{\text{obs}}}\bigr{)}$ for all $t\in\mathcal{T}$. This entire training process shifts the computational burden from inference to pre-training: rather than evaluating the posterior $\pi(\psi\mid\mathcal{D}_{\mathrm{\text{obs}}})$ and running posterior inference at test time, the model learns to map observational data directly to the corresponding predictive distribution. When the model is well fitted, the prior complies with the assumptions in Proposition˜1, and $\mathcal{D}_{\mathrm{\text{obs}}}$ is sufficiently large, the predicted $q_{\theta}$ accurately pins down the true CEPO.

Figure˜3 visually illustrates optimizing the causal data-prior loss using stochastic gradient descent: at each iteration, we sample a DGP $\psi_{i}\sim\pi$, generate an observational dataset $\mathcal{D}_{\mathrm{\text{obs}}}$ from this DGP, and select a query point $(\boldsymbol{{\mathrm{x}}},t)$. We compute (simulate) the ground-truth CEPO $\mu_{t}\left(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}\,;\,\psi_{i}\right)$ and feed both the observational data and query to the model. The model outputs a CEPO-PPD, and we update $\theta$ using gradient descent to increase the probability assigned to the true CEPO value. Through training, $\theta$ minimizes the data-prior loss and implicitly learns to perform posterior-predictive inference, and estimate the predictive distribution $\pi^{\mu_{t}}$, without ever explicitly computing the posterior.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.07918/assets/x3.png)

Figure 3: Causal Data-Prior Training. At each iteration an index ψ i ∼ π \\psi\_{i}\\sim\\pi is sampled (left), yielding the DGP P ⁡ ( X, T { Y t } ∈ 𝒯 ) \\operatorname{P}^{\\psi\_{i}}(X,T,\\{Y\_{t}\\}\_{t\\in\\mathcal{T}},Y). From this DGP we simulate an observational context 𝒟 obs \\mathcal{D}\_{\\mathrm{\\text{obs}}} and a query 𝐱 (\\boldsymbol{{\\mathrm{x}}},t) with its true μ 𝐗 =; \\mu\_{t}(\\boldsymbol{{\\mathrm{X}}}=\\boldsymbol{{\\mathrm{x}}}\\,;\\,\\psi\_{i}) (centre). Passing (\\boldsymbol{{\\mathrm{x}}},t,\\mathcal{D}\_{\\mathrm{\\text{obs}}}) through the transformer predicts the CEPO-PPD q θ ⋅ ∣ q\_{\\theta}(\\cdot\\mid\\boldsymbol{{\\mathrm{X}}}=\\boldsymbol{{\\mathrm{x}}},t,\\mathcal{D}\_{\\mathrm{\\text{obs}}}) (in yellow), which is derived from an implicit posterior \\pi(\\cdot\\mid\\mathcal{D}\_{\\mathrm{\\text{obs}}}) that is never explicitly computed (right). We train \\theta to minimize the causal data-prior loss (bottom).

Point & Distributional Estimation of Causal Effects. Given observational data $\mathcal{D}_{\mathrm{\text{obs}}}$ from an underlying $\psi^{\star}$, a natural point estimate for CEPOs is the expectation of the predicted CEPO-PPD, that is, $\operatorname{\mathbb{E}}^{q_{\theta}}\left[\mu\mid\boldsymbol{{\mathrm{X}}},t,\mathcal{D}_{\mathrm{\text{obs}}}\right]\approx\mu_{t}(\boldsymbol{{\mathrm{X}}};\psi^{\star})$. These CEPO estimates can also form point estimates for CATEs using ˜3, and for ATEs using ˜2 by empirical averaging across units in $\mathcal{D}_{\mathrm{\text{obs}}}$.

Beyond point estimation, the estimated CEPO-PPDs can also capture the epistemic uncertainty about the causal effects. We can use $q_{\theta}$ to construct credible intervals around CEPOs, CATEs, and ATEs via sampling from $q_{\theta}(\cdot\mid\boldsymbol{{\mathrm{X}}},t=1,\mathcal{D}_{\mathrm{\text{obs}}})$ and $q_{\theta}(\cdot\mid\boldsymbol{{\mathrm{X}}},t=0,\mathcal{D}_{\mathrm{\text{obs}}})$. We can then use these intervals to quantify the uncertainty of our estimated causal effects.

## 4 Implementing CausalPFN

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.07918/assets/x4.png)

Figure 4: Prior construction. Sample diverse base tables (OpenML or synthetic TabPFN), select covariates X, draw treatment T with a random propensity model, select columns μ 0, 1 \\mu\_{0},\\mu\_{1} and add zero-mean noise to form Y Y\_{0},Y\_{1}, and.

While Section˜3 presents the framework in its most general form of identification and treatment set $\mathcal{T}$, for implementation we focus on binary treatments, i.e., $\mathcal{T}=\{0,1\}$, and strong ignorability. These assumptions reflect the most common settings encountered by practitioners and serve as a natural starting point. Extending the implementation and algorithms to more general settings is left for future work.

A Scalable Prior. Here, we focus on designing an appropriate prior $\pi$ over DGPs that satisfies the theoretical requirements established in Proposition˜1. This prior must balance two factors: First, it should contain a rich set of DGPs with sufficient coverage to approximate real-world scenarios—similar to the priors used in successful tabular predictive models like TabPFN [^43] [^44], TabDPT [^68], and TabICL [^90]. Second, and uniquely for causal inference, all DGPs in our prior must satisfy strong ignorability which directly implies identifiability of the prior. Moreover, the generated DGPs must allow us to access the ground-truth CEPOs, as required by the causal data-prior loss in Definition˜3 for training.

To address these requirements, we develop a procedure that can transform *any* base table from standard tabular priors into a valid causal dataset, illustrated by Figure˜4: (i) retrieve a base table with $N$ rows from either a large library of tabular data <sup>1</sup> or synthesize it (details in Section˜D.1); (ii) randomly select columns with a varying number of covariates as $\boldsymbol{{\mathrm{X}}}$; (iii) pick two other columns, relabel them as $\mu_{0}(\boldsymbol{{\mathrm{X}}}),\mu_{1}(\boldsymbol{{\mathrm{X}}})$; (iv) optionally add zero-mean noise to $\mu_{0}(\boldsymbol{{\mathrm{X}}})$ and $\mu_{1}(\boldsymbol{{\mathrm{X}}})$ to obtain $Y_{0}$ and $Y_{1}$, or simply set $Y_{0}=\mu_{0}(\boldsymbol{{\mathrm{X}}})$ and $Y_{1}=\mu_{1}(\boldsymbol{{\mathrm{X}}})$; these four steps simulate samples from the joint distribution $(\boldsymbol{{\mathrm{X}}},Y_{0},Y_{1})$; (v) generate a random function $f$, leveraging similar synthetic functions as in [^43] to map covariates to their treatment logits; (vi) sample binary treatments $T\sim\mathrm{Bernoulli}\left(\mathrm{Sigmoid}\left(f(\boldsymbol{{\mathrm{X}}})\right)\right)$; (vii) finally, form the observed outcomes $Y:=Y_{T}$.

The procedure above “simulates” a collection $\{t^{(n)},\boldsymbol{{\mathrm{x}}}^{(n)},\mu_{0}^{(n)},\mu_{1}^{(n)},y^{(n)}\}_{n=1}^{N}$ from an underlying DGP that can be used to sample the observational data and obtain CEPOs necessary for training (recall Figure˜3). This approach guarantees strong ignorability *by design*: since treatment $T$ is determined solely from $\boldsymbol{{\mathrm{X}}}$, it is conditionally independent from the potential outcomes $Y_{0},Y_{1}$. Additionally, by applying the sigmoid function, we ensure $0<P(T=1\mid\boldsymbol{{\mathrm{X}}})<1$, satisfying positivity. While this procedure primarily targets binary treatments, it can naturally extend to finite discrete treatments.

For the diversity aspect of $\pi$, we rely on the empirical success of existing tabular foundation models and the deliberate design in our generation process. Sampling covariates directly from a mix of real and synthetic tables yields data that is more likely to reflect the scenarios the model will face at inference. We assume no distributional assumptions on covariates and potential outcomes. Section˜D.1 details additional mechanisms for controlling treatment effect heterogeneity and positivity in our synthetic DGPs, as well as the detailed configurations of the prior-generation process.

Model Architecture & Parallel Training. We model $q_{\theta}$ using a PFN-style transformer encoder that receives a sequence of row tokens as *context* (i.e., $\mathcal{D}_{\mathrm{\text{obs}}}$), where each token embeds a triplet $(t^{(n)},\boldsymbol{{\mathrm{x}}}^{(n)},y^{(n)})$. At every iteration, we embed $B_{Q}$ batched *query* tokens $(t,\boldsymbol{{\mathrm{x}}})$. We then apply 20 layers of self-attention and MLP layers, followed by a final projection layer to get $q_{\theta}\left(\cdot\mid\boldsymbol{{\mathrm{x}}},t,\mathcal{D}_{\mathrm{\text{obs}}}\right)$ for all the $(t,\boldsymbol{{\mathrm{x}}})$ pairs in the batched query. The transformer uses the asymmetric masking used in PFNs: both context and query tokens attend only to the context tokens, ensuring that the predicted CEPO-PPDs are mutually independent.

To model each CEPO-PPD, we approximate it with a quantized histogram. We discretize the outcome axis into $L=1024$ bins and let the network project the query tokens into $L$ logits. We then apply SoftMax to turn the logits into a quantized distribution $q_{\theta}(\cdot\mid\boldsymbol{{\mathrm{x}}},t,\mathcal{D}_{\mathrm{\text{obs}}})[\ell],\forall\ell\in[L]$. At each round of gradient update, we place a Gaussian with a small $\sigma$ at the true CEPO ${\mu}_{t}(\boldsymbol{{\mathrm{x}}})$ and integrate it over bins to obtain Gaussian quantized probabilities ${\mathcal{N}({\mu}_{t}(\boldsymbol{{\mathrm{x}}}),\sigma^{2})}[\ell]$ and minimize the *histogram loss*:

$$
\texttt{HL}\bigl{[}{\mu}_{t}(\boldsymbol{{\mathrm{x}}})\,\|\,q_{\theta}\bigr{]}=-\sum_{\ell=1}^{L}{\mathcal{N}({\mu}_{t}(\boldsymbol{{\mathrm{x}}}),\sigma^{2})}[\ell]\cdot\log q_{\theta}[\ell].
$$

This loss is an approximate form of the causal data-prior loss ˜9, which matches in the limit $\sigma\to 0$ and $L\to\infty$. The histogram loss affords a tractable proxy for the continuous CEPO-PPD.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.07918/assets/x5.png)

Figure 5: Architecture, Training, and Inference Details. (Left): An observational data, and a batch of B Q B\_{Q} queries along with their true CEPO values are sampled from the prior. Each observational row, containing the treatment, covariates, and the outcome form a context token, while query tokens consist of only the treatment and covariates. (Middle): The context and query tokens are fed into a transformer encoder with an asymmetric attention masking, where both context and query tokens attend only to the context tokens. (Bottom-Right): The output tokens are projected into a 1024-dimensional logit vector and softmaxed to form a discrete posterior-predictive distribution (CEPO-PPD). Then, the true CEPO value corresponding to each output token is smoothed by adding narrow-width Gaussian noise, and training is done by minimizing the cross-entropy (histogram) loss; this is a trick commonly used for stabilizing training in histogram losses. (Top-Right): At inference time, we return the CEPO-PPD mean as the point estimate and sample from CEPO-PPD to estimate credible intervals.

The detailed architecture and training setup along with the procedure for obtaining point estimates and intervals of the causal effects are illustrated and revisited in Figure˜5. For further details on parameter counts, optimization settings, training compute, and inference-time techniques, see Section˜D.2.

## 5 Experiments

Baseline Causal Effect Estimators. We compare to a broad suite of baselines. This includes double machine learning (DML) [^17] [^8] [^30], doubly robust learner (DR-Learner) [^56] [^51], as well as the T-, S-, X-, and domain adaptation learner (DA-Learner), all part of the EconML package [^12]. Moreover, we include deep-learning–based methods such as TarNet [^100], DragonNet [^102], and RA-Net [^21], implemented via the CATENets library [^20]. Finally, we compare to inverse propensity weighting (IPW) [^94], Bayesian regression trees (BART) [^41] [^16], and generalized random forests (GRF) [^8]. All the baselines, except for IPW, provide both CATE and ATE estimates.

*Importantly, we tune most of the baselines with cross-validation via grid search. The set of hyperparameter, along with the results with default hyperparameters are all detailed in Section˜D.3*.

Benchmarks with Ground-Truth Effects. A handful of benchmarks provide ground-truth causal effects, allowing us to directly measure estimation errors. We report the relative error for ATE, and the precision in estimation of heterogeneous effects (PEHE) for CATE, defined as the root mean squared deviation between predicted and true CATEs [^41]. Table˜1 compares CausalPFN to all baselines on four standard set of datasets: 100 realizations of IHDP [^92] [^41], 10 realizations of ACIC 2016 [^28], and the Lalonde CPS and Lalonde PSID cohorts [^57] with their causal effects provided by RealCause [^80] (we use the first 10 realizations). Our model demonstrates superior performance on both CATE and ATE tasks, remaining within the top three models across all the benchmarks, except for the ATE on IHDP datasets. To assess the overall performance of each method on CATE, we calculate the average rank of each method (based on PEHE) on all 130 realizations, as PEHEs are not standardized across different datasets. For ATEs, we average the relative errors directly. CausalPFN outperforms all the baselines on both average metrics. Notably, unlike other methods that are trained directly on the target datasets, our model is trained entirely on simulated data and *never* sees the evaluation data. While some baseline estimators in Table˜1 perform well on specific datasets, they underperform on others. In contrast, the consistent performance of CausalPFN suggests that amortized approaches can potentially eliminate the manual burden of task-specific estimator design.

Table 1: CATE & ATE results. Columns correspond to benchmark suites: IHDP, ACIC 2016, Lalonde CPS/PSID. (left half) mean PEHE and the average rank when pooling all tasks. (right half) mean ATE relative error and its average across all tasks. Lalonde PEHE is in thousands. Top-three per column are blue, purple, and orange. Cells with “—” indicate that the method is not applicable or has diverged.

Method Mean PEHE $\pm$ Standard Error $(\downarrow\text{better})$ Mean ATE Relative Error $\pm$ Standard Error $(\downarrow\text{better})$ IHDP ACIC 2016 Lalonde CPS Lalonde PSID Avg. IHDP ACIC 2016 Lalonde CPS Lalonde PSID Avg. ($\times 10^{3}$) ($\times 10^{3}$) Rank Error CausalPFN 0.58 $\pm$ 0.07 0.92 $\pm$ 0.11 8.83 $\pm$ 0.04 13.98 $\pm$ 0.43 2.22 $\pm$ 0.16 0.21 $\pm$ 0.04 0.04 $\pm$ 0.01 0.08 $\pm$ 0.02 0.20 $\pm$ 0.03 0.18 $\pm$ 0.03 T-Learner 1.90 $\pm$ 0.34 1.03 $\pm$ 0.08 9.21 $\pm$ 0.09 13.43 $\pm$ 0.42 4.16 $\pm$ 0.25 0.21 $\pm$ 0.04 0.04 $\pm$ 0.01 0.27 $\pm$ 0.03 0.03 $\pm$ 0.01 0.19 $\pm$ 0.03 X-Learner 2.83 $\pm$ 0.46 0.85 $\pm$ 0.14 12.19 $\pm$ 0.42 20.37 $\pm$ 0.78 5.78 $\pm$ 0.27 0.19 $\pm$ 0.03 0.03 $\pm$ 0.01 0.86 $\pm$ 0.07 0.70 $\pm$ 0.08 0.27 $\pm$ 0.03 BART 2.50 $\pm$ 0.38 0.68 $\pm$ 0.10 12.78 $\pm$ 0.11 20.86 $\pm$ 0.43 5.99 $\pm$ 0.25 0.48 $\pm$ 0.11 0.04 $\pm$ 0.01 1.01 $\pm$ 0.02 0.83 $\pm$ 0.03 0.52 $\pm$ 0.09 DragonNet 2.16 $\pm$ 0.25 2.11 $\pm$ 0.18 10.31 $\pm$ 0.29 15.64 $\pm$ 0.53 6.08 $\pm$ 0.22 0.22 $\pm$ 0.03 0.07 $\pm$ 0.02 0.52 $\pm$ 0.08 0.40 $\pm$ 0.07 0.24 $\pm$ 0.03 S-Learner 3.45 $\pm$ 0.60 1.19 $\pm$ 0.15 12.77 $\pm$ 0.09 22.00 $\pm$ 0.48 6.37 $\pm$ 0.29 0.20 $\pm$ 0.04 0.05 $\pm$ 0.01 1.01 $\pm$ 0.02 0.95 $\pm$ 0.02 0.31 $\pm$ 0.04 TarNet 1.80 $\pm$ 0.14 2.20 $\pm$ 0.20 — 18.93 $\pm$ 0.42 6.64 $\pm$ 0.17 0.23 $\pm$ 0.04 0.07 $\pm$ 0.02 — 0.76 $\pm$ 0.04 0.26 $\pm$ 0.03 GRF 3.67 $\pm$ 0.60 1.32 $\pm$ 0.28 12.40 $\pm$ 0.19 22.39 $\pm$ 0.45 6.68 $\pm$ 0.28 0.18 $\pm$ 0.03 0.07 $\pm$ 0.02 0.81 $\pm$ 0.06 0.80 $\pm$ 0.05 0.52 $\pm$ 0.09 DR-Learner 3.45 $\pm$ 0.54 1.09 $\pm$ 0.15 20.05 $\pm$ 1.88 33.99 $\pm$ 8.71 6.74 $\pm$ 0.27 0.16 $\pm$ 0.03 0.06 $\pm$ 0.02 1.68 $\pm$ 0.67 0.75 $\pm$ 0.07 0.31 $\pm$ 0.07 Forest DML 4.31 $\pm$ 0.70 1.42 $\pm$ 0.29 171.0 $\pm$ 50.6 22.66 $\pm$ 0.51 7.35 $\pm$ 0.28 0.09 $\pm$ 0.01 0.04 $\pm$ 0.01 2.31 $\pm$ 0.66 0.96 $\pm$ 0.03 0.32 $\pm$ 0.08 RA-Net 2.35 $\pm$ 0.19 2.35 $\pm$ 0.24 11.50 $\pm$ 0.31 16.95 $\pm$ 0.78 7.57 $\pm$ 0.19 0.23 $\pm$ 0.03 0.07 $\pm$ 0.02 0.75 $\pm$ 0.06 0.44 $\pm$ 0.05 0.27 $\pm$ 0.03 IPW — — — — — 0.23 $\pm$ 0.04 0.24 $\pm$ 0.05 0.25 $\pm$ 0.03 0.05 $\pm$ 0.01 0.22 $\pm$ 0.03

Policy Evaluation on Marketing Randomized Trials. Ground-truth CATEs are only available for synthetic or semi-synthetic datasets. However, if a randomized controlled trial (RCT) is available, we can still evaluate the quality of a CATE estimator by assessing the performance of policies derived from it. A common tool for evaluating such policies is the *Qini curve* [^91], which plots the cumulative treatment effect when units are ranked in descending order of their predicted CATE.

Formally, let $(y^{(n)},t^{(n)})_{n=1}^{N}$ denote outcomes and binary treatments from an RCT, and let $\widehat{\tau}_{n}$ be the corresponding CATE estimates, ordered so that $\widehat{\tau}_{1}\geq\cdots\geq\widehat{\tau}_{N}$. Define

$$
\lambda(q)\coloneqq\sum_{n=1}^{\lfloor qN\rfloor}\Bigl{(}\tfrac{t^{(n)}y^{(n)}}{r(q)}-\tfrac{(1-t^{(n)})y^{(n)}}{1-r(q)}\Bigr{)},\qquad Q(q)\coloneqq q\cdot\lambda(q)/\lambda,\qquad 0\leq q\leq 1,
$$

where $r(q)=\tfrac{1}{\lfloor qN\rfloor}\sum_{n}^{\lfloor qN\rfloor}t^{(n)}$ is the empirical treatment rate for the first $q$ -quantile of units. Because the data comes from an RCT, $\lambda(q)$ unbiasedly estimates the ATE for the top $q$ -quantile of units ranked by predicted CATEs. Plotting $Q(q)$ against the treated fraction $q$ yields the (normalized) Qini curve, and the area under this curve is called the *Qini score*. A random ranking produces a baseline curve as a straight line from $(0,0)$ to $(1,1)$. The higher the Qini curve lies above this line, the better the model prioritizes high-impact units with larger CATE values, leading to greater lift and policy benefit.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.07918/assets/x6.png)

Figure 6: Hill (1) & Hill (2) Qini curves.

We benchmark CausalPFN on five large marketing RCTs from the scikit-uplift library [^73]. The first dataset, Hillstrom [^42], includes 64,000 customers randomly assigned to one of three treatments: no e-mail, an e-mail advertising men’s merchandise, or an e-mail advertising women’s merchandise. The outcome is whether a website visit occurred within two weeks (binary). We consider two causal tasks: Hill <sup>(1)</sup> – Men’s-merchandise e-mail (treatment) vs. no e-mail (control), and Hill <sup>(2)</sup> – Women’s-merchandise e-mail vs. no e-mail. We estimate CATEs using CausalPFN (five-fold honest splitting) and X Learner. Figure˜6 shows Qini curves where CausalPFN closely matches X Learner across the targeting range. Notably, Hill <sup>(2)</sup> shows much greater gains, *suggesting focusing on women’s-merchandise ad campaigns, compared to men’s, can drive more gains in the number of website visits*. We also evaluate CausalPFN on four larger campaigns—Lenta, Retail Hero (X5), Megafon (Mega), and Criteo [^60] [^93] [^77] [^115] —each with $\sim$ $10^{6}$ rows. For tractability, we compute Qini scores on stratified $50$ k subsamples; Table˜2 shows CausalPFN achieves the best mean performance. However, when we run it on full tables (see Table˜5 of Section˜D.4), we observe a drop in performance, which aligns with known context-length limitations of PFN-style transformers on large tables [^104]. Still, the strong subsample results highlight the potential of scaling CausalPFN to longer contexts, which remains an important future direction.

Uncertainty & Calibration. Recall from Section˜3 that for each unit covariate $\boldsymbol{{\mathrm{x}}}$, CausalPFN can produce both point estimates and credible intervals for the CATE and CEPOs. We do so by drawing 10,000 samples from the quantized distributions $q_{\theta}(\cdot\mid\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}},t,\mathcal{D}_{\mathrm{\text{obs}}})$ and construct credible intervals at any desired significance level $\alpha$. Here, we evaluate these intervals, focusing on the model’s calibration. We also assess a key assumption from Proposition˜1—whether the inference-time DGP $\psi^{\star}$ lies within the support of the prior $\pi$, and how the model behaves when this assumption is violated.

We define families of synthetic DGPs to simulate both in-distribution and out-of-distribution (OOD) scenarios. Each DGP samples covariates $\boldsymbol{{\mathrm{X}}}$ from a uniform distribution, defines a treatment logit function $f$ and CEPO functions $\mu_{t}$ for $t\in\{0,1\}$, assigns treatment via $T\sim\mathrm{Bernoulli}\left(\mathrm{Sigmoid}\left(f(\boldsymbol{{\mathrm{X}}})\right)\right)$, and generates potential outcomes as $Y_{t}=\mu_{t}(\boldsymbol{{\mathrm{X}}})+\epsilon_{t}$, where $\epsilon_{t}$ is drawn from a standard Uniform, Gaussian, or Laplace. We consider two DGP families; Sinusoidal, where $f$ and $\mu_{t}$ are functions with sinusoidal components, and Polynomial, where the functions $f$ and $\mu_{t}$ are polynomials of varying degree (see Section˜D.5 for detailed configurations). CausalPFN is trained either on the same family it is tested on, or on a different one (OOD).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.07918/assets/x7.png)

Figure 7: Calibration. (Left): CATE coverage vs. nominal credibility. In-distribution DGPs (blue) lie on or above the diagonal (calibrated/conservative), while OOD DGPs (orange) fall below it (overconfident). (Middle): Across model–DGP pairs, CATE ICE (x-axis) exceeds regression ICE (y-axis). (Right): Temperature scaling based on regression ICE ensures the model is either calibrated or conservative for both in- and out-of-distribution DGPs.

For a unit with covariates $\boldsymbol{{\mathrm{x}}}$ and significance level $\alpha$, we say the true CATE is *covered* if $\tau(\boldsymbol{{\mathrm{x}}})$ lies within the predicted $100(1-\alpha)\%$ interval obtained using samples from $q_{\theta}$. Plotting Bayesian coverage against nominal levels of $\alpha$ yields the CATE calibration curve. As shown in Figure˜7 (left), CausalPFN is reliably calibrated under in-distribution settings but becomes severely overconfident when evaluated on OOD DGPs ($\psi^{\star}\not\sim\pi$). This aligns with prior observations that neural models often exhibit pathological overconfidence under distribution shift [^36] [^85].

To correct this, we apply a temperature parameter $\theta_{T}$ to the SoftMax that outputs the quantized CEPO-PPD from the logits of the model. We aim to tune $\theta_{T}$ to minimize the calibration error. However, direct CATE calibration is impossible because $\tau(\boldsymbol{{\mathrm{x}}})$ is never observed at test-time. Instead, we introduce the *regression calibration* based on observational data: an observed triple $(t,\boldsymbol{{\mathrm{x}}},y)$ is covered by the predicted credible interval when $y$ lies inside the model’s predicted interval for the CEPO-PPD $\mu_{t}\left(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}\,;\,\psi^{\star}\right)$. With that in mind, we let $\widehat{\mathrm{cov}}_{\mu}(\alpha)$ and $\widehat{\mathrm{cov}}_{\tau}(\alpha)$ denote the Bayesian coverage at level $\alpha$ for the regression and CATE calibration curves, respectively, and define

$$
\mathrm{ICE}_{\mu}\coloneqq\int_{0}^{1}\bigl{(}\widehat{\mathrm{cov}}_{\mu}(\alpha)-\alpha\bigr{)}\,\mathop{}\!\mathrm{d}\alpha,\text{ and }\qquad\mathrm{ICE}_{\tau}\coloneqq\int_{0}^{1}\bigl{(}\widehat{\mathrm{cov}}_{\tau}(\alpha)-\alpha\bigr{)}\,\mathop{}\!\mathrm{d}\alpha,
$$

as the integrated coverage error (ICE) for regression and CATE (negative values $=$ overconfidence).

Note that we do not expect $\widehat{\mathrm{cov}}_{\mu}$ to be calibrated: regression intervals combine epistemic uncertainty of the CEPO with the irreducible (aleatoric) noise in $Y$, so $\mathrm{ICE}_{\mu}$ is biased. Still, it holds a useful signal. Across all model–DGP pairs in Figure˜7 (middle), we consistently observe $\mathrm{ICE}_{\mu}\leq\mathrm{ICE}_{\tau}$: the regression curve sits at or below the CATE curve. While $\mathrm{ICE}_{\tau}$ is inaccessible without having the true CATE, $\mathrm{ICE}_{\mu}$ is computable from observational data. Consequently, temperature-scaling the logits to lift $\widehat{\mathrm{cov}}_{\mu}$ to the diagonal also calibrates the CATE intervals or makes them conservative. We thus tune $\theta_{T}$ by grid search to drive $\mathrm{ICE}_{\mu}$ to zero using a 5-fold calibration on the observational data. The calibrated curves in Figure˜7 (right) confirm that, after temperature scaling, CausalPFN’s overconfidence on the OOD test-sets disappears. Additional synthetic train-/test-DGP pairs and real-world data experiments appear in Section˜D.5.

Comparison to TabPFN. We also compare against the latest version of TabPFN [^44], plugging its regression output as a proxy for CEPO. As Table˜3 shows, TabPFN is surprisingly competitive without any causal tuning, yet CausalPFN outperforms it on every benchmark except ACIC 2016. To isolate the benefit of training on a causal prior, compared to the predictive *non-identifiable* prior in TabPFN, we fine-tune it on our prior for 48 hours on an H100 GPU. This causal fine-tuning boosts the performance and confirms the added value of identifiable priors for causal effect estimation.

Table 3: TabPFN Comparison. PEHE (left half) alongside ATE relative error (right half). TabPFN <sup>⋆</sup> is the latest TabPFN model [^44] tuned with our prior. Best numbers are highlighted.

Method PEHE $\pm$ Standard Error ($\downarrow$ better) ATE Relative Error $\pm$ Standard Error ($\downarrow$ better) IHDP ACIC 2016 Lalonde CPS Lalonde PSID IHDP ACIC 2016 Lalonde CPS Lalonde PSID ($\times 10^{3}$) ($\times 10^{3}$) CausalPFN (Ours) 0.58 $\pm$ 0.07 0.92 $\pm$ 0.11 8.83 $\pm$ 0.04 13.98 $\pm$ 0.41 0.21 $\pm$ 0.04 0.04 $\pm$ 0.01 0.08 $\pm$ 0.02 0.20 $\pm$ 0.04 TabPFN <sup>⋆</sup> (Ours) 0.90 $\pm$ 0.16 0.47 $\pm$ 0.05 8.97 $\pm$ 0.06 14.90 $\pm$ 0.95 0.21 $\pm$ 0.04 0.03 $\pm$ 0.01 0.17 $\pm$ 0.02 0.22 $\pm$ 0.08 TabPFN 0.95 $\pm$ 0.20 0.54 $\pm$ 0.08 9.45 $\pm$ 0.19 18.7 $\pm$ 0.83 0.21 $\pm$ 0.04 0.03 $\pm$ 0.01 0.32 $\pm$ 0.05 0.60 $\pm$ 0.07

## 6 Related Work

Single–Dataset Estimators. Common methods for causal effect estimation are trained and applied on a single dataset. Representative examples include the X-, S-, DR-, and RA-Learners, as well as IPW and DML [^12]. Alongside these approaches, several neural variants such as TARNet [^100], DragonNet [^102], CEVAE [^67], and NCMs [^108] [^109] have been proposed; however, all of them still require per-dataset training and do not amortize across various datasets.

Amortized Causal Inference. Amortized methods train a *single* network to map observational data to causal quantities across *multiple* DGPs. Existing methods either first recover a causal graph representing the observational data and then compute interventions on that graph [^98] [^71], mirroring ideas from causal-discovery [^87] [^114] [^52] [^66] [^50] [^49], while others infer effects end-to-end [^82] [^113] [^15]. In parallel, amortized approaches have also been explored for causal decision-making, where the goal is to learn policies or decisions that generalize across environments or tasks [^59] [^58].While all these methods can conceptually be used for amortized causal effect estimation, none of them provide a ready-to-use estimator that can surpass the specialized single-dataset estimators on standard benchmarks. They either rely on synthetic datasets as proof-of-concept or require multiple datasets similar to the one given at inference to train their estimators. Contrary to prior work, our method is trained once and produces causal effects without any access to, or adaptation on, the test-time DGPs. CausalPFN, through large-scale training, yields out-of-the-box performance that surpasses the specialized single-dataset estimators, setting a new milestone for amortized methods.

Scaling In-Context Transformers. In-context learning with transformers has shown impressive results across a range of domains [^14] [^110] [^19] [^26] [^105]. Although the underlying mechanisms responsible for this success remain an active area of research [^1] [^24] [^84] [^106] [^62] [^111] [^107] [^9] [^89], increasing model size and training data have consistently and undoubtedly led to stronger performance. This success has recently extended to tabular prediction with models such as TabPFN [^43] [^44], TabDPT [^68], and TabICL [^90], which are trained on broad prior distributions and perform well on real-world data without fine-tuning. CausalPFN complements these works, demonstrating that—with sufficient scale and training—in-context learning can also be effectively adapted to causal inference.

## 7 Conclusions, Limitations, and Future Work

In this paper, we introduced a practical paradigm for amortized causal effect estimation that combines Bayesian causal inference with large-scale tabular training. Despite learning solely from simulated data, CausalPFN matches, and often outperforms, specialized causal estimators across diverse real-world domains. Through amortization, we significantly reduce the burden of estimator selection at inference time, and to foster adoption, we have open-sourced the code and presets.

That said, several important limitations remain: (i) Our approach fundamentally assumes strong ignorability, which is an untestable assumption in practice. Without this condition, CausalPFN has no guarantees of validity. Domain expertise still remains essential to determine whether this method is appropriate or whether alternative approaches should be used. (ii) Our theoretical guarantees rely on idealistic assumptions: a well-specified prior and asymptotically large datasets. We lack finite-sample theory characterizing the estimator’s behavior in practical settings. Investigating robustness to prior misspecification and developing finite-sample guarantees remain open problems. Recent work on theory of valid adjustment sets [^18] may offer promising directions for addressing these challenges. (iii) Performance degradation is evident on the largest marketing tables (Table˜5), reflective of the known size-scalability trade-off inherent to PFN-style models [^44]. (iv) While CausalPFN already supports multi-arm discrete treatments with a finite set $\mathcal{T}$, we have only implemented it for the binary $\mathcal{T}$. Additionally, extending to the continuous treatment setting where $\mathcal{T}$ is not finite remains fully unexplored. (v) Finally, our entire implementation relies on the strong ignorability or backdoor assumption. Extending our framework to richer domain-informed priors like instrumental variables can broaden the framework’s reach, although designing scalable priors for such cases is non-trivial.

## References

## Appendix Contents

\[sections\] \[sections\]1

## Appendix A Notation, Definitions, and Assumptions

Sample Space. Let $\mathcal{B}$ be the Borel $\sigma$ -algebra of $\operatorname{\mathbb{R}}$. Let the random variable $Z=\left(\boldsymbol{{\mathrm{X}}},T,Y\right)$ denote all the observed random variables, defined on the Polish space $\mathcal{Z}$ with $\sigma$ -algebra $\mathcal{B}_{\mathcal{Z}}$. In particular, $\boldsymbol{{\mathrm{X}}}\in\mathcal{X}$, $T\in\mathcal{T}$ with finite $\mathcal{T}$, and $Y\in\operatorname{\mathbb{R}}$. To reason about counterfactuals, we enlarge the space and let $\tilde{Z}=\left(\boldsymbol{{\mathrm{X}}},T,\{Y_{t}\}_{t\in\mathcal{T}},Y\right)$ be the random variable denoting all the observed and unobserved (potential outcome) variables, defined on the extended sample space $\tilde{\mathcal{Z}}$ with $\sigma$ -algebra $\mathcal{B}_{\tilde{\mathcal{Z}}}$.

Data–Generating Parameters. Let $\Psi$ be a Polish parameter space with $\sigma$ -algebra $\mathcal{B}_{\Psi}$. For any $\psi\in\Psi$, a parameterized DGP is specified by a probability measure $\operatorname{P}^{\psi}$ on $(\tilde{}\mathcal{Z},\mathcal{B}_{\tilde{}\mathcal{Z}})$, and as a result, it also induces a (marginal) probability measure $\operatorname{P}_{\mathrm{\text{obs}}}^{\psi}$ on the observational space $(\mathcal{Z},\mathcal{B}_{\mathcal{Z}})$. We slightly abuse the notation and write $\psi$ for both the random variable and its realized value where the meaning is clear from context. We impose our first mild *regularity* condition that guarantees measurability of all objects introduced later and makes the consistency proofs well-posed.

###### Assumption 2 (Measurability).

The mappings $\psi\mapsto\operatorname{P}^{\psi}$ is measurable, in the sense that $\psi\mapsto\operatorname{P}^{\psi}\left(B\right)$ is $\mathcal{B}_{\Psi}$ -measurable for each $B\in\mathcal{B}_{\tilde{\mathcal{Z}}}$. Moreover, the mapping $[\cdot]=\psi\mapsto\operatorname{P}_{\mathrm{\text{obs}}}^{\psi}$ is measurable and its image $[\Psi]=\{\operatorname{P}_{\mathrm{\text{obs}}}^{\psi}\,;\,\psi\in\Psi\}$ is a Borel subset of a Polish space.

Parameteric CEPOs. For any $t\in\mathcal{T}$, parameter $\psi\in\Psi$, and $\boldsymbol{{\mathrm{X}}}\sim\operatorname{P}^{\psi}$, we define CEPOs as

$$
\mu_{t}\left(\boldsymbol{{\mathrm{X}}}\,;\,\psi\right)\coloneqq\operatorname{\mathbb{E}}^{\operatorname{P}^{\psi}}\left[Y_{t}\mid\boldsymbol{{\mathrm{X}}}\right].
$$

Prior and Posterior Distributions. We define the prior $\pi$ as a probability measure on the parameter space $(\Psi,\mathcal{B}_{\Psi})$. Since Assumption˜2 holds, we can define the joint distribution $\operatorname{P}^{\pi}$ of $\left(\left(\tilde{Z}_{1},\tilde{Z}_{2},\ldots,\right),\psi\right)$ by letting $\psi\sim\pi$ and $(\tilde{Z}_{i})_{i\geq 1}\mid\psi$ i.i.d. from $\operatorname{P}^{\psi}$. Specifically, we use the same notation $\operatorname{P}^{\pi}$ to denote its marginal distributions and write $\boldsymbol{{\mathrm{X}}}\sim\operatorname{P}^{\pi}$ as the marginal distribution of the observed covariates in $\tilde{Z}_{1}$. Using this measure, we define our second *regularity* assumption, which involves the CEPOs:

###### Assumption 3 (Integrability).

For all $t_{0}\in\mathcal{T}$, and $\operatorname{P}^{\pi}$ -almost all $\boldsymbol{{\mathrm{x}}}_{0}$, $\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi)$ exists and $\operatorname{\mathbb{E}}^{\pi}\left[|\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi)|\right]<\infty$.

Given the joint distribution $\operatorname{P}^{\pi}$, we can define the observational data $\mathcal{D}^{n}_{\mathrm{\text{obs}}}\coloneqq\left(Z_{1},Z_{2},\ldots,Z_{n}\right)$ as the first $n$ observed random variables, sampled from the marginal distribution of $\operatorname{P}^{\pi}$. This allows us to define the posterior distribution $\pi(\psi\mid\mathcal{D}_{\mathrm{\text{obs}}}^{n})$ as the conditional distribution $\operatorname{P}^{\pi}\left(\psi\mid\mathcal{D}_{\mathrm{\text{obs}}}^{n}\right)$. We then define the posterior-predictive distribution (CEPO-PPD) for any $t\in\mathcal{T}$ and $B\in\mathcal{B}$ as follows:

$$
\pi^{\mu_{t}}\left(B\mid\boldsymbol{{\mathrm{X}}},\mathcal{D}_{\mathrm{\text{obs}}}^{n}\right)\coloneqq\int_{\Psi}\mathbb{I}\left(\mu_{t}\left(\boldsymbol{{\mathrm{X}}}\,;\,\psi\right)\in B\right)\mathop{}\!\mathrm{d}\pi\left(\psi\mid\mathcal{D}_{\mathrm{\text{obs}}}^{n}\right).
$$

Model. Given a query $(t,\boldsymbol{{\mathrm{x}}})$ and context (observational data) $\mathcal{D}_{\mathrm{\text{obs}}}^{n}$, a model with parameters $\theta$ yields the predictive distribution $q_{\theta}(\,\cdot\mid\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}},t,\mathcal{D}_{\mathrm{\text{obs}}}^{n})$ over CEPO values. For both CEPO-PPDs $\pi^{\mu_{t}}$ and the model $q_{\theta}$, we will use the same symbol for the measures and their (Lebesgue) densities. Finally, unless stated otherwise, we assume both $q_{\theta}$ and $\pi^{\mu_{t}}$ have full support on $\operatorname{\mathbb{R}}$.

## Appendix B Consistency Result

### B.1 Re-Statement of

###### (Formal).

Under Assumptions˜2 and 3, there exists sets $\mathcal{X}_{0}\subseteq\mathcal{X}$ with $\operatorname{P}^{\pi}\left(\mathcal{X}_{0}\right)=1$ and $\Psi^{\star}\subseteq\Psi$ with $\pi\left(\Psi^{\star}\right)=1$, such that for all $t_{0}\in\mathcal{T},\boldsymbol{{\mathrm{x}}}_{0}\in\mathcal{X}_{0}$, and all $\psi^{\star}\in\Psi^{\star}$, if $Z_{1},Z_{2},\ldots\sim\operatorname{P}^{\psi^{\star}}_{\mathrm{\text{obs}}}$ i.i.d., then

$$
\lim_{n\to\infty}\operatorname{\mathbb{E}}^{\pi^{\mu_{t_{0}}}}\bigl{[}\mu\mid\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0},\mathcal{D}_{\mathrm{\text{obs}}}^{n}\bigr{]}\overset{a.s.}{=}\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi^{\star}),
$$

if and only if the support of the prior $\pi$ only contains $\psi$ with identifiable CEPOs almost everywhere..

### B.2 Preliminaries for the Proof of

Here, we introduce some concepts to simplify the statement of the proof. We start by characterizing the set of DGPs with the same observational distributions:

###### Definition 4 (Observational Quotient Space).

Let $\Phi\coloneqq\Psi/\sim$ be the set of equivalence classes under the the following relation: $\psi_{1}\sim\psi_{2}$ if and only if $\operatorname{P}^{\psi_{1}}_{\mathrm{\text{obs}}}=\operatorname{P}^{\psi_{2}}_{\mathrm{\text{obs}}}$. Since $\operatorname{P}^{\psi}_{\mathrm{\text{obs}}}$ is characterized by the equivalence class of $\psi$, we can override the measurable map $[\cdot]:\Psi\to\Phi$ in Assumption˜2 and write $[\psi]$ to denote the equivalence class corresponding to the parameter $\psi$.

The Joint Measure $\Pi$. For technical convenience, we define a single joint measure $\Pi$ on all the random variables defined so far, as the pushforward measure of $\operatorname{P}^{\pi}$ by the following mapping:

$$
\Big{(}\psi,(\tilde{Z}_{i})_{i\geq 1}\Big{)}\mapsto\Big{(}\psi,(\tilde{Z}_{i})_{i\geq 1},[\psi],(Z_{i})_{i\geq 1}\Big{)}.
$$

Note that by Assumption˜2, the mapping above is measurable and thus $\Pi$ is well-defined. In particular, we have the following conditional independence statement:

$$
\Pi\big{(}(Z_{i})_{i\geq 1}\mid\psi,[\psi]\big{)}=\Pi\big{(}(Z_{i})_{i\geq 1}\mid[\psi]\big{)}\implies(Z_{i})_{i\geq 1}\perp\!\!\!\!\perp_{\Pi}\psi\mid[\psi].
$$

The joint measure $\Pi$ lets us view each quantity as a marginal or conditional under a single umbrella. Importantly, we can easily rewrite the expected CEPO-PPD in Proposition˜1 as follows:

$$
\operatorname{\mathbb{E}}^{\pi^{\mu_{t_{0}}}}\bigl{[}\mu\mid\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0},\mathcal{D}_{\mathrm{\text{obs}}}^{n}\bigr{]}=\operatorname{\mathbb{E}}[\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi)\mid\mathcal{D}_{\mathrm{\text{obs}}}^{n}]
$$

where the expectation on the R.H.S. is taken w.r.t. $\Pi$ with $\psi$ and $\mathcal{D}_{\mathrm{\text{obs}}}^{n}$ being the random objects under $\Pi$. In fact, unless stated otherwise, all of the expectations will be w.r.t. $\Pi$ hereafter.

Notation Remark. We use the same symbol $\Pi$ to denote the *joint* measure, as well as any of its *marginal* or *conditional* measures. The distinction will be clear from context. For example, writing $\Pi\left(\Psi^{\star}\right)$ for some $\Psi^{\star}\subseteq\Psi$ refers to the marginal distribution of $\Pi$ on the parameters, i.e., $\pi\left(\Psi^{\star}\right)$. Similarly, a statement like $\Pi$ -almost all $\psi$ is equivalent to $\pi$ -almost all $\psi$. Moreover, writing $\boldsymbol{{\mathrm{X}}}\sim\Pi$ means that $\boldsymbol{{\mathrm{X}}}$ is drawn from the marginal distribution of $\Pi$ on the observed covariates of $Z_{1}$.

Simplifying Proposition˜1. Using $\Pi$ and these notation abuses, we can simplify Proposition˜1 and write the following equivalent statement that is easier to ration about:

###### .

Under Assumptions Assumptions˜2 and 3, there exists a set $\mathcal{X}_{0}\subseteq\mathcal{X}$ with $\Pi(\mathcal{X}_{0})=1$ where for all $\boldsymbol{{\mathrm{x}}}_{0}\in\mathcal{X}_{0}$ and $t_{0}\in\mathcal{T}$ we have that: for $\Pi$ -almost all $\psi^{\star}$ and $\mathcal{D}_{\mathrm{\text{obs}}}^{n}\sim\Pi(Z_{1},\ldots Z_{n}\mid\psi^{\star})$

$$
\lim_{n\to\infty}\operatorname{\mathbb{E}}\left[\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi)\,\big{|}\,\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0},\mathcal{D}^{n}_{\mathrm{\text{obs}}}\right]\stackrel{{\scriptstyle a.s.}}{{=}}\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi^{\star}),
$$

if and only if we have identifiable CEPOs $\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi)$ for $\Pi$ -almost all $\psi$.

With that established, we will proceed with the proof of this more simplified form of Proposition˜1.

### B.3 Proof

For any given $t$ and $\boldsymbol{{\mathrm{x}}}$, $\mu_{t}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}\,;\,\psi)$ can be viewed as a functional mapping $\psi\in\Psi$ to an element in $\operatorname{\mathbb{R}}$. We start of by defining an important functional $g_{t}$ that maps from the quotient $\Phi$ to $\operatorname{\mathbb{R}}$ instead:

$$
g_{t}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}\,;\,\phi)=\operatorname{\mathbb{E}}\left[\mu_{t}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}\,;\,\psi)\mid\psi\in[\phi]^{-1}\right],
$$

where $[\phi]^{-1}$ is the preimage of the quotient mapping $[\cdot]$ and denotes all of the DGPs that produce the same observational distribution characterized by $\phi$. It should be noted that condition $\psi\in[\phi]^{-1}$ defines a valid sub- $\sigma$ -algebra, and a valid conditional expectation as a result, only if $\Pi$ is well-defined under the regularity Assumptions Assumptions˜2 and 2.

Now, we present a corollary of Doob’s consistency theorem without proof (Corollary 2.3 of [^78]) which we heavily leverage. The result is re-stated to match our parallel notation:

###### Theorem 2 (Corollary of Doob’s Consistency Theorem).

Suppose $\mathcal{Z}$ and $\Phi$ are Borel measurable spaces, and let $\mathcal{B}_{\mathcal{Z}}$ and $\mathcal{B}_{\Phi}$ be their induced Borel $\sigma$ -algebras of $\mathcal{Z}$ and $\Phi$. For each $\phi\in\Phi$, let $\operatorname{P}^{\phi}$ be a probability measure on $\left(\mathcal{Z},\mathcal{B}_{\mathcal{Z}}\right)$, and $\nu$ a probability measure on $(\Phi,\mathcal{B}_{\Phi})$. Then for a given measurable functional $g:\Phi\to\operatorname{\mathbb{R}}$, we assume the following assumptions hold:

1. Polish. $\mathcal{Z}$ and $\Phi$ are each Borel subsets of a Polish space.
2. Measurability. $\phi\mapsto\operatorname{P}^{\phi}\left(B\right)$ is measurable for every $B\in\mathcal{B}_{\mathcal{Z}}$.
3. No Redundancy. $\phi\neq\phi^{\prime}\implies\operatorname{P}^{\phi}\neq\operatorname{P}^{\phi^{\prime}}$.
4. Integrability. $\operatorname{\mathbb{E}}^{\nu}\left[|g({\phi})|\right]<\infty$.

Abusing notation slightly, and using $\phi$ to denote both a random variable and its realization, we define the extended joint probability measure $\nu_{\text{tot}}$ on $\big{(}(Z_{1},Z_{2},\ldots),\phi\big{)}$ defined by first drawing $\phi\sim\nu$, and then sampling $Z_{1},Z_{2},\ldots\mid\phi$ i.i.d. from $\operatorname{P}^{\phi}$. Then, there exists $\Phi_{0}\subseteq\Phi$ with $\nu(\Phi_{0})=1$ and for any $\phi_{0}\in\Phi_{0}$ and $Z_{1},Z_{2},\ldots\sim\operatorname{P}^{\phi_{0}}$ i.i.d., we have that

$$
\lim_{n\to\infty}\operatorname{\mathbb{E}}^{\Pi_{\text{tot}}}\left[g({\phi})\mid Z_{1},\ldots,Z_{n}\right]\stackrel{{\scriptstyle a.s.}}{{=}}g(\phi_{0}).
$$

We now write an important Lemma that uses Theorem˜2 to establish a consistency result connecting the CEPO-PPDs the new functionals defined in ˜20:

###### Lemma 3.

Under Assumptions Assumptions˜2 and 3, there exists a set $\mathcal{X}_{0}\subseteq\mathcal{X}$ such that for all $\boldsymbol{{\mathrm{x}}}_{0}\in\mathcal{X}_{0}$ and $t_{0}\in\mathcal{T}$: for $\Pi$ -almost all $\phi_{0}\in\Phi_{0}$, if $\mathcal{D}^{n}_{\mathrm{\text{obs}}}\sim\Pi(Z_{1},\ldots,Z_{n}\mid\phi_{0})$ <sup>2</sup>, then

$$
\lim_{n\to\infty}\operatorname{\mathbb{E}}\left[\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi)\mid\mathcal{D}^{n}_{\mathrm{\text{obs}}}\right]\stackrel{{\scriptstyle a.s.}}{{=}}g_{t_{0}}(\boldsymbol{{\mathrm{x}}}_{0}\,;\,\phi_{0}).
$$

###### Proof.

According to Assumption˜3, a subset $\mathcal{X}_{0}\subseteq\mathcal{X}$ exists with $\Pi(\mathcal{X}_{0})=1$ where for all $t_{0}\in\mathcal{T}$ and $x_{0}\in\mathcal{X}_{0}$ we have that $\operatorname{\mathbb{E}}\left[|\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi)|\right]<\infty$. Now, we verify that using this, a similar integrability statement can be made for the new functional $g_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\phi)$ which will be used later:

$$
\displaystyle\operatorname{\mathbb{E}}\left[|g_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\phi)|\right]
$$
 
$$
\displaystyle=\operatorname{\mathbb{E}}\Big{[}\big{|}\operatorname{\mathbb{E}}\left[\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi)\mid\psi\in[\phi]^{-1}\right]\big{|}\Big{]}
$$
 
$$
\displaystyle\leq\operatorname{\mathbb{E}}\Big{[}\operatorname{\mathbb{E}}\left[|\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi)|\mid\psi\in[\phi]^{-1}\right]\Big{]}
$$
 
$$
\displaystyle=\operatorname{\mathbb{E}}[|\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi)|]<\infty.
$$

Now, we use Theorem˜2 and plug in $\mathcal{Z},\Phi,\mathcal{B}_{\mathcal{Z}}$, and $\mathcal{B}_{\Phi}$ directly from our notation. Note that $\operatorname{P}^{\psi}_{\mathrm{\text{obs}}}$ can be written as a distribution $\operatorname{P}^{\phi}$ when $\phi=[\psi]$. This allows us to work with the Doob’s corollary on the observational quotient space. Furthermore, we show that for the functionals $g_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\phi)$ on this quotient space all of the assumptions in Theorem˜2 hold:

1. Polish. Directly follows from the image $[\Psi]$ being Polish in Assumption˜2.
2. Measurability. Also follows from the measurability in Assumption˜2.
3. No Redundancy. Follows from the definition of the quotient space.
4. Integrability. Follows from ˜23, ˜24, and ˜25.

Further, we replace the marginals $\Pi\Big{(}\left(Z_{1},Z_{2},\ldots\right),\phi\Big{)}$ and $\Pi(\phi)$ in place of $\nu_{\text{tot}}$ and $\nu$ in Theorem˜2, respectively, and use it to obtain a statement for the functional $g_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\phi)$. As a result, all of the three assumptions (i), (ii), (iii), and (iv) are verified for these functionals and Theorem˜2 yields $\Pi$ -almost all $\phi_{0}$ claim for the quotient space where $\forall t_{0}\in\mathcal{T}$ and $\boldsymbol{{\mathrm{x}}}_{0}\in\mathcal{X}_{0}$ we have that:

$$
\lim_{n\to\infty}\operatorname{\mathbb{E}}\left[g_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\phi)\mid\mathcal{D}_{\mathrm{\text{obs}}}^{n}\right]\stackrel{{\scriptstyle a.s.}}{{=}}g_{t_{0}}(\boldsymbol{{\mathrm{x}}}_{0}\,;\,\phi_{0}).
$$

The final ingredient is to move from the posterior-predictive defined in ˜26 on the quotient space, to the base parameter space $\Psi$ in the CEPO-PPD:

$$
\displaystyle~\operatorname{\mathbb{E}}\left[g_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\phi)\mid\mathcal{D}_{\mathrm{\text{obs}}}^{n}\right]
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle~\operatorname{\mathbb{E}}\Big{[}\operatorname{\mathbb{E}}\left[\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi)\mid\psi\in[\phi]^{-1}\right]\mid\mathcal{D}^{n}_{\mathrm{\text{obs}}}\Big{]}\quad\text{(total expectation)}
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle~\operatorname{\mathbb{E}}\Big{[}\operatorname{\mathbb{E}}\left[\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi)\mid\mathcal{D}^{n}_{\mathrm{\text{obs}}},\psi\in[\phi]^{-1}\right]\mid\mathcal{D}^{n}_{\mathrm{\text{obs}}}\Big{]}\quad\text{Since }\left(\psi\perp\!\!\!\!\perp_{\Pi}\mathcal{D}_{\mathrm{\text{obs}}}^{n}\mid\phi\right)
$$
 
$$
\displaystyle=
$$
 
$$
\displaystyle~\operatorname{\mathbb{E}}\left[\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi)\mid\mathcal{D}^{n}_{\mathrm{\text{obs}}}\right].\quad\text{(total expectation)}
$$

When repeated for all $t_{0}\in\mathcal{T}$ and $\boldsymbol{{\mathrm{x}}}_{0}\in\mathcal{X}_{0}$, ˜30 and ˜26 prove the lemma. ∎

Lemma˜3 establishes a consistency result between the CEPO-PPDs (L.H.S. of ˜19) and the new functionals that we defined on the quotient space (R.H.S. of ˜22). With the consistency proven in the observational quotient space, all that remains to be proven is to connect these functionals to the original CEPOs. This is where identifiability comes into play.

We now use Lemma˜3 to define $\mathcal{X}_{0}$, and rather than a $\Pi$ -almost all statement on the quotient space, we explicitly denote a set $\Phi_{0}\subseteq\Phi$ on the quotient space with $\Pi(\Phi_{0})=1$ where for all $\phi_{0}\in\Phi_{0}$ ˜22 holds. We also define $\Psi_{0}\coloneqq[\Phi_{0}]^{-1}$ as the preimage of $\Phi_{0}$ under the quotient mapping. Verify from the definition of the pushforward ˜16 that when $\Pi(\Phi_{0})=1$ we have $\Pi(\Psi_{0})=1$. The measure $1$ set $\Psi_{0}$ on the parameter space is an important set we use to prove the two sides of the proposition.

In what follows, fix $t_{0}\in\mathcal{T}$ and $\boldsymbol{{\mathrm{x}}}_{0}\in\mathcal{X}_{0}$:

Identifiability $\Rightarrow$ Consistency. Under identifiability, there exists $\Psi_{1}\subseteq\Psi$ where $\Pi(\Psi_{1})=1$, and for all $\psi_{0}^{\prime},\psi_{0}^{\prime\prime}\in\Psi_{1}$ we have that $\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi_{0}^{\prime})=\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi_{0}^{\prime\prime})$ iff $[\psi_{0}^{\prime}]=[\psi_{0}^{\prime\prime}]$. Consequently, define $\Psi^{\star}\coloneqq\Psi_{1}\cap\Psi_{0}$ and verify that $\forall\psi^{\star}\in\Psi^{\star}$, the domain of the conditional expectation below is constant on $\Psi^{\star}$ (thus constant a.e.) and equal to the CEPO for any representative element $\psi^{\star}\in\Psi^{\star}$:

$$
\operatorname{\mathbb{E}}\Big{[}\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi\mid[\psi]=[\psi^{\star}]\Big{]}=\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi^{\star}).
$$

The L.H.S. of ˜31 is simply $g_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,[\psi^{\star}])$ and plugging it into the R.H.S. of ˜22 provides an almost sure convergence for all $\psi^{\star}\in\Psi^{\star}$. Since $\Pi(\Psi^{\star})=1$, this is equivalent to saying that the consistency holds for $\Pi$ -almost all $\psi^{\star}$, as stated in the proposition, and proves one side.

Consistency $\Rightarrow$ Identifiability. When consistency holds, a set $\Psi^{\star}\subseteq\Psi$ exists with $\Pi(\Psi^{\star})=1$ where ˜19 holds absolutely everywhere on $\Psi^{\star}$. Similarly, according to Lemma˜3, ˜22 holds for absolutely every $\psi_{0}\in\Psi_{0}$. Using these two identities, we can define $\Psi_{1}\coloneqq\Psi^{\star}\cap\Psi_{0}$ with $\Pi(\Psi_{1})=1$ where the following holds for every $\psi_{1}\in\Psi_{1}$:

$$
g_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,[\psi_{1}])=\mu_{t}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi_{1}).
$$

Therefore, for any such $\psi_{1}$ the CEPO can be written as a functional of the observational distribution $\operatorname{P}^{\psi_{1}}_{\mathrm{\text{obs}}}$ which is equivalently characterized by $[\psi_{1}]$. By Definition˜1, this means that $\mu_{t_{0}}(\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}}_{0}\,;\,\psi_{1})$ is identifiable for any element in a measure $1$ subset $\Psi_{1}$ of the parameters. This is the same as saying CEPOs are identifiable for $\Pi$ -almost all $\psi$ and yields the other side.

Repeating the above statement for all $t_{0}\in\mathcal{T}$ and $\boldsymbol{{\mathrm{x}}}_{0}\in\mathcal{X}_{0}$ ($\mathcal{X}_{0}$ in Lemma˜3) finishes the proof.

## Appendix C Validity of the Causal Data-Prior Loss

Here, we show that the causal data-prior loss is equivalent to the expected forward KL divergence between the CEPO-PPDs and the parameterized distribution $q_{\theta}$. For the theoretical justification, we assume a fixed observation size $n$ and define $\mathcal{D}_{\mathrm{\text{obs}}}\coloneqq\mathcal{D}_{\mathrm{\text{obs}}}^{n}$ with a dropped superscript for simplicity.

###### Definition 5.

Let $\operatorname{P}_{\mathrm{\text{obs}}}\left(Z_{1},\ldots,Z_{n}\right)\coloneqq\int_{\Psi}\operatorname{P}^{\psi}_{\mathrm{\text{obs}}}\left(Z_{1},\ldots,Z_{n}\right)\mathop{}\!\mathrm{d}\pi(\psi)$, be the marginal observational distribution under $\pi$. Then, the expected forward-KL divergence between $\pi^{\mu_{t}}$ and $q_{\theta}$, is defined as

$$
\mathcal{L}_{t}^{\mathrm{KL}}(\theta)\coloneqq\operatorname{\mathbb{E}}_{\mathcal{D}_{\mathrm{\text{obs}}}\cup\{\boldsymbol{{\mathrm{x}}}\}\ \sim\ \operatorname{P}_{\mathrm{\text{obs}}}}\left[\mathrm{D}_{\mathrm{KL}}\left(\pi^{\mu_{t}}\left(\cdot\mid\boldsymbol{{\mathrm{x}}},\mathcal{D}_{\mathrm{\text{obs}}}\right)\ \middle\|\ q_{\theta}\left(\cdot\mid\boldsymbol{{\mathrm{x}}},t,\mathcal{D}_{\mathrm{\text{obs}}}\right)\right)\right],
$$

where $\mathcal{D}_{\mathrm{\text{obs}}}\cup\{\boldsymbol{{\mathrm{x}}}\}\ \sim\ \operatorname{P}_{\mathrm{\text{obs}}}$ means to sample $\psi\sim\pi$ first and then sample $\mathcal{D}_{\mathrm{\text{obs}}}\cup\{\boldsymbol{{\mathrm{x}}}\}\ \sim\ \operatorname{P}^{\psi}_{\mathrm{\text{obs}}}$.

###### Proposition 4.

The causal data-prior loss from Definition˜3 and the expected forward-KL divergence in Definition˜5 have the same optima. In other words, for all $t\in\mathcal{T}$,

$$
\arg\min_{\theta}\mathcal{L}_{t}^{\mathrm{KL}}(\theta)=\arg\min_{\theta}\mathcal{L}_{t}(\theta).
$$

###### Proof.

First off, we note that

$$
\displaystyle{\mathcal{L}}^{\mathrm{KL}}_{t}(\theta)
$$
 
$$
\displaystyle=\operatorname{\mathbb{E}}_{\mathcal{D}_{\mathrm{\text{obs}}}\cup\{\boldsymbol{{\mathrm{x}}}\}\ \sim\ \operatorname{P}_{\mathrm{\text{obs}}}}\left[\int_{\operatorname{\mathbb{R}}}\log\frac{\pi^{\mu_{t}}(\mu\mid\boldsymbol{{\mathrm{x}}},\mathcal{D}_{\mathrm{\text{obs}}})}{q_{\theta}(\mu\mid\boldsymbol{{\mathrm{x}}},t,\mathcal{D}_{\mathrm{\text{obs}}})}\mathop{}\!\mathrm{d}\pi^{\mu_{t}}(\mu\mid\boldsymbol{{\mathrm{x}}},\mathcal{D}_{\mathrm{\text{obs}}})\right].
$$

From ˜14, we can see that the measure $\pi^{\mu_{t}}(\cdot\mid\boldsymbol{{\mathrm{x}}},\mathcal{D}_{\mathrm{\text{obs}}})$ is the pushforward of the measure $\pi(\cdot\mid\mathcal{D}_{\mathrm{\text{obs}}})$ by the function $f(\psi)=\mu_{t}(\boldsymbol{{\mathrm{x}}}\,;\,\psi)$. Hence, for any measurable function $h:\operatorname{\mathbb{R}}\to\operatorname{\mathbb{R}}$, we get

$$
\displaystyle\operatorname{\mathbb{E}}_{\mu\sim\pi^{\mu_{t}}(\cdot\mid\boldsymbol{{\mathrm{x}}},\mathcal{D}_{\mathrm{\text{obs}}})}\left[h(\mu)\right]=\operatorname{\mathbb{E}}_{\psi\sim\pi\left(\cdot\mid\mathcal{D}_{\mathrm{\text{obs}}}\right)}\left[h(f(\psi))\right]=\operatorname{\mathbb{E}}_{\psi\sim\pi\left(\cdot\mid\mathcal{D}_{\mathrm{\text{obs}}}\right)}\left[h(\mu_{t}(\boldsymbol{{\mathrm{x}}}\,;\,\psi))\right].
$$

Setting $h(\mu)=\log\frac{\pi^{\mu_{t}}(\mu\mid\boldsymbol{{\mathrm{x}}},\mathcal{D}_{\mathrm{\text{obs}}})}{q_{\theta}(\mu\mid\boldsymbol{{\mathrm{x}}},t,\mathcal{D}_{\mathrm{\text{obs}}})}$, and combining with ˜36 and 35 yields

$$
\displaystyle{\mathcal{L}}^{\mathrm{KL}}_{t}(\theta)
$$
 
$$
\displaystyle=\operatorname{\mathbb{E}}_{\mathcal{D}_{\mathrm{\text{obs}}}\cup\{\boldsymbol{{\mathrm{x}}}\}\ \sim\ \operatorname{P}_{\mathrm{\text{obs}}}}\left[\operatorname{\mathbb{E}}_{\psi\sim\pi(\cdot\mid\mathcal{D}_{\mathrm{\text{obs}}})}\left[\log\frac{\pi^{\mu_{t}}(\mu_{t}(\boldsymbol{{\mathrm{x}}}\,;\,\psi)\mid\boldsymbol{{\mathrm{x}}},\mathcal{D}_{\mathrm{\text{obs}}})}{q_{\theta}(\mu_{t}(\boldsymbol{{\mathrm{x}}}\,;\,\psi)\mid\boldsymbol{{\mathrm{x}}},t,\mathcal{D}_{\mathrm{\text{obs}}})}\right]\right]
$$
 
$$
\displaystyle=\operatorname{\mathbb{E}}_{\mathcal{D}_{\mathrm{\text{obs}}}\cup\{\boldsymbol{{\mathrm{x}}}\}\ \sim\ \operatorname{P}_{\mathrm{\text{obs}}},\;\psi\sim\pi(\cdot\mid\mathcal{D}_{\mathrm{\text{obs}}})}\left[\log\frac{\pi^{\mu_{t}}(\mu_{t}(\boldsymbol{{\mathrm{x}}}\,;\,\psi)\mid\boldsymbol{{\mathrm{x}}},\mathcal{D}_{\mathrm{\text{obs}}})}{q_{\theta}(\mu_{t}(\boldsymbol{{\mathrm{x}}}\,;\,\psi)\mid\boldsymbol{{\mathrm{x}}},t,\mathcal{D}_{\mathrm{\text{obs}}})}\right].
$$

Next, we use the Bayes’ rule to derive

$$
\underbrace{\operatorname{P}_{\mathrm{\text{obs}}}\left(\mathcal{D}_{\mathrm{\text{obs}}}\right)}_{\text{evidence}}\underbrace{\pi(\psi\mid\mathcal{D}_{\mathrm{\text{obs}}})}_{\text{posterior}}=\underbrace{\pi(\psi)}_{\text{prior}}\underbrace{\operatorname{P}^{\psi}\left(\mathcal{D}_{\mathrm{\text{obs}}}\right)}_{\text{likelihood}}.
$$

Combining ˜38 and ˜39, we get

$$
\displaystyle{\mathcal{L}}^{\mathrm{KL}}_{t}(\theta)
$$
 
$$
\displaystyle=\operatorname{\mathbb{E}}_{\psi\sim\pi,\ \mathcal{D}_{\mathrm{\text{obs}}}\cup\{\boldsymbol{{\mathrm{x}}}\}\ \sim\ \operatorname{P}^{\psi}_{\mathrm{\text{obs}}}}\left[\log\frac{\pi^{\mu_{t}}(\mu_{t}(\boldsymbol{{\mathrm{x}}}\,;\,\psi)\mid\boldsymbol{{\mathrm{x}}},\mathcal{D}_{\mathrm{\text{obs}}})}{q_{\theta}(\mu_{t}(\boldsymbol{{\mathrm{x}}}\,;\,\psi)\mid\boldsymbol{{\mathrm{x}}},t,\mathcal{D}_{\mathrm{\text{obs}}})}\right]
$$
 
$$
\displaystyle=\operatorname{\mathbb{E}}_{\psi\sim\pi,\ \mathcal{D}_{\mathrm{\text{obs}}}\cup\{\boldsymbol{{\mathrm{x}}}\}\ \sim\ \operatorname{P}^{\psi}_{\mathrm{\text{obs}}}}\left[-\log{q_{\theta}(\mu_{t}(\boldsymbol{{\mathrm{x}}}\,;\,\psi)\mid\boldsymbol{{\mathrm{x}}},t,\mathcal{D}_{\mathrm{\text{obs}}})}\right]+\text{constant term in }\theta
$$
 
$$
\displaystyle={\mathcal{L}}_{t}(\theta)+\text{constant term in }\theta.
$$

When repeated for all $t\in\mathcal{T}$, (42) shows the equivalence and finishes the proof. ∎

Remark 1. In general, the forward-KL divergence loss is not computable, as it requires having access to the true CEPO-PPD. However, with this identity established, we can justify the use the equivalent causal data-prior loss which is readily computable. ˜34 lets us interpret this optimization as minimizing the forward–KL divergence. Driving down $\mathcal{L}_{t}(\theta)$ therefore also decreases $\mathcal{L}^{\mathrm{KL}}_{t}(\theta)$, which reaches its infimum $0$ exactly when, for almost all $\mathcal{D}_{\mathrm{\text{obs}}}\cup\{\boldsymbol{{\mathrm{x}}}\}\sim\operatorname{P}^{\pi}_{\mathrm{\text{obs}}}$, we have that $q_{\theta}(\,\cdot\mid\boldsymbol{{\mathrm{x}}},t,\mathcal{D}_{\mathrm{\text{obs}}})=\pi^{\mu_{t}}(\,\cdot\mid\boldsymbol{{\mathrm{x}}},\mathcal{D}_{\mathrm{\text{obs}}})$ -almost everywhere. While this equality is rarely attainable, the equivalence guarantees that minimizing the causal data-prior loss pushes $q_{\theta}$ toward the closest CEPO-PPD representable by the chosen parameterization $\theta$.

Remark 2. The theoretical equivalence is proved only for a fixed treatment $t\in\mathcal{T}$ and a fixed, finite sample size $n\in\{1,2,\ldots\}$. In practice, the training loss is minimized while *randomizing* both $t$ and $n$. If the optimizer attains a global minimum of this randomized objective, the same argument yields

$$
q_{\theta}\bigl{(}\cdot\mid\boldsymbol{{\mathrm{x}}},t,\mathcal{D}_{\mathrm{\text{obs}}}\bigr{)}=\pi^{\mu_{t}}\bigl{(}\cdot\mid\boldsymbol{{\mathrm{x}}},\mathcal{D}_{\mathrm{\text{obs}}}\bigr{)},\text{ for every }t\in\mathcal{T},\text{ and for all but finitely many }n=|\mathcal{D}_{\mathrm{\text{obs}}}|.
$$

Hence the approximation $q_{\theta}\bigl{(}\cdot\mid\boldsymbol{{\mathrm{x}}},t,\mathcal{D}_{\mathrm{\text{obs}}}\bigr{)}\approx\pi^{\mu_{t}}\bigl{(}\cdot\mid\boldsymbol{{\mathrm{x}}},\mathcal{D}_{\mathrm{\text{obs}}}\bigr{)}$ can effectively extend to all the treatment values and to almost every sample size we care about in practice.

## Appendix D Experimental Details

### D.1 Prior Generation & Simulating DGPs

As illustrated in Figure˜4, our prior generation consists of retrieving or synthesizing a base table, subsampling covariates $\boldsymbol{{\mathrm{X}}}$ and CEPOs $\mu_{0}$ and $\mu_{1}$, synthesizing treatments $T$, potential outcomes $Y_{t}$, and finally, observed outcomes $Y$. We break down each of the components:

Data Sources for the Base Tables. We draw the base tables from two sources: (i) real-world tables from OpenML, and (ii) fully synthetic data.

- We use the OpenML collections used in [^35], AMLB [^34], and TabZilla [^76], all listed in [^68]. To widen coverage, we also add tables from CTR23 [^29] and CC18 [^13]. All OpenML IDs are in [this link](https://drive.google.com/file/d/1NXib83Lc7jGOPJx554p-I3sxFrcWeF52).<sup>3</sup> Data leakage is ruled out as none of the tables that share covariates or outcomes with our test sets (Lalonde, IHDP, ACIC, Criteo, Megafon, Hillstrom, Lenta, X5) are included in training. Moreover, the propensities are sampled purely synthetically, following the approach described below.
- For additional diversity, we generate synthetic tables using the random neural networks used to train TabPFN v1, with the same hyperparameters described in [^43]. Inputs, from a standard Gaussian distribution, are fed into the network, and a subset of the outputs and hidden neurons are selected to construct the tabular data. Some columns are discretized at random to produce categorical and ordinal variables to reflect the structure of real-world tabular domains. While TabPFN v2 [^44] is a newer and stronger model, its training data is not publicly available, so we restrict ourselves to the v1 generator to ensure transparent evaluation and leakage control.

CEPOs with Heterogeneity Control. Once the base table is given, we randomly select two columns and name them $\mu_{\text{raw},0}$ and $\mu_{\text{raw},1}$. However, in practice, we observe that directly using such columns for CEPOs can result in large variances (*heterogeneity*) for CATEs. We therefore apply a light-weight post-processing inspired by RealCause [^80].

The post-processing requires a heterogeneity hyperparameter $\gamma$, which we sample uniformly from $[0,1]$ during prior generation. Then, for $N$ units (rows) extracted from the base table, let ${\tau}_{\text{raw}}^{(n)}=\mu^{(n)}_{\text{raw},1}-\mu^{(n)}_{\text{raw},0}$ be the CATE for unit $n\in[N]$, and ${\lambda}_{\text{raw}}=\tfrac{1}{N}\sum_{n=1}^{N}{\tau}_{\text{raw}}^{(n)}$ the sample ATE. We draw i.i.d. $\{\alpha^{(n)}\}_{n=1}^{N}\sim\mathrm{Unif}[0,1]$ and construct the final $\gamma$ -augmented CEPOs as

$$
\displaystyle\mu_{1}^{(n)}
$$
 
$$
\displaystyle\coloneqq\bigl{[}\alpha^{(n)}+(1-\alpha^{(n)})\gamma\bigr{]}{\mu}_{\text{raw},1}^{(n)}+(1-\gamma)(1-\alpha^{(n)})({\mu}_{\text{raw},0}^{(n)}+{\lambda_{\text{raw}}}),
$$
$$
\displaystyle\mu_{0}^{(n)}
$$
 
$$
\displaystyle\coloneqq\bigl{[}(1-\alpha^{(n)})+\alpha^{(n)}\gamma\bigr{]}{\mu}_{\text{raw},0}^{(n)}+(1-\gamma)\alpha^{(n)}({\mu}_{\text{raw},1}^{(n)}-{\lambda_{\text{raw}}}).
$$

A simple algebraic check shows

$$
\tau^{(n)}\;\coloneqq\;\mu_{1}^{(n)}-\mu_{0}^{(n)}\;=\;\gamma\,\tau_{\text{raw}}^{(n)}+(1-\gamma){\lambda_{\text{raw}}},\quad\operatorname{\mathrm{Var}}[\tau\mid\boldsymbol{{\mathrm{x}}}]=\gamma^{2}\operatorname{\mathrm{Var}}[{\tau}_{\text{raw}}\mid\boldsymbol{{\mathrm{x}}}].
$$

Hence, while preserving the average treatment effect, $\gamma=0$ yields a dataset with a zero variance CATE (fully homogeneous), whereas $\gamma=1$ recovers the original heterogeneity.

Outcomes. After constructing the CEPO columns $\mu_{0}(\boldsymbol{{\mathrm{x}}})$ and $\mu_{1}(\boldsymbol{{\mathrm{x}}})$, we need to turn them into potential outcomes by adding zero-mean noises. To avoid tying the data to a specific parametric noise model, we introduce two additional *nuisance* columns, $\eta_{0}(\boldsymbol{{\mathrm{x}}})$ and $\eta_{1}(\boldsymbol{{\mathrm{x}}})$, sampled from the base table. Let $\epsilon_{t}$ be random scalars, independent from $\boldsymbol{{\mathrm{x}}}$, with $\operatorname{\mathbb{E}}[\epsilon_{t}]=0$. We define the potential outcomes as

$$
Y_{t}\;=\;\mu_{t}(\boldsymbol{{\mathrm{x}}})\;+\;\eta_{t}(\boldsymbol{{\mathrm{x}}})\,\epsilon_{t},\quad t\in\mathcal{T}.
$$

This construction preserves the conditional means, that is $\operatorname{\mathbb{E}}[Y_{t}\mid\boldsymbol{{\mathrm{x}}}]=\mu_{t}(\boldsymbol{{\mathrm{x}}})$. The input-dependent scale factors $\eta_{t}(\boldsymbol{{\mathrm{x}}})$ allow for heteroscedastic noises and capture a richer family of outcome distributions than additive parametric noise models. For our training, we sample $\epsilon_{t}$ from a Gaussian with a variance uniformly drawn from $(0,\operatorname{\mathrm{Var}}(\mu_{t})]$. This choice of noise values ensures a similar noise scale to the scale of CEPOs, resulting in training data with a more informative signal-to-noise ratio.

Propensities with Positivity Control. Given a covariate vector $\boldsymbol{{\mathrm{x}}}$, the strong ignorability assumption requires the propensity values $0<\operatorname{P}(T=1\mid\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}})<1$. Hence, due to the invertibility of the sigmoid function, it is sufficient to generate treatment logits, through any function $f:\boldsymbol{{\mathrm{X}}}\to\operatorname{\mathbb{R}}$, and then apply a sigmoid function to get values within $(0,1)$. To simulate different degrees of confounding, we choose $f$ by randomly selecting one of the following mechanisms:

- Randomized treatments (RCT). Treatments are independent of covariates, i.e., $f$ is constant. We sample $c\sim\texttt{Logistic}(0,1)$ and set ${f}(\boldsymbol{{\mathrm{x}}})=c$ to get uniform propensities.
- Linear logits. Draw the random vector $\boldsymbol{{\mathrm{w}}}$ from a standard Gaussian and set ${f}(\boldsymbol{{\mathrm{x}}})=\boldsymbol{{\mathrm{w}}}^{\top}\boldsymbol{{\mathrm{x}}}$.
- Non-linear logits. Feed $\boldsymbol{{\mathrm{x}}}$ into a randomly initialized MLP, similar architecture to that of [^43], to get $f(\boldsymbol{{\mathrm{x}}})$.

Empirically, we observe that the above procedure yields an artificially high level of positivity, which is not reflective of real-world scenarios. We therefore apply a light-weight post-processing transform, inspired by RealCause [^80], to better control the positivity level. Concretely, we sample a parameter $\xi\in[0,1]$ and *exacerbate* extreme propensity scores to mimic poor positivity:

$$
\operatorname{P}(T=1\mid\boldsymbol{{\mathrm{X}}}=\boldsymbol{{\mathrm{x}}})\;\coloneqq\;\xi\,\mathrm{Sigmoid}\left(f(\boldsymbol{{\mathrm{x}}})\right)\;+\;(1-\xi)\,\mathbb{I}\bigl{[}{f}(\boldsymbol{{\mathrm{x}}})>0\bigr{]}.
$$

Here, $\xi=1$ leaves the original positivity intact. However, for smaller $\xi$ values, the support of the treated and control groups become increasingly disjoint, leading to low-positivity scenarios.

Treatment Assignment. Finally, each unit’s treatment is drawn as $T\sim\mathrm{Bernoulli}\left(\mathrm{Sigmoid}\left(f(\boldsymbol{{\mathrm{X}}})\right)\right)$, and the observed outcome $Y$ is also derived by selecting the assigned potential outcome $Y\coloneqq Y_{T}$.

Collectively, all of the steps above simulate different DGPs, with various levels of positivity and heterogeneity, extracted from real and synthetic sources of tabular data. This procedure creates a broad prior $\pi$ for CausalPFN, which is necessary for the model to work well in practice.

### D.2 Model Details

Architecture & Training. We represent each context row $(t,\boldsymbol{{\mathrm{x}}},y)$ and query row $(t,\boldsymbol{{\mathrm{x}}})$ as single tokens by summing up (1) a treatment embedding for $t$, (2) a covariate embedding for $\boldsymbol{{\mathrm{x}}}$ (padded to length $F=100$), and (3) an outcome embedding for $y$ (only for context rows). We use linear layers for embeddings and omit the positional encodings to preserve the permutation invariance of the context set, similar to other PFN-style transformers.

All tokens—context and query—are passed into a 20-layer transformer, with a hidden size of 384, QK-normalization (RMS) <sup>4</sup>, and a parallel SwiGLU-activated [^101] feed-forward block.

The transformer’s query outputs are then projected to a 1024-dimensional logit vector, then softmaxed at a fixed temperature of $\theta_{T}=1.0$ to form a discrete CEPO posterior over the interval $[-10,10]$. We then scale the interval to match the scale of the outcomes and clip the out-of-range values. At inference time, we return the posterior mean as the point estimate and sample 10,000 times to estimate credible intervals at any desired significance level $\alpha$.

The full model has approximately 20M parameters and is trained in two stages: (i) a predictive phase that mimics standard predictive PFN training from [^68], and (ii) a causal phase that optimizes the CEPO loss. We use AdamW [^54] with warmup and cosine annealing for the predictive phase, and switch to the schedule-free optimizer [^25] in the causal phase. The model is trained with a maximum context length of 16K in the first phase and 2,048 in the second. We use four A100 GPUs trained for at most one week for the initial phase, and two days on an H100 for the second phase.

Finally, to enhance parallel training, we batch both the queries and the tables. That is, rather than sampling only one DGP and one query token, each gradient update samples $B_{t}$ DGPs, draws $B_{q}$ queries per DGP, and concatenates everything into a single tensor. The tensor is then passed through the transformer to get $B_{t}B_{q}$ CEPO-PPDs. The final loss is averaged over all the batches. See Algorithm˜1 for a detailed demonstration of CausalPFN’s training algorithm.

Algorithm 1 Parallel training of CausalPFN.

Prior $\pi$, DGPs and CEPO values $\operatorname{P}^{\psi}_{\mathrm{\text{obs}}}$, $\mu_{t}(\cdot\!\,;\,\psi)$, model $q_{\theta}$, DGP batch size $B_{t}$, query batch size $B_{q}$, fixed feature length $F$, and histogram loss HL ˜10.

while not converged do

  Sample $\psi[1],\ldots,\psi[B_{t}]\sim\pi$

  Sample $\mathcal{D}_{\mathrm{\text{obs}}}[i]\sim\operatorname{P}^{\psi[i]}_{\mathrm{\text{obs}}},\forall 1\leq i\leq B_{t}$

  Randomly sample query treatments $t^{(i,j)}$ for ${1\leq i\leq B_{t},1\leq j\leq B_{q}}$

  Sample query covariates $\boldsymbol{{\mathrm{x}}}^{(i,j)}\sim\operatorname{P}^{\psi}_{\mathrm{\text{obs}}}[i]$ for ${1\leq i\leq B_{t},1\leq j\leq B_{q}}$

  Set ${\mu}^{(i,j)}\leftarrow\mu_{t^{(i,j)}}\left(\boldsymbol{{\mathrm{x}}}^{(i,j)}\,;\,\psi[i]\right)$

  Pad $\boldsymbol{{\mathrm{x}}}^{(i,j)}$ with zeros such that $\boldsymbol{{\mathrm{x}}}^{(i,j)}\in\mathbb{R}^{F}$

   $\widehat{\mathcal{L}}\leftarrow\frac{1}{B_{t}\cdot B_{q}}\sum_{i,j}\texttt{HL}\left[{\mu}^{(i,j)}\|q_{\theta}(\cdot\mid\boldsymbol{{\mathrm{x}}}^{(i,j)},t^{(i,j)},\mathcal{D}_{\mathrm{\text{obs}}}[i])\right]$

  Update $\theta$ using the gradients $\nabla_{\theta}\widehat{\mathcal{L}}$

end while

Handling Large Tables at Inference Time. CausalPFN’s default maximum context length is set to 4,096 at inference, but real-world tables may contain millions of rows. Training PFN-style transformers on such long contexts can be challenging due to hardware or architectural constraints. While some tabular foundation models such as TabICL [^90] modify the architecture itself, [^104] show that, retrieving a small relevant subset of rows for each query at inference time allows a model with a short context length to better generalize to longer contexts.

We adopt this retrieval philosophy in CausalPFN to enable causal effect estimation on large tables. First, we fit a lightweight gradient boosting regressor on the context data to produce weak CATE estimates for each covariate. This regressor estimates CATE by regressing outcomes on the treatment and covariates and then taking the difference in predicted outcomes between $T=1$ and $T=0$. This step is applied *only* when the table is too large to fit within the model’s maximum context window. We then (i) sort both the context rows and the queries based on their weak CATE estimates, which effectively stratifies the data; (ii) partition the ordered queries into consecutive mini-batches; and (iii) for each query batch, use a fast bisection search to select a contiguous window of context rows whose weak CATE estimate range most closely matches that of the batch. As a result, each batch is exposed only to a neighborhood of rows with similar causal effects, allowing all CEPO predictions to be computed with short forward passes.

### D.3 Baseline Hyperparameters and Results without Hyperparameter Tuning

No Hyperparameter Tuning. Table˜4 summarizes the performance of all methods without hyperparameter tuning. Indeed, in this setting, CausalPFN consistently attains the first or second best results on heterogeneous treatment effect, and overall minimum average relative error on ATE.

EconML Hyperparameters. For the results without hyper parameter tuning in Table˜4, we ran the models with the recommended hyper parameters in the Jupyter notebooks from EconML [^12]. For the tuned results in Table˜1, we performed a grid search on both the propensity and outcome models with the following search space:

- Model family $\in$ $\{$ Random forest (with $100$ trees), Gradient boosting $\}$
- Maximum depth $\in$ $\{3,\,5\}$
- Minimum samples per leaf $\in$ $\{10,\,50\}$

Each candidate configuration was evaluated using three-fold cross-validation. For DR-Learner and DML, we additionally expanded the covariates with quadratic terms (polynomial degree 2).

CATE Nets. For the results without hyper parameter tuning in Table˜4, we ran the models with the default hyperparameters and a batch size of $512$. For the tuned results in Table˜1, we perform a grid search on the hyperparameters for the neural architecture:

- Number of layers $\in$ $\{2,3\}$
- Representation dimension $\in$ $\{128,256\}$
- Number of hidden output layers $\in$ $\{1,2\}$
- Width of the hidden output layers $\in$ $\{128,256\}$

the rest of the hyperaparameters are left unchanged.

BART & GRF. The GRF implementation includes an internal tune option. We enable this option for the tuned experiment reported in Table˜1 and disable it for the untuned experiment in Table˜4. BART, on the other hand, offers no comparable hyperparameter-tuning routine. Its only alternative, a full cross-fit, is prohibitively slow uses a rudimentary Bayesian routine. Thus, the BART scores appear unchanged in both Table˜1 and Table˜4.

Table 4: CATE & ATE results. PEHE (left half) alongside ATE relative error and its overall average (right half). PEHE for Lalonde CPS/PSID is shown in thousands. Best numbers are in blue; second best are in purple.

Method PEHE $\pm$ Standard Error ($\downarrow$ better) ATE Relative Error $\pm$ Standard Error ($\downarrow$ better) IHDP ACIC 2016 Lalonde CPS Lalonde PSID IHDP ACIC 2016 Lalonde CPS Lalonde PSID Avg. ($\times 10^{3}$) ($\times 10^{3}$) CausalPFN 0.58 $\pm$ 0.07 0.92 $\pm$ 0.11 8.83 $\pm$ 0.04 13.98 $\pm$ 0.43 0.20 $\pm$ 0.04 0.04 $\pm$ 0.01 0.07 $\pm$ 0.02 0.20 $\pm$ 0.04 0.18 $\pm$ 0.03 T Learner 2.28 $\pm$ 0.34 1.41 $\pm$ 0.11 9.24 $\pm$ 0.05 14.01 $\pm$ 0.41 0.21 $\pm$ 0.04 0.05 $\pm$ 0.01 0.37 $\pm$ 0.02 0.09 $\pm$ 0.03 0.20 $\pm$ 0.03 DragonNet 2.13 $\pm$ 0.24 2.23 $\pm$ 0.20 10.3 $\pm$ 0.39 16.2 $\pm$ 0.78 0.19 $\pm$ 0.04 0.09 $\pm$ 0.02 0.55 $\pm$ 0.10 0.40 $\pm$ 0.07 0.23 $\pm$ 0.03 GRF 4.26 $\pm$ 0.69 1.36 $\pm$ 0.30 12.1 $\pm$ 0.22 21.2 $\pm$ 0.48 0.18 $\pm$ 0.03 0.07 $\pm$ 0.02 0.81 $\pm$ 0.06 0.80 $\pm$ 0.05 0.27 $\pm$ 0.03 DR Learner 3.82 $\pm$ 0.49 1.09 $\pm$ 0.09 14.34 $\pm$ 0.72 34.04 $\pm$ 3.64 0.19 $\pm$ 0.03 0.04 $\pm$ 0.01 0.91 $\pm$ 0.10 46.8 $\pm$ 42.9 0.28 $\pm$ 0.03 RA Net 2.08 $\pm$ 0.19 2.08 $\pm$ 0.19 12.5 $\pm$ 0.37 19.9 $\pm$ 1.71 0.20 $\pm$ 0.03 0.07 $\pm$ 0.03 0.93 $\pm$ 0.05 0.67 $\pm$ 0.06 0.28 $\pm$ 0.03 TarNet 1.88 $\pm$ 0.15 2.26 $\pm$ 0.20 11.9 $\pm$ 0.13 18.4 $\pm$ 0.45 0.20 $\pm$ 0.04 0.06 $\pm$ 0.02 0.95 $\pm$ 0.02 0.68 $\pm$ 0.03 0.29 $\pm$ 0.03 S Learner 3.06 $\pm$ 0.52 1.36 $\pm$ 0.12 12.86 $\pm$ 0.04 21.81 $\pm$ 0.44 0.23 $\pm$ 0.05 0.05 $\pm$ 0.01 1.01 $\pm$ 0.01 0.94 $\pm$ 0.01 0.33 $\pm$ 0.04 DML 3.73 $\pm$ 0.61 1.20 $\pm$ 0.24 133.9 $\pm$ 13.8 27.51 $\pm$ 1.83 0.11 $\pm$ 0.02 0.05 $\pm$ 0.01 3.91 $\pm$ 0.63 0.96 $\pm$ 0.09 0.50 $\pm$ 0.10 BART 2.50 $\pm$ 0.39 0.68 $\pm$ 0.11 12.7 $\pm$ 0.11 20.8 $\pm$ 0.45 0.50 $\pm$ 0.11 0.04 $\pm$ 0.01 1.01 $\pm$ 0.02 0.83 $\pm$ 0.03 0.53 $\pm$ 0.08 IPW 5.70 $\pm$ 0.89 3.21 $\pm$ 0.62 10.94 $\pm$ 0.06 18.20 $\pm$ 0.45 0.23 $\pm$ 0.04 0.24 $\pm$ 0.05 0.25 $\pm$ 0.03 0.05 $\pm$ 0.01 0.81 $\pm$ 0.03 X Learner 3.00 $\pm$ 0.47 1.02 $\pm$ 0.16 13.01 $\pm$ 0.10 20.27 $\pm$ 0.69 0.19 $\pm$ 0.03 0.03 $\pm$ 0.01 1.06 $\pm$ 0.02 0.74 $\pm$ 0.06 0.92 $\pm$ 0.03

### D.4 Marketing Experiments

Datasets. Apart from Hill <sup>(1)</sup> and Hill <sup>(2)</sup>, which were explained in the main text. We also run experiments on the following datasets:

1. Criteo. 25M ad-exposure records from Criteo’s online *incrementality tests*: a randomly selected *held-out* audience is shielded from seeing an advert, while the treated audience is shown the ad; the target is a post-impression conversion flag. We use a readily provided 2.5M stratified subset of this dataset from sklift.
2. Retail-Hero (X5). Transaction logs from the X5 Retail Group hackathon. Customers are randomly offered personalized coupons (treatment); the outcome records whether the customer subsequently purchased the promoted items.
3. Lenta. SMS-based promotion experiment run by the Russian grocery chain Lenta. The treatment group receives a marketing text message, and the outcome is an in-store visit after the campaign window.
4. Megafon (Mega). Synthetic yet domain-faithful data released for the MegaFon Uplift competition. Users are randomly offered a telecom upsell offer (treatment), and the outcome indicates whether they accepted the offer.

Qini Evaluation. To build Qini curves we follow scikit-uplift’s recommended five-fold *stratified* split based on the outcome and the treatment [^73]. In each fold, we hold out 20% of the data as test rows and train the baseline models on the remaining 80%. For CausalPFN we use that same 80% as context tokens and treat the held-out 20% as queries. We then rank the rows based on their CATE estimates to compute the Qini curves and the corresponding Qini scores.

Context Length Challenges. In all the marketing experiments, we have increased the model’s maximum context length from the default 4,096 to 50,000 tokens. This context length is sufficient for the subsampled datasets in Table˜2. However, extending beyond 50K for the *full-table* runs is not feasible in GPU memory. We thus use the retrieval approach explained in Section˜D.2 to achieve CATE estimates for this setting. Table˜5 shows CausalPFN’s performance (with the retrieval approach) compared to the baselines on the full-table datasets. We conjecture that the relative under-performance compared to Table˜2 is due to this retrieval heuristic.

Table 5: Normalized Qini scores ($\uparrow$ better). Scores are normalized per dataset such that the top-performing model achieves 1.0 (highlighted in bold). All datasets use full stratified subsamples: Hill <sup>(1)</sup> and Hill <sup>(2)</sup> (64K rows), Criteo (2.5M rows), X5 (200K rows), Lenta (687K rows), and Mega (600K rows).

| Method | Hill <sup>(1)</sup> | Hill <sup>(2)</sup> | Criteo | X5 | Lenta | Mega | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S Learner | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.913 | 0.985 |
| X Learner | 0.975 | 0.980 | 0.994 | 0.965 | 0.868 | 0.997 | 0.963 |
| DA Learner | 0.985 | 0.964 | 0.955 | 0.969 | 0.903 | 1.000 | 0.963 |
| T Learner | 0.991 | 0.972 | 0.902 | 0.953 | 0.833 | 0.987 | 0.940 |
| CausalPFN | 0.992 | 0.968 | 0.939 | 0.746 | 0.947 | 0.954 | 0.924 |

### D.5 Calibration, Coverage, and Credible Intervals

The Synthetic DGPs. For the calibration results in Figure˜7, we use two families of synthetic DGPs, polynomials and sinusoidals. As a general recipe, each DGP defines a treatment logit function $f(\boldsymbol{{\mathrm{x}}})\in\mathbb{R}$ and assigns treatments by sampling from the $\mathrm{Bernoulli}\left(\mathrm{Sigmoid}(f(\boldsymbol{{\mathrm{x}}}))\right)$. Moreover, each DGP specifies two CEPO functions $\mu_{0},\mu_{1}:\boldsymbol{{\mathrm{x}}}\to\operatorname{\mathbb{R}}$. It then samples the potential outcomes by $Y_{t}=\mu_{t}\left(\boldsymbol{{\mathrm{x}}}\right)+\epsilon_{t}$ for $t\in\{0,1\}$, where the noise terms $\epsilon_{t}\sim\mathrm{Normal}(0,1)$, $\mathrm{Laplace}(0,1)$, or $\mathrm{Uniform}(-1,1)$ with equal probability. We now describe each DGP family in more detail:

1. Polynomial. We first draw the number of features $d\sim\mathrm{Unif}\{10,\dots,20\}$ and sample covariate vectors $\boldsymbol{{\mathrm{x}}}\sim\mathrm{Unif}[-2,2]^{d}$. We then fix a maximum degree $\deg\in\{1,2,3,4\}$, augment covariates with powers $\boldsymbol{{\mathrm{x}}}_{\text{ext}}=(x_{1},\dots,x_{d},x_{1}^{2},\dots,x_{d}^{\deg})$, sample weights $\boldsymbol{{\mathrm{w}}}_{\mu_{0}},\boldsymbol{{\mathrm{w}}}_{\mu_{1}},\boldsymbol{{\mathrm{w}}}_{T}\sim\mathrm{Unif}[-5,5]^{d\times\deg+1}$, and define
	$$
	f(\boldsymbol{{\mathrm{x}}})=\boldsymbol{{\mathrm{w}}}_{T}^{\top}\boldsymbol{{\mathrm{x}}}_{\text{ext}},\quad\mu_{t}(\boldsymbol{{\mathrm{x}}})=\boldsymbol{{\mathrm{w}}}_{\mu_{t}}^{\top}\boldsymbol{{\mathrm{x}}}_{\text{ext}}\;\text{ for }\;t\in\{0,1\}.
	$$
	Degrees 1, 2, 3, and 4 give the Linear, Quadratic, Cubic, and Quartic sub-families; each degree adds new terms and is therefore a super-set of all lower degrees. We train on one degree family and test on the others to probe generalization.
2. Sinusoidal. We draw the number of features $d\sim\mathrm{Unif}\{5,\dots,10\}$ and sample covariate vectors $\boldsymbol{{\mathrm{x}}}\sim\mathrm{Unif}[-3,5]^{d}$. We then sample weight vectors $\boldsymbol{{\mathrm{w}}}_{\mu_{0}},\boldsymbol{{\mathrm{w}}}_{\mu_{1}},\boldsymbol{{\mathrm{w}}}_{T}\sim\mathrm{Unif}[-10,6]^{d}$, and a frequency $\omega\in\mathbb{R}^{+}$. We define the treatment logit function and the CEPOs as
	$$
	f(\boldsymbol{{\mathrm{x}}})\;=\;\sin\bigl{(}\omega\,\left\{\boldsymbol{{\mathrm{w}}}_{T}^{\top}\boldsymbol{{\mathrm{x}}}\right\}\bigr{)}+\boldsymbol{{\mathrm{w}}}_{T}^{\top}\boldsymbol{{\mathrm{x}}},\quad\mu_{t}(\boldsymbol{{\mathrm{x}}})\;=\;\sin\bigl{(}\omega\,\left\{\boldsymbol{{\mathrm{w}}}_{\mu_{t}}^{\top}\boldsymbol{{\mathrm{x}}}\right\}\bigr{)}+\boldsymbol{{\mathrm{w}}}_{\mu_{t}}^{\top}\boldsymbol{{\mathrm{x}}}\;\text{ for }\;t\in\{0,1\}.
	$$
	For training DGPs, we create three sub-families: Linear ($\omega=0$), L1 ($\omega\!\in\![0,1]$) and L2 ($\omega\!\in\!(1,2]$). For test-time DPGs, we use the following: Linear ($\omega=0$), L1 ($\omega\!\in\![0.5,1]$), L2 ($\omega\!\in\!(1.5,2]$), and L3 ($\omega\!\in\!(2.5,3]$). This allows us to measure extrapolation to unseen frequencies. For example, an L2-trained model has seen DGPs from L1 and L2, but not L3.

Synthetic Experiments on Sinusoidal. Figure˜8 shows both the regression curve $\widehat{cov}_{\mu}$ (orange) and the CATE curve $\widehat{cov}_{\tau}$. The model is overly confident in OOD scenarios (e.g., L2 tested on an L1 trained model) and either well-calibrated or conservative otherwise. The figure also shows that the regression curve is always below the blue CATE curve. Once calibration is done on the regression curve, as shown in Figure˜9, the $\mathrm{ICE}_{\mu}$ becomes smaller, resulting in a well-calibrated or conservative model, even on OOD scenarios.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.07918/assets/x8.png)

Figure 8: CATE and regression calibration curves for the models trained on synthetic sinusoidal datasets, before calibration.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.07918/assets/x9.png)

Figure 9: CATE and regression calibration curves for the models trained on synthetic sinusoidal datasets, after calibration.

Synthetic Experiments on Polynomial. Similar to the sinusoidal setting, the uncalibrated curves in Figure˜10 show that the model becomes overly confidence when tested on OOD data (e.g., testing a model trained on Quadatic data on Cubic DGP). However, applying the regression calibration results in near-perfect CATE calibration, as shown in Figure˜11.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.07918/assets/x10.png)

Figure 10: CATE and regression calibration curves for the models trained on synthetic polynomial datasets, before calibration.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.07918/assets/x11.png)

Figure 11: CATE and regression calibration curves for the models trained on synthetic polynomial datasets, after calibration.

Calibration of the Large-scale CausalPFN. We evaluate the calibration curves of the large-scale pre-trained CausalPFN on both synthetic and real-world benchmarks in Figure˜12, Figure˜13, and Figure˜14. The model generally appears conservative. This may be attributed to the Gaussian smoothing used in the histogram loss; yet, this smoothing is necessary to achieve stability in training. Regardless, across all datasets, post-hoc regression calibration improves reliability: the calibrated (pink) curves adhere far more closely to the diagonal than their uncalibrated (blue) counterparts. In Figures˜12 and 13 the improvement is almost perfect, while in Figure˜14 it corrects the base model’s strong conservatism on IHDP and ACIC 2016 and achieves near-ideal alignment on the Lalonde CPS/PSID datasets.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.07918/assets/x12.png)

Figure 12: CausalPFN’s CATE calibration curves on synthetic sinusoidal datasets, before and after calibration.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.07918/assets/x13.png)

Figure 13: CausalPFN’s CATE calibration curves on synthetic polynomial datasets, before and after calibration.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.07918/assets/x14.png)

Figure 14: CausalPFN’s CATE calibration curves on the semi-synthetic benchmarks, before and after calibration.

[^1]: Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. In *The Eleventh International Conference on Learning Representations*, 2023.

[^2]: Ahmed Alaa and Mihaela Van Der Schaar. Validating causal inference models via influence functions. In *International Conference on Machine Learning*, pages 191–201. PMLR, 2019.

[^3]: Ahmed M. Alaa and Mihaela van der Schaar. Bayesian inference of individualized treatment effects using multi-task gaussian processes. In *Advances in Neural Information Processing Systems*, volume 30, 2017.

[^4]: Christophe Andrieu, Nando De Freitas, Arnaud Doucet, and Michael I Jordan. An introduction to MCMC for machine learning. *Machine Learning*, 50:5–43, 2003.

[^5]: Joshua Angrist and Guido Imbens. Identification and estimation of local average treatment effects, 1995.

[^6]: Joshua D Angrist and Jörn-Steffen Pischke. *Mastering ’Metrics: The path from cause to effect*. Princeton University Press, 2014.

[^7]: Joshua D Angrist, Guido W Imbens, and Donald B Rubin. Identification of causal effects using instrumental variables. *Journal of the American statistical Association*, 91(434):444–455, 1996.

[^8]: Susan Athey, Julie Tibshirani, and Stefan Wager. Generalized random forests. *The Annals of Statistics*, 47(2):1148–1178, 2019. doi: 10.1214/18-AOS1709.

[^9]: Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians: Provable in-context learning with in-context algorithm selection. In *Advances in Neural Information Processing Systems*, volume 36, pages 57125–57211, 2023.

[^10]: Vahid Balazadeh, Keertana Chidambaram, Viet Nguyen, Rahul G Krishnan, and Vasilis Syrgkanis. Sequential decision making with expert demonstrations under unobserved heterogeneity. *Advances in Neural Information Processing Systems*, 37:65476–65498, 2024.

[^11]: Alexander Balke and Judea Pearl. Bounds on treatment effects from studies with imperfect compliance. *Journal of the American statistical Association*, 92(439):1171–1176, 1997.

[^12]: Keith Battocchi, Eleanor Dillon, Maggie Hei, Greg Lewis, Paul Oka, Miruna Oprescu, and Vasilis Syrgkanis. EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation, 2019. URL [https://github.com/py-why/EconML](https://github.com/py-why/EconML). Version 0.15.0.

[^13]: Bernd Bischl, Giuseppe Casalicchio, Matthias Feurer, Pieter Gijsbers, Frank Hutter, Michel Lang, Rafael Gomes Mantovani, Jan van Rijn, and Joaquin Vanschoren. OpenML benchmarking suites. In *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*, 2021.

[^14]: Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In *Advances in Neural Information Processing Systems*, volume 33, pages 1877–1901, 2020.

[^15]: Lucius EJ Bynum, Aahlad Manas Puli, Diego Herrero-Quevedo, Nhi Nguyen, Carlos Fernandez-Granda, Kyunghyun Cho, and Rajesh Ranganath. Black box causal inference: Effect estimation via meta prediction. *arXiv:2503.05985*, 2025.

[^16]: Stephanie C. Y. Chan, Adam Santoro, Andrew K. Lampinen, Jane X. Wang, Aaditya Singh, Pierre H. Richemond, Jay McClelland, and Felix Hill. Data distributional properties drive emergent in-context learning in transformers, 2022.

[^17]: Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. Double/debiased machine learning for treatment and causal parameters. *arXiv:1608.00060*, 2016.

[^18]: Davin Choo, Chandler Squires, Arnab Bhattacharyya, and David Sontag. Probably approximately correct high-dimensional causal effect estimation given a valid adjustment set. *arXiv preprint arXiv:2411.08141*, 2024.

[^19]: Julian Coda-Forno, Marcel Binz, Zeynep Akata, Matt Botvinick, Jane Wang, and Eric Schulz. Meta-in-context learning in large language models. *Advances in Neural Information Processing Systems*, 36:65189–65201, 2023.

[^20]: Alicia Curth. CATENets: Sklearn-style Implementations of Neural Network-based Conditional Average Treatment Effect (CATE) Estimators. [https://github.com/AliciaCurth/CATENets](https://github.com/AliciaCurth/CATENets), 2021. GitHub repository, commit 821bfb0. Accessed: 2025-05-11.

[^21]: Alicia Curth and Mihaela van der Schaar. On inductive biases for heterogeneous treatment effect estimation. In *Advances in Neural Information Processing Systems*, volume 34, pages 15883–15894, 2021a.

[^22]: Alicia Curth and Mihaela van der Schaar. Nonparametric estimation of heterogeneous treatment effects: From theory to learning algorithms. In *Proceedings of the 24th International Conference on Artificial Intelligence and Statistics (AISTATS)*. PMLR, 2021b.

[^23]: Alicia Curth, David Svensson, James Weatherall, and Mihaela van der Schaar. Really doing great at estimating cate? a critical look at ml benchmarking practices in treatment effect estimation. In *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*, 2021.

[^24]: Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why can GPT learn in-context? language models secretly perform gradient descent as meta-optimizers. In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 4005–4019. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.findings-acl.247.

[^25]: Aaron Defazio, Xingyu Yang, Ahmed Khaled, Konstantin Mishchenko, Harsh Mehta, and Ashok Cutkosky. The road less scheduled. *Advances in Neural Information Processing Systems*, 37:9974–10007, 2024.

[^26]: Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, et al. A survey on in-context learning. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 1107–1128, 2024.

[^27]: Joseph L Doob. Application of the theory of martingales. *Le calcul des probabilites et ses applications*, pages 23–27, 1949.

[^28]: Vincent Dorie, Jennifer Hill, Uri Shalit, Marc Scott, and Dan Cervone. Automated versus do-it-yourself methods for causal inference: Lessons learned from a data analysis competition. *Statistical Science*, 34(1):43–68, 2019.

[^29]: Sebastian Felix Fischer, Matthias Feurer, and Bernd Bischl. OpenML-CTR23 – A curated tabular regression benchmarking suite. In *AutoML Conference (Workshop)*, 2023.

[^30]: Dylan J Foster and Vasilis Syrgkanis. Orthogonal statistical learning. *The Annals of Statistics*, 51(3):879–908, 2023.

[^31]: Michele Jonsson Funk, Daniel Westreich, Chris Wiesen, Til Stürmer, M Alan Brookhart, and Marie Davidian. Doubly robust estimation of causal effects. *American Journal of Epidemiology*, 173(7):761–767, 2011.

[^32]: Marta Garnelo, Dan Rosenbaum, Christopher Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo Rezende, and S. M. Ali Eslami. Conditional neural processes. In *Proceedings of the 35th International Conference on Machine Learning*, volume 80, pages 1704–1713, 2018a.

[^33]: Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J Rezende, SM Eslami, and Yee Whye Teh. Neural processes. *arXiv:1807.01622*, 2018b.

[^34]: Pieter Gijsbers, Marcos LP Bueno, Stefan Coors, Erin LeDell, Sébastien Poirier, Janek Thomas, Bernd Bischl, and Joaquin Vanschoren. AMLB: An AutoML benchmark. *Journal of Machine Learning Research*, 25(101):1–65, 2024.

[^35]: Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux. Why do tree-based models still outperform deep learning on typical tabular data? In *Advances in Neural Information Processing Systems*, 2022.

[^36]: Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In *International Conference on Machine Learning*, pages 1321–1330. PMLR, 2017.

[^37]: P Richard Hahn, Jared S Murray, and Carlos M Carvalho. Bayesian regression tree models for causal inference: Regularization, confounding, and heterogeneous effects (with discussion). *Bayesian Analysis*, 15(3):965–1056, 2020.

[^38]: Kai Helli, David Schnurr, Noah Hollmann, Samuel Müller, and Frank Hutter. Drift-resilient tabpfn: In-context learning temporal distribution shifts on tabular data. *Advances in Neural Information Processing Systems*, 37:98742–98781, 2024.

[^39]: Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, and Yuxuan Chen. Query-key normalization for transformers. *arXiv preprint arXiv:2010.04245*, 2020.

[^40]: Miguel A Hernán and James M Robins. Causal inference, 2010.

[^41]: Jennifer L Hill. Bayesian nonparametric modeling for causal inference. *Journal of Computational and Graphical Statistics*, 20(1):217–240, 2011.

[^42]: Kevin Hillstrom. Minethatdata e-mail analytics and data mining challenge dataset. [https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html), 2008. Accessed: 2025-05-11.

[^43]: Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. TabPFN: A transformer that solves small tabular classification problems in a second. In *The Eleventh International Conference on Learning Representations*, 2023.

[^44]: Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. *Nature*, 637(8045):319–326, 2025.

[^45]: Guido W Imbens and Donald B Rubin. Bayesian inference for causal effects in randomized experiments with noncompliance. *The annals of statistics*, pages 305–327, 1997.

[^46]: Guido W Imbens and Donald B Rubin. *Causal inference in statistics, social, and biomedical sciences*. Cambridge University Press, 2015.

[^47]: Andrew Jesson, Sören Mindermann, Uri Shalit, and Yarin Gal. Identifying causal-effect inference failure with uncertainty-aware models. *Advances in Neural Information Processing Systems*, 33:11637–11649, 2020.

[^48]: Michael I Jordan, Zoubin Ghahramani, Tommi S Jaakkola, and Lawrence K Saul. An introduction to variational methods for graphical models. *Machine Learning*, 37:183–233, 1999.

[^49]: Hamidreza Kamkari, Vahid Balazadeh, Vahid Zehtab, and Rahul G Krishnan. Order-based structure learning with normalizing flows. *arXiv:2308.07480*, 2023.

[^50]: Nan Rosemary Ke, Silvia Chiappa, Jane X Wang, Jorg Bornschein, Anirudh Goyal, Melanie Rey, Theophane Weber, Matthew Botvinick, Michael Curtis Mozer, and Danilo Jimenez Rezende. Learning to induce causal structure. In *International Conference on Learning Representations*, 2023.

[^51]: Edward H. Kennedy. Towards optimal doubly robust estimation of heterogeneous causal effects. *Electronic Journal of Statistics*, 17(2):3008–3049, 2023.

[^52]: Ilyes Khemakhem, Ricardo Monti, Robert Leech, and Aapo Hyvarinen. Causal autoregressive flows. In *International Conference on Artificial Antelligence and Statistics*, pages 3520–3528. PMLR, 2021.

[^53]: Hyunjik Kim, Andriy Mnih, Jonathan Schwarz, Marta Garnelo, Ali Eslami, Dan Rosenbaum, Oriol Vinyals, and Yee Whye Teh. Attentive neural processes. In *International Conference on Learning Representations*, 2019.

[^54]: Diederik P Kingma. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014.

[^55]: Andrei Konstantinov, Stanislav Kirpichenko, and Lev Utkin. Heterogeneous treatment effect with trained kernels of the nadaraya–watson regression. *Algorithms*, 16(5):226, 2023.

[^56]: Sören R. Künzel, Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. Metalearners for estimating heterogeneous treatment effects using machine learning. *Proceedings of the National Academy of Sciences*, 116(10):4156–4165, 2019.

[^57]: Robert J LaLonde. Evaluating the econometric evaluations of training programs with experimental data. *The American Economic Review*, pages 604–620, 1986.

[^58]: Allison Lau, Younwoo Choi, Vahid Balazadeh, Keertana Chidambaram, Vasilis Syrgkanis, and Rahul G Krishnan. Personalized adaptation via in-context preference learning. *arXiv preprint arXiv:2410.14001*, 2024.

[^59]: Jonathan Lee, Annie Xie, Aldo Pacchiano, Yash Chandak, Chelsea Finn, Ofir Nachum, and Emma Brunskill. Supervised pretraining can learn in-context reinforcement learning. *Advances in Neural Information Processing Systems*, 36:43057–43083, 2023.

[^60]: Lenta LLC. Lenta uplift dataset. [https://github.com/maks-sh/scikit-uplift](https://github.com/maks-sh/scikit-uplift), 2020. Accessed: 2025-05-11.

[^61]: Fan Li, Peng Ding, and Fabrizia Mealli. Bayesian causal inference: a critical review. *Philosophical Transactions of the Royal Society A*, 381(2247):20220153, 2023a.

[^62]: Yingcong Li, Muhammed Emrullah Ildiz, Dimitris Papailiopoulos, and Samet Oymak. Transformers as algorithms: Generalization and stability in in-context learning. In *International Conference on Machine Learning*, pages 19565–19594. PMLR, 2023b.

[^63]: Antonio R Linero and Joseph L Antonelli. The how and why of bayesian nonparametric causal inference. *Wiley Interdisciplinary Reviews: Computational Statistics*, 15(1):e1583, 2023.

[^64]: Manqing Liu, David R Bellamy, and Andrew L Beam. Dag-aware transformer for causal effect estimation. *arXiv preprint arXiv:2410.10044*, 2024.

[^65]: Si-Yang Liu and Han-Jia Ye. TabPFN Unleashed: A Scalable and Effective Solution to Tabular Classification Problems. *arXiv:2502.02527*, 2025.

[^66]: Lars Lorch, Scott Sussex, Jonas Rothfuss, Andreas Krause, and Bernhard Schölkopf. Amortized inference for causal structure learning. *Advances in Neural Information Processing Systems*, 35:13104–13118, 2022.

[^67]: Christos Louizos, Uri Shalit, Joris M Mooij, David Sontag, Richard Zemel, and Max Welling. Causal effect inference with deep latent-variable models. In *Advances in Neural Information Processing Systems*, volume 30, 2017.

[^68]: Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Hamidreza Kamkari, Alex Labach, Jesse C Cresswell, Keyvan Golestan, Guangwei Yu, Maksims Volkovs, and Anthony L Caterini. TabDPT: Scaling Tabular Foundation Models. *arXiv:2410.18164*, 2024a.

[^69]: Yuchen Ma, Valentyn Melnychuk, Jonas Schweisthal, and Stefan Feuerriegel. DiffPO: A causal diffusion model for learning distributions of potential outcomes. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024b. URL [https://openreview.net/forum?id=merJ77Jipt](https://openreview.net/forum?id=merJ77Jipt).

[^70]: David P MacKinnon, Amanda J Fairchild, and Matthew S Fritz. Mediation analysis. *Annu. Rev. Psychol.*, 58(1):593–614, 2007.

[^71]: Divyat Mahajan, Jannes Gladrow, Agrin Hilmkil, Cheng Zhang, and Meyer Scetbon. Zero-shot learning of causal models. *arXiv:2410.06128*, 2024a.

[^72]: Divyat Mahajan, Ioannis Mitliagkas, Brady Neal, and Vasilis Syrgkanis. Empirical analysis of model selection for heterogeneous causal effect estimation. In *The Twelfth International Conference on Learning Representations*, 2024b.

[^73]: Irina Elisova Maksim Shevchenko. User guide for uplift modeling and casual inference. [https://www.uplift-modeling.com/en/latest/user\_guide/index.html](https://www.uplift-modeling.com/en/latest/user_guide/index.html), 2020.

[^74]: Charles F Manski. Identification problems in the social sciences. *Sociological methodology*, pages 1–56, 1993.

[^75]: Calvin McCarter. What exactly has TabPFN learned to do? *arXiv:2502.08978*, 2025.

[^76]: Duncan McElfresh, Sujay Khandagale, Jonathan Valverde, Vishak Prasad C, Ganesh Ramakrishnan, Micah Goldblum, and Colin White. When do neural nets outperform boosted trees on tabular data? In *Advances in Neural Information Processing Systems*, 2023.

[^77]: Megafon PJSC. Megafon uplift dataset. [https://github.com/maks-sh/scikit-uplift](https://github.com/maks-sh/scikit-uplift), 2020. Accessed: 2025-05-11.

[^78]: Jeffrey W Miller. A detailed treatment of Doob’s theorem. *arXiv:1801.03122*, 2018.

[^79]: Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers Can Do Bayesian Inference. In *International Conference on Learning Representations*, 2022.

[^80]: Brady Neal, Chin-Wei Huang, and Sunand Raghupathi. Realcause: Realistic causal inference benchmarking. *arXiv:2011.15007*, 2020.

[^81]: Radford M Neal. *Bayesian learning for neural networks*, volume 118. Springer Science & Business Media, 2012.

[^82]: Hamed Nilforoshan, Michael Moor, Yusuf Roohani, Yining Chen, Anja Šurina, Michihiro Yasunaga, Sara Oblak, and Jure Leskovec. Zero-shot causal learning. *Advances in Neural Information Processing Systems*, 36:6862–6901, 2023.

[^83]: Arman Oganisian and Jason A Roy. A practical introduction to bayesian estimation of causal effects: Parametric and nonparametric approaches. *Statistics in Medicine*, 40(2):518–551, 2021.

[^84]: Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads. *arXiv:2209.11895*, 2022.

[^85]: Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift. In *Advances in Neural Information Processing Systems*, volume 32, 2019.

[^86]: Judea Pearl. *Causality*. Cambridge University Press, 2009.

[^87]: Jonas Peters, Joris M Mooij, Dominik Janzing, and Bernhard Schölkopf. Causal discovery with continuous additive noise models. *The Journal of Machine Learning Research*, 15(1):2009–2053, 2014.

[^88]: Jonas Peters, Dominik Janzing, and Bernhard Schölkopf. *Elements of causal inference: Foundations and learning algorithms*. The MIT Press, 2017.

[^89]: Maxime Peyrard and Kyunghyun Cho. Meta-statistical learning: Supervised learning of statistical inference. *arXiv preprint arXiv:2502.12088*, 2025.

[^90]: Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. TabICL: A Tabular Foundation Model for In-Context Learning on Large Data. *arXiv:2502.05564*, 2025.

[^91]: Nicholas Radcliffe. Using control groups to target on predicted lift: Building and assessing uplift models. Technical report, Stochastic Solutions, 2007.

[^92]: Craig T Ramey, Donna M Bryant, Barbara H Wasik, Joseph J Sparling, Kaye H Fendt, and Lisa M La Vange. Infant health and development program for low birth weight, premature infants: Program elements, family participation, and child intelligence. *Pediatrics*, 89(3):454–465, 1992.

[^93]: Retail Hero. Retail hero (x5) uplift dataset. [https://github.com/maks-sh/scikit-uplift](https://github.com/maks-sh/scikit-uplift), 2020. Accessed: 2025-05-11.

[^94]: Paul R Rosenbaum and Donald B Rubin. The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1):41–55, 1983.

[^95]: Donald B Rubin. Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66(5):688, 1974.

[^96]: Donald B Rubin. Bayesian inference for causal effects: The role of randomization. *The Annals of Statistics*, pages 34–58, 1978.

[^97]: Donald B Rubin. Causal inference using potential outcomes: Design, modeling, decisions. *Journal of the American Statistical Association*, 100(469):322–331, 2005.

[^98]: Meyer Scetbon, Joel Jennings, Agrin Hilmkil, Cheng Zhang, and Chao Ma. A fixed-point approach for causal generative modeling. In *Proceedings of the 41st International Conference on Machine Learning*, volume 235, pages 43504–43541, 2024.

[^99]: Alejandro Schuler, Michael Baiocchi, Robert Tibshirani, and Nigam Shah. A comparison of methods for model selection when estimating individual treatment effects. *arXiv preprint arXiv:1804.05146*, 2018.

[^100]: Uri Shalit, Fredrik D Johansson, and David Sontag. Estimating individual treatment effect: generalization bounds and algorithms. In *International conference on machine learning*, pages 3076–3085. PMLR, 2017.

[^101]: Noam Shazeer. Glu variants improve transformer. *arXiv preprint arXiv:2002.05202*, 2020.

[^102]: Claudia Shi, David Blei, and Victor Veitch. Adapting neural networks for the estimation of treatment effects. In *Advances in Neural Information Processing Systems*, volume 32, 2019.

[^103]: Yishai Shimoni, Chen Yanover, Ehud Karavani, and Yaara Goldschmnidt. Benchmarking framework for performance-evaluation of causal inference analysis. *arXiv preprint arXiv:1802.05046*, 2018.

[^104]: Valentin Thomas, Junwei Ma, Rasa Hosseinzadeh, Keyvan Golestan, Guangwei Yu, Maks Volkovs, and Anthony L Caterini. Retrieval & fine-tuning for in-context tabular models. *Advances in Neural Information Processing Systems*, 37:108439–108467, 2024.

[^105]: Julius Vetter, Manuel Gloeckler, Daniel Gedon, and Jakob H Macke. Effortless, simulation-efficient bayesian inference using tabular foundation models. *arXiv preprint arXiv:2504.17660*, 2025.

[^106]: Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In *International Conference on Machine Learning*, pages 35151–35174. PMLR, 2023.

[^107]: Johannes von Oswald, Maximilian Schlegel, Alexander Meulemans, Seijin Kobayashi, Eyvind Niklasson, Nicolas Zucchet, Nino Scherrer, Nolan Miller, Mark Sandler, Blaise Agüera y Arcas, Max Vladymyrov, Razvan Pascanu, and João Sacramento. Uncovering mesa-optimization algorithms in transformers. *arXiv:2309.05858*, 2023.

[^108]: Kevin Xia, Kai-Zhan Lee, Yoshua Bengio, and Elias Bareinboim. The causal-neural connection: Expressiveness, learnability, and inference. *Advances in Neural Information Processing Systems*, 34:10823–10836, 2021.

[^109]: Kevin Xia, Yushu Pan, and Elias Bareinboim. Neural causal models for counterfactual identification and estimation. *arXiv preprint arXiv:2210.00035*, 2022.

[^110]: Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. In *International Conference on Learning Representations*, 2022.

[^111]: Steve Yadlowsky, Lyric Doshi, and Nilesh Tripuraneni. Pretraining data mixtures enable narrow model selection capabilities in transformer models. *arXiv:2311.00871*, 2023.

[^112]: Han-Jia Ye, Si-Yang Liu, and Wei-Lun Chao. A closer look at TabPFN v2: Strength, limitation, and extension. *arXiv:2502.17361*, 2025.

[^113]: Jiaqi Zhang, Joel Jennings, Agrin Hilmkil, Nick Pawlowski, Cheng Zhang, and Chao Ma. Towards causal foundation model: on duality between causal inference and attention. *arXiv:2310.00809*, 2023.

[^114]: Xun Zheng, Bryon Aragam, Pradeep K Ravikumar, and Eric P Xing. DAGs with NO TEARS: Continuous Optimization for Structure Learning. In *Advances in Neural Information Processing Systems*, volume 31, 2018.

[^115]: Émilie Diemert, Olivier Teytaud, Guillaume Oblé, and Florent Meynet. A large scale benchmark for uplift modeling. In *Proceedings of the AdKDD Workshop*, 2018. URL [https://www.cs.cornell.edu/˜diemert/criteo-uplift-dataset/](https://www.cs.cornell.edu/~diemert/criteo-uplift-dataset/).