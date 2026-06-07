---
title: "GIT-BO: High-Dimensional Bayesian Optimization with Tabular Foundation Models"
source: "https://ar5iv.labs.arxiv.org/html/2505.20685"
author:
published:
created: 2026-06-03
description: "Bayesian optimization (BO) effectively optimizes expensive black-box functions but faces significant challenges in high-dimensional spaces (dimensions exceeding 100) due to the curse of dimensionality. Existing high-di…"
tags:
  - "clippings"
---
Rosen Ting-Ying Yu  
Center of Computational Science and Engineering  
Massachusetts Institute of Technology  
Cambridge, MA 02139  
rosenyu@mit.edu &Cyril Picard  
Department of Mechanical Engineering  
Massachusetts Institute of Technology  
Cambridge, MA 02139  
cyrilp@mit.edu &Faez Ahmed  
Department of Mechanical Engineering  
Massachusetts Institute of Technology  
Cambridge, MA 02139  
faez@mit.edu

###### Abstract

Bayesian optimization (BO) effectively optimizes expensive black-box functions but faces significant challenges in high-dimensional spaces (dimensions exceeding 100) due to the curse of dimensionality. Existing high-dimensional BO methods typically leverage low-dimensional embeddings or structural assumptions to mitigate this challenge, yet these approaches frequently incur considerable computational overhead and rigidity due to iterative surrogate retraining and fixed assumptions. To address these limitations, we propose Gradient-Informed Bayesian Optimization using Tabular Foundation Models (GIT-BO), an approach that utilizes a pre-trained tabular foundation model (TFM) as a surrogate, leveraging its gradient information to adaptively identify low-dimensional subspaces for optimization. We propose a way to exploit internal gradient computations from the TFM’s forward pass by creating a gradient-informed diagnostic matrix that reveals the most sensitive directions of the TFM’s predictions, enabling optimization in a continuously re-estimated active subspace without the need for repeated model retraining. Extensive empirical evaluation across 23 synthetic and real-world benchmarks demonstrates that GIT-BO consistently outperforms four state-of-the-art Gaussian process-based high-dimensional BO methods, showing superior scalability and optimization performances especially as dimensionality increases up to 500 dimensions. This work establishes foundation models, augmented with gradient-informed adaptive subspace identification, as highly competitive alternatives to traditional Gaussian process-based approaches for high-dimensional Bayesian optimization tasks.

## 1 Introduction

Bayesian optimization (BO) is widely recognized as an effective and sample-efficient technique for black-box optimization, essential in machine learning [^1] [^2], engineering design [^3] [^4] [^5] [^6], and hyperparameter tuning [^7] [^8]. However, BO can struggle in high-dimensional spaces (especially when dimension $D>$ 100) due to the curse of dimensionality, making it difficult for conventional Gaussian process (GP)-based methods to effectively explore complex objective functions [^9] [^10] [^11] [^12] [^13]. Existing high-dimensional BO approaches mitigate these challenges by exploiting low-dimensional subspaces, but still rely heavily on iterative model retraining and structural assumptions, incurring substantial computational overhead [^14] [^15] [^13] [^16].

Recent advances in tabular foundation models (TFM) have proposed using Prior-Data Fitted Networks (PFNs) to bypass surrogate refitting in Bayesian Optimization. Initially demonstrated for simple low-dimensional BO problems [^17], the PFN-based BO was further validated for constrained engineering problems [^6], using the parallel processing capabilities of the transformer architecture to outperform other methods in both speed and performance. TabPFN v2, a recent TFM model, extends the capabilities of TabPFN to handle inputs up to 500 dimensions, demonstrating superior performance in zero-shot classification, regression, and time series prediction tasks [^18].

While TabPFN v2 presents an exciting opportunity as a high-dimensional BO surrogate due to its computational efficiency and predictive accuracy, its fixed-parameter foundation model architecture restricts its ability to dynamically adapt kernel hyperparameters. This is in contrast to traditional GP models, which adjust their kernel parameters during optimization to effectively identify principal searching directions in high-dimensional spaces [^16] [^11] [^12]. Addressing this limitation requires an approach that can leverage the strengths of foundation models while introducing adaptive mechanisms for subspace exploration, a gap we aim to bridge in this work.

To address this, we introduce Gradient-Informed Tabular Bayesian Optimization (GIT-BO), a framework that utilizes gradient information from the foundation model’s forward pass to dynamically guide adaptive subspace exploration. Specifically, our contributions are as follows:

- We propose GIT-BO, a novel Bayesian optimization approach that employs a tabular foundation model as a surrogate, demonstrating superior scaling properties and performance on complex optimization landscapes where traditional GP-based methods struggle.
- Our gradient-informed subspace identification technique adaptively learns promising search directions in high-dimensional spaces, exploiting the foundation model’s adaptive gradient knowledge during the optimization trajectory.
- We compare GIT-BO against four popular state-of-the-art (SOTA) GP high-dimensional BO methods across twenty-three diverse benchmarks, including synthetic functions and real-world problems, introducing foundation model surrogates as a viable alternative for complex Bayesian optimization tasks.

The rest of the paper is structured as follows. Section 2 states the background problem and discusses related work. Section 3 presents the GIT-BO algorithm. Section 4 defines the benchmark high-dimensional BO algorithms of interest, the test problem set, and the evaluation methods for algorithm performance. The results are presented in Section 5, and Section 6 is the concluding discussion.

## 2 Background

#### Bayesian Optimization and the High-Dimensional Challenge

Bayesian optimization is a sample-efficient approach for optimizing expensive black-box functions where the objective is to find $x^{*}\in\arg\min_{x\in\mathcal{X}}f(x)$ with $\mathcal{X}=[0,1]^{D}$. The algorithm leverages a surrogate model, typically a GP, to approximate the objective function, and calculate the acquisition function to determine the next query points until it converges to the optimum. As dimensionality $D$ grows, effectively exploring the space often requires more observations $n$, pushing GPs into regimes of high computational cost ($O(n^{3})$ complexity) [^19] [^10] [^20].

#### Existing Strategies for Tackling High-Dimensional Bayesian Optimization Challenge

Several baseline approaches have been developed to address high-dimensional BO challenges. Random Embedding Bayesian Optimization (REMBO) projects the optimization domain onto a randomly selected low-dimensional linear subspace [^21] [^22] [^13] [^23] [^24]. However, REMBO suffers from distorted objective values due to clipping points back into the feasible domain, impairing GP calibration [^21]. Hashing-Enhanced Subspace Bayesian Optimization (HESBO) replaces Gaussian projections with a sparse hashing sketch that keeps every point feasible, at the price of a slightly smaller chance of containing the active subspace [^22]. Another approach, Adaptive Linear Embedding Bayesian Optimization (ALEBO), creates a more effective optimization framework by employing linear constraints that preserve the embedding’s linearity to further improve optimization performance [^13]. All three methods, however, require the embedding dimension $d$ to be defined by users. BAXUS lifts this restriction by growing a nested sequence of random sub-spaces as data accumulate [^23].

A complementary line of work models the full space but exploits structure. TURBO maintains multiple GP-driven trust regions that expand when progress is made and shrink otherwise, achieving strong results up to 200 variables with thousands of evaluations [^15]. SAASBO places half-Cauchy shrinkage priors on GP length-scales, automatically selecting a sparse axis-aligned subset of salient variables [^16]. However, its computational cost grows cubically with the sample count rather than the dimension. While these methods offer significant improvements, they still entail limitations such as fixed-dimensional embedding choices or incremental computational overhead, highlighting the need for alternative approaches that can more dynamically and efficiently address high-dimensional optimization problems.

#### Foundation Models as Bayesian Optimization Surrogates

Tabular foundation models have emerged as promising alternatives to GP-based methods in Bayesian optimization. Prior-data Fitted Networks (PFNs) [^25] demonstrated that pre-trained tabular foundation models can effectively replace GPs as surrogate models. PFNs4BO [^17] showed competitive performance on low-dimensional BO tasks with significantly reduced computational overhead compared to GPs. However, PFNs inherently lack the adaptive capability of GPs in dynamically updating kernel hyperparameters during optimization, limiting their ability to discover low-rank structures within high-dimensional spaces.

Interestingly, a single forward pass yields internal gradients with respect to inputs and hidden states. Studies in the field of Large Language Models (LLMs) show that leveraging gradient information can help guide model behavior [^26] or perform rapid adaptation to distribution shifts [^27]. These findings motivate our exploration of gradient-informed techniques within Bayesian optimization, particularly to harness the intrinsic structure of high-dimensional search spaces.

#### Gradient-Enhanced Bayesian Optimization

Previous work demonstrates how derivative information can accelerate Bayesian optimization through various mechanisms or capture the search space structure [^28] [^29] [^30]. When explicit derivatives are available, a direct integration of gradient information is to jointly model function values and gradients in a Gaussian process [^28]. Strategies such as the parallel knowledge gradient method for batch BO [^29] and sampling the most informative observation subset using GP log-likelihood gradients [^30] have shown faster convergence of BO. However, these methods inherently depend on precise GP modeling, which becomes less effective as dimensionality grows.

To overcome these limitations, we introduced the Gradient-Informed Bayesian Optimization using Tabular Foundation Models (GIT-BO) algorithm—a method that does not require fitting a GP. Our approach fundamentally departs from the GP gradient-enhanced BO methods: We combine the gradient information obtained directly from the forward pass of tabular foundation models and integrate subspace identification, an established strategy in high-dimensional optimization. We can approximate high-dimensional target measures with low-dimensional updates by adapting the principal feature detection framework based on $\phi$ -Sobolev inequalities [^31]. This approach combines the computational efficiency of foundation models with subspace identification capabilities, enabling effective exploration in high-dimensional spaces.

## 3 The GIT-BO Algorithm

We now present our GIT-BO high-dimensional BO framework. The framework consists of four main components: the surrogate model (TabPFN v2), the gradient-based subspace identification, the choice of acquisition function, and the algorithm that ties these together.

#### Surrogate Modeling with TabPFN

We use TabPFN, specifically the 500-dimensional TabPFN v2 TFM model from [^18], as our surrogate model to predict the objective function values using in-context learning. The optimization objective is defined as minimizing an unknown function $f(x)$ over a domain $x\in\mathcal{X}\subset\mathbb{R}^{D}$. At any BO iteration, let $D_{n}={(x_{i},y_{i})}_{i=1}^{n}$ be the dataset of points sampled so far and their observed function values $y_{i}=f(x_{i})$. In a standard BO setting, one would fit a GP posterior $p(f|\mathcal{D}n)$ to these data. Here instead, we leverage TabPFN to obtain a predictive model. We provide the dataset $D_{n}$ to TabPFN as input context (taking a set of labeled examples as part of its input sequence) and query the model for its prediction at any candidate point $x$. TabPFN then returns an approximate posterior predictive distribution $\text{PFN}(y\mid x,D_{n})$ informed by both the new point $x$ and the context of observed samples. In practice, TabPFN produces a prediction in a single forward pass as an approximation to the Bayesian posterior mean $\mu_{n}(x)$ for $f(x)$ as this predictive mean given the provided data. In our method, we review the previous series of PFN work [^32] and implement the calculation of predictive mean $\mu_{n}(x)$ and predictive variance $\sigma_{n}^{2}(x)$ with the latest TabPFN v2 regressor model.

#### Gradient-Informed Subspace Identification and Sampling

For active subspace identification, we leverage gradient information $\nabla_{x}\mu_{n}(x)$ obtained through one-step backpropagation on TabPFN’s predictive mean values, which naturally varies across data points at each iteration. Inspired by studies on dimension reduction in nonlinear Bayesian inverse problems, which employs techniques $\phi$ -Sobolev inequalities and gradient-based approaches, we approximate the diagnostic (Fisher information) matrix ${H}=\mathbb{E}_{\mu}[\nabla_{x}\mu_{n}(x)\,\nabla_{x}\mu_{n}(x)^{\top}]$ [^33] [^34] [^31] [^35]. In this study of high-dimensional problems with only $D>100$, we select the top $r\;(D>>r)$ dominant eigenvectors in ${H}$ as the principal vectors ${V}_{r}$ that span our gradient-informed active subspace (GI subspace). Next, we restrict the next exploration to this subspace and map the search candidates back to the original space by $x_{\text{cand}}=\bar{x}_{ref}+{V}_{r}{z},\;{z}\sim\mathrm{U}([-1,1]^{r})$, where the reference point $\bar{x}_{ref}$ is the centroid of the observed data. Finally, we pass $x_{\text{cand}}$ to the acquisition function for determining the next sample to evaluate. In GIT-BO, we recalculate the GI subspace at each iteration as the subspace changes when observed data increases. For computational practicality and exploration-exploitation balance, we default to $r=15$ in GIT-BO; a detailed sensitivity analysis on the choice of $r$ is presented in the Appendix.

#### Acquisition Function

We implement the Thompson Sampling (TS) acquisition as our acquisition function due to its previous success in high-dimensional BO [^15]. For TS, we approximate it by sampling from the predictive distribution at each sample $x$ and draw a fixed number of 512 random samples $\tilde{f}(\cdot)$ from the surrogate’s posterior. The candidate $x_{\text{next}}$ with the highest sampled value is selected as the next query $\tilde{f}(x_{\text{next}})$. This leverages the full predictive uncertainty and tends to balance exploration and exploitation implicitly [^36] [^15]. The key difference here is that we restrict $x$ to our learned GI subspace. After selecting the next query point $x_{\text{next}}$, we evaluate the true objective function to obtain $y_{\text{next}}$.

Putting everything together, Figure 1 and Algorithm 1 outlines the GIT-BO procedure combining TabPFN with gradient-informed subspace search. The implementation details of GIT-BO are stated in the Appendix.

Figure 1: GIT-BO algorithm overview. The method operates in four stages: (1) Initial observed samples are collected in the high-dimensional space $\mathbb{R}^{D}$; (2) TabPFN v2, a fixed-weight tabular foundation model generates predictions of the objective space at inference time using in-context learning; (3) Gradient information from TabPFN’s forward pass ($\nabla\hat{\mu}(x)$) is used to identify a low-dimensional gradient-informed (GI) subspace. The predicted mean and variance are used for acquisition value calculations $\hat{\mu}(x),\hat{\sigma}^{2}(x)$; (4) The next sample point ($x_{\text{next}}$) is selected within this GI subspace with the highest acquisition value and appended to the dataset for iterative search until stopping criteria is met.

Algorithm 1 Gradient-Informed Bayesian Optimization using Tabular Foundation Models (GIT-BO)

 objective $f$, domain $\mathcal{X}\!\subset\!\mathbb{R}^{D}$, init. size $n_{0}$, iteration budget $I$, subspace dim. $r$

 Draw $n_{0}$ LHS points $x_{i}$ and set $y_{i}=f(x_{i})$;  $D\leftarrow\{(x_{i},y_{i})\}_{i=1}^{n_{0}}$

 for $i=1\;\text{to}\;I$ do

  Fit TabPFN on $D$ to obtain mean $\mu_{n}$ and variance $\sigma^{2}_{n}$

  Calculate gradient $\nabla_{x}\mu_{n}(x_{i})$ for all $(x_{i},y_{i})\!\in\!D$

  Form diagnostic matrix ${H}=\mathbb{E}_{\mu}[\nabla_{x}\mu_{n}(x)\,\nabla_{x}\mu_{n}(x)^{\top}]$;

   ${V}_{r}\leftarrow$ top- $r$ eigvectors of ${H}$

  Set $x_{\text{cand}}\leftarrow x_{\text{ref}}+V_{r}z$, with $x_{\text{ref}}\leftarrow\bar{x}$ and $z\sim\mathrm{U}([-1,1]^{r})$

   $x_{\text{next}}\leftarrow\arg\max_{j}\mathrm{ThompsonSampling}_{n}(x_{\text{cand}})$

  Evaluate $y_{\text{next}}=f(x_{\text{next}})$ and append the query point data $D\leftarrow D\cup\{(x_{\text{next}},y_{\text{next}})\}$

 end for

 return $x^{\star}=\arg\min_{(x,y)\in D}y$

## 4 Experiment

This section evaluates GIT-BO’s performance on a diverse collection of synthetic optimization benchmarks and real-world engineering problems.

### 4.1 Experiment Setups

This section outlines our empirical approach to evaluating and comparing different high-dimensional Bayesian optimization algorithms, specifically focusing on assessing their performance across various complex synthetic and engineering benchmarks.

#### Benchmark Algorithms

We benchmark GIT-BO against random search [^37] and four high-dimensional BO methods, including SAASBO [^16], TURBO [^15], HESBO [^22], and ALEBO [^13]. These high-dimensional GP BO algorithms have been introduced in Section 2. For HESBO and ALEBO, we varied $d_{\text{embedding}}\in\{10,20\}$ and denote the choice as HESBO/ALEBO (d={10, 20}). The implementation of SAASBO and TURBO is taken from BoTorch tutorials [^38]. The implementation of HESBO and ALEBO are taken from their original published paper and code [^22] [^13]. Additional details of the algorithm implementation are listed in the Appendix.

#### Test Problems

This study incorporates a diverse set of high-dimensional optimization problems, including 11 synthetic problems and 12 real-world benchmarks. Synthetic and scalable problems include: Ackley, Rosenbrock, Dixon-Price, Levy, Powell, Griewank, Rastrigin, Styblin-Tang, and Michalewicz. Additionally, we tested on two Synthetic problems in LassoBench (LassoSyntheticMedium and LassoSyntheticHigh) and one real-world hyperparameter optimization (HPO) problem LassoDNA. The rest of the application problems are collected from previous optimization studies and conference benchmarks: the power system optimization problems in CEC2020, Rover, SVM HPO, MOPTA08 car problem, and two Mazda car problems. As this study focus on the high-dimensional characteristic of problem, we make all our benchmark problems single and unconstrained for testing. Therefore, we have applied penalty transforms to all real-world problems with constraints and perform average weighting to the two multi-objective Mazda problems. The details of benchmark transformation and implementation are detailed in the Appendix. Among the 23 benchmarks, 10 (Synthetic + Rover) are scalable problems. To evaluate the algorithms’ performance with respect to dimensionality, we solve the scalable problems for $D=\{100,200,300,400,500\}$. Therefore, we have experimented with a total of $5\times 10+13=63$ different variants of the benchmark problems. The benchmarks’ details and implementations are listed in the Appendix.

#### Algorithm Tests

The algorithm evaluation aims to thoroughly compare GIT-BO to current SOTA Bayesian optimization techniques. This study focuses on minimizing the objective function for the given test problems. For each test problem, our experiment consists of 20 independent trials, each utilizing a distinct random seed. To ensure fair comparison, we initialize each algorithm with an identical set of 200 samples, generated through Latin Hypercube Sampling with consistent random seeds across all trials. During each iteration, each algorithm selects one sample to evaluate next.

To execute this extensive benchmarking process, we utilized a distributed server infrastructure featuring Intel Xeon Platinum 8480+ CPUs and NVIDIA H100 GPUs. Individual experiments were conducted with the same amount of compute allocated: a single H100 GPU node with 24 CPU cores and 250GB RAM.

### 4.2 Evaluation Metrics

#### Optimization Fixed-budget Convergence Analysis

Fixed-budget evaluations is a technique for comparing the efficiency of optimization algorithms by allotting specific computational resources for their execution [^39]. We apply a “fixed-iteration” budget approach in our study, initially running all algorithms (GIT-BO, TurBO, HESBO, SAASBO, and ALEBO) for 100 iterations. Additionally, we run GIT-BO, TurBO, and HESBO for an extra 100 iterations (total of 200 iterations) because these methods are roughly two orders of magnitude faster than SAASBO and ALEBO, allowing us to explore their long-term convergence behavior more thoroughly.

#### Statistical Ranking

For comprehensively compare and evaluate the performance of the Bayesian optimization algorithms, statistical ranking techniques are employed instead of direct performance measurements of the optimization outcome. In this study, we define the optimization performance result as the median of the minimum result found (final incumbent) across the 20 optimization trials of each algorithm. By statistically ranking the results, we were able to standardize the comparisons across different problems, since various optimization challenges can produce objective values of vastly different magnitudes. Furthermore, using this ranking allowed us to reduce the distorting effects of unusual or extreme data points that might influence our evaluation.

We conduct our statistical analysis using the Friedman and Wilcoxon signed-rank tests, complemented by Holm’s alpha correction. These non-parametric approaches excel at processing benchmarking result data without assuming specific distributions, which is critical for handling optimization results with outliers. These statistical methods effectively handle the dependencies in our setup, where we used the same initial samples and seeds to test all algorithms. The Wilcoxon signed-rank test addresses paired comparisons between algorithms, while the Friedman test manages problem-specific grouping effects. For multiple algorithm comparisons, we used Holm’s alpha correction to control error rates [^40] [^41] [^42].

#### Algorithm Runtime Recording

Each algorithm is timed for the total time it takes to run one experiment trial. We take the overall mean over the 20 trials and over all benchmark problems, resulting in a single value average time ($t_{\text{avg}}$) to represent the average runtime of each algorithm.

## 5 Results and Evaluations

In this section, we report the algorithm performance on a subset of the 63 benchmark optimization experiments and the ablation study on GIT-BO and vanilla TabPFN v2 BO. The complete optimization results for all synthetic and real-world engineering problems are detailed in the appendix.

#### Overall Statistical Ranking and Algorithm Runtime Tradeoffs

Across all problem variants, Figure 2 (a) shows that GIT-BO achieves the top statistical performance rank based on the optimization results (1.97), followed by SAASBO (3.51), ALEBO (d=10) (4.11), and HESBO (d=20) (5.14). Even when we enforce a strict iteration limit at 100 iterations, GIT-BO still outperforms other GP-based algorithms in the statistical rank demonstrated in Figure 2 (b). A detailed rank-iteration evolution curve in Figure 2 (c) shows that GIT-BO becomes dominant within the first 25 iterations and keeps that lead. Figure 2 (d) plots the average run time $t_{\text{avg}}$ of each algorithm versus the statistical rank. GIT-BO requires $\approx 10^{3}$ sec per trial, two orders of magnitude faster than SAASBO and ALEBO. With the first place in rank and third place in $t_{\text{avg}}$, GIT-BO is the Pareto frontier of speed and quality.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2505.20685/assets/x2.png)

Figure 2: (a) and (b): Statistical ranking of the overall performance across 63 benchmark problems. GIT-BO ranked the top at optimization results at both fixed 100- and 200-iteration evaluations. (c): Plot of average algorithm rank at each iteration. Due to the compute limit, we only run ALEBO and SAASBO for 100 iterations. We compare their final results after 100 iterations with other algorithms run for 200 iterations. The plot shows that GIT-BO achieves a fast convergence at ranking performance as it remains the top of all algorithms after 25 iterations. (d): Plot of average time vs overall statistical rank. The shorter time and smaller rank perform better (bottom left corner), so we show GIT-BO at the Paerto front as the best algorithm.

#### Scalable Synthetic Benchmarks (100 ≤\\leq D ≤\\leq 500)

Overall, GIT-BO outperforms other SOTA algorithms in 35 out of 45 variants of the scalable synthetic benchmark problems. The convergence results for these scalable synthetic problems can be summarized in three categories in Figure 3. First, GIT-BO has absolute domination over several problems (e.g., Ackley), outperforming all other algorithms in all dimensions. The second category, the most common observed ones, are that ALEBO (d=10) or SAASBO match or outperform GIT-BO in $D=100$; however, GIT-BO maintains its top optimization efficiency in the higher dimensions $D\geq 300$ (e.g., Levy). Finally, GIT-BO struggles with very few problems (e.g., Michalewicz), where the current SOTA methods with the trust region and linear embedding finding strategies dominate.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2505.20685/assets/x3.png)

Figure 3: Optimization results on the synthetic benchmarks comparing our method against baseline algorithms. The solid line represents the median best function value achieved over 20 trials, with shaded regions indicating the 95% confidence interval. GIT-BO ranks the top for nine variants out of 15, showing better performance at higher dimensional problems and faster convergence for the first 100 iterations. GIT-BO also has a lower variance compared to other baselines. In practice this means GIT-BO not only converges faster in expectation but also delivers reliably strong results without requiring multiple restarts, reducing both computational cost and risk. Full statistical tests and per-problem plots are provided in the Appendix.

#### Real-world Problem

Figure 4 illustrates the convergence results for a subset of eight real-world problems. GIT-BO outperforms all baselines on the four 100+D CEC engineering tasks on power system optimization and the 388D SVM hyperparameter optimization problem, showing its great potential to tackle real-world applications. GIT-BO achieves the second and third best in the LassoDNA problem and car problems (MOPTA08 Car and Mazda), respectively. One possible explanation is that the distribution of these engineering benchmarks might be very different from the training distribution of our TFM, TabPFN v2. Although SAASBO achieves the best for these problems, it takes on average three hours to solve them, while the second best, GIT-BO, takes only fifteen minutes. This showcases that GIT-BO offers a compelling trade-off for resource-constrained engineering optimization tasks where time-to-solution is critical.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2505.20685/assets/x4.png)

Figure 4: Optimization results on the engineering benchmarks comparing our method against baseline algorithms. The solid line represents the median best function value achieved over 20 trials, with shaded regions indicating the 95% confidence interval. GIT-BO performs the best for the CEC problems, and maintains a 1.15 average ranking over the real-world problem set.

#### Ablation Study

To rigorously assess the necessity of gradient-informed subspace exploration, we conducted an ablation study comparing GIT-BO with a variant employing the vanilla TabPFN v2 model without adaptive gradient-informed subspace identification. Two acquisition functions are tested with the vanilla TabPFN v2: expected improvement acquisition (EI) (used in the original PFNs4BO [^17]) and Thompson Sampling (TS), which is used by GIT-BO. The convergence plot results shown in Figure 5 clearly indicate that both vanilla TabPFN v2 with EI and TS has failed the high-dimensional search significantly compared to GIT-BO. Specifically, we observe a drastic degradation in convergence speed and final optimization outcomes. This confirms our hypothesis that vanilla TabPFN v2, regardless of the selection of the acquisition function, fails to capture crucial directions in high-dimensional spaces without the assistance of the GI subspace. This reinforces the critical role of our proposed gradient-based refinement strategy in achieving robust performance.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2505.20685/assets/x5.png)

Figure 5: Ablation studies of vanilla TabPFNv2 BO with GIT-BO. GIT-BO consistently outperform other algorithms without GI subspace search across all tested problems (8.6 times better regrets), indicating GIT-BO is the best approach and vanilla TabPFNv2 is not an ideal candidate for performing high-dimensional BO.

## 6 Discussion

#### Novelties and Strengths

Our results demonstrate that a tabular foundation model can serve as a top-class surrogate in high-dimensional Bayesian optimization. By combining TabPFN’s one-shot regression with a data-driven, gradient-informed subspace that is updated after every evaluation, GIT-BO converges faster than carefully tuned Gaussian-process baselines on sixty-three benchmarks, maintaining its lead even as dimensionality grows to 500D. The approach removes the computational bottleneck of GP retraining and latent space searching while preserving the exploration–exploitation balance that characterizes principled BO (two orders of magnitude higher than SAASBO). Taken together, these properties position GIT-BO as an accessible “drop-in” optimizer for engineers who already rely on large language models and are now seeking equally powerful tools for structured data and design optimization.

#### Limitations

Our approach, although effective, faces several practical limitations. First, the computational requirements of TabPFN present significant hardware constraints. As a large foundation model, TabPFN requires GPU acceleration for efficient inference. This limits accessibility for users without GPUs. Second, the current implementation imposes a hard dimensionality cap of 500D, as the foundation model was trained specifically with this input dimension limit. Problems exceeding this dimensionality cannot be directly addressed without architectural modifications or using ensembles. Expanding to higher-dimensional spaces would require retraining the entire foundation model with an expanded feature capacity, representing a substantial computational investment. Third, although GIT-BO is largely parameter-free, the rank $r$ of the GI subspace and the choice of reference point $x_{\text{ref}}$ for back-projection could influence the algorithm’s performance. We provide an initial sensitivity analysis in the Appendix. However, future work will explore using an automated heuristic or meta-learning of these hyperparameters.

#### Societal Impact

GIT-BO advances high-dimensional BO by leveraging gradient-informed subspace identification from pre-trained tabular foundation models, significantly improving optimization robustness and scalability. This methodological innovation can empower researchers and practitioners to efficiently tackle previously challenging optimization problems in critical application areas of engineering design where objective evaluation is expensive [^43] [^44] [^45]. Additionally, the pre-trained nature of foundation models enables effective reuse across diverse tasks, potentially lowering computational overhead by eliminating training application-specific prediction models in repeated optimization studies. This especially benefits adaptive experimentation, such as drug and material discovery, where the research results and the researcher’s schedule depend on the optimization algorithm’s output [^4] [^46] [^47]. Thus, GIT-BO not only democratizes high-dimensional Bayesian optimization methodologies but also contributes indirectly to reducing resource consumption in long-term scientific research and industrial optimization cycles.

#### Conclusion and Future Work

In this paper, we introduced Gradient-Informed Bayesian Optimization using Tabular Foundation Models (GIT-BO), a method integrating the computational efficiency of tabular foundation models with an adaptive, gradient-informed subspace identification strategy for Bayesian optimization. Our comprehensive evaluations across synthetic and real-world benchmarks demonstrate that GIT-BO consistently surpasses state-of-the-art Gaussian process-based methods in high-dimensional optimization scenarios (up to 500 dimensions). By eliminating iterative surrogate retraining and dynamically identifying critical subspaces, GIT-BO also accelerates convergence and reduces computational costs by orders of magnitude compared to traditional GP methods.

The capability to efficiently and effectively handle previously intractable high-dimensional optimization tasks broadens the practical applicability of Bayesian optimization. This advancement is particularly impactful for automated machine learning pipelines, scientific experimentation, and complex engineering design problems where dimensionality has historically impeded efficient optimization. As tabular foundation models continue to grow in model size and improve the prediction accuracy, we anticipate even greater optimization performance from approaches like GIT-BO, further extending Bayesian optimization to even more challenging real-world scenarios. Future work can leverage our method in multiple directions, such as constrained optimization, mixed-integer problems, and multi-objective optimization, through modifications to the acquisition functions and embedding techniques.

## Appendix A Additional Details on Benchmark Algorithms and Problems Implementations

### A.1 Reproducibility Statement

The code for experiments and the GIT-BO algorithm is provided in the supplemental material zip file. It can also be found in this anonymous link: [https://anonymous.4open.science/r/GITBO-9EFE/README.md](https://anonymous.4open.science/r/GITBO-9EFE/README.md).

### A.2 GIT-BO Implementation Details

The GIT-BO algorithm was implemented using Python 3.12 with the TabPFN v2.0.6 implementation and model (link: [https://github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN) and [https://huggingface.co/Prior-Labs/TabPFN-v2-reg](https://huggingface.co/Prior-Labs/TabPFN-v2-reg), license: Prior Lab License (a derivative of the Apache 2.0 license ([http://www.apache.org/licenses/](http://www.apache.org/licenses/)))). Instead of using the off-the-shelf TabPFN code for GIT-BO, we make a wrapper that converts the numpy calculations to PyTorch for torch.backward() gradient calculations. The system uses BoTorch v0.12.0 and PyTorch 2.6.0+cu126 for the underlying optimization framework. For the gradient-informed subspace identification, a rank of 15 was selected for the GI subspace matrix. All experiments were conducted on a GNU/Linux 6.5.0-15-generic x86\_64 system running Ubuntu 22.04.3 LTS as the operating system, ensuring a consistent computational environment across all benchmark tests.

### A.3 GP-based Benchmark Algorithms Implementation Details

We benchmark GIT-BO against four high-dimensional BO methods, including SAASBO [^16], TURBO [^15], HESBO [^22], and ALEBO [^13].

- SAASBO: The implementation is taken from [^38] (link: [https://github.com/pytorch/botorch/blob/main/tutorials/saasbo/saasbo.ipynb](https://github.com/pytorch/botorch/blob/main/tutorials/saasbo/saasbo.ipynb), license: MIT license, last accessed: May 1st, 2025)
- TURBO: The implementation is taken from [^38] (link: [https://github.com/pytorch/botorch/blob/main/tutorials/turbo\_1/turbo\_1.ipynb](https://github.com/pytorch/botorch/blob/main/tutorials/turbo_1/turbo_1.ipynb), license: MIT license, last accessed: May 1st, 2025)
- HESBO: The implementation is taken from the original implementation [^22] (link: [https://github.com/aminnayebi/HesBO](https://github.com/aminnayebi/HesBO), license: none, last accessed: May 1st, 2025)
- ALEBO: The implementation is taken from the original implementation [^13] (link: [https://github.com/facebookresearch/alebo](https://github.com/facebookresearch/alebo), license: CC BY-NC 4.0, last accessed: May 1st, 2025)

As most BO code is archived in the BoTorch or Ax Platform <sup>1</sup> library, we use Botorch 0.12.0 for SAASBO and TURBO. However, since Ax Platform has officially deprecated ALEBO, we have created another environment to run ALEBO using the previous version of Ax Platform and related packages that are compatible with the original code of ALEBO. <sup>2</sup> The environment setups are detailed in the provided code zip file.

### A.4 Benchmark Problems Implementaion Details

The source and license details of our benchmark problems are described in the following paragraphs.

#### Synthetic Problems:

The implementations for the nine synthetic functions are taken from Botorch [^38] (link: [https://github.com/pytorch/botorch/blob/main/botorch/test\_functions/synthetic.py](https://github.com/pytorch/botorch/blob/main/botorch/test_functions/synthetic.py), license: MIT license, last accessed: May 1st, 2025). The bounds of each problem are the default implementation in Botorch. Detailed equations for each problem can be found here: [https://www.sfu.ca/˜ssurjano/optimization.html](https://www.sfu.ca/~ssurjano/optimization.html).

#### CEC2020 Power System Problems:

We examine a subset of six problems, specifically those with design spaces exceeding 100 dimensions, from the CEC2020 test suite [^48] (link: [https://github.com/P-N-Suganthan/2020-Bound-Constrained-Opt-Benchmark](https://github.com/P-N-Suganthan/2020-Bound-Constrained-Opt-Benchmark), license: no licence, last accessed: May 1st, 2025). The code is initially in MATLAB, and we translate it into Python, running pytest to ensure the implementations are correct. While these problems incorporate equality constraints ($h_{j}(x)$), they are transformed into inequality constraints ($g_{j}(x)$) using the methodology outlined in the original paper [^48], as constraint handling is not the primary focus of this research. These transformed constraints are subsequently incorporated into the objective function $f(x)$ as penalty terms.

$$
g_{j}(x)=|h_{j}(x)|-\epsilon\leq 0\;,\;\epsilon=10^{-4}\;,\;j=1\sim C
$$
 
$$
f_{penalty}(x)=f(x)+\rho\sum^{C}_{j=1}max(0,g_{j}(x))
$$

We set a different $\rho$ penalty factor for each problem, respectively, to make the objective and constraint values have a similar effect on $f_{penalty}(x)$.

Table 1: CEC2020 Benchmark Problems Penalty Transform Factor $\rho$

| Problems | CEC34 | CEC35 | CEC36 | CEC37 | CEC38 | CEC39 |
| --- | --- | --- | --- | --- | --- | --- |
| $\rho$ | 0.01 | 0.0002 | 0.001 | 0.04 | 0.02 | 0.04 |

#### LassoBench Benchmarks:

The implementation is taken from the original implementation [^49] (link: [https://github.com/ksehic/LassoBench](https://github.com/ksehic/LassoBench), license: MIT and BSD-3-Clause license, last accessed: May 1st, 2025).

#### Rover:

The implementation is taken from [^50] (link: [https://github.com/zi-w/Ensemble-Bayesian-Optimization](https://github.com/zi-w/Ensemble-Bayesian-Optimization), license: MIT license, last accessed: May 1st, 2025).

#### MOPTA08 Car:

The MOPTA08 executables are taken from the paper [^23] ’s personal website (link: [https://leonard.papenmeier.io/2023/02/09/mopta08-executables.html](https://leonard.papenmeier.io/2023/02/09/mopta08-executables.html), license: no license, last accessed: May 1st, 2025). The MOPTA08 Car’s penalty transformation follows the formation of [^16] ’s supplementary material.

#### Mazda Bechmark Problems:

The implementation is taken from [^51] (link: [https://ladse.eng.isas.jaxa.jp/benchmark/](https://ladse.eng.isas.jaxa.jp/benchmark/), license: no license, last accessed: May 1st, 2025). The Mazda problem has two raw forms: a 4-objectives problem 148D (Mazda\_SCA) and a 5-objectives 222D problem (Mazda), and both of them have inequality constraints. For both problems, we equally weight each objective to form a single objective and perform a penalty transform:

$$
f_{multiobj\_penalty}(x)=\frac{1}{N}\sum^{N}_{i=1}f(x)+\rho\sum^{C}_{j=1}max(0,g_{j}(x))
$$

where $N$ is the number of objectives, $C$ is the number of inequality constraints, and we use $\rho=$ 10 for both variants of Mazda problem.

Table 2 summarizes the type of problems and their respective tested dimensions.

Table 2: High-Dimensional Benchmark Problems

| Problems | Source | Type | Dimension ($D$) Tested |
| --- | --- | --- | --- |
| Ackley | Botorch [^38] | Synthetic | 100, 200, 300, 400, 500 |
| Dixon-Price | Botorch [^38] | Synthetic | 100, 200, 300, 400, 500 |
| Griewank | Botorch [^38] | Synthetic | 100, 200, 300, 400, 500 |
| Levy | Botorch [^38] | Synthetic | 100, 200, 300, 400, 500 |
| Michalewicz | Botorch [^38] | Synthetic | 100, 200, 300, 400, 500 |
| Powell | Botorch [^38] | Synthetic | 100, 200, 300, 400, 500 |
| Rastrigin | Botorch [^38] | Synthetic | 100, 200, 300, 400, 500 |
| Rosenbrock | Botorch [^38] | Synthetic | 100, 200, 300, 400, 500 |
| Styblinski-Tang | Botorch [^38] | Synthetic | 100, 200, 300, 400, 500 |
| CEC34 | CEC2020 Benchmark Suite [^48] | Real-World | 118 |
| CEC35 | CEC2020 Benchmark Suite [^48] | Real-World | 153 |
| CEC36 | CEC2020 Benchmark Suite [^48] | Real-World | 158 |
| CEC37 | CEC2020 Benchmark Suite [^48] | Real-World | 126 |
| CEC38 | CEC2020 Benchmark Suite [^48] | Real-World | 126 |
| CEC39 | CEC2020 Benchmark Suite [^48] | Real-World | 126 |
| LassoSyntMedium | LassoBench [^49] | Synthetic | 100 |
| LassoSyntHigh | LassoBench [^49] | Synthetic | 300 |
| LassoDNA | LassoBench [^49] | Real-World | 180 |
| Rover | Previous BO studies [^50] [^12] | Real-World | 100, 200, 300, 400, 500 |
| MOPTA08 CAR | Previous BO studies [^15] [^23] | Real-World | 124 |
| MAZDA | Mazda Car Bechmark [^51] | Real-World | 222 |
| MAZDA SCA | Mazda Car Bechmark [^51] | Real-World | 148 |
| SVM | Previous BO studies [^16] [^12] [^23] | Real-World | 388 |

## Appendix B Parameter Sweep

We conducted a parameter sweep to evaluate the sensitivity of GIT-BO’s optimization performance to the chosen dimensionality ($r$) of the gradient-informed active subspace. The results are presented in Figure 6. The empirical findings suggest that very high-dimensional subspaces (e.g., $r=40$ for 100-dimensional problems) result in significant performance degradation due to diminished effectiveness of the gradient-informed search direction. Conversely, lower-dimensional subspaces generally showed superior performance. Although problem-specific hyperparameter optimization would likely enhance these results, we maintained fixed parameter values across ($r=15$) all benchmarks to ensure fair comparison, demonstrating that GIT-BO consistently achieves top-ranked performance on average.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2505.20685/assets/x6.png)

Figure 6: Performance comparison across various gradient-informed subspace dimensions ( r ) in GIT-BO. Median optimization results across 20 trials are shown, with shaded regions depicting 95% confidence intervals. High-dimensional subspaces (e.g., = 40 r=40 ) exhibit notable performance deterioration, and lower subspace dimensions tend to perform consistently better.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2505.20685/assets/x7.png)

Figure 7: Rank of the final optimization solution performance comparison across various gradient-informed subspace dimensions ( r ) in GIT-BO.

## Appendix C Full Results

We present the comprehensive optimization results for all benchmark problems investigated in the main paper. Figures 8 to 11 provide the complete convergence curves for synthetic and real-world benchmarks, respectively. The results consistently demonstrate GIT-BO’s robust performance across different classes of high-dimensional problems, clearly indicating its advantage in both convergence speed and final solution quality compared to baseline methods. Notably, in Figure 12, the statistical ranking of the variance in the final results for 20 trials shows that GIT-BO has a lower variance than other baselines, which means that GIT-BO converges faster in expectation and delivers reliably strong results without requiring multiple restarts, practically reducing computational cost and risk.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2505.20685/assets/x8.png)

Figure 8: Full convergence curves on engineering benchmarks illustrating the median best solutions obtained across multiple trials. Confidence intervals (95%) are shown as shaded areas around the median lines. The minimal median best solution among all algorithms at the final iteration is marked in “x”. The results demonstrate that SAASBO excels in the Rover and car (MOPTA08 and Mazdas) problems, and GIT-BO leads the CEC power system optimization problem. Though GIT-BO is second-best for real-world problems, engineers might benefit from its 100x faster runtime if time is a hard constraint.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2505.20685/assets/x9.png)

Figure 9: Violin plot statistical summary of final optimization performance for engineering functions, summarizing results over 20 trials.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2505.20685/assets/x10.png)

Figure 10: Detailed optimization results for synthetic benchmarks demonstrating the relative performance of GIT-BO versus competing algorithms. Median best-found function values are illustrated by solid lines, with shaded bands denoting 95% confidence intervals. The minimal median best solution among all algorithms at the final iteration is marked in “x”. Results highlight the robustness of GIT-BO in achieving superior performance across diverse synthetic problems, especially in challenging higher dimensional settings ( D > D> 300). The only two exceptions that GIT-BO fails dramatically in all s are Styblinski-Tang and Michalewicz, which can potentially be explained by the “No Free Lunch” theorem 52.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2505.20685/assets/x11.png)

Figure 11: Violin plot representations of final optimization performance for synthetic functions, summarizing results over 20 trials.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2505.20685/assets/x12.png)

Figure 12: Statistical ranking of the overall variance across (a) Real-World and (b) Synthetic benchmark problems. GIT-BO consistently ranks at the top, showing the lowest variance during optimization convergence.

[^1]: Ian Dewancker, Michael McCourt, and Scott Clark. Bayesian optimization for machine learning: A practical guidebook. arXiv preprint arXiv:1612.04858, 2016.

[^2]: Jasper Snoek, Hugo Larochelle, and Ryan P Adams. Practical bayesian optimization of machine learning algorithms. Advances in neural information processing systems, 25, 2012.

[^3]: Nilay Kumar, Mayesha Sahir Mim, Alexander Dowling, and Jeremiah J Zartman. Reverse engineering morphogenesis through bayesian optimization of physics-based models. npj Systems Biology and Applications, 10(1):49, 2024.

[^4]: Ke Wang and Alexander W Dowling. Bayesian optimization for chemical products and functional materials. Current Opinion in Chemical Engineering, 36:100728, 2022.

[^5]: Yichi Zhang, Daniel W Apley, and Wei Chen. Bayesian optimization for materials design with mixed quantitative and qualitative variables. Scientific reports, 10(1):4924, 2020.

[^6]: Rosen Yu, Cyril Picard, and Faez Ahmed. Fast and accurate bayesian optimization with pre-trained transformers for constrained engineering problems. Structural and Multidisciplinary Optimization, 68(3):66, 2025.

[^7]: Aaron Klein, Stefan Falkner, Simon Bartels, Philipp Hennig, and Frank Hutter. Fast bayesian optimization of machine learning hyperparameters on large datasets. In Artificial intelligence and statistics, pages 528–536. PMLR, 2017.

[^8]: Jia Wu, Xiu-Yun Chen, Hao Zhang, Li-Dong Xiong, Hang Lei, and Si-Hao Deng. Hyperparameter optimization for machine learning models based on bayesian optimization. Journal of Electronic Science and Technology, 17(1):26–40, 2019.

[^9]: Mohit Malu, Gautam Dasarathy, and Andreas Spanias. Bayesian optimization in high-dimensional spaces: A brief survey. In 2021 12th International Conference on Information, Intelligence, Systems & Applications (IISA), pages 1–8. IEEE, 2021.

[^10]: Xilu Wang, Yaochu Jin, Sebastian Schmitt, and Markus Olhofer. Recent advances in bayesian optimization. ACM Computing Surveys, 55(13s):1–36, 2023.

[^11]: Carl Hvarfner, Erik Orm Hellsten, and Luigi Nardi. Vanilla bayesian optimization performs great in high dimension. arXiv preprint arXiv:2402.02229, 2024.

[^12]: Zhitong Xu and Shandian Zhe. Standard gaussian process is all you need for high-dimensional bayesian optimization. arXiv preprint arXiv:2402.02746, 2024.

[^13]: Ben Letham, Roberto Calandra, Akshara Rai, and Eytan Bakshy. Re-examining linear embeddings for high-dimensional bayesian optimization. Advances in neural information processing systems, 33:1546–1558, 2020.

[^14]: Santu Rana, Cheng Li, Sunil Gupta, Vu Nguyen, and Svetha Venkatesh. High dimensional bayesian optimization with elastic gaussian process. In International conference on machine learning, pages 2883–2891. PMLR, 2017.

[^15]: David Eriksson, Michael Pearce, Jacob Gardner, Ryan D Turner, and Matthias Poloczek. Scalable global optimization via local bayesian optimization. Advances in neural information processing systems, 32, 2019.

[^16]: David Eriksson and Martin Jankowiak. High-dimensional bayesian optimization with sparse axis-aligned subspaces. In Uncertainty in Artificial Intelligence, pages 493–503. PMLR, 2021.

[^17]: Samuel Müller, Matthias Feurer, Noah Hollmann, and Frank Hutter. Pfns4bo: In-context learning for bayesian optimization. arXiv preprint arXiv:2305.17535, 2023.

[^18]: Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637(8045):319–326, 2025.

[^19]: Haitao Liu, Yew-Soon Ong, Xiaobo Shen, and Jianfei Cai. When gaussian process meets big data: A review of scalable gps. IEEE transactions on neural networks and learning systems, 31(11):4405–4423, 2020.

[^20]: Maria Laura Santoni, Elena Raponi, Renato De Leone, and Carola Doerr. Comparison of high-dimensional bayesian optimization algorithms on bbob. ACM Transactions on Evolutionary Learning, 4(3):1–33, 2024.

[^21]: Ziyu Wang, Frank Hutter, Masrour Zoghi, David Matheson, and Nando De Feitas. Bayesian optimization in a billion dimensions via random embeddings. Journal of Artificial Intelligence Research, 55:361–387, 2016.

[^22]: Amin Nayebi, Alexander Munteanu, and Matthias Poloczek. A framework for bayesian optimization in embedded subspaces. In International Conference on Machine Learning, pages 4752–4761. PMLR, 2019.

[^23]: Leonard Papenmeier, Luigi Nardi, and Matthias Poloczek. Increasing the scope as you learn: Adaptive bayesian optimization in nested subspaces. Advances in Neural Information Processing Systems, 35:11586–11601, 2022.

[^24]: Juliusz Krzysztof Ziomek and Haitham Bou Ammar. Are random decompositions all we need in high dimensional bayesian optimisation? In International Conference on Machine Learning, pages 43347–43368. PMLR, 2023.

[^25]: Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers can do bayesian inference. arXiv preprint arXiv:2112.10510, 2021.

[^26]: Min Cai, Yuchen Zhang, Shichang Zhang, Fan Yin, Dan Zhang, Difan Zou, Yisong Yue, and Ziniu Hu. Self-control of llm behaviors by compressing suffix gradient into prefix controller. arXiv preprint arXiv:2406.02721, 2024.

[^27]: Qi Deng, Shuaicheng Niu, Ronghao Zhang, Yaofo Chen, Runhao Zeng, Jian Chen, and Xiping Hu. Learning to generate gradients for test-time adaptation via test-time training layers. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 16235–16243, 2025.

[^28]: Mohamed Osama Ahmed, Bobak Shahriari, and Mark Schmidt. Do we need “harmless” bayesian optimization and “first-order” bayesian optimization. NIPS BayesOpt, 6:22, 2016.

[^29]: Jian Wu and Peter Frazier. The parallel knowledge gradient method for batch bayesian optimization. Advances in neural information processing systems, 29, 2016.

[^30]: Qiyu Wei, Haowei Wang, Zirui Cao, Songhao Wang, Richard Allmendinger, and Mauricio A Álvarez. Gradient-based sample selection for faster bayesian optimization. arXiv preprint arXiv:2504.07742, 2025.

[^31]: Matthew TC Li, Youssef Marzouk, and Olivier Zahm. Principal feature detection via $\phi$ -sobolev inequalities. Bernoulli, 30(4):2979–3003, 2024.

[^32]: Samuel Müller, Matthias Feurer, Noah Hollmann, and Frank Hutter. PFNs4BO: In-context learning for Bayesian optimization. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 25444–25470. PMLR, 23–29 Jul 2023.

[^33]: Olivier Zahm, Tiangang Cui, Kody Law, Alessio Spantini, and Youssef Marzouk. Certified dimension reduction in nonlinear bayesian inverse problems. Mathematics of Computation, 91(336):1789–1835, 2022.

[^34]: Matthew TC Li, Tiangang Cui, Fengyi Li, Youssef Marzouk, and Olivier Zahm. Sharp detection of low-dimensional structure in probability measures via dimensional logarithmic sobolev inequalities. arXiv preprint arXiv:2406.13036, 2024.

[^35]: Alexander Ly, Maarten Marsman, Josine Verhagen, Raoul PPP Grasman, and Eric-Jan Wagenmakers. A tutorial on fisher information. Journal of Mathematical Psychology, 80:40–55, 2017.

[^36]: Bobak Shahriari, Kevin Swersky, Ziyu Wang, Ryan P Adams, and Nando De Freitas. Taking the human out of the loop: A review of bayesian optimization. Proceedings of the IEEE, 104(1):148–175, 2015.

[^37]: James Bergstra and Yoshua Bengio. Random search for hyper-parameter optimization. The journal of machine learning research, 13(1):281–305, 2012.

[^38]: Maximilian Balandat, Brian Karrer, Daniel R. Jiang, Samuel Daulton, Benjamin Letham, Andrew Gordon Wilson, and Eytan Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. In Advances in Neural Information Processing Systems 33, 2020.

[^39]: Nikolaus Hansen, Anne Auger, Dimo Brockhoff, and Tea Tušar. Anytime performance assessment in blackbox optimization benchmarking. IEEE Transactions on Evolutionary Computation, 26(6):1293–1305, 2022.

[^40]: Hassan Ismail Fawaz, Germain Forestier, Jonathan Weber, Lhassane Idoumghar, and Pierre-Alain Muller. Deep learning for time series classification: a review. Data Mining and Knowledge Discovery, 33(4):917–963, 2019.

[^41]: Dimo Brockhoff and Tea Tušar. Gecco 2023 tutorial on benchmarking multiobjective optimizers 2.0. In Proceedings of the Companion Conference on Genetic and Evolutionary Computation, pages 1183–1212, 2023.

[^42]: Cyril Picard and Faez Ahmed. Untrained and unmatched: Fast and accurate zero-training classification for tabular engineering data. Journal of Mechanical Design, 146(9), 2024.

[^43]: Hongyan Wang, Hua Xu, and Zeqiu Zhang. High-dimensional multi-objective bayesian optimization with block coordinate updates: Case studies in intelligent transportation system. IEEE Transactions on Intelligent Transportation Systems, 25(1):884–895, 2023.

[^44]: James R Deneault, Jorge Chang, Jay Myung, Daylond Hooper, Andrew Armstrong, Mark Pitt, and Benji Maruyama. Toward autonomous additive manufacturing: Bayesian optimization on a 3d printer. MRS Bulletin, 46:566–575, 2021.

[^45]: Rémi Lam, Matthias Poloczek, Peter Frazier, and Karen E Willcox. Advances in bayesian optimization with applications in aerospace engineering. In 2018 AIAA Non-Deterministic Approaches Conference, page 1656, 2018.

[^46]: Ryan-Rhys Griffiths and José Miguel Hernández-Lobato. Constrained bayesian optimization for automatic chemical design using variational autoencoders. Chemical science, 11(2):577–586, 2020.

[^47]: Benjamin J Shields, Jason Stevens, Jun Li, Marvin Parasram, Farhan Damani, Jesus I Martinez Alvarado, Jacob M Janey, Ryan P Adams, and Abigail G Doyle. Bayesian reaction optimization as a tool for chemical synthesis. Nature, 590(7844):89–96, 2021.

[^48]: Abhishek Kumar, Guohua Wu, Mostafa Z Ali, Rammohan Mallipeddi, Ponnuthurai Nagaratnam Suganthan, and Swagatam Das. A test-suite of non-convex constrained optimization problems from the real-world and some baseline results. Swarm and Evolutionary Computation, 56:100693, 2020.

[^49]: Kenan Šehić, Alexandre Gramfort, Joseph Salmon, and Luigi Nardi. Lassobench: A high-dimensional hyperparameter optimization benchmark suite for lasso. In International Conference on Automated Machine Learning, pages 2–1. PMLR, 2022.

[^50]: Zi Wang, Clement Gehring, Pushmeet Kohli, and Stefanie Jegelka. Batched large-scale bayesian optimization in high-dimensional spaces. In International Conference on Artificial Intelligence and Statistics, pages 745–754. PMLR, 2018.

[^51]: Takehisa Kohira, Hiromasa Kemmotsu, Oyama Akira, and Tomoaki Tatsukawa. Proposal of benchmark problem based on real-world car structure design optimization. In Proceedings of the Genetic and Evolutionary Computation Conference Companion, pages 183–184, 2018.

[^52]: David H Wolpert and William G Macready. No free lunch theorems for optimization. IEEE transactions on evolutionary computation, 1(1):67–82, 1997.