---
title: "TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models"
source: "https://ar5iv.labs.arxiv.org/html/2511.08667"
author:
published:
created: 2026-05-30
description: "We present TabPFN, a trained Transformer that can do supervised classification for small tabular datasets in less than a second, needs no hyperparameter tuning and is competitive with state-of-the-art classification methods."
tags:
  - "clippings"
---
![[Uncaptioned image]](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/figures/prior-logo.png)

## Abstract

The first tabular foundation model, TabPFN, and its successor TabPFNv2 have impacted tabular AI substantially, with dozens of methods building on it and hundreds of applications across different use cases. This report introduces TabPFN-2.5, the next generation of our tabular foundation model, built for datasets with up to 50,000 data points and 2,000 features, a 20× increase in data cells compared to TabPFNv2. TabPFN-2.5 is now the leading method for the industry standard benchmark TabArena (which contains datasets with up to 100,000 training data points), substantially outperforming tuned tree-based models and matching the accuracy of AutoGluon 1.4, a complex four-hour tuned ensemble that even includes the previous TabPFNv2. Remarkably, default TabPFN-2.5 has a 100% win rate against default XGBoost on small to medium-sized classification datasets (≤10,000 data points, 500 features) and a 87% win rate on larger datasets up to 100K samples and 2K features (85% for regression). For production use cases, we introduce a new distillation engine that converts TabPFN-2.5 into a compact MLP or tree ensemble, preserving most of its accuracy while delivering orders-ofmagnitude lower latency and plug-and-play deployment. This new release will immediately strengthen the performance of the many applications and methods already built on the TabPFN ecosystem.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x1.png)

Figure 1: TabPFN-2.5 performance on the standard TabArena-lite benchmark \[ erickson2025tabarena \], TabPFNv2 classification subset. TabPFN-2.5 outperforms any other model in a forward pass, and marks a strong leap from TabPFNv2. When fine-tuned on real data, Real-TabPFN-2.5 shows even stronger performance. The horizontal dotted line stands for AutoGluon 1.4 extreme mode tuned for 4 hours, an ensemble of models including TabPFNv2.

## 1 Introduction

Tabular data is ubiquitous, forming the backbone of decision-making in countless domains, from finance to healthcare. For decades, traditional tabular machine learning—built on gradient-boosted trees \[chen2016xgboost, prokhorenkova2018catboost, lightgbm\], random forests \[breiman2001random\], and linear or additive models—has been the workhorse of applied data science. Yet these methods remain limited: they require extensive dataset-specific tuning, often provide uncalibrated or unreliable uncertainty estimates without significant modification, and lack the generalization and transferability of modern foundation models.

Tabular foundation models (TFMs) offer a new paradigm. They address these limitations by pretraining on large synthetic distributions of tabular tasks and performing inference via in-context learning instead of gradient descent. They are training-free predictors meta-trained to yield strong calibration, without the need for time-consuming and labor-intensive hyperparameter tuning necessary for gradient-boosted trees. Their strong generalization makes them particularly attractive for data-scarce domains.

Our initial release, TabPFNv1 \[hollmann2022tabpfnv1\] served as a proof-of-concept that a transformer could learn a Bayesian-like inference algorithm, though it was limited to small (up to 1k samples), clean, numerical-only data. Our successor, TabPFNv2 \[Hollmann2025tabpfnv2\], scaled this idea into a practical model for datasets up to 10,000 samples. TabPFNv2 handles the messy and heterogeneous data seen in the real world—including categorical features, missing values, and outliers.

This paper describes the next release of TabPFN: TabPFN-2.5. Our key contributions are:

- SOTA Performance: In a forward pass, TabPFN-2.5 outperforms tuned tree-based models (like XGBoost and CatBoost) and matches the accuracy of AutoGluon 1.4 tuned for 4 hours—a complex ensemble that includes all previous methods, even TabPFNv2.
- Improved Scalability: We scale the power of in-context learning to datasets of up to 50,000 samples (5x increase over TabPFNv2) and 2,000 features (4x increase), making TFMs viable for a much wider range of real-world problems <sup>1</sup>. While TabPFN-2.5 was designed for up to 50,000 rows, we note that this limit is not strict and report strong results on benchmarks with up to 100,000 training samples.
- Fast Inference: We dramatically improve inference speed. We introduce TabPFN-as-MLP/TreeEns, a proprietary output engine, that yields an MLP or tree ensemble, combining most of TabPFN’s accuracy with the low-latency inference and easy deployment of MLPs and tree ensembles.

We begin by surveying the growing ecosystem of TabPFN applications and extensions (Section 2). We then describe our methodological advances (Section 3) and present the experimental results (Section 4). We then discuss how to get the best speed out of TabPFN on common hardware (Section 5) as well as our non-commercial open-source license (Section 6). We conclude by discussing the remaining limitations and opportunities for future work (Sections 7). For installation and usage examples, see the online documentation at [https://docs.priorlabs.ai/](https://docs.priorlabs.ai/).

## 2 Ecosystem & Adoption

We now discuss community adoption, methods built on top of TabPFN, and our TabPFN extensions.

### 2.1 Community Adoption

Since its release, TabPFNv2 has become a widely used baseline for tabular ML. The Nature paper \[Hollmann2025tabpfnv2\] has been cited in almost 400 papers within 10 months of its publication, and the open-source package has surpassed 2,000,000 downloads on PyPI <sup>2</sup>. Adoption spans both research and production, especially in settings with sparse data or frequent retraining requirements. This widespread adoption has matured TabPFN from a research model into a stable product. With feedback from our community of nearly 1,500 users on Discord, and hundreds of closed GitHub issues, we have shipped numerous stability fixes, and cross-platform device compatibility. In addition to commercial use-cases, we also collected 100 published use cases across a broad range of areas (please see Appendix B for a detailed list):

- Healthcare and Life Sciences. Adoption is strongest in healthcare (50+ published applications), driven by TabPFN’s exceptional performance in data-scarce settings—a common challenge in medicine. Use cases span oncology, neurology, cardiology, and pharmacology, powering applications like diagnosis, prognosis, and treatment response prediction from complex multimodal (clinical, imaging, omics) data.
- Financial Services, Banking, and Insurance. While we see strong commercial traction, public-facing use cases are rare due to the competitive, private nature of this industry (3 collected). Applications in this domain typically involve proprietary forecasting, uplift modeling, and risk assessments.
- Energy and Utilities. We’ve identified 15 published cases centered on complex forecasting and optimization. Key applications include environmental forecasting (algal blooms, wildfire risk), renewable-energy nowcasting, and process/asset optimization across water, oil & gas.
- Manufacturing and Industrial. The 12 diverse published use cases in this area highlight TabPFN’s flexibility. Applications include anomaly detection in IIoT security, predictive maintenance for rotating machinery, physics-aware optimization for battery thermal modeling, and semiconductor test optimization.
- Other Industries 19 further applications demonstrate broad utility, spanning geoscience, agriculture, materials, and engineering. These range from microbiome classification and lunar regolith analysis to soil property modeling, fuel-blend optimization and crop yield forecasting.

### 2.2 A Foundational Layer for New Research

Beyond direct application, TabPFN now serves as a foundational layer for new research domains. Its ability to act as a powerful, pre-trained “algorithm-in-a-box” has unlocked new approaches to complex problems. We expect TabPFN-2.5 to directly boost performance in all these areas:

- Time Series Forecasting: TabPFN-TS \[hoo2024tabpfn\_ts\] extends TabPFN to time-series forecasting by incorporating temporal context into its in-context learning mechanism, outperforming specialized time-series models without any retraining.
- Node Classification in Graphs: Various works \[Hayler2025GraphsTablesZeroShot, eremeev2025turningtabularfoundationmodels\] represent graph nodes as tabular instances with relational and structural features, directly using tabular foundation models like TabPFN to solve the problem.
- Data Streams: TabPFNv2 was used for in-context learning on Evolving Data Streams \[Lourenco2025ICLStreams\]. TabPFN can *adapt to non-stationary data streams* online, without retraining, enabling continual learning in evolving environments.
- Reinforcement Learning: TabPFNv2 was used to replace gradient-based policy optimization with in-context optimization over trajectories, creating a powerful general-purpose optimizer for RL tasks \[Schiff2025TabPFNRL\].
- Bayesian optimization: GIT-BO \[Yu2025GITBO\] uses TabPFNv2 inside of high-dimensional Bayesian Optimization, as it enables efficient search in high-dimensional and heterogeneous design spaces.
- Multimodal Learning & Encoding: TabPFN is used to integrate tabular data with other modalities. It can serve as a *frozen tabular encoder* to generate robust embeddings for combination with data like images (e.g., in the TIME framework \[luo2025timetabpfnintegratedmultimodalengine\]), or handle modalities in a unified manner by adding modality-specific projectors \[Lourenco2025ICLStreams\].
- Causal Inference: Do-PFN \[robertson\_dopfn\], CausalPFN \[balazadeh\_causalpfn\], and CausalFM \[feuerriegel\_causalfm\] pre-train PFNs to predict interventional outcomes, and show strong performance in estimating causal effects.

### 2.3 The TabPFN-Extensions Ecosystem

We maintain the TabPFN–Extensions repository ([https://github.com/PriorLabs/tabpfn-extensions](https://github.com/PriorLabs/tabpfn-extensions)), which offers extensions around the core model, developed together with a growing community around TabPFN. These extensions leverage TabPFN capabilities for:

- Interpretability. SHAP values, feature selection, partial dependence.
- Unsupervised Tasks. Data generation, augmentation, outlier detection.
- Advanced Modeling. Many-class classification, regression-via-classifier.
- Performance & Integration. Lightweight HPO, ensembling, and integration with tree/forest baselines.

Figure 11 in the appendix provides a minimal workflow to help users pick the right components for their task.

## 3 Model Overview

TabPFN-2.5 follows the same general design as TabPFNv2 but introduces deeper architectures, richer synthetic priors, and new calibration and inference modules. We summarize only the key changes here.

#### Data.

We improved our prior data generation substantially, broadened the set of distributions and scaled up to more data points and more features, while keeping the prediction tasks difficult. Like the original TabPFNv2, TabPFN-2.5 is trained purely on synthetically generated data. We also release a version that is fine-tuned on real data following Real-TabPFN \[garg2025realtabpfn\]. It is trained on a curated corpus of 43 real-world tabular datasets sourced from OpenML and Kaggle, deduplicated against all internal benchmarks and the full TabArena suite. We refer to this version as Real-TabPFN-2.5, and report strong improvement in Figures 3 and 4. See Appendix C for details on training and deduplication.

#### Architecture.

We follow the alternating-attention transformer design of TabPFNv2, which attends across both data points and features to achieve permutation invariance, but introduces some changes:

- We increase the network depth from 12 to 18 layers for our regression model and 24 layers for our classification model.
- We simultaneously increase the feature group size (the number of features being embedded together), which allows for faster training and inference. We use a group size of 3 for TabPFN-2.5, compared to 2 for TabPFNv2.
- For our regression models, we found a small improvement by replacing the linear encoder used in TabPFNv2 by a 2-layer MLP.
- Finally, we add 64 additional “thinking” rows to the input dataset of TabPFN-2.5, which are learned during pretraining. Inspired by results from the LLM literature \[MerrillSabharwal2025\_ExactExpressivePower, GoyalEtAl2024\_ThinkBeforeYouSpeak\], these rows give additional computational capacity to the model and can also act as attention sinks to help the model ignore other rows \[DarcetEtAl2024\_VisionTransformersNeedRegisters\].

Other core components from TabPFNv2—feature/sample dual attention, caching separation of training/test context, and positional feature embeddings—remain unchanged.

#### Preprocessing.

We aggregate predictions across multiple dataset permutations and feature transformations to enhance robustness and generalization. In the updated TabPFN-2.5 configuration, additional feature transformations are introduced to enhance robustness against outlier-prone feature distributions and to increase the diversity among the individual estimators. Specifically, we combine robust scaling and soft clipping (following \[holzmuller2024realmlp\]) with quantile transformations and standard scaling to balance stability and sensitivity across features. Following TabPFNv2, we also include singular value decomposition (SVD) components as additional features in some of the estimators, capturing high-energy directions of variance that provide complementary global structure information.

#### Hyperparameter Tuning of TabPFN with TabPFN.

TabPFN’s hyperparameter space spans architectural, training, and prior-data parameters, making exhaustive grid search computationally infeasible. To explore this space efficiently, we adopted a surrogate-based optimization strategy.

We first trained $\approx 100$ models on a broad but sparse grid of hyperparameter configurations drawn from plausible prior ranges and evaluated them on a curated in-house validation suite, producing a compact set of hyperparameter–performance pairs.

With $\sim 50$ hyperparameters and only $100$ datapoints, direct interpolation was prone to overfitting. We therefore used a regression model well-suited for data-scarce structured prediction—our previous TabPFNv2 model—as a surrogate to predict validation performance over a denser grid of $10{,}000$ configurations. This self-referential “TabPFN-tunes-TabPFN” strategy efficiently surfaced promising regions of the search space for full, compute-intensive training runs.

#### Tuning custom metrics.

TabPFN-2.5 adds new post-processing capabilities that enhance both calibration and metric-specific optimization. Our framework now supports tuning the classifier’s decision threshold, enabling direct optimization of metrics beyond accuracy—such as the F1-score—by adjusting the operating point to the desired trade-off between precision and recall. For multiclass classification, it allows to apply temperature scaling to the final softmax outputs to improve probability calibration. This threshold tuning procedure can yield substantial performance improvements (see Appendix H). Unless otherwise noted, however, all classification results in this report are computed using uncalibrated, default scores, without temperature scaling or threshold tuning.

#### Reducing inference costs.

Despite being a larger model than TabPFNv2, TabPFN-2.5 is between 1x and 2.3x faster thanks to optimized preprocessing and larger feature groups, as shown in Figure 20. This allows TabPFN-2.5 to scales inference to datasets with up to $50{,}000$ rows and 2,000 features. Furthermore, we found large speed gain in this adoption of FlashAttention-3 \[shah2024fa3\] and parallel evaluation across multiple GPUs.

#### Creating fast, deployable models.

To improve deployment flexibility, we developed a proprietary distillation engine that, given a training data set, outputs a multi-layer perceptron (TabPFN-2.5-as-MLP) or tree ensemble classifier (TabPFN-2.5-as-TreeEns) whose performance is close to the one of TabPFN on this dataset (see Figure 7). In contrast to TabPFN, this resulting MLP or tree ensemble classifier is dataset-specific, does not perform in-context learning, takes as input a single data point, and has very low latency and memory footprint for making predictions. It can also be seamlessly integrated into existing production pipelines, including those constrained by latency, interpretability, or regulatory requirements that hinder a change in the class of models being deployed. This increases TabPFN-2.5’s practical use in real-world decision systems. Other types of models could easily be supported.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x2.png)

Table 1: Summary of TabPFN model variants. Max Rows and Features are the recommended maximum sizes. The models also fit larger datasets but are not built and evaluated for these settings.

## 4 Experimental Results

We first demonstrate state-of-the-art performance on the industry standard benchmark TabArena and using our own benchmarking framework. Then, we report our advances to reduce inference latency. Finally, we demonstrate that TabPFN-2.5 yields new state-of-the-art performance for causal machine learning.

### 4.1 Performance on the Industry Standard Benchmark TabArena

TabArena \[erickson2025tabarena\] is the most curated tabular benchmark, based on the largest number of candidate datasets considered, and created by open-source contributors from a wide range of institutions. It will appear at the NeurIPS 2025 Datasets & Benchmarks track and is thus most up-to-date. We follow the paper’s recommendation to benchmark on “TabArena-Lite”, which is a cheaper but representative version of the full benchmark using only one test fold. The benchmark contains a set of 51 datasets selected from 1053 to be representative of real-world tabular data. See erickson2025tabarena for the list of datasets.

#### Pushing the limit on small to medium-sized datasets.

Figure 3 shows results for TabPFN-2.5 on TabArena-Lite with up to 10,000 data points and 500 features, demonstrating that TabPFN-2.5, in a forward pass, outperforms the wide range of existing tabular prediction methods. On classification, TabPFN-2.5 in a forward pass outperforms AutoGluon 1.4, an ensemble tuned for four hours and including best other methods (even TabPFNv2). Using our Real-TabPFN-2.5 variant fine-tuned on real datasets (deduplicated from TabArena datasets) widens the lead even further. On the other hand, our regression model benefits much more from tuning and outperforms AutoGluon 1.4 after being tuned for 60 configurations.

#### Scaling to larger datasets.

Figure 4 shows a similar experiment on all the TabArena datasets, with up to 100,000 data points and 2,000 features, clearly ranking TabPFN-2.5 as the best default model, and outperforming (for regression datasets) or approaching (for classification datasets) AutoGluon 1.4 (tuned for 4 hours) when tuned. Again, we highlight the very strong default performance of Real-TabPFN-2.5 on these larger classification datasets, beating in one forward pass any other tuned and ensembled model.

#### A significant improvement upon TabPFNv2.

Comparing the default performance of TabPFN-2.5 and TabPFNv2, we see a big leap in performance in Figure 3. In addition, looking at performance on each dataset in TabArena (TabPFNv2 compatible subset) in Figure 2, we see that TabPFN-2.5 clearly outperforms TabPFNv2 on almost all datasets, and is never much worse. In Appendix G, we detail the results on TabArena-Lite, showing the pairwise win rates of the different models, and comparing TabPFN-2.5 to other foundation models like TabICL \[qu2025tabicl\] or LimiX \[zhang2025limix\].

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x3.png)

Figure 3: TabArena-Lite results on classification (left) and regression (right), restricted to datasets with less than 10K training samples and 500 features. Note that tuning for TabPFN-2.5 is only based on 60 random configs compared to 200 for the baselines. The vertical dotted line stands for AutoGluon 1.4 extreme mode tuned for 4 hours, an ensemble of models including TabPFNv2 \[ autogluon\_tabular \].

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x5.png)

Figure 4: TabArena-Lite results on classification (left) and regression (right), evaluated on all datasets, going up to 100K training rows and 2K features. Note that tuning for TabPFN-2.5 is only based on 60 random configs compared to 200 for the baselines, and that we removed the "dt-pfn" option from our tuning search space for the 4 largest datasets in the benchmark to reduce the tuning time. The vertical dotted line stands for AutoGluon 1.4 extreme mode tuned for 4 hours, an ensemble of models including TabPFNv2 \[ autogluon\_tabular \].

### 4.2 Performance on Internal Benchmarks

#### A diverse internal benchmark.

In addition to the public TabArena benchmark, we built our own benchmarking framework using proprietary data. It includes over 100 use cases from healthcare, finance, insurance, retail and manufacturing. This benchmark focuses on comparing to gradient-boosted decision tree libraries that are frequently used in industry (XGBoost \[chen2016xgboost\], CatBoost \[prokhorenkova2018catboost\], LightGBM \[lightgbm\]), both in their default version and tuned for one hour. In all cases, we show the results of three standard gradient-boosted tree libraries (LightGBM, XGBoost and CatBoost). We tune all of the baselines for 1hr, using random search on the established search spaces from \[Hollmann2025tabpfnv2\]. TabPFN is tuned using our AutoTabPFN system, resulting in a tuned and ensembled model.

#### TabPFN-2.5 shows strong results up to 50,000 samples and 2,000 features.

Figure 5 and Figure 6 show results on our internal benchmark for classification and regression datasets with up to 50,000 data points and 500 features. We can see on these figures that TabPFN outperforms in one forward pass all our tuned baselines. In Section F, we also show strong results on datasets with 500 to 2,000 features, and provide more details on how we normalize the performance of each model across datasets.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x7.png)

Figure 5: Results from our internal benchmark on classification datasets with up to 50,000 data points. More details on the normalization is available in Appendix F. In the scatter plots (right), each point represents a different dataset from our internal benchmark, and the axes measure the normalized performance of TabPFN-2.5 and CatBoost (either default or tuned for 1 hour) on this dataset.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x11.png)

Figure 6: Results from our internal benchmark on regression datasets with up to 50,000 data points. More details on the normalization is available in Appendix F. In the scatter plots (right), each point represent a different dataset from our internal benchmark, and the axis measure the normalized performance of TabPFN-2.5 and CatBoost (either default or tuned for 1 hour) on this dataset

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x15.png)

Figure 7: TabPFN-as-MLP still outperforms tree-based models while having much faster inference speed than TabPFN. For baseline, light blue represents performance when tuned for 1 hour, and darker blue default performance. For TabPFN, we report default performance.

### 4.3 Measuring TabPFN-2.5 Training and Inference Speed

Figure 8 shows how TabPFN-2.5 classification speed scales with training set size, when using one or four GPUs, as we vary the number of rows and columns in the dataset. The time measured includes both the time to process the training rows (equivalent to the combination of “training” a classical ML model) and “prediction” time on test rows. We can observe the expected scaling in $\mathcal{O}(r^{2}\min(c,500)+r\min(c,500)^{2})$, where $r$ is the number of rows and $c$ is the number of columns, due to dual attention over rows and capped per-estimator feature subsampling at 500 features. Section 5 contains results for regression, performance on common models of GPU, for reference, and a measurement of the speedup from TabPFNv2. The inference speed reported here reflects the latency of the full in-context learning model.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x16.png)

(a) one H100 GPU

### 4.4 Fast Inference with TabPFN-2.5-as-MLP

We benchmark TabPFN-2.5-as-MLP against tuned LightGBM, XGBoost, and CatBoost models, as well as the standard TabPFN-2.5 model, on our curated collection of internal open source datasets with less than 10k data points. Figure 7 illustrates representative test-split performance. Empirically, TabPFN-2.5-as-MLP offers competitive accuracy while reducing inference cost, making it attractive for high-throughput or resource-constrained deployment scenarios.

### 4.5 TabPFN for Causal Inference

#### RealCause Benchmark.

To systematically evaluate TabPFN’s potential as a causal estimator, we leverage the RealCause benchmark \[neal\_realcause\], a semi-synthetic benchmark which begins with real-world randomized control trial (RCT) data and synthetically creates observable confounding effects.<sup>3</sup> We measure the Precision in Estimating Heterogeneous Effects (PEHE), which corresponds to the root-mean-squared error between predicted and RealCause’s ground-truth CATE values <sup>4</sup>. In Figure 10, we show that PFN-based methods for CATE-estimation dominate the leaderboard, occupying the first seven positions. TabPFN-2.5 applied as a T-Learner, a simple two-model approach that fits a separate model to the treatment and control observations, achieves the strongest overall performance, outperforming specialized tree- and deep-learning-based methods \[wager\_causal\_forest\]. We also observe in Figure 10 that for each of our three meta-learners, TabPFN-2.5 performs better out-of-the-box than TabPFNv2 and HPO <sup>5</sup>. This result shows that improvements in base model predictive performance transfer to the problem of causal inference.

#### Foundation Models for Causal Inference.

While we show strong results in unconfounded settings, real-world causal inference often involves imperfect data and latent confounders. A growing line of work aims to pre-train PFNs explicitly for causal reasoning—for example, predicting interventional outcomes or learning causal structures directly \[balazadeh\_causalpfn, dhir\_interv, feuerriegel\_causalfm, robertson\_dopfn, sauter\_activa\]. We view this as one of the most exciting frontiers for foundation models: extending TabPFN’s reasoning from predicting what is to inferring what would happen if, and ultimately, understanding why.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/figures/realcause_pretty.png)

Figure 9: PFN-based CATE estimators dominate RealCause, outperforming specialized tree- and deep-learning-based methods for causal inference. Choice of propensity and outcome model is important for CATE estimation.

## 5 How to Get Optimal Fit + Predict Speed from TabPFN-2.5

To achieve good performance, we recommend the following:

- Use a dedicated GPU or GPUs: We recommend NVIDIA H100 or A100 GPUs. Any dedicated GPU supported by PyTorch is compatible, but some models may not have enough memory for larger datasets or perform slowly. Integrated GPUs, MPS (Apple Silicon), and CPUs are also supported, but are only suitable for small datasets.
- Use multiple GPUs: For larger datasets, fit + predict time can be dramatically reduced by parallelizing inference over several GPUs. To enable this, set the device parameter of TabPFNClassifier and TabPFNRegressor.
- Use batch inference: Unless the fitted-model cache is enabled (see below), the model is retrained each time.predict() is called. This means that it is much faster to make a prediction for all your test points in a single.predict() call. If you run out of memory, split the test points into batches of $1000$ to $10000$ and call.predict() for each batch.
- Use PyTorch 2.8 or above: TabPFN-2.5 also supports earlier versions of PyTorch, but these may have lower performance.
- For small datasets, enable the fitted-model cache: This is an experimental feature that trains and stores the model during.fit(), making subsequent.predict() calls fast by using a KV-Cache. It is enabled by setting the fit\_mode parameter of TabPFNClassifier and TabPFNRegressor to fit\_with\_cache. However, with this setting classification models will consume approximately 6.1 KB of GPU memory and 48.8 KB of CPU memory per cell in the training dataset (regression models about $25$ % less), thus it is currently only suitable for small training datasets. For larger datasets and CPU-based inference, we recommend the TabPFN-as-MLP/Tree output engine.
- If speed is important for your application, you may consider optimizing the memory\_saving\_mode and n\_preprocessing\_jobs parameters of TabPFNClassifier and TabPFNRegressor. See the code documentation for further information.

Figure 18 in the appendix shows the inference latency you can expect for three common models of GPU, when using one or four GPUs. It also shows the maximum dataset size that fits in memory for each GPU.

## 6 License and Availability

We release TabPFN-2.5 under our TABPFN-2.5 License v1.0 designed to be permissive for research and internal evaluation. It explicitly allows testing, evaluation, and internal benchmarking, so an organization can download the model and run preliminary assessments on its own datasets.

The key restriction is that the model, its derivatives, and its outputs cannot be used for any commercial or production purpose. This includes, but is not limited to, revenue-generating products, competitive benchmarking for procurement, client deliverables, or using the model’s results for internal commercial decision-making.

For production use cases, we offer a Commercial Enterprise License. This provides access to our proprietary high-speed inference engine, dedicated support, integration tooling, and other internal models.

Please contact us at sales@priorlabs.ai for commercial licensing inquiries. The full non-commercial mode license text can be found at [https://huggingface.co/Prior-Labs/tabpfn\_2\_5/blob/main/LICENSE](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/LICENSE).

### 6.1 Cloud API

We provide a managed TabPFN-2.5 cloud endpoint, which runs on our optimized GPU infrastructure. This is the recommended option for users who do not have a dedicated local GPU or for those who wish to use TabPFN commercially without purchasing a full on-premise license.

The API is accessible via a simple Python SDK <sup>6</sup> (pip install tabpfn-client) or a standard REST API, allowing for integration into both non-commercial and commercial applications.

## 7 Conclusion and The Road Ahead

We are excited about this release. Taken together, our experiments on public (TabArena) and private benchmarks demonstrate that TabPFN-2.5 sets a new state-of-the-art for tuning-free tabular models. Built for datasets up to 50,000 rows and 2,000 features, TabPFN-2.5 matches the performance of complex 4-hour-tuned ensembles - ensembles that even include our previous TabPFNv2 - and in a forward pass outperforms any other tuned model on the unrestricted public TabArena benchmark (which contains datasets with up to 100,000 training data points).

The next step is scaling to datasets with millions of rows. We are actively developing new techniques - including retrieval, fine-tuning, and novel architectures - and anticipate that systems based on Tabular Foundation Models (TFMs) will define state-of-the-art performance for datasets with millions of data points very soon.

Our broader vision beyond this release is to tackle the entire stack of problems with tabular-like data, including time series, multimodal tabular data, causal inference, unsupervised tasks, integration of domain knowledge and decision support, ultimately building the core intelligence engine for reasoning over structured and multimodal data.

## Appendix A Contributors

### Model dev & Deployment

Léo Grinsztajn, Klemens Flöge, Oscar Key, Felix Birkel, Brendan Roof, Phil Jund, Benjamin Jäger, Adrian Hayler, Dominik Safaric, Simone Alessi, Felix Jablonski, Mihir Manium, Rosen Yu, Anurag Garg, Jake Robertson, Shi Bin (Liam) Hoo, Vladyslav Moroshan, Magnus Bühler, Lennart Purucker, Bernhard Schölkopf, Noah Hollmann, Frank Hutter

### Distribution & Product

Clara Cornu, Lilly Charlotte Wehrhahn, Alessandro Bonetto, Sauraj Gambhir

## Appendix B TabPFN Use Case Overview

TabPFNv2 has been applied to a broad set of use cases. We now list 100 published use cases across different industries.

## Healthcare and Life Sciences

We collected 51 published TabPFN use cases in this area, by far more than in any other area; we attribute this partly to the scarcity of data in healthcare and life sciences, and partly to the open publishing culture in this area. Use cases span oncology, neurology, cardiology, psychiatry, nephrology, and pharmacology. Applications include diagnosis, prognosis, and treatment response prediction from multimodal clinical, imaging, and omics data, often under severe data scarcity.

1. TabPFN was applied to distinguish cancer patients from healthy individuals using immune system profiles from peripheral blood, facilitating predictions of immunotherapy responses \[hc\_usecase1\_bostongene\_tabpfn\]. [Link](https://www.linkedin.com/pulse/how-bostongene-utilized-tabpfn-identify-immune-system-profiles-vexle/)
2. A machine learning model employing TabPFN was developed for non-invasive diagnostic prediction of minimal change disease in patients with nephrotic syndrome, utilizing clinical biomarkers \[hc\_usecase2\_mcd\_scirep\]. [Link](https://www.nature.com/articles/s41598-024-73898-4)
3. TabPFN was integrated into a system for analyzing T-cell receptor repertoires combined with clinical biomarkers to forecast immunotherapy outcomes in cancer patients, as explored by researchers at BostonGene \[hc\_usecase3\_immunotypes\_cancercell\]. [Link](https://www.cell.com/cancer-cell/fulltext/S1535-6108\(24\)00132-6)
4. TabPFN enabled early detection of stillbirth risks through analysis of cardiotocography data, supporting improved prenatal care \[hc\_usecase4\_stillbirth\_slas\]. [Link](https://www.sciencedirect.com/science/article/pii/S2472630324000852)
5. Predictive modeling for postoperative outcomes following anterior cervical corpectomy utilized TabPFN to assess patient demographics and surgical parameters \[hc\_usecase5\_acc\_asj\]. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC11366553/)
6. A hybrid model incorporating TabPFN was introduced to predict dementia progression in Parkinson’s disease patients, handling small datasets and missing values effectively \[hc\_usecase6\_pd\_dementia\_lightgbm\_tabpfn\]. [Link](https://journals.sagepub.com/doi/full/10.1177/20552076241272585)
7. A machine learning model based on TabPFN was developed to predict 90-day unfavorable outcomes in stroke patients with distal vessel occlusions using CT perfusion imaging \[hc\_usecase7\_dmvo\_ajnr\]. [Link](https://www.ajnr.org/content/early/2024/10/28/ajnr.A8547.abstract)
8. TabPFN was utilized in chemoproteomics for identifying small-molecule fragment-protein interactions, aiding ligand discovery in drug development \[hc\_usecase8\_chemoproteomics\_science\]. [Link](https://www.science.org/doi/abs/10.1126/science.adk5864)
9. TabPFN facilitated the prediction of non-invasive ventilation outcomes in patients with acute hypoxemic respiratory failure, supporting early identification of treatment failures \[hc\_usecase9\_niv\_tabpfn\]. [Link](https://www.researchgate.net/profile/Antonio-Esquinas/publication/393595503_Early-prediction-of-non-invasive_ventilation_outcome_using_the_TabPFN_machine_learning_model_a_multi-centre_validation_study/links/68718bc56e247f362b18c4b8/Early-prediction-of-non-invasive-ventilation-outcome-using-the-TabPFN-machine-learning-model-a-multi-centre-validation-study.pdf)
10. An interpretable Transformer-based model leveraging TabPFN was created to predict intravenous immunoglobulin resistance in pediatric patients with Kawasaki disease \[hc\_usecase10\_kawasaki\_tabpfnv2\]. [Link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327564)
11. TabPFN was employed in visual representation techniques for prostate cancer diagnosis, converting clinical biomarkers and symptom data into formats suitable for analysis \[hc\_usecase51\_prostate\_visual\_rep\]. [Link](https://www.mdpi.com/2306-5354/11/7/635)
12. TabPFN was used to combine clinical, MR morphological, and delta-radiomics features to predict lymphovascular invasion in invasive breast cancer patients \[hc\_usecase11\_lvi\_breast\_tabpfn\]. [Link](https://journals.sagepub.com/doi/full/10.1177/15330338251362050)
13. TabPFN is proposed to predict mental health trajectories through digital phenotyping, enabling proactive and personalized interventions in precision psychiatry \[hc\_usecase12\_precision\_psychiatry\_tabpfn\]. [Link](https://onlinelibrary.wiley.com/doi/epdf/10.1002/mdr2.70017)
14. TabPFN contributed to cardiovascular disease risk stratification using clinical features from a large patient cohort, incorporating interpretability techniques \[hc\_usecase13\_ml\_health\_tabpfn\]. [Link](https://github.com/Bruno-LSo/ML-Health-TABPFN)
15. TabPFN outperformed traditional machine learning models for early prediction of acute kidney injury in hospitalized patients, demonstrating generalizability across datasets \[hc\_usecase14\_aki\_ssrn\_tabpfn\]. [Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5397006)
16. TabPFN was integrated into a framework for predicting postoperative mobility and discharge destinations in older adults using sensor data \[hc\_usecase15\_postop\_mobility\_sensors\]. [Link](https://www.mdpi.com/1424-8220/25/16/5021)
17. TabPFN supported the prediction of infant temperament from maternal mental health data, aiding early identification of at-risk infants \[hc\_usecase16\_infant\_temperament\_tabpfn\]. [Link](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2025.1659987/abstract)
18. TabPFN was employed to characterize clinical risk profiles for complications in type 2 diabetes mellitus patients, focusing on neuropathy and retinopathy \[hc\_usecase17\_t2dm\_complications\_tabpfn\]. [Link](https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2025.1657366/abstract)
19. TabPFN was extended with a longitudinal-to-cross-sectional transformation to forecast Alzheimer’s disease progression on neuroimaging datasets \[hc\_usecase18\_ad\_l2c\_tabpfn\]. [Link](https://arxiv.org/abs/2508.17649)
20. TabPFN supported uncertainty calibration evaluation in medical data using variational techniques \[hc\_usecase19\_uncertainty\_vbll\_tabpfn\]. [Link](https://arxiv.org/abs/2509.10048)
21. TabPFN was applied to predict tumor response to chemotherapy in cholangiocarcinoma patients using RNA expression landscapes \[hc\_usecase20\_cholangio\_aacr\_tabpfn\]. [Link](https://aacrjournals.org/clincancerres/article/31/13_Supplement/A020/763312)
22. TabPFN was incorporated into a generative model framework for tasks like data augmentation and imputation in biomedicine \[hc\_usecase21\_tabpfgen\]. [Link](https://arxiv.org/abs/2406.05216)
23. TabPFN facilitated the prediction of gallstone malignancy risks through analysis of associated disease factors \[hc\_usecase22\_gallstone\_malignancy\_tabpfn\]. [Link](https://www.mdpi.com/2077-0383/14/17/6091)
24. TabPFN was used in classifying tuberculosis treatment outcomes based on clinical and sociodemographic data from national registries \[hc\_usecase23\_tb\_outcomes\_tabpfn\]. [Link](https://www.researchsquare.com/article/rs-7502054/v1)
25. TabPFN contributed to early prediction of gestational diabetes using cell-free DNA and genetic scores from early pregnancy blood samples \[hc\_usecase24\_gdm\_cfdna\_tabpfn\]. [Link](https://www.medrxiv.org/content/10.1101/2025.09.03.25334985v1)
26. TabPFN was used for predicting schizophrenia based on sense of agency features, emphasizing interpretability \[hc\_usecase25\_schizophrenia\_soa\_tabpfn\]. [Link](https://www.sciencedirect.com/science/article/abs/pii/S187620182500317X)
27. TabPFN was integrated into a physiologically based pharmacokinetic model for predicting dissolution and absorption of amorphous solid dispersions in drug development \[hc\_usecase26\_pbpk\_asd\_tabpfn\]. [Link](https://doi.org/10.1016/j.jconrel.2025.114123)
28. TabPFN enabled classification of respiratory diseases from sound data, addressing clinical spectrum diversity \[hc\_usecase27\_respiratory\_sounds\_tabpfn\]. [Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5529540)
29. TabPFN was applied to small-data tabular learning in drug discovery, handling data scarcity and distribution shifts \[hc\_usecase28\_drug\_discovery\_small\_data\_tabpfn\]. [Link](https://chemrxiv.org/engage/chemrxiv/article-details/68d29b1cf2aff1677025b18f)
30. TabPFN facilitated prediction of coronary heart disease risk in patients with cardiovascular-kidney-metabolic syndrome, optimizing evaluation in small samples \[hc\_usecase29\_ckm\_chd\_tabpfn\]. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC12437168/)
31. TabPFN was used to predict success of allogeneic stem cell mobilization in donors, aiding transplant therapies \[hc\_usecase30\_stem\_cell\_mobilization\_tabpfn\]. [Link](https://www.biorxiv.org/content/10.1101/2025.09.17.676674v1.full)
32. TabPFN contributed to predicting manual strength using anthropometric data, focusing on accuracy and interpretability \[hc\_usecase31\]. [Link](https://pubmed.ncbi.nlm.nih.gov/41021732/)
33. TabPFN supported uncertainty-guided model selection for biomolecule efficacy prediction, enhancing ensemble optimization in drug discovery, as studied at GSK \[hc\_usecase32\]. [Link](https://www.arxiv.org/abs/2510.02476)
34. TabPFN was utilized in a multitask deep learning framework for optimizing in vitro fertilization decisions, including embryo transfer and pregnancy prediction \[hc\_usecase33\]. [Link](https://dspace.mit.edu/bitstream/handle/1721.1/162969/zheng-zhengr-meng-eecs-2025-thesis.pdf?sequence=1&isAllowed=y)
35. TabPFN enabled a framework for early Long COVID detection through causal gene identification and interpretability \[hc\_usecase34\]. [Link](https://www.medrxiv.org/content/10.1101/2025.10.02.25337138v1.full.pdf)
36. TabPFN was used in a foundation model approach for neoadjuvant therapy recommendations in breast cancer, integrating multi-omics data \[hc\_usecase35\]. [Link](https://www.medrxiv.org/content/10.1101/2025.10.03.25337255v1)
37. Recent work has demonstrated explainable machine learning pipelines for coronary artery disease stratification from routine clinical data \[healthcare\_explainable\_cad\_algorithms2025\]. [Link](https://www.mdpi.com/1999-4893/18/11/693)
38. TabPFN facilitated prediction of recurrence and progression in oral potentially malignant disorder patients post-surgery \[hc\_usecase36\]. [Link](https://journals.lww.com/international-journal-of-surgery/abstract/9900/artificial_intelligence_for_predicting.3354.aspx)
39. TabPFN supported prediction of occult lymph node metastasis in non-small cell lung cancer patients treated with stereotactic ablative radiotherapy \[hc\_usecase37\]. [Link](https://www.redjournal.org/article/S0360-3016\(25\)05890-0/fulltext)
40. TabPFN was used in stroke diagnosis, addressing dataset imbalance and model interpretability for clinical decisions \[hc\_usecase38\]. [Link](https://www.ijsab.com/jsr-volume-9-issue-1/8205)
41. TabPFN was integrated into a multimodal thesis framework for clinical predictions using tabular and phenotypic data from large-scale projects \[hc\_usecase39\]. [Link](https://irep.mbzuai.ac.ae/items/3e3d4c0d-dbcb-4d5b-a23e-e28aea840660)
42. TabPFN was used to predict diabetes-related hypo- and hyperglycemia during hemodialysis using continuous glucose monitoring data, facilitating improved patient management \[hc\_usecase40\]. [Link](https://www.medrxiv.org/content/10.1101/2025.10.24.25338707v1)
43. TabPFN was applied to enhance diagnosis of hypervascular thyroid nodules using multimodal ultrasound features \[hc\_usecase52\_thyroid\_hypervascular\_multimodal\]. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC12432950/)
44. TabPFN was integrated with radiomics and clinical features to predict endovascular treatment success in femoropopliteal chronic total occlusions, supporting interventional planning \[hc\_usecase53\_fp\_cto\_radiomics\]. [Link](https://www.researchgate.net/publication/396892115_Radiomics_enhance_the_prediction_of_endovascular_treatment_success_for_femoropopliteal_chronic_total_occlusions_a_proof-of-concept_study)
45. TabPFN was applied to CorvisST biomechanical indices to classify corneal disorders, improving diagnostic accuracy in ophthalmology \[hc\_usecase41\_corvisst\_corneal\]. [Link](https://pubmed.ncbi.nlm.nih.gov/41130662/)
46. TabPFN was incorporated into a non-invasive sleep staging framework using respiratory sound features, advancing passive sleep monitoring \[hc\_usecase42\_sleepstage\_resp\_sounds\]. [Link](https://www.mdpi.com/1424-8220/25/20/6282)
47. TabPFN supported prediction of vancomycin blood concentrations to optimize antimicrobial dosing strategies in clinical practice \[hc\_usecase43\_vancomycin\_mimic4\]. [Link](https://journal.china-pharmacy.com/en/article/doi/10.6039/j.issn.1001-0408.2025.19.16/)
48. TabPFN was used to predict negative self-rated oral health in adults, identifying risk factors for targeted public-health interventions \[hc\_usecase44\_sroh\_jdent\]. [Link](https://www.sciencedirect.com/science/article/pii/S0300571225006104)
49. TabPFN was extended to very high-dimensional feature spaces to enable robust analysis of biomedical data, improving stability and interpretability in clinical applications \[hc\_usecase45\_tabpfn\_wide\]. [Link](https://arxiv.org/abs/2510.06162)
50. TabPFN predicted gastrointestinal bleeding risk in pediatric Henoch–Schönlein purpura patients, supporting early clinical intervention \[hc\_usecase50\_gibleed\_hsp\]. [Link](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1630807/full)
51. TabPFN was used as the pre-trained backbone (embeddings + in-context learning) for silica nanoparticle cellular toxicity prediction \[other\_usecase22\_silica\_np\_toxicity\_tabpfn\]. [Link](https://www.researchsquare.com/article/rs-7735307/v1)

## Financial Services, Banking, and Insurance

While we have seen strong customer interest in this area, this is not reflected by the relatively few published use cases (only 3) we managed to collect; we attribute this to the domain’s competitive nature and disinclination to publish.

1. TabPFN was applied to usage-based premium calculations in actuarial science, leveraging driving behavior data from IoT devices \[fin\_usecase1\_nonlife\_transformers\_actuarial\]. [Link](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/article/10.1007/s13385-024-00388-2&casa_token=5LdKiRIXfwEAAAAA:45MEDhjSq66DEqh96gk0NTWrhozhvBbd73mH-oMMuukD0EeHxH1fx3DTtp7h_l04IAjDuJXnpO2uHaHxjw)
2. TabPFN facilitated cross-selling of health insurance products through deep learning analysis of customer data \[fin\_usecase2\_crosssell\_health\_insurance\]. [Link](https://ieeexplore.ieee.org/abstract/document/10475046)
3. TabPFN was used in corporate bond recovery rate prediction for credit risk management \[fin\_usecase3\_recovery\_rate\_github\]. [Link](https://github.com/hoanguyen94/Recovery-rate-prediction)

## Energy and Utilities

We collected 15 use cases focused on environmental forecasting (algal blooms, wildfire, rainfall), renewable‑energy nowcasting, process/asset optimization across water, oil & gas, and materials.

1. TabPFN was employed to predict river algal blooms through multi-classification of chlorophyll-a concentrations, aiding water management \[energy\_usecase1\_river\_algal\_tabpfn\]. [Link](https://koreascience.kr/article/JAKO202427157640711.page)
2. TabPFN facilitated wildfire propagation prediction in Canadian conifer forests, classifying fire types for environmental risk assessment \[energy\_usecase2\_wildfire\_automl\]. [Link](https://www.sciencedirect.com/science/article/pii/S157495412400253X)
3. TabPFN was integrated into a machine learning framework for optimizing energy consumption at wastewater treatment plants \[energy\_usecase3\_wwtp\_tabpfnreg\]. [Link](https://www.researchgate.net/publication/390516459_Machine_learning_framework_for_energy_consumption_optimization_using_the_TabPFNRegressor_algorithm)
4. TabPFN supported rainfall forecast post-processing using historical error patterns from environmental data \[energy\_usecase4\_rainfall\_tabpfn\]. [Link](https://github.com/aarxshi/rainfall_tabpfn)
5. TabPFN enabled solar forecast error adjustment, particularly during rapid weather changes, as developed by Open Climate Fix \[energy\_usecase5\_solar\_adjuster\_ocf\]. [Link](https://gist.github.com/anshulg954/5f4423ee6b3d3151fa8d0d7fcd98d3eb)
6. TabPFN was applied to predict ash fusibility in high-alkali coal for improved energy production \[energy\_usecase6\_ash\_fusibility\_high\_alkali\]. [Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5406504)
7. TabPFN contributed to predicting Henry coefficients for alkanes in zeolites, aiding hydroisomerization in sustainable fuel production \[energy\_usecase7\_henry\_zeolites\]. [Link](https://pubs.acs.org/doi/full/10.1021/acs.jpcc.5c03868)
8. TabPFN facilitated shape-selectivity modeling in zeolites for long-chain alkane hydroisomerization, optimizing catalyst design \[energy\_usecase8\_shape\_selectivity\_zeolites\]. [Link](https://doi.org/10.4233/uuid:f36da034-5cb3-42ca-a53d-d351f68a9ffa)
9. TabPFN was used in an integrated framework for estimated ultimate recovery prediction and fracturing optimization in shale gas reservoirs \[energy\_usecase9\_shale\_eur\_fracturing\]. [Link](https://www.researchgate.net/publication/395761327_Coupling_EUR_Prediction_with_Fracturing_Optimization_An_Integrated_Machine_Learning_Framework_for_Shale_Gas_Development)
10. TabPFN supported core data augmentation for enhanced reservoir parameter prediction in oil and gas exploration \[energy\_usecase10\_core\_augmentation\_reservoir\]. [Link](https://www.researchgate.net/publication/395434405_Enhancing_Reservoir_Parameter_Prediction_Workflows_via_Advanced_Core_Data_Augmentation)
11. TabPFN was employed to optimize energy performance in multistage centrifugal pumps through entropy generation analysis \[energy\_usecase11\_multistage\_pump\_tabpfn\]. [Link](https://www.sciencedirect.com/science/article/abs/pii/S0360544225040411)
12. TabPFN contributed to physics-informed regression for evaluating solar-reflective materials in facade temperature modeling \[energy\_usecase12\_physinf\_facade\]. [Link](https://arxiv.org/pdf/2507.16174)
13. TabPFN was applied to generate advanced global heat flow maps at 0.2° resolution, integrating high-resolution geophysical data to improve geothermal resource modeling \[energy\_usecase13\_global\_heatflow\_02\]. [Link](https://www.researchgate.net/publication/396728153_The_First_02_Resolution_Global_Continental_Heat_Flow_Map_Advancing_Fine-Scale_Geothermal_Modeling)
14. TabPFN contributed to FuelCast, standardizing benchmarks for ship fuel consumption prediction and improving efficiency in maritime operations \[energy\_usecase14\_fuelcast\]. [Link](https://arxiv.org/abs/2510.08217)
15. TabPFN was used as the main supervised classifier to automatically identify thunderstorm ground enhancements from particle detector and environmental measurements \[energy\_usecase15\_tge\_tabpfn\]. [Link](https://arxiv.org/abs/2510.25125)

## Manufacturing and Industrial

We collected 12 diverse use cases including anomaly detection, predictive maintenance, physics‑aware optimization—spanning IIoT security, rotating machinery, semiconductor testing, geotechnical/optical sensing, machining, battery thermal modeling, and concrete mix design.

1. TabPFN enabled early fault classification in rotating machinery, addressing data scarcity in industrial scenarios \[manuf\_usecase1\_rotating\_faults\_tabpfn\]. [Link](https://ieeexplore.ieee.org/abstract/document/10318062)
2. TabPFN facilitated microcontroller performance prediction, aiding semiconductor screening with minimal supervision, as studied at Infineon Technologies \[manuf\_usecase2\_mcu\_performance\_tabpfn\]. [Link](https://iris.polito.it/handle/11583/3002056)
3. TabPFN was applied to caisson inclination prediction in ultra-deep construction, combining data denoising techniques \[manuf\_usecase3\_caisson\_inclination\_ml\]. [Link](https://www.sciencedirect.com/science/article/abs/pii/S2214391225001734)
4. TabPFN supported event classification in phase-sensitive optical time-domain reflectometry systems for distributed fiber sensing \[manuf\_usecase4\_photdr\_event\_classification\]. [Link](https://opg.optica.org/oe/fulltext.cfm?uri=oe-33-17-36646&id=575783)
5. TabPFN was integrated into an adaptive ensemble for intrusion detection in Industrial Internet of Things networks \[manuf\_usecase5\_wfetab\_iiot\_ids\]. [Link](https://rdcu.be/eASzJ)
6. TabPFN enabled a random forest-based framework for attack recognition in Internet of Things networks, improving interpretability \[manuf\_usecase6\_rf\_tabpfn\_iot\_attack\]. [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11142329)
7. TabPFN facilitated geotechnical site characterization for predicting soil strength and imputing mechanical parameters \[manuf\_usecase7\_geotech\_site\_char\_tabpfn\]. [Link](https://arxiv.org/abs/2509.03191)
8. TabPFN was used in cryogenic-assisted abrasive waterjet machining for improving surface integrity in titanium alloys \[manuf\_usecase8\_cryo\_awj\_ti64\]. [Link](https://www.sciencedirect.com/science/article/abs/pii/S2214993725004531)
9. TabPFN supported in-context learning for thermal behavior prediction in nano-phase change materials for battery systems \[manuf\_usecase9\_nano\_pcm\_thermal\_icl\]. [Link](https://www.sciencedirect.com/science/article/pii/S036054422504335X)
10. TabPFN was applied to explainable strength evaluation in multicomponent concrete mixtures \[manuf\_usecase10\_multicomponent\_concrete\]. [Link](https://www.mdpi.com/1996-1944/18/19/4456)
11. TabPFN was integrated into a multimodal fusion framework linking microstructure to friction behavior in martensitic stainless steel, improving wear resistance in materials engineering applications \[manuf\_usecase11\_martensitic\_friction\_multimodal\]. [Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5346149)
12. TabPFN supported multiscale modeling to predict soil salinity in arid farmland, advancing sustainable agricultural management in regions such as Xinjiang \[manuf\_usecase12\_soil\_salinity\_multiscale\]. [Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5591702)

## Other Industries

We collected 19 further heterogeneous TabPFN applications spanning geoscience, forensic science, agriculture, materials, and engineering domains—ranging from microbiome classification and lunar regolith analysis to soil property modeling, crop yield and phenology forecasting, fuel-blend optimization, and spatial regression.

1. TabPFN was modified for microbiome data classification in metagenomics, matching species abundance patterns with synthetic priors \[other\_usecase1\_microbiome\_zero\_inflated\]. [Link](https://openreview.net/forum?id=3I0bVvUj25)
2. TabPFN enabled lunar regolith analysis for classifying meteorite compositions from spectral data \[other\_usecase2\_lunar\_meteorites\]. [Link](https://www.sciencedirect.com/science/article/pii/S2095268624001010)
3. TabPFN facilitated winter wheat yield forecasting in agricultural regions by integrating climate and remote sensing data \[other\_usecase3\_winter\_wheat\_yield\_ssrn\]. [Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5380177)
4. TabPFN was applied to flood impact assessment on housing prices by geographic areas \[other\_usecase4\_flood\_housing\_prices\_ml\_climate\]. [Link](https://github.com/melina-thegarza/ml-climate/blob/main/doc/ML_Climate___Final.pdf)
5. TabPFN showed the strongest performance on 31 predictive soil modeling datasets containing 30 to 460 samples \[other\_usecase5\_soil\_mapping\_new\_default\]. [Link](https://arxiv.org/abs/2508.09888)
6. TabPFN was applied to shallow natural gas hazard prediction in tunnel construction \[other\_usecase6\_shallow\_gas\_tunnel\_tabpfn\]. [Link](https://www.sciencedirect.com/science/article/pii/S2590123025029366)
7. TabPFN supported automated feature engineering for energy consumption forecasting in domain-specific applications \[other\_usecase7\_autoenergy\_feature\_eng\]. [Link](https://www.sciencedirect.com/science/article/pii/S0950705125013413)
8. TabPFN enabled Australian rice phenology prediction using remote sensing and weather data for crop management \[other\_usecase8\_rice\_phenology\_tabpfn\]. [Link](https://www.mdpi.com/2072-4292/17/17/3050)
9. TabPFN was applied to a multi-stage framework for predicting fuel blend properties through automated feature engineering \[other\_usecase9\_fuel\_blend\_framework\]. [Link](https://chemrxiv.org/engage/chemrxiv/article-details/68dc888d3e708a7649ff0ec9)
10. TabPFN enabled kriging prior regression for incorporating spatial context in soil mapping predictions \[other\_usecase10\_kriging\_prior\_regression\]. [Link](https://arxiv.org/abs/2509.09408)
11. TabPFN was applied to predicting electric vehicle crash severity using deep learning models \[other\_usecase11\_ev\_crash\_severity\]. [Link](https://www.arxiv.org/abs/2509.11449)
12. TabPFN enhanced clone-type recognition across programming languages through metrics-driven analysis, improving stability and interpretability in software engineering \[other\_usecase12\_clone\_type\]. [Link](https://wiley.authorea.com/users/980519/articles/1346750-metrics-first-language-aware-clone-type-recognition-auditable-signals-across-c-c-java-and-python)
13. TabPFN was used to predict biomass-derived hard carbon performance in sodium-ion batteries, facilitating material selection for energy storage systems \[other\_usecase13\_hard\_carbon\_sib\]. [Link](https://arxiv.org/abs/2510.12833)
14. TabPFN informed the development of TabImpute, enabling efficient zero-shot imputation for missing tabular data and improving preprocessing pipelines \[other\_usecase14\_tabimpute\]. [Link](https://www.arxiv.org/abs/2510.02625)
15. TabPFN, alongside TabICL and related foundation models, was evaluated for intrusion detection, improving cybersecurity performance in IoT networks \[other\_usecase16\_cyber\_fm\_tabpfn\_tabicl\]. [Link](https://www.mdpi.com/2079-9292/14/19/3792)
16. TabPFN supported continual learning for tabular data streams in resource-constrained environments \[other\_usecase17\_imlp\_continual\]. [Link](https://arxiv.org/html/2510.04660v1)
17. TabPFN contributed to assessing robustness of language models for data fitting under irrelevant variations \[other\_usecase19\_llm\_data\_fitting\_robustness\]. [Link](https://arxiv.org/pdf/2508.19563)
18. TabPFN was used in forensic science to advance biogeographical ancestry predictions \[Heinzel2025\]. [Link](https://www.sciencedirect.com/science/article/pii/S1872497325000705)
19. TabPFN was used as a benchmark model for predicting avocado alternate bearing from Sentinel-2 and climate features \[other\_usecase21\_avocado\_alt\_bearing\]. [Link](https://www.preprints.org/manuscript/202510.2413)

## Appendix C Data Contamination and Deduplication for Real-TabPFN-2.5

To ensure fair evaluation and eliminate data contamination, we implemented an enhanced multi-tiered deduplication and filtering pipeline for Real-TabPFN-2.5. While based on the methodology used for Real-TabPFN \[garg2025realtabpfn\], the process was extended to deduplicate the training datasets against all internal benchmarks, our curated in-house validation suite, and the public TabArena benchmark \[erickson2025tabarena\]. Our deduplication procedure combines automated cross-referencing of dataset identifiers, feature schemas, and row- and column-level hashes with manual metadata inspection to ensure that no training dataset overlaps with, or is derived from, any evaluation dataset. Datasets failing these criteria were excluded from the final training corpus.

### C.1 Training Datasets

The following table lists the datasets curated for fine-tuning, along with their sources and access links.

| Name | Source |
| --- | --- |
| [artificial-characters](https://www.openml.org/search?type=data&sort=runs&status=active&id=1459) | OpenML |
| [BNG(breast-w)](https://www.openml.org/search?type=data&status=active&id=251) | OpenML |
| [BNG(tic-tac-toe)](https://www.openml.org/search?type=data&status=active&id=137) | OpenML |
| [connect\_4](https://www.openml.org/d/40668) | OpenML |
| [eeg-eye-state](https://www.openml.org/search?type=data&sort=runs&status=active&id=1471) | OpenML |
| [Employee-Turnover-at-TECHCO](https://openml.org/search?type=data&status=active&id=43551) | OpenML |
| [eye\_movements](https://openml.org/search?type=data&status=active&id=1044) | OpenML |
| [FOREX\_eurpln-hour-High](https://www.openml.org/search?type=data&status=active&id=41787&sort=runs) | OpenML |
| [gas-drift](https://www.openml.org/search?type=data&sort=runs&status=active&id=1476) | OpenML |
| [higgs](https://openml.org/search?type=data&status=active&id=23512) | OpenML |
| [Intersectional-Bias-Assessment-(Training-Data)](https://openml.org/search?type=data&status=active&id=44201) | OpenML |
| [law-school-admission-binary](https://openml.org/search?type=data&status=active&id=43904) | OpenML |
| [Medical-Appointment](https://openml.org/search?type=data&status=active&id=43617) | OpenML |
| [microaggregation2](https://www.openml.org/search?type=data&status=active&id=41671&sort=runs) | OpenML |
| [fried](https://www.openml.org/search?type=data&sort=runs&id=901&status=active) | OpenML |
| [mushroom](https://www.openml.org/search?type=data&status=active&id=43923&sort=runs) | OpenML |
| [NewspaperChurn](https://openml.org/search?type=data&status=active&id=44226) | OpenML |
| [nursery](https://openml.org/search?type=data&status=active&id=1568) | OpenML |
| [WBCAtt](https://www.openml.org/search?type=data&status=active&id=46676&sort=runs) | OpenML |
| [Internet Firewall Data](https://www.openml.org/search?type=data&sort=runs&id=43039&status=active) | OpenML |
| [aam\_avaliacao\_dataset](https://www.kaggle.com/datasets/himselfthedecker/aam-avaliacao-dataset) | Kaggle |
| [Air Traffic Data](https://www.kaggle.com/datasets/rohanshetty678/air-traffic-data) | Kaggle |
| [ansible-defects-prediction](https://www.kaggle.com/datasets/stefadp/ansibledefectsprediction) | Kaggle |
| [AV Healthcare Analytics II](https://www.kaggle.com/datasets/nehaprabhavalkar/av-healthcare-analytics-ii) | Kaggle |
| [Candidate Selection](https://www.kaggle.com/datasets/tarunchilkur/client) | Kaggle |
| [Cardio Disease](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) | Kaggle |
| [Classification - Crop Damages in India (2015-2019)](https://www.kaggle.com/datasets/aniketng21600/crop-damage-information-in-india) | Kaggle |
| [CSGO Round Winner Classification](https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification) | Kaggle |
| [Flower Type Prediction Machine Hack](https://www.kaggle.com/datasets/vpkprasanna/flower-type-prediction-machine-hack) | Kaggle |
| [Horse Racing - Tipster Bets](https://www.kaggle.com/datasets/gunner38/horseracing/data) | Kaggle |
| [How severe the accident could be](https://www.kaggle.com/datasets/kanuriviveknag/road-accidents-severity-dataset) | Kaggle |
| [hr-comma-sep](https://www.kaggle.com/datasets/pankeshpatel/hrcommasep) | Kaggle |
| [ip-network-traffic-flows-labeled-with-87-apps](https://www.kaggle.com/datasets/jsrojas/ip-network-traffic-flows-labeled-with-87-apps) | Kaggle |
| [Janatahack cross-sell prediction](https://www.kaggle.com/datasets/pawan2905/jantahack-cross-sell-prediction) | Kaggle |
| [L&T Vehicle Loan Default Prediction](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction) | Kaggle |
| [League of Legends Diamond Games (First 15 Minutes)](https://www.kaggle.com/datasets/benfattori/league-of-legends-diamond-games-first-15-minutes) | Kaggle |
| [Richter’s Predictor Modeling Earthquake Damage](https://www.kaggle.com/code/franciscoescobar/richter-s-predictor-modeling-earthquake-damage) | Kaggle |
| [Server Logs - Suspicious](https://www.kaggle.com/datasets/kartikjaspal/server-logs-suspicious) | Kaggle |
| [Sloan Digital Sky Survey DR14](https://www.kaggle.com/datasets/lucidlenn/sloan-digital-sky-survey) | Kaggle |
| [Sloan Digital Sky Survey DR16](https://www.kaggle.com/datasets/muhakabartay/sloan-digital-sky-survey-dr16) | Kaggle |
| [Term Deposit Prediction Data Set](https://www.kaggle.com/datasets/brajeshmohapatra/term-deposit-prediction-data-set) | Kaggle |
| [trajectory-based-ship-classification](https://www.kaggle.com/datasets/danielamigo/trajectorybasedshipclassification/data) | Kaggle |
| [Travel Insurance](https://www.kaggle.com/datasets/mhdzahier/travel-insurance) | Kaggle |

## Appendix D Details on Causal Inference Results

#### Causal Inference

Most real-world decision problems ultimately hinge on causal questions—understanding what would happen if we intervened, rather than merely observing correlations. Estimating Conditional Average Treatment Effects (CATEs) is one of the central ways to answer these “what-if” questions: how would an individual’s outcome change if a treatment were applied versus withheld?

#### Unconfounded Settings.

Many causal inference methods require unconfoundedness, which broadly states that there are no features not included in the dataset that influence both the treatment variable and the outcome \[rosenbaum\_propensity\]. While recent studies have begun to challenge the validity and verifiability of this assumption \[karlsson\_unconf, robertson\_dopfn\], there are presently a wide variety of causal inference methods designed for the unconfounded setting \[curth\_doing\_great, oprescu\_econml\].

#### Importance of Base Model.

Recent empirical findings have shown that when unconfoundedness holds, CATE estimation can be framed as an AutoML problem \[vanderschueren\_autocate\], as many CATE estimators require a choice of classification or regression model to approximate the likelihood (propensity) of a treatment and an outcome given an individual’s features. Parallel studies \[zhang\_one\_model, robertson\_dopfn\] have shown that TabPFN is an especially strong choice for meta-learners such as the X-, T-, and S-Learner \[kunzel\_meta\_learners\], hypothesizing that its strong performance in tabular prediction transfers to the problem of causal inference.

Table 3: Description of causal inference datasets in the RealCause benchmark.

| Characteristic | ACIC-2016 | IHDP | Lalonde-CPS | Lalonde-PSID |
| --- | --- | --- | --- | --- |
| Realizations | 10 | 100 | 100 | 100 |
| Samples | 4,802 | 747 | 16,177 | 2,675 |
| Features | 58 | 25 | 8 | 8 |

## Appendix E The TabPFN Ecosystem

Figure 11 provides a minimal user workflow through components in the TabPFN–Extensions ecosystem.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/figures/TabPFN_workflow.png)

Figure 11: A minimal user workflow through components in the TabPFN–Extensions ecosystem.

## Appendix F Additional Internal Benchmark Details

### F.1 Details on the normalization

For benchmarking, we normalize scores per dataset to enable averaging and clearer comparison across datasets, ensuring that differences in dataset difficulty do not bias comparisons. For each dataset, we linearly scale scores between 0 (worse model on this dataset) and 1 (best model). For each model, the default and tuned versions are considered as two different models for the normalization. Bar heights show the mean normalized performance, and error bars denote the standard error of the mean (SEM) across datasets, reflecting uncertainty from dataset variability.

### F.2 Additional results on many features

In Figure 12, we show results on an internal set of datasets containing from 500 to 2,000 features showing strong default performance.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x18.png)

Figure 12: TabPFN-2.5 default performs well up to 2,000 features. In our internal benchmark on datasets from 500 features to 2,000 features, we can see that for both classification (left) and regression (right), the default TabPFN-2.5 outperforms any other default model and is better than any tuned single model for regression.

## Appendix G Detailed TabArena Results

In addition to the results shown in Section 4, we also report the pairwise winrates of different models on TabArena in Figure 13 (for TabPFNv2 compatible datasets with less than 10k rows and 500 features) and Figure 14 (all datasets up to 100k training rows and 2k features).

We also compare our TabPFN-2.5 model to other foundation models in more detail below. In Figure 15, we show that TabPFN-2.5 outperforms TabICL when we restrict TabArena to only datasets for which TabICL is designed, and in Figure 16, we show much better performance when compared to LimiX’s results on datasets with less than 50,000 samples and 2,000 features, which corresponds to the datasets on which the TabArena maintainers could run LimiX at the time of writing (see this [link](https://github.com/autogluon/tabarena/pull/208)).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x20.png)

Figure 13: TabArena-Lite pairwise win rates on classification (left) and regression (right), restricted to TabPFNv2 compatible datasets (less than 10K training samples and 500 features ). Note that tuning for TabPFN-2.5 is only based on 60 random configs compared to 200 for the baselines.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x22.png)

Figure 14: TabArena-Lite pairwise win rates on classification (left) and regression (right), evaluated on all datasets (up to 100k training samples and 2K features ). Note that tuning for TabPFN-2.5 is only based on 60 random configs compared to 200 for the baselines.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x24.png)

Figure 15: Comparison with TabICL \[ qu2025tabicl \]. In this plot, we show the performance of TabPFN-2.5 and TabICL on a TabArena-lite subset compatible with TabICL, restricting to classification datasets with less than 500 features. On this subset for which TabICL is designed, we see that TabPFN-2.5 significantly outperforms TabICL.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x25.png)

Figure 16: Comparison with LimiX \[ zhang2025limix \]. In this plot, we show the performance of TabPFN-2.5 and LimiX on datasets from TabArena-Lite with less than 50,000 training samples and less than 2,000 features, which corresponds to the datasets on which the TabArena maintainers could run LimiX at the time of writing (see this link ). On this subset, we see that TabPFN-2.5 significantly outperforms LimiX. Note that these results are still unverified by the original authors at the time of writing and thus not included in the main paper results.

## Appendix H Results with Tuned Decision Thresholds

Starting with TabPFN-2.5, our framework supports tuning the decision threshold to optimize for specific metrics. Figure 17 quantifies the performance gains that this procedure can yield, illustrating substantial improvement in F1-score for several imbalanced datasets when tuning the threshold.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x26.png)

Figure 17: F1-score sometimes improves substantially by decision threshold tuning. The plot shows the difference in F1-score (macro) between a model with an optimized decision threshold and the same model using a default (untuned) threshold. This demonstrates the effectiveness of the tuning procedure for metric-specific optimization.

## Appendix I Supplementary Inference Time Details

Figure 18 shows the inference latency you can expect for three common models of GPUs. Figure 19 shows that the time scales linearly with the number of test rows. Figure 20 compares the fit + training time of TabPFN-2.5 vs TabPFNv2, showing that TabPFN-2.5 is significantly faster, showing between 1x and 2.3x speedup depending on the dataset size.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x27.png)

Figure 18: Time taken, in seconds, to train TabPFN-2.5 models on various training set sizes, and then make predictions on 500 test rows, using three common models of NVIDIA GPU: T4 15GB, A100 SXM 40GB, H100 SXM 80GB. Performance is shown for 100, 300, and features. Datasets with more than features have the same performance as datasets with, as each estimator will subsample to features. Incomplete lines indicate that the GPU had insufficient memory for that dataset size.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x28.png)

Figure 19: The time taken by TabPFN-2.5 to train and predict scales linearly in the test set size, shown here for a classification model trained on datasets of 500 rows × \\times 10 features, 5,000 rows 100 features, and 20,000 rows 500 features. Measured on one H100 GPU.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2511.08667/assets/x29.png)

Figure 20: TabPFN-2.5 is significantly faster than TabPFNv2. Comparison of the time taken to fit + predict TabPFN-2.5 vs TabPFNv2 on different number of rows and features. Measured for 100 test points on 1 H100, using the same number of estimators (8). Note that this is measured using the v2 and v2.5 versions available on the latest release of the TabPFN package, and thus is on top of the performance improvements since the original release of TabPFNv2.