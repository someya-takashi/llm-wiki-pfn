---
title: "Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models"
source: "https://ar5iv.labs.arxiv.org/html/2510.21204"
author:
published:
created: 2026-06-03
description: "Since the seminal work of TabPFN (hollmann2022tabpfn, ), research on tabular foundation models (TFMs) based on in-context learning (ICL) has challenged long-standing paradigms in machine learning.Without seeing any re…"
tags:
  - "clippings"
---
Xiyuan Zhang  
Amazon  
&Danielle C. Maddix  
Amazon  
&Junming Yin  
Amazon  
&Nick Erickson  
Amazon  
&Abdul Fatir Ansari  
Amazon  
&Boran Han  
Amazon  
&Shuai Zhang  
Amazon  
&Leman Akoglu  
Amazon and CMU  
&Christos Faloutsos  
Amazon and CMU  
&Michael W. Mahoney  
Amazon  
&Cuixiong Hu  
Amazon  
&Huzefa Rangwala  
Amazon  
&George Karypis  
Amazon  
&Bernie Wang  
Amazon

###### Abstract

Since the seminal work of TabPFN [^16], research on tabular foundation models (TFMs) based on in-context learning (ICL) has challenged long-standing paradigms in machine learning. Without seeing any real-world data, models pretrained on purely synthetic datasets generalize remarkably well across diverse datasets, often using only a moderate number of in-context examples. This shifts the focus in tabular machine learning from model architecture design to the design of synthetic datasets, or, more precisely, to the prior distributions that generate them. Yet the guiding principles for prior design remain poorly understood. This work marks the first attempt to address the gap. We systematically investigate and identify key properties of synthetic priors that allow pretrained TFMs to generalize well. Based on these insights, we introduce Mitra <sup>1</sup>, a TFM trained on a curated mixture of synthetic priors selected for their diversity, distinctiveness, and performance on real-world tabular data. Mitra consistently outperforms state-of-the-art TFMs, such as TabPFNv2 [^17] and TabICL [^29], across both classification and regression benchmarks, with better sample efficiency.

## 1 Introduction

Tabular data lie at the core of many real-world applications, including healthcare, finance, e-commerce, and the sciences [^32]. Predictive modeling on tabular data is central to statistical data analysis and underpins decision-making systems across these diverse domains [^14]. Tree-based models, such as random forests, gradient boosting, and ensemble methods [^3] [^7], have long dominated tabular predictions, due to their strong empirical performance and ease of use. However, these methods are typically tailored to individual datasets, and they exhibit limited ability to transfer across different distributions. Despite advances in transfer learning, a truly general-purpose approach to tabular prediction has remained elusive, until the introduction of TabPFN [^16].

Inspired by the success of large language models (LLMs), TabPFN and its follow-up works have introduced the notion of tabular foundation models (TFMs) based on in-context learning (ICL) [^16] [^17] [^29] [^2] [^5]. These models are pretrained on synthetic tabular tasks, and they make predictions on real downstream tasks by conditioning on labeled training samples from the downstream tasks as in-context examples. Synthetic data, generated on-the-fly during pretraining, provides broad task coverage and enables adaptation, without the need for large amounts of real-world downstream data.

While most previous TFM efforts focus on architectural innovations [^24] [^9] [^29], we advocate that greater attention should instead be on the design of the data priors—the distributions used to generate the synthetic datasets. Existing work typically relies on fixed or heuristic data priors [^29] [^16] [^17] [^2] [^5], leaving fundamental questions open. For example: *What makes a synthetic data prior effective? How should one construct a mixture of priors for better generalization?*

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x1.png)

Figure 1: Mixture of priors in Mitra, Generalizability Matrix 𝐆 \\mathbf{G} and Performance Vector 𝐏 \\mathbf{P}. We consider three factors for good priors: (1) the performance of a pretrained TFM using data generated solely from that prior on real tabular data (Point 1 and ); (2) its diversity and (3) distinctiveness when included in a mixture (Points 2 and 3, captured by the diagonal and off-diagonal entries of ). This leads to our mixture comprising SCM, gradient boosting, random forest and decision tree priors.

In this paper, we investigate these questions, with the goal of identifying key properties of synthetic priors used for pretraining TFMs. Our findings sharpen a vague rule of thumb that “diversity of the prior is important.” We show that the effectiveness of a synthetic prior depends on: (1) the performance of a TFM pretrained *solely on data generated from that prior*, when evaluated on real tabular data; (2) its *diversity*, i.e., how difficult it is for a TFM pretrained on this prior to overfit on its own distribution; and (3) *distinctiveness within a mixture of priors*, i.e., how hard it is for data generated from this prior to be predicted by TFMs pretrained on other priors. See Figure 1 for a simplified illustration.

Based on our insights, we construct a mixture of synthetic priors that consists of *structural causal models* (SCM) [^16] and *tree-based priors* (TBP), including gradient boosting, random forest, decision tree, and extra tree models. We choose SCMs because we show that they are diverse and achieve the best standalone performance on real tabular datasets. We choose TBPs because we identify that TFMs pretrained with data from SCMs do not always generalize well to all types of data generated from TBPs, showing the distinctiveness property of TBPs.

Our mixture of priors enables effective coverage of the diverse distributions in real-world tabular data. Notably, our priors are *model-agnostic*, and they consistently demonstrate performance improvement for both row-based 1D attention [^16] [^2] and more advanced element-based 2D attention architectures [^17] [^5]. Building on the 2D attention architecture, we propose Mitra, a TFM pretrained with our mixture of priors that obtains state-of-the-art (SOTA) results. Mitra outperforms existing TFMs, e.g., TabPFNv2 [^17], TabICL [^29], and other strong baselines, on both classification and regression tasks. Additionally, Mitra models pretrained with our prior mixture demonstrate better sample efficiency, i.e., they consistently have stronger performance with fewer in-context examples. This highlights the benefits of our principled prior mixture analysis.

We summarize our major contributions as follows:

## 2 Related Work

##### Traditional and Deep Learning-Based Tabular Models.

Historically, the tabular domain has been mainly dominated by traditional statistical methods, most notably gradient boosting (GB) decision trees, e.g., XGBoost [^3], LightGBM [^20], and CatBoost [^28]. These methods are widely adopted in practice due to their strong performance, robustness, and interpretability. To further improve generalization and automation, ensemble-based systems (e.g., AutoGluon [^7]) combine multiple base learners and automatically optimize hyperparameters and stacking strategies. More recently, deep learning-based approaches have been proposed to model complex and rich interactions in tabular data [^31] [^19] [^13] [^12]. For example, RealMLP [^18] introduces an optimized multilayer perceptron (MLP) architecture, tuned over a broad set of meta-benchmark datasets. While both traditional statistical methods and neural networks remain competitive on many real-world benchmarks, they require retraining from scratch for each new dataset, and they struggle to generalize across different distributions. This need for repeated per-data retraining poses scalability challenges and limits model reuse in real-world applications.

##### TFMs: Semantically-Rich Models.

Recent work has explored adapting LLMs to structured data by serializing tables into text. TabLLM [^15], GTL [^34], and Tabula [^10] enable zero-shot or few-shot inference by formatting rows and tasks into promptable inputs. TP-BERTa [^36] propose a pretrained language model tailored for tabular prediction tasks, with relative magnitude tokenization and intra-feature attention mechanism. Beyond text serialization, several approaches focus on non-textual pretraining by leveraging cross-table techniques. These include independent featurizers [^39], modular encodings with multi-task masked reconstruction [^37], and prompt masked table modeling [^38]. CARTE [^21] leverages pretraining on knowledge graphs and column-level metadata to facilitate downstream tabular tasks. These methods often rely heavily on semantic metadata (e.g., column names or textual descriptions), curated schemas, or large-scale real corpora. This dependence limits their applicability in settings where this auxiliary information is unavailable or unreliable, and it may require expensive inference costs in practical deployments due to their reliance on LLMs.

##### TFMs: ICL-Based Models.

A complementary line of work frames tabular prediction as an ICL task, where models are pretrained on synthetic or real datasets and downstream datasets are used as in-context examples. These efforts primarily focus on two directions: (1) designing better data priors; and (2) improving model architectures. In the first direction, TabPFN [^16] introduces this paradigm by pretraining a Transformer [^33] on SCM-generated synthetic data to simulate ICL for tabular classification problems. TabDPT [^23] extends this approach by incorporating real datasets during pretraining. Several subsequent models—including TabForestPFN [^2], Attic [^5], and TabICL [^29] —combine tree-based priors with SCMs for synthetic data generation. Specifically, TabForestPFN and Attic mix SCMs with decision trees, and TabICL combines SCMs with XGBoost. These approaches adopt tree-based priors in a heuristic manner, without explaining why incorporating these priors is beneficial. In the second direction, Attic [^5] introduces an element-based attention mechanism, which treats each cell (rather than every row or column) in a table as a separate token in the Transformer. TabPFNv2 [^17] adopts a similar element-wise Transformer architecture, and it further improves scalability and generalization by refining the synthetic prior distributions. More recently, TabICL [^29] proposes a two-stage architecture to first build fixed-dimensional embeddings of rows, followed by an efficient attention mechanism for ICL. Recent efforts have also explored how to scale ICL through compressing prompts, selecting informative contexts [^9] [^35] [^24] and hypernetworks [^26] [^1].

## 3 Mitra: TFM Pretrained on a Mixture of Priors

In this section, we describe how we characterize effective priors with the following three criteria: (1) strong performance on real datasets; (2) diversity; and (3) distinctiveness within the mixture. These characteristics together lead to the development of a new SOTA TFM, Mitra.

### 3.1 Data-Generating Priors

To pretrain a TFM on purely synthetic data in the supervised tabular learning setting, a data-generating prior $\mathcal{G}_{i}$ takes as input uniformly randomly generated hyper-parameters, e.g., feature size, number of samples, class count (for classification tasks), and categorical feature count; and it outputs a dataset $\mathcal{D}^{(i)}=\{(\mathbf{x}_{n}^{(i)},y_{n}^{(i)})\}_{n=1}^{N_{i}}$. This dataset consists of $N_{i}$ feature-label pairs, where $\mathbf{x}^{(i)}_{n}\in\mathbb{R}^{d}$ denotes a $d$ -dimensional feature vector with continuous or categorical attributes; and $y^{(i)}_{n}\in\mathcal{Y}\subset\mathbb{Z}$ denotes the corresponding target label for a classification task, or $y^{(i)}_{n}\in\mathcal{Y}\subset\mathbb{R}$ the target value for a regression task. The entire dataset can be represented as a two-dimensional matrix/table $D^{(i)}\in\mathbb{R}^{N_{i}\times(d+1)}$. (See Appendix A.1 for details on problem settings and preliminaries.)

In Mitra, we propose to use a mixture of $M$ data-generating priors $\{\mathcal{G}_{i}\}_{i=1}^{M}$. Concretely, we include the following types of priors: structural causal models (SCMs); and tree-based priors (TBPs).

SCM. The data-generating prior, $\mathcal{G}_{\text{SCM}}$, which was originally introduced in the pretraining of TabPFN [^16], is capable of capturing causal relationships among columns observed in tabular data. Moreover, SCMs model both feature dependencies and the conditional distribution $p(y|\mathbf{x})$. To sample a dataset $\mathcal{D}$ from $\mathcal{G}_{\text{SCM}}$, a directed acyclic graph (DAG) is first randomly constructed, after which the features $\mathbf{x}$ and target $y$ are generated in a sequential manner, following the conditional dependencies defined by the DAG structure.

TBP. The data-generating tree-based priors, $\mathcal{G}_{\text{TBP}}$, consist of trees and ensembles of trees, which are known for their strong predictive performance on tabular tasks and which are commonly used to model decision boundaries in tabular data. TBPs primarily focus on modeling $p(y|\mathbf{x})$ using complex threshold-based splits, with ensemble trees helping to smooth the resulting axis-aligned decision boundaries. In this work, we consider $\mathcal{G}_{\text{DT}}$, $\mathcal{G}_{\text{ET}}$, $\mathcal{G}_{\text{GB}}$, and $\mathcal{G}_{\text{RF}}$, where DT, ET, GB, RF refer to decision tree, extra tree, gradient boosting, and random forest, respectively. To use these models as data generators, they are first fit on a synthetically generated training dataset, after which features $\mathbf{x}$ are sampled from a simple distribution (e.g., multivariate standard normal) and targets $y$ are drawn according to the learned conditional distribution $\hat{p}(y\mid\mathbf{x})$. (See Appendix A.2.1 for additional details on the data generation process.) We group these priors together, and we refer to them as “indirectly sampled” priors. In addition, we introduce a custom “directly sampled” RF prior, $\mathcal{G}_{\text{DSRF}}$, that does not require model fitting. Instead, it directly constructs random conditional distributions $p(y\mid\mathbf{x})$ by sampling random split indices and thresholds. (See Appendix A.2.2 for details on each of the priors.)

![[Uncaptioned image]](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x2.png)

Table 1: Three factors of prior importance. Each entry 𝐆 i j \\mathbf{G}\_{ij} in the Generalizability Matrix represents the AUC of a TFM pretained on data generated from 𝒢 \\mathcal{G}\_{i} and evaluated on data from \\mathcal{G}\_{j}. Each element 𝐏 \\mathbf{P}\_{i} in the Performance Vector corresponds to the AUC of a TFM pretained on data from and evaluated on a real-world dataset. We use a color gradient to visually indicate the relative quality of each prior. The best/worst-performing priors are shown in the darkest green/red.

### 3.2 Prior Mixture Promoting Diversity and Distinctiveness

Here, we provide an in-depth study on characteristics of effective mixture $\mathcal{G}^{\prime}=\{\mathcal{G}^{\prime}_{i}\}_{i=1}^{M^{\prime}}$ of $M^{\prime}$ data-generating priors from $\mathcal{G}=\{\mathcal{G}_{i}\}_{i=1}^{M}$, where $M^{\prime}\leq M$. We first pretrain $M$ models on data from each candidate prior, and we evaluate them across priors to construct a generalizability matrix $\mathbf{G}\in\mathbb{R}^{M\times M}$, where the rows of $\mathbf{G}$ denote the model pretrained on draws from $\mathcal{G}_{i}$, and the columns denote the test data generated on draws from $\mathcal{G}_{j}$, with $\mathbf{G}_{ij}$ denoting the metric value. We also evaluate these $M$ models on real-world datasets to form a performance vector $\mathbf{P}\in\mathbb{R}^{M}$, where $\mathbf{P}_{i}$ denotes the performance of a model pretrained on data from $\mathcal{G}_{i}$ and evaluated on these real datasets. We find three key factors when characterizing good priors: (1) Performance, quantified by a higher value of $\mathbf{P}_{i}$; (2) Diversity, quantified by a lower diagonal value $\mathbf{G}_{ii}$, which indicates greater difficulty in overfitting to the same prior; (3) Distinctiveness, quantified by a lower off-diagonal value $\mathbf{G}_{ij}$, which shows how well a model pretrained on $\mathcal{G}_{i}$ performs on data from $\mathcal{G}_{j}$. More specifically, given a current mixture $\mathcal{G}^{\prime}$, the maximum of the off-diagonal $\mathbf{G}_{ij}$ in the $j^{\text{th}}$ column for $i$ such that $\mathcal{G}_{i}\in\mathcal{G}^{\prime}$ is a measure of the distinctiveness of $\mathcal{G}_{j}$. Then, adding the prior $\mathcal{G}_{j}$ with the smallest maximum, i.e., $\min_{1\leq j\leq M}\max_{1\leq i\leq M,\mathcal{G}_{i}\in\mathcal{G}^{\prime}}\mathbf{G}_{ij}$, increases the coverage of the mixture. Ultimately, prior importance balances performance and diversity.

Table 1 provides an illustrative example using the AUC metric of how these three factors help explain the prior importance findings in our ablation study in Table 6 (below; see Appendix C.1 for similar findings in other aggregated metrics). We see that $\mathcal{G}_{\text{SCM}}$ is of high quality due to both diversity (low diagonal value $\mathbf{G}_{ii}$) and strong performance on real datasets $\mathbf{P}$. Interestingly, while $\mathcal{G}_{\text{DSRF}}$ ranks second in performance on $\mathbf{P}$, the third best performing prior $\mathcal{G}_{\text{ET}}$ shows higher quality based on its ability to increase the diversity and distinctiveness, as validated in Table 6. For diversity, the diagonal of $\mathcal{G}_{\text{DSRF}}$ is significantly higher than the diagonal of $\mathcal{G}_{\text{ET}}$ ($0.984$ vs. $0.871$). For distinctiveness, the off-diagonals in the SCM row of $\mathbf{G}$ measure how much unique information is added from the next prior. We see that the model pretrained with $\mathcal{G}_{\text{SCM}}$ predicts on test data drawn from $\mathcal{G}_{\text{DSRF}}$ significantly better than that from $\mathcal{G}_{\text{ET}}$ ($0.960$ vs. $0.751$), showing that $\mathcal{G}_{\text{ET}}$ has higher distinctiveness.

### 3.3 Pretraining TFM on Prior Mixture

We use the insights from the prior subsection to assign weights $w_{i}$ to each prior in the mixture $\mathcal{G}^{\prime}=\{\mathcal{G}_{i}^{\prime}\}_{i=1}^{M^{\prime}}$, subject to $\sum_{i=1}^{M^{\prime}}w_{i}=1$. During pretraining, we sample generators $\mathcal{G}_{i}^{\prime}$ proportional to $w_{i}$, and we generate synthetic datasets for pretraining. Given a sampled table $D^{(i)}$ from a generator $\mathcal{G}_{i}^{\prime}$ in the mixture $\mathcal{G}^{\prime}$, we randomly sub-sample $s$ entries as the support set or in-context examples and $q$ entries as the query samples. We then optimize the likelihood over the masked query labels: $\mathcal{L}=\mathbb{E}_{D}[\log p_{\theta}(y_{\text{qry}_{1}:\text{qry}_{q}}|\mathbf{x}_{\text{sup}_{1}:\text{sup}_{s}},y_{\text{sup}_{1}:\text{sup}_{s}},\mathbf{x}_{\text{qry}_{1}:\text{qry}_{q}})].$ Algorithm 1 in Appendix A.3 summarizes our synthetic prior generation procedure. We present more pretraining details in Appendix B.3.2. Figure 2 illustrates the overall Mitra model pipeline with two popular architectures: one-dimensional row-wise attention [^16] [^2], and two-dimensional element-wise attention [^5] [^17]. Our priors are model-agnostic, and we demonstrate their effectiveness across both architectures in the next section.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x3.png)

Figure 2: Mitra Pipeline: Model agnostic visualized with either 1D or 2D attention.

## 4 Empirical Results

In this section, we show that Mitra achieves SOTA performance on both classification and regression tasks (Section 4.2). We demonstrate that Mitra is model agnostic and consistently improves the performance with both 1D attention (Mitra 1D) and 2D attention architectures (Section 4.3). Furthermore, we highlight Mitra’s better sample efficiency (Section 4.4), strong performance when combined with advanced ensembling techniques (Section 4.5), and strong fine-tuning performance (Section 4.6). We also conduct an ablation study to quantify the importance of each prior (Section 4.7), which supports our findings in Section 3.2. Finally, we analyze the scaling law with respect to both model size and synthetic dataset size (Section 4.8). See Appendix B for additional experimental setting details and Appendix C for additional experimental results.

### 4.1 Experimental Settings

Datasets. For the classification task, we compare Mitra on three established 10-fold benchmarks: TabRepo [^30]; Tabzilla [^25]; and AutoML benchmarks [^11]. We additionally evaluate on a concurrent benchmark TabArena [^8] in Appendix C.4. For the regression task, we compare on the 10-fold TabRepo [^30] benchmark. We evaluate both Mitra and its variant Mitra 1D that is trained on a 1D attention model with the same mixture of priors. To compare with 1D models, e.g., TabPFN that support features up to 100, and 2D models, e.g., TabPFNv2 that support features up to 500, we evaluate on both small-feature and large-feature benchmarks. For the small-feature benchmark, we use 66 TabRepo classification datasets, 75 TabZilla classification datasets, and 10 TabRepo regression datasets, that have up to 3,000 rows and 100 features following TabPFN [^16]. To evaluate on large-feature benchmarks, we use 29 classification datasets from AMLB benchmark with up to 10,000 rows and 500 features. This is consistent with the evaluation protocol of TabPFNv2 [^17]. We provide the dataset IDs in Appendix B.1.

Baselines. We compare Mitra with both SOTA TFMs (TabPFNv2 [^17], TabICL [^29], Attic [^5], TabPFN [^16], TabForestPFN [^2]), and competitive classical and neural baselines requiring dataset-specific tuning (RealMLP [^18], AutoGluon [^7], LightGBM [^20], XGBoost [^3], CatBoost [^28], MLP [^7]). For the latter, we use their bagged version implemented in AutoGluon 1.3.

Metrics. For classification tasks, we report aggregated metrics including AUC-ROC (AUC), accuracy (ACC) and cross-entropy (CE). For regression tasks, we report R <sup>2</sup>, root mean squared error (RMSE), and mean absolute error (MAE). Aggregated metrics can be disproportionately influenced by a small number of datasets with extreme performance, as noted in prior work [^30]. To address this, we complement these metrics with more robust and comprehensive rank-based metrics that better capture relative performance across datasets: average rank, Elo [^6], winrate, rescaled accuracy (RAcc), and champion delta (C $\Delta$). We provide the definitions of these metrics in Appendix B.2.

Evaluation Protocols. For TFMs including Mitra, we evaluate the following three settings: (1) *in-context learning* (ICL) performance; (2) *ICL with ensembling* techniques of feature shuffling, class order shuffling and random feature transformations [^17] [^29], denoted as “+e” in the following sections; and (3) *fine-tuning* that continues training the model on the training set of the target data, denoted as “+f” in the following sections. We use “bagging” to describe fine-tuning with bagging ensemble, and we show its advanced ensemble performance in Section 4.5.

Table 2: Mitra wins on all three classification benchmarks (We show the overall merged results and detail the separate benchmark results in Appendix C.4, i.e., Table 13 (TabRepo), Table 14 (TabZilla), Table 15 (AMLB) due to space limits). Winner/runner-up in /. +e means ensembling in ICL, and +f means fine-tuning. The $95\%$ confidence interval is shown in parentheses for the Elo. Aggregated metrics show mean and std (shown in parentheses) of the corresponding metric.

<table><tbody><tr><td rowspan="2">Model</td><td colspan="5">Ranking Metrics</td><td colspan="3">Aggregated Metrics</td></tr><tr><td>Avg. Rank <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Elo <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Winrate <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>RAcc <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>AUC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>ACC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>CE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>Mitra (+ef)</td><td>7.2</td><td>1136 <sub>(+4/-4)</sub></td><td>0.69</td><td>0.82</td><td>20.1</td><td>0.905 <sub>(0.124)</sub></td><td>0.858 <sub>(0.143)</sub></td><td>0.328 <sub>(0.317)</sub></td></tr><tr><td>Attic (+ef)</td><td>7.4</td><td>1128 <sub>(+4/-4)</sub></td><td>0.68</td><td>0.81</td><td>21.7</td><td>0.903 <sub>(0.125)</sub></td><td>0.857 <sub>(0.143)</sub></td><td>0.332 <sub>(0.317)</sub></td></tr><tr><td>TabPFNv2 (+e)</td><td>8.0</td><td>1107 <sub>(+4/-4)</sub></td><td>0.65</td><td>0.8</td><td>23.3</td><td>0.901 <sub>(0.13)</sub></td><td>0.856 <sub>(0.144)</sub></td><td>0.338 <sub>(0.318)</sub></td></tr><tr><td>TabPFNv2 (+ef)</td><td>8.6</td><td>1085 <sub>(+4/-4)</sub></td><td>0.62</td><td>0.76</td><td>25.3</td><td>0.897 <sub>(0.129)</sub></td><td>0.846 <sub>(0.15)</sub></td><td>0.363 <sub>(0.341)</sub></td></tr><tr><td>TabICL (+e)</td><td>9.5</td><td>1053 <sub>(+4/-4)</sub></td><td>0.58</td><td>0.75</td><td>30.9</td><td>0.889 <sub>(0.14)</sub></td><td>0.836 <sub>(0.15)</sub></td><td>0.367 <sub>(0.323)</sub></td></tr><tr><td>Mitra (+e)</td><td>9.7</td><td>1046 <sub>(+3/-3)</sub></td><td>0.57</td><td>0.73</td><td>31.2</td><td>0.896 <sub>(0.131)</sub></td><td>0.847 <sub>(0.148)</sub></td><td>0.36 <sub>(0.328)</sub></td></tr><tr><td>TabPFNv2</td><td>9.8</td><td>1043 <sub>(+4/-3)</sub></td><td>0.56</td><td>0.73</td><td>29.2</td><td>0.891 <sub>(0.139)</sub></td><td>0.846 <sub>(0.147)</sub></td><td>0.352 <sub>(0.324)</sub></td></tr><tr><td>Attic (+e)</td><td>9.9</td><td>1037 <sub>(+4/-3)</sub></td><td>0.55</td><td>0.73</td><td>31.8</td><td>0.896 <sub>(0.13)</sub></td><td>0.848 <sub>(0.148)</sub></td><td>0.364 <sub>(0.328)</sub></td></tr><tr><td>TabICL</td><td>10.6</td><td>1015 <sub>(+4/-4)</sub></td><td>0.52</td><td>0.7</td><td>33.5</td><td>0.884 <sub>(0.141)</sub></td><td>0.832 <sub>(0.152)</sub></td><td>0.374 <sub>(0.323)</sub></td></tr><tr><td>Mitra</td><td>10.6</td><td>1015 <sub>(+4/-4)</sub></td><td>0.52</td><td>0.69</td><td>32.9</td><td>0.891 <sub>(0.134)</sub></td><td>0.841 <sub>(0.15)</sub></td><td>0.368 <sub>(0.33)</sub></td></tr><tr><td>Mitra 1D (+f)</td><td>11.2</td><td>992 <sub>(+4/-4)</sub></td><td>0.49</td><td>0.68</td><td>34.5</td><td>0.893 <sub>(0.13)</sub></td><td>0.842 <sub>(0.15)</sub></td><td>0.367 <sub>(0.331)</sub></td></tr><tr><td>Attic</td><td>11.2</td><td>992 <sub>(+3/-4)</sub></td><td>0.49</td><td>0.68</td><td>35.2</td><td>0.884 <sub>(0.139)</sub></td><td>0.834 <sub>(0.156)</sub></td><td>0.376 <sub>(0.332)</sub></td></tr><tr><td>CatBoost</td><td>11.3</td><td>988 <sub>(+4/-4)</sub></td><td>0.48</td><td>0.67</td><td>35.7</td><td>0.888 <sub>(0.133)</sub></td><td>0.837 <sub>(0.15)</sub></td><td>0.375 <sub>(0.324)</sub></td></tr><tr><td>TabForestPFN (+f)</td><td>11.6</td><td>980 <sub>(+4/-4)</sub></td><td>0.47</td><td>0.65</td><td>35.3</td><td>0.886 <sub>(0.136)</sub></td><td>0.84 <sub>(0.148)</sub></td><td>0.377 <sub>(0.331)</sub></td></tr><tr><td>RealMLP</td><td>12.2</td><td>958 <sub>(+4/-4)</sub></td><td>0.44</td><td>0.62</td><td>35.4</td><td>0.878 <sub>(0.142)</sub></td><td>0.827 <sub>(0.164)</sub></td><td>0.412 <sub>(0.394)</sub></td></tr><tr><td>XGBoost</td><td>13.1</td><td>926 <sub>(+4/-4)</sub></td><td>0.4</td><td>0.58</td><td>39.5</td><td>0.883 <sub>(0.133)</sub></td><td>0.833 <sub>(0.149)</sub></td><td>0.388 <sub>(0.323)</sub></td></tr><tr><td>LightGBM</td><td>13.4</td><td>917 <sub>(+4/-4)</sub></td><td>0.38</td><td>0.56</td><td>39.8</td><td>0.876 <sub>(0.141)</sub></td><td>0.829 <sub>(0.152)</sub></td><td>0.393 <sub>(0.328)</sub></td></tr><tr><td>Random Forest</td><td>13.7</td><td>903 <sub>(+4/-4)</sub></td><td>0.36</td><td>0.54</td><td>44.1</td><td>0.874 <sub>(0.14)</sub></td><td>0.822 <sub>(0.152)</sub></td><td>0.471 <sub>(0.423)</sub></td></tr><tr><td>MLP</td><td>13.7</td><td>903 <sub>(+4/-4)</sub></td><td>0.36</td><td>0.51</td><td>40.1</td><td>0.869 <sub>(0.145)</sub></td><td>0.82 <sub>(0.161)</sub></td><td>0.414 <sub>(0.346)</sub></td></tr><tr><td>Mitra 1D</td><td>14.0</td><td>894 <sub>(+4/-4)</sub></td><td>0.35</td><td>0.55</td><td>42.5</td><td>0.869 <sub>(0.143)</sub></td><td>0.815 <sub>(0.163)</sub></td><td>0.414 <sub>(0.351)</sub></td></tr><tr><td>TabForestPFN</td><td>14.3</td><td>883 <sub>(+4/-4)</sub></td><td>0.34</td><td>0.52</td><td>42.6</td><td>0.864 <sub>(0.153)</sub></td><td>0.814 <sub>(0.167)</sub></td><td>0.414 <sub>(0.353)</sub></td></tr></tbody></table>

Table 3: Mitra also demonstrates better performance with 1D attention model, showing that our priors are *model-agnostic*. Winner/runner-up in shades of green /. The $95\%$ confidence interval is shown in parentheses for the Elo. The columns in the aggregated metrics are mean and std (shown in parentheses) of the corresponding metric.

<table><tbody><tr><td rowspan="2">Model</td><td colspan="5">Ranking Metrics</td><td colspan="3">Aggregated Metrics</td></tr><tr><td>Avg. Rank <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Elo <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Winrate <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>RAcc <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>AUC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>ACC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>CE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>Mitra 1D (+f)</td><td>3.0</td><td>1057 <sub>(+7/-7)</sub></td><td>0.6</td><td>0.68</td><td>16.5</td><td>0.886 <sub>(0.135)</sub></td><td>0.835 <sub>(0.155)</sub></td><td>0.38 <sub>(0.349)</sub></td></tr><tr><td>TabForestPFN (+f)</td><td>3.2</td><td>1038 <sub>(+7/-6)</sub></td><td>0.56</td><td>0.64</td><td>18.8</td><td>0.878 <sub>(0.142)</sub></td><td>0.832 <sub>(0.154)</sub></td><td>0.391 <sub>(0.337)</sub></td></tr><tr><td>TabPFN (+e)</td><td>3.4</td><td>1012 <sub>(+6/-7)</sub></td><td>0.52</td><td>0.6</td><td>24.4</td><td>0.862 <sub>(0.155)</sub></td><td>0.809 <sub>(0.17)</sub></td><td>0.418 <sub>(0.346)</sub></td></tr><tr><td>Mitra 1D</td><td>3.7</td><td>972 <sub>(+6/-6)</sub></td><td>0.45</td><td>0.53</td><td>27.2</td><td>0.865 <sub>(0.147)</sub></td><td>0.812 <sub>(0.164)</sub></td><td>0.417 <sub>(0.343)</sub></td></tr><tr><td>TabPFN</td><td>3.8</td><td>970 <sub>(+7/-6)</sub></td><td>0.45</td><td>0.53</td><td>26.3</td><td>0.86 <sub>(0.156)</sub></td><td>0.808 <sub>(0.17)</sub></td><td>0.426 <sub>(0.349)</sub></td></tr><tr><td>TabForestPFN</td><td>3.9</td><td>951 <sub>(+7/-6)</sub></td><td>0.42</td><td>0.49</td><td>27.7</td><td>0.859 <sub>(0.159)</sub></td><td>0.81 <sub>(0.169)</sub></td><td>0.419 <sub>(0.349)</sub></td></tr></tbody></table>

Table 4: Mitra wins on TabRepo 10-fold regression benchmark. Winner/runner-up in shades of green /. +e means adding ensemble in ICL, and +f means adding fine-tuning.

<table><tbody><tr><td rowspan="2">Model</td><td colspan="5">Ranking Metrics</td><td colspan="3">Aggregated Metrics</td></tr><tr><td>Avg. Rank <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Elo <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Winrate <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>RAcc <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>R <sup>2</sup> <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>RMSE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>MAE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>Mitra (+ef)</td><td>4.3</td><td>1140 <sub>(+20/-20)</sub></td><td>0.7</td><td>0.82</td><td>10.9</td><td>0.636 <sub>(0.306)</sub></td><td>2401.274 <sub>(7700.93)</sub></td><td>1351.15 <sub>(4100.89)</sub></td></tr><tr><td>TabPFNv2 (+e)</td><td>5.1</td><td>1090 <sub>(+22/-18)</sub></td><td>0.63</td><td>0.72</td><td>12.7</td><td>0.615 <sub>(0.332)</sub></td><td>2374.55 <sub>(7495.47)</sub></td><td>1304.54 <sub>(3960.35)</sub></td></tr><tr><td>RealMLP</td><td>5.8</td><td>1044 <sub>(+19/-20)</sub></td><td>0.56</td><td>0.7</td><td>16</td><td>0.627 <sub>(0.304)</sub></td><td>2424.34 <sub>(7574.57)</sub></td><td>1385.57 <sub>(4209.89)</sub></td></tr><tr><td>CatBoost</td><td>5.8</td><td>1044 <sub>(+19/-21)</sub></td><td>0.56</td><td>0.69</td><td>15.7</td><td>0.629 <sub>(0.301)</sub></td><td>2465.09 <sub>(7711.582)</sub></td><td>1444.57 <sub>(4383.87)</sub></td></tr><tr><td>TabPFNv2 (+ef)</td><td>6.1</td><td>1023 <sub>(+18/-18)</sub></td><td>0.53</td><td>0.62</td><td>15.5</td><td>0.600 <sub>(0.335)</sub></td><td>2372.76 <sub>(7513.58)</sub></td><td>1295.36 <sub>(3936.25)</sub></td></tr><tr><td>Mitra (+e)</td><td>6.4</td><td>1008 <sub>(+19/-21)</sub></td><td>0.51</td><td>0.63</td><td>19.5</td><td>0.604 <sub>(0.311)</sub></td><td>2469.12 <sub>(7922.80)</sub></td><td>1372.43 <sub>(4166.35)</sub></td></tr><tr><td>TabPFNv2</td><td>6.4</td><td>1008 <sub>(+19/-21)</sub></td><td>0.51</td><td>0.59</td><td>15.9</td><td>0.601 <sub>(0.347)</sub></td><td>2436.86 <sub>(7790.76)</sub></td><td>1337.64 <sub>(4063.73)</sub></td></tr><tr><td>XGBoost</td><td>6.7</td><td>989 <sub>(+19/-18)</sub></td><td>0.48</td><td>0.62</td><td>18.2</td><td>0.625 <sub>(0.298)</sub></td><td>2572.80 <sub>(7975.02)</sub></td><td>1573.52 <sub>(4767.09)</sub></td></tr><tr><td>LightGBM</td><td>6.8</td><td>984 <sub>(+19/-19)</sub></td><td>0.47</td><td>0.61</td><td>20.3</td><td>0.629 <sub>(0.289)</sub></td><td>2665.90 <sub>(8218.40)</sub></td><td>1571.68 <sub>(4762.81)</sub></td></tr><tr><td>Mitra</td><td>7.1</td><td>963 <sub>(+19/-20)</sub></td><td>0.44</td><td>0.59</td><td>20.6</td><td>0.599 <sub>(0.317)</sub></td><td>2465.02 <sub>(7858.76)</sub></td><td>1387.39 <sub>(4214.92)</sub></td></tr><tr><td>MLP</td><td>8.1</td><td>904 <sub>(+20/-20)</sub></td><td>0.36</td><td>0.5</td><td>22.3</td><td>0.595 <sub>(0.328)</sub></td><td>2778.66 <sub>(8748.78)</sub></td><td>1557.71 <sub>(4729.35)</sub></td></tr><tr><td>Random Forest</td><td>9.4</td><td>804 <sub>(+22/-22)</sub></td><td>0.23</td><td>0.32</td><td>27.9</td><td>0.585 <sub>(0.319)</sub></td><td>2797.35 <sub>(8626.88)</sub></td><td>1705.43 <sub>(5161.03)</sub></td></tr></tbody></table>

### 4.2 Mitra achieves SOTA classification and regression performance

Classification. We merge the three classification benchmarks and keep a unique set of 137 datasets to report an overall ranking and aggregated performance in Table 2. Detailed method configurations and individual benchmark results on TabRepo (Table 13), TabZilla (Table 14), and AMLB (Table 15) are provided in Appendix C.4. Mitra wins in the overall results and across all three benchmarks with varying feature dimensionality or sample size, and it consistently achieves the best performance with fine-tuning and ensembling, across both ranking-based and aggregated metrics. Notably, the ICL performance of Mitra closely matches that of TabPFNv2, despite being pretrained on a maximum of 16 features (one-tenth of the maximum pretraining features in TabPFNv2), which showcases the strong generalizability of our priors.

Regression. We evaluate on TabRepo regression datasets (Table 4)). Mitra again demonstrates the best performance across the various metrics, showing that our mixture of prior is task-agnostic.

### 4.3 Mitra priors are model agnostic

As shown in Table 3, when pretrained with the same mixture of priors, Mitra 1D also outperforms other 1D attention-based counterparts, e.g., TabPFN and TabForestPFN, which rely on less diverse priors. This highlights that our mixture of priors is model-agnostic and can consistently enhance performance across different architectures. Note that TabForestPFN does not offer native ensemble logic, and TabPFN does not offer native fine-tuning logic. We report results under the capabilities available in the respective baselines to ensure a fair comparison.

### 4.4 Mitra is more sample efficient

We compare the sample efficiency of Mitra against leading TFMs, i.e., TabPFNv2 and TabICL, in Table 5. We down-sample the number of ICL examples of TabRepo classification benchmark to 10%, 25%, 50%, and 75% of the original size, and Mitra consistently achieves better performance with ensemble and fine-tuning. We demonstrate in Appendix C.2 that such improvement can be attributed to the increased diversity of priors in the mixture, which enhances the model’s ability to generalize from limited data.

Table 5: Mitra shows better sample efficiency. Winner in shades of green.

<table><thead><tr><th rowspan="2">Model</th><th colspan="5">Ranking Metrics</th><th colspan="3">Aggregated Metrics</th></tr><tr><th>Avg. Rank <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th>Elo <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Winrate <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>RAcc <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>C <math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th>AUC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>ACC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>CE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr></thead><tbody><tr><td>Mitra (ds=1)</td><td>4.2</td><td>1234 <sub>(+8/-8)</sub></td><td>0.77</td><td>0.88</td><td>13.7</td><td>0.882 <sub>(0.125)</sub></td><td>0.84 <sub>(0.162)</sub></td><td>0.36 <sub>(0.361)</sub></td></tr><tr><td>TabPFNv2 (ds=1)</td><td>4.5</td><td>1217 <sub>(+7/-8)</sub></td><td>0.75</td><td>0.87</td><td>16.3</td><td>0.879 <sub>(0.127)</sub></td><td>0.838 <sub>(0.162)</sub></td><td>0.372 <sub>(0.361)</sub></td></tr><tr><td>TabICL (ds=1)</td><td>5.6</td><td>1144 <sub>(+7/-7)</sub></td><td>0.67</td><td>0.82</td><td>25.4</td><td>0.858 <sub>(0.145)</sub></td><td>0.816 <sub>(0.171)</sub></td><td>0.406 <sub>(0.369)</sub></td></tr><tr><td>Mitra (ds=0.75)</td><td>5.3</td><td>1163 <sub>(+8/-7)</sub></td><td>0.69</td><td>0.84</td><td>20.3</td><td>0.876 <sub>(0.129)</sub></td><td>0.835 <sub>(0.165)</sub></td><td>0.371 <sub>(0.366)</sub></td></tr><tr><td>TabPFNv2 (ds=0.75)</td><td>5.5</td><td>1155 <sub>(+8/-7)</sub></td><td>0.68</td><td>0.83</td><td>21.9</td><td>0.872 <sub>(0.134)</sub></td><td>0.832 <sub>(0.165)</sub></td><td>0.384 <sub>(0.365)</sub></td></tr><tr><td>TabICL (ds=0.75)</td><td>6.7</td><td>1081 <sub>(+7/-7)</sub></td><td>0.59</td><td>0.77</td><td>29.6</td><td>0.852 <sub>(0.146)</sub></td><td>0.81 <sub>(0.172)</sub></td><td>0.419 <sub>(0.372)</sub></td></tr><tr><td>Mitra (ds=0.5)</td><td>6.7</td><td>1083 <sub>(+7/-7)</sub></td><td>0.59</td><td>0.78</td><td>28.3</td><td>0.868 <sub>(0.135)</sub></td><td>0.827 <sub>(0.168)</sub></td><td>0.388 <sub>(0.372)</sub></td></tr><tr><td>TabPFNv2 (ds=0.5)</td><td>6.9</td><td>1072 <sub>(+7/-8)</sub></td><td>0.58</td><td>0.77</td><td>29.4</td><td>0.864 <sub>(0.138)</sub></td><td>0.824 <sub>(0.168)</sub></td><td>0.401 <sub>(0.369)</sub></td></tr><tr><td>TabICL (ds=0.5)</td><td>7.9</td><td>1012 <sub>(+7/-7)</sub></td><td>0.51</td><td>0.72</td><td>34.9</td><td>0.844 <sub>(0.146)</sub></td><td>0.799 <sub>(0.176)</sub></td><td>0.439 <sub>(0.375)</sub></td></tr><tr><td>Mitra (ds=0.25)</td><td>9.4</td><td>923 <sub>(+7/-7)</sub></td><td>0.4</td><td>0.64</td><td>41.1</td><td>0.849 <sub>(0.143)</sub></td><td>0.808 <sub>(0.174)</sub></td><td>0.43 <sub>(0.378)</sub></td></tr><tr><td>TabPFNv2 (ds=0.25)</td><td>9.6</td><td>912 <sub>(+7/-7)</sub></td><td>0.39</td><td>0.62</td><td>42.8</td><td>0.843 <sub>(0.147)</sub></td><td>0.802 <sub>(0.178)</sub></td><td>0.441 <sub>(0.372)</sub></td></tr><tr><td>TabICL (ds=0.25)</td><td>10.5</td><td>855 <sub>(+8/-8)</sub></td><td>0.32</td><td>0.56</td><td>46.3</td><td>0.819 <sub>(0.152)</sub></td><td>0.776 <sub>(0.181)</sub></td><td>0.487 <sub>(0.38)</sub></td></tr><tr><td>Mitra (ds=0.1)</td><td>12.1</td><td>740 <sub>(+9/-9)</sub></td><td>0.21</td><td>0.36</td><td>54.1</td><td>0.808 <sub>(0.151)</sub></td><td>0.771 <sub>(0.18)</sub></td><td>0.515 <sub>(0.377)</sub></td></tr><tr><td>TabPFNv2 (ds=0.1)</td><td>12.4</td><td>719 <sub>(+9/-10)</sub></td><td>0.19</td><td>0.33</td><td>55.1</td><td>0.801 <sub>(0.156)</sub></td><td>0.764 <sub>(0.182)</sub></td><td>0.519 <sub>(0.376)</sub></td></tr><tr><td>TabICL (ds=0.1)</td><td>12.7</td><td>689 <sub>(+10/-10)</sub></td><td>0.16</td><td>0.24</td><td>56.8</td><td>0.777 <sub>(0.153)</sub></td><td>0.742 <sub>(0.182)</sub></td><td>0.561 <sub>(0.384)</sub></td></tr></tbody></table>

### 4.5 Mitra shows the best performance with advanced ensembling techniques

To further boost performance, we implement a bagging-based ensemble for Mitra, denoted as Mitra (bagging). Specifically, we fine-tune a separate Mitra instance for each fold of an 8-fold (stratified) cross-validation ensemble [^22], and we aggregate their predictions via uniform averaging at test time. This allows Mitra to benefit from both data-level diversity and model-level robustness. While cross-validation ensembles are widely used for achieving top performance with classical tabular models [^7], to the best of our knowledge, our work is the first to demonstrate cross-validation ensembles for fine-tuned TFMs. We compare Mitra (bagging) against the strongest ensemble methods reported in previous works on TabRepo: the Post-Hoc Ensemble (PHE) of TabPFNv2, and the AutoGluon 1.3 best quality preset, which combines a diverse set of classical and neural tabular models. As shown in Figure 3, across different training budgets from 300, 600, 900 and 3600 seconds per dataset, Mitra (bagging) consistently outperforms both TabPFNv2 PHE and the AutoGluon ensemble, demonstrating the best performance with advanced ensembling techniques in the most competitive settings. Table 16 reports more details on ranking and aggregated performance over the unified classification benchmark.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x4.png)

Figure 3: Mitra (bagging) shows the best performance compared with TabPFNv2 PHE, AutoGluon best-quality and Mitra (+ef) across 300- to 3600-second budgets on 1-fold TabRepo benchmark.

### 4.6 Mitra shows the best fine-tuning performance

We compare the combined fine-tuning and ensemble performance of Mitra with TabPFNv2, TabICL, and Attic on TabRepo as a function of the number of estimators in the ensemble. As shown in Figure 4, Mitra consistently shows better fine-tuning performance across various ensemble sizes. Fine-tuning and ensembling of TabPFNv2 barely improves their ensemble-alone performance.<sup>2</sup> A likely reason for the strong gains from fine-tuning in Mitra is that it is pretrained with a maximum of 16 input features, so that adapting to downstream datasets with larger feature spaces provides substantial benefits. We also hypothesize that more diverse priors in Mitra contribute to its fine-tuning effectiveness, as they enable the model to generalize from a broader set of inductive biases, making the model more generalizable and adaptable to task-specific fine-tuning.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x5.png)

Figure 4: Mitra shows significantly better fine-tuning performance across ensemble sizes.

### 4.7 Prior Importance Ablation Study

We systematically study the importance of each prior in the Mitra prior mixture by iteratively adding the best-performing prior at each step. (See Appendix C.3 for details.) Table 6 presents several key findings that support and extend the analysis in Table 1 from Section 3.2: (1) Both diversity and performance on real datasets are important. Among all priors, data generated from $\mathcal{G}_{\text{SCM}}$ shows the highest importance, aligning with its high entry in the performance vector and low diagonal in Table 1. In contrast, despite being the second-best stand-alone prior, $\mathcal{G}_{\text{DSRF}}$ contributes the least when added to the mixture, due to its high diagonal ($0.984$) and strong overlap with other priors (off-diagonals $>0.96$). On the other hand, while $\mathcal{G}_{\text{RF}}$ exhibits low overlap (lowest diagonal of $0.761$ and off-diagonals in $[0.708,0.768]$), it also shows the worst performance on real datasets, as indicated by the performance vector, thus explaining why it is not as beneficial in the mixture. (2) Our mixture of priors promotes complementary strengths and improves generalization. We observe that combining $\mathcal{G}_{\text{SCM}}$ with every tree-based prior improves over either alone, which emphasizes the importance of a prior mixture. In particular, we see that combining $\mathcal{G}_{\text{ET}}$ with $\mathcal{G}_{\text{SCM}}$ significantly boosts performance, yielding an Elo improvement of 63. This supports the findings in Table 1 that adding $\mathcal{G}_{\text{ET}}$ into the mixture is effective since it is both diverse, as measured by its lower diagonal ($0.871$), and distinctive, as measured by its low off-diagonal in the SCM row ($0.751$). Similarly, we see that adding $\mathcal{G}_{\text{GB}}$ to the mixture further improves the performance. Adding the remaining priors $\mathcal{G}_{\text{DT}}$, $\mathcal{G}_{\text{DSRF}}$, $\mathcal{G}_{\text{RF}}$ leads to a few configurations with similarly good performance on real datasets. This aligns with our findings in Table 1 that they are less important due to either lower performance on real datasets or higher diagonal or off-diagonal values. We choose to include these priors in the final mixture to represent a more complete family of tree priors. Moreover, including them in the full mixture improves sample efficiency (See Appendix C.2 that shows Mitra is more sample efficient than $\{\mathcal{G}_{\text{SCM}},\mathcal{G}_{\text{ET}},\mathcal{G}_{\text{GB}}\}$ (Mitra-Mix2) and Attic [^5].)

Table 6: TabRepo 10-fold classification ablation study on the effect of each prior in the mixture consisting of $50\%$ SCM and $50\%$ TBPs with equal weighting. The $95\%$ confidence interval is shown in parentheses for the Elo. The columns in the aggregated metrics are mean and std (shown in parentheses) of the corresponding metric. Note that SCM + DT is a variant of Attic [^5].

<table><tbody><tr><td rowspan="2">Model</td><td colspan="5">Ranking Metrics</td><td colspan="3">Aggregated Metrics</td></tr><tr><td>Avg. Rank <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Elo <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Winrate <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>RAcc <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>AUC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>ACC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>CE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>RF</td><td>15.9</td><td>815 <sub>(+7/-6)</sub></td><td>0.25</td><td>0.38</td><td>42.8</td><td>0.831 <sub>(0.147)</sub></td><td>0.782 <sub>(0.177)</sub></td><td>0.521 <sub>(0.468)</sub></td></tr><tr><td>DT</td><td>15.4</td><td>840 <sub>(+6/-7)</sub></td><td>0.28</td><td>0.43</td><td>42.0</td><td>0.839 <sub>(0.148)</sub></td><td>0.791 <sub>(0.177)</sub></td><td>0.523 <sub>(0.532)</sub></td></tr><tr><td>GB</td><td>15.0</td><td>855 <sub>(+6/-6)</sub></td><td>0.3</td><td>0.52</td><td>38.3</td><td>0.843 <sub>(0.147)</sub></td><td>0.796 <sub>(0.177)</sub></td><td>0.501 <sub>(0.464)</sub></td></tr><tr><td>ET</td><td>14.3</td><td>883 <sub>(+6/-6)</sub></td><td>0.36</td><td>0.58</td><td>36.2</td><td>0.847 <sub>(0.140)</sub></td><td>0.803 <sub>(0.169)</sub></td><td>0.503 <sub>(0.579)</sub></td></tr><tr><td>DSRF</td><td>13.9</td><td>899 <sub>(+6/-6)</sub></td><td>0.36</td><td>0.58</td><td>34.5</td><td>0.849 <sub>(0.144)</sub></td><td>0.799 <sub>(0.175)</sub></td><td>0.473 <sub>(0.410)</sub></td></tr><tr><td>SCM</td><td>11.1</td><td>1000 <sub>(+5/-6)</sub></td><td>0.5</td><td>0.68</td><td>25.5</td><td>0.857 <sub>(0.142)</sub></td><td>0.812 <sub>(0.176)</sub></td><td>0.416 <sub>(0.378)</sub></td></tr><tr><td>SCM + RF</td><td>10.9</td><td>1005 <sub>(+5/-6)</sub></td><td>0.5</td><td>0.71</td><td>25.2</td><td>0.858 <sub>(0.143)</sub></td><td>0.816 <sub>(0.173)</sub></td><td>0.413 <sub>(0.374)</sub></td></tr><tr><td>SCM + DT</td><td>10.3</td><td>1025 <sub>(+5/-5)</sub></td><td>0.53</td><td>0.73</td><td>24.5</td><td>0.861 <sub>(0.135)</sub></td><td>0.816 <sub>(0.169)</sub></td><td>0.412 <sub>(0.369)</sub></td></tr><tr><td>SCM + GB</td><td>10.2</td><td>1028 <sub>(+6/-5)</sub></td><td>0.54</td><td>0.73</td><td>22.3</td><td>0.859 <sub>(0.142)</sub></td><td>0.817 <sub>(0.172)</sub></td><td>0.411 <sub>(0.374)</sub></td></tr><tr><td>SCM + DSRF</td><td>9.4</td><td>1058 <sub>(+6/-5)</sub></td><td>0.58</td><td>0.75</td><td>21.3</td><td>0.862 <sub>(0.140)</sub></td><td>0.819 <sub>(0.171)</sub></td><td>0.407 <sub>(0.371)</sub></td></tr><tr><td>SCM + ET</td><td>9.3</td><td>1063 <sub>(+5/-5)</sub></td><td>0.59</td><td>0.75</td><td>20.7</td><td>0.866 <sub>(0.133)</sub></td><td>0.823 <sub>(0.169)</sub></td><td>0.398 <sub>(0.371)</sub></td></tr><tr><td>SCM + ET + DT</td><td>10.2</td><td>1028 <sub>(+5/-5)</sub></td><td>0.54</td><td>0.74</td><td>22.4</td><td>0.869 <sub>(0.129)</sub></td><td>0.824 <sub>(0.168)</sub></td><td>0.403 <sub>(0.371)</sub></td></tr><tr><td>SCM + ET + RF</td><td>9.9</td><td>1042 <sub>(+6/-5)</sub></td><td>0.56</td><td>0.75</td><td>22.4</td><td>0.870 <sub>(0.129)</sub></td><td>0.825 <sub>(0.166)</sub></td><td>0.401 <sub>(0.369)</sub></td></tr><tr><td>SCM + ET + DSRF</td><td>9.6</td><td>1050 <sub>(+5/-5)</sub></td><td>0.57</td><td>0.74</td><td>22.1</td><td>0.860 <sub>(0.139)</sub></td><td>0.815 <sub>(0.172)</sub></td><td>0.410 <sub>(0.373)</sub></td></tr><tr><td>SCM + ET + GB</td><td>8.9</td><td>1076 <sub>(+5/-6)</sub></td><td>0.6</td><td>0.77</td><td>20.9</td><td>0.870 <sub>(0.129)</sub></td><td>0.827 <sub>(0.164)</sub></td><td>0.396 <sub>(0.368)</sub></td></tr><tr><td>SCM + ET + GB + DSRF</td><td>9.7</td><td>1046 <sub>(+6/-6)</sub></td><td>0.56</td><td>0.73</td><td>22.2</td><td>0.860 <sub>(0.142)</sub></td><td>0.817 <sub>(0.171)</sub></td><td>0.408 <sub>(0.366)</sub></td></tr><tr><td>SCM + ET + GB + RF</td><td>9.6</td><td>1050 <sub>(+6/-6)</sub></td><td>0.57</td><td>0.75</td><td>22.2</td><td>0.868 <sub>(0.131)</sub></td><td>0.823 <sub>(0.167)</sub></td><td>0.403 <sub>(0.367)</sub></td></tr><tr><td>SCM + ET + GB + DT</td><td>9.4</td><td>1059 <sub>(+6/-5)</sub></td><td>0.58</td><td>0.76</td><td>21.1</td><td>0.870 <sub>(0.130)</sub></td><td>0.826 <sub>(0.165)</sub></td><td>0.397 <sub>0.369</sub></td></tr><tr><td>SCM + ET + GB + DT + DSRF</td><td>9.5</td><td>1053 <sub>(+5/-5)</sub></td><td>0.57</td><td>0.75</td><td>22.9</td><td>0.859 <sub>(0.140)</sub></td><td>0.813 <sub>(0.172)</sub></td><td>0.411 <sub>(0.368)</sub></td></tr><tr><td>SCM + ET + GB + DT + RF</td><td>9.3</td><td>1062 <sub>(+6/-5)</sub></td><td>0.59</td><td>0.76</td><td>21.5</td><td>0.866 <sub>(0.134)</sub></td><td>0.823 <sub>(0.167)</sub></td><td>0.402 <sub>(0.369)</sub></td></tr><tr><td>SCM + ET + GB + DT + RF + DSRF(Mitra)</td><td>9.3</td><td>1062 <sub>(+6/-6)</sub></td><td>0.59</td><td>0.77</td><td>21.4</td><td>0.868 <sub>(0.133)</sub></td><td>0.822 <sub>(0.168)</sub></td><td>0.403 <sub>(0.369)</sub></td></tr></tbody></table>

### 4.8 Scaling Behavior for TFMs

Beyond prior construction, two critical factors influencing pretraining are the model size and the amount of training data. We investigate their respective scaling behaviors in Figure 5, by evaluating the performance on TabRepo given different model sizes and varying amount of pretraining data. Specifically, we vary the model depth across 6 configurations (4, 8, 12, 16, 20, 24 layers), all pretrained with the same on-the-fly generated mixture priors. For the *model size* scaling law, we observe that larger models achieve better performance in early training stages and also converge to higher final accuracy. However, performance gains begin to saturate beyond 12 layers, indicating a trade-off between model capacity and computational efficiency when selecting the appropriate model size. Regarding the *sample size* scaling law, we find that performance improves rapidly in the early stages and then gradually plateaus, with diminishing returns after approximately 18K steps. In our setting each step involves 2,048 new synthetic datasets, so this suggests the model saturates after encountering around 37 million unique datasets.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x6.png)

Figure 5: Model size scaling law. We vary the model size across six configurations (4, 8, 12, 16, 20, and 24 layers), each pretrained on 45 million unique samples. The plot zooms in on pretraining steps beyond 2,000 to provide a clearer view of performance trends. A larger version focusing only on this zoomed-in region is provided in Figure 25 in Appendix.

## 5 Conclusion and Discussion

Conclusion. We provide the first systematic investigation into the role of synthetic priors in pretraining TFMs, demonstrating how prior effectiveness depends on both its standalone performance on real tabular datasets as well as its diversity and distinctiveness within a mixture. Based on our analysis, we construct a diverse, high-performing, and model-agnostic mixture of synthetic priors. Leveraging this mixture, we develop Mitra, a SOTA TFM that consistently outperforms existing TFMs and other strong tabular baselines, across both classification and regression tasks. Limitations are discussed in Appendix D. Broader Impact. Our work advances the understanding and design of synthetic data for pretraining foundation models in structured domains, reducing the need for costly real labeled data and reducing privacy risks associated with training on sensitive real-world records.

## References

## Appendix A Mitra

In this section, we provide details on the in-context learning (ICL) preliminaries, our data generation methods and priors, and the overall Mitra algorithm.

### A.1 TFM Preliminaries

To pretrain a TFM on purely synthetic data, each dataset $\mathcal{D}=\{(\mathbf{x}_{n},y_{n})\}_{n=1}^{N}$ is sampled from a prior distribution $\mathcal{G}$ that generates datasets with varying numbers of features, samples, classes (for classification tasks), and categorical attributes. Once a TFM $f_{\theta}$ has been pretrained, it can be used to perform ICL as follows. The model is given a support set $\mathcal{D}_{\text{sup}}$ consisting of $s=N_{\text{sup}}$ labeled rows $\{(\mathbf{x}_{\text{sup}_{n}},y_{\text{sup}_{n}})\}_{n=1}^{s}$, along with $q=N_{\text{qry}}$ unlabeled query rows $\{\mathbf{x}_{\text{qry}_{n}}\}_{n=1}^{q}$, where $s+q=N$. It then predicts the corresponding query labels $\{y_{\text{qry}_{n}}\}_{n=1}^{q}$ in a single forward pass:

$$
\hat{y}_{\text{qry}_{1}},\cdots,\hat{y}_{\text{qry}_{q}}=f_{\theta}([(\mathbf{x}_{\text{sup}_{1}},y_{\text{sup}_{1}}),\cdots,(\mathbf{x}_{\text{sup}_{s}},y_{\text{sup}_{s}}),\mathbf{x}_{\text{qry}_{1}},\cdots,\mathbf{x}_{\text{qry}_{q}}]),
$$

without the need to update its parameter $\theta$.

### A.2 Data Generation

Here, we discuss the specific parameters and modeling choices in the data generation for feature-target pairs $(\mathbf{x},y)\in\mathcal{D}$ and the details on the priors that we use in our data-generating mixture of priors in Mitra. For simplicity, Figure 6 shows a visualization of a 2D dataset generated from our mixture of priors in Mitra. In addition, Figure 7 shows a t-SNE visualization of high-dimensional data samples from our mixture used during pretraining. We see both continuous and categorical features represented. The classification labels $y$ are depicted in color.

#### A.2.1 Feature and Target Generation

##### Feature 𝐱.\\mathbf{x}.

We design $\mathbf{x}$ to include $N_{\text{cont}}$ continuous and $N_{\text{cat}}$ categorical components, such that $N_{\text{cont}}+N_{\text{cat}}=d$. The number of categorical components is determined by $N_{\text{cat}}=\lfloor p_{\text{cat}}(d+1)\rfloor$, where $p_{\text{cat}}$ is a categorical percentage uniformly sampled. We then uniformly sample the $N_{\text{cat}}$ categorical feature indices $\mathcal{I}_{\text{cat}}$ in $\mathcal{I}=\{1,\dots,d\}$ without replacement. For each continuous feature index $j\in\mathcal{I}_{\text{cont}}=\mathcal{I}\setminus\mathcal{I}_{\text{cat}}$, we take $\mathbf{x}_{j}$ to be i.i.d. Gaussian noise. For each categorical feature $\mathbf{x}_{k}$, we generate its number of classes from a Geometric distribution. Lastly, we model each $\mathbf{x}_{k}$ via a multinomial distribution over its number of classes $N_{\text{c}}^{k}$.

##### Target yy.

For each target $y\in\mathcal{Y}$, we handle its generation differently depending on whether the generating prior uses “direct” or “indirect” sampling. For direct sampling, we directly simulate $y$ from a random conditional distribution $p(y\mid\mathbf{x})$. For indirect sampling, we first require fitting a classifier or regressor on a synthetically generated training dataset $(\mathbf{x},y)$ for the corresponding task. In classification tasks, the label $y\in\mathcal{Y}\subset\mathbb{Z}$ is generated similarly to the aforementioned label generating process for the categorical features. In regression tasks, we take $y$ to be normal. The final label $y_{2}$ is generated as the output from the fitted estimator on a new feature vector $\mathbf{x}_{2}$.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x7.png)

Figure 6: Randomly generated 2D Dataset from mixture of priors in Mitra. The features 𝐱 ∈ ℝ 2 \\mathbf{x}\\in\\mathbb{R}^{2} drawn from our mixture of SCM and TBP data-generating priors in on classification tasks. The classification labels y are shown in color.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x8.png)

Figure 7: t-SNE visualization of random high-dimensional dataset from mixture of priors in Mitra. The features 𝐱 ∈ ℝ d \\mathbf{x}\\in\\mathbb{R}^{d} drawn from our mixture of SCM and TBP data-generating priors in. The classification labels y are shown in color.

#### A.2.2 Synthetic Data-Generating Priors

We include a mixture of SCMs with TBPs, with both indirectly (ET, GB, DT, RF) and directly sampled priors (SCM, DSRF).

##### Indirectly Sampled Priors.

Algorithm 2 shows the data-generating procedure for “indirectly” sampled priors. We refer to these priors as indirectly sampled methods since they first require training data $D=[\mathbf{X},\mathbf{Y}]\in\mathbb{R}^{B\times(d+1)}$ to fit the estimator, i.e., classifier or regressor for the corresponding task, where $B$ denotes the base size. Then, we generate features $\mathbf{X}_{2}\in\mathbb{R}^{N\times d}$ according to the feature generation procedure described in Subsection A.2.1. The final $\mathbf{Y}_{2}\in\mathbb{R}^{N}$ is output from the fitted estimator by predicting on this $\mathbf{X}_{2}$ input. We note that TabForestPFN [^2] and Attic [^5] use an indirectly sampled DT prior, and TabICL [^29] uses an indirectly sampled XGBoost prior. Our data generation is similar to that in TabForestPFN [^2] and Attic [^5] with the differences that we directly fit a Classifier for classification tasks rather than using a Regressor, and use our direct multinomial label generation procedure (see subsection A.2.1) for both the target labels and categorical feature labels. Hence, we eliminate the need for using the quantile transform to bucketize the continuous values to labels. For these indirectly sampled TBPs, we use the classifiers and regressors from scikit-learn [^27].

##### Directly Sampled Priors.

Algorithm 5 shows the data-generating procedure for our directly sampled random forest (DRSF) TBP prior. We refer to these priors as directly sampled because they first sample a function $f\in\mathcal{F}$ from function space $\mathcal{F}$ and then generate the targets $y_{n}=f(\mathbf{x}_{n})$ for data $\mathbf{x}_{n}$. Hence, directly sampled methods only need to form $\mathbf{X}_{2}\in\mathbb{R}^{N\times d}$ once and then the directly-sampled data-generating priors directly output $\mathbf{Y}_{2}\in\mathbb{R}$.

For DSRF, we generate the features $\mathbf{X}_{2}$ using the same feature generation process from Subsection A.2.1. For DSRF, we must first construct random trees from $\mathcal{F}$ (see Algorithm 4). To do so, we sample the following: (1) random split indices in $\{0,\dots d-1\}$, where $d$ denotes the feature dimension; (2) random split thresholds in a specified range thres-int. Each node in the tree stores its corresponding split index and split threshold. We store the split intervals in a dictionary to track the sub-intervals corresponding to the feature split index to sample from on future splits as the algorithm progresses. We follow the convention from DTs, where the tree is split on a feature index $i$ and value $v$, and all datapoints such that $\mathbf{x}[i]\leq v$, are split to the left side of the tree and those such that $\mathbf{x}[i]>v$ are on the right side of the tree. We also control the number of nodes with no children. We construct the leaf node labels differently depending on the task type. For classification, we uniformly randomly sample a starting index in $\{0,\dots,N_{c}-1\}$, where $N_{c}$ denotes the number of classes, and we increment it for each subsequent leaf node added modulo the number of classes. For regression, we sample the leaf nodes from a Gaussian distribution. We sample number of estimators (random trees) $N_{e}$, and we traverse each tree until a leaf node is reached to get a target value for that tree (see Algorithm 3). Lastly, we compute the final label $y$ using majority voting over these $N_{e}$ values for classification tasks and using the mean for regression tasks.

For details on the SCM prior, see Appendix C.1 of the TabPFN paper [^16].

### A.3 Algorithm

Algorithm 1 provides an overview of our Mitra method.

The population version of the likelihood in Section 3.3 takes a form of an expectation over the table $D$, with each table sampled from the prior mixture. Given the sampled table, the model assumes $q$ query rows are conditionally independent given the in-context examples, so that the log-likelihood within the expectation is decomposed into a sum of $q$ individual terms, one per query row. When the query label corresponds to a classification task, its (conditional) distribution is assumed to be categorical, making the training objective equivalent to minimizing the cross-entropy loss. When the query label corresponds to a regression task, its (conditional) distribution is assumed to be Gaussian and the training objective corresponds to minimizing the Mean Squared Error (MSE) loss.

Algorithm 1 Mitra: Mixture of Synthetic Priors for pretraining TFMs

Set of data generators $\{\mathcal{G}_{i}\}_{i=1}^{M}$, with mixture weights $\{w_{i}\}_{i=1}^{M}$ such that $\sum_{i=1}^{M}w_{i}=1$,

Support set size $s$, query set size $q$, number of pretraining steps $T$.

for $t=1$ to $T$ do

  Sample generator index $i\sim w_{i}$, for $i=1,\dots,M$.

  Sample synthetic dataset $\mathcal{D}^{(i)}=\{(\mathbf{x}_{n}^{(i)},y_{n}^{(i)})\}_{n=1}^{N_{i}}\leftarrow\mathcal{G}_{i}$.

  Randomly partition $\mathcal{D}^{(i)}$ into support set $\mathcal{D}^{(i)}_{\text{sup}}$ and query set $\mathcal{D}_{\text{qry}}$ with $|\mathcal{D}^{(i)}_{\text{sup}}|=s$ and $|\mathcal{D}^{(i)}_{\text{qry}}|=q$.

  Construct input sequence: $[(\mathbf{x}^{(i)}_{\text{sup}_{1}},y^{(i)}_{\text{sup}_{1}}),\ldots,(\mathbf{x}^{(i)}_{\text{sup}_{s}},y^{(i)}_{\text{sup}_{s}}),\mathbf{x}^{(i)}_{\text{qry}_{1}},\ldots,\mathbf{x}^{(i)}_{\text{qry}_{q}}]$.

  Predict query labels via TFM: $\hat{y}^{(i)}_{\text{qry}_{1:q}}=f_{\theta}(\cdot)$.

  Compute loss: $\mathcal{L}=\log p_{\theta}(y^{(i)}_{\text{qry}_{1:q}}|\mathcal{D}^{(i)}_{\text{sup}},\mathbf{x}^{(i)}_{\text{qry}_{1:q}})$.

  Update model parameters $\theta$ using gradient of $\mathcal{L}$.

end for

Algorithm 2 Indirectly Sampled TBPs.

Task generator TBP, base size $B$, feature dimension $d$, number of samples $N$, number of classes $N_{c}$, task type $\mathcal{T}\in\{\textsc{Classification},\textsc{Regression}\}$.

Generate training data $\mathbf{D}=[\mathbf{X},\mathbf{Y}]\in\mathbb{R}^{B\times(d+1)}$, where $\mathbf{X}=[\mathbf{x}_{1};\dots;\mathbf{x}_{B}]\in\mathbb{R}^{B\times d}$ and $\mathbf{Y}=[y_{1},\dots,y_{B}]^{\top}\in\mathbb{R}^{B}$, according to the procedure in Subsection A.2.1.

Instantiate model $z\leftarrow\begin{cases}\texttt{TBPClassifier},&\text{if }\mathcal{T}=\textsc{Classification};\\
\texttt{TBPRegressor},&\text{otherwise}.\end{cases}$

Fit model: $z.\texttt{fit}(\mathbf{X},\mathbf{Y})$.

Sample testing features $\mathbf{X}_{2}\in\mathbb{R}^{N\times d}$.

Generate predictions: $\mathbf{Y}_{2}\leftarrow z.\texttt{predict}(\mathbf{X}_{2})\in\mathbb{R}^{N}$.

return $[\mathbf{X}_{2},\mathbf{Y}_{2}]\in\mathbb{R}^{N\times(d+1)}$.

Algorithm 3 Target Generation From DT Traversal

function DT-Traversal($\mathbf{x}$, tree)

   $(\texttt{ind},\texttt{thres})\leftarrow\texttt{tree.value}$

  if tree.target is not None then $\triangleright$ Leaf node reached

    $y\leftarrow\texttt{tree.target}$

  else if $\mathbf{x}[\texttt{ind}]\leq\texttt{thres}$ then

    $y\leftarrow$ DT-Traversal($\mathbf{x}$, tree.left) $\triangleright$ Traverse left subtree

  else

    $y\leftarrow$ DT-Traversal($\mathbf{x}$, tree.right) $\triangleright$ Traverse right subtree

  end if

  return $y$

end function

Algorithm 4 Construct Random Decision Tree (DT)

Feature dimension $d$, number of classes $N_{c}$, task type $\mathcal{T}\in\{\textsc{Classification},\textsc{Regression}\}$, minimum and maximum tree depths $d_{\min}>0$, $d_{\max}$, split threshold interval thres-int, probability of no children $p_{\text{nc}}$, Gaussian parameters for regression leaf nodes $(\mu,\sigma)$, global leaf counter $N_{\text{leaf}}=0$.

class Node(depth, $d_{\min}$, $d_{\max}$, thres-int)

  value $\leftarrow$ None $\triangleright$ Split rule: (feature index, threshold)

  left, right $\leftarrow$ None $\triangleright$ Child nodes

  d $\leftarrow$ depth $\triangleright$ Node depth

  target $\leftarrow$ None $\triangleright$ Leaf prediction value

  rs $\leftarrow\{\}$ $\triangleright$ Dict. of (feature index, split intervals)

  Store $d_{\min},d_{\max}$, and thres-int $\triangleright$ Min and max depths, threshold interval

end class

function RandDT(\[lb, ub\]=thres-int, tree=None, depth=0, rs = {}, ind = None)

  tree $\leftarrow$ Node(depth, $d_{\min},d_{\max},\texttt{thres-int}$)

  if depth $>0$ then

   tree.rs $\leftarrow$ rs;  tree.rs\[ind\] $\leftarrow$ \[lb, ub\]

  else if depth $=0$ and $\mathcal{T}$ = Classification then

   target $\sim\mathcal{U}(\{0,\dots,N_{c}-1\})$

  end if

   $\texttt{ind}\sim\mathcal{U}(\{0,\dots,d{-}1\})$

   $[\text{{lb}},\text{{ub}}]\leftarrow\texttt{tree.rs[ind]}$ if ind in tree.rs, else tree.thres-int

   $\texttt{thres}\sim\mathcal{U}([\text{{lb}},\text{{ub}}])$

  tree.value $\leftarrow$ (ind, thres)

  if $(\texttt{tree.d}<d_{\max}$ and $p\sim\mathcal{U}([0,1])\geq p_{\text{nc}})$ or $\texttt{tree.d}<d_{\min}$ then $\triangleright$ Add children

   tree.left $\leftarrow$ RandDT(\[lb, thres\], tree.left, tree.d + 1, tree.rs, ind)

   tree.right $\leftarrow$ RandDT(\[thres, ub\], tree.right, tree.d + 1, tree.rs, ind)

  else $\triangleright$ Assign target value to leaf node

   if $\mathcal{T}$ = Classification then

     tree.target $\leftarrow N_{\text{leaf}}\bmod N_{c}$

      $N_{\text{leaf}}\leftarrow N_{\text{leaf}}+1$

   else

     tree.target $\sim\mathcal{N}(\mu,\sigma)$

   end if

  end if

  return tree

end function

Algorithm 5 Directly Sampled Random Forest (DSRF)

Feature dimension $d$, number of samples $N$, number of classes $N_{c}$, task type $\mathcal{T}\in\{\textsc{Classification},\textsc{Regression}\}$, number of estimators $N_{e}$, minimum depth $d_{\min}>0$, maximum depth $d_{\max}$, split threshold interval thres-int, probability of terminal node $p_{\text{nc}}$, Gaussian parameters for regression leaves $(\mu,\sigma)$.

Generate feature matrix $\mathbf{X}_{2}\in\mathbb{R}^{N\times d}$ as described in Subsection A.2.1.

for $n=1$ to $N$ do

  for $i=1$ to $N_{e}$ do

    $\texttt{tree}\leftarrow\textsc{RandDT}()$ $\triangleright$ Sample a random DT from function space $\mathcal{F}$ (Alg. 4)

    $\texttt{target}[i]\leftarrow\textsc{DT-Traversal}(\mathbf{X}_{2}[n,:],\texttt{tree})$ $\triangleright$ Evaluate tree on input (Alg. 3)

  end for

  if $\mathcal{T}=\textsc{Classification}$ then

    $\mathbf{Y}_{2}[n]\leftarrow\textsc{MajorityVote}(\texttt{target})$

  else

    $\mathbf{Y}_{2}[n]\leftarrow\textsc{Mean}(\texttt{target})$

  end if

end for

return $[\mathbf{X}_{2},\mathbf{Y}_{2}]\in\mathbb{R}^{N\times(d+1)}$

## Appendix B Experimental Settings

In this section, we discuss the details on the benchmarking datasets, metrics, and implementation.

### B.1 Benchmarking Datasets

Table 7 provides the description and statistics of the benchmarking datasets with various number of rows and features. As discussed in Section 4.1, to compare with 1D models (e.g., TabPFN that support features up to 100) and 2D models (e.g., TabPFNv2) that support features up to 500, we evaluate on both small-feature and large-feature benchmarks. For the small-feature benchmark, we use 66 TabRepo classification datasets, 75 TabZilla classification datasets, and 10 TabRepo regression datasets, that have up to 3,000 rows and 100 features following TabPFN [^16]. To evaluate on large-feature benchmarks, we use 29 classification datasets from AMLB benchmark with up to 10,000 rows and 500 features. We additionally evaluate on a large-feature regression benchmark of 28 datasets from AMLB and OpenML-CTR23 benchmarks, with up to 10,000 rows and 500 features. This is consistent with the evaluation protocol of TabPFNv2 [^17].

The dataset task IDs are provided as follows:

TabRepo: 2, 11, 37, 2073, 2077, 3512, 3549, 3560, 3581, 3583, 3606, 3608, 3616, 3623, 3664, 3667, 3690, 3702, 3704, 3747, 3749, 3766, 3783, 3793, 3799, 3800, 3812, 3903, 3913, 3918, 9904, 9905, 9906, 9909, 9915, 9924, 9925, 9926, 9970, 9971, 9979, 14954, 125920, 125921, 146800, 146818, 146819, 168757, 168784, 190137, 190146, 359954, 359955, 359956, 359958, 359959, 359960, 359962, 359963, 361333, 361335, 361336, 361339, 361340, 361341, 361345

AMLB: 2073, 146818, 146820, 168350, 168757, 168784, 168911, 190137, 190146, 190392, 190410, 190411, 359954, 359955, 359956, 359958, 359959, 359960, 359961, 359962, 359963, 359964, 359965, 359968, 359969, 359970, 359972, 359974, 359975

TabZilla: 4, 9, 10, 11, 14, 15, 16, 18, 22, 23, 25, 27, 29, 31, 35, 37, 39, 40, 42, 47, 48, 50, 53, 54, 59, 2079, 2867, 3512, 3540, 3543, 3549, 3560, 3561, 3602, 3620, 3647, 3731, 3739, 3748, 3779, 3797, 3902, 3903, 3913, 3917, 3918, 9946, 9957, 9971, 9978, 9979, 9984, 10089, 10093, 10101, 14954, 14967, 125920, 125921, 145793, 145799, 145847, 145977, 145984, 146024, 146063, 146065, 146192, 146210, 146800, 146817, 146818, 146819, 146821, 146822

TabRepoReg: 167210, 359930, 359931, 359932, 359933, 359935, 359942, 359944, 359950, 359951

AMLB + OpenML-CTR23: \[167210, 233215, 359930, 359931, 359932, 359933, 359934, 359939, 359940, 359942, 359944, 359945, 359948, 359950, 359951, 360945, 361235, 361236, 361237, 361243, 361251, 361256, 361258, 361259, 361617, 361619, 361621, 361622

Table 7: Benchmarking Datasets Description and Statistics.

| Dataset | Task | Num. Tables | Max Num. Rows | Max Num. Features |
| --- | --- | --- | --- | --- |
| TabRepo | Classification | 66 | 3,000 | 100 |
| AMLB | Classification | 29 | 10,000 | 500 |
| Tabzilla | Classification | 75 | 3,000 | 100 |
| TabRepoReg | Regression | 10 | 3,000 | 100 |
| AMLB+OpenML-CTR23 | Regression | 28 | 10,000 | 500 |

### B.2 Metrics

To ensure a comprehensive evaluation of model performance across datasets and tasks, we employ a diverse set of ranking and aggregated metrics for both classification and regression tasks. Below, we provide their formal definitions used in this work. When computing the rank-based metrics, each fold of a dataset is considered as a separate evaluation unit.

#### B.2.1 Ranking-Based Metrics

To assess model performance across a suite of datasets, we adopt the following ranking-based metrics:

##### Average Rank.

Let $\mathcal{M}$ denote the set of models and $\mathcal{D}$ the set of datasets. For a model $m\in\mathcal{M}$ and dataset $\delta\in\mathcal{D}$, let $\text{rank}_{\delta}(m)$ denote the model’s rank (1 is best) on dataset $\delta$ according to a performance metric (e.g., accuracy). The average rank is defined as:

$$
\text{Avg. Rank}(m)=\frac{1}{|\mathcal{D}|}\sum_{\delta\in\mathcal{D}}\text{rank}_{\delta}(m).
$$

##### Elo Rating.

Elo rating generalizes pairwise win/loss outcomes into a competitive rating system. Each model is treated as a player, and its rating is updated based on pairwise performance comparisons across datasets. The final Elo rating reflects the model’s relative strength across all pairwise matchups. We implement the metric based on existing work [^6].

##### Winrate.

Winrate captures the proportion of datasets on which a model outperforms other models. Formally, for a model $m$:

$$
\text{Winrate}(m)=\frac{1}{|\mathcal{D}||\mathcal{M}|-1}\sum_{\delta\in\mathcal{D}}\sum_{\begin{subarray}{c}m^{\prime}\in\mathcal{M}\\
m^{\prime}\neq m\end{subarray}}\left(\mathbbm{1}[\text{E}_{\delta}(m)<\text{E}_{\delta}(m^{\prime})]+\frac{1}{2}\mathbbm{1}[\text{E}_{\delta}(m)=\text{E}_{\delta}(m^{\prime})]\right),
$$

where error $\text{E}=1-\text{AUC}$ for classification task, and $\text{E}=1-\text{R}^{2}$ for regression task, and $\mathbbm{1}[\cdot]$ denotes the indicator function. Tie contributes half a win.

##### Rescaled Accuracy.

To address the effect of dataset difficulty on raw scores, we scale by the best-performing model within each dataset:

$$
\text{RAcc}_{\delta}(m)=1-\text{Rescaled Error}_{\delta}(m),\text{Rescaled Error}_{\delta}(m)=\frac{\text{E}(m)-\text{E}(m^{*})}{\text{E}(m^{\prime})-\text{E}(m^{*})},
$$

where $m^{*}=\arg\min_{m}\text{E}(m)$ and $m^{\prime}=\arg\max_{m}\text{E}(m)$ respectively denote the model with the smallest and largest errors. Error $\text{E}=1-\text{AUC}$ for classification task, and $\text{E}=1-\text{R}^{2}$ for regression task. The overall Rescaled Accuracy is the average over datasets:

$$
\text{RAcc}(m)=\frac{1}{|\mathcal{D}|}\sum_{\delta\in\mathcal{D}}\text{RAcc}_{\delta}(m).
$$

##### Champion Delta.

Let $m^{*}=\arg\min_{m}\text{E}(m)$ denote the champion model. The Champion Delta for a model $m$ is defined as:

$$
\text{C}\Delta(m)=\bigg(1-\frac{\text{E}(m^{*})}{\text{E}(m)}\bigg)\times 100.
$$

This reflects the percentage performance margin between the current model and the best-performing model. Error $\text{E}=1-\text{AUC}$ for classification task, and $\text{E}=1-\text{R}^{2}$ for regression task.

#### B.2.2 Aggregated Classification Metrics

For aggregated classification metrics, we report standard metrics averaged across datasets.

##### AUC (Area Under ROC Curve).

For a binary classifier, AUC measures the area under the Receiver Operating Characteristic curve, which plots the true positive rate (TPR) against the false positive rate (FPR). Formally:

$$
\text{AUC}=\int_{0}^{1}\text{TPR}(t)\,d\,\text{FPR}(t),
$$

where $t$ is a threshold on predicted probability scores. For multiclass classification, we adopt the one-vs-one (OvO) strategy and compute the average AUC over all pairwise class comparisons.

##### Accuracy (ACC).

Accuracy is the fraction of correctly classified instances:

$$
\text{ACC}=\frac{1}{N}\sum_{i=1}^{N}\mathbbm{1}[y_{i}=\hat{y}_{i}],
$$

where $y_{i}$ is the ground-truth label, and $\hat{y}_{i}$ is the predicted label, and $\mathbbm{1}[\cdot]$ denotes the indicator function.

##### Cross-Entropy (CE).

Let $p_{i}$ be the predicted class probability vector for instance $i$, and $y_{i}$ the one-hot encoded ground-truth vector. The cross-entropy loss is:

$$
\text{CE}=-\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C}y_{i,c}\log p_{i,c},
$$

where $C$ is the number of classes.

#### B.2.3 Aggregated Regression Metrics

For aggregated regression metrics, we report standard metrics averaged across regression datasets.

##### R2\\text{R}^{2} (Coefficient of Determination).

The $\text{R}^{2}$ score quantifies the proportion of variance explained by the model:

$$
\text{R}^{2}=1-\frac{\sum_{i=1}^{N}(y_{i}-\hat{y}_{i})^{2}}{\sum_{i=1}^{N}(y_{i}-\bar{y})^{2}},
$$

where $\bar{y}$ is the mean of the ground-truth values.

##### Root Mean Squared Error (RMSE).

RMSE penalizes large prediction errors more heavily:

$$
\text{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_{i}-\hat{y}_{i})^{2}},
$$

##### Mean Absolute Error (MAE).

MAE measures the average magnitude of prediction errors:

$$
\text{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_{i}-\hat{y}_{i}|.
$$

### B.3 Training and Inference

Our implementation is based on PyTorch. We discuss the specific pretraining details and model hyperparameters below.

#### B.3.1 Transformer Architecture

Mitra is built on Transformer architecture [^33] with 12 layers, 512 embedding size and 4 attention heads. Each Transformer layer includes both row-wise attention and column-wise attention implemented using FlashAttention [^4]. The resulting model contains 72M parameters. Mitra 1D is built on Transformer architecture, and each layer contains row-wise attention. The resulting model contains 37M parameters.

#### B.3.2 Mitra Pretraining

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x9.png)

Figure 8: Comparison of training accuracy with respect to pretraining steps, with data generated from the mixture of priors in Mitra, vs. data from its prior individually.

For pretraining Mitra, we use eight 40GB A100 GPUs. Mitra is trained on 45 million synthetically generated datasets. This training takes approximately 60 hours on 8 GPUs (Nvidia A100s). To normalize features, we apply uniform quantile transform based on the support set, followed by standard normalization based on the mean and standard deviation from the support set. For regression tasks, we additionally apply min-max normalization for the target column using the minimum and maximum values of the support set for each table. Figure 8 illustrates that training curves tend to converge quickly around 2000 steps and are oscillatory. Interestingly, Figure 8 highlights the diversity findings in the diagonal of the ACC generalizability matrix in Table 9. We see that the training accuracy converges to a lower value for models generated with data from more diverse priors. The mixture priors of Mitra lie in between the less diverse TBPs (DSRF, GB) and the more diverse TBPs (RF, ET, DT), and slightly below SCM, which indicates that the TBPs add diversity to the mixture. In our experiments, we observe that the validation performance on real-world datasets improves as the pre-training continues.

#### B.3.3 Ensembling and Finetuning Parameters for TFMs

For models incorporating ensembling, we use the default number of estimators for each model, i.e., 4 for TabPFNv2 on classification tasks, 8 for TabPFNv2 on regression tasks, 32 for TabICL, 3 for TabPFN. We find that on TabZilla and AMLB classification benchmarks, TabPFNv2 with 8 estimators performs better than using 4 estimators and the performance saturates after that. Accordingly, we increase the number of estimators to 8 for TabPFNv2 on these two benchmarks to ensure competitive performance. All models with +f are fine-tuned for 50 epochs, which is a setting that typically triggers early stopping on most datasets. Note that TabPFN only supports up to 100 features so it is not evaluated on the AMLB classification dataset. In addition, TabPFN and TabICL do not support regression. Attic has been observed to have training stability issues on regression tasks, and its reported performance is inferior to that of XGBoost, which we include as a baseline for regression. Table 8 summarizes the number of estimators for each dataset.

Table 8: Number of Estimators for the Ensemble TFMs.

| Dataset | Mitra (+ef) | Attic (+ef) | TabPFNv2 (+ef) | TabPFNv2 (+e) | Mitra (+e) | Attic (+e) | TabICL (+e) | TabPFN (+e) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabRepo | 4 | 4 | 4 | 4 | 32 | 32 | 32 | 3 |
| AMLB | 8 | 8 | 8 | 8 | 32 | 32 | 32 | – |
| Tabzilla | 8 | 8 | 8 | 8 | 32 | 32 | 32 | 3 |
| Reg. | 8 | – | 8 | 8 | 32 | – | – | – |

#### B.3.4 Statistical Model Hyperparameters

For RealMLP, LightGBM, XGBoost, CatBoost, MLP, we use their default hyperparameters in AutoGluon [^7].

## Appendix C Additional Empirical Results

In this section, we report additional empirical results including the generalizability matrix $\mathbf{G}$ and performance vector $\mathbf{P}$ computed in other aggregated metrics, additional sample efficiency results and 2D decision boundary visualizations, ablations, classification and regression per dataset results, and the further improved performance of Mitra on the aggregated metrics using the advanced ensembling bagging method.

### C.1 Generalizability Matrix and Performance Vector Metrics

Here, we show the generalizability matrix $\mathbf{G}$ and performance vector $\mathbf{P}$ on TabRepo on metrics, i.e., accuracy (ACC) (Table 9) and cross-entropy (CE) (Table 10) in addition to the AUC table reported in Table 1. The rows of $\mathbf{G}$ denote the model pretrained with data generated from each prior. We then test those pretrained models on synthetic data generated from each prior distribution. We generate $100$ tables, each with $N=1000$ samples (rows), where the number of samples in the support $s=800$ and the number of samples in the query $q=200$.

We see that similar findings hold across the various aggregated metrics in these tables. In particular, Table 9 further emphasizes the distinctiveness of the TBP, ET, where the off-diagonal $\mathbf{G}_{ij}$ corresponding to the model pretrained with data drawn from $\mathcal{G}_{\text{ET}}$ is only $0.577$. For the ranking-based metrics of each individual prior, see Table 6.

Table 9: Diversity (diagonal) and distinctiveness (off-diagonals) of each prior distribution. Each entry $\mathbf{G}_{ij}$ in the Generalizability Matrix represents the ACC of a TFM pretained on data generated from $\mathcal{G}_{i}$ and evaluated on data from $\mathcal{G}_{j}$. Each element $\mathbf{P}_{i}$ in the Performance Vector corresponds to the ACC of a TFM pretained on data from $\mathcal{G}_{i}$ and evaluated on real-world datasets.

| Train |  |  |  | Test |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | SCM | ET | GB | DT | RF | DSRF | TabRepo |
| SCM | 0.841 <sub>(0.155)</sub> | 0.577 <sub>(0.249)</sub> | 0.902 <sub>(0.093)</sub> | 0.747 <sub>(0.198)</sub> | 0.562 <sub>(0.241)</sub> | 0.909 <sub>(0.113)</sub> | 0.812 <sub>(0.176)</sub> |
| ET | 0.808 <sub>(0.159)</sub> | 0.715 <sub>(0.210)</sub> | 0.928 <sub>(0.077)</sub> | 0.854 <sub>(0.141)</sub> | 0.613 <sub>(0.239)</sub> | 0.941 <sub>(0.096)</sub> | 0.803 <sub>(0.169)</sub> |
| GB | 0.800 <sub>(0.174)</sub> | 0.625 <sub>(0.258)</sub> | 0.942 <sub>(0.060)</sub> | 0.814 <sub>(0.177)</sub> | 0.588 <sub>(0.252)</sub> | 0.944 <sub>(0.092)</sub> | 0.796 <sub>(0.177)</sub> |
| DT | 0.799 <sub>(0.168)</sub> | 0.712 <sub>(0.213)</sub> | 0.929 <sub>(0.077)</sub> | 0.861 <sub>(0.134)</sub> | 0.611 <sub>(0.238)</sub> | 0.939 <sub>(0.098)</sub> | 0.791 <sub>(0.177)</sub> |
| RF | 0.807 <sub>(0.160)</sub> | 0.629 <sub>(0.235)</sub> | 0.904 <sub>(0.096)</sub> | 0.785 <sub>(0.177)</sub> | 0.602 <sub>(0.230)</sub> | 0.907 <sub>(0.112)</sub> | 0.782 <sub>(0.177)</sub> |
| DSRF | 0.806 <sub>(0.211)</sub> | 0.672 <sub>(0.235)</sub> | 0.927 <sub>(0.094)</sub> | 0.840 <sub>(0.154)</sub> | 0.607 <sub>(0.240)</sub> | 0.952 <sub>(0.081)</sub> | 0.799 <sub>(0.175)</sub> |

Table 10: Diversity (diagonal) and distinctiveness (off-diagonals) of each prior distribution. Each entry $\mathbf{G}_{ij}$ in the Generalizability Matrix represents the CE of a TFM pretained on data generated from $\mathcal{G}_{i}$ and evaluated on data from $\mathcal{G}_{j}$. Each element $\mathbf{P}_{i}$ in the Performance Vector corresponds to the CE of a TFM pretained on data from $\mathcal{G}_{i}$ and evaluated on real-world datasets.

| Train |  |  |  | Test |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | SCM | ET | GB | DT | RF | DSRF | TabRepo |
| SCM | 0.366 <sub>(0.359)</sub> | 1.054 <sub>(0.609)</sub> | 0.309 <sub>(0.275)</sub> | 0.690 <sub>(0.521)</sub> | 1.128 <sub>(0.627)</sub> | 0.273 <sub>(0.303)</sub> | 0.416 <sub>(0.378)</sub> |
| ET | 0.465 <sub>(0.387)</sub> | 0.764 <sub>(0.565)</sub> | 0.220 <sub>(0.233)</sub> | 0.399 <sub>(0.381)</sub> | 1.001 <sub>(0.642)</sub> | 0.173 <sub>(0.270)</sub> | 0.503 <sub>(0.579)</sub> |
| GB | 0.469 <sub>(0.403)</sub> | 1.022 <sub>(0.720)</sub> | 0.162 <sub>(0.165)</sub> | 0.535 <sub>(0.503)</sub> | 1.107 <sub>(0.714)</sub> | 0.163 <sub>(0.254)</sub> | 0.501 <sub>(0.464)</sub> |
| DT | 0.475 <sub>(0.400)</sub> | 0.777 <sub>(0.574)</sub> | 0.216 <sub>(0.231)</sub> | 0.383 <sub>(0.368)</sub> | 1.007 <sub>(0.648)</sub> | 0.177 <sub>(0.271)</sub> | 0.523 <sub>(0.532)</sub> |
| RF | 0.457 <sub>(0.370)</sub> | 0.967 <sub>(0.623)</sub> | 0.273 <sub>(0.264)</sub> | 0.579 <sub>(0.474)</sub> | 1.028 <sub>(0.628)</sub> | 0.248 <sub>(0.296)</sub> | 0.521 <sub>(0.468)</sub> |
| DSRF | 0.449 <sub>(0.492)</sub> | 0.868 <sub>(0.632)</sub> | 0.219 <sub>(0.296)</sub> | 0.445 <sub>(0.420)</sub> | 1.021 <sub>(0.655)</sub> | 0.134 <sub>(0.222)</sub> | 0.473 <sub>(0.410)</sub> |

### C.2 Sample Efficiency

Table 11 illustrates the improved sample efficiency of Mitra compared to its ablations: Mitra-Mix2 (SCM + ET + GB) and Attic (a variant of SCM + DT). In particular, the differences in the Elo become larger as the downsampling ratio, ds, is further decreased with the largest gains occuring when $\text{ds}=0.1$. This result emphasizes the generalizability of our mixture priors in comparison to mixtures of less priors in data-scarce scenarios.

Table 11: More diverse priors (Mitra) show better sample efficiency. Winner under each down-sampling ratio $\text{ds}\in\{1.0,0.75,0.5,0.25,0.1\}$ in bold.

<table><thead><tr><th rowspan="2">Model</th><th colspan="5">Ranking Metrics</th><th colspan="3">Aggregated Metrics</th></tr><tr><th>Avg. Rank <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th>Elo <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Winrate <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>RAcc <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>C <math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th>AUC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>ACC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>CE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr></thead><tbody><tr><td>Mitra-Mix2 (ds=1.0)</td><td>4.4</td><td>1234 <sub>(+8/-8)</sub></td><td>0.76</td><td>0.88</td><td>12.1</td><td>0.8817 <sub>(0.1256)</sub></td><td>0.8407 <sub>(0.1607)</sub></td><td>0.3612 <sub>(0.3624)</sub></td></tr><tr><td>Mitra (ds=1.0)</td><td>4.5</td><td>1227 <sub>(+8/-8)</sub></td><td>0.75</td><td>0.88</td><td>11.7</td><td>0.8818 <sub>(0.125)</sub></td><td>0.8396 <sub>(0.1625)</sub></td><td>0.3609 <sub>(0.3614)</sub></td></tr><tr><td>Attic (ds=1.0)</td><td>4.6</td><td>1221 <sub>(+8/-8)</sub></td><td>0.75</td><td>0.87</td><td>12.8</td><td>0.8812 <sub>(0.125)</sub></td><td>0.839 <sub>(0.1628)</sub></td><td>0.3682 <sub>(0.3607)</sub></td></tr><tr><td>Mitra (ds=0.75)</td><td>5.6</td><td>1151 <sub>(+7/-8)</sub></td><td>0.67</td><td>0.83</td><td>18.5</td><td>0.8758 <sub>(0.1289)</sub></td><td>0.8344 <sub>(0.1647)</sub></td><td>0.3716 <sub>(0.3668)</sub></td></tr><tr><td>Attic (ds=0.75)</td><td>5.6</td><td>1150 <sub>(+7/-7)</sub></td><td>0.67</td><td>0.83</td><td>19.0</td><td>0.8743 <sub>(0.13)</sub></td><td>0.8324 <sub>(0.165)</sub></td><td>0.3795 <sub>(0.3651)</sub></td></tr><tr><td>Mitra-Mix2 (ds=0.75)</td><td>5.7</td><td>1146 <sub>(+7/-7)</sub></td><td>0.66</td><td>0.82</td><td>18.6</td><td>0.8755 <sub>(0.1292)</sub></td><td>0.8338 <sub>(0.1641)</sub></td><td>0.3727 <sub>(0.3672)</sub></td></tr><tr><td>Mitra (ds=0.5)</td><td>7.1</td><td>1060 <sub>(+7/-7)</sub></td><td>0.56</td><td>0.76</td><td>26.6</td><td>0.8684 <sub>(0.135)</sub></td><td>0.8267 <sub>(0.1679)</sub></td><td>0.3888 <sub>(0.3723)</sub></td></tr><tr><td>Mitra-Mix2 (ds=0.5)</td><td>7.2</td><td>1056 <sub>(+7/-7)</sub></td><td>0.56</td><td>0.76</td><td>26.3</td><td>0.8674 <sub>(0.135)</sub></td><td>0.8257 <sub>(0.1695)</sub></td><td>0.3899 <sub>(0.3726)</sub></td></tr><tr><td>Attic (ds=0.5)</td><td>7.3</td><td>1047 <sub>(+7/-7)</sub></td><td>0.55</td><td>0.75</td><td>27.7</td><td>0.8665 <sub>(0.1351)</sub></td><td>0.824 <sub>(0.1683)</sub></td><td>0.3965 <sub>(0.3696)</sub></td></tr><tr><td>Mitra (ds=0.25)</td><td>9.9</td><td>888 <sub>(+7/-8)</sub></td><td>0.36</td><td>0.59</td><td>39.7</td><td>0.8492 <sub>(0.1427)</sub></td><td>0.8076 <sub>(0.1744)</sub></td><td>0.4317 <sub>(0.3781)</sub></td></tr><tr><td>Mitra-Mix2 (ds=0.25)</td><td>10.0</td><td>884 <sub>(+7/-7)</sub></td><td>0.36</td><td>0.59</td><td>40.1</td><td>0.8488 <sub>(0.1412)</sub></td><td>0.807 <sub>(0.1746)</sub></td><td>0.4327 <sub>(0.3776)</sub></td></tr><tr><td>Attic (ds=0.25)</td><td>10.2</td><td>874 <sub>(+7/-8)</sub></td><td>0.35</td><td>0.58</td><td>41.5</td><td>0.8457 <sub>(0.1439)</sub></td><td>0.8043 <sub>(0.1748)</sub></td><td>0.4416 <sub>(0.3727)</sub></td></tr><tr><td>Mitra (ds=0.1)</td><td>12.5</td><td>700 <sub>(+9/-9)</sub></td><td>0.18</td><td>0.25</td><td>53.3</td><td>0.8078 <sub>(0.1505)</sub></td><td>0.7702 <sub>(0.1799)</sub></td><td>0.5158 <sub>(0.3772)</sub></td></tr><tr><td>Mitra-Mix2 (ds=0.1)</td><td>12.7</td><td>681 <sub>(+11/-10)</sub></td><td>0.17</td><td>0.23</td><td>53.8</td><td>0.8065 <sub>(0.1499)</sub></td><td>0.7662 <sub>(0.1796)</sub></td><td>0.5182 <sub>(0.3766)</sub></td></tr><tr><td>Attic (ds=0.1)</td><td>12.7</td><td>680 <sub>(+8/-11)</sub></td><td>0.16</td><td>0.22</td><td>53.7</td><td>0.7995 <sub>(0.1532)</sub></td><td>0.7648 <sub>(0.1794)</sub></td><td>0.5237 <sub>(0.377)</sub></td></tr></tbody></table>

### C.3 Ablations

In Table 6, we perform an ablation study to analyze the importance of each prior in our mixture. We begin with ranking the performance of the model pretrained on data drawn from each prior individually. For the next step, we select the prior with the best performance, which in this case is SCM. We then add each remaining prior one-by-one with that selected prior to determine the next best pairing. We continue this procedure until we have added all the priors in Mitra in a forward process. We see that the ranking of the priors in decreasing order is $\{\text{SCM},\text{ET},\text{GB},\text{DT},\text{RF},\text{DSRF}\}$, which aligns with the performance, diversity and distinctiveness findings from the generalizability matrices $\mathbf{G}$ and performance vectors $\mathbf{P}$ in the various metrics in Tables 1 and Tables 9 - 10.

In Table 12, we study the effect of the percentage $p$ between SCM and the tree-based priors (TBP) in our mixture of priors in Mitra on the TabRepo dataset. In our experimental results, we set $p=0.5$ for an equal weighting of SCM and TBP. We see that on TabRepo there are several values of $p$ that show improved performance of the mixture over SCM alone ($p=1.0$) and TBP alone ($p=0)$. The average rankings for $p=0.5,0.4,0.6$ are tied at 3.2 with slightly better Elo for $p=0.5$ at 1040 vs. 1030 for the other variants. In addition, SCM alone has better performance over the TBPs alone with an average ranking of 3.7 vs. 4.4. The mixture of TBPs without SCM has a lack of distinctiveness since the models trained on data drawn from a TBP can predict on data drawn from these other TBPs better than models trained on SCM can, as measured by the off-diagonals of the generalizability matrix $\mathbf{G}$. Removing SCM from the mixture also removes the top performing prior as measured by the performance vector $\mathbf{P}$. Note that priors works have combined SCM with 1 TBP but not multiple TBPs. In addition, these works have not studied the effect of the percentage of SCM and the corresponding TBP, e.g., TabICL [^29] combines $p=0.7$ for SCM with $p=0.3$ XGBoost-based SCM, and TabForestPFN [^2] and Attic [^5] combine $p=0.5$ SCM with $p=0.5$ DT.

Table 12: TabRepo 10-fold classification ablation study on the effect of the percentage $p$ of SCM in the mixture of priors and the total amount of tree-based priors (TBP) is $1-p$.

<table><thead><tr><th rowspan="2"><math><semantics><mi>p</mi> <annotation>p</annotation></semantics></math></th><th colspan="5">Ranking Metrics</th><th colspan="3">Aggregated Metrics</th></tr><tr><th>Avg. Rank <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th>Elo <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Winrate <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>RAcc <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>C <math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th>AUC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>ACC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>CE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr></thead><tbody><tr><td>0.5 (Mitra)</td><td>3.2</td><td>1040 <sub>(+9/-10)</sub></td><td>0.57</td><td>0.69</td><td>14.3</td><td>0.868 <sub>(0.133)</sub></td><td>0.822 <sub>(0.168)</sub></td><td>0.403 <sub>(0.369)</sub></td></tr><tr><td>0.4</td><td>3.2</td><td>1030 <sub>(+9/-9)</sub></td><td>0.55</td><td>0.68</td><td>15.1</td><td>0.864 <sub>(0.133)</sub></td><td>0.817 <sub>(0.171)</sub></td><td>0.406 <sub>(0.369)</sub></td></tr><tr><td>0.6</td><td>3.2</td><td>1030 <sub>(+9/-9)</sub></td><td>0.55</td><td>0.67</td><td>14.6</td><td>0.865 <sub>(0.135)</sub></td><td>0.822 <sub>(0.169)</sub></td><td>0.404 <sub>(0.368)</sub></td></tr><tr><td>0.7</td><td>3.3</td><td>1029 <sub>(+10/-10)</sub></td><td>0.55</td><td>0.66</td><td>15.3</td><td>0.863 <sub>(0.139)</sub></td><td>0.818 <sub>(0.171)</sub></td><td>0.406 <sub>(0.368)</sub></td></tr><tr><td>1.0 (SCM)</td><td>3.7</td><td>983 <sub>(+10/-10)</sub></td><td>0.47</td><td>0.57</td><td>19.1</td><td>0.857 <sub>(0.142)</sub></td><td>0.812 <sub>(0.176)</sub></td><td>0.416 <sub>(0.378)</sub></td></tr><tr><td>0.0 (TBP)</td><td>4.4</td><td>888 <sub>(+10/-11)</sub></td><td>0.32</td><td>0.35</td><td>29.5</td><td>0.856 <sub>(0.137)</sub></td><td>0.808 <sub>(0.168)</sub></td><td>0.451 <sub>(0.392)</sub></td></tr></tbody></table>

Table 13: Mitra wins on TabRepo 10-fold classification benchmark. Winner/runner-up in /. +e means ensembling in ICL, and +f means fine-tuning. The $95\%$ confidence interval is shown in parentheses for the Elo. The columns in the aggregated metrics are mean and std (shown in parentheses) of the corresponding metric.

<table><tbody><tr><td rowspan="2">Model</td><td colspan="5">Ranking Metrics</td><td colspan="3">Aggregated Metrics</td></tr><tr><td>Avg. Rank <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Elo <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Winrate <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>RAcc <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>AUC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>ACC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>CE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>Mitra (+ef)</td><td>7.8</td><td>1141 <sub>(+5/-5)</sub></td><td>0.69</td><td>0.8</td><td>17.9</td><td>0.882 <sub>(0.124)</sub></td><td>0.837 <sub>(0.164)</sub></td><td>0.370 <sub>(0.367)</sub></td></tr><tr><td>Attic (+ef)</td><td>7.9</td><td>1135 <sub>(+5/-6)</sub></td><td>0.69</td><td>0.79</td><td>19.2</td><td>0.880 <sub>(0.125)</sub></td><td>0.835 <sub>(0.163)</sub></td><td>0.377 <sub>(0.366)</sub></td></tr><tr><td>TabPFNv2 (+e)</td><td>8.4</td><td>1120 <sub>(+5/-6)</sub></td><td>0.67</td><td>0.78</td><td>20.4</td><td>0.879 <sub>(0.126)</sub></td><td>0.835 <sub>(0.164)</sub></td><td>0.382 <sub>(0.367)</sub></td></tr><tr><td>TabPFNv2 (+ef)</td><td>8.9</td><td>1102 <sub>(+6/-6)</sub></td><td>0.64</td><td>0.76</td><td>21.6</td><td>0.878 <sub>(0.125)</sub></td><td>0.831 <sub>(0.162)</sub></td><td>0.396 <sub>(0.367)</sub></td></tr><tr><td>Mitra (+e)</td><td>9.6</td><td>1076 <sub>(+5/-6)</sub></td><td>0.61</td><td>0.74</td><td>25.8</td><td>0.877 <sub>(0.126)</sub></td><td>0.831 <sub>(0.164)</sub></td><td>0.394 <sub>(0.367)</sub></td></tr><tr><td>Attic (+e)</td><td>10.0</td><td>1063 <sub>(+5/-5)</sub></td><td>0.59</td><td>0.74</td><td>26.7</td><td>0.877 <sub>(0.126)</sub></td><td>0.830 <sub>(0.164)</sub></td><td>0.401 <sub>(0.367)</sub></td></tr><tr><td>TabICL (+e)</td><td>10.2</td><td>1057 <sub>(+5/-6)</sub></td><td>0.58</td><td>0.73</td><td>28.6</td><td>0.859 <sub>(0.144)</sub></td><td>0.813 <sub>(0.172)</sub></td><td>0.415 <sub>(0.374)</sub></td></tr><tr><td>TabPFNv2</td><td>10.3</td><td>1055 <sub>(+5/-5)</sub></td><td>0.58</td><td>0.72</td><td>24.8</td><td>0.866 <sub>(0.136)</sub></td><td>0.823 <sub>(0.167)</sub></td><td>0.396 <sub>(0.371)</sub></td></tr><tr><td>Mitra</td><td>10.7</td><td>1043 <sub>(+5/-5)</sub></td><td>0.56</td><td>0.71</td><td>28.1</td><td>0.868 <sub>(0.133)</sub></td><td>0.821 <sub>(0.168)</sub></td><td>0.403 <sub>(0.369)</sub></td></tr><tr><td>TabICL</td><td>11.4</td><td>1020 <sub>(+6/-5)</sub></td><td>0.53</td><td>0.69</td><td>30.7</td><td>0.853 <sub>(0.144)</sub></td><td>0.807 <sub>(0.176)</sub></td><td>0.422 <sub>(0.375)</sub></td></tr><tr><td>Attic</td><td>11.7</td><td>1009 <sub>(+5/-5)</sub></td><td>0.51</td><td>0.67</td><td>30.7</td><td>0.858 <sub>(0.141)</sub></td><td>0.812 <sub>(0.174)</sub></td><td>0.413 <sub>(0.370)</sub></td></tr><tr><td>CatBoost</td><td>11.9</td><td>1003 <sub>(+5/-5)</sub></td><td>0.50</td><td>0.68</td><td>32.3</td><td>0.865 <sub>(0.130)</sub></td><td>0.816 <sub>(0.168)</sub></td><td>0.416 <sub>(0.374)</sub></td></tr><tr><td>Mitra 1D (+f)</td><td>12.4</td><td>987 <sub>(+5/-6)</sub></td><td>0.48</td><td>0.65</td><td>31.7</td><td>0.868 <sub>(0.133)</sub></td><td>0.822 <sub>(0.171)</sub></td><td>0.405 <sub>(0.377)</sub></td></tr><tr><td>TabForestPFN (+f)</td><td>12.6</td><td>980 <sub>(+5/-5)</sub></td><td>0.47</td><td>0.64</td><td>33.2</td><td>0.861 <sub>(0.135)</sub></td><td>0.817 <sub>(0.167)</sub></td><td>0.417 <sub>(0.373)</sub></td></tr><tr><td>RealMLP</td><td>13.4</td><td>955 <sub>(+6/-5)</sub></td><td>0.44</td><td>0.6</td><td>33.3</td><td>0.851 <sub>(0.140)</sub></td><td>0.802 <sub>(0.183)</sub></td><td>0.453 <sub>(0.402)</sub></td></tr><tr><td>TabPFNv1 (+e)</td><td>14.2</td><td>929 <sub>(+5/-5)</sub></td><td>0.40</td><td>0.55</td><td>36.8</td><td>0.832 <sub>(0.151)</sub></td><td>0.787 <sub>(0.183)</sub></td><td>0.463 <sub>(0.385)</sub></td></tr><tr><td>LightGBM</td><td>14.3</td><td>926 <sub>(+5/-5)</sub></td><td>0.40</td><td>0.58</td><td>36.1</td><td>0.858 <sub>(0.132)</sub></td><td>0.812 <sub>(0.169)</sub></td><td>0.427 <sub>(0.373)</sub></td></tr><tr><td>XGBoost</td><td>14.3</td><td>924 <sub>(+5/-6)</sub></td><td>0.39</td><td>0.57</td><td>37.5</td><td>0.859 <sub>(0.131)</sub></td><td>0.813 <sub>(0.166)</sub></td><td>0.429 <sub>(0.369)</sub></td></tr><tr><td>Mitra 1D</td><td>15.0</td><td>902 <sub>(+5/-6)</sub></td><td>0.36</td><td>0.54</td><td>38.3</td><td>0.842 <sub>(0.140)</sub></td><td>0.794 <sub>(0.179)</sub></td><td>0.448 <sub>(0.381)</sub></td></tr><tr><td>Random Forest</td><td>15.0</td><td>901 <sub>(+6/-5)</sub></td><td>0.36</td><td>0.52</td><td>40.9</td><td>0.844 <sub>(0.136)</sub></td><td>0.797 <sub>(0.170)</sub></td><td>0.540 <sub>(0.513)</sub></td></tr><tr><td>TabPFNv1</td><td>15.0</td><td>901 <sub>(+5/-5)</sub></td><td>0.36</td><td>0.51</td><td>38.4</td><td>0.829 <sub>(0.151)</sub></td><td>0.785 <sub>(0.184)</sub></td><td>0.469 <sub>(0.386)</sub></td></tr><tr><td>MLP</td><td>15.3</td><td>890 <sub>(+5/-5)</sub></td><td>0.35</td><td>0.47</td><td>37.7</td><td>0.839 <sub>(0.141)</sub></td><td>0.794 <sub>(0.180)</sub></td><td>0.464 <sub>(0.387)</sub></td></tr><tr><td>TabForestPFN</td><td>15.6</td><td>880 <sub>(+6/-6)</sub></td><td>0.34</td><td>0.5</td><td>40.2</td><td>0.834 <sub>(0.156)</sub></td><td>0.791 <sub>(0.185)</sub></td><td>0.447 <sub>(0.380)</sub></td></tr></tbody></table>

Table 14: Mitra wins on TabZilla 10-fold classification benchmark. Winner/runner-up in /. +e means adding ensemble in ICL, and +f means adding fine-tuning.

<table><tbody><tr><td rowspan="2">Model</td><td colspan="5">Ranking Metrics</td><td colspan="3">Aggregated Metrics</td></tr><tr><td>Avg. Rank <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Elo <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Winrate <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>RAcc <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>AUC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>ACC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>CE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>Mitra (+ef)</td><td>8.6</td><td>1110 <sub>(+5/-4)</sub></td><td>0.66</td><td>0.82</td><td>20.7</td><td>0.913 <sub>(0.132)</sub></td><td>0.867 <sub>(0.143)</sub></td><td>0.304 <sub>(0.302)</sub></td></tr><tr><td>Attic (+ef)</td><td>8.8</td><td>1102 <sub>(+5/-5)</sub></td><td>0.65</td><td>0.82</td><td>21.8</td><td>0.912 <sub>(0.133)</sub></td><td>0.867 <sub>(0.142)</sub></td><td>0.305 <sub>(0.302)</sub></td></tr><tr><td>TabPFNv2 (+e)</td><td>9.5</td><td>1079 <sub>(+5/-5)</sub></td><td>0.61</td><td>0.8</td><td>25.1</td><td>0.909 <sub>(0.142)</sub></td><td>0.863 <sub>(0.144)</sub></td><td>0.315 <sub>(0.301)</sub></td></tr><tr><td>TabPFNv2 (+ef)</td><td>9.9</td><td>1067 <sub>(+5/-4)</sub></td><td>0.6</td><td>0.77</td><td>25.7</td><td>0.903 <sub>(0.143)</sub></td><td>0.851 <sub>(0.156)</sub></td><td>0.345 <sub>(0.342)</sub></td></tr><tr><td>TabICL (+e)</td><td>10.3</td><td>1055 <sub>(+4/-4)</sub></td><td>0.58</td><td>0.77</td><td>30.0</td><td>0.907 <sub>(0.141)</sub></td><td>0.850 <sub>(0.145)</sub></td><td>0.333 <sub>(0.305)</sub></td></tr><tr><td>Mitra (+e)</td><td>10.3</td><td>1055 <sub>(+4/-4)</sub></td><td>0.58</td><td>0.77</td><td>30.2</td><td>0.907 <sub>(0.143)</sub></td><td>0.860 <sub>(0.142)</sub></td><td>0.325 <sub>(0.295)</sub></td></tr><tr><td>Attic (+e)</td><td>10.6</td><td>1043 <sub>(+5/-5)</sub></td><td>0.56</td><td>0.77</td><td>31.4</td><td>0.906 <sub>(0.142)</sub></td><td>0.862 <sub>(0.141)</sub></td><td>0.328 <sub>(0.295)</sub></td></tr><tr><td>Mitra</td><td>11.1</td><td>1028 <sub>(+4/-5)</sub></td><td>0.54</td><td>0.74</td><td>30.8</td><td>0.905 <sub>(0.141)</sub></td><td>0.858 <sub>(0.140)</sub></td><td>0.329 <sub>(0.295)</sub></td></tr><tr><td>TabPFNv2</td><td>11.2</td><td>1026 <sub>(+5/-4)</sub></td><td>0.54</td><td>0.74</td><td>30.9</td><td>0.901 <sub>(0.149)</sub></td><td>0.856 <sub>(0.144)</sub></td><td>0.327 <sub>(0.305)</sub></td></tr><tr><td>TabICL</td><td>11.2</td><td>1025 <sub>(+4/-4)</sub></td><td>0.54</td><td>0.74</td><td>32.3</td><td>0.903 <sub>(0.142)</sub></td><td>0.851 <sub>(0.143)</sub></td><td>0.338 <sub>(0.303)</sub></td></tr><tr><td>Attic</td><td>11.6</td><td>1014 <sub>(+5/-4)</sub></td><td>0.52</td><td>0.73</td><td>33.8</td><td>0.901 <sub>(0.141)</sub></td><td>0.851 <sub>(0.145)</sub></td><td>0.340 <sub>(0.297)</sub></td></tr><tr><td>Mitra 1D (+f)</td><td>12.3</td><td>990 <sub>(+5/-5)</sub></td><td>0.48</td><td>0.69</td><td>34.5</td><td>0.901 <sub>(0.139)</sub></td><td>0.850 <sub>(0.146)</sub></td><td>0.348 <sub>(0.319)</sub></td></tr><tr><td>CatBoost</td><td>12.7</td><td>979 <sub>(+4/-5)</sub></td><td>0.47</td><td>0.68</td><td>35.8</td><td>0.898 <sub>(0.142)</sub></td><td>0.848 <sub>(0.145)</sub></td><td>0.352 <sub>(0.299)</sub></td></tr><tr><td>RealMLP</td><td>12.7</td><td>979 <sub>(+5/-4)</sub></td><td>0.47</td><td>0.66</td><td>33.2</td><td>0.895 <sub>(0.149)</sub></td><td>0.844 <sub>(0.157)</sub></td><td>0.380 <sub>(0.402)</sub></td></tr><tr><td>TabPFNv1 (+e)</td><td>12.7</td><td>979 <sub>(+4/-4)</sub></td><td>0.47</td><td>0.67</td><td>35.1</td><td>0.892 <sub>(0.152)</sub></td><td>0.837 <sub>(0.158)</sub></td><td>0.367 <sub>(0.331)</sub></td></tr><tr><td>TabForestPFN (+f)</td><td>12.8</td><td>976 <sub>(+5/-5)</sub></td><td>0.47</td><td>0.66</td><td>35.4</td><td>0.895 <sub>(0.144)</sub></td><td>0.849 <sub>(0.146)</sub></td><td>0.359 <sub>(0.327)</sub></td></tr><tr><td>TabPFNv1</td><td>13.3</td><td>960 <sub>(+5/-5)</sub></td><td>0.44</td><td>0.63</td><td>36.4</td><td>0.891 <sub>(0.153)</sub></td><td>0.836 <sub>(0.157)</sub></td><td>0.374 <sub>(0.333)</sub></td></tr><tr><td>MLP</td><td>13.8</td><td>942 <sub>(+5/-5)</sub></td><td>0.42</td><td>0.59</td><td>37.7</td><td>0.892 <sub>(0.150)</sub></td><td>0.843 <sub>(0.150)</sub></td><td>0.365 <sub>(0.314)</sub></td></tr><tr><td>XGBoost</td><td>14.3</td><td>929 <sub>(+5/-5)</sub></td><td>0.40</td><td>0.59</td><td>39.4</td><td>0.892 <sub>(0.143)</sub></td><td>0.841 <sub>(0.148)</sub></td><td>0.367 <sub>(0.303)</sub></td></tr><tr><td>Mitra 1D</td><td>14.4</td><td>924 <sub>(+5/-5)</sub></td><td>0.39</td><td>0.61</td><td>39.8</td><td>0.886 <sub>(0.151)</sub></td><td>0.834 <sub>(0.155)</sub></td><td>0.380 <sub>(0.333)</sub></td></tr><tr><td>Random Forest</td><td>14.7</td><td>915 <sub>(+5/-5)</sub></td><td>0.38</td><td>0.57</td><td>43.4</td><td>0.888 <sub>(0.150)</sub></td><td>0.836 <sub>(0.151)</sub></td><td>0.458 <sub>(0.482)</sub></td></tr><tr><td>TabForestPFN</td><td>14.7</td><td>913 <sub>(+5/-4)</sub></td><td>0.38</td><td>0.59</td><td>39.8</td><td>0.884 <sub>(0.156)</sub></td><td>0.834 <sub>(0.159)</sub></td><td>0.384 <sub>(0.348)</sub></td></tr><tr><td>LightGBM</td><td>14.8</td><td>911 <sub>(+5/-5)</sub></td><td>0.37</td><td>0.55</td><td>40.4</td><td>0.879 <sub>(0.157)</sub></td><td>0.835 <sub>(0.152)</sub></td><td>0.375 <sub>(0.308)</sub></td></tr></tbody></table>

Table 15: Mitra wins on AMLB 10-fold classification benchmark. Winner/runner-up in /. +e means ensembling in ICL, and +f means fine-tuning. The $95\%$ confidence interval is shown in parentheses for the Elo. The columns in the aggregated metrics are mean and std (shown in parentheses) of the corresponding metric.

<table><tbody><tr><td rowspan="2">Model</td><td colspan="5">Ranking Metrics</td><td colspan="3">Aggregated Metrics</td></tr><tr><td>Avg. Rank <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Elo <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Winrate <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>RAcc <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>AUC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>ACC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>CE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>Mitra (+ef)</td><td>5.8</td><td>1202 <sub>(+11/-11)</sub></td><td>0.76</td><td>0.84</td><td>17.6</td><td>0.926 <sub>(0.076)</sub></td><td>0.858 <sub>(0.124)</sub></td><td>0.341 <sub>0.292</sub></td></tr><tr><td>Attic (+ef)</td><td>6.2</td><td>1186 <sub>(+10/-10)</sub></td><td>0.74</td><td>0.83</td><td>19.4</td><td>0.926 <sub>(0.076)</sub></td><td>0.857 <sub>(0.124)</sub></td><td>0.344 <sub>(0.293)</sub></td></tr><tr><td>TabPFNv2 (+e)</td><td>6.9</td><td>1156 <sub>(+10/-10)</sub></td><td>0.71</td><td>0.81</td><td>19.7</td><td>0.927 <sub>0.0752</sub></td><td>0.858 <sub>(0.124)</sub></td><td>0.345 <sub>(0.298)</sub></td></tr><tr><td>TabICL (+e)</td><td>7.5</td><td>1129 <sub>(+9/-9)</sub></td><td>0.67</td><td>0.78</td><td>23.7</td><td>0.919 <sub>(0.080)</sub></td><td>0.847 <sub>(0.125)</sub></td><td>0.360 <sub>(0.290)</sub></td></tr><tr><td>TabPFNv2 (+ef)</td><td>7.8</td><td>1117 <sub>(+9/-9)</sub></td><td>0.66</td><td>0.76</td><td>24.2</td><td>0.921 <sub>(0.077)</sub></td><td>0.848 <sub>(0.133)</sub></td><td>0.368 <sub>(0.320)</sub></td></tr><tr><td>TabPFNv2</td><td>9.4</td><td>1059 <sub>(+9/-8)</sub></td><td>0.58</td><td>0.74</td><td>27.1</td><td>0.923 <sub>(0.077)</sub></td><td>0.852 <sub>(0.126)</sub></td><td>0.359 <sub>(0.305)</sub></td></tr><tr><td>TabICL</td><td>9.4</td><td>1058 <sub>(+9/-9)</sub></td><td>0.58</td><td>0.7</td><td>29.1</td><td>0.914 <sub>(0.087)</sub></td><td>0.841 <sub>(0.126)</sub></td><td>0.372 <sub>(0.293)</sub></td></tr><tr><td>Mitra 1D (+f)</td><td>10.6</td><td>1014 <sub>(+8/-9)</sub></td><td>0.52</td><td>0.66</td><td>30.9</td><td>0.921 <sub>(0.077)</sub></td><td>0.850 <sub>(0.124)</sub></td><td>0.360 <sub>(0.294)</sub></td></tr><tr><td>CatBoost</td><td>10.8</td><td>1006 <sub>(+8/-9)</sub></td><td>0.51</td><td>0.64</td><td>33.1</td><td>0.916 <sub>(0.078)</sub></td><td>0.844 <sub>(0.126)</sub></td><td>0.376 <sub>(0.304)</sub></td></tr><tr><td>TabForestPFN (+f)</td><td>10.8</td><td>1005 <sub>(+8/-9)</sub></td><td>0.51</td><td>0.65</td><td>30.1</td><td>0.920 <sub>(0.078)</sub></td><td>0.848 <sub>(0.126)</sub></td><td>0.364 <sub>(0.298)</sub></td></tr><tr><td>Mitra (+e)</td><td>11.2</td><td>993 <sub>(+8/-9)</sub></td><td>0.49</td><td>0.62</td><td>36.1</td><td>0.911 <sub>(0.085)</sub></td><td>0.831 <sub>(0.147)</sub></td><td>0.406 <sub>(0.350)</sub></td></tr><tr><td>Attic (+e)</td><td>11.3</td><td>986 <sub>(+9/-9)</sub></td><td>0.48</td><td>0.63</td><td>34.9</td><td>0.910 <sub>(0.085)</sub></td><td>0.832 <sub>(0.147)</sub></td><td>0.401 <sub>(0.348)</sub></td></tr><tr><td>Mitra</td><td>12.6</td><td>942 <sub>(+9/-9)</sub></td><td>0.42</td><td>0.54</td><td>39.0</td><td>0.909 <sub>(0.087)</sub></td><td>0.826 <sub>(0.147)</sub></td><td>0.415 <sub>(0.354)</sub></td></tr><tr><td>RealMLP</td><td>12.6</td><td>940 <sub>(+9/-10)</sub></td><td>0.42</td><td>0.58</td><td>35.0</td><td>0.911 <sub>(0.082)</sub></td><td>0.834 <sub>(0.138)</sub></td><td>0.406 <sub>(0.330)</sub></td></tr><tr><td>Attic</td><td>12.7</td><td>935 <sub>(+9/-9)</sub></td><td>0.41</td><td>0.58</td><td>38.1</td><td>0.908 <sub>(0.086)</sub></td><td>0.829 <sub>(0.148)</sub></td><td>0.405 <sub>(0.353)</sub></td></tr><tr><td>XGBoost</td><td>13.1</td><td>924 <sub>(+9/-10)</sub></td><td>0.4</td><td>0.54</td><td>38.2</td><td>0.912 <sub>(0.080)</sub></td><td>0.839 <sub>(0.127)</sub></td><td>0.388 <sub>(0.308)</sub></td></tr><tr><td>LightGBM</td><td>13.4</td><td>912 <sub>(+9/-9)</sub></td><td>0.38</td><td>0.51</td><td>38.0</td><td>0.910 <sub>(0.082)</sub></td><td>0.838 <sub>(0.132)</sub></td><td>0.389 <sub>(0.314)</sub></td></tr><tr><td>Random Forest</td><td>13.6</td><td>905 <sub>(+9/-9)</sub></td><td>0.37</td><td>0.5</td><td>42.3</td><td>0.908 <sub>(0.087)</sub></td><td>0.833 <sub>(0.131)</sub></td><td>0.446 <sub>(0.330)</sub></td></tr><tr><td>MLP</td><td>14.5</td><td>869 <sub>(+10/-10)</sub></td><td>0.33</td><td>0.46</td><td>39.9</td><td>0.897 <sub>(0.097)</sub></td><td>0.820 <sub>(0.146)</sub></td><td>0.424 <sub>(0.340)</sub></td></tr><tr><td>Mitra 1D</td><td>15.3</td><td>834 <sub>(+10/-10)</sub></td><td>0.28</td><td>0.44</td><td>45.1</td><td>0.899 <sub>(0.090)</sub></td><td>0.818 <sub>(0.147)</sub></td><td>0.423 <sub>(0.347)</sub></td></tr><tr><td>TabForestPFN</td><td>15.5</td><td>828 <sub>(+9/-9)</sub></td><td>0.28</td><td>0.43</td><td>44.4</td><td>0.899 <sub>(0.089)</sub></td><td>0.817 <sub>(0.146)</sub></td><td>0.423 <sub>(0.344)</sub></td></tr></tbody></table>

### C.4 Classification

We report detailed results for individual benchmarks in Table 13 (TabRepo), Table 14 (TabZilla), and Table 15 (AMLB), with aggregated results across all three benchmarks in Table 2 of the main text. These tables show that Mitra (+ef) has the best performance across all 3 of these benchmark datasets, which is consistent with the aggregated results.

To further enhance performance, we apply the advanced bagging strategy introduced in Section 4.5 of the main text to both Mitra and the top-performing baseline Attic. Table 16 shows the results of Mitra (bagging) and Attic (bagging) evaluated on the unified set of the three classification benchmark datasets. Notably, Mitra (bagging) further improves Mitra (+ef) and shows even larger gains compared to other baselines.

We additionally evaluate Mitra and baselines on the TabArena benchmark [^8] in Table 17. Mitra remains the state-of-the-art tabular models on TabArena. Specifically, Mitra achieves Pareto efficiency both in training time and inference time, with TabPFNv2 (HPO + ensemble) being significantly slower. Mitra is the strongest single model, outperforming all methods even when they perform hyperparameter tuning for 200 iterations. Mitra is only outperformed once the hyperparameter configurations of TabPFNv2 are ensembled together. We leave constructing a search space for Mitra for HPO and HPO + ensemble results as future work.

Table 16: Adding Mitra (bagging) and Attic (bagging) in the aggregated classification benchmark results in Table 2 of the main text. +e means adding ensemble in ICL, and +f means adding fine-tuning. The $95\%$ confidence interval is shown in parentheses for the Elo. The columns in the aggregated metrics are mean and std (shown in parentheses) of the corresponding metric.

<table><thead><tr><th rowspan="2">Model</th><th colspan="5">Ranking Metrics</th><th colspan="3">Aggregated Metrics</th></tr><tr><th>Avg. Rank <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th>Elo <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Winrate <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>RAcc <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>C <math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th>AUC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>ACC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>CE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr><tr><th>Mitra (bagging)</th><th>7.9</th><th>1135 <sub>(+3/-4)</sub></th><th>0.68</th><th>0.82</th><th>20.8</th><th>0.905 <sub>(0.125)</sub></th><th>0.86 <sub>(0.144)</sub></th><th>0.325 <sub>(0.318)</sub></th></tr></thead><tbody><tr><td>Mitra (+ef)</td><td>8.2</td><td>1124 <sub>(+4/-4)</sub></td><td>0.67</td><td>0.82</td><td>20.7</td><td>0.904 <sub>(0.124)</sub></td><td>0.858 <sub>(0.144)</sub></td><td>0.327 <sub>(0.318)</sub></td></tr><tr><td>Attic (bagging)</td><td>8.4</td><td>1119 <sub>(+4/-4)</sub></td><td>0.66</td><td>0.81</td><td>22.9</td><td>0.9 <sub>(0.131)</sub></td><td>0.855 <sub>(0.144)</sub></td><td>0.333 <sub>(0.319)</sub></td></tr><tr><td>Attic (+ef)</td><td>8.5</td><td>1115 <sub>(+4/-4)</sub></td><td>0.66</td><td>0.81</td><td>22.3</td><td>0.903 <sub>(0.125)</sub></td><td>0.858 <sub>(0.143)</sub></td><td>0.331 <sub>(0.318)</sub></td></tr><tr><td>TabPFNv2 (+e)</td><td>9.1</td><td>1094 <sub>(+4/-3)</sub></td><td>0.63</td><td>0.79</td><td>23.8</td><td>0.901 <sub>(0.13)</sub></td><td>0.856 <sub>(0.144)</sub></td><td>0.337 <sub>(0.319)</sub></td></tr><tr><td>TabPFNv2 (+ef)</td><td>9.7</td><td>1073 <sub>(+4/-3)</sub></td><td>0.6</td><td>0.76</td><td>25.8</td><td>0.897 <sub>(0.129)</sub></td><td>0.846 <sub>(0.151)</sub></td><td>0.362 <sub>(0.342)</sub></td></tr><tr><td>TabICL (+e)</td><td>10.7</td><td>1041 <sub>(+4/-4)</sub></td><td>0.56</td><td>0.74</td><td>31.2</td><td>0.888 <sub>(0.14)</sub></td><td>0.836 <sub>(0.15)</sub></td><td>0.366 <sub>(0.324)</sub></td></tr><tr><td>Mitra (+e)</td><td>11.0</td><td>1032 <sub>(+4/-3)</sub></td><td>0.55</td><td>0.73</td><td>31.7</td><td>0.896 <sub>(0.132)</sub></td><td>0.847 <sub>(0.148)</sub></td><td>0.359 <sub>(0.328)</sub></td></tr><tr><td>TabPFNv2</td><td>11.1</td><td>1030 <sub>(+4/-3)</sub></td><td>0.54</td><td>0.73</td><td>29.6</td><td>0.891 <sub>(0.139)</sub></td><td>0.846 <sub>(0.147)</sub></td><td>0.352 <sub>(0.324)</sub></td></tr><tr><td>Attic (+e)</td><td>11.3</td><td>1023 <sub>(+3/-3)</sub></td><td>0.53</td><td>0.73</td><td>32.2</td><td>0.896 <sub>(0.131)</sub></td><td>0.848 <sub>(0.148)</sub></td><td>0.363 <sub>(0.328)</sub></td></tr><tr><td>TabICL</td><td>11.9</td><td>1004 <sub>(+3/-3)</sub></td><td>0.51</td><td>0.7</td><td>33.9</td><td>0.884 <sub>(0.141)</sub></td><td>0.833 <sub>(0.152)</sub></td><td>0.373 <sub>(0.324)</sub></td></tr><tr><td>Mitra</td><td>12.0</td><td>1001 <sub>(+4/-3)</sub></td><td>0.5</td><td>0.68</td><td>33.5</td><td>0.891 <sub>(0.134)</sub></td><td>0.841 <sub>(0.151)</sub></td><td>0.368 <sub>(0.331)</sub></td></tr><tr><td>Mitra 1D (+f)</td><td>12.6</td><td>979 <sub>(+3/-4)</sub></td><td>0.47</td><td>0.67</td><td>34.9</td><td>0.892 <sub>(0.13)</sub></td><td>0.843 <sub>(0.15)</sub></td><td>0.365 <sub>(0.331)</sub></td></tr><tr><td>Attic</td><td>12.7</td><td>979 <sub>(+4/-4)</sub></td><td>0.47</td><td>0.67</td><td>35.7</td><td>0.884 <sub>(0.139)</sub></td><td>0.834 <sub>(0.156)</sub></td><td>0.375 <sub>(0.332)</sub></td></tr><tr><td>CatBoost</td><td>12.8</td><td>976 <sub>(+4/-4)</sub></td><td>0.47</td><td>0.67</td><td>36.0</td><td>0.888 <sub>(0.133)</sub></td><td>0.837 <sub>(0.15)</sub></td><td>0.374 <sub>(0.324)</sub></td></tr><tr><td>TabForestPFN (+f)</td><td>13.0</td><td>969 <sub>(+4/-4)</sub></td><td>0.46</td><td>0.65</td><td>35.6</td><td>0.886 <sub>(0.136)</sub></td><td>0.84 <sub>(0.148)</sub></td><td>0.375 <sub>(0.33)</sub></td></tr><tr><td>RealMLP</td><td>13.6</td><td>948 <sub>(+3/-3)</sub></td><td>0.43</td><td>0.62</td><td>35.7</td><td>0.878 <sub>(0.142)</sub></td><td>0.827 <sub>(0.164)</sub></td><td>0.411 <sub>(0.394)</sub></td></tr><tr><td>XGBoost</td><td>14.6</td><td>914 <sub>(+3/-3)</sub></td><td>0.38</td><td>0.58</td><td>39.8</td><td>0.883 <sub>(0.133)</sub></td><td>0.833 <sub>(0.149)</sub></td><td>0.388 <sub>(0.323)</sub></td></tr><tr><td>LightGBM</td><td>14.9</td><td>905 <sub>(+4/-4)</sub></td><td>0.37</td><td>0.56</td><td>40.1</td><td>0.876 <sub>(0.141)</sub></td><td>0.829 <sub>(0.153)</sub></td><td>0.392 <sub>(0.328)</sub></td></tr><tr><td>MLP</td><td>15.3</td><td>893 <sub>(+4/-4)</sub></td><td>0.35</td><td>0.51</td><td>40.4</td><td>0.869 <sub>(0.145)</sub></td><td>0.82 <sub>(0.161)</sub></td><td>0.413 <sub>(0.346)</sub></td></tr><tr><td>Random Forest</td><td>15.3</td><td>892 <sub>(+4/-4)</sub></td><td>0.35</td><td>0.54</td><td>44.5</td><td>0.874 <sub>(0.14)</sub></td><td>0.822 <sub>(0.153)</sub></td><td>0.47 <sub>(0.424)</sub></td></tr><tr><td>Mitra 1D</td><td>15.6</td><td>882 <sub>(+4/-4)</sub></td><td>0.34</td><td>0.54</td><td>42.8</td><td>0.869 <sub>(0.143)</sub></td><td>0.815 <sub>(0.163)</sub></td><td>0.414 <sub>(0.351)</sub></td></tr><tr><td>TabForestPFN</td><td>15.8</td><td>873 <sub>(+4/-4)</sub></td><td>0.33</td><td>0.52</td><td>42.8</td><td>0.864 <sub>(0.154)</sub></td><td>0.814 <sub>(0.167)</sub></td><td>0.414 <sub>(0.353)</sub></td></tr></tbody></table>

Table 17: Additional results on TabArena. “Single” represents single-model results without HPO. “HPO” represents results after 200-iteration hyperparameter optimization. “HPO + ensemble” represents the ensemble results of hyperparameter optimization. Mitra is the strongest single model, outperforming all methods even when they perform hyperparameter tuning. “Train Time” and “Infer Time” represent the median training and inference time in seconds per 1K rows. Mitra achieves Pareto efficiency both in training time and inference time.

| Model | Avg. Rank $\downarrow$ | Elo $\uparrow$ | Winrate $\uparrow$ | C $\Delta$ $\downarrow$ | Train Time | Infer Time |
| --- | --- | --- | --- | --- | --- | --- |
| TabPFNv2 (HPO + ensemble) | 6.2 | 1745 | 0.89 | 0.05 | 3445.6 | 48.2 |
| Mitra (single) | 7.4 | 1699 | 0.86 | 0.07 | 457.2 | 34.6 |
| TabM (HPO + ensemble) | 9.9 | 1620 | 0.8 | 0.1 | 2828.4 | 1.6 |
| TabICL (single) | 10.1 | 1617 | 0.8 | 0.07 | 8.9 | 1.7 |
| RealMLP (HPO + ensemble) | 10.9 | 1597 | 0.78 | 0.09 | 6796.3 | 12.4 |
| TabPFNv2 (HPO) | 10.9 | 1596 | 0.78 | 0.08 | 3445.6 | 1 |
| AutoGluon1.3 (4h) | 12.6 | 1551 | 0.74 | 0.1 | 2309.2 | 2.6 |
| TabPFNv2 (single) | 12.7 | 1548 | 0.74 | 0.1 | 4.1 | 0.4 |
| LightGBM (HPO + ensemble) | 13.7 | 1525 | 0.72 | 0.12 | 647.6 | 1.7 |
| TabM (HPO) | 14.2 | 1512 | 0.71 | 0.11 | 2828.4 | 0.2 |
| LightGBM (HPO) | 16.2 | 1470 | 0.66 | 0.12 | 647.6 | 0.3 |
| CatBoost (HPO + ensemble) | 16.5 | 1465 | 0.66 | 0.12 | 1465.9 | 0.7 |
| CatBoost (HPO) | 17.3 | 1444 | 0.64 | 0.12 | 1465.9 | 0.1 |
| TabM (single) | 17.8 | 1437 | 0.63 | 0.14 | 10.4 | 0.2 |
| CatBoost (single) | 17.8 | 1434 | 0.63 | 0.14 | 5.7 | 0.1 |
| ModernNCA (HPO) | 18 | 1428 | 0.62 | 0.12 | 5944.9 | 0.5 |
| XGBoost (HPO + ensemble) | 18.5 | 1420 | 0.61 | 0.13 | 766.1 | 1.9 |
| EBM (HPO + ensemble) | 20 | 1390 | 0.58 | 0.16 | 1109.1 | 0.2 |
| XGBoost (HPO) | 20.1 | 1386 | 0.58 | 0.14 | 766.1 | 0.3 |
| RealMLP (HPO) | 20.2 | 1383 | 0.57 | 0.13 | 6796.3 | 0.7 |
| ModernNCA (HPO + ensemble) | 20.4 | 1381 | 0.57 | 0.13 | 5944.9 | 8.4 |
| ModernNCA (single) | 20.8 | 1368 | 0.56 | 0.15 | 14.8 | 0.3 |
| TorchMLP (HPO + ensemble) | 20.9 | 1370 | 0.56 | 0.14 | 2862.1 | 2.2 |
| FastaiMLP (HPO + ensemble) | 21.1 | 1366 | 0.55 | 0.16 | 1358.6 | 8.1 |
| TabDPT (single) | 22.7 | 1329 | 0.52 | 0.15 | 27.5 | 8.9 |
| EBM (HPO) | 23.1 | 1323 | 0.51 | 0.17 | 1109.1 | 0 |
| EBM (single) | 23.8 | 1307 | 0.49 | 0.18 | 5.3 | 0.1 |
| RealMLP (single) | 25.6 | 1270 | 0.45 | 0.16 | 22.5 | 1.6 |
| FastaiMLP (HPO) | 25.6 | 1273 | 0.45 | 0.17 | 1358.6 | 0.9 |
| ExtraTrees (HPO + ensemble) | 26.1 | 1261 | 0.44 | 0.18 | 370.9 | 1.5 |
| TorchMLP (HPO) | 27 | 1241 | 0.42 | 0.16 | 2862.1 | 0.2 |
| XGBoost (single) | 28.2 | 1214 | 0.39 | 0.17 | 2.4 | 0.2 |
| ExtraTrees (HPO) | 28.6 | 1205 | 0.39 | 0.2 | 370.9 | 0.2 |
| RandomForest (HPO + ensemble) | 30.5 | 1159 | 0.34 | 0.2 | 527.4 | 1.4 |
| LightGBM (single) | 30.7 | 1159 | 0.34 | 0.18 | 2.9 | 0.1 |
| RandomForest (HPO) | 33.1 | 1094 | 0.29 | 0.21 | 527.4 | 0.1 |
| TorchMLP (single) | 33.4 | 1087 | 0.28 | 0.22 | 10.4 | 0.2 |
| FastaiMLP (single) | 34.6 | 1056 | 0.25 | 0.23 | 4.7 | 0.6 |
| Linear (HPO + ensemble) | 35.2 | 1032 | 0.24 | 0.29 | 88.6 | 0.3 |
| Linear (HPO) | 36.2 | 1005 | 0.22 | 0.29 | 88.6 | 0.1 |
| RandomForest (single) | 36.3 | 1000 | 0.21 | 0.26 | 0.4 | 0.1 |
| Linear (single) | 36.8 | 985 | 0.21 | 0.31 | 2.3 | 0.1 |
| ExtraTrees (single) | 37.7 | 952 | 0.19 | 0.28 | 0.4 | 0.1 |
| KNN (HPO + ensemble) | 42.6 | 716 | 0.08 | 0.48 | 3 | 0.2 |
| KNN (HPO) | 43.7 | 623 | 0.05 | 0.5 | 3 | 0 |
| KNN (single) | 45.2 | 415 | 0.02 | 0.59 | 0.1 | 0 |

Table 18: AMLB 10 fold regression benchmark. +e means adding ensemble in ICL, and +f means adding fine-tuning. The $95\%$ confidence interval is shown in parentheses for the Elo. The columns in the aggregated metrics are mean and std (shown in parentheses) of the corresponding metric.

<table><tbody><tr><td rowspan="2">Model</td><td colspan="5">Ranking Metrics</td><td colspan="3">Aggregated Metrics</td></tr><tr><td>Avg. Rank <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Elo <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Winrate <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>RAcc <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td><math><semantics><msup><mi>𝐑</mi> <mn>2</mn></msup> <annotation>\mathbf{R}^{2}</annotation></semantics></math> <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>RMSE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>MAE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>TabPFNv2 (+e)</td><td>4.4</td><td>1137 <sub>(+13/-12)</sub></td><td>0.69</td><td>0.82</td><td>14.8</td><td>0.683 <sub>(0.324)</sub></td><td>1720.58 <sub>(5869.98)</sub></td><td>1002.44 <sub>(3340.38)</sub></td></tr><tr><td>RealMLP</td><td>5</td><td>1097 <sub>(+12/-13)</sub></td><td>0.64</td><td>0.77</td><td>16.4</td><td>0.685 <sub>(0.317)</sub></td><td>1743.95 <sub>(5919.86)</sub></td><td>991.60 <sub>(3310.13)</sub></td></tr><tr><td>Mitra (+ef)</td><td>5.1</td><td>1091 <sub>(+12/-11)</sub></td><td>0.63</td><td>0.79</td><td>18.3</td><td>0.678 <sub>(0.322)</sub></td><td>1773.96 <sub>(6086.74)</sub></td><td>1064.63 <sub>(3520.63)</sub></td></tr><tr><td>TabPFNv2 (+ef)</td><td>5.1</td><td>1089 <sub>(+12/-11)</sub></td><td>0.63</td><td>0.77</td><td>16.7</td><td>0.672 <sub>(0.328)</sub></td><td>1749.62 <sub>(5972.03)</sub></td><td>1017.13 <sub>(3337.77)</sub></td></tr><tr><td>CatBoost</td><td>5.3</td><td>1077 <sub>(+11/-12)</sub></td><td>0.61</td><td>0.75</td><td>19.1</td><td>0.678 <sub>(0.315)</sub></td><td>1773.29 <sub>(6051.57)</sub></td><td>1083.56 <sub>(3611.41)</sub></td></tr><tr><td>TabPFNv2</td><td>5.9</td><td>1041 <sub>(+12/-12)</sub></td><td>0.56</td><td>0.69</td><td>21.2</td><td>0.669 <sub>(0.333)</sub></td><td>1889.36 <sub>(6334.77)</sub></td><td>1140.71 <sub>(3827.81)</sub></td></tr><tr><td>LightGBM</td><td>6.6</td><td>995 <sub>(+12/-12)</sub></td><td>0.49</td><td>0.65</td><td>22.3</td><td>0.676 <sub>(0.311)</sub></td><td>1827.53 <sub>(6250.13)</sub></td><td>1140.90 <sub>(3827.07)</sub></td></tr><tr><td>XGBoost</td><td>6.8</td><td>984 <sub>(+11/-12)</sub></td><td>0.47</td><td>0.65</td><td>21.7</td><td>0.670 <sub>(0.314)</sub></td><td>1804.76 <sub>(6167.97)</sub></td><td>1121.92 <sub>(3770.51)</sub></td></tr><tr><td>Mitra (+e)</td><td>7.6</td><td>933 <sub>(+12/-12)</sub></td><td>0.4</td><td>0.59</td><td>28.8</td><td>0.650 <sub>(0.332)</sub></td><td>1964.38 <sub>(6586.97)</sub></td><td>1263.88 <sub>(4148.94)</sub></td></tr><tr><td>MLP</td><td>8.3</td><td>883 <sub>(+13/-13)</sub></td><td>0.34</td><td>0.47</td><td>28.1</td><td>0.638 <sub>(0.362)</sub></td><td>2153.05 <sub>(7161.28)</sub></td><td>1105.91 <sub>(3649.86)</sub></td></tr><tr><td>Mitra</td><td>8.5</td><td>869 <sub>(+12/-12)</sub></td><td>0.32</td><td>0.51</td><td>31.6</td><td>0.641 <sub>(0.337)</sub></td><td>1992.53 <sub>(6648.48)</sub></td><td>1294.52 <sub>(4233.03)</sub></td></tr><tr><td>Random Forest</td><td>9.4</td><td>805 <sub>(+13/-14)</sub></td><td>0.24</td><td>0.37</td><td>34.9</td><td>0.639 <sub>(0.325)</sub></td><td>1960.06 <sub>(6652.34)</sub></td><td>1233.41 <sub>(4145.83)</sub></td></tr></tbody></table>

### C.5 Regression

In Table 18, we report additional regression results on a larger benchmark that combines AMLB and OpenML-CTR23 with features up to 500 and rows up to 10k. This benchmark contains more datasets with a larger number of rows than the other benchmark datasets. Results show that Mitra (+ef) and TabPFNV2 (+ef) have similar performances, with TabPFNv2 (+e) having the top performance. We choose to separate the two regression benchmarks (Table 4 and Table 18) to show differences in performance on these small-scale and large-scale regression datasets. These results show that Mitra performs better on small-scale datasets. Its performance limitation on this larger-scale dataset can be explained by the fact that in pretraining it only sees up to 16 features and up to 640 rows. Notably, it can outperform TabPFNv2 on benchmarks with up to 100 features and 3k rows, and on some benchmarks with up to 500 features and 10k rows, despite being pretrained on one-tenth of the maximum pretraining features and one-third of the maximum pretraining rows in TabPFNv2. These findings suggest that Mitra generalizes well beyond its pretraining regime. Future work includes increasing the maximum number of rows and features during the pretraining process.

### C.6 Critical Differences

We visualize the critical differences between Mitra and baselines across three classification benchmarks (TabRepo in Figure 9, TabZilla in Figure 10 and AMLB in Figure 11) and two regression benchmarks (TabRepo in Figure 12 and AMLB + OpenML-CTR23 in Figure 13).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x10.png)

Figure 9: Critical difference plot on TabRepo classification benchmark.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x11.png)

Figure 10: Critical difference plot on TabZilla classification benchmark.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x12.png)

Figure 11: Critical difference plot on AMLB classification benchmark.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x13.png)

Figure 12: Critical difference plot on TabRepo regression benchmark.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x14.png)

Figure 13: Critical difference plot on AMLB + OpenML-CTR23 regression benchmark.

### C.7 Timing Efficiency

We compute the running time of Mitra and baselines on eight 40GB A100 machines. In Figure 14, we present the performance metrics (Elo, winrate, and AUC) alongside the average running time on the TabRepo benchmark. Mitra (+ef) achieves a gain of 138 Elo over CatBoost while being approximately 3.5× faster. We have additionally measured single-GPU fine-tuning performance. Mitra (+ef) on a single GPU takes similar time (89 seconds) as CatBoost (83 seconds) while achieving a gain of 138 Elo. We present more details on training and inference times on the TabArena benchmark in Table 17.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x15.png)

Figure 14: Average running time vs. performance metrics (Elo, winrate, and AUC) on TabRepo.

### C.8 Decision Boundary Visualizations

We visualize the decision boundaries of Mitra and baseline methods on a set of representative 2D simulated datasets. Each dataset consists of 1,000 samples drawn from a known ground-truth distribution, with 10% used as support samples and the remaining 90% as query samples. For Mitra and TabPFNv2, we adopt the ICL setting without ensembling. For classical models, we use their default hyperparameters. Overall, Mitra demonstrates effective few-shot generalization capabilities.<sup>3</sup> As illustrated in Figure 15 and Figure 21, when the data distribution is axis-aligned, Mitra produces more regular and less fragmented decision boundaries than TabPFNv2. This result suggests that a lower functional complexity that appears to support better generalization. On other representative 2D datasets—GP data (Figure 16), linearly separable data (Figure 18), Gaussian mixtures (Figure 19), sine waves (Figure 20), and star-shaped distributions (Figure 23)—Mitra has decision boundaries that fall between those of tree-based classifiers and the TabPFNv2 model, which highlights the effect of pre-training on a mixture of synthetic priors. In the spiral (Figure 22) and Swiss roll (Figure 24) examples, the Gaussian Process (GP) classifier shows strong performance, which motivates our future work to incorporate GP-based priors into the pretraining mixture.

## Appendix D Limitations and Future Work

While our current mixture of priors demonstrates strong performance, it can be further improved by employing hyperparameter optimization (HPO) to adapt the mixture weights for specific downstream tasks or domains. In addition, we plan to incorporate other continuous priors, e.g., Gaussian Processes, which model smooth boundaries directly into the mixture, to better generalize to tasks outside of tabular domain, e.g., time series forecasting. Lastly, although Mitra achieves competitive results overall, it does not consistently outperform TabPFNv2 on large-feature regression tasks, and we plan to scale pretraining to datasets with larger numbers of rows and features to yield further gains in generalization to real-world, high-dimensional settings.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x16.png)

Figure 15: Decision boundaries of Mitra and baselines on 2D checkerboard data. shows more regular and less fragmented decision boundaries than TabPFNv2.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x17.png)

Figure 16: Decision boundaries of Mitra and baselines on 2D Gaussian Process data.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x18.png)

Figure 17: Decision boundaries of Mitra and baselines on 2D Julia set data.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x19.png)

Figure 18: Decision boundaries of Mitra and baselines on 2D linearly separable data.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x20.png)

Figure 19: Decision boundaries of Mitra and baselines on 2D nonlinear Gaussian mixture data.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x21.png)

Figure 20: Decision boundaries of Mitra and baselines on 2D sine wave data.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x22.png)

Figure 21: Decision boundaries of Mitra and baselines on 2D sinusoidal checkerboard data. shows more regular and less fragmented decision boundaries than TabPFNv2.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x23.png)

Figure 22: Decision boundaries of Mitra and baselines on 2D spiral data.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x24.png)

Figure 23: Decision boundaries of Mitra and baselines on 2D star data.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x25.png)

Figure 24: Decision boundaries of Mitra and baselines on 2D Swiss roll data (from 27 ).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2510.21204/assets/x26.png)

Figure 25: Large view of Figure 5, showing only the zoomed-in region for greater clarity.

[^1]: David Bonet, Daniel Mas Montserrat, Xavier Giró-i Nieto, and Alexander G Ioannidis. Hyperfast: Instant classification for tabular data. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 11114–11123, 2024.

[^2]: Felix den Breejen, Sangmin Bae, Stephen Cha, and Se-Young Yun. Fine-tuned in-context learning transformers are excellent tabular data classifiers. arXiv preprint arXiv:2405.13396, 2024.

[^3]: Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pages 785–794, 2016.

[^4]: Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.

[^5]: Felix den Breejen and Se-Young Yun. Attic: A new architecture for tabular in-context learning transformers.

[^6]: Arpad E Elo. The proposed uscf rating system, its development, theory, and applications. Chess life, 22(8):242–247, 1967.

[^7]: Nick Erickson, Jonas Mueller, Alexander Shirkov, Hang Zhang, Pedro Larroy, Mu Li, and Alexander Smola. Autogluon-tabular: Robust and accurate automl for structured data. arXiv preprint arXiv:2003.06505, 2020.

[^8]: Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas, and Frank Hutter. Tabarena: A living benchmark for machine learning on tabular data. arXiv preprint arXiv:2506.16791, 2025.

[^9]: Benjamin Feuer, Robin Schirrmeister, Valeriia Cherepanova, Chinmay Hegde, Frank Hutter, Micah Goldblum, Niv Cohen, and Colin White. Tunetables: Context optimization for scalable prior-data fitted networks. Advances in Neural Information Processing Systems, 37:83430–83464, 2024.

[^10]: Josh Gardner, Juan C Perdomo, and Ludwig Schmidt. Large scale transfer learning for tabular data via language modeling. arXiv preprint arXiv:2406.12031, 2024.

[^11]: Pieter Gijsbers, Marcos LP Bueno, Stefan Coors, Erin LeDell, Sébastien Poirier, Janek Thomas, Bernd Bischl, and Joaquin Vanschoren. Amlb: an automl benchmark. Journal of Machine Learning Research, 25(101):1–65, 2024.

[^12]: Yury Gorishniy, Akim Kotelnikov, and Artem Babenko. Tabm: Advancing tabular deep learning with parameter-efficient ensembling. arXiv preprint arXiv:2410.24210, 2024.

[^13]: Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. Revisiting deep learning models for tabular data. Advances in neural information processing systems, 34:18932–18943, 2021.

[^14]: Trevor Hastie, Robert Tibshirani, Jerome Friedman, et al. The elements of statistical learning, 2009.

[^15]: Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David Sontag. Tabllm: Few-shot classification of tabular data with large language models. In International Conference on Artificial Intelligence and Statistics, pages 5549–5581. PMLR, 2023.

[^16]: Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. Tabpfn: A transformer that solves small tabular classification problems in a second. In International Conference on Learning Representations, 2023.

[^17]: Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637(8045):319–326, 2025.

[^18]: David Holzmüller, Léo Grinsztajn, and Ingo Steinwart. Better by default: Strong pre-tuned mlps and boosted trees on tabular data. Advances in Neural Information Processing Systems, 37:26577–26658, 2024.

[^19]: Arlind Kadra, Marius Lindauer, Frank Hutter, and Josif Grabocka. Well-tuned simple nets excel on tabular datasets. Advances in neural information processing systems, 34:23928–23941, 2021.

[^20]: Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. Lightgbm: A highly efficient gradient boosting decision tree. Advances in neural information processing systems, 30, 2017.

[^21]: Myung Jun Kim, Léo Grinsztajn, and Gaël Varoquaux. Carte: pretraining and transfer for tabular learning. arXiv preprint arXiv:2402.16785, 2024.

[^22]: Anders Krogh and Jesper Vedelsby. Neural network ensembles, cross validation, and active learning. In G. Tesauro, D. Touretzky, and T. Leen, editors, Advances in Neural Information Processing Systems, volume 7. MIT Press, 1994.

[^23]: Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Hamidreza Kamkari, Alex Labach, Jesse C Cresswell, Keyvan Golestan, Guangwei Yu, Maksims Volkovs, and Anthony L Caterini. Tabdpt: Scaling tabular foundation models. arXiv preprint arXiv:2410.18164, 2024.

[^24]: Junwei Ma, Valentin Thomas, Guangwei Yu, and Anthony Caterini. In-context data distillation with tabpfn. arXiv preprint arXiv:2402.06971, 2024.

[^25]: Duncan McElfresh, Sujay Khandagale, Jonathan Valverde, Vishak Prasad C, Ganesh Ramakrishnan, Micah Goldblum, and Colin White. When do neural nets outperform boosted trees on tabular data? Advances in Neural Information Processing Systems, 36:76336–76369, 2023.

[^26]: Andreas Müller, Carlo Curino, and Raghu Ramakrishnan. Mothernet: A foundational hypernetwork for tabular classification. arXiv preprint arXiv:2312.08598, 2023.

[^27]: Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. Scikit-learn: Machine learning in python. Journal of Machine Learning Research, 12(Oct):2825–2830, 2011.

[^28]: Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. Catboost: unbiased boosting with categorical features. Advances in neural information processing systems, 31, 2018.

[^29]: Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. TabICL: A tabular foundation model for in-context learning on large data. arXiv preprint arXiv:2502.05564, 2025.

[^30]: David Salinas and Nick Erickson. Tabrepo: A large scale repository of tabular model evaluations and its automl applications. arXiv preprint arXiv:2311.02971, 2023.

[^31]: Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild, C Bayan Bruss, and Tom Goldstein. Saint: Improved neural networks for tabular data via row attention and contrastive pre-training. arXiv preprint arXiv:2106.01342, 2021.

[^32]: Boris Van Breugel and Mihaela Van Der Schaar. Why tabular foundation models should be a research priority. arXiv preprint arXiv:2405.01147, 2024.

[^33]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[^34]: Xumeng Wen, Han Zhang, Shun Zheng, Wei Xu, and Jiang Bian. From supervised to generative: A novel paradigm for tabular deep learning with large language models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3323–3333, 2024.

[^35]: Derek Xu, Olcay Cirit, Reza Asadi, Yizhou Sun, and Wei Wang. Mixture of in-context prompters for tabular pfns. arXiv preprint arXiv:2405.16156, 2024.

[^36]: Jiahuan Yan, Bo Zheng, Hongxia Xu, Yiheng Zhu, Danny Z Chen, Jimeng Sun, Jian Wu, and Jintai Chen. Making pre-trained language models great on tabular prediction. arXiv preprint arXiv:2403.01841, 2024.

[^37]: Yazheng Yang, Yuqi Wang, Guang Liu, Ledell Wu, and Qi Liu. Unitabe: A universal pretraining protocol for tabular foundation model in data science. arXiv preprint arXiv:2307.09249, 2023.

[^38]: Chao Ye, Guoshan Lu, Haobo Wang, Liyao Li, Sai Wu, Gang Chen, and Junbo Zhao. Towards cross-table masked pretraining for web data mining. In Proceedings of the ACM Web Conference 2024, pages 4449–4459, 2024.

[^39]: Bingzhao Zhu, Xingjian Shi, Nick Erickson, Mu Li, George Karypis, and Mahsa Shoaran. Xtab: Cross-table pretraining for tabular transformers. arXiv preprint arXiv:2305.06090, 2023.