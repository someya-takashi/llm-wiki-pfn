---
title: "TabICL: A Tabular Foundation Model for In-Context Learning on Large Data"
source: "https://ar5iv.labs.arxiv.org/html/2502.05564"
author:
published:
created: 2026-05-30
description: "The long-standing dominance of gradient-boosted decision trees on tabular data is currently challenged by tabular foundation models using In-Context Learning (ICL): setting the training data as context for the test dat…"
tags:
  - "clippings"
---
Jingang Qu    David Holzmüller    Gaël Varoquaux    Marine Le Morvan

###### Abstract

The long-standing dominance of gradient-boosted decision trees on tabular data is currently challenged by tabular foundation models using In-Context Learning (ICL): setting the training data as context for the test data and predicting in a single forward pass without parameter updates. While the very recent TabPFNv2 foundation model (2025) excels on tables with up to 10K samples, its alternating column- and row-wise attentions make handling large training sets computationally prohibitive. So, can ICL be effectively scaled and deliver a benefit for larger tables? We introduce TabICL, a tabular foundation model for classification, pretrained on synthetic datasets with up to 60K samples and capable of handling 500K samples on affordable resources. This is enabled by a novel two-stage architecture: a column-then-row attention mechanism to build fixed-dimensional embeddings of rows, followed by a transformer for efficient ICL. Across 200 classification datasets from the TALENT benchmark, TabICL is on par with TabPFNv2 while being systematically faster (up to 10 times), and significantly outperforms all other approaches. On 56 datasets with over 10K samples, TabICL surpasses both TabPFNv2 and CatBoost, demonstrating the potential of ICL for large data. Inference code and pre-trained models are available at [https://github.com/soda-inria/tabicl](https://github.com/soda-inria/tabicl).

Machine Learning, ICML

## 1 Introduction

Tabular data, structured in rows and columns, is widely used in industries like healthcare [^5] and finance [^23], where tabular classification problems underpin numerous real-world applications. In other data modalities, foundation models –particularly Large Language Models (LLMs) [^52] – have significantly advanced the ability to tackle new tasks and few-shot learning. This is largely due to their remarkable in-context learning (ICL) capabilities [^4], which enable them to capture patterns in prompts without requiring parameter updates. This success combined with the pervasiveness of tables have spurred interest in tabular foundation models [^44].

While LLMs are primarily designed to model natural language, attempts to fine-tune them for tabular data have emerged [^19] [^11]. These efforts rely on table serialization, which is the process of converting table rows into text or sentences which can then be tokenized. [^13] fine-tuned a Llama 3-8B on a corpus of serialized tables and demonstrated the effectiveness of this approach compared to tree-based models in the few-shot setting. However, such language models-based approaches are limited by the size of their context window to fit large serialized tables (e.g. up to 32 or 64 shots in [^13]). Moreover, it is unclear whether LLMs can effectively handle numerical values [^42]. Finally, as evidence shows that LLMs are pretrained on many popular datasets [^3], their evaluation on tabular prediction tasks should also be conducted with care.

Taking a markedly different approach, [^21] introduced TabPFN, a transformer-based tabular foundation model for classification tasks, pretrained on synthetic tabular datasets only. A key feature of TabPFN is ICL with tables, which eliminates the need for tokenization and enables efficient handling of small tables with up to 1K samples and 100 features. Very recently, the same authors introduced TabPFNv2 [^22], an improved version that significantly outperforms tree-based and neural network competitors on small-to-medium datasets with up to 10K samples and 500 features. The great promise of tabular ICL has spurred a new line of research (see Section 2.3), yet the quadratic cost of self attention is a threat to scalability of all these. TabPFNv2 uses a two-way attention mechanism alternating between column-wise and row-wise attentions, which limits its scalability for large datasets. In real-world scenarios, where industrial datasets can contain millions of samples [^38], the high computational and memory demands of TabPFNv2 hinder its practicality.

In this paper, we introduce TabICL, a scalable and efficient tabular foundation model designed for classification tasks. Pretrained on synthetic datasets with up to 60K samples, TabICL can effectively handle datasets with up to 500K samples and 500 features, significantly expanding the scalability boundaries of ICL for tables and cementing it as a foundational technique for tabular foundation models.

To handle tables of arbitrary sizes with architectural changes, TabICL treats individual cells as basic elements: each column is seen as a set of cell values capturing feature-specific distribution and semantics, and each row consists of interdependent feature values providing a holistic view of each sample. TabICL employs a two-stage architecture to achieve efficient ICL for tabular data. First, it encodes rows (excluding target labels) into dense vector embeddings. Each embedding is designed to be capable of capturing the entire table information. This stage effectively collapses the column dimension to substantially reduce computational complexity and memory footprint for subsequent ICL. Second, it combines these compact yet informative embeddings with corresponding labels and then performs ICL. Therefore, the core of TabICL lies in its embedding strategy of the first stage, which is supposed to transform rows into semantically rich embeddings.

Words and phrases in text often carry clear semantics, and are thus naturally associated to informative embeddings [^33]. However, tabular data lacks such an inherent structure: cell values can be ambiguous without metadata such as column names or data types. To address this challenge, TabICL adopts a well-constrained embedding strategy combining (1) distribution-aware column-wise feature embedding to capture statistical regularities within each column and (2) attention-based row-wise interaction to model dependencies across columns, thereby constructing semantically grounded representations for tabular data.

Feature embedding involves mapping scalar cell values into high-dimensional vectors for each feature, serving as a critical factor in model performance [^16]. Since features often exhibit vastly different distributions, previous approaches typically use feature-specific embedding modules without parameter sharing, which, however, limits cross-table transferability. In this work, we reformulate feature embedding as a *set-input* problem, where a permutation-invariant set of cell values acts as input, and the output comprises corresponding one-to-one embeddings. To achieve this, we leverage the Set Transformer [^30], a model specifically designed to process sets through efficient induced self-attention. It excels in tasks such as identifying extrema and counting unique elements, enabling the discovery of distribution-related metadata within each column and enhancing the ability to distinguish between features of different data types.

The feature embeddings are then processed per-row by another transformer, and aggregated into a single vector using learnable \[CLS\] tokens. This effectively captures complex feature interactions and accommodates a varying number of features. Overall, this column-then-row attention-based embedding achieves efficient sparse attention across all cells by leveraging the column/row inherent structure of tabular data as a strong inductive bias. Finally, the resulting row-wise embeddings are handled by a final transformer for ICL.

In addition to the above innovations, we introduce several improvements: (1) We refine pretraining synthetic datasets of TabPFN by adding a new tree-based data-generating model to incorporate the inductive biases of tree-based models [^18] [^8]; (2) We adopt curriculum learning by gradually scaling the pretraining dataset size from 1K to 60K; (3) To address classification problems with over 10 classes (the pretraining limit), we use hierarchical classification [^40], breaking them into a hierarchical structure of subproblems with $\leq$ 10 classes. The increase in the number of tasks is largely offset by the fast ICL-powered predictions of TabICL.

To summarize our contributions: (1) We present TabICL, a novel scalable tabular foundation model for classification tasks that can accommodate any number of samples, features, and classes. In practice, TabICL handles up to 500K samples and 500 features with around 20GB of GPU memory; (2) We introduce a distribution-aware feature embedding approach that handles features with diverse properties in a unified manner, unlocking new possibilities for cross-table transferability; (3) TabICL performs tasks in a single forward pass and is orders of magnitude faster than tabular methods requiring hyper-parameter tuning while still providing better performance in most cases. TabICL is also consistently faster than TabPFNv2 (up to 10 times), with efficiency gains increasing as dataset size grows; (4) We evaluate TabICL on the TALENT benchmark [^50], comprising 200 classification datasets across various domains and sizes (up to 150K samples). TabICL performs comparably to TabPFNv2 on medium-sized datasets and significantly outperforms all other methods. On the 55 large datasets with over 10K samples, TabICL surpasses both TabPFNv2 and CatBoost [^9]. These results demonstrate the potential of ICL for large data.

## 2 Related Work

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x1.png)

Figure 1: An overview of the architecture of TabICL. First, column-wise embedding transforms each cell of the input table into an embedding vector using a transformer TF col \\text{TF}\_{\\text{col}} ( Section 3.2 ), producing E. Next, row-wise interaction prepends four trainable \[CLS\] tokens to, applies rotary positional encoding, and processes row-by-row with a transformer row \\text{TF}\_{\\text{row}}. Concatenating the outputs of \[CLS\] tokens yields the final row-wise embeddings H. Finally, dataset-wise ICL operates on and uses a Transformer icl \\text{TF}\_{\\text{icl}} to predict the target labels for the test set in a single forward pass. Overall, TabICL consists of three transformers.

### 2.1 Foundation Models and In-Context Learning

In recent years, deep learning (DL) has been transformed by the emergence of foundation models, which are pretrained on massive, diverse datasets and serve as versatile backbones for downstream tasks. These transformer-based models enable In-Context Learning (ICL): performing tasks by analyzing prompts containing input-output pairs, without explicit training or parameter updates. ICL operates as a form of on-the-fly reasoning. The mechanism underlying ICL remains elusive, with prevailing explanations framing it as implicit Bayesian inference [^46] [^35], gradient descent optimization [^45], and algorithmic learning [^14].

### 2.2 Tabular Deep Learning Models

Gradient-boosted decision trees (GBDTs), such as CatBoost and XGBoost [^6], have long dominated the tabular domain. However, there are growing efforts to develop tabular DL models. Recent studies indicate a narrowing performance gap between GBDTs and tabular DL models [^49] [^17].

As tabular DL improves, cross-table transferability emerges as an important topic. Notable efforts in this direction include XTab [^53] and CARTE [^24], which incorporate transferable components that are typically shareable backbone networks and dataset-specific components that require fine-tuning for each new task. The advent of tabular foundation models can bring new possibilities to cross-table learning, paving the way for large-scale pretraining and transfer learning across tables.

### 2.3 TabPFN and its Offsprings

TabPFN, short for Tabular Prior-Data Fitted Network, is a tabular foundation model. It is a transformer pretrained on extensive synthetic datasets to perform tabular classification tasks through ICL. TabPFN interprets ICL from a Bayesian perspective as an approximate posterior predictive distribution over synthetic datasets. Several variants aim to enhance its scalability, including distilling training data into a compact learned context via prompt tuning [^32] [^12], selecting the most relevant subset of training data for each test sample [^48] [^43] [^27], replacing quadratic with linear attention [^51], and generating small task-specific neural networks via an ICL-based hypernetwork [^34]. However, most variants do not structurally improve TabPFN but instead act as prompt engineering to reduce in-context samples.

Other approaches try to improve the quality of pre-training data, such as TabForestPFN [^8] incorporating tree-based synthetic datasets and TabDPT [^31] curating and using real-world datasets.

Very recently (January 2025), TabPFNv2 was released and largely improved TabPFN in terms of both prediction performance and scalability. Our contributed model, TabICL, achieves comparable performance to TabPFNv2 while being more scalable and computationally efficient.

## 3 The TabICL Architecture

We consider a tabular classification task with an input space $\mathcal{X}\in\mathbb{R}^{m}$ and a target space $\mathcal{Y}\in[1,\cdots,C]$. Given a training dataset of input-output pairs $\mathcal{D}_{\text{train}}=\{(x_{\text{train}}^{(i)},y_{\text{train}}^{(i)})\}_{i=1}^{n_{\text{tr}}}$ and test samples $X_{\text{test}}=\{x_{\text{test}}^{(i)}\}_{i=1}^{n_{\text{te}}}$, our goal is to predict the class probabilities $p(\cdot|x_{\text{test}},\mathcal{D}_{\text{train}})$.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x2.png)

(a) Efficient induced self-attention

### 3.1 High-level Structure: Embedding then ICL

TabICL comprises two key modules: a tabular embedding module followed by an ICL module, with labels utilized exclusively in the ICL module. The tabular embedding module encodes table rows into dense vector representations while taking the inherent column-row structure into account. This module consists of two components: distribution-aware column-wise embedding, which captures statistical characteristics of individual columns, and context-aware row-wise interaction, which models dependencies between features. The ICL module subsequently processes both training and test embeddings through a transformer, enabling the prediction of the entire test set in a single forward pass through ICL. The overall architecture is depicted in Figure 1.

### 3.2 Distribution-aware Column-wise Embedding

The column-wise embedding (or feature embedding) maps each scalar cell in a column $c_{j}\in\mathbb{R}^{n}$ into a $d$ -dimensional embedding. Unlike typical approaches that assign a separate linear layer to each column [^16], we embed all columns through a shareable Set Transformer $\text{TF}_{\text{col}}$, as illustrated in Figure 2(b) and formulated as follows:

$$
\displaystyle W,B
$$
 
$$
\displaystyle=\text{TF}_{\text{col}}(c_{j})
$$
 
$$
\displaystyle\hskip-40.00006pt\in\mathbb{R}^{n\times d}
$$
 
$$
\displaystyle e_{j}
$$
 
$$
\displaystyle=W\odot c_{j}+B
$$
 
$$
\displaystyle\hskip-40.00006pt\in\mathbb{R}^{n\times d}
$$

Note that each cell in a column is assigned its own weight and bias. Essentially, feature embedding can be framed as a *set-input* problem, where $\text{TF}_{\text{col}}$ serves as a hypernetwork taking as input a permutation-invariant set of cell values and generating distribution-aware weights and biases.

The operations within $\text{TF}_{\text{col}}$ unfold as follows:

$$
\displaystyle U
$$
 
$$
\displaystyle=\text{Lin}(c)
$$
 
$$
\displaystyle\hskip-30.00005pt\in\mathbb{R}^{n\times d}
$$
 
$$
\displaystyle\leavevmode\hbox to0pt{\vbox to0pt{\pgfpicture\makeatletter\hbox{\thinspace\lower 0.0pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }{}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope\hbox to0.0pt{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}M
$$
 
$$
\displaystyle=\text{MAB}_{1}(V_{I},U_{\text{train}},U_{\text{train}})
$$
 
$$
\displaystyle\hskip-30.00005pt\in\mathbb{R}^{k\times d}
$$
 
$$
\displaystyle V
$$
 
$$
\displaystyle=\text{MAB}_{2}(U,M,M)
$$
 
$$
\displaystyle\hskip-30.00005pt\in\mathbb{R}^{n\times d}\leavevmode\hbox to0pt{\vbox to0pt{\pgfpicture\makeatletter\hbox{\thinspace\lower 0.0pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }{}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope\hbox to0.0pt{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}
$$
 
$$
\displaystyle W,B
$$
 
$$
\displaystyle=\text{Lin}(V)
$$
 
$$
\displaystyle\hskip-30.00005pt\in\mathbb{R}^{n\times d}
$$

where Lin represents a linear layer, MAB denotes multi-head attention block, and IASB stands for induced self-attention block (Figure 2(a)), introduced by [^30]. A column $c$ (omitting the subscript $j$ for simplicity) is projected into a $d$ -dimensional space ($d=128$) via a linear layer, processed by ISAB consisting of $\text{MAB}_{1}$ and $\text{MAB}_{2}$, and passed through linear layers to generate $W$ and $B$.

ISAB reduces self-attention complexity to $\mathcal{O}(n)$ while preserving the ability to capture global information through a two-stage attention mechanism. In $\text{MAB}_{1}$, inducing vectors $V_{I}$ act as queries and attend to training samples $U_{\text{train}}$ to generate induced representations $M$. In $\text{MAB}_{2}$, inputs $U$ (including both training and test samples) serve as queries and attend back to $M$, enabling global information to propagate across all training samples. We use $k=128$ inducing vectors, 4 attention heads in the MABs, and 3 ISAB blocks (only one ISAB block is shown in Equations 4 and 5 for clarity). Crucially, only train samples serve as keys and values in $\text{MAB}_{1}$. This ensures that the outputs of ISAB depend solely on training data, thereby preventing data leakage.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x4.png)

Figure 3: Learned column-wise embeddings encode statistical properties of feature distributions. Visualization of these embeddings projected onto their first two principal components for 40,000 features from synthetic datasets.

To get insights into the information captured in the learned feature embeddings, we visualize $M\in\mathbb{R}^{k\times d}$ (Equation 4) produced by the final ISAB block of $\text{TF}_{\text{col}}$. We summarize $M$ by summing along the first dimension (i.e., aggregating the induced representations of all inducing vectors) to obtain a single vector per column, and then apply Principal Component Analysis. Figure 3 reveals that columns with similar skewness (resp. kurtosis) tend to cluster together, i.e., non-symmetric distributions are differentiated from symmetrical ones (Figure 3 left), and heavy-tailed distributions are differentiated from light-tailed ones. This suggests that $\text{TF}_{\text{col}}$ encodes distributional properties in a structured manner, and cells in a column are probably embedded to reflect their statistical role (e.g., min, max, mean, mode). Therefore, the learned feature embeddings should distinguish features based on their unique distributional properties, effectively serving as feature identifiers. This contrasts with methods that rely on semantic column names [^24] or learn feature identifier vectors [^28].

### 3.3 Context-aware Row-wise Interaction

After obtaining all feature embeddings $E=[e_{1},\cdots,e_{m}]\in\mathbb{R}^{n\times m\times d}$, a 3-layer transformer with 8 attention heads, denoted by $\text{TF}_{\text{row}}$, processes $E$ for inter-feature interactions. To aggregate the embeddings into a single vector, four learnable \[CLS\] tokens are prepended to each row of $E$, and their final outputs are concatenated together. We use four tokens to provide richer representations with a total embedding size of $4\times d=512$ for subsequent ICL, while maintaining a lower embedding size ($d=128$) for $\text{TF}_{\text{col}}$ and $\text{TF}_{\text{row}}$ to reduce memory consumption.

A defining characteristic of tabular data is that columns do not have a natural ordering. Ideally, tabular methods should be invariant to permutations of columns. Unlike TabPFN (v1), TabICL naturally incorporates this invariance through the row-wise attention. However, we experimentally observed that $\text{TF}_{\text{row}}$ can suffer from a representation collapse issue. As mentioned earlier, features are identified by their distributional properties after column-wise embedding. Consequently, features originating from similar distributions thus become less distinguishable. In the extreme case where all features are drawn from the same distribution, $\text{TF}_{\text{row}}$ cannot differentiate a sample from any of its column-permuted versions, leading to nearly identical representations for originally distinct samples, i.e., representation collapse. This phenomenon is exemplified by the balance scale dataset [^39], as shown in Figure 4. In this dataset, all features follow the same discrete distribution and can take only 5 values, making collapse highly probable. After processing with $\text{TF}_{\text{row}}$, we observe that many samples collapse to the same representation (rightmost plot), despite being originally distinct (leftmost plot).

To break the symmetry between identically distributed features, we incorporate rotary positional embedding [^41] in $\text{TF}_{\text{row}}$. RoPE is widely adopted in recent LLMs [^10] and directly encodes relative positional information into the attention mechanism by rotating the query and key vectors. The rotation angle is determined by the position $p$ in the sequence and the dimension index $i$, defined as $\theta_{i}=p/(\text{base}^{2i/d})$, where $d$ is the embedding dimension and base is the frequency scaling factor. More details can be found in Appendix C.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x5.png)

Figure 4: An example of learning collapse without RoPE for the balance scale dataset. ( Upper ) The histograms show that the four features follow the same discrete distribution. ( Lower ) The t-SNE visualization of the input features and learned row-wise embeddings H with and without RoPE demonstrates that RoPE can effectively alleviate representation collapse. The colors represent 3 classes.

While the use of RoPE breaks permutational invariance, it can mitigate the representation collapse. For example, on the balance scale dataset, RoPE preserves distinct representations for different samples (Figure 4 middle). Following [^47], we set a large scaling factor of 100,000 for RoPE to enhance generalization to a larger number of features than those seen during training (up to 100). To approximately restore permutation invariance, TabICL adopts the same strategy as other TabPFN-like models by ensembling predictions over multiple column permutations.

### 3.4 Dataset-wise In-Context Learning

After converting all samples into embeddings $H\in\mathbb{R}^{n\times 4d}$, training labels are mapped to the same space as $H$ using one-hot encoding. The embeddings of $X$ and $y$ for the training set are added to create the final training embeddings $H_{\text{train}}$. We then process $H_{\text{train}}$ and $H_{\text{test}}$ using a 12-layer Transformer with 4 attention heads, denoted by $\text{TF}_{\text{icl}}$. The embeddings in $H_{\text{train}}$ can attend to one another while those in $H_{\text{test}}$ can only attend to $H_{\text{train}}$. Finally, a two-layer MLP converts the outputs of $H_{\text{test}}$ into class probabilities for the test samples.

## 4 Pretraining and Inference

### 4.1 Improved Pretraining Synthetic Datasets

TabICL is pre-trained exclusively on synthetic datasets. To ensure realistic dependencies among variables, we generate these datasets using structural causal models (SCMs), following the approach proposed in TabPFN (v1). We first sample a directed acyclic graph (DAG) to define dependencies, following the structure of a fully connected MLP, where each neuron corresponds to a variable. Each feature $c$ is then modeled as a function $f$ of its parent variables $\text{Pa}(c)$ in the graph, with added independent noise $\epsilon$, *i.e.*, $c=f(\text{Pa}(c))+\epsilon$. Compared to previous work, we enrich the dataset generation in two ways: (i) we introduce tree-based SCMs to benefit from their inductive biases, and (ii) increase the diversity of modeling functions $f$.

#### Tree-based generation

As tree-based models excel on tabular data, we introduce tree-based SCMs to leverage their ability to model complex interactions and hierarchical dependencies between variables. We define $f$ using an XGBoost regression model, as it is widely favored by practitioners and supports multi-output regression. At each layer of the graph, an XGBoost model is trained on fake targets drawn from Gaussian noise, taking the values of the parent variables as input. The obtained predictions then become the values of the child variables. To balance data generation, we combine SCMs (70%) with tree-based SCMs (30%). Appendix B gives details and examples of generated data.

#### Diversifying activation functions

In TabPFN (v1), $f$ is defined as a random affine mapping (a linear layer) with an activation function chosen from $[\text{Identity},\text{Tanh},\text{Leaky ReLU},\text{ELU}]$. We enrich this set with 15 additional activation functions to enhance the diversity of non-linear dependencies, introducing for example non-monotone or discontinuous functions. We also include random activation functions sampled from Gaussian processes with random kernels. Gaussian Process functions have been used for synthetic data generation for time series foundation models [^1], but not as activation functions and with different types of kernels. Finally, we added an option to use different activation functions across layers and applied standardization followed by random rescaling before each activation function. Figure B.1 visualizes the employed activation functions.

### 4.2 Curriculum Learning for Large-scale Pretraining

Similar to pretraining LLMs on shorter sentences before moving to longer ones, we gradually increase the size of synthetic datasets (i.e., the number of samples) while adjusting the micro batch size $N_{\mathcal{B}}$ used for gradient accumulation to accommodate memory constraints, as follows:

1. $N_{\mathcal{B}}=4$ with a fixed size of 1,024 for 100K steps;
2. $N_{\mathcal{B}}=1$ with the size randomly drawn from a log-uniform distribution between 1K and 40K over 2K steps. Activation checkpointing is enabled for datasets exceeding 10K samples, and we accordingly reduce the number of features to avoid out-of-memory issues;
3. $N_{\mathcal{B}}=1$ with the size uniformly sampled between 40K and 60K for 50 steps, training only $\text{TF}_{\text{icl}}$ while freezing all other components.

Each step consists of 512 datasets with the number of features ($\leq 100$) and classes ($\leq 10$) randomly sampled. FlashAttention and automatic mixed precision are applied globally. The pretraining took 2 weeks on three A100 GPUs with 40GB memory using PyTorch (10, 3, and 1 days for stage 1, 2, and 3, respectively). Section D.1 gives more pretraining details.

### 4.3 Hierarchical Class-extension Strategy

We tackle many-class classification problems ($>10$ classes) through hierarchical classification [^40]. Specifically, we recursively and evenly partition classes into subgroups of up to 10 classes, forming a multi-level classification tree. A classification problem with $k$ classes requires a hierarchy with depth $r=\left\lceil\log_{10}k\right\rceil$. Each node in the tree corresponds to a sub-task that predicts probabilities for its subgroup. During inference, the final probability for a given class is obtained by multiplying the probabilities across all relevant nodes from root to leaf.

As noted earlier, labels are used exclusively in the final ICL block. Consequently, the hierarchical tree is constructed during dataset-wise ICL, with all sub-tasks sharing the learned row embeddings $H$ and the same $\text{TF}_{\text{icl}}$ for ICL predictions. This kind of sharing greatly enhances the efficiency of TabICL in hierarchical classification scenarios.

### 4.4 Memory-efficient Inference

Using FlashAttention, which offers linear memory complexity with respect to sequence length, we observed that the peak activation memory can be effectively modeled by a polynomial regression during inference: $\alpha_{1}\times\text{batch\_size}+\alpha_{2}\times\text{seq\_len}+\alpha_{3}\times\text{batch\_size}\times\text{seq\_len}+\alpha_{4}$. This enables us to dynamically adjust the batch size based on sequence length and available GPU memory. The batch dimension serves different roles upon the context: it represents the number of columns for column-wise embedding, the sample size for row-wise interaction, and the number of datasets for dataset-wise ICL. Inspired by [^37], intermediate activations can be offloaded to CPU and disk as needed to further reduce memory consumption. These optimizations enable TabICL to handle datasets with 100K samples and 500 features using only 5 GB of GPU memory and 32 GB of RAM. This suffices for most real-world applications. See Section D.2 for details.

## 5 Experiments

### 5.1 Benchmark

We evaluate TabICL on the benchmark by [^50], referred to as “TALENT” here. It comprises 200 classification datasets (120 binary and 80 multiclass datasets) and comes with results for over 30 baselines. We will first analyze the 188 datasets with at most 10 classes, since these can be handled natively by TabICL and TabPFNv2.

Datasets are split into 64% training, 16% validation, and 20% test data. While most models rely on the validation set for early stopping and hyperparameter tuning, TabICL and TabPFNv2 do not. Nonetheless, we opted to train TabICL and TabPFNv2 using only the training data, which places them at a disadvantage. On the flip side, both TabICL and TabPFNv2 leverage ensembles, unlike the other deep learning models. A fair comparison would require training the other models as an ensemble as well. However, this is computationally expensive, was not part of the original benchmark, and is not implemented in these experiments. Both TabICL and TabPFNv2 average over 32 predictions obtained by randomly shuffling columns and classes and using different pre-processors. TabICL applies $z$ -normalization with or without power transformation to all input features. See [^22] for details on TabPFNv2. To avoid out-of-memory issues, we subsample training sets to 30K samples for TabPFNv2, while TabICL could predict on all datasets without subsampling. In addition, we disable automatic mixed precision across all methods during inference to ensure reproducibility and consistent time measurements.

### 5.2 Results

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x6.png)

Figure 5: Accuracy and training/inference times on the TALENT benchmark up to 10 classes. For TabICL and TabPFNv2, the time reported corresponds to training+inference on an A100 GPU. For the other models, it includes both training and hyperparameter tuning. Since 50 track only the training time using the best hyperparameters found after 100 tuning steps, we approximate the total training time by multiplying this value by 100. For TabPFN (v1), we do not report the time since the original time measurement does not include the inference time.

#### TabICL obtains state-of-the-art accuracy.

Figure 5 shows accuracies relative to the tuned MLP for all models. TabICL obtains the best median relative accuracy across all datasets while being much faster than traditional state-of-the-art models: the geometric mean training+inference time per 1K samples for TabICL is 1.1 seconds, while tuning CatBoost on a CPU takes around 3 minutes, and RealMLP and ModernNCA on a GPU take around 7 minutes. Looking at the ranks of each method based on the obtained accuracies, the critical difference diagram in Figure E.1 shows that TabICL and TabPFN2 outperform competitors by a wide margin, while the difference between the two of them is not statistically significant.

#### Speedup over TabPFNv2.

Figure 6 shows that TabICL is 1.5 $\times$ faster than TabPFNv2 on small datasets and 3-10 $\times$ faster on large datasets. This is facilitated by the hybrid architecture of TabICL, using fewer of the expensive row-wise and column-wise attention layers with smaller embedding dimension before ICL on tokenized rows. On a dataset with 10,000 samples and 100 features, TabICL takes roughly 20 seconds versus 1 minute 40 seconds for TabPFNv2, while for a dataset of 1000 samples and 10 features, TabICL takes 1 second compared to 2 seconds for TabPFNv2. In Figure A.1, we fit simple scaling laws to predict the runtime of TabICL and TabPFN2 based on the number of samples and features. These scaling laws show that the average speedup of TabICL remains around five-fold for large datasets.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x7.png)

Figure 6: Speedup of TabICL vs. TabPFNv2 on datasets with less than 30K samples and at most 10 classes.

#### TabICL enables ICL for large datasets.

While TabPFNv2 achieves excellent performance on datasets up to 10K samples, it has only been pre-trained with up to 2048 training samples and can fail on datasets above 30K samples due to its memory usage. Figure 7 shows that unlike TabPFNv2, the performance of TabICL remains strong for larger datasets. While ICL is often used in few-shot settings, this demonstrates that foundation models can compete with the best deep learning or tree-based models, even in large-sample regimes where the latter have the most potential. Additional results in Appendix A demonstrate that TabICL also performs well for many classes, features, and high ratios of categorical features.

#### TabICL produces reliable probabilities.

In many practical situations, the quality of the estimated probabilities is crucial for decision-making. Therefore, we also report the log loss (a.k.a. cross-entropy loss), which is a proper scoring rule and therefore rewards accurate prediction of probabilities [^15]. Since TabICL does not leverage hyperparameter tuning, its predictions are not specifically optimized towards accuracy. The critical difference diagrams in Appendix E show that TabICL and TabPFN2 significantly outperform accuracy-tuned competitors on the log loss, though the difference between the two is not significant, indicating that both produce more reliable probability estimates than the other models.

#### TabICL remains effective with more than 10 classes.

On datasets with more than 10 classes, we apply the hierarchical classification strategy from Section 4.3 to TabICL. Thanks to its use of label-independent row embeddings that can be shared between all sub-classifiers, TabICL scales efficiently to a large number of classes. Figure 8 shows that TabICL still achieves the second best result on these datasets in terms of mean normalized accuracy, while TabPFNv2 cannot natively handle more than 10 classes.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x8.png)

Figure 7: Model rankings as a function of sample size. Each point is the rank of one method on one dataset. Lower rank is better. The lines show the bootstrap median and 10% / 90% bootstrap confidence intervals of a piecewise linear fit.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x9.png)

Figure 8: Normalized accuracy across the 12 datasets with more than 10 classes. Colors identify tree-based models (green), deep learning models (pink), retrieval models (gray), transformers (orange), and in-context models (blue).

## 6 Conclusion

We introduce TabICL, a novel tabular foundation model that extends the scalability of existing tabular foundation models by an order of magnitude. Evaluated on datasets with up to 100K training samples, it delivers excellent performance without hyperparameter tuning, making it nearly two orders of magnitude faster than other tabular methods requiring hyper-parameter tuning. TabICL achieves this through a hybrid architecture and memory-saving optimizations. Compared to the newly released and leading tabular foundation model — TabPFNv2, our TabICL achieves comparable performance while being more scalable and faster.

#### Limitations

Like other foundation models, TabICL suffers from slow inference speed, although TabPFNv2 has demonstrated that this issue can be alleviated to some extent through caching. Currently, TabICL is limited to classification problems, but as [^22] showed, regression problems can be treated with similar methodology. Our evaluation inherits the strengths and weaknesses of the TALENT benchmark. In particular, TALENT as most other benchmarks trains single models tuned using the holdout method, while ensembling models evaluated using cross-validation can improve performance. Bagging increases computational cost, whereas TabICL can simply be trained on the full data without needing a validation set.

#### Outlook

In-context learning for tabular data was originally introduced as a speedup for small tables [^21]. The expressive power of a forward pass in a transformer may seem limited, and one might wonder if in-context learning loses its edge, given sufficient data. We find that even with large data, pre-training can create implicit priors that give in-context transformers a competitive advantage.

## Acknowledgements

We thank Léo Grinsztajn for coming up with the idea to add a tree-based part to the prior. We also thank Lennart Purucker and Samuel Müller for interesting discussions.

## References

## Appendix A Further Experiments

#### Predicting the runtime of TabICL and TabPFNv2.

To obtain rough predictions of the time for a forward pass (training and inference) in TabICL and TabPFN2, we leverage the runtime complexity of the column- and row-wise attention modules. On a table with $n$ rows and $m$ columns, intra-column attention has a time complexity of $O(n^{2}m)$ since it is performed separately for each column, while intra-row attention has a time complexity of $O(nm^{2})$. Together, we obtain a complexity of $O(nm(n+m))$, where $nm$ is the number of cells of the table. We plot this quantity on the $x$ -axis of Figure A.1 and fit models of the form

$$
\text{time }=\alpha+\beta(nm(n+m))^{\gamma}
$$

with MSLE loss, that is, a MSE loss between the log-transformed time and the log-transformed prediction. To facilitate a simpler comparison, we fix $\gamma\coloneqq 0.8$ for both models since it yields a good fit. We suspect that $\gamma=1$ should be used asymptotically, but then more additional terms would be needed to obtain a good fit. In this model, the speedup of TabICL over TabPFNv2 for large datasets approaches $5$, while it is $1.4$ for small datasets.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x10.png)

Figure A.1: Training+inference times of TabICL and TabPFNv2. Each point represents the time on one dataset on an A100 GPU. We only use datasets with less than 30,000 samples because TabPFNv2 is applied with subsampling on larger datasets to avoid RAM overflow.

#### Metafeatures.

We analyze the dependence of the ranks of TabICL, TabPFNv2, CatBoost, and ModernNCA based on different dataset metafeatures. To this end, we fit piecewise linear regression models with pre-defined nodes to predict the ranks of these models depending on a single metafeature.

Figure A.2 shows the scaling with the number of classes. The performance of TabICL deteriorates on datasets with three classes but not on datasets with more classes. It is unclear if this is really due to the number of classes or caused by a different characteristic present in the three-class datasets on the benchmark.

Figure A.3 shows the scaling with the number of features. These plots show that TabICL behaves well for large numbers of features even though it tokenizes entire rows in the middle of the network before seeing any labels.

Figure A.4 shows the dependence on the ratio of categorical to numerical variables. In general, the performance of TabICL and TabPFNv2 deteriorates somewhat in the presence of many categorical variables. However, it is notable that TabICL performs slightly better than TabPFNv2 on such datasets even though TabPFNv2 has a more sophisticated categorical feature generation in its prior.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x11.png)

Figure A.2: Dependency of the benchmark rank on the number of classes. Each point is the rank of one method on one dataset. Lower rank is better. The lines show the bootstrap median and 10% / 90% bootstrap confidence intervals of a piecewise linear fit.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x12.png)

Figure A.3: Dependency of the benchmark rank on the number of features. Each point is the rank of one method on one dataset. Lower rank is better. The lines show the bootstrap median and 10% / 90% bootstrap confidence intervals of a piecewise linear fit.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x13.png)

Figure A.4: Dependency of the benchmark rank on the ratio of categorical to overall features. Each point is the rank of one method on one dataset. Lower rank is better. The lines show the bootstrap median and 10% / 90% bootstrap confidence intervals of a piecewise linear fit.

## Appendix B Synthetic Datasets for Pretraining

### B.1 SCM prior with more activation functions

In the TabPFN prior, we replace the activation layers by the following sequence of layers:

- A standardization layer that standardizes each feature across the batch (samples) dimension
- A random rescaling layer that, for each neuron $i$, computes $x_{i}\leftarrow\exp(2a)(x_{i}+b)$, with $a,b\sim\mathcal{N}(0,1)$ sampled once per layer.
- A random activation function. With probability $1/2$, each layer uses the same type of activation function, otherwise each layer samples the type of activation function independently.

On top of the original activation functions $\{$ Identity, tanh, LeakyReLU, ELU $\}$, we add the following activation functions:

- ReLU
- ReLU6 [^29]
- SELU [^26]
- SiLU [^20]
- Softplus
- $\operatorname{Hardtanh}(x)=\max(-1,\min(1,x))$
- Signum function
- Sine
- $\mathrm{RBF}(x)=\exp(-x^{2})$
- Exponential function
- $f(x)=\sqrt{|x|}$
- $f(x)=1_{|x|\leq 1}$
- $f(x)=x^{2}$
- $f(x)=|x|$
- A random function $f(x)=\phi(x)^{\top}\mathbf{z}$, where $\mathbf{z}\sim\mathcal{N}(0,1)$ and the feature map $\phi$ is defined randomly as
	$$
	\displaystyle\phi(x)
	$$
	 
	$$
	\displaystyle\coloneqq\frac{\mathbf{w}}{\|\mathbf{w}\|_{2}}\odot\sin(\mathbf{a}x+\mathbf{b})\in\mathbb{R}^{N},
	$$
	$$
	\displaystyle N
	$$
	 
	$$
	\displaystyle\coloneqq 256,
	$$
	$$
	\displaystyle b_{i}
	$$
	 
	$$
	\displaystyle\sim\mathcal{U}[0,2\pi],
	$$
	$$
	\displaystyle a_{i}
	$$
	 
	$$
	\displaystyle\sim\mathcal{U}[0,N],
	$$
	$$
	\displaystyle w_{i}
	$$
	 
	$$
	\displaystyle\coloneqq a_{i}^{-\exp(u)},
	$$
	$$
	\displaystyle u
	$$
	 
	$$
	\displaystyle\sim\mathcal{U}[0.7,3.0]~.
	$$
	Here, all random parameters are drawn once per layer, and $\odot$ is an element-wise product. This is motivated by the fact that for fixed feature map $\phi$, the random function $f(x)=\phi(x)^{\top}\mathbf{z}$ is a Gaussian process with covariance kernel $k(x,x^{\prime})=\phi(x)^{\top}\phi(x^{\prime})$. The design of $\phi$ is inspired by random Fourier features [^36]. The randomly drawn exponent $-\exp(u)$ leads to different decays of coefficients with increasing frequency, which produces different levels of smoothness for the sampled function as shown in Figure B.1 (right) [^7]. This activation function applies standardization directly before the random function, such that the random rescaling has no effect.

Here, the random function is sampled with a probability ten times higher than the other activation functions to account for the fact that it can represent many different functions.

Figure B.1 visualizes the used activation functions. Figure B.2 shows datasets from the resulting prior.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x14.png)

Figure B.1: Activation functions from the SCM prior. Left: Non-random activation functions from the SCM prior without standardization and random rescaling. Activation functions are randomly flipped along the x - and/or y -axis to fully utilize the space in the plot. Right: Random instantiations of the random activation function from the SCM prior, including the automatic standardization of the inputs.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x16.png)

Figure B.2: Randomly generated 2D datasets from the SCM prior. The color corresponds to the class label.

### B.2 Tree-based SCM prior

The tree-based SCM prior replaces the linear and activation layers in the SCM prior by XGBoost models fitted on random data. More specifically, these models are generated as follows:

- Sample n\_estimators and max\_depth independently as $\min\{4,1+\operatorname{Exponential}(\lambda=0.5)\}$ and $\min\{4,2+\operatorname{Exponential}(\lambda=0.5)\}$, respectively.
- For each layer with $n$ input and $m$ output neurons, fit an XGBoost multi-output regressor with the parameters above on the layer inputs $x_{i}$ with standard normal targets $y_{i}\in\mathbb{R}^{m}$, then use the fitted model to predict on the given inputs.

Figure B.3 shows datasets from the tree SCM prior.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x17.png)

Figure B.3: Randomly generated 2D datasets from the tree-based SCM prior. The color corresponds to the class label.

## Appendix C Rotary Positional Embedding

Rotary Positional Encoding (RoPE) is a technique used in Transformer models to represent positional information in the input data. RoPE encodes positional information directly into the attention mechanism through a rotation matrix applied to the query and key vectors in self-attention. Given a position $p$, a frequency $\omega$, and a vector $x$ (query or key vector):

$$
x_{p}=\text{RoPE}(x,p)=R(p)x
$$

where $R(p)$ is a rotation matrix applied to the vector $x$, encoding positional information using sinusoidal functions. The rotation matrix $R(p)$ is based on a sinusoidal function and applies rotations independently to 2-dimensional subspaces of $x$. For each 2D pair of vector components $(x_{2i},x_{2i+1})$, the rotation is defined as:

$$
\displaystyle R(p)\begin{bmatrix}x_{2i}\\
x_{2i+1}\end{bmatrix}
$$
 
$$
\displaystyle\coloneqq\begin{bmatrix}\cos(\theta_{i})&-\sin(\theta_{i})\\
\sin(\theta_{i})&\cos(\theta_{i})\end{bmatrix}\begin{bmatrix}x_{2i}\\
x_{2i+1}\end{bmatrix},
$$
$$
\displaystyle\theta_{i}
$$
 
$$
\displaystyle\coloneqq\frac{p}{10000^{2i/d}}
$$

where $d$ is the dimensionality of the embedding and $\omega_{i}=10000^{2i/d}$ determines the frequency for each dimension. The above equation indicates that for each dimension pair $(2i,2i+1)$, the vector is rotated by an angle proportional to the position $p$ and the frequency $\omega_{i}$. RoPE directly modifies the query $Q$ and key $K$ vectors in the self-attention mechanism:

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{(R(p_{Q})Q)\cdot(R(p_{K})K)^{T}}{\sqrt{d}}\right)V~.
$$

Because relative positional information is preserved in the inner product, the attention scores naturally encode the relative distance between positions $p_{Q}$ and $p_{K}$.

[^2] provides an alternative perspective on RoPE that aligns well with its use in our work. At lower indices $i$, the rotation angle changes more rapidly with increasing $p$, resulting in high-frequency oscillations that resemble random noise and encode positional information. Conversely, at higher indices $i$, the rotation angle changes more slowly with increasing $p$, producing stable values that carry semantic information. In our case, RoPE effectively introduces noise to each feature as its identifier in a controlled, predictable, and generalizable manner.

RoPE is also claimed to exhibit long-term decay, where tokens become less correlated as their relative distance increases. However, this claim is questioned by [^2], as it relies on an unrealistic oversimplification that queries and keys are equal.

## Appendix D Setup of TabICL

### D.1 Pretraining details

As outlined in the main paper, we employ curriculum learning to progressively increase the size of synthetic datasets (i.e., the number of samples) during pretraining. This process unfolds in three stages by adjusting the micro-batch size used for gradient accumulation:

1. $N_{\mathcal{B}}=4$ with a fixed size of 1,024 the first 100K steps.
2. $N_{\mathcal{B}}=1$ with the size randomly drawn from a log-uniform distribution between 1K and 40K over 2K steps. Activation checkpointing is enabled for datasets exceeding 10K samples, and we accordingly reduce the number of features to avoid out-of-memory issues.
3. $N_{\mathcal{B}}=1$ with the size uniformly sampled between 40K and 60K for 50 steps, training only $\text{TF}_{\text{icl}}$ while all other components remain frozen.

Each step comprises 512 datasets. In the first stage, all datasets contain an equal number of samples. In the second and third stages, datasets in each micro-batch have the same number of samples, but the number of samples varies between different micro-batches. We use Adam [^25] and clip the gradient norm to 1. The learning rate schedules for pretraining are shown in Figure 1(c).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x18.png)

(a) Cosine decay with warmup for stage 1

### D.2 Memory-efficient inference

By employing FlashAttention, we have observed that the inference peak GPU memory consumption of the three transformers of TabICL for column-wise embedding $\text{TF}_{\text{col}}$, row-wise interaction $\text{TF}_{\text{row}}$, and dataset-wise ICL $\text{TF}_{\text{icl}}$ can be well approximated through the following polynomial regression:

$$
\text{MEM}=\alpha_{1}\times\text{batch\_size}+\alpha_{2}\times\text{seq\_len}+\alpha_{3}\times\text{batch\_size}\times\text{seq\_len}+\alpha_{4}
$$

It is important to note that the specific meanings of batch size and sequence length vary across different transformers, as outlined below:

|  | Batch Size | Sequence Length |
| --- | --- | --- |
| $\text{TF}_{\text{col}}$ | Number of features | Number of samples |
| $\text{TF}_{\text{row}}$ | Number of samples | Number of features |
| $\text{TF}_{\text{icl}}$ | Number of datasets | Number of samples |

Table D.1: Notions of batch size and sequence length for different transformers

Given an input $X\in\mathbb{R}^{b\times n\times m}$, where $b$, $n$, and $m$ represent the number of datasets, the number of samples, and the number of features, respectively, $X$ is first reshaped to $\mathbb{R}^{(b\times m)\times n}$ and processed by $\text{TF}_{\text{col}}$ to get $E=\mathbb{R}^{(b\times m)\times n\times d}$. Subsequently, $E$ is reshaped to $\mathbb{R}^{(b\times n)\times m\times d}$ and passed to $\text{TF}_{\text{row}}$, which generates $H\in\mathbb{R}^{b\times n\times 4d}$. Finally, $H$ is fed into $\text{TF}_{\text{icl}}$ to predict the test set entirely through ICL. We can see that it is necessary to set appropriate batch sizes for different transformers in order to efficiently utilize GPU resources and avoid out-of-memory errors. This is precisely where the aforementioned polynomial regression comes into play.

To this end, we first systematically tracked peak GPU memory usage of different transformers by varying both batch size and sequence length on a A100 GPU with 40GB memory, and then we fit the parameters of the above polynomial regression to the tracked data, as shown below:

$$
\displaystyle\text{MEM}_{\text{col}}
$$
 
$$
\displaystyle=(0.0708\times\text{batch\_size})+(7.29\times 10^{-6}\times\text{seq\_len})+(0.00391\times\text{batch\_size}\times\text{seq\_len})+137.62
$$
 
$$
\displaystyle\text{MEM}_{\text{row}}
$$
 
$$
\displaystyle=(-2.07\times 10^{-5}\times\text{batch\_size})+(2.27\times 10^{-4}\times\text{seq\_len})+(0.00537\times\text{batch\_size}\times\text{seq\_len})+138.54
$$
 
$$
\displaystyle\text{MEM}_{\text{icl}}
$$
 
$$
\displaystyle=(-0.260\times\text{batch\_size})+(4.77\times 10^{-7}\times\text{seq\_len})+(0.0195\times\text{batch\_size}\times\text{seq\_len})+140.58
$$

The estimated memory is measured in megabytes (MB).

In addition to adjusting the batch size, we also offload intermediate activations to the CPU or disk as needed to further alleviate GPU memory constraints. Figure D.2 illustrates the CPU and GPU memory consumption for a large dataset containing 100K samples and 500 features (80% training set and 20% test set). As shown, only 5GB of GPU memory and 25GB of CPU memory are utilized, making this a highly affordable computational setup.

Figure D.3 shows the CPU and GPU memory consumption for a larger dataset containing 500K samples and 500 features (80% training set and 20% test set). As depicted, the computation requires less than 14GB of GPU memory. The CPU memory usage reaches approximately 120GB, which, however, can be significantly reduced through optional disk offloading via memory mapping.

We can observe that GPU memory consumption exhibits periodic fluctuations during the column-wise embedding and row-wise interaction phases. This is because the entire dataset is automatically divided into multiple batches, with the batch size determined dynamically based on the polynomial regression mentioned earlier. Additionally, during column-wise embedding, the output of $\text{TF}_{\text{col}}$ is progressively offloaded to CPU. As a result, we can see an incremental increase in CPU memory usage throughout this stage.

We can also see that enabling automatic mixed precision highly reduces both memory consumption and computation time.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x21.png)

(a) CPU and GPU Memory Usage without Automatic Mixed Precision

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x23.png)

(a) CPU and GPU Memory Usage without Automatic Mixed Precision

## Appendix E Average Performance and Rankings

In this section, we present the average rank of all methods across different dataset categories, including binary datasets, multi-class datasets ($\leq$ 10 classes), small classification datasets ($\leq$ 10K samples), and large classification datasets ($>$ 10K samples). The rankings are computed based on accuracy, AUC, and Log Loss. The average rank is given in critical difference diagrams, with Wilcoxon-Holm tests with a significance level 0.05. The lower the rank value, the better the performance.

It is important to note that accuracy is used as the objective metric for hyperparameter tuning in tabular methods.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x25.png)

(a) Accuracy

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x28.png)

(a) Accuracy

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x31.png)

(a) Accuracy

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x34.png)

(a) Accuracy

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2502.05564/assets/x37.png)

(a) Accuracy

[^1]: Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P., Kapoor, S., et al. Chronos: Learning the language of time series. *arXiv preprint arXiv:2403.07815*, 2024.

[^2]: Barbero, F., Vitvitskyi, A., Perivolaropoulos, C., Pascanu, R., and Veličković, P. Round and Round We Go! What makes Rotary Positional Encodings useful?, October 2024.

[^3]: Bordt, S., Nori, H., and Caruana, R. Elephants never forget: Testing language models for memorization of tabular data. *arXiv preprint arXiv:2403.06644*, 2024.

[^4]: Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language Models are Few-Shot Learners, July 2020.

[^5]: Cartella, F., Anunciacao, O., Funabiki, Y., Yamaguchi, D., Akishita, T., and Elshocht, O. Adversarial Attacks for Tabular Data: Application to Fraud Detection and Imbalanced Data, January 2021.

[^6]: Chen, T. and Guestrin, C. XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794, August 2016. doi: 10.1145/2939672.2939785.

[^7]: Da Costa, N., Pförtner, M., Da Costa, L., and Hennig, P. Sample path regularity of gaussian processes from the covariance kernel. *arXiv preprint arXiv:2312.14886*, 2023.

[^8]: den Breejen, F., Bae, S., Cha, S., and Yun, S.-Y. Why In-Context Learning Transformers are Tabular Data Classifiers, May 2024.

[^9]: Dorogush, A. V., Ershov, V., and Gulin, A. CatBoost: Gradient boosting with categorical features support, October 2018.

[^10]: Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.

[^11]: Fang, X., Xu, W., Tan, F. A., Zhang, J., Hu, Z., Qi, Y., Nickleach, S., Socolinsky, D., Sengamedu, S., and Faloutsos, C. Large language models(llms) on tabular data: Prediction, generation, and understanding – a survey, 2024.

[^12]: Feuer, B., Schirrmeister, R. T., Cherepanova, V., Hegde, C., Hutter, F., Goldblum, M., Cohen, N., and White, C. TuneTables: Context Optimization for Scalable Prior-Data Fitted Networks. *arXiv preprint arXiv:2402.11137*, 2024.

[^13]: Gardner, J., Perdomo, J. C., and Schmidt, L. Large Scale Transfer Learning for Tabular Data via Language Modeling, June 2024.

[^14]: Garg, S., Tsipras, D., Liang, P., and Valiant, G. What Can Transformers Learn In-Context? A Case Study of Simple Function Classes, August 2023.

[^15]: Gneiting, T. and Raftery, A. E. Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477):359–378, 2007.

[^16]: Gorishniy, Y., Rubachev, I., and Babenko, A. On embeddings for numerical features in tabular deep learning. *Advances in Neural Information Processing Systems*, 35:24991–25004, 2022.

[^17]: Gorishniy, Y., Kotelnikov, A., and Babenko, A. TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling, November 2024.

[^18]: Grinsztajn, L., Oyallon, E., and Varoquaux, G. Why do tree-based models still outperform deep learning on tabular data?, July 2022.

[^19]: Hegselmann, S., Buendia, A., Lang, H., Agrawal, M., Jiang, X., and Sontag, D. Tabllm: Few-shot classification of tabular data with large language models. In *International Conference on Artificial Intelligence and Statistics*, pp. 5549–5581. PMLR, 2023.

[^20]: Hendrycks, D. and Gimpel, K. Gaussian error linear units (gelus). *arXiv preprint arXiv:1606.08415*, 2016.

[^21]: Hollmann, N., Müller, S., Eggensperger, K., and Hutter, F. Tabpfn: A transformer that solves small tabular classification problems in a second. *arXiv preprint arXiv:2207.01848*, 2022.

[^22]: Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., and Hutter, F. Accurate predictions on small data with a tabular foundation model. *Nature*, 637(8045):319–326, January 2025. ISSN 1476-4687. doi: 10.1038/s41586-024-08328-6.

[^23]: Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L.-w. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Anthony Celi, L., and Mark, R. G. MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3(1):160035, May 2016. ISSN 2052-4463. doi: 10.1038/sdata.2016.35.

[^24]: Kim, M. J., Grinsztajn, L., and Varoquaux, G. CARTE: Pretraining and Transfer for Tabular Learning, May 2024.

[^25]: Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014.

[^26]: Klambauer, G., Unterthiner, T., Mayr, A., and Hochreiter, S. Self-normalizing neural networks. *Neural Information Processing Systems*, 30, 2017.

[^27]: Koshil, M., Nagler, T., Feurer, M., and Eggensperger, K. Towards Localization via Data Embedding for TabPFN. 2024.

[^28]: Kossen, J., Band, N., Lyle, C., Gomez, A. N., Rainforth, T., and Gal, Y. Self-attention between datapoints: Going beyond individual input-output pairs in deep learning. *Advances in Neural Information Processing Systems*, 34:28742–28756, 2021.

[^29]: Krizhevsky, A. Convolutional deep belief networks on cifar-10.

[^30]: Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S., and Teh, Y. W. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks, May 2019.

[^31]: Ma, J., Thomas, V., Hosseinzadeh, R., Kamkari, H., Labach, A., Cresswell, J. C., Golestan, K., Yu, G., Volkovs, M., and Caterini, A. L. TabDPT: Scaling Tabular Foundation Models, October 2024a.

[^32]: Ma, J., Thomas, V., Yu, G., and Caterini, A. In-Context Data Distillation with TabPFN. *arXiv preprint arXiv:2402.06971*, 2024b.

[^33]: Mikolov, T., Chen, K., Corrado, G., and Dean, J. Efficient Estimation of Word Representations in Vector Space, September 2013.

[^34]: Müller, A., Curino, C., and Ramakrishnan, R. MotherNet: A Foundational Hypernetwork for Tabular Classification, December 2023.

[^35]: Müller, S., Hollmann, N., Arango, S. P., Grabocka, J., and Hutter, F. Transformers Can Do Bayesian Inference, August 2024.

[^36]: Rahimi, A. and Recht, B. Random features for large-scale kernel machines. *Neural Information Processing Systems*, 20, 2007.

[^37]: Rajbhandari, S., Ruwase, O., Rasley, J., Smith, S., and He, Y. ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning, April 2021.

[^38]: Rubachev, I., Kartashev, N., Gorishniy, Y., and Babenko, A. Tabred: Analyzing pitfalls and filling the gaps in tabular deep learning benchmarks. *arXiv preprint arXiv:2406.19380*, 2024.

[^39]: Siegler, R. Balance Scale. UCI Machine Learning Repository, 1976. DOI: https://doi.org/10.24432/C5488X.

[^40]: Silla, C. N. and Freitas, A. A. A survey of hierarchical classification across different application domains. *Data Mining and Knowledge Discovery*, 22(1):31–72, January 2011. ISSN 1573-756X. doi: 10.1007/s10618-010-0175-9.

[^41]: Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. RoFormer: Enhanced Transformer with Rotary Position Embedding, November 2023.

[^42]: Thawani, A., Pujara, J., Szekely, P. A., and Ilievski, F. Representing Numbers in NLP: A Survey and a Vision, March 2021.

[^43]: Thomas, V., Ma, J., Hosseinzadeh, R., Golestan, K., Yu, G., Volkovs, M., and Caterini, A. Retrieval & Fine-Tuning for In-Context Tabular Models, June 2024.

[^44]: van Breugel, B. and van der Schaar, M. Why Tabular Foundation Models Should Be a Research Priority, June 2024.

[^45]: von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., and Vladymyrov, M. Transformers learn in-context by gradient descent, May 2023.

[^46]: Xie, S. M., Raghunathan, A., Liang, P., and Ma, T. An Explanation of In-context Learning as Implicit Bayesian Inference, July 2022.

[^47]: Xiong, W., Liu, J., Molybog, I., Zhang, H., Bhargava, P., Hou, R., Martin, L., Rungta, R., Sankararaman, K. A., Oguz, B., Khabsa, M., Fang, H., Mehdad, Y., Narang, S., Malik, K., Fan, A., Bhosale, S., Edunov, S., Lewis, M., Wang, S., and Ma, H. Effective Long-Context Scaling of Foundation Models, November 2023.

[^48]: Xu, D., Cirit, O., Asadi, R., Sun, Y., and Wang, W. Mixture of In-Context Prompters for Tabular PFNs, May 2024.

[^49]: Ye, H.-J., Yin, H.-H., and Zhan, D.-C. Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later, July 2024.

[^50]: Ye, H.-J., Liu, S.-Y., Cai, H.-R., Zhou, Q.-L., and Zhan, D.-C. A Closer Look at Deep Learning Methods on Tabular Datasets, January 2025.

[^51]: Zeng, Y., Kang, W., and Mueller, A. C. Tabflex: Scaling tabular learning to millions with linear attention. In *NeurIPS 2024 Third Table Representation Learning Workshop*, 2024.

[^52]: Zhou, C., Li, Q., Li, C., Yu, J., Liu, Y., Wang, G., Zhang, K., Ji, C., Yan, Q., He, L., Peng, H., Li, J., Wu, J., Liu, Z., Xie, P., Xiong, C., Pei, J., Yu, P. S., and Sun, L. A comprehensive survey on pretrained foundation models: A history from BERT to ChatGPT. *International Journal of Machine Learning and Cybernetics*, November 2024. ISSN 1868-808X. doi: 10.1007/s13042-024-02443-6.

[^53]: Zhu, B., Shi, X., Erickson, N., Li, M., Karypis, G., and Shoaran, M. XTab: Cross-table Pretraining for Tabular Transformers, May 2023.