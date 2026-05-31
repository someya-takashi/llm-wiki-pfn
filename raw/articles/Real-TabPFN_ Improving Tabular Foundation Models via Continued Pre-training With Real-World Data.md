---
title: "Real-TabPFN: Improving Tabular Foundation Models via Continued Pre-training With Real-World Data"
source: "https://ar5iv.labs.arxiv.org/html/2507.03971"
author:
published:
created: 2026-05-31
description: "Foundation models for tabular data, like TabPFN, achieve strong performance on small datasets when pre-trained solely on synthetic data. We show that this performance can be significantly boosted by a targeted continue…"
tags:
  - "clippings"
---
Anurag Garg    Muhammad Ali    Noah Hollmann    Lennart Purucker    Samuel Müller    Frank Hutter

###### Abstract

Foundation models for tabular data, like TabPFN, achieve strong performance on small datasets when pre-trained solely on synthetic data. We show that this performance can be significantly boosted by a targeted continued pre-training phase. Specifically, we demonstrate that leveraging a small, curated collection of large, real-world datasets for continued pre-training yields superior downstream predictive accuracy compared to using broader, potentially noisier corpora like CommonCrawl or GitTables. Our resulting model, Real-TabPFN, achieves substantial performance gains on 29 datasets from the OpenML AutoML Benchmark.

Tabular Foundation Model, TabPFN

## 1 Introduction

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2507.03971/assets/x1.png)

Figure 1: Per Dataset Normalized ROC Comparison of TabPFN (default) and Real-TabPFN (ours) on the 29 datasets from the OpenML AutoML Benchmark Datasets. Wilcoxon p refers to the two-sided Wilcoxon signed-rank test p value.

Until recently, traditional tree-based algorithms like XGBoost [^3] and CatBoost [^7] have consistently outperformed neural networks on tabular prediction tasks [^12]. However, TabPFNv2 has recently demonstrated improved performance on small datasets (up to 10,000 samples and 500 features), significantly advancing deep learning for tabular data.

Although TabPFNv2 delivers strong average performance, it is still not universally best-in-class: many datasets are better handled by well-tuned tree ensembles or by careful hyper-parameter searches on TabPFNv2 itself. It is not easy to improve the TabPFNv2 model further, since it has already been exhaustively trained on over 100 million synthetic tables that approximate a broad prior over tabular problems. However, even small accuracy gains could translate into tangible benefits, such as fewer hospital re-admissions or more precise credit-risk scoring, domains in which tabular data dominates.

We enhance TabPFNv2’s in-context learning by continuing to pre-train it on a carefully selected set of real-world tables from OpenML [^30] and Kaggle <sup>1</sup>. The resulting model, Real-TabPFN, consistently outperforms TabPFNv2 on the OpenML AutoMLBenchmark classification tasks (see Figure 1). In practice, Real-TabPFN serves as a stronger off-the-shelf baseline for tabular classification than the default TabPFNv2 model.

Our contributions are:

- We empirically show that using real-world datasets with synthetic data during the pre-training of a tabular foundation model boosts in-context learning performance, opening a promising research direction.
- The new model Real-TabPFN and its public weights <sup>2</sup>, an extension of TabPFNv2 obtained by continued pre-training on real-world data. Real-TabPFN’s in-context learning outperforms its predecessor on $29$ small tabular datasets.

## 3 Pre-training and Evaluation Data

To study continued pre-training, we had to decide on the data for continuing pre-training and for evaluation.

Evaluation Data. We adopt the same datasets as used by [^15] for TabPFNv2 to evaluate our method. We use the same 29 datasets from the OpenML AutoML Benchmark [^11]; see Appendix B. All datasets contain up to 10,000 samples and 500 features.

Continued Pre-training Data. Unlike the domains of natural language processing and computer vision, where many carefully curated datasets, such as ImageNet [^6], COCO [^21], FineWeb [^26], and C4 [^28] are available, comparably high-quality resources for tabular learning remain scarce.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2507.03971/assets/plots/all_data_sizes_720p.png)

Figure 2: Distribution of dataset sizes (number of rows and features) from various sources. The prevalence of smaller datasets in broad corpora like CommonCrawl and GitTable contrasts with the larger datasets from OpenML and Kaggle.

Recent efforts have attempted to address this gap. Notable contributions include the Web Table Corpus [^1], TabLib [^9], and GitTables [^16]. Beyond these, researchers either generate synthetic datasets or rely on existing repositories like UCI [^18], OpenML [^30], and Kaggle. We adopt the latter, manually curating $71$ high-quality datasets from OpenML and Kaggle. Figure 2 shows the distribution of the number of features and rows across datasets from different sources; Appendix A lists the curated datasets used for continued pre-training. We apply minimal preprocessing to the $71$ datasets: categorical features are encoded with Scikit-learn’s [^25] OrdinalEncoder; and if the target variable has more than ten classes, we retain the nine most common classes and merge the remainder into a single tenth class.

### Data Contamination.

We carefully avoided data contamination [^17] between training and the evaluation set. We implemented a multi-tiered filtering process: (1) We only select datasets exceeding 10,000 samples since all our evaluation datasets have fewer samples. (2) We cross-referenced dataset IDs, names, and shapes to identify potential duplicates. (3) We compared feature names across datasets to detect similar or identical data structures. (4) We generated hashes of both rows and columns to identify potential data duplications at a granular level. (5) We manually inspected metadata to ensure no training and evaluation datasets share a common source dataset or a subsampled version of that dataset.  
We exclude any dataset from the pre-training data that does not meet these criteria.

## 4 Method: Continued Pre-training of TabPFN

Our method bridges purely synthetic training (e.g., TabPFN [^15] [^14]) and purely real-data training (e.g., TabDPT [^24]) by leveraging the complementary strengths of both paradigms.

Concretely, we adopt a *two-stage approach*. Stage 1 relies on the original TabPFNv2 checkpoint, pre-trained by [^15] on a large, diverse set of synthetic tables. This serves as our starting point. Stage 2 continues pre-training exclusively on a curated collection of heterogeneous real-world tables. This approach contrasts with *mixed* training, where synthetic and real samples are fed to the model simultaneously, as in [^8]. We opted for a two-stage approach as it is easier to apply and builds directly on a strong existing synthetic base model.

Although continued pre-training has shown remarkable success in language models [^13], its potential for tabular foundation models remains largely unexplored. By pre-training on diverse real datasets rather than narrow task-specific data, our approach improves generalization while preserving cross-domain adaptability.

To enable robust continued pre-training, we retained the original TabPFNv2 architecture and trained with a reduced learning rate of $3\times 10^{-7}$ using the AdamW optimizer [^22] together with a linear warm‑up followed by a cosine annealing schedule [^23].

Moreover, we added a regularizer penalizing distance to the L2-Starting-Point (L2-SP) [^20] to the pre-training objective. This penalizes large deviations from the initial pre-trained weights, and is used to mitigate catastrophic forgetting [^19]. More precisely, let $\mathbf{w}^{0}$ denote the parameter vector of the pre-trained base model from which continued pre-training begins. The L2-SP penalty then regularizes the model parameters towards this initial vector. It is formally defined as:

$$
\Omega(\mathbf{w})=\frac{\alpha}{2}\left\lVert\mathbf{w}-\mathbf{w}^{0}\right\rVert_{2}^{2},
$$

where $\alpha$ controls the strength of the regularization penalty, and $\left\lVert\cdot\right\rVert_{2}$ denotes the L2 norm. We add the L2 norm to the cross-entropy loss: $\mathcal{L}\;=\;\mathcal{L}_{\text{CE}}\;+\;\Omega(\mathbf{w})$ to obtain our final pre-training objective. We used a regularization strength $\alpha$ of 0.003.

We continued pre-training for 20,000 steps with a batch size of 1 (i.e., a single dataset). Choosing a batch size of 1 is a simple approach that naturally handles the varying feature dimensions of real-world datasets without requiring padding or truncation. Critically, it also allowed us to maximize the training context for each dataset up to 20,000 samples, limited primarily by GPU memory, rather than by batching constraints. For datasets larger than this limit, we sample uniformly up to 20,000 rows. Per batch, we split the data into 60% context (TabPFN’s training data) and 40% query (TabPFN’s testing data) for the forward pass. The training was performed on a single Nvidia RTX 2080 Ti GPU. To stay within GPU memory limits, we further capped each dataset at 400,000 total cells, adjusting the number of samples accordingly for datasets with too many attributes.

## 5 Experiments and Results

We follow the evaluation protocol of [^15] and evaluate Real-TabPFN with 10-fold cross-validation per dataset. Furthermore, we reuse the performance values for additional baselines from the results reported by [^15]. The baselines were tuned for ROC-AUC via five-fold cross-validated random search under a four-hour time budget. We run Real-TabPFN and TabPFNv2 ourselves without hyperparameter tuning to focus on their in-context learning performance.

Figure 1 compares TabPFNv2 and Real-TabPFN per dataset. We observe that Real-TabPFN significantly outperforms TabPFNv2. Additionally, Figure 3 shows that Real-TabPFN improves the mean normalized ROC-AUC from 0.954 to 0.976 and naturally outperforms all baselines on average, like TabPFNv2. We provide a table with various additional performance metrics for all methods in Appendix C.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2507.03971/assets/x2.png)

Figure 3: Mean Normalized ROC AUC Comparsion of Real-TabPFN with all the default and the tuned versions of the baselines on the AutoMLBenchmark. Scores were normalized per dataset, with 1.0 representing the best and 0.0 the worst performance with respect to all baselines.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2507.03971/assets/x3.png)

Figure 4: Increase in normalized ROC AUC as the continued-pre-training context grows. The gains are shown relative to the base TabPFNv2 model performance which was synthetically pre-trained with 2,048 context size.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2507.03971/assets/x4.png)

Figure 5: Increase in normalized ROC AUC as the training data source is varied. The gains are shown relative to the base TabPFNv2 model performance which was synthetically pre-trained.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2507.03971/assets/x5.png)

Figure 6: Change in normalized ROC AUC as the training data source is varied. The changes are shown relative to the base TabPFNv2 model performance which was synthetically pre-trained.

### Effect of Context Size.

We investigate the impact of the size of datasets during continued pre-training by testing pre-training with datasets from $2{,}048$ to $20{,}000$ (our GPU memory limit) samples. Figure 4 shows that downstream accuracy increases with a larger context size.

### OpenML vs Kaggle.

To understand the impact of our final curated $71$ training data on model performance, we repeated continued pre-training with three corpora: (1) only Kaggle, (2) only OpenML, and (3) the union of both. As Figure 5 shows, OpenML alone delivers a $+0.019$ gain, while Kaggle alone gives $+0.015$. Combining them yields the strongest boost, $+0.022$, confirming that heterogeneous sources provide complementary supervision signals. This finding indicates that while OpenML datasets provide slightly better performance individually, the combination of both data sources yields the best performing model.

### Effect of Training Data Source.

To evaluate the impact of different training data sources, we also experimented with two alternative corpora: (1) CommonCrawl [^32] and (2) GitTables [^29]. We applied aggressive filtering by evaluating datasets with Logistic Regression [^4] and Random Forest [^2] and subsequently removing noisy datasets, followed by our data contamination pipeline (see Section 3). This resulted in approximately $97,000$ CommonCrawl and $658$ GitTables datasets.

Figure 6 compares performance using these two corpora. The model trained on CommonCrawl (approximately 100 data points and 7 features on average per dataset; see Figure 2) exhibits decreased performance, primarily because the small dataset size did not sufficiently benefit the model during the continued pre-training phase, ultimately leading to a performance drop.

In contrast, GitTables (approximately 1000 data points and 9 features on average per dataset; see Figure 2) leads to performance improvements. The biggest performance improvements are achieved with our manually curated OpenML and Kaggle datasets (10k to 100k data points and on average tens of features). We intentionally chose a smaller, curated set of datasets from OpenML and Kaggle to effectively prevent data contamination, which is why we did not combine them with GitTables or CommonCrawl datasets.

## 6 Conclusion and Future Work

We show that continued pre-training of TabPFNv2 on curated, real-world tabular data yields a stronger default model, Real-TabPFN, which we will open-source. Bridging the synthetic-to-real gap, Real-TabPFN outperforms the default TabPFNv2 on most of the datasets and outperforms every other state-of-the-art baseline on all evaluated datasets. Additional experiments deliver the same message: seeing *more context* —whether temporal (longer windows) or statistical (a richer mix of bigger datasets) during continued pre-training produces larger improvements.

## Acknowledgements

This research was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under grant number 539134284, through EFRE (FEIH\_2698644) and the state of Baden-Württemberg. We also acknowledge funding by the European Union (via ERC Consolidator Grant DeepLearning 2.0, grant no. 101045765). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them. L.P. acknowledges funding by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under SFB 1597 (SmallData), grant number 499552394; and funding by ELSA – European Lighthouse on Secure and Safe AI funded by the European Union under Grant Agreement No. 101070617. F.H. acknowledges the financial support of the Hector Foundation. Finally, we thank the reviewers for their feedback, which has helped us improve the manuscript.

## References

## Appendix A Training Datasets

The following table lists the $71$ datasets curated for continued pre-training, along with their source and access link.

| Name | Source |
| --- | --- |
| [aam\_avaliacao\_dataset](https://www.kaggle.com/datasets/himselfthedecker/aam-avaliacao-dataset) | Kaggle |
| [Air Traffic Data](https://www.kaggle.com/datasets/rohanshetty678/air-traffic-data) | Kaggle |
| [ansible-defects-prediction](https://www.kaggle.com/datasets/stefadp/ansibledefectsprediction) | Kaggle |
| [AV Healthcare Analytics II](https://www.kaggle.com/datasets/nehaprabhavalkar/av-healthcare-analytics-ii) | Kaggle |
| [Candidate Selection](https://www.kaggle.com/datasets/tarunchilkur/client) | Kaggle |
| [Cardio Disease](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) | Kaggle |
| [CC\_Fraud\_Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Kaggle |
| [Classification - Crop Damages in India (2015-2019)](https://www.kaggle.com/datasets/aniketng21600/crop-damage-information-in-india) | Kaggle |
| [CSGO Round Winner Classification](https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification) | Kaggle |
| [Flower Type Prediction Machine Hack](https://www.kaggle.com/datasets/vpkprasanna/flower-type-prediction-machine-hack) | Kaggle |
| [Horse Racing - Tipster Bets](https://www.kaggle.com/datasets/gunner38/horseracing/data) | Kaggle |
| [How severe the accident could be](https://www.kaggle.com/datasets/kanuriviveknag/road-accidents-severity-dataset) | Kaggle |
| [HR Analysis Case Study](https://www.kaggle.com/datasets/shivan118/hranalysis) | Kaggle |
| [HR analysis](https://www.kaggle.com/datasets/anshika2301/hr-analytics-dataset) | Kaggle |
| [hr-comma-sep](https://www.kaggle.com/datasets/pankeshpatel/hrcommasep) | Kaggle |
| [ip-network-traffic-flows-labeled-with-87-apps](https://www.kaggle.com/datasets/jsrojas/ip-network-traffic-flows-labeled-with-87-apps) | Kaggle |
| [Janatahack cross-sell prediction](https://www.kaggle.com/datasets/pawan2905/jantahack-cross-sell-prediction) | Kaggle |
| [JanataHack Machine Learning for Banking](https://www.kaggle.com/code/neerunaveenjakhar/janatahack-machine-learning-for-banking/data) | Kaggle |
| [L&T Vehicle Loan Default Prediction](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction) | Kaggle |
| [League of Legends Diamond Games (First 15 Minutes)](https://www.kaggle.com/datasets/benfattori/league-of-legends-diamond-games-first-15-minutes) | Kaggle |
| [Multiple target variable classification - Hackathon](https://www.kaggle.com/datasets/ppsheth91/two-target-variables-classification-problem) | Kaggle |
| [Online News Popularity](https://www.kaggle.com/datasets/btphan/online-news-popularity-dataset) | Kaggle |
| [Online Shopper’s Intention](https://www.kaggle.com/datasets/henrysue/online-shoppers-intention) | Kaggle |
| [Phishing website Detector](https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector) | Kaggle |
| [Phishing websites Data](https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning) | Kaggle |
| [Pump it Up Data Mining the Water Table](https://www.kaggle.com/datasets/dylanli/pump-it-up-data-mining-the-water-table?select=Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_values.csv) | Kaggle |
| [Richter’s Predictor Modeling Earthquake Damage](https://www.kaggle.com/code/franciscoescobar/richter-s-predictor-modeling-earthquake-damage) | Kaggle |
| [Server Logs - Suspicious](https://www.kaggle.com/datasets/kartikjaspal/server-logs-suspicious) | Kaggle |
| [Sloan Digital Sky Survey DR14](https://www.kaggle.com/datasets/lucidlenn/sloan-digital-sky-survey) | Kaggle |
| [Sloan Digital Sky Survey DR16](https://www.kaggle.com/datasets/muhakabartay/sloan-digital-sky-survey-dr16) | Kaggle |
| [Success of Bank Telemarketing Data](https://www.kaggle.com/datasets/raosuny/success-of-bank-telemarketing-data) | Kaggle |
| [Term Deposit Prediction Data Set](https://www.kaggle.com/datasets/brajeshmohapatra/term-deposit-prediction-data-set) | Kaggle |
| [trajectory-based-ship-classification](https://www.kaggle.com/datasets/danielamigo/trajectorybasedshipclassification/data) | Kaggle |
| [Travel Insurance](https://www.kaggle.com/datasets/mhdzahier/travel-insurance) | Kaggle |
| [Amazon\_employee\_access](https://openml.org/search?type=data&status=active&id=4135) | OpenML |
| [artificial-characters](https://www.openml.org/search?type=data&sort=runs&status=active&id=1459) | OpenML |
| [Bank\_marketing\_data\_set\_UCI](https://openml.org/search?type=data&status=active&id=44234) | OpenML |
| [BNG(breast-w)](https://www.openml.org/search?type=data&status=active&id=251) | OpenML |
| [BNG(tic-tac-toe)](https://www.openml.org/search?type=data&status=active&id=137) | OpenML |
| [Click\_prediction\_small](https://www.openml.org/search?type=data&sort=runs&status=active&id=41434) | OpenML |
| [CreditCardSubset](https://openml.org/search?type=data&status=active&id=4154) | OpenML |
| [connect\_4](https://www.openml.org/d/40668) | OpenML |
| [eeg-eye-state](https://www.openml.org/search?type=data&sort=runs&status=active&id=1471) | OpenML |
| [electricity](https://openml.org/search?type=data&status=active&id=151) | OpenML |
| [elevators](https://www.openml.org/search?type=data&status=active&id=846&sort=runs) | OpenML |
| [Employee-Turnover-at-TECHCO](https://openml.org/search?type=data&status=active&id=43551) | OpenML |
| [eye\_movements](https://openml.org/search?type=data&status=active&id=1044) | OpenML |
| [FOREX\_eurpln-hour-High](https://www.openml.org/search?type=data&status=active&id=41787&sort=runs) | OpenML |
| [gas-drift-different-concentrations](https://www.openml.org/search?type=data&sort=runs&status=active&id=1477) | OpenML |
| [gas-drift](https://www.openml.org/search?type=data&sort=runs&status=active&id=1476) | OpenML |
| [higgs](https://openml.org/search?type=data&status=active&id=23512) | OpenML |
| [house\_16H](https://www.openml.org/search?type=data&status=active&id=821&sort=runs) | OpenML |
| [house\_8L](https://www.openml.org/search?type=data&status=active&id=843&sort=runs) | OpenML |
| [Intersectional-Bias-Assessment-(Training-Data)](https://openml.org/search?type=data&status=active&id=44201) | OpenML |
| [law-school-admission-binary](https://openml.org/search?type=data&status=active&id=43904) | OpenML |
| [magic](https://www.openml.org/search?type=data&sort=runs&status=active&id=40679) | OpenML |
| [MagicTelescope](https://openml.org/search?type=data&status=active&id=1120) | OpenML |
| [Medical-Appointment](https://openml.org/search?type=data&status=active&id=43617) | OpenML |
| [microaggregation2](https://www.openml.org/search?type=data&status=active&id=41671&sort=runs) | OpenML |
| [fried](https://www.openml.org/search?type=data&sort=runs&id=901&status=active) | OpenML |
| [mozilla4](https://openml.org/search?type=data&status=active&id=1046) | OpenML |
| [mushroom](https://www.openml.org/search?type=data&status=active&id=43923&sort=runs) | OpenML |
| [NewspaperChurn](https://openml.org/search?type=data&status=active&id=44226) | OpenML |
| [nursery](https://openml.org/search?type=data&status=active&id=1568) | OpenML |
| [okcupid\_stem](https://www.openml.org/search?type=data&status=active&id=45067&sort=runs) | OpenML |
| [pendigits](https://www.openml.org/search?type=data&status=active&id=32&sort=runs) | OpenML |
| [PhishingWebsites](https://openml.org/search?type=data&status=active&id=4534&sort=runs) | OpenML |
| [pol](https://www.openml.org/search?type=data&sort=runs&status=active&id=44122) | OpenML |
| [WBCAtt](https://www.openml.org/search?type=data&status=active&id=46676&sort=runs) | OpenML |
| [Bank Marketing](https://www.openml.org/search?type=data&sort=runs&id=1461&status=active) | OpenML |
| [Internet Firewall Data](https://www.openml.org/search?type=data&sort=runs&id=43039&status=active) | OpenML |

## Appendix B Evaluation Datasets

We use the same evaluation suite as TabPFNv2 to ensure direct comparability of results. All classification tasks from the AutoML Benchmark with fewer 10,000 samples and 500 features. The benchmark comprises diverse real-world tabular datasets, curated for complexity, relevance, and domain diversity.

| Name | OpenML ID | Domain | Features | Samples | Targets | Categorical Feats. |
| --- | --- | --- | --- | --- | --- | --- |
| ada | 41156 | Census | 48 | 4147 | 2 | 0 |
| Australian | 40981 | Finance | 14 | 690 | 2 | 8 |
| blood-transfusion-service-center | 1464 | Healthcare | 4 | 748 | 2 | 0 |
| car | 40975 | Automotive | 6 | 1728 | 4 | 6 |
| churn | 40701 | Telecommunication | 20 | 5000 | 2 | 4 |
| cmc | 23 | Public Health | 9 | 1473 | 3 | 7 |
| credit-g | 31 | Finance | 20 | 1000 | 2 | 13 |
| dna | 40670 | Biology | 180 | 3186 | 3 | 180 |
| eucalyptus | 188 | Agriculture | 19 | 736 | 5 | 5 |
| first-order-theorem-proving | 1475 | Computational Logic | 51 | 6118 | 6 | 0 |
| GesturePhase Segmentation Processed | 4538 | Human-Computer Interaction | 32 | 9873 | 5 | 0 |
| jasmine | 41143 | Natural Language Processing | 144 | 2984 | 2 | 136 |
| kc1 | 1067 | Software Engineering | 21 | 2109 | 2 | 0 |
| kr-vs-kp | 3 | Game Strategy | 36 | 3196 | 2 | 36 |
| madeline | 41144 | Artificial | 259 | 3140 | 2 | 0 |
| mfeat-factors | 12 | Handwriting Recognition | 216 | 2000 | 10 | 0 |
| ozone-level-8hr | 1487 | Environmental | 72 | 2534 | 2 | 0 |
| pc4 | 1049 | Software Engineering | 37 | 1458 | 2 | 0 |
| philippine | 41145 | Bioinformatics | 308 | 5832 | 2 | 0 |
| phoneme | 1489 | Audio | 5 | 5404 | 2 | 0 |
| qsar-biodeg | 1494 | Environmental | 41 | 1055 | 2 | 0 |
| Satellite | 40900 | Environmental Science | 36 | 5100 | 2 | 0 |
| segment | 40984 | Computer Vision | 16 | 2310 | 7 | 0 |
| steel-plates-fault | 40982 | Industrial | 27 | 1941 | 7 | 0 |
| sylvine | 41146 | Environmental Science | 20 | 5124 | 2 | 0 |
| vehicle | 54 | Image Classification | 18 | 846 | 4 | 0 |
| wilt | 40983 | Environmental | 5 | 4839 | 2 | 0 |
| wine-quality-white | 40498 | Food and Beverage | 11 | 4898 | 7 | 0 |
| yeast | 181 | Biology | 8 | 1484 | 10 | 0 |

## Appendix C Performance Comparison on 29 AMLB Classification Datasets

Scores are normalized on all the baselines (0 = worst, 1 = best) per dataset; all methods are tuned for ROC-AUC, so secondary metrics may not reflect their true rank.

Mean Normalized Mean Mean ROC Acc. F1 CE ECE ROC Acc. F1 CE ECE Time (s) ($\uparrow$) ($\uparrow$) ($\uparrow$) ($\downarrow$) ($\downarrow$) ($\uparrow$) ($\uparrow$) ($\uparrow$) ($\downarrow$) ($\downarrow$) Real-TabPFN 0.976 0.932 0.939 0.011 0.107 0.932 0.862 0.771 0.337 0.040 2.921 $\pm$ 0.01 $\pm$ 0.01 $\pm$ 0.01 $\pm$ 0.00 $\pm$ 0.01 $\pm$ 0.01 $\pm$ 0.02 $\pm$ 0.04 $\pm$ 0.03 $\pm$ 0.01 $\pm$ 0.57 TabPFN 0.954 0.906 0.920 0.036 0.111 0.929 0.857 0.767 0.347 0.042 2.793 (default) $\pm$ 0.01 $\pm$ 0.01 $\pm$ 0.01 $\pm$ 0.01 $\pm$ 0.02 $\pm$ 0.01 $\pm$ 0.02 $\pm$ 0.04 $\pm$ 0.03 $\pm$ 0.01 $\pm$ 0.49 Autogluon(V1, 0.928 0.888 0.916 0.040 0.108 0.926 0.856 0.769 0.311 0.041 9660.060 BQ) (tuned) $\pm$ 0.01 $\pm$ 0.02 $\pm$ 0.01 $\pm$ 0.01 $\pm$ 0.01 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.04 $\pm$ 0.03 $\pm$ 0.01 $\pm$ 514.65 XGB 0.842 0.748 0.759 0.268 0.367 0.920 0.844 0.739 0.432 0.066 14444.307 (tuned) $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.04 $\pm$ 0.08 $\pm$ 0.03 $\pm$ 11.99 CatBoost 0.832 0.776 0.790 0.186 0.285 0.920 0.844 0.741 0.408 0.057 14437.103 (tuned) $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.03 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.04 $\pm$ 0.06 $\pm$ 0.02 $\pm$ 4.79 LightGBM 0.781 0.720 0.767 0.252 0.361 0.915 0.841 0.741 0.443 0.063 14410.417 (tuned) $\pm$ 0.02 $\pm$ 0.03 $\pm$ 0.02 $\pm$ 0.03 $\pm$ 0.04 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.04 $\pm$ 0.11 $\pm$ 0.02 $\pm$ 1.37 CatBoost 0.761 0.731 0.783 0.170 0.249 0.913 0.839 0.748 0.404 0.053 5.874 (default) $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.04 $\pm$ 0.04 $\pm$ 0.01 $\pm$ 0.74 Random Forest 0.727 0.650 0.644 0.376 0.462 0.913 0.834 0.716 0.386 0.074 14404.904 (tuned) $\pm$ 0.02 $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.04 $\pm$ 0.03 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.05 $\pm$ 0.07 $\pm$ 0.02 $\pm$ 0.15 LightGBM 0.693 0.684 0.747 0.307 0.407 0.908 0.836 0.745 0.461 0.068 0.583 (default) $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.04 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.04 $\pm$ 0.06 $\pm$ 0.02 $\pm$ 0.06 XGB 0.665 0.643 0.725 0.330 0.533 0.906 0.834 0.743 0.468 0.079 0.814 (default) $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.04 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.04 $\pm$ 0.06 $\pm$ 0.02 $\pm$ 0.09 Random Forest 0.640 0.633 0.672 0.553 0.425 0.907 0.833 0.727 0.432 0.073 0.488 (default) $\pm$ 0.04 $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.04 $\pm$ 0.03 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.04 $\pm$ 0.19 $\pm$ 0.02 $\pm$ 0.03 SVM 0.571 0.531 0.537 0.292 0.169 0.887 0.810 0.680 0.455 0.044 14412.047 (tuned) $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.04 $\pm$ 0.04 $\pm$ 0.01 $\pm$ 3.05 MLP 0.512 0.442 0.480 0.345 0.294 0.883 0.802 0.664 0.493 0.058 2.133 (default) $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.05 $\pm$ 0.05 $\pm$ 0.02 $\pm$ 0.19 MLP (sklearn) 0.458 0.411 0.448 0.432 0.306 0.877 0.800 0.653 0.764 0.059 14408.730 (tuned) $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.04 $\pm$ 0.03 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.06 $\pm$ 0.65 $\pm$ 0.02 $\pm$ 0.34 Log. Regr. 0.401 0.354 0.391 0.386 0.241 0.874 0.789 0.637 inf 0.049 14406.416 (tuned) $\pm$ 0.04 $\pm$ 0.03 $\pm$ 0.04 $\pm$ 0.04 $\pm$ 0.03 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.04 $\pm$ 0.03 $\pm$ 0.02 $\pm$ 0.47 SVM 0.388 0.406 0.430 0.357 0.202 0.872 0.794 0.672 0.482 0.046 2.887 (default) $\pm$ 0.04 $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.04 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.04 $\pm$ 0.03 $\pm$ 0.01 $\pm$ 0.60 Log. Regr. 0.209 0.185 0.186 0.483 0.348 0.857 0.778 0.600 0.529 0.062 0.609 (default) $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.03 $\pm$ 0.04 $\pm$ 0.04 $\pm$ 0.02 $\pm$ 0.02 $\pm$ 0.04 $\pm$ 0.03 $\pm$ 0.02 $\pm$ 0.10

[^1]: Bizer, C., Meusel, R., Lehmberg, O., Ritze, D., and Zope, S. Web data commons - web table corpus 2015 / english-language relational subset, 2015. URL [https://madata.bib.uni-mannheim.de/208/](https://madata.bib.uni-mannheim.de/208/).

[^2]: Breiman, L. Random forests. *Machine Learning*, 45(1):5–32, 2001.

[^3]: Chen, T. and Guestrin, C. Xgboost: A scalable tree boosting system. *CoRR*, abs/1603.02754, 2016. URL [http://arxiv.org/abs/1603.02754](http://arxiv.org/abs/1603.02754).

[^4]: Cox, D. R. The regression analysis of binary sequences. *Journal of the Royal Statistical Society. Series B*, 20(2):215–242, 1958.

[^5]: den Breejen, F., Bae, S., Cha, S., and Yun, S.-Y. Fine-tuned in-context learning transformers are excellent tabular data classifiers, 2025. URL [https://arxiv.org/abs/2405.13396](https://arxiv.org/abs/2405.13396).

[^6]: Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition*, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

[^7]: Dorogush, A. V., Gulin, A., Gusev, G., Kazeev, N., Prokhorenkova, L. O., and Vorobev, A. Fighting biases with dynamic boosting. *CoRR*, abs/1706.09516, 2017. URL [http://arxiv.org/abs/1706.09516](http://arxiv.org/abs/1706.09516).

[^8]: D’souza, A., Swetha, M., and Sarawagi, S. Synthetic tabular data generation for imbalanced classification: The surprising effectiveness of an overlap class. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 39, pp. 16127–16134, 2025.

[^9]: Eggert, G., Huo, K., Biven, M., and Waugh, J. Tablib: A dataset of 627m tables with context, 2023. URL [https://arxiv.org/abs/2310.07875](https://arxiv.org/abs/2310.07875).

[^10]: Gardner, J., Perdomo, J. C., and Schmidt, L. Large scale transfer learning for tabular data via language modeling, 2024. URL [https://arxiv.org/abs/2406.12031](https://arxiv.org/abs/2406.12031).

[^11]: Gijsbers, P., Bueno, M. L. P., Coors, S., LeDell, E., Poirier, S., Thomas, J., Bischl, B., and Vanschoren, J. Amlb: an automl benchmark, 2023. URL [https://arxiv.org/abs/2207.12560](https://arxiv.org/abs/2207.12560).

[^12]: Grinsztajn, L., Oyallon, E., and Varoquaux, G. Why do tree-based models still outperform deep learning on tabular data?, 2022. URL [https://arxiv.org/abs/2207.08815](https://arxiv.org/abs/2207.08815).

[^13]: Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., and Smith, N. A. Don‘t stop pretraining: Adapt language models to domains and tasks. In Jurafsky, D., Chai, J., Schluter, N., and Tetreault, J. (eds.), *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pp. 8342–8360, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.740. URL [https://aclanthology.org/2020.acl-main.740/](https://aclanthology.org/2020.acl-main.740/).

[^14]: Hollmann, N., Müller, S., Eggensperger, K., and Hutter, F. Tabpfn: A transformer that solves small tabular classification problems in a second, 2023. URL [https://arxiv.org/abs/2207.01848](https://arxiv.org/abs/2207.01848).

[^15]: Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., and Hutter, F. Accurate predictions on small data with a tabular foundation model. *Nature*, 637(8045):319–326, 2025. doi: 10.1038/s41586-024-08328-6. URL [https://doi.org/10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6).

[^16]: Hulsebos, M., Demiralp, Ç., and Groth, P. Gittables: A large-scale corpus of relational tables. *CoRR*, abs/2106.07258, 2021. URL [https://arxiv.org/abs/2106.07258](https://arxiv.org/abs/2106.07258).

[^17]: Jiang, M., Liu, K. Z., Zhong, M., Schaeffer, R., Ouyang, S., Han, J., and Koyejo, S. Investigating data contamination for pre-training language models, 2024. URL [https://arxiv.org/abs/2401.06059](https://arxiv.org/abs/2401.06059).

[^18]: Kelly, M., Longjohn, R., and Nottingham, K. The uci machine learning repository, 2007. URL [https://archive.ics.uci.edu](https://archive.ics.uci.edu/).

[^19]: Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., and Hadsell, R. Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13):3521–3526, March 2017. ISSN 1091-6490. doi: 10.1073/pnas.1611835114. URL [http://dx.doi.org/10.1073/pnas.1611835114](http://dx.doi.org/10.1073/pnas.1611835114).

[^20]: Li, X., Grandvalet, Y., and Davoine, F. Explicit inductive bias for transfer learning with convolutional networks, 2018. URL [https://arxiv.org/abs/1802.01483](https://arxiv.org/abs/1802.01483).

[^21]: Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L., and Dollár, P. Microsoft coco: Common objects in context, 2015. URL [https://arxiv.org/abs/1405.0312](https://arxiv.org/abs/1405.0312).

[^22]: Loshchilov, I. and Hutter, F. Decoupled weight decay regularization, 2017a.

[^23]: Loshchilov, I. and Hutter, F. SGDR: Stochastic gradient descent with warm restarts. In *International Conference on Learning Representations*, 2017b. URL [https://openreview.net/forum?id=Skq89Scxx](https://openreview.net/forum?id=Skq89Scxx).

[^24]: Ma, J., Thomas, V., Hosseinzadeh, R., Kamkari, H., Labach, A., Cresswell, J. C., Golestan, K., Yu, G., Volkovs, M., and Caterini, A. L. Tabdpt: Scaling tabular foundation models, 2024. URL [https://arxiv.org/abs/2410.18164](https://arxiv.org/abs/2410.18164).

[^25]: Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12:2825–2830, 2011.

[^26]: Penedo, G., Kydlíček, H., allal, L. B., Lozhkov, A., Mitchell, M., Raffel, C., Werra, L. V., and Wolf, T. The fineweb datasets: Decanting the web for the finest text data at scale, 2024. URL [https://arxiv.org/abs/2406.17557](https://arxiv.org/abs/2406.17557).

[^27]: Qu, J., Holzmüller, D., Varoquaux, G., and Morvan, M. L. Tabicl: A tabular foundation model for in-context learning on large data, 2025. URL [https://arxiv.org/abs/2502.05564](https://arxiv.org/abs/2502.05564).

[^28]: Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. *CoRR*, abs/1910.10683, 2019. URL [http://arxiv.org/abs/1910.10683](http://arxiv.org/abs/1910.10683).

[^29]: Tran, Q. M., Hoang, S. N., Nguyen, L. M., Phan, D., and Lam, H. T. Tabularfm: An open framework for tabular foundational models, 2024. URL [https://arxiv.org/abs/2406.09837](https://arxiv.org/abs/2406.09837).

[^30]: Vanschoren, J., van Rijn, J. N., Bischl, B., and Torgo, L. Openml: networked science in machine learning. *SIGKDD Explorations*, 15(2):49–60, 2013. doi: 10.1145/2641190.2641198. URL [http://doi.acm.org/10.1145/2641190.264119](http://doi.acm.org/10.1145/2641190.264119).

[^31]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I. Attention is all you need. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), *Advances in Neural Information Processing Systems*, volume 30. Curran Associates, Inc., 2017.

[^32]: Yin, P., Neubig, G., tau Yih, W., and Riedel, S. Tabert: Pretraining for joint understanding of textual and tabular data, 2020. URL [https://arxiv.org/abs/2005.08314](https://arxiv.org/abs/2005.08314).