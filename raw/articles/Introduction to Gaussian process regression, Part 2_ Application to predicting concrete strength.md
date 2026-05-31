---
title: "Introduction to Gaussian process regression, Part 2: Application to predicting concrete strength"
source: "https://medium.com/data-science-at-microsoft/introduction-to-gaussian-process-regression-part-2-application-to-predicting-concrete-strength-3facef04f639"
author:
  - "[[Kaixin Wang]]"
published: 2022-10-11
created: 2026-05-31
description: "The mathematical formulation and a simple example of Gaussian Process Regression (GPR) served as the basis for my first article in this two-"
tags:
  - "clippings"
---
![](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*ICWfXsbvkG01R7Ki)

Photo by Lightscape on Unsplash.

The mathematical formulation and a simple example of Gaussian Process Regression (GPR) served as the basis for my [first article](https://medium.com/data-science-at-microsoft/introduction-to-gaussian-process-regression-part-1-the-basics-3cb79d9f155f?sk=81fa41fcbb67ac893de2e800f9119964) in this two-part series. In this concluding article, I delve deeper by applying GPR to solve a real-world regression problem — predicting the compressive strength of concrete material. As we shall see, GPR provides the point estimate with the associated prediction uncertainty, and it can be easily interpreted using SHapley Additive exPlanations (SHAP) analysis.

## Background

Modeling the behavior of high-performance concrete material is difficult due to its complex property. The goal of the study is to predict concrete strength using its chemical compositions via Machine Learning.[¹](#0dd7) The concrete dataset contains around 1000 concrete samples. The feature set includes properties such as the amount of cement (kg per cubic meter), blast furnace slag (kg per cubic meter), fly ash (kg per cubic meter), water (kg per cubic meter), coarse aggregate (kg per cubic meter), fine aggregate (kg per cubic meter), as well as the age (days) of the concrete when tested. The target variable is the concrete compressive strength (in megapascals, or MPa), a value ranging between 0 and 100. Figure 1 below shows the distribution of the features and the target variable.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*RMvwJ5Pq5kWFOjvYWA0Gmg.png)

*Table 1:* Summary statistics of features and target variable in the concrete dataset.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*_GSIPr0Y--arImKL5rw8uQ.png)

*Figure 1:* Distribution of features and target in the concrete dataset.

The authors of the original study [¹](#0dd7) used Artificial Neural Network (ANN) models to make predictions. In this article, GPR is applied to tackle the Machine Learning challenge, which achieved performance comparable to ANN while at the same time providing the uncertainty level of the predictions.

## Methodology

There are a few steps that need to be implemented before establishing the ML model. As shown in Table 1, different features have differing ranges of values. To lower the impact of scaling them, the features are standardized using the standard scaler in scikit-learn. The formula of standard scaling is as follows:

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*2an1OajnztEtkdLi-BcoNw.png)

where μ is the mean value of *x* and σ is its standard deviation.

Figure 2 below shows the correlation heatmap of the concrete dataset after scaling the features. Cement, superplasticier, and age are the top three features that have a positive correlation with strength, whereas compositions such as fly ash and aggregates are negatively correlated with the target.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*S7AsQ7kJF_ryH4vBiEFJZw.png)

*Figure 2:* Correlation heatmap.

The other important step is to split the dataset into training and testing sets so that the model performance can be evaluated. Figure 3 shows the distribution of concrete strength in the training and testing sets by applying a random train-test split with an 80/20 percent ratio.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*amUE0LlPzU5uCEa7BaXEmQ.png)

*Figure 3:* Distribution of target variable in training and testing set (80/20 percent ratio).

The GPR model training process can now begin. The kernel chosen is the combination of linear and Radial Basis Function (RBF) kernels, a robust combination that can interpolate and extrapolate the data well. The optimal set of hyperparameters (i.e., the variance and lengthscale parameters of the RBF kernel) is determined using the built-in optimizer in GPflow.[²](#9f4f) Figure 4 below shows the predicted versus true concrete strength, from which it can be seen that the model achieved similar performance on the testing set compared to the training set, with a determination of coefficient (i.e., R²) of approximately 0.90 and root mean squared error (RMSE) of approximately 5.4. The results are close to those from the Neural Network model, where the training R² is about 0.945 and the testing R² is about 0.92.

## Get Kaixin Wang’s stories in your inbox

Join Medium for free to get updates from this writer.

Figure 5 shows the 95-percent confidence interval associated with each prediction. We observe that most of the confidence intervals cover the true value of the target variable, except for a few points with large prediction errors. We see that those intervals have a relatively larger bandwidth, which means the model is “aware” of the error and the prediction isn’t as confident as the others.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*3Ak89vDWwCOmhA8yiSigmg.png)

*Figure 4:* Predicted concrete strength using GPR versus true values.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*KjVSpzr89lOCIHv6ZFa7aw.png)

*Figure 5:* Predicted concrete strength using GPR versus true values, with the associated 95-percent prediction confidence interval.

## Model interpretation

ML models are also called “black box” models because it’s often hard to interpret the relation established between the inputs and output. To better understand the role that each feature plays in the GPR model, we utilize SHapley Additive exPlanations (SHAP) analysis to visualize feature importance.

SHAP analysis is an approach based on game theory to explain the output of an ML model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory.[³](#1754) Different types of “explainers” can be applied when interpreting different ML architectures, such as the tree explainer for ensemble tree methods and deep learning explainer for deep neural networks, among others. Kernel explainer, a generic explainer that can be used for any type of model, is used in studying the GPR model.

Here we look at two types of SHAP plots. Figure 6 shows the SHAP summary waterfall plot, which ranks the features based on their SHAP values, where a larger SHAP value indicates that the feature has a stronger impact on the prediction. Each dot is colored corresponding to the feature value, from which the association between the input and prediction can be easily visualized. Figure 7 shows the SHAP dependence plots, or feature interaction plots, which contain both the distribution between the SHAP values and feature value, and the most correlated feature with each predictor.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*66Yg0xJFw4VrYyUjsARn3g.png)

*Figure 6:* SHAP summary waterfall plot based on the optimized GPR model.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*qeaobCZ2vCGZuM_wI2Cbbg.png)

*Figure 7:* SHAP dependence (feature interaction) plot based on the optimized GPR model.

From the SHAP summary plot, we observe that cement is the most important feature in predicting concrete strength, where a higher amount of cement typically results in higher strength, aligning with the observations from the correlation map. The second top feature is the age of the concrete, which has a quadratic relation with the output. From the dependence plots, it’s further proved that there’s a strong linear trend in the distribution of cement, and a quadratic relation in the plot of the age feature. In addition, we see Cement and Age interact the most with Water, aligning with the fact that the water-to-cement ratio (c/m) is one of the most influential factors that make an impact in the strength of concrete.

## Discussion and conclusion

In this article we have looked at a real-world application of GPR in materials science. The model achieved performance comparable to the neural network established in the original research study, and it also revealed the level of uncertainty in the predictions. GPR is becoming more and more widely applied in solving ML problems because of its probabilistic non-parametric property, as well its good performance in interpolation and extrapolation, which provides the point estimate and the prediction confidence intervals.

[*Kaixin Wang*](https://www.linkedin.com/in/kaixinwang/) *is on LinkedIn*.

## References

1\. Yeh, I.-C. Modeling of strength of high-performance concrete using artificial neural networks. *Cement and Concrete research* **28**, 1797–1808 (1998).

2\. Matthews, A. G. de G. *et al.* [GPflow: A Gaussian process library using TensorFlow](http://jmlr.org/papers/v18/16-537.html). *Journal of Machine Learning Research* **18**, 1–6 (2017).

3\. Lundberg, S. M. & Lee, S.-I. [A unified approach to interpreting model predictions](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf). in *Advances in neural information processing systems 30* (eds. Guyon, I. et al.) 4765–4774 (Curran Associates, Inc., 2017).