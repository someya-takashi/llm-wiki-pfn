---
type: concept
aliases: [ICL, In-Context Learning, 文脈内学習]
tags: [meta-learning, transformer, amortized-inference]
related: [[prior-data-fitted-networks]], [[bayesian-inference]]
sources: [[sources/2021-transformers-can-do-bayesian-inference]], [[sources/2022-tabpfn]]
updated: 2026-05-30
---

# In-Context Learning（ICL, 文脈内学習）

## 一言で

**文脈内学習（ICL; In-Context Learning, モデルの重みを一切更新せず、入力＝文脈として与えたラベル付き例の系列から、その場でタスクを解く能力）**。学習＝勾配でパラメータを更新すること、という常識に対し、ICL では予測に必要な「学習」が**順伝播の内部**で起こる。大規模言語モデル（LLM）でプロンプトに数例を入れると新しいタスクを解けてしまう現象として広く知られるようになった。

## PFN との関係（このリポジトリでの位置づけ）

[[prior-data-fitted-networks]]（PFN）は、この ICL を **偶発的な創発ではなく、設計された事前分布のもとで意図的に行う** 枠組みと見なせる。
- LLM の ICL: 大量の自然言語で次トークン予測を訓練した副産物として、文脈内の例から学ぶ能力が「創発」する。
- PFN の ICL: 「データセットを文脈として与え、保留点を当てる」タスクで明示的に訓練する。入力は (x, f(x)) のラベル付き例の系列（または集合）で、出力はテスト点の予測。

どちらも共通して、**重み更新なし・順伝播一回**で「訓練データを見て予測する」を実現する。この構図と理論は PFN 原典（[[sources/2021-transformers-can-do-bayesian-inference]]）が確立し、TabPFN（[[sources/2022-tabpfn]]）はそれを表形式データに持ち込んだ代表例で、論文要旨でも自手法を「ICL を行う」と明言している。

## なぜ "学習" が順伝播で起きるのか

理論的な見方の一つは **償却ベイズ推論（amortized Bayesian inference）**。訓練を通じてネットワークは「文脈（訓練データ）→ 事後予測分布」という写像そのものを学ぶ。一度学んでしまえば、新しい文脈に対しては写像を 1 回評価するだけでよい（個別の最適化を「償却」する）。この見方では ICL ≒ 近似ベイズ推論であり、[[bayesian-inference]] と [[prior-data-fitted-networks]] を結ぶ橋になっている。

## 関連ページ

- [[prior-data-fitted-networks]] — ICL を事前分布設計のもとで行う枠組み
- [[bayesian-inference]] — ICL を償却ベイズ推論として解釈する視点
- [[sources/2022-tabpfn]] — 表形式データへの ICL 適用例
