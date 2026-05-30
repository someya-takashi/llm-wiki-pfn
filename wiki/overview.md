---
type: overview
updated: 2026-05-30
---

# PFN（Prior-Data Fitted Networks）総括

> このページは PFN 領域全体の地図です。原典を ingest するたびに随時更新します。

## PFN とは

**Prior-Data Fitted Network（PFN, 事前分布からサンプリングした合成データセットで一度だけ訓練し、推論時には重み更新なしに in-context でベイズ推論を近似するニューラルネットワーク）**。「新しいデータが来たらモデルを当てはめる」という機械学習の基本動作を、「訓練済みネットワークにデータセットを丸ごと 1 回通す」へ置き換えるのが核心。詳細は [[prior-data-fitted-networks]]。

## 主要な系譜・パラダイム

現時点の地図（ingest が進むにつれ拡張）:

- **理論的基盤**: [[bayesian-inference]] の事後予測分布（PPD）を、ニューラルネットで **償却（amortize）** して近似する。この PPD 近似が PFN 全体の "解こうとしている問題"。PFN 原典（[[sources/2021-transformers-can-do-bayesian-inference]]）が、訓練損失最小化＝PPD への KL 最小化であることを証明した。
- **推論の枠組み**: [[in-context-learning]]。重み更新なし・順伝播一回で、文脈（訓練データ）から予測する。大規模言語モデルの ICL と同じ構図を、設計された事前分布のもとで意図的に行う。
- **原典**: Transformers Can Do Bayesian Inference（[[sources/2021-transformers-can-do-bayesian-inference]]）。PFN の枠組みと理論を最初に提示。固定 [[gaussian-process]] をほぼ完璧に模倣し、扱いにくい GP/BNN でも MCMC/SVI より 200〜10000 倍速い近似を達成。回帰用の予測ヘッド「リーマン分布」も導入。
- **代表手法**: TabPFN（[[sources/2022-tabpfn]]）。表形式データ向けに [[structural-causal-model]] と BNN の混合を事前分布として設計し、小規模表データで GBDT を上回り、1 時間級 AutoML に 1 秒未満で並ぶ。原典を実用に引き上げたマイルストーン。
- **事前分布の設計**: PFN の性能は事前分布の設計に決定的に依存する。TabPFN では「単純な因果構造ほど高確率」（オッカムの剃刀）を [[structural-causal-model]] で表現したのが鍵。ベンチマークや事前分布の素材として [[gaussian-process]] が繰り返し登場する。

## オープンな課題

- **スケーラビリティ**: アテンションが入力サンプル数に二次（$O(n^2)$）。大規模データへの拡張（線形アテンション等）。
- **データ型の一般化**: カテゴリ特徴量・欠損値・無情報特徴量への頑健性。表形式以外への一般化。
- **事前分布設計**: データセット依存の事前分布選択、因果（介入・反事実）への拡張。
- **較正と信頼性**: 較正（calibration）、分布外ロバスト性、公平性、説明可能性。

## 関連ページ

- [[prior-data-fitted-networks]] ・ [[in-context-learning]] ・ [[bayesian-inference]] ・ [[structural-causal-model]]
- [[index]]
