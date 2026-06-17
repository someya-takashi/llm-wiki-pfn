---
type: source
source_path: raw/articles/Understanding the basics of Markov Chain Monte Carlo (MCMC) Methods.md
source_kind: blog
title: "Understanding the basics of Markov Chain Monte Carlo (MCMC) Methods"
authors: [Sarowar Ahmed]
year: 2024
venue: Medium
ingested: 2026-06-17
tags: [mcmc, nuts, pymc, bayesian-inference, probabilistic-programming, tutorial]
translation: "[[translations/2024-mcmc-basics-pymc3]]"
---

# Understanding the basics of MCMC Methods（MCMC の超入門＋PyMC3/NUTS 実コード）

> 原典: [[translations/2024-mcmc-basics-pymc3]] ・ `raw/articles/Understanding the basics of Markov Chain Monte Carlo (MCMC) Methods.md`（Medium）
> 著者・年・媒体: Sarowar Ahmed / 2024（Medium, 2024-06-23）

## 一言まとめ

**MCMC（Markov Chain Monte Carlo, マルコフ連鎖モンテカルロ）をひと言で定義し、`PyMC3` で身長データから母平均 μ・母標準偏差 σ の事後分布を NUTS サンプラーで推定する短いコード例を示した、ごく軽量な入門ブログ。** 概念面は既存の MCMC 三部作（[[sources/2021-mcmc-explained]]／[[sources/2019-conceptual-intro-mcmc]]／[[sources/2018-mcmc-simple-introduction]]）と大きく重複するが、**「確率的プログラミング言語（PPL）で実際に MCMC を回す」実践レシピ**という点だけが固有の補完になる。[[markov-chain-monte-carlo]] の「ツールで動かす」面の最小例。

## 背景と問題意識

MCMC は「定常分布が望む分布（典型的には事後分布）に一致するマルコフ連鎖を作り、そこからサンプルする」手法で、**直接サンプリングが難しい分布**を扱える（[[markov-chain-monte-carlo]]）。本稿の動機は理論ではなく「手を動かす」こと: 限られた身長データから、母集団の平均と不確実性をベイズ推論で推定する具体タスクを、ライブラリで解いてみせる。

## 提案手法 / 主張（PyMC3 + NUTS の実レシピ）

身長 `[170,172,…,173]`（10点）から、正規分布の母平均 μ・母標準偏差 σ を推定する:

1. **モデル定義**: 身長 ~ Normal(μ, σ)。
2. **事前分布**: μ ~ Normal(170, 10)、σ ~ HalfNormal(10)（σ は非負なので半正規）。
3. **尤度**: 観測身長に正規尤度。
4. **事後サンプリング**: `pm.sample(2000)` が **NUTS（No-U-Turn Sampler, MCMC の一種）** で事後からサンプルし、`pm.summary` / `traceplot` / ヒストグラムで μ・σ の事後を可視化。

ポイントは **確率的プログラミング（probabilistic programming）** の発想——「事前×尤度のモデルを*宣言*すれば、サンプラー（NUTS）が事後を自動で出す」——であり、ユーザーは詳細釣り合いや提案分布のチューニングを書かずに済む。

<figure>

![](../../raw/assets/2024-mcmc-basics-pymc3/01-mcmc-height-illustration.png)

<figcaption>図（再掲）: 標準モンテカルロ（独立な抽出）と MCMC（相関した抽出）の対比。どちらも目標分布から出発するが、MCMC は連鎖ゆえサンプルが相関する。</figcaption>
</figure>

## 実験結果と知見

定量ベンチはない。要点は「宣言的にモデルを書く → NUTS が事後を出す → μ・σ の点推定と不確実性が得られる」という**最小の動くワークフロー**を見せること。標準モンテカルロが独立サンプルを生むのに対し、MCMC は連鎖なので相関サンプルを生む、という対比図も添える。

## 限界・批判的視点

- **非常に軽量で、既存3本と概念が重複**する。MCMC の中身（詳細釣り合い・Metropolis–Hastings・収束/burn-in 診断・ESS・次元の呪い）は説明せず、**NUTS をブラックボックスとして使う**だけ。理論は [[sources/2019-conceptual-intro-mcmc]]、実務診断（R-hat・blocking・Gibbs/DE）は [[sources/2018-mcmc-simple-introduction]] に譲る。
- 収束診断・事前感度・NUTS の中身（HMC＋軌跡長自動調整）には触れない。`pymc3` は旧版（現在は PyMC）で API も更新されている。

## PFN との接続

- **ここで NUTS が回している「データセットごとの事後サンプリング」こそ、PFN が順伝播1回に償却して消す手作業**。[[prior-data-fitted-networks]]（PFN）は、新しいデータごとに `pm.sample` を待つ代わりに、事前分布から合成したデータで一度だけ事前訓練し、推論時は近似計算ゼロで事後予測分布（[[bayesian-inference]]）を返す。
- **NUTS＝PFN 原典の速度比較相手そのもの**。[[sources/2021-transformers-can-do-bayesian-inference]] は、扱いにくい事後を NUTS で近似する従来法に対し、PFN が同等以上を **NUTS 比 1000〜10000 倍速**で達成すると示した。本稿の「PyMC3 で NUTS を回す」典型例は、その「遅い基準」の実物。
- **PPL ↔ PFN の対比が綺麗**。確率的プログラミングは「**事前×尤度のモデルを書けば、サンプラーが事後を出す**」枠組み。PFN は「**事前＝データ生成器を書けば、順伝播が事後予測を出す**」枠組みで、いわば**償却版の確率的プログラミング**。本稿の宣言的ワークフローは、PFN が何を「即時化」したのかを具体的に映す鏡になる。

## 用語と略称

- **MCMC** = Markov Chain Monte Carlo（マルコフ連鎖モンテカルロ）→ [[markov-chain-monte-carlo]]
- **NUTS** = No-U-Turn Sampler（HMC ベースの MCMC。PyMC のデフォルトサンプラー、PFN 原典のベンチ相手）
- **PyMC3 / PyMC** = Python の確率的プログラミングライブラリ（モデルを宣言するとサンプラーが事後を出す）
- **確率的プログラミング（probabilistic programming, PPL）** = 確率モデルを宣言的に記述し推論を自動化する枠組み
- **事前 / 尤度 / 事後** = Prior / Likelihood / Posterior（μ に Normal、σ に HalfNormal を事前に置く）
- **half-normal（半正規）分布** = 正の値だけを取る分布。σ のような非負パラメータの事前に使う
- **事後予測分布（PPD）** = 次データの予測分布。PFN の近似ターゲット → [[bayesian-inference]]
- **amortize（償却）** = 事前の一括計算で個別推論を安価にすること（PFN の鍵）

## 関連ページ

- [[markov-chain-monte-carlo]] — MCMC 概念ページ（本稿は PyMC/NUTS で実際に回す最小例）
- [[bayesian-inference]] — 事後・事後予測と償却推論
- [[prior-data-fitted-networks]] — per-dataset の NUTS 実行を順伝播1回に償却する枠組み（償却版 PPL）
- [[sources/2021-transformers-can-do-bayesian-inference]] — PFN 原典。NUTS（MCMC）比 1000〜10000 倍速
- [[sources/2018-mcmc-simple-introduction]] — MCMC の応用・実務（Gibbs/DE/診断）
- [[sources/2019-conceptual-intro-mcmc]] — MCMC の厳密・包括版
- [[sources/2021-mcmc-explained]] — MCMC の informal な仕組み
- [[translations/2024-mcmc-basics-pymc3]] — 本記事の全文翻訳（PyMC3 コード含む）
