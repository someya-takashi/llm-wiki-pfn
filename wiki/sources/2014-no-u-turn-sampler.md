---
type: source
source_path: raw/articles/The No-U-Turn Sampler_ Adaptively Setting Path Lengths in Hamiltonian Monte Carlo.md
source_kind: paper
title: "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo"
authors: [Matthew D. Hoffman, Andrew Gelman]
year: 2014
venue: "JMLR 15 (2014) / arXiv 1111.4246"
ingested: 2026-06-17
tags: [mcmc, hamiltonian-monte-carlo, nuts, dual-averaging, bayesian-inference, stan, leapfrog]
translation: "[[translations/2014-no-u-turn-sampler]]"
---

# The No-U-Turn Sampler（NUTS の原典）

> 原典: [[translations/2014-no-u-turn-sampler]] ・ `raw/articles/The No-U-Turn Sampler_ ... .md`（arXiv 1111.4246, JMLR 2014, ar5iv 版）
> 著者・年・媒体: Matthew D. Hoffman, Andrew Gelman（Columbia University）/ 2014（JMLR 15）

## 一言まとめ

**ハミルトニアンモンテカルロ（HMC; Hamiltonian Monte Carlo, 勾配を使ってランダムウォークを避ける高次元向け MCMC）の2つの厄介な手調整パラメータを自動化した原典論文。** (1) ステップ数 $L$ を「U ターン基準」で自動決定する **No-U-Turn Sampler（NUTS）**、(2) ステップサイズ $\epsilon$ を**双対平均化（dual averaging）**で自動調整する方式、を提案。結果、**ユーザーは目標受容率を1つ指定するだけで走る「ターンキー（turnkey）」サンプラー**になり、よく調整した HMC と同等以上、ランダムウォーク Metropolis やギブスより桁違いに効率的。確率的プログラミング言語 **Stan の中核アルゴリズム**であり、**PFN 原典 [[sources/2021-transformers-can-do-bayesian-inference]] が「速度のゴールドスタンダード」として比較した相手そのもの**（PFN は NUTS 比 1000〜10000 倍速）。[[markov-chain-monte-carlo]] で名前だけ挙げてきた HMC/NUTS の、これが一次資料。

## 背景と問題意識：HMC は速いが手調整が要る

ランダムウォーク Metropolis やギブス（[[markov-chain-monte-carlo]]）は、相関した高次元事後で**非効率なランダムウォーク**に陥り収束が遅い。**HMC** は、各パラメータ $\theta$ に補助の運動量 $r$ を足し、同時密度 $p(\theta,r)\propto\exp\{\mathcal L(\theta)-\frac12 r\cdot r\}$ を「位置エネルギー＋運動エネルギー」の物理系とみなして、**対数事後の勾配で導かれるリープフロッグ（leapfrog）軌跡**を走らせる。これにより前のサンプルから**遠くまで一気に**移動でき、独立サンプル1つあたり $O(D^{5/4})$（ランダムウォークは $O(D^2)$）。だが代償が2つ:
- **勾配が要る**（自動微分で緩和できる）。
- **$\epsilon$ と $L$ の手調整が要る**。$\epsilon$ が大→不正確で棄却増、小→無駄。$L$ が小→ランダムウォーク、大→**U ターンして足跡をたどり計算が二重に無駄**。特に $L$ は「短すぎ/長すぎ/ちょうど」の尺度が無く、予備実行と専門知識が要る——これが BUGS/JAGS/PyMC のような汎用エンジンへの組み込みを阻む。

## 提案手法 / 主張

**① No-U-Turn 基準で $L$ を消す.** 軌跡が「U ターン」した（出発点 $\theta$ へ戻り始めた）かを、$\frac{d}{dt}\frac{\|\tilde\theta-\theta\|^2}{2}=(\tilde\theta-\theta)\cdot\tilde r$ の符号で判定する（負なら戻り始めた）。だが「単純に負になったら止める」だと**時間反転可能性が壊れ正しい分布に収束しない**。NUTS は**再帰的な倍化（doubling）**でこれを解決: 前後ランダムに $1\to2\to4\to\dots$ ステップ進み、葉が位置・運動量状態の**平衡二分木**を作る（図1）。木の**いずれかの平衡部分木が U ターン**したら停止し、**スライス変数 $u$** と慎重な候補集合 $\mathcal C/\mathcal B$ の設計で**詳細釣り合いを保つ**。効率版（アルゴリズム3）は $O(j)$ メモリ（$O(2^j)$ でなく）で、平均してより大きなジャンプをする遷移核を使う。

**② 双対平均化で $\epsilon$ を消す.** 目標の平均受容率 $\delta$ に合わせるよう、Nesterov の双対平均化 [^20] を MCMC 適応に転用して $\log\epsilon$ を**バーンイン中に自動調整**し、その後凍結。HMC の最適 $\delta\approx0.65$、NUTS は $\delta\approx0.6$（$0.45$〜$0.65$ で鈍感）。初期値は `FindReasonableEpsilon` ヒューリスティックで。**アルゴリズム6（双対平均化つき NUTS）は $\delta$ を1つ指定するだけで走る**＝手調整ゼロ。

## 実験結果と知見

4つの目標分布（250次元多変量正規 MVN・ベイズロジスティック回帰 LR・階層ロジスティック回帰 300+次元 HLR・確率的ボラティリティ 3001次元 SV）で評価。効率は**勾配評価あたりの ESS（有効サンプルサイズ）**で測る。
- 双対平均化は $h$ を目標 $\delta$ にうまく強制（図3）、$\bar\epsilon$ は数百反復で収束（図4。SV のみバーンインが長く遅い）。
- NUTS の軌跡長はほぼ**2の整数乗**に集中（U ターンが倍化完了後に起こる＝半軌跡を捨てるのは時々）（図5）。
- **NUTS ≥ HMC**: LR で同等、MVN・SV で **HMC 最良の約3倍**。しかも HMC の調整実行コストを割り引いてさえ NUTS が勝つ（図6）。HMC の最適 $\lambda$ は4問題で約100倍ばらつく＝手調整の難しさ。
- **NUTS ≫ RWM・ギブス**: 強相関250次元で、RWM は100万サンプルでほぼ未探索、ギブスも一部未探索、NUTS は1000サンプルで広く独立に探索（図7）。RWM/ギブスは実質独立サンプル1つに $O(D^2)$。

## 限界・批判的視点

- **勾配が必要**で、**連続・ほぼ微分可能な変数のみ**。離散変数は積分消去するか別手法（ギブス等）が要る。
- 硬い制約（0事後確率領域）があると HMC は仕事を無駄にする（NUTS はそこまでの点からサンプルでき幾分マシ）。
- 単純な運動エネルギー $\frac12 r\cdot r$ のみ。**質量行列 $M$**（$M^{-1}\approx$ 共分散）や **RMHMC**（位置依存質量、$O(D^3)$）への拡張は本論文外。
- 双対平均化のパラメータ（$\gamma,t_0,\kappa$）は手で少数試して決めた経験則。

## PFN との接続

- **これが PFN 原典の「速度のゴールドスタンダード」NUTS そのもの**。[[sources/2021-transformers-can-do-bayesian-inference]] は、扱いにくいハイパー事前分布付き [[gaussian-process]] や BNN の事後を NUTS で近似する従来法に対し、PFN が同等以上を **NUTS 比 1000〜8000 倍（GP）／10000 倍（BNN）速く**達成すると示した。本論文は、その「遅いが正確で手間いらず」の基準が**なぜ高効率・不偏・turnkey なのか**を与える。
- **NUTS は“最良の per-dataset ベイズ推論”、PFN はそれを償却**。NUTS は最も洗練された MCMC で、**新しいデータごとに勾配を計算し、二分木を伸ばし、バーンイン・収束を待つ**。[[prior-data-fitted-networks]]（PFN）はこの per-dataset の推論を**事前訓練に一度だけ償却**し、推論時は順伝播1回で事後予測（[[bayesian-inference]]）を返す。
- **「人間をループから外す」目標は共通、コスト構造が逆**。NUTS（→ Stan）の狙いは「手調整ゼロの turnkey 推論」で、PFN（→ TabPFN）の狙いは「per-dataset 学習ゼロの即時推論」。だが NUTS は**依然データごとにフルのサンプラーを回す**のに対し、PFN は**順伝播1回**。本論文の「turnkey 化」と PFN の「償却」は、自動ベイズ推論の2つの到達点として対比できる（[[questions/pfn-bayesian-inference-evaluation-settings]] の「収束 NUTS を物差しに」の中身がこの NUTS）。
- **評価の物差しとしての位置づけ**: [[questions/evaluating-pfn-with-known-gmm]] / [[questions/pfn-bayesian-inference-evaluation-settings]] の「閉形式が無い prior では収束 NUTS をゴールドスタンダードに」は、本論文の NUTS が**効率的で不偏な真値の代理**だから成立する。その理論的裏付け（ESS・収束）は [[sources/2019-conceptual-intro-mcmc]]。

## 用語と略称

- **HMC** = Hamiltonian Monte Carlo（ハミルトニアンモンテカルロ。運動量＋勾配で動く MCMC）→ [[markov-chain-monte-carlo]]
- **NUTS** = No-U-Turn Sampler（HMC の $L$ を U ターン基準で自動化した変種）
- **リープフロッグ（leapfrog）** = ハミルトン力学を離散時間でシミュレートする体積保存・可逆な積分器
- **運動量 $r$ / 運動エネルギー $\frac12 r\cdot r$** = HMC が導入する補助変数とそのエネルギー
- **U ターン基準** = $(\theta^+-\theta^-)\cdot r^{\pm}<0$。軌跡が後戻りし始めたら停止
- **倍化（doubling）/ 二分木** = 前後に $2^j$ ステップ伸ばして候補集合を作る再帰手続き
- **スライス変数 $u$** = 詳細釣り合いを保つために導入する補助一様変数
- **双対平均化（dual averaging）** = Nesterov の凸最適化を MCMC 適応に転用し $\epsilon$ を自動調整
- **$\epsilon$ / $L$** = ステップサイズ / ステップ数（HMC の手調整パラメータ。NUTS は $L$ を消す）
- **ESS（有効サンプルサイズ）** = 相関サンプルが実質何個の独立サンプルに相当するか
- **ターンキー（turnkey）** = 手調整なしですぐ使えること。Stan の設計目標
- **Stan** = NUTS を中核に据えた自動ベイズ推論システム（本論文が予告）
- **amortize（償却）** = 事前の一括計算で個別推論を安価にすること（PFN の鍵）

## 関連ページ

- [[markov-chain-monte-carlo]] — MCMC 概念ページ（HMC/NUTS の位置づけ。本論文がその一次資料）
- [[sources/2021-transformers-can-do-bayesian-inference]] — PFN 原典。NUTS 比 1000〜10000 倍速の比較相手が本論文の NUTS
- [[prior-data-fitted-networks]] — per-dataset の NUTS 推論を順伝播1回に償却する PFN
- [[bayesian-inference]] — NUTS が近似する事後・事後予測と償却推論
- [[sources/2019-conceptual-intro-mcmc]] — MCMC の厳密な理論（ESS・自己相関・typical set）。NUTS が効く理由の背景
- [[sources/2021-mcmc-explained]] / [[sources/2018-mcmc-simple-introduction]] — Metropolis–Hastings・実務（NUTS の手前の基本手法）
- [[questions/pfn-bayesian-inference-evaluation-settings]] / [[questions/evaluating-pfn-with-known-gmm]] — 「収束 NUTS を物差しに」の NUTS が本論文
- [[gaussian-process]] — PFN 原典が NUTS で近似した扱いにくい事後（ハイパー事前分布付き GP）
- [[translations/2014-no-u-turn-sampler]] — 本論文の全文翻訳（§1〜5＋アルゴリズム、図解12枚）
