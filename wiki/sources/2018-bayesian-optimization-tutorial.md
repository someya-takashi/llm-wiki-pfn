---
type: source
source_path: raw/articles/A Tutorial on Bayesian Optimization.md
source_kind: article
title: "A Tutorial on Bayesian Optimization"
authors: [Peter I. Frazier]
year: 2018
venue: arXiv 1807.02811
tags: [bayesian-optimization, gaussian-process, acquisition-function, expected-improvement, knowledge-gradient, tutorial]
ingested: 2026-05-31
translation: "[[translations/2018-bayesian-optimization-tutorial]]"
---

# ベイズ最適化のチュートリアル（arXiv 2018）

> 原典: [[translations/2018-bayesian-optimization-tutorial]] ・ `raw/articles/A Tutorial on Bayesian Optimization.md`（arXiv:1807.02811）
> 著者・年: Peter I. Frazier（Cornell University）/ 2018

## 一言まとめ

**ベイズ最適化（BayesOpt; Bayesian Optimization, 評価が高コストなブラックボックス関数を、サロゲート＋獲得関数で少ない評価回数で大域最適化する手法）** の体系的入門。「ガウス過程（[[gaussian-process]]）で目的関数の代理と不確実性をモデル化し、**獲得関数**でどこを次に評価するか決める」というループを核に、期待改善・知識勾配・エントロピー探索の 3 大獲得関数と、ノイズ/並列/制約/多忠実度などの「エキゾチック」拡張を解説する。この wiki では **`bayesian-optimization` 概念のリファレンス**であり、ハイパーパラメータ調整という形で機械学習全体（PFN 系の学習・評価を含む）の土台に位置する。

## 背景と問題意識

最適化したい目的関数 $f$ が「評価に数時間かかる／ブラックボックス（凹・線形などの構造なし）／導関数が得られない／ノイズを伴う」場合、勾配降下やニュートン法は使えず、評価回数も数百回が限度になる。深層ニューラルネットのハイパーパラメータ調整、工学設計、材料・創薬の実験計画、強化学習などがこれに当たる。**ベイズ最適化は「高コスト・ブラックボックス・導関数フリーな大域最適化」専用の方法論**で、評価 1 回 1 回を最大限に活かすために、これまでの観測からベイズ的に「次にどこを測ると最も得か」を計算する。

## 提案手法 / 主張（チュートリアルの骨子）

BayesOpt は 2 部品のループ（アルゴリズム 1）:

1. **統計モデル（ほぼ必ず GP 回帰）**: 観測点から、未評価点 $x$ での $f(x)$ の事後分布（正規分布、平均 $\mu_n(x)$・分散 $\sigma_n^2(x)$）を閉形式で与える。詳細は [[gaussian-process]] / [[sources/2020-gp-regression-tutorial]]。カーネル（べき指数 / Matérn）が関数の滑らかさを、ハイパーパラメータ（MLE / MAP / 完全ベイズで推定）が事前の信念を決める。
2. **獲得関数（acquisition function）**: 現在の事後分布をもとに「点 $x$ を評価する価値」を測るスカラー関数。安価に評価でき、これを最大化する点を次にサンプリングする。

**3 大獲得関数**:
- **期待改善（EI; Expected Improvement）**: 「これまでの最良値 $f^*_n$ をどれだけ更新できそうか」の期待値 $E_n[[f(x)-f^*_n]^+]$。閉形式を持ち、**高い期待品質（$\Delta_n$）と高い不確実性（$\sigma_n$）のトレードオフ**（＝探索 vs 活用）を自然に取る。最も普及。
- **知識勾配（KG; Knowledge Gradient）**: 「評価済み点しか返せない」という EI の仮定を外し、**事後平均の最大値 $\mu^*_n$ の改善期待** $E_n[\mu^*_{n+1}-\mu^*_n]$ を価値とする。サンプル点での改善でなく**全域の事後の改善**を見るため、ノイズあり・多忠実度・導関数観測などで EI を大きく上回る。
- **エントロピー探索（ES）/ 予測エントロピー探索（PES）**: 大域最適 $x^*$ の位置の**微分エントロピー（不確実性）の減少**を最大化。PES は相互情報量で再定式化し第 1 項を閉形式化。

論文独自の貢献として、**ノイズあり評価への EI の自然な一般化**（KGCP）を決定理論的に正当化した点を挙げている。

## 実験結果と知見（チュートリアルの主張）

- EI/KG/ES/PES は **1 段（$N=n+1$）では最適**だが、多段（$N>n+1$）では明らかには最適でない（§4.4）。多段最適は確率的動的計画で原理的に計算可能だが「次元の呪い」で実用困難。一方、計算できる特殊設定では **KG が多段最適の 98% 以内、ES が多段最適**と、1 段獲得関数が驚くほど善戦する。
- 「**エキゾチック**」拡張（§5）では、サンプル点での改善でなく全域事後の改善が価値の源泉になるため、**KG/ES/PES が EI に対し優位**になる: ノイズあり、並列評価（最大 $q=128$）、高コスト制約、多忠実度/多情報源、ランダム環境条件/マルチタスク（交差検証フォールド $w$ など）、導関数観測。

## 限界・批判的視点

- **次元の制約**: BayesOpt は典型的に $d\leq 20$ の連続領域向け。高次元は未解決の研究課題（§7）で、構造を活用する手法が模索されている。
- **GP への依存**: ほぼ全ての BayesOpt が GP をサロゲートに使うが、$O(N^3)$ のコスト（[[gaussian-process]] 参照）と、問題によっては GP 以外のモデルの方が良い可能性が指摘されている。→ **この「GP 以外の高速サロゲート」という空白こそ、後年 [[tabular-foundation-model]]（TabPFN 系）が埋めようとした方向**。
- **多段最適の実用性**: 近似多段法はまだ実用展開できる状態にない。

## PFN 文脈での意義（なぜこの wiki に重要か）

1. **PFN の応用先（下流タスク）**: ベイズ最適化は「高速で較正のよい予測分布を出すサロゲート」を必要とする。TabPFN 系（[[prior-data-fitted-networks]] / [[tabular-foundation-model]]）はまさにそれを 1 回の前向き計算で提供するため、**GP の代わりに TabPFN をサロゲートに使う BayesOpt**（例: 高次元 BO の GIT-BO で TabPFNv2 を利用、[[sources/2026-tabpfn-3]] のエコシステム言及）が成立する。本チュートリアルの「サロゲート＋獲得関数」の枠組みは、その応用を理解する前提。
2. **GP 理解の補完**: §3 の GP 回帰は [[gaussian-process]] / [[sources/2020-gp-regression-tutorial]] と同じ内容を「最適化に使う」観点から再説明する。PFN が**模倣・近似の対象**にする GP の事後予測分布（[[bayesian-inference]] の PPD）が、ここでは獲得関数の入力になる。
3. **ハイパーパラメータ調整の原理**: 深層モデル（PFN を含む）のハイパラ調整に BayesOpt が広く使われる。PFN の学習設定探索の背景知識でもある。

## 用語と略称

- **BayesOpt** = Bayesian Optimization（ベイズ最適化）→ [[bayesian-optimization]]
- **獲得関数（acquisition function）** = 事後分布から「次にどこを評価すると得か」を測るスカラー関数
- **EI** = Expected Improvement（期待改善）。$E_n[[f(x)-f^*_n]^+]$
- **KG** = Knowledge Gradient（知識勾配）。事後平均の最大値の改善期待
- **ES / PES** = Entropy Search / Predictive Entropy Search（エントロピー探索 / 予測エントロピー探索）
- **サロゲート（surrogate）** = 高コストな目的関数を代理する安価なモデル（BayesOpt では GP）
- **探索 vs 活用（exploration vs exploitation）** = 不確実性の高い未知領域を測るか、期待値の高い既知領域を測るかのトレードオフ
- **多忠実度（multi-fidelity）** = 精度とコストの異なる複数の情報源 $f(x,s)$ から最適化する設定
- **GP / GPR** = Gaussian Process / GP Regression → [[gaussian-process]]
- **PPD / 事後予測分布** = 観測を条件にした予測分布 → [[bayesian-inference]]
- **MLE / MAP** = 最尤推定 / 最大事後推定（カーネルのハイパーパラメータ推定法）

## 関連ページ

- [[bayesian-optimization]] — 本チュートリアルが解説する概念そのもの（この source がリファレンス）
- [[gaussian-process]] — BayesOpt のサロゲート（§3 GP 回帰）
- [[sources/2020-gp-regression-tutorial]] — GP 回帰の入門（本チュートリアル §3 と対）
- [[bayesian-inference]] — 獲得関数が用いる事後予測分布の枠組み
- [[tabular-foundation-model]] / [[prior-data-fitted-networks]] — GP の代わりに使えるサロゲート候補（TabPFN を BO に応用）
- [[translations/2018-bayesian-optimization-tutorial]] — 本文 §1〜7 の翻訳
