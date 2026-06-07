---
type: concept
aliases: [BayesOpt, BO, Bayesian Optimization, ベイズ最適化]
tags: [optimization, gaussian-process, acquisition-function, black-box]
related:
  - "[[gaussian-process]]"
  - "[[bayesian-inference]]"
  - "[[tabular-foundation-model]]"
  - "[[prior-data-fitted-networks]]"
sources:
  - "[[sources/2018-bayesian-optimization-tutorial]]"
  - "[[sources/2023-pfns4bo]]"
  - "[[sources/2025-git-bo]]"
updated: 2026-06-03
---

# Bayesian Optimization（BayesOpt, ベイズ最適化）

## 一言で

**ベイズ最適化（BayesOpt; Bayesian Optimization, 評価が高コストなブラックボックス関数を少ない評価回数で大域最適化する手法）**。$\max_{x\in A}f(x)$ を、$f$ が「評価に数時間かかる／ブラックボックス（構造を仮定できない）／導関数なし／ノイズあり」という条件下で解く。中核は次の 2 部品の反復ループ:

1. **サロゲート（代理モデル）**: 観測から $f$ の事後分布を作る。ほぼ必ず**ガウス過程回帰**（[[gaussian-process]]）を使い、各点で予測平均 $\mu_n(x)$ と不確実性 $\sigma_n(x)$ を与える。
2. **獲得関数（acquisition function）**: その事後分布から「次に $x$ を評価する価値」をスカラーで測る。安価に評価でき、これを最大化する点を次にサンプリングする。

評価 → 事後更新 → 獲得関数最大化 → 次の評価、を予算 $N$ 回まで回す。典型的に $d\leq 20$ の連続領域向け。体系的な解説は [[sources/2018-bayesian-optimization-tutorial]]（Frazier のチュートリアル）。

## 獲得関数の主なバリエーション

| 獲得関数 | 価値の定義 | 特徴 |
|---|---|---|
| **EI（Expected Improvement, 期待改善）** | $E_n[[f(x)-f^*_n]^+]$ | 閉形式。最良値の更新期待。探索 vs 活用を自然に取る。最普及 |
| **KG（Knowledge Gradient, 知識勾配）** | $E_n[\mu^*_{n+1}-\mu^*_n]$ | 事後平均の最大値の改善。全域の事後変化を見る。ノイズ・多忠実度等で EI を上回る |
| **ES / PES（Entropy Search / Predictive ES）** | 大域最適 $x^*$ の位置のエントロピー減少 | 情報量基準。PES は相互情報量で再定式化 |

EI は「サンプル点での改善」のみを価値とするのに対し、KG/ES/PES は「測定が**全域の事後**をどう変えるか」を見るため、後述のエキゾチック問題で優位になる。**探索 vs 活用（exploration vs exploitation）**——不確実性の高い未知領域を測るか期待値の高い既知領域を測るか——のトレードオフが BayesOpt の本質。

## エキゾチック拡張

「標準問題」（単純な実行可能集合・ノイズなし・導関数なし）の仮定を破る設定を Frazier は「エキゾチック」と呼ぶ: **ノイズあり評価**（KGCP＝EI のノイズあり一般化）、**並列評価**（複数点を同時提案）、**高コスト制約** $g_i(x)\geq 0$、**多忠実度 / 多情報源**（精度・コストの異なる $f(x,s)$）、**ランダム環境条件 / マルチタスク**（$\max_x\int f(x,w)p(w)dw$、交差検証フォールド等）、**導関数観測**。これらでは KG/ES/PES が EI より有効。

## なぜ PFN の文脈で重要か

1. **PFN の応用先（下流タスク）— その正準的実体が PFNs4BO**　BayesOpt は「高速で較正のよい予測分布を出すサロゲート」を必要とする。標準の GP はサロゲートとして $O(N^3)$ のコストと次元の制約を抱える。**[[prior-data-fitted-networks]]（PFN）は 1 回の前向き計算で予測分布を返す**ため、GP の代わりのサロゲートになりうる。これを正面から実装したのが **PFNs4BO**（[[sources/2023-pfns4bo]]、PFN 原典と同じ著者グループ）で、PFN を GP の「ベイズ的ドロップイン置換」として BO に据え、固定 GP の事後をほぼ厳密に再現しつつ（EI/PI/UCB をリーマン分布から厳密計算）、HPO ベンチで GP ベースの SOTA（HEBO）を上回ることを示した。さらに**高次元（100〜500 次元）BO**では **GIT-BO**（[[sources/2025-git-bo]]）が TabPFN v2 をサロゲートに使う。固定基盤モデルは GP のようにカーネルを適応できないが、GIT-BO は **TabPFN の順伝播の勾配から“探索すべき低次元部分空間”を毎反復つくり直す**ことでこれを補い、GP ベース高次元 BO（SAASBO 等）を 2 桁速く凌駕した（PFNs4BO が低次元、GIT-BO が高次元、と棲み分く）。TabPFN-3（[[sources/2026-tabpfn-3]]）のエコシステムでも BayesOpt は下流応用として挙がる。

   - **GP では難しい拡張が事前分布の設計だけで入る**: PFNs4BO は、最適解の位置のヒント（**ユーザー事前分布** $p(D|\rho,I)$）、出力に効かない**無関係次元**の無視、そして**獲得関数（知識勾配 KG）を直接学習する非近視眼的 BO** を、いずれも「事前分布を書いて一度訓練する」だけで実現する。GP/経験ベイズには組み込みにくいこれらが、PFN では自然に扱える点が本質的な利点。

2. **GP を共有の土台にする**　BayesOpt のサロゲートも、PFN が**模倣・近似の対象**にするものも、ともに [[gaussian-process]] の事後予測分布（[[bayesian-inference]] の PPD）。Frazier §3 の GP 回帰は [[sources/2020-gp-regression-tutorial]] と同内容を「最適化に使う」視点で説明する。BayesOpt では PPD が**獲得関数の入力**になる。

3. **ハイパーパラメータ調整**　深層モデル（PFN を含む）のハイパラ調整に BayesOpt が広く使われる。

## 関連ページ

- [[gaussian-process]] — BayesOpt のサロゲート（事後分布を与える）
- [[bayesian-inference]] — 獲得関数が用いる事後予測分布（PPD）の枠組み
- [[tabular-foundation-model]] — GP の代わりに使えるサロゲート候補（TabPFN を BO に応用）
- [[prior-data-fitted-networks]] — 1 回の前向き計算で PPD を返す＝高速サロゲート
- [[sources/2023-pfns4bo]] — PFN を GP の置換として BO に据えた正準的論文（ユーザー事前分布・無関係次元・非近視眼 KG）
- [[sources/2025-git-bo]] — TabPFNv2 を高次元 BO サロゲートに（順伝播の勾配で能動部分空間を毎反復同定、最大500次元で SOTA）
- [[sources/2018-bayesian-optimization-tutorial]] — 本概念のリファレンス（Frazier のチュートリアル）
- [[sources/2020-gp-regression-tutorial]] — サロゲートに使う GP 回帰の入門
- [[questions/pfn-and-bayesian-optimization]] — PFN と BO の関係（BO のサロゲートを GP→PFN に差し替える）初心者向け図解
