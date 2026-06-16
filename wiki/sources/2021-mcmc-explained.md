---
type: source
source_path: raw/articles/Monte Carlo Markov Chain (MCMC) explained.md
source_kind: blog
title: "Monte Carlo Markov Chain (MCMC) explained"
authors: [Shivam Agrahari]
year: 2021
venue: "Towards Data Science"
ingested: 2026-06-16
tags: [mcmc, monte-carlo, markov-chain, metropolis-hastings, approximate-inference, bayesian-inference, sampling]
translation: "[[translations/2021-mcmc-explained]]"
---

# Monte Carlo Markov Chain (MCMC) explained（MCMC の入門解説）

> 原典: [[translations/2021-mcmc-explained]] ・ `raw/articles/Monte Carlo Markov Chain (MCMC) explained.md`（Towards Data Science）
> 著者・年・媒体: Shivam Agrahari / 2021（Towards Data Science, 2021-07-27）

## 一言まとめ

**MCMC（Markov Chain Monte Carlo, マルコフ連鎖モンテカルロ）を「モンテカルロ（乱数で量を近似）」＋「マルコフ連鎖（定常分布へ収束する状態遷移）」に分解して説き、扱いにくい事後分布から正規化定数を計算せずにサンプリングする Metropolis–Hastings アルゴリズムまでを導く入門記事。** PFN（[[prior-data-fitted-networks]]）の文脈では、これは**ベイズ推論で事後が解けないときの定番の近似手法**であり、直前に取り込んだ2本の入門（[[sources/2020-understanding-bayesian-inference]] / [[sources/2024-bayesian-inference-step-by-step]]）が高レベルで触れた「近似推論」の MCMC を、アルゴリズムの中身まで開いて見せる。

## 背景と問題意識：なぜサンプリングが要るのか

ベイズ推論で欲しい事後分布 $P(\theta\mid\mathcal D)=\frac{P(\mathcal D\mid\theta)P(\theta)}{P(\mathcal D)}$ は、分子（尤度×事前）は書けても、分母の**周辺尤度（証拠）$P(\mathcal D)$**——あらゆる $\theta$ にわたる積分——が計算困難（intractable, 解析的に解けない）になりがち。記事はまず**サンプリング**の発想を導入する: 分布から標本を引いて、その平均で「扱いにくい和・積分」を近似する。一般に期待値 $s=\int p(x)f(x)\,dx=\mathbb E_p[f(x)]$ を、標本平均 $\hat s_n=\frac1n\sum_{i=1}^n f(x^{(i)})$ で近似する（中心極限定理により $n$ を増やすほど誤差が縮む）——これが**モンテカルロ推定**。曲線下面積を「長方形に撒いた乱点のうち曲線下に落ちた割合」で測る例が直感を与える。

問題は、**そもそも目的の分布から直接サンプリングできないことがある**こと。そこで**マルコフ連鎖**を使い、扱いにくい分布から効率的にサンプルを引く——これが MCMC。

## 提案手法 / 主張：マルコフ連鎖 → 定常分布 → 詳細釣り合い → Metropolis–Hastings

**マルコフ性（Markov property）**: 次の状態へ移る確率は**現在の状態だけ**に依存し、そこに至る過去の履歴に依存しない（$P(X_{n+1}\mid X_n,\dots,X_1)=P(X_{n+1}\mid X_n)$）。「雨／洗車 → 濡れた地面 → 滑る」の例で、「滑る」確率を測るには「地面が濡れているか」だけ分かれば十分、と説明。この性質を持つ過程が**マルコフ連鎖**。

**定常分布（stationary distribution）**: 状態分布 $s$ に固定の遷移行列 $Q$ を掛け続ける（$s_{i+1}=s_i Q$）と、やがて $sQ=s$ で変化しなくなる不動点に達する。重要なのは、**この定常分布が初期分布に依存しない**こと（3状態の数値例＋Python コードで実演）。MCMC の発想は、**サンプリングしたい目的分布（＝事後分布）を、ある巧妙に作ったマルコフ連鎖の定常分布になるよう設計する**ことにある。

**詳細釣り合い（detailed balance）＝正規化定数を消す鍵**: 定常分布が目的分布 $p$ に一致することを保証する十分条件が

$$
p(X_1)\,T(X_1\to X_2)=p(X_2)\,T(X_2\to X_1).
$$

「$X_1$ にいて $X_2$ へ流れる確率」と「逆向き」が釣り合えば、連鎖は $p$ を定常分布に持つ。これにより、**事後 $\propto$ 尤度×事前**だけ分かれば（正規化定数 $P(\mathcal D)$ を知らなくても）サンプリングできる。

**Metropolis–Hastings**: $p(x)=f(x)/Z$（$Z$ は扱いにくい正規化定数）からサンプリングする具体アルゴリズム。遷移を「**提案確率 $g$**（現在のサンプルから次候補を提案。例: $g(X_2\mid X_1)=\mathcal N(X_1,\sigma)$）」と「**受容確率 $A$**（候補を採るか）」の2段に分け、詳細釣り合いに代入する。$p=f/Z$ を入れると**両辺で $Z$ が打ち消える**のがミソで、整理すると

$$
\frac{A(X_1\to X_2)}{A(X_2\to X_1)}=\underbrace{\frac{f(X_2)}{f(X_1)}}_{R_f}\cdot\underbrace{\frac{g(X_2\mid X_1)}{g(X_1\mid X_2)}}_{R_g},
$$

から受容確率 $A=\min(1,\,R_f R_g)$ をとる（記事の図は比 $R_f R_g$ まで）。手続きは「①ランダムな状態から始める→②$g$ で次候補を引く→③$A$ を計算→④確率 $A$ のコインで受容/棄却→⑤長く繰り返す」。連鎖がまだ定常状態に達していない最初の数サンプルは捨てる（**burn-in 期間**）。

## 実験結果と知見

入門記事のため定量実験はないが、要点は「**目的分布を直接サンプルできなくても、それを定常分布に持つマルコフ連鎖を回せば事後分布の標本が得られる**」「**詳細釣り合いのおかげで正規化定数 $Z=P(\mathcal D)$ を一切計算しなくてよい**」という2点。3状態の数値例で「初期分布によらず同じ定常分布に収束する」ことを Python で実演している。

## 限界・批判的視点

- **多峰（multi-modal）分布に弱い**: 連鎖がモードの一つに詰まり、偏った（バイアスのある）サンプルになって推定精度が落ちる。
- **高次元で遅い**: Metropolis–Hastings はサンプル空間が高次元になると非常に遅い。記事は改良版として**ハミルトニアンモンテカルロ（HMC）**に言及（PFN 原典がベンチに使う **NUTS** は HMC の自動化版）。
- アルゴリズムの収束保証の理論には深入りせず（Extras で『Deep Learning』本の固有値による収束の図を引用するに留まる）。

## PFN との接続（なぜこの wiki に置くか）

- **MCMC ＝ PFN が「償却」で置き換える近似推論の代表**。記事が描く「データ（事後）ごとに毎回チェーンを回してサンプルを溜める」コストの高さこそ、[[prior-data-fitted-networks]] が解消した相手。PFN は事前分布から合成したデータで一度だけ訓練し、推論時は近似計算ゼロで事後予測分布（PPD）を順伝播一回で返す＝**償却ベイズ推論（amortized inference, [[bayesian-inference]]）**。
- **PFN 原典の速度比較相手そのもの**。[[sources/2021-transformers-can-do-bayesian-inference]] は、扱いにくいハイパー事前分布付き GP や BNN の事後を **NUTS（No-U-Turn Sampler, HMC ベースの MCMC）** で近似する従来法と比べ、PFN が同等以上の精度を **NUTS 比 1000〜8000 倍（GP）／10000 倍（BNN）速く**達成すると示した。本記事はその「遅い基準」MCMC の内部を理解する助けになる。
- **「正規化定数 $Z$ を回避＝事後 ∝ 尤度×事前」は前ソースのトリックと同一**。[[sources/2020-understanding-bayesian-inference]] が一般論として述べた「正規化定数を無視して argmax をとる」を、MCMC では詳細釣り合いという具体形で実現している。
- **評価の物差しとしての MCMC**。[[questions/pfn-bayesian-inference-evaluation-settings]] の「(E) 収束 NUTS をゴールドスタンダードに」は、まさにこの MCMC を真値の代理に使う話。閉形式の事後がない prior でも、十分収束させた MCMC を基準に PFN の精度を測れる。

## 用語と略称

- **MCMC** = Markov Chain Monte Carlo（マルコフ連鎖モンテカルロ。マルコフ連鎖でモンテカルロ推定を行う手法群）
- **モンテカルロ推定** = 乱数で引いた標本の平均で、扱いにくい積分・期待値を近似する
- **マルコフ性 / マルコフ連鎖** = 次状態が現在状態だけに依存する性質 / その性質を持つ確率過程
- **遷移確率 $Q$（$T$）** = 状態間を移る確率（の行列）
- **定常分布（stationary distribution）** = $sQ=s$ を満たす不動点の分布。初期分布に依存しない
- **詳細釣り合い（detailed balance）** = $p(A)T(A\to B)=p(B)T(B\to A)$。連鎖の定常分布を目的分布 $p$ に一致させる十分条件
- **Metropolis–Hastings** = 提案 $g$ ＋受容 $A$ で詳細釣り合いを満たす MCMC アルゴリズム
- **提案確率 $g$ / 受容確率 $A$** = 次候補を提案する分布 / その候補を採用する確率
- **正規化定数 $Z$（分配関数）** = $p(x)=f(x)/Z$ の $Z$。＝周辺尤度 $P(\mathcal D)$。計算困難の主因で、MH は打ち消して回避する
- **burn-in 期間** = 連鎖が定常状態に達する前の初期サンプルを捨てる区間
- **HMC / NUTS** = Hamiltonian Monte Carlo / No-U-Turn Sampler（高次元で速い改良 MCMC。PFN 原典の比較相手）
- **VI** = Variational Inference（変分推論。MCMC と並ぶもう一つの近似推論）
- **amortize（償却）** = 事前の一括計算で個別推論を安価にすること（PFN の鍵）

## 関連ページ

- [[bayesian-inference]] — 近似推論（MCMC）が近似しようとする事後・PPD と、それを償却する PFN
- [[prior-data-fitted-networks]] — MCMC の反復サンプリングを順伝播1回に償却する枠組み
- [[sources/2021-transformers-can-do-bayesian-inference]] — PFN 原典。NUTS（MCMC）比 1000〜10000 倍速を実証
- [[sources/2020-understanding-bayesian-inference]] — 近似推論（MCMC/VI）を高レベルで導入。本記事はその MCMC を深掘り
- [[sources/2019-conceptual-intro-mcmc]] — 同じ MCMC の厳密・包括版（重点サンプリング・ESS・自己相関・次元の呪い・typical set・アンサンブル法）
- [[sources/2024-bayesian-inference-step-by-step]] — ベイズ則・共役の入門（事後が閉形式で出る例）。MCMC は閉形式が無い場合の道具
- [[questions/pfn-bayesian-inference-evaluation-settings]] — 「収束 NUTS をゴールドスタンダードに」＝MCMC を PFN 精度の物差しに
- [[translations/2021-mcmc-explained]] — 本記事の全文翻訳
