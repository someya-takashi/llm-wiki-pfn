---
type: concept
aliases: [EM, Expectation-Maximization, 期待値最大化, EM アルゴリズム, E-step, M-step, ELBO, evidence lower bound, 変分下界, coordinate ascent, 座標上昇法]
tags: [latent-variable-model, optimization, approximate-inference, maximum-likelihood]
related:
  - "[[bayesian-inference]]"
  - "[[gaussian-mixture-model]]"
  - "[[markov-chain-monte-carlo]]"
  - "[[prior-data-fitted-networks]]"
sources:
  - "[[sources/2019-em-algorithm-explained]]"
  - "[[sources/2022-em-by-examples]]"
  - "[[sources/2020-gmm-overview]]"
  - "[[sources/2024-gmm-em-clustering]]"
updated: 2026-06-17
---

# Expectation-Maximization（EM, 期待値最大化）

## 一言で

**EM（Expectation-Maximization, 期待値最大化）**は、**潜在変数（latent variable, 観測されない隠れ変数 $z$）を持つモデルの最尤推定（MLE）または最大事後確率推定（MAP）**を、2ステップの反復で求める汎用アルゴリズム（Dempster+ 1977）。観測 $y$・潜在 $z$・パラメータ $\theta$ があるとき、欲しいのは尤度 $p(y\mid\theta)=\sum_z p(y\mid z,\theta)p(z\mid\theta)$ の最大化だが、$z$ が隠れているため直接の argmax が難しい。**「パラメータを知るには所属（$z$）が要る／所属を決めるにはパラメータが要る」という鶏と卵の問題**を、初期値から **E-step（$z$ の事後を計算）↔ M-step（$\theta$ を更新）**の交互反復で解く。[[gaussian-mixture-model]]（GMM）・k-means（球状クラスタを仮定した EM の特別な変種）・隠れマルコフモデル（HMM）など、潜在変数モデル一般の当てはめに使われる。

> 理論（なぜ機能するか）の入門は [[sources/2019-em-algorithm-explained]]（Chloe Bi / Medium）。下界（lower bound）・イェンセンの不等式・KL ダイバージェンス・座標上昇法で「EM がなぜ収束するか」を導く。GMM への具体適用（E/M-step の式）は [[gaussian-mixture-model]] / [[sources/2020-gmm-overview]] / [[sources/2024-gmm-em-clustering]]。

## なぜ機能するか（下界 ＝ ELBO と KL ギャップ）

EM の核心は、**直接最大化が難しい対数尤度 $\log p(y\mid\theta)$ を、扱いやすい下界（lower bound）に置き換えて最大化する**こと（[[sources/2019-em-algorithm-explained]]）。任意の潜在変数の分布 $q(z)$ を導入し、ベイズ則から出発して両辺の対数・$q(z)$ に関する期待値をとると:

$$
\log p(y\mid\theta)=\underbrace{\int q(z)\log\frac{p(y\mid z,\theta)p(z\mid\theta)}{q(z)}\,dz}_{F(q,\theta)\ \text{（下界）}}+\underbrace{\int q(z)\log\frac{q(z)}{p(z\mid y,\theta)}\,dz}_{\mathrm{KL}(q\,\|\,p(z\mid y,\theta))\ \ge 0}.
$$

第2項は $q(z)$ と真の事後 $p(z\mid y,\theta)$ の **KL ダイバージェンス**で常に非負（イェンセンの不等式：$\log$ が凹だから）。よって第1項 $F(q,\theta)$ は対数尤度の**下界**であり、これは **ELBO（evidence lower bound, 変分下界）**そのもの。EM はこの下界を交互に持ち上げる:

- **E-step**: $\theta$ を固定し、**$F$ を $q$ について最大化**。尤度は $q$ に依存しないので、これは **KL を最小化**＝$q(z)=p(z\mid y,\theta^{(t)})$（**真の事後**）と置くことに等しい。これで下界が対数尤度に**接する**。
- **M-step**: $q$ を固定し、$F$ を $\theta$ について最大化: $\theta_t=\arg\max_\theta \int q_t(z)\log\big(p(y\mid z,\theta)p(z\mid\theta)\big)\,dz$。

E↔M を繰り返すと対数尤度の**局所最大**へ単調に登る（各反復で悪化しない＝デバッグの目安）。

## 別の見方・性質

- **座標上昇法（coordinate ascent）**: $F(q,\theta)$ を $q$ と $\theta$ の2座標で交互に最大化する、と見なせる。
- **微分不要**: 目的関数が直接微分可能でなく勾配降下/上昇が使えないときでも動く。
- **vs (S)GD**: SGD は目的関数の微分可能性を要し、大規模データでは GPU が要る。EM は map-reduce で並列化しやすい（マッパー＝E-step、リデューサー＝M-step）。
- **限界**: 初期値依存で**局所最適**に収束する。

## 代表的な適用例（同じ EM の別の顔）

見かけの違う問題が、すべて「E-step（潜在の事後で割り当て）↔ M-step（MLE で更新）」という同じ EM に帰着する（[[sources/2022-em-by-examples]]）:

- **GMM（ガウス混合）**: 潜在＝所属成分。E-step が責任 $r_{nk}$、M-step が $\mu_k,\Sigma_k,\pi_k$ を更新（[[gaussian-mixture-model]]）。
- **k-means**: GMM で**共分散を球状・割り当てをハード**にした特別な変種。E-step＝最近傍重心へ割り当て、M-step＝重心を平均に。
- **Two Coins（二項混合）**: 表確率の異なる2枚のコインを毎回ランダムに選んで投げる。潜在＝どのコインか。E-step＝各試行をベルヌーイ尤度の比で各コインへ**分数的にソフト割り当て**（例: 表8回の試行で $p(\text{A}\mid D)=0.564$ なら $0.564\times8$ 枚をコイン A に帰属）、M-step＝帰属した表/裏の割合で確率を更新。**EM がガウス以外（ベルヌーイ/二項）でも同じ形で動く**ことを示す最も有名な教育例。
- **HMM（隠れマルコフモデル）**: 潜在＝隠れ状態系列（Baum–Welch ＝ HMM の EM）。

## VI（変分推論）との関係

下界 $F(q,\theta)$ は **ELBO** であり、これは**変分推論（Variational Inference, VI）**が最大化するものと同じ。EM は「E-step で $q$ を**真の事後 $p(z\mid y,\theta)$ そのもの**に置ける（事後が解ける）場合の VI」と見なせる。事後が解けないときは、$q$ を扱いやすいパラメータ族に制限して ELBO を最大化する＝VI になる。つまり **EM ⊂ VI**（の特別な場合）。VI は MCMC（[[markov-chain-monte-carlo]]）と並ぶ**近似推論**の二大手法で、EM/ELBO はその橋渡しにある（[[bayesian-inference]]）。

## なぜ PFN の文脈で重要か

- **EM ＝ PFN が償却で置き換える、もう一つの per-dataset 反復推論**。MCMC（サンプリング）・VI（下界最大化）・EM（潜在変数の MLE）はいずれも**データセットごとに反復を回す**。[[prior-data-fitted-networks]]（PFN）はこの反復を**事前訓練に一度だけ償却**し、推論時は順伝播1回で事後予測（[[bayesian-inference]]）を返す。
- **E-step の出力＝潜在変数の事後**。GMM なら責任 $r_{nk}=p(z_k{=}1\mid x_n)$ で、これはパラメータ既知なら解析的に出る＝[[questions/pfn-bayesian-inference-evaluation-settings]] の「ベイズ最適事後を PFN 精度の物差しに」の中身。
- **KL を介した最適化という共通言語**。EM の E-step は $\mathrm{KL}(q\|p(z\mid y))$ を最小化し、VI は ELBO（＝−KL＋定数）を最大化する。PFN 原典（[[sources/2021-transformers-can-do-bayesian-inference]]）も、訓練損失が真の事後予測分布への**前向き KL の最小化**に一致することを示した。「KL を最小化して事後（予測）に寄せる」という発想を、EM/VI は per-dataset の反復で、PFN は事前訓練で償却して実現する、という対比ができる。

## 関連ページ

- [[gaussian-mixture-model]] — EM の代表的な適用先（責任＝E-step の事後、E/M-step の具体式）
- [[bayesian-inference]] — 近似推論（MCMC/VI/EM）と、それを償却する PFN
- [[markov-chain-monte-carlo]] — EM/VI と並ぶ近似推論（サンプリング側）
- [[prior-data-fitted-networks]] — per-dataset の反復推論（EM 含む）を順伝播1回に償却
- [[sources/2019-em-algorithm-explained]] — EM の理論（下界・Jensen・KL・座標上昇）の入門（本概念の一次資料）
- [[sources/2022-em-by-examples]] — EM の3実例（K-Means・Two Coins 二項混合・GMM）で統一的に示す
- [[sources/2020-gmm-overview]] — GMM の文脈での EM 導出
- [[questions/pfn-bayesian-inference-evaluation-settings]] — 既知モデルの事後（E-step 出力）を PFN 精度の物差しに
