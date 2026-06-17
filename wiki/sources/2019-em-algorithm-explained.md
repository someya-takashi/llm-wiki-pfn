---
type: source
source_path: raw/articles/The EM Algorithm Explained.md
source_kind: blog
title: "The EM Algorithm Explained"
authors: [Chloe Bi]
year: 2019
venue: Medium
ingested: 2026-06-17
tags: [expectation-maximization, latent-variable-model, evidence-lower-bound, kl-divergence, variational-inference, optimization]
translation: "[[translations/2019-em-algorithm-explained]]"
---

# The EM Algorithm Explained（EM の「なぜ機能するか」）

> 原典: [[translations/2019-em-algorithm-explained]] ・ `raw/articles/The EM Algorithm Explained.md`（Medium）
> 著者・年・媒体: Chloe Bi / 2019（Medium, 2019-02-08）

## 一言まとめ

**期待値最大化（EM; Expectation-Maximization, [[expectation-maximization]]）を、直感（身長×性別の鶏と卵）と理論（下界＝ELBO・イェンセンの不等式・KL ダイバージェンス・座標上昇）の両面から説いた、EM 一般の解説記事。** GMM 記事3本が EM を「GMM の当てはめ手順」として扱ったのに対し、本稿は **EM そのものを主役にし、「なぜ反復が対数尤度の局所最大に収束するか」を下界最大化として証明的に示す**。これにより EM が **変分推論（VI）の特別な場合**であること、近似推論（MCMC/VI/EM）の一角であることが見え、PFN の「KL 最小化で事後（予測）に寄せる」発想との接続も鮮明になる。

## 背景と問題意識

潜在変数モデルでは、観測 $y$・潜在 $z$・パラメータ $\theta$ があり、尤度 $p(y\mid\theta)$ を最大化したい。だが $\theta$ は $z$ に依存し $z$ は隠れているので、**最尤推定の argmax を直接解けない**。身長の例: 男女の身長分布を推定したいが、各標本の性別（$z$）が分からない。「グループ（パラメータ）を知るには所属が要る／所属を決めるにはパラメータが要る」という**鶏と卵の問題**。EM は「初期のランダムな推測から始めて反復せよ」と答える。**k-means は、クラスタが球状という仮定を置いた EM の特別な変種**である、という指摘も本稿のポイント。

## 提案手法 / 主張（下界＝ELBO の最大化）

**核心**: 直接最大化が難しい $\log p(y\mid\theta)$ を、扱いやすい**下界**に置き換える。任意の潜在分布 $q(z)$ を導入し、ベイズ則 → 対数 → $q(z)$ に関する期待値、と進めると:

$$
\log p(y\mid\theta)=\underbrace{\int q(z)\log\frac{p(y\mid z,\theta)p(z\mid\theta)}{q(z)}dz}_{F(q,\theta)}+\underbrace{\int q(z)\log\frac{q(z)}{p(z\mid y,\theta)}dz}_{\mathrm{KL}(q\,\|\,p(z\mid y,\theta))\ge 0}.
$$

$\log$ が凹（2階微分 $-1/x^2<0$）なので**イェンセンの不等式**から第1項 $F(q,\theta)$ は対数尤度の**下界（＝ELBO, evidence lower bound）**。第2項は $q$ と真の事後の **KL** で非負。EM はこの下界を交互に持ち上げる:

- **E-step**: $\theta^{(t)}$ 固定で $F$ を $q$ について最大化 → KL を最小化 → **$q(z)=p(z\mid y,\theta^{(t)})$（真の事後）**。これで下界が対数尤度に接する。
- **M-step**: $q$ 固定で $F$ を $\theta$ について最大化 → $\theta_t=\arg\max_\theta\int q_t(z)\log\big(p(y\mid z,\theta)p(z\mid\theta)\big)dz$。

これを繰り返すと対数尤度の**局所最大**へ単調に登る（図: Duke の収束可視化、下界が尤度に接しては上がる）。「なぜ式1〜4で微分＝0 を解かないのか？」への答えは「微分が難しい／$F$ は対数の和で扱いやすいから」。

**別の見方**: EM は**座標上昇法**（$q$ と $\theta$ を交互に最大化）。**微分不可能な目的関数でも動く**点が勾配降下と違う。SGD は微分可能性を要し大規模では GPU が要るが、EM は **map-reduce で並列化**しやすい（マッパー＝E-step、リデューサー＝M-step）。応用は **ガウス混合（GMM）・クラスタリング（k-means）・隠れマルコフモデル（HMM）**。

## 実験結果と知見

定量実験はない。眼目は「**下界（ELBO）を E/M で交互に最大化すると対数尤度が単調に上がる**」という収束の理屈と、Wikipedia/Duke の可視化（ガウス混合の収束アニメ・座標上昇の等高線図）。

## 限界・批判的視点

- **局所最適**に収束する（初期値依存）。大域最適の保証はない。
- 記事は導出のスケッチで、イェンセンの不等式の厳密な適用や一般の収束証明（Wu 1983）には踏み込まない。
- GMM/HMM の具体的な E/M-step 更新式は示さない（→ [[gaussian-mixture-model]] / [[sources/2020-gmm-overview]]）。
- EM/VI は**点推定（MLE/MAP）**寄りで、パラメータ自体の完全な事後は出さない（完全ベイズは別途）。

## PFN との接続

- **EM ＝ PFN が償却で置き換える per-dataset 反復推論の一角**。MCMC（[[markov-chain-monte-carlo]]）・VI・EM はいずれもデータごとに反復を回す。[[prior-data-fitted-networks]]（PFN）はこれを事前訓練に償却し順伝播1回で事後予測（[[bayesian-inference]]）を返す。
- **ELBO が EM↔VI の橋**。下界 $F(q,\theta)$ は VI が最大化する ELBO そのもの。**EM は「E-step で $q$ を真の事後に置ける場合の VI」**で、事後が解けないときに $q$ を制限すると VI になる（EM ⊂ VI）。近似推論の地図上で EM の位置が定まる。
- **「KL 最小化で事後（予測）に寄せる」という共通言語**。EM の E-step は $\mathrm{KL}(q\|p(z\mid y))$ を最小化する。PFN 原典（[[sources/2021-transformers-can-do-bayesian-inference]]）は、訓練損失が真の事後予測分布への**前向き KL の最小化**に一致すると示した。同じ KL 最小化を、EM/VI は per-dataset の反復で、PFN は事前訓練で**償却**して実現する、という対比ができる。
- **E-step の出力＝潜在の事後**は、既知モデルなら解析的に出る＝[[questions/pfn-bayesian-inference-evaluation-settings]] の「ベイズ最適事後を PFN 精度の物差しに」（GMM の責任など）の中身。

## 用語と略称

- **EM** = Expectation-Maximization（期待値最大化）→ [[expectation-maximization]]
- **潜在変数（latent variable）$z$** = 観測されない隠れ変数（例: 所属クラスタ）
- **E-step / M-step** = $q$（潜在の事後）を更新 / $\theta$（パラメータ）を更新
- **下界 / ELBO（evidence lower bound, 変分下界）** = $F(q,\theta)$。対数尤度の下界
- **イェンセンの不等式（Jensen's inequality）** = 凹関数で $\mathbb E[f(x)]\le f(\mathbb E[x])$（log は凹）
- **KL ダイバージェンス** = 2分布の非対称な近さ。下界と対数尤度のギャップ $\mathrm{KL}(q\|p(z\mid y))$
- **座標上昇法（coordinate ascent）** = 変数を交互に最適化する手法（EM の別の見方）
- **VI** = Variational Inference（変分推論。ELBO を最大化する近似推論。EM の一般化）
- **HMM** = Hidden Markov Model（隠れマルコフモデル。EM の応用先）
- **MLE / MAP** = 最尤推定 / 最大事後確率推定
- **amortize（償却）** = 事前の一括計算で個別推論を安価にすること（PFN の鍵）

## 関連ページ

- [[expectation-maximization]] — EM の概念ページ（本稿が理論の一次資料）
- [[gaussian-mixture-model]] — EM の代表的な適用先（責任＝E-step の事後）
- [[bayesian-inference]] — 近似推論（MCMC/VI/EM）と償却推論
- [[markov-chain-monte-carlo]] — EM/VI と並ぶ近似推論（サンプリング側）
- [[prior-data-fitted-networks]] — per-dataset の反復推論を順伝播1回に償却
- [[sources/2021-transformers-can-do-bayesian-inference]] — PFN 原典。損失＝事後予測への前向き KL 最小化
- [[sources/2020-gmm-overview]] — GMM 文脈での EM 導出
- [[sources/2022-em-by-examples]] — EM の3実例（K-Means・Two Coins・GMM）の実装チュートリアル
- 参考（外部）: Carl Rasmussen 講義ノート（mlg.eng.cam.ac.uk）、Andrew Ng cs229 notes8（EM の標準的解説）
- [[translations/2019-em-algorithm-explained]] — 本記事の全文翻訳
