---
type: concept
aliases: [GMM, Gaussian Mixture Model, ガウス混合モデル, 混合ガウスモデル, mixture model, 混合モデル, soft clustering, ソフトクラスタリング, responsibility, 責任, BIC, AIC, 情報量規準]
tags: [probabilistic-modeling, clustering, latent-variable-model, density-estimation]
related:
  - "[[bayesian-inference]]"
  - "[[expectation-maximization]]"
  - "[[markov-chain-monte-carlo]]"
  - "[[gaussian-process]]"
  - "[[prior-data-fitted-networks]]"
sources:
  - "[[sources/2024-gmm-em-clustering]]"
  - "[[sources/2025-gmm-sklearn-guide]]"
  - "[[sources/2020-gmm-overview]]"
updated: 2026-06-17
---

# Gaussian Mixture Model（GMM, ガウス混合モデル）と EM

## 一言で

**ガウス混合モデル（GMM; Gaussian Mixture Model）**は、「データは $K$ 個のガウス分布の**混合**から生成される」と仮定する確率モデル。密度は $p(x)=\sum_{c=1}^{K}\pi_c\,\mathcal N(x;\mu_c,\Sigma_c)$（$\pi_c$＝各成分の混合係数で $\sum_c\pi_c=1$）。各成分が自前の平均 $\mu_c$・共分散 $\Sigma_c$ を持つので、**楕円形・任意の向きのクラスタ**を表せ、各点に「どのクラスタにどれだけ属するか」の**確率（ソフトクラスタリング）**を与える。多峰（複数ピーク）の分布も表せる**密度推定器**でもある。各点のクラスタ所属は観測されない**潜在変数（latent variable）**なので、当てはめには **EM（期待値最大化）アルゴリズム**を使う。

> 実装つきの入門は [[sources/2024-gmm-em-clustering]]（Tejas Pawar / Medium）。合成3次元データに GMM を EM で適合させ、E-step/M-step の反復でクラスタが収束する様子を可視化し、k-means（球状・ハード割り当て）との違いを対比する。理論から導く厳密なリファレンスは [[sources/2020-gmm-overview]]（Patacchiola, *Mathematics for Machine Learning* 第11章ベース）。

## 潜在変数モデルとしての GMM

GMM は**潜在変数モデル（latent variable model）**として導ける（[[sources/2020-gmm-overview]]）。各点 $x_n$ は「どの成分から来たか」を表す観測されない潜在変数 $z$ に生成される（$z\to x$）。$z$ をワンホットにすればハード割り当て、$z$ 上に分布 $p(z)=\pi$ を置けばソフト割り当て＝混合になる。GMM の尤度は、この潜在変数を**周辺化（marginalize, 和で消す）**して得られる:

$$
p(x\mid\theta)=\sum_z p(x\mid\theta,z)\,p(z\mid\theta)=\sum_{k=1}^K \pi_k\,\mathcal N(x\mid\mu_k,\Sigma_k).
$$

この「潜在を積分消去して周辺尤度を得る」構図は、[[bayesian-inference]] の事後予測（パラメータ/潜在を積分消去）と同型。新規データは**祖先サンプリング**（まず成分を $\pi$ で選び、次にそのガウスから引く）で生成できる。注意: GMM は**ガウス密度の重み付き和**で、ガウス確率変数の重み付き和（$\mathcal N(a\mu_x+b\mu_y,\,a^2\Sigma_x+b^2\Sigma_y)$）とは別物。

## EM アルゴリズム（E-step / M-step）

EM は**潜在変数を持つモデルの最尤推定（MLE）**を反復で求める汎用アルゴリズム。**「なぜ反復が収束するか」という一般論（下界＝ELBO の最大化・KL ギャップ・座標上昇）は [[expectation-maximization]] にまとめた**ので、ここでは GMM への具体適用（E/M-step の式）を示す。GMM では次の2ステップを交互に回す。

- **E-step（期待）**: 現在のパラメータのもとで、各点 $x_i$ が各クラスタ $c$ に属する**責任（responsibility）$r_{ic}$＝潜在変数の事後確率**を計算する。

$$
r_{ic}=\frac{\pi_c\,\mathcal N(x_i;\mu_c,\Sigma_c)}{\sum_{c'}\pi_{c'}\,\mathcal N(x_i;\mu_{c'},\Sigma_{c'})}
$$

分母は全クラスタにわたる和（正規化）で、これが「ソフト割り当て」の実体。

- **M-step（最大化）**: 責任を重みにしてパラメータを更新する。$m_c=\sum_i r_{ic}$ として

$$
\pi_c=\frac{m_c}{m},\quad \mu_c=\frac1{m_c}\sum_i r_{ic}x_i,\quad \Sigma_c=\frac1{m_c}\sum_i r_{ic}(x_i-\mu_c)^{\!\top}(x_i-\mu_c).
$$

E-step（事後を計算）→ M-step（パラメータ更新）を、**対数尤度の変化が閾値未満**になるまで繰り返す。EM は反復のたびに尤度を単調に増やす（決して悪化しない＝デバッグの目安）が、**局所最適に収束する（初期値依存）**。

**なぜ閉形式で一発で解けず反復が要るのか**（[[sources/2020-gmm-overview]]）: 単一ガウスは事後が単峰で ML 解が1ステップで出るが、GMM は**非識別可能（unidentifiable）**——ラベルの入れ替え（$\mu_1=a,\mu_2=b$ と $\mu_1=b,\mu_2=a$ が同値）で**事後が多峰・対数尤度が非凸**になる。さらに「パラメータを知るには所属ラベルが要る／ラベルを付けるにはパラメータが要る」という**鶏と卵の問題**がある。EM はこれを「パラメータ固定でラベル（責任）を推定（E-step）↔ ラベル固定でパラメータを更新（M-step）」の交互反復で解く。**特異性（singularity）**——成分が1点に崩壊し $\sigma\to0$ で尤度が発散する——にも注意（正則化が要る）。

**成分数 $K$ の選択**: $K$ は事前に決める必要があるが、**BIC（ベイズ情報量規準）/ AIC（赤池情報量規準）**——「データへの当てはまり（尤度）」と「モデルの複雑さ（パラメータ数）」をトレードオフし、過剰な成分にペナルティを与える規準——を複数の $K$ で計算し、**スコア最小の $K$ を選ぶ**のが定石（[[sources/2025-gmm-sklearn-guide]]）。実務では scikit-learn の `GaussianMixture`（`.fit` / `.predict_proba`／ソフト確率 / `.bic`）で当てはめからモデル選択まで行える。

## なぜ PFN の文脈で重要か

1. **既知 GMM ＝ PFN 精度評価の「ベイズ最適事後」の物差し**。[[questions/pfn-bayesian-inference-evaluation-settings]] の「(B) 分類のベイズ最適事後を作る」は、**パラメータ既知の GMM** からデータを生成すると各点のクラスタ事後確率（＝E-step の $r_{ic}$）が**解析的に厳密計算できる**ことを使い、PFN のクラス確率の正しさを厳密に突き合わせる設定。GMM は「正解の事後が手に入る」貴重な合成問題を与える。
2. **EM ＝ PFN が償却で置き換える「もう一つの反復的近似推論」**。扱いにくい推論は **MCMC（サンプリング, [[markov-chain-monte-carlo]]）・変分推論（VI）・EM（潜在変数の MLE）** などで近似され、いずれも**データセットごとに反復を回す**。[[prior-data-fitted-networks]]（PFN）はこの per-dataset の反復を**事前訓練に一度だけ償却**し、推論時は順伝播1回で事後予測（[[bayesian-inference]]）を返す。EM の E-step が出す「潜在変数の事後（責任）」は、PFN が文脈から内在化する種類の量。
3. **混合モデルは密度推定・事前分布の素材**。GMM は多峰な分布を表せる密度推定器で、表形式基盤モデル（[[tabular-foundation-model]]）の密度推定タスクや、多峰な合成データの生成と地続き。「ソフトクラスタリング＝潜在クラスへの事後」という見方は、PFN が出す較正された予測分布の親戚。

## GMM と GP の違い（混同注意）

[[gaussian-process]]（GP）と名前が似るが別物。**GP は「関数そのものの上のガウス分布」**（無限次元）で回帰・不確実性に使う。**GMM は「有限個のガウスの重み付き和」**で、クラスタリング・密度推定に使う。共通点は「ガウスを部品にしたベイズ的モデリング」という系譜だけ。

## 関連ページ

- [[sources/2024-gmm-em-clustering]] — GMM/EM をフルスクラッチ実装した入門（本概念の一次資料）
- [[sources/2025-gmm-sklearn-guide]] — scikit-learn 実装＋BIC/AIC による成分数選択
- [[sources/2020-gmm-overview]] — 理論リファレンス（潜在変数の周辺化・責任=事後・非識別可能性・鶏と卵の EM 導出）
- [[expectation-maximization]] — GMM を当てはめる EM の一般論（下界=ELBO 最大化・KL・座標上昇）
- [[bayesian-inference]] — 近似推論（MCMC/VI/EM）と、それを償却する PFN
- [[markov-chain-monte-carlo]] — EM と並ぶ反復的近似推論（サンプリング側）
- [[prior-data-fitted-networks]] — per-dataset の反復推論（EM 含む）を順伝播1回に償却
- [[gaussian-process]] — 同じ「ガウスを部品にしたモデリング」系譜（ただし別物）
- [[questions/pfn-bayesian-inference-evaluation-settings]] — 既知 GMM のベイズ最適事後を PFN 精度の物差しに
