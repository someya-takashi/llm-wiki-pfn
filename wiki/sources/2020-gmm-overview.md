---
type: source
source_path: raw/articles/An overview of Gaussian Mixture Models.md
source_kind: blog
title: "An overview of Gaussian Mixture Models"
authors: [Massimiliano Patacchiola]
year: 2020
venue: "mpatacchiola.github.io (blog)"
ingested: 2026-06-17
tags: [gaussian-mixture-model, expectation-maximization, latent-variable-model, maximum-likelihood, responsibility, clustering, density-estimation]
translation: "[[translations/2020-gmm-overview]]"
---

# An overview of Gaussian Mixture Models（GMM の理論的概観）

> 原典: [[translations/2020-gmm-overview]] ・ `raw/articles/An overview of Gaussian Mixture Models.md`（個人ブログ mpatacchiola.github.io）
> 著者・年・媒体: Massimiliano Patacchiola / 2020（個人ブログ。書籍 *Mathematics for Machine Learning* 第11章ベース）

## 一言まとめ

**ガウス混合モデル（GMM; [[gaussian-mixture-model]]）を、教科書 *Mathematics for Machine Learning*（Deisenroth+ 2020）第11章に基づいて理論から導く、最も厳密な GMM 入門。** 1変量・多変量ガウスの最尤推定（ML）→ **潜在変数モデル（$z\to x$）と周辺化** → GMM の尤度 → **責任 ＝ 潜在変数の事後分布**であることの明示 → **非識別可能性（label switching）で対数尤度が非凸**であること → **鶏と卵の問題**としての EM の動機づけ → E/M-step の全式（1変量・多変量）→ from-scratch numpy 実装、までを通す。前2本の GMM 記事（[[sources/2024-gmm-em-clustering]] フルスクラッチ実装・[[sources/2025-gmm-sklearn-guide]] sklearn 応用）が**結果と実装中心**だったのに対し、本稿は**「なぜそうなるか」を導出から押さえる理論リファレンス**で、PFN への接続（責任＝事後、潜在変数の周辺化、多峰事後）も最も鮮明。

## 背景と問題意識：単一ガウスでは足りない

体重データに単一ガウスを ML 当てはめすると、平均・分散の閉形式 $\mu_{\rm ML}=\frac1N\sum_n x_n$、$\sigma^2_{\rm ML}=\frac1N\sum_n(x_n-\mu)^2$ で1ステップで解ける（事後が単峰だから）。だが実データは身長・年齢・性別など**潜在的な変動要因**で複数の部分母集団に分かれ、単一ガウスでは不連続が残る。そこで「データは複数のガウスの**混合**から来た」と考える。

## 提案手法 / 主張：潜在変数 → GMM → 責任 → EM

**① 潜在変数モデル**: 各点 $x_n$ は潜在変数 $z$（どの成分から来たか）に生成される（$z\to x$）。$z$ をワンホットにすればハード割り当て、$z$ 上に分布 $p(z)=\pi=[\pi_1,\dots,\pi_K]$ を置けば**ソフト割り当て**＝混合モデル。

**② GMM の尤度＝潜在変数の周辺化**: $p(x\mid\theta)=\sum_z p(x\mid\theta,z)p(z\mid\theta)=\sum_{k=1}^K \pi_k\,\mathcal N(x\mid\mu_k,\Sigma_k)$。GMM は**ガウス密度の重み付き和**（ガウス確率変数の重み付き和 $\mathcal N(a\mu_x+b\mu_y,\,a^2\Sigma_x+b^2\Sigma_y)$ とは別物、と明確に区別）。新規データは**祖先サンプリング**（まず成分を $\pi$ で選び、次にそのガウスから引く）で生成できる。

**③ 責任 ＝ 事後分布**（本稿の白眉）: 点 $x_n$ が成分 $k$ から来た事後確率

$$
r_{nk}=p(z_k=1\mid x_n)=\frac{\pi_k\,\mathcal N(x_n\mid\mu_k,\sigma_k)}{\sum_{j=1}^K \pi_j\,\mathcal N(x_n\mid\mu_j,\sigma_j)}
$$

を**責任（responsibility）＝ソフトラベル**と呼ぶ。GMM の対数尤度を $\mu_k$ で偏微分すると、単一ガウスの微分にこの $r_{nk}$ が因子として現れる。$r_{nk}$ は確率ベクトル（$\sum_k r_{nk}=1$）で、$N_k=\sum_n r_{nk}$ が成分 $k$ の総責任。

**④ 非識別可能性と非凸性**: 単一ガウスは事後が単峰だが、GMM はラベルの入れ替え（$\mu_1=a,\mu_2=b$ と $\mu_1=b,\mu_2=a$ が同値）で**事後が多峰＝対数尤度が非凸**。一意の ML 解が無いので閉形式で一発では解けない。

**⑤ 鶏と卵 → EM**: 「パラメータを知るにはラベルが要る／ラベルを付けるにはパラメータが要る」という循環。**第1シナリオ（パラメータ既知→責任を計算）と第2シナリオ（ラベル既知→ML でパラメータ）を交互に回す**のが EM。E-step で責任 $r_{nk}$（事後）を計算 → M-step で $\mu_k=\frac1{N_k}\sum_n r_{nk}x_n$、$\Sigma_k=\frac1{N_k}\sum_n r_{nk}(x_n-\mu_k)(x_n-\mu_k)^\top$、$\pi_k=N_k/N$ を更新 → 負の対数尤度が閾値 $\epsilon$ を切るまで反復。**EM は各反復で尤度を単調増加させる（決して悪化しない＝デバッグの目安）**。

## 実験結果と知見

体重データ（507点）に単一ガウスを当てはめると $69.148\pm13.333$。GMM（$K=2$）は平均 ~55 と ~75 の2成分に収束し、単一ガウスより当てはまりが良い。$K=5$ はさらに良いが過適合のリスク（検証集合で対処）。負の対数尤度は最初の数反復で急減し以後単調減少。

## 限界・批判的視点

- **初期化が繊細**で崩壊解（1成分に点が集中）に陥りやすい。
- **特異性（singularity）**: 成分が1点に張り付くと $\sigma\to0$ で対数尤度が無限大に発散しうる（正則化が要る）。
- **成分数 $K$ の選択が難しい**（本稿は BIC/AIC には触れず；それは [[sources/2025-gmm-sklearn-guide]] が補う）。
- EM は MAP/MLE の**点推定**で、パラメータ自体の不確実性（事後分布）は出さない（→ 変分ベイズ GMM・ディリクレ過程混合が拡張）。

## PFN との接続

- **責任 ＝ 事後分布が完全に明示される**点が、PFN 評価との接続を最も鮮明にする。パラメータ既知の GMM なら各点のクラスタ事後 $r_{nk}$ が**解析的に厳密計算**でき、[[questions/pfn-bayesian-inference-evaluation-settings]] の「分類のベイズ最適事後＝PFN 精度の物差し」をそのまま作れる。
- **潜在変数の周辺化** $p(x\mid\theta)=\sum_z p(x\mid\theta,z)p(z\mid\theta)$ は、[[bayesian-inference]] が「事後予測は潜在/パラメータを積分消去して得る」と説くのと同型。PFN はこうした周辺化（積分）を順伝播1回に償却する。
- **多峰な事後・非凸尤度・鶏と卵**は、EM が**データセットごとに反復を要する**理由そのもの。[[prior-data-fitted-networks]]（PFN）はこの per-dataset の反復推論（EM/MCMC/VI, [[markov-chain-monte-carlo]]）を事前訓練に償却し、推論時は順伝播1回で済ませる。
- **ガウスの良い性質（周辺・条件付きもガウス）**が GP（[[gaussian-process]]）やカルマンフィルタの土台でもある、と本稿が明記する点は、PFN が物差し/事前分布に使う [[gaussian-process]] と同じ「ガウスを部品にしたモデリング」系譜を再確認させる（GMM≠GP は別物）。
- 三部作の中で本稿は**理論リファレンス**: 実装は [[sources/2024-gmm-em-clustering]]、sklearn＋モデル選択は [[sources/2025-gmm-sklearn-guide]]、導出と概念は本稿。

## 用語と略称

- **GMM** = Gaussian Mixture Model（ガウス密度の重み付き和 $\sum_k\pi_k\mathcal N$）→ [[gaussian-mixture-model]]
- **EM** = Expectation-Maximization（潜在変数モデルの ML/MAP を E/M-step で反復推定。Dempster+ 1977）
- **潜在変数（latent variable）$z$** = 各点がどの成分から来たか（観測されない）
- **周辺化（marginalization）** = 潜在変数を和/積分で消すこと（$\sum_z$）
- **責任（responsibility）$r_{nk}$** = $p(z_k{=}1\mid x_n)$。点 $x_n$ の成分 $k$ への事後確率＝ソフトラベル
- **ML / MAP** = 最尤推定 / 最大事後確率推定
- **非識別可能性（unidentifiability）** = ラベル入れ替えで複数の最適解 → 事後多峰・対数尤度が非凸
- **祖先サンプリング（ancestral sampling）** = 親（カテゴリ $\pi$）→子（ガウス）の順にサンプル
- **特異性（singularity）** = 成分が1点に崩壊し $\sigma\to0$ で尤度発散
- **$N_k=\sum_n r_{nk}$** = 成分 $k$ の総責任（実効的なデータ点数）
- **amortize（償却）** = 事前の一括計算で個別推論を安価にすること（PFN の鍵）

## 関連ページ

- [[gaussian-mixture-model]] — GMM/EM の概念ページ（本稿が理論リファレンス）
- [[sources/2024-gmm-em-clustering]] — GMM/EM のフルスクラッチ実装（3次元クラスタリング）
- [[sources/2025-gmm-sklearn-guide]] — sklearn 実装＋BIC/AIC によるモデル選択
- [[bayesian-inference]] — 潜在変数の周辺化・事後予測と、それを償却する PFN
- [[markov-chain-monte-carlo]] — EM と並ぶ反復的近似推論（サンプリング側）
- [[prior-data-fitted-networks]] — per-dataset の反復推論を順伝播1回に償却
- [[gaussian-process]] — 同じ「ガウスを部品にしたモデリング」系譜（GMM とは別物）
- [[questions/pfn-bayesian-inference-evaluation-settings]] — 既知 GMM の責任（＝事後）を PFN 精度の物差しに
- [[translations/2020-gmm-overview]] — 本記事の全文翻訳（導出と numpy 実装つき）
