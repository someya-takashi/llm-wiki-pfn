---
type: source
source_path: raw/articles/Gaussian Mixture Models Explained_ Applying GMM and EM for Effective Data Clustering.md
source_kind: blog
title: "Gaussian Mixture Models Explained: Applying GMM and EM for Effective Data Clustering"
authors: [Tejas Pawar]
year: 2024
venue: Medium
ingested: 2026-06-17
tags: [gaussian-mixture-model, expectation-maximization, clustering, latent-variable-model, soft-clustering, density-estimation]
translation: "[[translations/2024-gmm-em-clustering]]"
---

# Gaussian Mixture Models Explained（GMM と EM によるクラスタリング）

> 原典: [[translations/2024-gmm-em-clustering]] ・ `raw/articles/Gaussian Mixture Models Explained_ ... .md`（Medium）
> 著者・年・媒体: Tejas Pawar / 2024（Medium, 2024-05-08）

## 一言まとめ

**ガウス混合モデル（GMM; Gaussian Mixture Model, データが複数のガウス分布の混合から生成されると仮定する確率的クラスタリングモデル）を、期待値最大化（EM; Expectation-Maximization）アルゴリズムで当てはめる実装チュートリアル。** 合成3次元データに GMM を EM で適合させ、E-step／M-step の反復でクラスタが収束していく様子を可視化する。K-means の「球状クラスタ・ハード割り当て」と対比し、GMM の **柔軟な共分散（楕円形クラスタ）＋ソフトクラスタリング（各点が各クラスタに属する確率）** を強調する。PFN wiki にとっては、(1) [[questions/pfn-bayesian-inference-evaluation-settings]] が PFN の精度評価に使う「**既知 GMM のベイズ最適事後**」の中身、(2) EM が **MCMC/変分推論と並ぶ反復的な近似推論**であり PFN の償却と対比できる点、で接続する。

## 背景と問題意識：なぜ GMM か

クラスタリングは教師なし学習の基本で、データ内の本来的なグループを見つける。代表的な **k-means** は各点を1つのクラスタに割り当て（ハードクラスタリング）、クラスタを球状と仮定するため、楕円形・重なり・非線形構造に弱い。GMM はこれを確率モデル化で克服する: データは $K$ 個のガウス成分の**混合** $p(x)=\sum_{c=1}^{K}\pi_c\,\mathcal N(x;\mu_c,\Sigma_c)$（$\pi_c$＝混合係数、$\sum_c\pi_c=1$）から生成されると仮定する。各成分が自前の平均・共分散を持つので**楕円形・任意の向き**を表せ、各点に「どのクラスタにどれだけ属するか」の**確率（ソフト割り当て）**を与える。多峰（複数ピーク）の密度も表現でき、密度推定器としても使える。

ここで各点のクラスタ所属 $z_i$ は観測されない**潜在変数（latent variable）**であり、これが「混合モデルの推定が一筋縄でいかない」理由。潜在変数があると尤度の直接最大化が難しいので、**EM** を使う。

## 提案手法 / 主張：EM アルゴリズム（E-step / M-step）

EM は**潜在変数を持つモデルの最尤推定**を反復で求める汎用アルゴリズム。GMM では2ステップを交互に回す:

- **E-step（期待）**: 現在のパラメータのもとで、各点 $x_i$ が各クラスタ $c$ に属する**責任（responsibility）$r_{ic}$＝潜在変数の事後確率**を計算する。

$$
r_{ic}=\frac{\pi_c\,\mathcal N(x_i;\mu_c,\Sigma_c)}{\sum_{c'}\pi_{c'}\,\mathcal N(x_i;\mu_{c'},\Sigma_{c'})}
$$

（分母は全クラスタにわたる和＝正規化。これがソフト割り当ての実体）。

- **M-step（最大化）**: 責任を重みに使ってパラメータを更新する。$m_c=\sum_i r_{ic}$（クラスタ $c$ の総責任）として、

$$
\pi_c=\frac{m_c}{m},\qquad \mu_c=\frac1{m_c}\sum_i r_{ic}x_i,\qquad \Sigma_c=\frac1{m_c}\sum_i r_{ic}(x_i-\mu_c)^{\!\top}(x_i-\mu_c).
$$

E-step（事後を計算）→ M-step（パラメータ更新）を、**対数尤度の変化が閾値（例 1e-3）未満**になるまで繰り返す。記事の実装では3クラスタの広がった楕円体データに対し **40反復で収束**し、1→10→25反復の可視化でクラスタ中心が真の中心へ寄っていく。

## 実験結果と知見

合成3次元データ（3クラスタ、異なる平均・共分散・サイズ、重なりあり）で、GMM+EM が **重なり合う楕円体クラスタを正しく分離**（K-means が苦手とするケース）。鍵は**適応的共分散**（中心だけでなく形・向きも合わせる）と**確率に基づく割り当て**（重なり領域での繊細なグループ分け）。収束は対数尤度の停滞で判定。

## 限界・批判的視点

- **EM は局所最適に収束する**（初期値依存）。記事は初期化を乱択しており大域最適の保証はない。多峰・初期値の問題は MCMC と同様に存在する。
- **成分数 $K$ を事前に決め打ち**（記事は3）。実際は情報量規準などで選ぶ必要があり、記事は触れない。
- **共分散が縮退するリスク**（1点に張り付くと特異）や正則化には触れない。EM は MAP/MLE の**点推定**であり、パラメータ自体の不確実性（事後分布）は出さない（→ ベイズ版は変分ベイズ GMM）。
- Medium の入門記事で、理論（EM の Jensen 不等式・下界最大化としての導出）は省略。

## PFN との接続

- **「既知 GMM のベイズ最適事後」＝ PFN 精度評価の物差し**。[[questions/pfn-bayesian-inference-evaluation-settings]] の「(B) 分類のベイズ最適事後を作る」は、まさに**パラメータ既知の GMM** からデータを生成し、各点のクラスタ事後確率（本稿の E-step の $r_{ic}$ に相当）を**解析的に厳密計算**して、PFN のクラス確率の正しさを突き合わせる、という設定。本稿はその「ベイズ最適事後」が何かを具体的に与える。
- **EM ＝ PFN が償却で置き換える「もう一つの反復的近似推論」**。[[bayesian-inference]] / [[markov-chain-monte-carlo]] で見たように、扱いにくい推論は **MCMC（サンプリング）・変分推論（VI）・EM（潜在変数の MLE）** などで近似する。いずれも**データセットごとに反復を回す**。[[prior-data-fitted-networks]]（PFN）は、こうした per-dataset の反復推論を**事前訓練に一度だけ償却**し、推論時は順伝播1回で事後予測を返す。EM の E-step が出す「潜在変数の事後（責任）」は、PFN が文脈から内在化する種類の量。
- **混合モデルは PFN の事前分布・密度推定の素材**。GMM のような混合構造は、TabPFN 系が扱う多峰な表形式データの生成源になりうるし、表形式基盤モデル（[[tabular-foundation-model]]）の密度推定タスクは混合分布の当てはめと地続き。「ソフトクラスタリング＝潜在クラスへの事後」という見方は、PFN が出す較正された予測分布の親戚。
- **ガウス分布つながり**: GMM は単一ガウスの混合で、[[gaussian-process]]（関数上のガウス）とは別物だが、どちらも「ガウスを部品にしたベイズ的モデリング」という同じ系譜にある。

## 用語と略称

- **GMM** = Gaussian Mixture Model（ガウス混合モデル。$\sum_c\pi_c\mathcal N(x;\mu_c,\Sigma_c)$）→ [[gaussian-mixture-model]]
- **EM** = Expectation-Maximization（期待値最大化。潜在変数モデルの MLE を E-step/M-step の反復で求める）
- **責任（responsibility）$r_{ic}$** = 点 $x_i$ がクラスタ $c$ に属する事後確率（E-step の出力＝ソフト割り当て）
- **混合係数 $\pi_c$** = 各成分の重み（$\sum_c\pi_c=1$）
- **潜在変数（latent variable）** = 観測されない変数（ここでは各点のクラスタ所属 $z_i$）
- **ソフト/ハードクラスタリング** = 各点を確率で複数クラスタに / 1つのクラスタに割り当てる
- **k-means** = 重心への距離で割り当てる、球状・ハードクラスタリングの代表
- **多峰（multimodal）** = 複数のピークを持つ分布
- **MLE / MAP** = 最尤推定 / 最大事後確率推定（EM が求める点推定）
- **amortize（償却）** = 事前の一括計算で個別推論を安価にすること（PFN の鍵）

## 関連ページ

- [[gaussian-mixture-model]] — GMM/EM の概念ページ（本稿が一次資料）
- [[sources/2025-gmm-sklearn-guide]] — 姉妹記事。こちらは sklearn 実装＋BIC/AIC による成分数選択
- [[sources/2020-gmm-overview]] — 姉妹記事。GMM/EM を理論から導く厳密リファレンス（潜在変数・責任=事後・非識別可能性）
- [[bayesian-inference]] — 近似推論（MCMC/VI/EM）と、それを償却する PFN
- [[markov-chain-monte-carlo]] — EM と並ぶ反復的近似推論（サンプリング側）
- [[prior-data-fitted-networks]] — per-dataset の反復推論（EM 含む）を順伝播1回に償却
- [[questions/pfn-bayesian-inference-evaluation-settings]] — 既知 GMM のベイズ最適事後を PFN 精度の物差しに（本稿の E-step が中身）
- [[gaussian-process]] — 同じ「ガウスを部品にしたモデリング」系譜（別物）
- [[translations/2024-gmm-em-clustering]] — 本記事の全文翻訳（Python 実装含む）
