---
type: source
source_path: raw/articles/Understanding the EM Algorithm by Examples (with code and visualization).md
source_kind: blog
title: "Understanding the EM Algorithm by Examples (with code and visualization)"
authors: [Clarice Wang]
year: 2022
venue: Medium
ingested: 2026-06-17
tags: [expectation-maximization, gaussian-mixture-model, k-means, mixture-model, latent-variable-model, clustering]
translation: "[[translations/2022-em-by-examples]]"
---

# Understanding the EM Algorithm by Examples（3例で見る EM: K-Means / Two Coins / GMM）

> 原典: [[translations/2022-em-by-examples]] ・ `raw/articles/Understanding the EM Algorithm by Examples ... .md`（Medium）
> 著者・年・媒体: Clarice Wang / 2022（Medium, 2022-07-31）

## 一言まとめ

**期待値最大化（EM; [[expectation-maximization]]）を、3つの具体例——K-Means（クラスタリング）・Two Coins（二項混合）・ガウス混合（GMM）——で横並びに見せ、「これらはすべて同じ EM プロセス（E-step↔M-step）の実例だ」と統一して示す実装つきチュートリアル。** 概念面は既存の [[expectation-maximization]] / GMM 3本（[[sources/2024-gmm-em-clustering]] ほか）と大きく重複するが、**固有の補完は「Two Coins（コインの二項混合）」例**——EM をガウス以外（ベルヌーイ/二項）に適用し、表・裏を各コインに**分数的にソフト割り当て**する様子を具体的な数値で見せる、EM の最も有名な教育例。

## 背景と問題意識

まず MLE（最尤推定）: $\theta_{\rm MLE}=\arg\max_\theta p(D\mid\theta)$。ガウスなら平均・分散の閉形式で1ステップ。だが「男女の身長を別々のガウスでモデル化したいが性別（潜在変数）が分からない」ような場合、MLE を直接使えない——**潜在変数の存在が EM を必要とする理由**。EM は4ステップ（初期推測 → E-step: 所属確率で割り当て → M-step: MLE でパラメータ再推定 → 収束テスト）で、初期値依存ゆえ複数の初期推測を試すのが定石。

## 提案手法 / 主張（3つの例で EM を統一）

**① K-Means（クラスタリング）**: `make_blobs` で3クラスタ600点。E-step＝各点を最近傍の重心へ**ハード割り当て**、M-step＝重心を割り当て点の平均に更新。重心が動かなくなるまで反復。**K-Means＝EM の特別な変種**（球状クラスタ・ハード割り当て）。

**② Two Coins（二項混合・本稿の固有価値）**: コイン A・B（表確率は未知, 真値 0.6/0.2）を毎回ランダムに選び10投×多数試行。どちらを使ったかは**潜在変数**。E-step: ある試行 $D$（例: 表8・裏2）について各コインからの尤度 $p(D\mid\theta_A)=\theta_A^{8}(1-\theta_A)^{2}=(0.8)^8(0.2)^2=0.00671$、$p(D\mid\theta_B)=(0.7)^8(0.3)^2=0.00519$ を計算し、**事後（重み）** $p(\text{A}\mid D)=\frac{0.00671}{0.00671+0.00519}=0.564$。K-Means と違い**試行を分数的に割り当てる**——8枚の表のうち $0.564\times8=4.512$ 枚をコイン A に、残りを B に帰属。M-step: 各コインに帰属した表/裏の割合で $\theta$ を更新。15反復で真値 0.6/0.2 に収束。**EM をガウス以外（ベルヌーイ/二項）に適用した例**で、責任＝事後＝ソフト割り当てが最も分かりやすい。

**③ ガウス混合（GMM）**: 3つのガウス（µ,σ=[-10,3]/[30,7]/[3,5]、各1000点）。E-step＝各点の各ガウスへのメンバーシップ確率 $r$（行ごとに正規化）、M-step＝$\mu'_d=\frac{\sum_i r_{d,i}x_i}{\sum r_d}$、$\sigma'_d=\sqrt{\sum_i r_{d,i}(x_i-\mu'_d)^2}$ を MLE で更新。20反復で3分布に収束。

## 実験結果と知見

3例とも EM が真のパラメータ/構造に収束: K-Means の重心、Two Coins の 0.603/0.196（真値 0.6/0.2）、GMM の3ガウス。可視化で反復ごとの収束を見せる。眼目は「クラスタリング・コイン・ガウス混合という**見かけの違う問題が、すべて E-step（潜在の事後で割り当て）↔M-step（MLE で更新）という同じ EM**だ」という統一。

## 限界・批判的視点

- **既存の [[expectation-maximization]] / GMM 3本と大きく重複**。EM の理論（下界=ELBO・なぜ収束するか）は扱わず（→ [[sources/2019-em-algorithm-explained]]）、手順と可視化に終始。
- 初期値依存・局所最適に触れるのみ。成分数 $K$ の選択（BIC/AIC）は扱わない。
- コードは Medium の gist 埋め込み（iframe）で、クリップした markdown には含まれず本翻訳では割愛。

## PFN との接続

GMM/EM の PFN 接続は [[expectation-maximization]] / [[gaussian-mixture-model]] にまとめたとおりで、本稿はそれを補強する:
- **Two Coins の「重み＝事後確率」がソフト割り当ての最小例**。$p(\text{A}\mid D)$ は潜在変数の事後で、パラメータ既知なら解析的に出る＝[[questions/pfn-bayesian-inference-evaluation-settings]] の「ベイズ最適事後を PFN 精度の物差しに」を、ガウスでなく**二項（コイン）**でも作れることを示す。
- **K-Means/Two-Coins/GMM の統一**は、EM が「潜在変数モデル一般の per-dataset 反復推論」であることを具体化し、[[prior-data-fitted-networks]]（PFN）がこの反復を順伝播1回に償却する、という対比をどの問題形でも成り立たせる。
- 本稿の Two Coins は、PFN 原典 [[sources/2021-transformers-can-do-bayesian-inference]] が「コイン投げ（ベルヌーイ）の事後を厳密に解ける物差し」に使う発想（[[questions/pfn-bayesian-inference-evaluation-settings]] の Beta–Bernoulli）と地続き。

## 用語と略称

- **EM** = Expectation-Maximization（期待値最大化）→ [[expectation-maximization]]
- **MLE** = Maximum Likelihood Estimation（最尤推定。$\arg\max_\theta p(D\mid\theta)$）
- **潜在変数（latent variable）** = 観測されない変数（性別・どのコイン・所属クラスタ）
- **混合問題（mixture problem）** = 複数の分布から生成されたデータのパラメータを当てる問題
- **メンバーシップ確率 / 重み / 責任** = 各点が各成分から来た事後確率（E-step の出力、ソフト割り当て）
- **K-Means** = 球状クラスタ・ハード割り当ての EM の特別な変種（重心＝各クラスタの平均）
- **重心（centroid）** = クラスタの中心（割り当て点の平均）
- **二項/ベルヌーイ分布** = コイン投げの分布（Two Coins 例）
- **amortize（償却）** = 事前の一括計算で個別推論を安価にすること（PFN の鍵）

## 関連ページ

- [[expectation-maximization]] — EM の概念ページ（本稿は K-Means/Two Coins/GMM の3実例）
- [[gaussian-mixture-model]] — ガウス混合（本稿の3例目）
- [[sources/2019-em-algorithm-explained]] — EM の理論（下界=ELBO・なぜ収束するか）
- [[sources/2024-gmm-em-clustering]] / [[sources/2020-gmm-overview]] — GMM/EM の実装・理論
- [[prior-data-fitted-networks]] — per-dataset の EM 反復を順伝播1回に償却
- [[questions/pfn-bayesian-inference-evaluation-settings]] — 既知モデル（GMM/コイン）の事後を PFN 精度の物差しに
- [[translations/2022-em-by-examples]] — 本記事の全文翻訳
