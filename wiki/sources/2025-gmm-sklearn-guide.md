---
type: source
source_path: raw/articles/Gaussian Mixture Models (GMM) Explained_ A Complete Guide with Python Examples.md
source_kind: blog
title: "Gaussian Mixture Models (GMM) Explained: A Complete Guide with Python Examples"
authors: [Lakhan Bukkawar]
year: 2025
venue: "GoPenAI (Medium)"
ingested: 2026-06-17
tags: [gaussian-mixture-model, expectation-maximization, clustering, scikit-learn, model-selection, bic-aic]
translation: "[[translations/2025-gmm-sklearn-guide]]"
---

# Gaussian Mixture Models (GMM) Explained: A Complete Guide（sklearn と BIC/AIC）

> 原典: [[translations/2025-gmm-sklearn-guide]] ・ `raw/articles/Gaussian Mixture Models (GMM) Explained_ ... .md`（GoPenAI / Medium）
> 著者・年・媒体: Lakhan Bukkawar / 2025（GoPenAI, Medium, 2025-03-18）

## 一言まとめ

**ガウス混合モデル（GMM; [[gaussian-mixture-model]]）を、scikit-learn の `GaussianMixture` で動かす実践ガイド。** ソフト vs ハードクラスタリング・GMM の PDF・EM の4ステップ・顧客セグメンテーション例・K-Means との比較表・限界を、`sklearn` の高水準 API（`.fit` / `.predict_proba` / `.bic`）で示す。**概念は直前に取り込んだ [[sources/2024-gmm-em-clustering]] と大きく重複**するが、固有の補完は2点——(1) **scikit-learn での実装**（前者は EM をフルスクラッチ実装だったのに対し、こちらはライブラリ利用）、(2) **BIC/AIC による成分数 k の選択**（前者が「k は決め打ち」で触れなかった穴を埋める）。

## 背景と問題意識

K-Means は各点を1クラスタに固定（ハード割り当て）し球状クラスタを仮定するため、重なり合う・楕円形のデータに弱い。GMM はデータを $K$ 個のガウスの混合 $P(x)=\sum_{i=1}^{K}\pi_i\,\mathcal N(x\mid\mu_i,\Sigma_i)$ とみなし、各成分が自前の平均 $\mu_i$・共分散 $\Sigma_i$・重み $\pi_i$ を持つので、**楕円形クラスタ**と各点への**確率的（ソフト）割り当て**を実現する。顧客セグメンテーション（支出額×購入頻度）のように「予算重視と高級品の中間」のような曖昧な所属を、ハード境界でなく確率で表せるのが利点。

## 提案手法 / 主張（sklearn + BIC/AIC）

- **EM の4ステップ**（初期化 → E-step: 各点の所属確率を計算 → M-step: μ・Σ・π を更新 → 収束まで反復）を概説。詳細な E/M-step の式は [[gaussian-mixture-model]] / [[sources/2024-gmm-em-clustering]] を参照。
- **scikit-learn 実装**: `GaussianMixture(n_components=k).fit(X)` で当てはめ、`.predict`（ハードラベル）・`.predict_proba`（ソフト確率）・`.bic(X)`（モデル選択）を使う。フルスクラッチの EM を書かずに済む実用レシピ。
- **顧客セグメンテーション例**: 3クラスタ（予算重視・通常・高級品）の重なるデータを GMM が分離。
- **GMM vs K-Means 比較表**: 形状（球 vs 楕円）・割り当て（ハード vs ソフト）・重なり対応（× vs ○）・速度（速 vs 遅）・使い所。
- **成分数 k の選択（本稿の固有貢献）**: **BIC（ベイズ情報量規準）/ AIC（赤池情報量規準）** を $k=1..9$ で計算し、**スコア最小の k を選ぶ**（記事の例は $k=3$ で最小）。両規準とも「データへの当てはまり（尤度）」と「モデルの複雑さ（パラメータ数）」をトレードオフし、過剰な成分数にペナルティを与える。

## 実験結果と知見

`sklearn` の2クラスタ合成データで、K-Means のハード割り当て（2色）と GMM のソフト確率（連続グラデーション）の違いを可視化。顧客データでは GMM が重なるセグメントを確率付きで分離。BIC スコアは $k=3$ で底を打つ V 字を描き、最適成分数の選択を示す。

## 限界・批判的視点

- **直前の [[sources/2024-gmm-em-clustering]] と内容が大きく重複**する。EM の理論的導出（Jensen 不等式・下界最大化）は無く、`sklearn` をブラックボックスとして使う。
- 記事自身が挙げる GMM の限界: **収束が遅い・初期化に敏感（局所最適）・成分数を事前に決める必要**（ただし BIC/AIC で緩和）。
- BIC/AIC は万能でなく、真のクラスタ数を当てる保証はない（モデル仮定が外れると誤る）。共分散の縮退・正則化には触れない。

## PFN との接続

GMM の PFN への接続は [[gaussian-mixture-model]] にまとめたとおりで、本稿はそれを補強する:
- **既知 GMM のソフト確率（`predict_proba`）＝ベイズ最適クラス事後**。パラメータ既知なら各点のクラスタ事後が解析的に出るので、[[questions/pfn-bayesian-inference-evaluation-settings]] の「分類のベイズ最適事後＝PFN 精度の物差し」を作れる。
- **EM ＝ per-dataset の反復近似推論**で、[[prior-data-fitted-networks]] が順伝播1回に償却する相手（[[markov-chain-monte-carlo]] と同列）。
- **BIC/AIC のモデル選択は PFN が暗黙に回避する手間**: GMM では「k をいくつにするか」を BIC で外から選ぶが、PFN/TabPFN は構造（成分数に相当する仮説）を事前分布上で周辺化し、推論時に明示的なモデル選択をしない。「複雑さのトレードオフを規準で選ぶ」古典と「事前分布で積分して償却する」PFN の対比が見える。

## 用語と略称

- **GMM** = Gaussian Mixture Model（ガウス混合モデル）→ [[gaussian-mixture-model]]
- **EM** = Expectation-Maximization（期待値最大化。E-step/M-step の反復で潜在変数モデルの MLE を求める）
- **ソフト/ハードクラスタリング** = 確率で複数クラスタに / 1つのクラスタに割り当てる
- **predict_proba** = sklearn で各点の各クラスタへの所属確率（ソフト割り当て）を返すメソッド
- **BIC** = Bayesian Information Criterion（ベイズ情報量規準。尤度−複雑さのペナルティ、低いほど良い）
- **AIC** = Akaike Information Criterion（赤池情報量規準。同様のモデル選択規準）
- **混合係数 π / 平均 μ / 共分散 Σ** = 各ガウス成分の重み・中心・形
- **MLE / MAP** = 最尤推定 / 最大事後確率推定
- **amortize（償却）** = 事前の一括計算で個別推論を安価にすること（PFN の鍵）

## 関連ページ

- [[gaussian-mixture-model]] — GMM/EM の概念ページ（本稿は sklearn 実装＋BIC/AIC を補う）
- [[sources/2024-gmm-em-clustering]] — GMM/EM を**フルスクラッチ実装**した姉妹記事（本稿は sklearn 版＋モデル選択）
- [[sources/2020-gmm-overview]] — GMM/EM を理論から導く厳密リファレンス（潜在変数・責任=事後・非識別可能性）
- [[bayesian-inference]] — 近似推論（EM/MCMC/VI）と償却
- [[prior-data-fitted-networks]] — per-dataset の反復推論・モデル選択を償却で回避
- [[questions/pfn-bayesian-inference-evaluation-settings]] — 既知 GMM のベイズ最適事後を PFN 精度の物差しに
- [[translations/2025-gmm-sklearn-guide]] — 本記事の全文翻訳（sklearn コード含む）
