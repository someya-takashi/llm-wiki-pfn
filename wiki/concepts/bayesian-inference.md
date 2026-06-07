---
type: concept
aliases: [Bayesian Inference, PPD, Posterior Predictive Distribution, 事後予測分布, ベイズ推論, amortized inference]
tags: [probabilistic-modeling, amortized-inference]
related:
  - "[[prior-data-fitted-networks]]"
  - "[[in-context-learning]]"
sources:
  - "[[sources/2021-transformers-can-do-bayesian-inference]]"
  - "[[sources/2022-tabpfn]]"
  - "[[sources/2025-causalpfn]]"
  - "[[sources/2025-do-pfn]]"
updated: 2026-06-03
---

# Bayesian Inference と事後予測分布（PPD）

## 一言で

**ベイズ推論（Bayesian inference）**は、「データを見る前の信念（事前分布）」を「データを見た後の信念（事後分布）」へ更新する枠組み。教師あり学習で実際に欲しいのは、未知のテスト点 $x$ のラベル分布を、訓練データ $D_{train}$ を条件として求めること —— これを **事後予測分布（PPD; Posterior Predictive Distribution）** と呼ぶ。[[prior-data-fitted-networks]]（PFN）が近似しようとしている当のターゲットがこの PPD である。

## PPD の定義と計算困難性

仮説空間 $\Phi$（あり得るデータ生成メカニズム $\phi$ の全体）を考えると、PPD は仮説について積分した形になる:

$$
p(y\mid x, D)\propto\int_{\Phi}p(y\mid x,\phi)\,p(D\mid\phi)\,p(\phi)\,d\phi.
$$

- $p(\phi)$: 仮説の**事前確率**（その生成メカニズムがどれだけもっともらしいか。例: 単純な構造ほど高確率＝オッカムの剃刀）。
- $p(D\mid\phi)$: その仮説のもとでの観測データの**尤度**。
- 直感的には「データをうまく説明し、かつ事前にもっともらしい仮説」ほど重く効く、無限個の仮説にわたる加重平均。

この積分は仮説空間が巨大なためふつう解析的に解けず、MCMC や変分推論などで近似されてきた。いずれもデータセットごとに高コスト。

## 償却（amortization）という発想

[[prior-data-fitted-networks]] の鍵は **償却ベイズ推論（amortized Bayesian inference）**。データセットごとに毎回近似計算をやり直す代わりに、「文脈（訓練データ）→ PPD」という写像をニューラルネットに**一度だけ**学習させておく。すると新しいデータには順伝播 1 回で PPD が得られる（事前の一括コストで個別推論を償却する）。Müller et al. の理論は、合成データ上の交差エントロピー最小化がこの PPD 近似に一致することを示した。

この「償却」が、ベイズ推論を [[in-context-learning]] や PFN の高速な順伝播推論に結びつける核心になっている。

## このリポジトリでの登場例

- PFN 原典（[[sources/2021-transformers-can-do-bayesian-inference]]）は、Prior-Data NLL 最小化が PPD との交差エントロピー（＝加法定数を除き KL ダイバージェンス）最小化に一致することを証明し（洞察1・系1.1）、固定ガウス過程（[[gaussian-process]]）の厳密 PPD をほぼ完璧に近似できることを実証した。扱いにくいハイパー事前分布付き GP や BNN でも MCMC/SVI より桁違いに速い。
- TabPFN（[[sources/2022-tabpfn]]）は、構造的因果モデル（[[structural-causal-model]]）と BNN の混合を事前分布 $p(\phi)$ とし、その PPD を Transformer で償却近似する。「単純な因果構造ほど高確率」という事前分布が、過適合しにくさ・滑らかな決定境界につながっている。
- CausalPFN（[[sources/2025-causalpfn]]）は、この償却ベイズ推論を**因果**へ拡張した例。データ生成過程（DGP）$\psi$ 上の事前分布から、因果推定対象＝条件付き期待潜在結果（CEPO）の**事後予測分布（CEPO-PPD）**を Transformer で償却近似する。鍵は、上の $p(\phi)$ に相当する DGP 事前分布が**無視可能性（ignorability）を満たす＝因果効果が同定可能**になるよう設計されている点。さらに同論文（付録 C）は、PFN の data-prior 損失の因果版を最小化することが、真の CEPO-PPD への**前向き KL ダイバージェンスの最小化**と同じ最適解を持つことを証明しており、原典の「交差エントロピー最小化＝PPD 近似」を因果推定対象へ一般化した形になっている。
- Do-PFN（[[sources/2025-do-pfn]]）は、償却の対象を「データセット→事後」ではなく **「観測データセット→条件付き介入分布（CID）$p(y\mid do(t),x)$」** に置いた例。SCM 上の事前分布 $p(\psi)$ から介入つきデータをシミュレートし、負の対数尤度最小化が真の CID への**前向き KL 最小化**に一致することを証明（命題1）。CausalPFN が潜在結果＋ignorability だったのに対し、Do-PFN は do 計算側から同じ「償却＝前向き KL 最適化」の構図を介入分布に適用する。償却ベイズ推論が予測（PPD）だけでなく**介入分布**まで射程に入れることを示す。

## 関連ページ

- [[prior-data-fitted-networks]] — PPD を償却近似する具体的枠組み
- [[in-context-learning]] — 償却ベイズ推論としての解釈
- [[structural-causal-model]] — TabPFN が用いる事前分布 $p(\phi)$ の中身
- [[sources/2022-tabpfn]]
- [[sources/2025-causalpfn]] — 償却ベイズ推論を因果（CEPO-PPD）へ拡張、data-prior 損失＝前向き KL
- [[sources/2025-do-pfn]] — 償却の対象を条件付き介入分布（CID）に置いた例（do 計算・前向き KL 最適）
- [[questions/pfn-paper-and-gaussian-process]] — PPD を厳密に解ける GP を物差しに PFN を検証した話・出力の仕組み
- [[questions/evaluating-pfn-gp-approximation]] — PFN の GP 近似を定量評価する指標（KL・RMSE・被覆率・CRPS）
