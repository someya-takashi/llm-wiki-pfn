---
type: source
source_path: raw/articles/Understanding Bayesian Inference.md
source_kind: blog
title: "Understanding Bayesian Inference"
authors: [Jonty Sinai]
year: 2020
venue: "Personal blog (jontysinai.github.io)"
ingested: 2026-06-16
tags: [bayesian-inference, posterior-predictive-distribution, approximate-inference, mcmc, variational-inference, mle, map]
translation: "[[translations/2020-understanding-bayesian-inference]]"
---

# Understanding Bayesian Inference（ベイズ推論を機械学習パラダイムとして理解する）

> 原典: [[translations/2020-understanding-bayesian-inference]] ・ `raw/articles/Understanding Bayesian Inference.md`（個人ブログ jontysinai.github.io）
> 著者・年・媒体: Jonty Sinai / 2020（個人ブログ, 2020-04-19）

## 一言まとめ

**ベイズ推論を「理論的枠組み」ではなく「機械学習のパラダイム」として説き、点推定しか出せない普通の学習（$\theta^{*}$ を1点に最適化）から、パラメータ $\theta$ を確率変数として事後分布で扱う見方へ橋渡しする入門記事。** 重要なのは、**予測分布（predictive distribution）$P(y^{*}\mid x^{*};\mathcal D)=\int_\theta P(y^{*}\mid x^{*},\theta)P(\theta\mid\mathcal D)\,d\theta$ ＝事後予測分布（PPD）まで到達**し、その計算困難性ゆえに **近似推論（MCMC・変分推論・期待値伝播）** が要る、という PFN の出発点そのものを平易に描く点。直前に取り込んだ [[sources/2024-bayesian-inference-step-by-step]] が「$\theta$ の事後と MAP」で止まるのに対し、本記事はその先（PPD と近似推論）を埋める。

## 背景と問題意識：なぜ点推定では足りないのか

機械学習の標準は、柔軟な関数 $f:\mathcal X\to\mathcal Y$（典型的にはニューラルネット）をパラメータ $\theta$ で表し、コスト関数を最小化して**最適な1点 $\theta^{*}$** を見つけること。だが $\theta^{*}$ が決まると予測は**完全に決定論的**になり、「この予測にどれだけ自信があるか」を言えない。記事は、**(1) データの分散が大きい・外れ値のとき**、**(2) サンプル数が少ないとき**に不確実性が欲しい、と動機づける。$\theta$ そのものの不確実性を反映した予測がしたい——これがベイズ推論の狙いである。

## 提案手法 / 主張：ベイズ則 → 事後 → 予測分布 → 近似推論

**ベイズの定理を「データの不確実性」と「予測の不確実性」の橋渡しに使う。**

$$
P(\theta\mid\mathcal D)=\frac{P(\mathcal D\mid\theta)\,P(\theta)}{P(\mathcal D)}.
$$

- **事前分布 $P(\theta)$**: データを見る前の $\theta$ への信念。「離散なら離散分布、非負なら正の分布」のように仮定を符号化（し検証）する道具。十分データがあれば事前の選択は効かなくなる（ベルンシュタイン＝フォン・ミーゼスの定理）。
- **尤度 $P(\mathcal D\mid\theta)$**: $\theta$ の関数として見た「観測データの起こりやすさ」。多くの ML 損失は**負の対数尤度（NLL）**としてここから導かれる。
- **事後分布 $P(\theta\mid\mathcal D)$**: 事前を尤度で作り替えたもの。点推定の代わりに**分布**が得られ、標本平均や**信用区間（credible interval, 例：中央95%）**で不確実性を報告できる。

**予測分布（＝事後予測分布 PPD）に到達する**のが本記事の核心。新しい点 $x^{*}$ の予測は、$\theta$ を1点に決めず**事後で積分消去**する:

$$
P(y^{*}\mid x^{*};\mathcal D)=\int_\theta P(y^{*}\mid x^{*},\theta)\,P(\theta\mid\mathcal D)\,d\theta.
$$

実務では $\theta$ を事後から $M$ 個サンプルし、**モンテカルロ推定**で予測平均 $\widehat y=\frac1M\sum_s f(x^{*};\theta_s)$ と標本分散（＝予測の不確実性）を出す。

**計算困難性 → 近似推論（approximate inference）。** 事後の計算は「大文字の Hard」になり得る。理由は (1) 尤度が高次元・非線形で全サンプルを含む、(2) **証拠 $P(\mathcal D)$ の計算が実質不可能**（あらゆるデータ値への確率割り当てが要る）。そこで事後を直接計算せず近似する。記事は3種を挙げる:

- **MCMC（マルコフ連鎖モンテカルロ）**: 直前のサンプルと目標分布だけから次の $\theta$ を引き、最適値付近のサンプルを溜める。分布を陽に持たない。
- **変分推論（VI; Variational Inference）**: 柔軟な関数 $q_\phi(\theta)$ を選び $\phi$ を最適化して $q_\phi(\theta)\approx P(\theta\mid\mathcal D)$ にする（$q$ は「ガイド」）。
- **期待値伝播（Expectation Propagation）**: 事後を因数分解可能な $q$ で近似し、メッセージパッシングで因子を更新。

**「鶏と卵」を解く鍵**: 事後を計算できないのにサンプルしたい——という循環を、**正規化定数 $P(\mathcal D)$ を無視して $P(\theta\mid\mathcal D)\propto P(\mathcal D\mid\theta)P(\theta)$** とし、さらに「$\mathrm{argmax}$ は定数倍に不変」を使って回避する。記事はこの循環を「前向き（事前・尤度）→後ろ向き（事後）」のグラフとして図示し、近似推論はそれを反復に展開したものだと説く。

最後に記事は、ベイズ推論を **(a) モデリング・(b) 最適化（MLE→MAP）・(c) 機械学習** の3つのパラダイムとして整理し、ボーナスで **「ベイズ推論＝データを低次元 $\theta$ へ圧縮（符号化）し、事後で復号する」** という情報理論的な見立てを添える。

## 実験結果と知見

入門記事のため実験はない。眼目は「**点推定 $\theta^{*}$ → 事後分布 → 予測分布（PPD）**」という流れと、「**事後（とくに証拠 $P(\mathcal D)$）が解けないから近似推論が要る**」という一点。MAP は事後の最頻値だが、予測分布では MAP1点でなく $\theta$ を多数サンプルして積分する、という区別も明示される。

## 限界・批判的視点

- **近似推論アルゴリズムの中身は扱わない**（MCMC/VI/EP は「それぞれ別記事」と明言）。本記事は枠組みの直感まで。
- **「データセットごとに近似をやり直す高コスト」という PFN の出発点には触れない**。記事の近似推論は1つのデータセットに対する反復最適化で、これを**多数のデータセットにわたって一度に償却**する発想（PFN）はスコープ外。
- ベイズ最適化・ガウス過程など具体的なモデルには立ち入らず、ニューラルネットのパラメータ事後を主例にする一般論。

## PFN との接続（なぜこの wiki に置くか）

- **予測分布 ＝ PFN の近似ターゲットそのもの**。記事の $P(y^{*}\mid x^{*};\mathcal D)=\int_\theta P(y^{*}\mid x^{*},\theta)P(\theta\mid\mathcal D)d\theta$ は、[[bayesian-inference]] で言う**事後予測分布（PPD）**であり、[[prior-data-fitted-networks]]（PFN）が Transformer の順伝播1回で出そうとしている当の量。記事が「$\theta$ を多数サンプルして平均・分散」と手続き的に説く部分を、PFN は**学習済みの順伝播＋リーマン分布出力**（[[questions/riemann-distribution-buckets]]）で一気に代替する。
- **近似推論を PFN は「償却」で置き換える**。記事が挙げる MCMC/VI/EP は、いずれも**新しいデータごとに反復計算をやり直す**。PFN の発想は、事前分布から合成したデータで一度だけ訓練し、推論時は近似計算ゼロで PPD を返す——すなわち**償却ベイズ推論（amortized inference）**（[[bayesian-inference]]・[[in-context-learning]]）。本記事は「PFN が不要にした重い計算」を具体的に見せてくれる前提資料。
- **「正規化定数 $P(\mathcal D)$ が解けない」が PFN 原典の動機と一致**。[[sources/2021-transformers-can-do-bayesian-inference]] も同じ計算困難性を出発点に、「事前からサンプルできれば PPD を直接学習できる」と転換した。
- **役割分担**: [[sources/2024-bayesian-inference-step-by-step]]＝$\theta$ の事後と MAP・共役事前まで（厳密な物差しの例）／本記事＝その先の**予測分布（PPD）と近似推論**まで。2本で「ベイズ推論の土台」を相補的にカバーする。

## 用語と略称

- **MLE** = Maximum Likelihood Estimate（最尤推定。尤度を最大化する $\theta$ の1点推定）
- **MAP** = Maximum A Posteriori（最大事後確率推定。事後を最大化する $\theta$。＝事後の最頻値）
- **事前 / 尤度 / 事後** = Prior $P(\theta)$ / Likelihood $P(\mathcal D\mid\theta)$ / Posterior $P(\theta\mid\mathcal D)$
- **証拠（evidence）** = $P(\mathcal D)$。ベイズ則の分母＝周辺尤度＝正規化定数。計算困難の主因
- **予測分布（predictive distribution）/ PPD** = $\int_\theta P(y^{*}\mid x^{*},\theta)P(\theta\mid\mathcal D)d\theta$。$\theta$ を事後で積分消去した予測。PFN の近似対象 → [[bayesian-inference]]
- **信用区間（credible interval）** = 事後分布の中央 X%（例 95%）。ベイズ版の不確実性区間
- **近似推論（approximate inference）** = 事後を直接計算せず近似する手法群
- **MCMC** = Markov Chain Monte Carlo（マルコフ連鎖モンテカルロ。サンプリングで近似）
- **VI** = Variational Inference（変分推論。柔軟な $q_\phi$ で事後を関数近似）
- **EP** = Expectation Propagation（期待値伝播。因数分解＋メッセージパッシング）
- **NLL** = Negative Log-Likelihood（負の対数尤度。多くの ML 損失の出所）
- **amortize（償却）** = 事前の一括計算で個別推論を安価にすること（PFN の鍵）

## 関連ページ

- [[bayesian-inference]] — 本記事の予測分布・近似推論を PPD・償却まで延長した概念ハブ
- [[prior-data-fitted-networks]] — 記事の「近似推論」を順伝播1回に償却する枠組み
- [[in-context-learning]] — 償却ベイズ推論としての PFN 推論
- [[sources/2021-transformers-can-do-bayesian-inference]] — 同じ計算困難性を出発点に PPD を直接学習する PFN 原典
- [[sources/2024-bayesian-inference-step-by-step]] — 姉妹入門（$\theta$ 事後と MAP・共役まで）。本記事は PPD＋近似推論まで踏み込む
- [[sources/2021-mcmc-explained]] — 本記事が挙げた近似推論（MCMC/VI）の MCMC を1段深掘り（詳細釣り合い・Metropolis–Hastings）
- [[questions/riemann-distribution-buckets]] — PFN が予測分布（平均・分散・分位点）を出す仕組み
- [[translations/2020-understanding-bayesian-inference]] — 本記事の全文翻訳
