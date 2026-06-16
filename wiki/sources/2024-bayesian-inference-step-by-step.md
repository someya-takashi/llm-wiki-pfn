---
type: source
source_path: raw/articles/Bayesian Inference A step-by-step guide.md
source_kind: blog
title: "Bayesian Inference: A step-by-step guide"
authors: [Rahuldhrh]
year: 2024
venue: Medium
ingested: 2026-06-16
tags: [bayesian-inference, prior, posterior, conjugate-prior, mle, bernoulli, sequential-bayes]
translation: "[[translations/2024-bayesian-inference-step-by-step]]"
---

# Bayesian Inference: A step-by-step guide（ベイズ推論の入門解説）

> 原典: [[translations/2024-bayesian-inference-step-by-step]] ・ `raw/articles/Bayesian Inference A step-by-step guide.md`（Medium）
> 著者・年・媒体: Rahuldhrh / 2024（Medium 個人ブログ, 2024-06-05）

## 一言まとめ

**ベイズ推論（Bayesian inference, データを見る前の信念＝事前分布を、観測データで事後分布へ更新する枠組み）を、最尤推定（MLE）との対比から説き起こし、コイン投げ（ベルヌーイ＋ベータ）と正規平均推定（正規＋正規）の2例で「事前→尤度→事後」を手を動かして追う入門記事。** 数学的に新しいことは何もないが、PFN（[[prior-data-fitted-networks]]）が「Transformer の順伝播一回で代行しようとしている当の計算」——事前分布をデータで事後へ更新する——を、最も素朴な形で体験できる教材として価値がある。

## 背景と問題意識：なぜ MLE では足りないのか

記事の出発点は **最尤推定（MLE; Maximum Likelihood Estimation, 観測データ $X$ の尤度 $P(X\mid\theta)$ を最大化するパラメータ $\theta$ を1点だけ選ぶ推定法）** の限界整理である。MLE には3つの弱点があるとする:

1. **パラメータ $\theta$ を「固定された定数」とみなす**（＝頻度論的, frequentist の立場）。$\theta$ 自身が分布を持つ可能性を表現できない。
2. **点推定（point estimate）しか出さず、不確実性を定量化しない**。「$\theta$ はだいたい 0.7、でも 0.5〜0.9 もあり得る」という幅が出せない。
3. **過適合（overfitting）しやすい**。とくにパラメータ数が多いほど、手元のデータに合わせ込みすぎる。

ベイズ的アプローチはこれを、$\theta$ を**確率変数**として事前分布 $P(\theta)$ を置き、観測データ $X$ で**事後分布** $P(\theta\mid X)$ へ更新することで克服する。この更新の道具が**ベイズ則（Bayes' rule）**:

$$
P(\theta\mid X)=\frac{P(X\mid\theta)\,P(\theta)}{P(X)}=\frac{P(X\mid\theta)\,P(\theta)}{\int P(X\mid\theta)\,P(\theta)\,d\theta}.
$$

分母 $P(X)=\int P(X\mid\theta)P(\theta)\,d\theta$ は**周辺尤度（marginal likelihood）**で、事後を確率分布として正規化する定数。

**「濡れたソファ」の直感**（記事の白眉）: 帰宅するとソファが濡れている。容疑者は「水をこぼした弟」と「侵入したサメ」。尤度だけ見ると $P(\text{濡れたソファ}\mid\text{サメ})>P(\text{濡れたソファ}\mid\text{弟})$（サメが来れば確実に濡れる）で、MLE はサメを選んでしまう。しかし事前確率 $P(\text{サメ})\ll P(\text{弟})$ を掛けると同時確率は逆転し、事後では弟が主犯になる。**「尤度 × 事前」で常識に合う結論が出る**、というベイズの効きどころを一目で見せる好例。

## 提案手法 / 主張：ベイズ推論の構成要素

記事は「尤度・事前・事後」を順に組み立てる。

- **尤度（likelihood）** $L(X,\theta)=P(X\mid\theta)$。コイン投げ（各試行が独立同分布 IID のベルヌーイ）なら $P(x_i\mid\mu)=\mu^{x_i}(1-\mu)^{1-x_i}$ を全試行で掛け合わせた積。
- **共役事前（conjugate prior）**。尤度 $P(X\mid\theta)$ と事前 $P(\theta)$ が同じ分布族に属し、結果の事後 $P(\theta\mid X)$ も同じ族になる組み合わせ。**事後が手計算で閉じた形に出る**ので扱いが楽。ベルヌーイ尤度にはベータ分布 $P(\mu\mid\alpha,\beta)\propto\mu^{\alpha-1}(1-\mu)^{\beta-1}$（$\alpha$＝成功回数、$\beta$＝失敗回数の擬似カウント）が共役。
- **事後（posterior）** をベイズ則で得て、必要なら **MAP 推定（Maximum A Posteriori, 事後分布を最大にする $\theta$ を1点選ぶ）** で点推定に落とす。

**2つの具体例**（いずれも共役なので事後が閉形式）:

- **例1（コイン投げ）**: データ $X=[1,1,1,1,1,1,1,0,0,0]$（表7・裏3）、事前は一様分布 $\theta\sim U(0,1)$。事後は $f(\theta\mid X)=\frac{11!}{7!\,3!}\theta^{7}(1-\theta)^{3}$ ＝ **ベータ分布 $\mathrm{Beta}(8,4)$** で、$\theta\approx0.7$ にピークを持つ釣鐘型になる（一様事前は $\mathrm{Beta}(1,1)$ なので、事後は $\mathrm{Beta}(1+7,\,1+3)$）。
- **例2（正規平均）**: 実数データ10点、母標準偏差は既知で3、事前は $\theta\sim\mathcal N(60,5^2)$。**正規事前 × 正規尤度の事後はふたたび正規分布**（共役）で $f(\theta\mid X)\sim\mathcal N(68.121,\,0.869)$。事前平均 60 とデータ平均（≈68.4）の**中間に引き寄せられ**、分散は事前より小さくなる（データで不確実性が縮む）。

**逐次ベイズ推論（sequential Bayesian inference）**: いったん得た事後を次の事前として再利用し、新しいデータ $Y$ でさらに更新する。これを繰り返せばデータを1つずつ取り込む**反復学習**になる。

## 実験結果と知見

入門記事なので実験はないが、要点は「**共役事前を使えば事後が閉形式で出る**」「**事後はデータ量が増えるほど締まり（分散が縮み）、事前とデータの綱引きで決まる**」という2つ。例2 の事後平均が事前とデータ平均の中間に来るのは、ベイズ更新の本質（事前の引力 vs データの引力）を数値で見せている。

## 限界・批判的視点

- **記事はパラメータ $\theta$ の事後と MAP 止まり**で、教師あり予測で本当に欲しい **事後予測分布（PPD; Posterior Predictive Distribution, 訓練データを所与とした新規テスト点 $x$ のラベル分布 $p(y\mid x,\mathcal D)=\int p(y\mid x,\theta)p(\theta\mid X)\,d\theta$）** には踏み込まない。PFN が近似対象にしているのはこの PPD（[[bayesian-inference]]）であり、本記事はその一歩手前（パラメータ事後）までを扱う、という位置づけを押さえておきたい。
- **すべて共役・低次元の例**。共役でない／高次元の事後は閉形式にならず、実際には MCMC（マルコフ連鎖モンテカルロ）や変分推論で近似する必要がある——その「計算困難性」こそが PFN の存在意義だが、記事はそこには触れない。
- 個人ブログゆえ数式の途中変形は省略（例2 は「長すぎるので省略」）。厳密な導出は別資料で補う必要がある。

## PFN との接続（なぜこの wiki に置くか）

- **PFN が償却するのはこの「事前→事後」更新そのもの**。記事は $\theta$ の事後をベイズ則で1ケースずつ手計算するが、PFN（[[prior-data-fitted-networks]]）は「事前分布からデータセットを大量合成して Transformer を一度訓練し、推論時は文脈（訓練データ）から事後予測を順伝播一回で返す」——つまり**ベイズ更新をニューラルネットに償却（amortize）**する。本記事の手計算は、その「中で起きていること」の最小モデルとして読める。
- **共役の2例＝PFN の精度評価の“厳密な物差し”**。本記事のコイン投げ（Beta–Bernoulli）と正規平均（Normal–Normal）は、まさに [[questions/pfn-bayesian-inference-evaluation-settings]] で挙げた「閉形式で厳密な事後が出るので、PFN のベイズ推論精度を厳密に検証できる自前の問題設定」に一致する。事後が手計算で出る＝PFN 出力と突き合わせる真値が手に入る。
- **逐次ベイズ推論 ≈ ICL の条件付け**。「事後を次の事前にしてデータを1点ずつ足す」逐次更新は、PFN/[[in-context-learning]] が**文脈にデータ点を追加するほど予測が締まる**挙動と同じ構図（データを増やすほど不確実性が縮む）。
- 既存の [[bayesian-inference]] 概念ページが PPD・償却という「PFN 寄りの上層」を扱うのに対し、本 source はその**土台（MLE との対比・共役事前・逐次更新）**を埋める入門リファレンス。

## 用語と略称

- **MLE** = Maximum Likelihood Estimation（最尤推定。尤度を最大化する $\theta$ を1点推定。$\theta$ を定数扱いする頻度論的手法）
- **MAP** = Maximum A Posteriori（最大事後確率推定。事後分布を最大化する $\theta$ を1点選ぶ）
- **事前分布 / 事後分布** = Prior / Posterior（データを見る前 / 後の $\theta$ の分布）
- **尤度（likelihood）** $P(X\mid\theta)$ = パラメータ $\theta$ のもとでデータ $X$ が観測される確率
- **共役事前（conjugate prior）** = 事前と事後が同じ分布族になる事前の選び方（事後が閉形式で出る）
- **ベルヌーイ分布 / ベータ分布** = 二値試行の分布 / その共役事前（$[0,1]$ 上の連続分布）
- **IID** = Independent and Identically Distributed（独立同分布）
- **周辺尤度（marginal likelihood）** $P(X)=\int P(X\mid\theta)P(\theta)d\theta$ = 事後を正規化する定数
- **逐次ベイズ推論（sequential Bayesian inference）** = 事後を次の事前に使ってデータを順に取り込む更新
- **PPD** = Posterior Predictive Distribution（事後予測分布。PFN の近似対象）→ [[bayesian-inference]]
- **amortize（償却）** = 事前の一括計算で個別推論を安価にすること

## 関連ページ

- [[bayesian-inference]] — 本記事の「事前→事後」を PPD・償却推論まで延長した概念ハブ
- [[prior-data-fitted-networks]] — このベイズ更新を Transformer の順伝播に償却する枠組み
- [[questions/pfn-bayesian-inference-evaluation-settings]] — 本記事の共役例（Beta–Bernoulli・Normal–Normal）＝ PFN 精度評価の厳密な物差し
- [[in-context-learning]] — 逐次ベイズ更新と対応する「文脈にデータを足すほど締まる」推論
- [[sources/2020-understanding-bayesian-inference]] — 姉妹入門。本記事は θ 事後＋MAP まで、あちらは予測分布(PPD)＋近似推論(MCMC/VI/EP)まで踏み込む
- [[sources/2021-transformers-can-do-bayesian-inference]] — 「ベイズ推論を Transformer で代行する」PFN 原典
- [[translations/2024-bayesian-inference-step-by-step]] — 本記事の全文翻訳
