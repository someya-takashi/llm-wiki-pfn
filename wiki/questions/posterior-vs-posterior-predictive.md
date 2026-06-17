---
type: question
asked: 2026-06-17
question: "事後分布と事後予測分布の違いは？ わかりやすく説明してほしい。"
sources_used:
  - "[[sources/2024-bayesian-inference-step-by-step]]"
  - "[[sources/2020-understanding-bayesian-inference]]"
  - "[[sources/2019-conceptual-intro-mcmc]]"
---

# 事後分布と事後予測分布の違い

> 問い「事後分布と事後予測分布の違いがわかりません」への回答。ベイズ推論の基礎（[[bayesian-inference]]）と、入門ソース群（[[sources/2024-bayesian-inference-step-by-step]] / [[sources/2020-understanding-bayesian-inference]] / [[sources/2019-conceptual-intro-mcmc]]）を統合して整理する。

## 一言でいうと

| | **事後分布（posterior）** $P(\theta\mid D)$ | **事後予測分布（PPD; posterior predictive）** $P(y^*\mid x^*,D)$ |
|---|---|---|
| **何の上の分布か** | **パラメータ $\theta$** の上の分布 | **未来の観測値 $y^*$**（次に出るデータ）の上の分布 |
| **答える問い** | 「データを見たいま、パラメータはどのへんが尤もらしい？」 | 「次に観測するデータはどうばらつく？」 |
| **不確実性の中身** | パラメータの不確実性だけ | パラメータの不確実性 **＋** 観測ノイズの両方 |
| **位置づけ** | 推論の**途中成果**（モデルの内部状態） | 予測の**最終成果**（実際に使う量） |

**核心**: 事後分布は「**モデルのつまみ（パラメータ）がどの値か**」の分布、事後予測分布は「**次のデータがどう出るか**」の分布。後者は前者を使って $\theta$ を積分消去（周辺化）して作る。

## 関係式：PPD は事後分布を「平均」したもの

$$
\underbrace{P(y^*\mid x^*, D)}_{\text{事後予測分布}}=\int \underbrace{P(y^*\mid x^*,\theta)}_{\text{尤度（θ を1つ決めた予測）}}\;\underbrace{P(\theta\mid D)}_{\text{事後分布（重み）}}\,d\theta
$$

読み方は「**ありうる各 $\theta$ で予測を立て、その $\theta$ が尤もらしい確率（＝事後分布）で重みづけ平均する**」。つまり **PPD ＝ 事後分布で重みづけした予測の混ぜ合わせ**。1つの $\theta$ に決め打ちせず「$\theta$ が分からない分の不確実性」も予測に流し込むのがポイント（[[bayesian-inference]]、[[sources/2019-conceptual-intro-mcmc]] §3.3）。

## 具体例1：コイン投げ（[[sources/2024-bayesian-inference-step-by-step]]）

表7回・裏3回を観測。表の出る確率を $\mu$ とする。

- **事後分布** $P(\mu\mid D)=\mathrm{Beta}(8,4)$ … これは **$0$〜$1$ 上の曲線**で、「$\mu$ は 0.7 あたりが尤もらしいが 0.5〜0.9 もありうる」という **$\mu$ についての分布**。次の1投がどうなるかは、まだ直接は言っていない。
- **事後予測分布**「次の1投が表」$=\displaystyle\int_0^1 \mu\,\mathrm{Beta}(\mu;8,4)\,d\mu=\mathbb{E}[\mu]=\frac{8}{12}\approx 0.667$ … これは **次の観測（表/裏）についての分布**（ベルヌーイ $p\approx0.667$）。$\mu$ を1点に決めず、事後の全範囲で平均してある。

「$\mu$ の最頻値 0.7 をそのまま使えばいいのでは？」と思うかもしれない。それが **MAP 推定（点推定）** だが、PPD は1点に決めず分布全体で平均するので、**$\mu$ の不確実性が大きいほど予測も適切に慎重（裾が広い）**になる。

## 具体例2：温度の測定（[[sources/2019-conceptual-intro-mcmc]] の演習）

5回のノイズ付き測定から真の気温 $T$ を推定。

- **事後分布** $P(T\mid D)$ … 「**真の気温 $T$** はどのへんか」の分布。
- **事後予測分布** $P(\hat T_6\mid D)$ … 「**次の6回目の測定値 $\hat T_6$**」がどうばらつくか。ここには $T$ の不確実性に加えて **測定ノイズ $\sigma_6$** も乗る。だから PPD は事後分布より**必ず幅が広く**（または同等に）なる ── 観測ノイズの分だけ上乗せされるから。

この「**PPD は事後分布＋観測ノイズで必ず広がる**」が、両者を取り違えないための実用的な目印になる。

## なぜ2つあるのか・どちらを使うか

- **事後分布**は「世界の仕組み（パラメータ）を理解・解釈したい」とき（例：効果の大きさ、係数の符号）。
- **事後予測分布**は「**次に何が起きるかを予測したい**」とき。教師あり学習で実際に欲しいのはこちら。$x^*$ を入れて $y^*$ の分布（平均・信頼区間）を返すのが目的だから。

MCMC（[[markov-chain-monte-carlo]]）の視点だと、まず事後分布から $\theta$ を多数サンプルし、各サンプルで予測を立てて平均すると PPD が得られる ── つまり**事後分布は PPD を作るための中間材料**。

## PFN との接続

[[prior-data-fitted-networks]]（PFN）は、**事後分布 $P(\theta\mid D)$ を明示的に作らず、いきなり事後予測分布 $P(y^*\mid x^*,D)$ を Transformer の順伝播1回で出力**する（[[bayesian-inference]] の「償却ベイズ推論」）。従来は「①事後分布を MCMC でサンプル → ②予測を平均して PPD」という2段だったのを、PFN は **②（最終的に欲しい PPD）を直接償却**する。[[sources/2019-conceptual-intro-mcmc]] が説く「**本当に欲しいのは事後そのものでなく事後上の積分（＝PPD のような期待値）**」という思想と、PFN の設計はぴたり一致する。

## 覚え方（たとえ）

- **事後分布**＝「料理人の腕前（$\theta$）はどのくらいか」の見立て。
- **事後予測分布**＝「次に出てくる一皿（$y^*$）はどんな味か」の予想。腕前の不確実性に、その日のムラ（観測ノイズ）も足して予想する。

## 用語と略称

- **事後分布（posterior）** $P(\theta\mid D)$ = データ $D$ を見た後のパラメータ $\theta$ の分布
- **事後予測分布（PPD; Posterior Predictive Distribution）** $P(y^*\mid x^*,D)$ = $\theta$ を積分消去した、新規観測 $y^*$ の予測分布。PFN の近似ターゲット
- **周辺化（marginalization）** = 興味のない変数（ここでは $\theta$）を積分して消すこと
- **MAP 推定** = Maximum A Posteriori（事後分布を最大化する $\theta$ の1点推定）
- **尤度（likelihood）** $P(y^*\mid x^*,\theta)$ = $\theta$ を1つ固定したときの予測確率

## 関連ページ

- [[bayesian-inference]] — 事後・PPD・償却推論の概念ハブ
- [[markov-chain-monte-carlo]] — 事後からサンプル→平均で PPD を作る従来手段
- [[prior-data-fitted-networks]] — PPD を順伝播1回で直接償却する枠組み
- [[sources/2024-bayesian-inference-step-by-step]] — θ の事後と MAP まで（コイン投げ・共役の入門）
- [[sources/2020-understanding-bayesian-inference]] — 予測分布（PPD）と近似推論まで踏み込む入門
- [[sources/2019-conceptual-intro-mcmc]] — 事後予測 §3.3、「欲しいのは事後上の積分」の精密化
- [[questions/pfn-bayesian-inference-evaluation-settings]] — PPD を物差しに PFN を評価する問題設定
