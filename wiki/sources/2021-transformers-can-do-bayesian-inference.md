---
type: source
source_path: raw/papers/TRANSFORMERS CAN DO BAYESIAN INFERENCE.pdf
source_kind: paper
title: Transformers Can Do Bayesian Inference
authors: [Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, Frank Hutter]
year: 2021
venue: ICLR 2022
ingested: 2026-05-30
tags: [pfn, bayesian-inference, in-context-learning, gaussian-process, bayesian-neural-network, riemann-distribution]
translation: "[[translations/2021-transformers-can-do-bayesian-inference]]"
---

# Transformers Can Do Bayesian Inference（PFN の原典）

> 原典: [[translations/2021-transformers-can-do-bayesian-inference]] ・ `raw/papers/TRANSFORMERS CAN DO BAYESIAN INFERENCE.pdf`
> 著者・年・会議: Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, Frank Hutter / 2021（ICLR 2022）

## 一言まとめ

**Prior-Data Fitted Network（PFN）** という枠組みを最初に提示した原典論文。「事前分布からデータセットを大量に合成して Transformer に『保留点を当てる』訓練をすると、その Transformer は 1 回の順伝播でベイズの事後予測分布（PPD）を近似できる」ことを理論と実験で示した。ガウス過程をほぼ完璧に模倣し、扱いにくい事前分布（ハイパー事前分布付き GP・BNN）でも MCMC/SVI より 200〜10000 倍速く同等以上の近似を達成する。[[sources/2022-tabpfn]]（TabPFN）はこの論文の表形式特化・大規模化版にあたる。

## 背景と問題意識

ベイズ推論は「データがどう生成されるか」という前提（事前分布）を明示でき、較正がよく解釈可能、という利点を持つ。しかし実際に予測へ使うには **事後予測分布（PPD; posterior predictive distribution, 訓練データ $\mathcal{D}$ を条件としたテスト点 $x$ のラベル分布）** を計算する必要があり、これはあらゆる仮説（タスク）$t$ について積分した量:

$$p(y\mid x,\mathcal{D})=\int_{t}p(y\mid x,t)\,p(t\mid\mathcal{D})$$

で、ほとんどの場合**扱いにくい（解析的に解けない）**。従来は事後分布 $p(t\mid\mathcal{D})$ を **MCMC（マルコフ連鎖モンテカルロ。正確だが遅い）** や **VI（変分推論。扱いやすい分布で近似）** で近似していたが、いずれも非正規化事後分布や密度値へのアクセスを要し、データセットごとに高コスト。詳細は [[bayesian-inference]]。

この論文の発想の転換は、「事後分布を一切作らず、PPD を**直接**ニューラルネットに学習させる」こと。必要なのは「事前分布からデータをサンプリングできる」という非常に弱い条件だけ。これにより、従来近似が難しかった事前分布も扱える。

## 提案手法 / 主張

**(1) Prior-Data Fitted Network（PFN）の枠組み（[[prior-data-fitted-networks]]）**
- 事前分布を「データセット生成器」とみなし、合成データセット $D\cup\{x,y\}\sim p(\mathcal{D})$ を大量に引く。
- **Prior-Data NLL** という損失 $\ell_\theta=\mathbb{E}_{D\cup\{x,y\}}[-\log q_\theta(y\mid x,D)]$ で、「データセット $D$ を見て保留点 $x$ のラベル $y$ を当てる」よう Transformer $q_\theta$ を訓練する（アルゴリズム1）。
- **理論的核心（洞察1・系1.1）**: この損失は PPD と近似 $q_\theta$ の交差エントロピー（＝加法定数を除き KL ダイバージェンス）の期待値に等しい。したがって損失最小化 = PPD への KL 最小化。**系1.2**: 分布族が十分表現力を持てば最適解は厳密な PPD に一致する。これが「Transformer はベイズ推論ができる」という題名の根拠。
- 過適合の概念がない点が重要（付録D）: 訓練損失の改善は常に「厳密な事後分布への近さ」の改善を意味し、より多く訓練するほど較正も良くなる。

**(2) Transformer アーキテクチャの適応（[[in-context-learning]]）**
- 位置エンコーディングを除いた Transformer エンコーダ。各 $(x,y)$ を 1 トークンにし、データセットを**集合**として順序不変に扱う。訓練点同士は相互に、クエリ点は訓練点にのみアテンションする（アテンションマスク）。テスト集合全体の PPD を 1 回の順伝播で出す。
- 推論時に勾配学習を行わない＝大規模言語モデルの **文脈内学習（ICL; in-context learning）** と同じ構図。PFN はそれを設計された事前分布のもとで意図的に行う。

**(3) リーマン分布（Riemann Distribution）— 回帰のための新しい予測ヘッド**
- 連続値の回帰をニューラルネットで扱うのは難しい。そこで出力範囲を、prior-data 上で等確率になるよう**バケットに離散化**し、各バケットの確率を予測する（分類問題に変換）。分布型強化学習に着想。非有界サポートは両端を半正規分布で置換。
- **定理1**: 十分なバケットがあれば、リーマン分布は任意の（リーマン積分可能・完全サポートの）連続分布を任意精度で近似できる。

## 実験結果と知見

- **GP 近似（厳密に評価可能）**: 固定ハイパラ GP の PPD（閉形式で厳密）に対し、PFN の平均・95% 信頼区間は真値とほぼ区別がつかない（図3）。メタ訓練データセットを増やすほど近づく。Attentive Neural Processes より明確に優れる。
- **扱いにくい事前分布**: ハイパー事前分布付き GP では MLE-II より 200 倍以上、NUTS より 1000〜8000 倍速く、真の PPD に最も近い。BNN では同じ性能を SVI より 1000 倍、NUTS より 10000 倍速く達成。
- **表形式分類（[[sources/2022-tabpfn]] の前身となる結果）**: GP 事前分布・**アーキテクチャ上の BNN 事前分布**（モデルの層数・幅・活性化までベイズ的に積分）で PFN を訓練。OpenML の小規模 20 データセット（30 訓練サンプル）で、**PFN-BNN が平均ランク最良**（XGBoost・Catboost・GP・KNN 等を上回る）。較正（ECE 0.025）は全ベースラインを圧倒。GPU で全 20 データセット 13 秒 ≒ XGBoost の 20 時間より 5000 倍以上速い。
- **few-shot 学習**: わずか 55 行の「ランダムな直線」事前分布で訓練＋Omniglot でファインチューンし、5-shot 5-way で最先端と同等（精度 0.865）。

## 限界・批判的視点

- **スケール**: 本論文の実験は小規模（表形式は 30〜100 サンプル、特徴量最大 60）。大規模問題へのスケールは今後の課題として明示。
- **アーキテクチャは流用**: Transformer をほぼそのまま使っており、このタスクに最適なアーキテクチャ探索は未着手。
- **事前分布依存**: PFN の品質は事前分布の設計と表現力に依存する（系1.2 は「分布族が $p$ を含むなら」という条件付き）。
- **few-shot のみファインチューニング併用**: 第7節だけは純粋な「訓練不要推論」から外れ、ターゲットドメインでのファインチューンを併用している。
- **リーマン分布のサポート設定**: 非有界分布では端の半正規スケーリングなど手当てが必要。

## 意義（なぜ重要か）

ベイズ推論の最大の障壁だった「PPD の計算困難性」を、**事前分布からのサンプリングだけ**を要件に、Transformer の 1 回の順伝播で**償却（amortize）**してしまう枠組みを確立した点で画期的。「学習アルゴリズムを設計する」のではなく「事前分布（データ生成器）を書く」だけでベイズ的予測器が手に入る、という設計パラダイムを開いた。これは [[in-context-learning]] を「意図的な近似ベイズ推論」として理論づけ、後続の [[sources/2022-tabpfn]]（TabPFN）や表形式基盤モデルの直接の出発点になった、PFN 系の**原典**である。

## 用語と略称

- **PFN** = Prior-(Data) Fitted Network（事前データ当てはめネットワーク）→ [[prior-data-fitted-networks]]
- **PPD** = Posterior Predictive Distribution（事後予測分布）→ [[bayesian-inference]]
- **Prior-Data NLL** = 事前データ負の対数尤度（PFN の訓練損失）
- **ICL** = In-Context Learning（文脈内学習）→ [[in-context-learning]]
- **GP** = Gaussian Process（ガウス過程。関数上の分布を与える確率モデル）→ [[gaussian-process]]
- **BNN** = Bayesian Neural Network（ベイズニューラルネット。重みを分布として扱う）
- **MCMC / NUTS** = マルコフ連鎖モンテカルロ／No-U-Turn Sampler（事後分布の標準的サンプリング近似）
- **VI / SVI** = 変分推論／確率的変分推論（Bayes-by-Backprop など）
- **MLE-II** = ハイパーパラメータの最大（事後）尤度推定
- **Riemann Distribution（リーマン分布）** = 出力範囲を等確率バケットに離散化した予測分布
- **ECE** = Expected Calibration Error（期待較正誤差。予測確率と実正解率のズレ）
- **amortize（償却）** = 事前の一括計算で個別推論を安価にすること

## 関連ページ

- [[prior-data-fitted-networks]] — 本論文が提示した中核概念（原典）
- [[bayesian-inference]] — 近似対象である PPD の理論的背景
- [[in-context-learning]] — 1 回の順伝播で文脈から予測する枠組み
- [[gaussian-process]] — 近似対象かつ事前分布として登場
- [[sources/2022-tabpfn]] — 本論文の表形式特化・大規模化版（TabPFN）
- [[translations/2021-transformers-can-do-bayesian-inference]] — 全文翻訳（付録 A〜H 含む）
