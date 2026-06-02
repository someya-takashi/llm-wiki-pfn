---
type: concept
aliases: [GP, GPs, Gaussian Process, ガウス過程]
tags: [probabilistic-modeling, bayesian-inference, kernel-method]
related:
  - "[[bayesian-inference]]"
  - "[[prior-data-fitted-networks]]"
  - "[[bayesian-optimization]]"
sources:
  - "[[sources/2019-gp-not-for-dummies]]"
  - "[[sources/2020-gp-regression-tutorial]]"
  - "[[sources/2022-gpr-part1-basics]]"
  - "[[sources/2022-gpr-part2-concrete]]"
  - "[[sources/2025-gp-intuitive-intro]]"
  - "[[sources/2021-transformers-can-do-bayesian-inference]]"
  - "[[sources/2022-tabpfn]]"
  - "[[sources/2025-tabicl]]"
  - "[[sources/2026-tabicl-v2]]"
  - "[[sources/2018-bayesian-optimization-tutorial]]"
  - "[[sources/2021-gp-models-intro]]"
updated: 2026-06-01
---

# Gaussian Process（GP, ガウス過程）

## 一言で

**ガウス過程（GP; Gaussian Process, 関数そのものの上に確率分布を置くノンパラメトリックなベイズモデル）**。任意の有限個の入力点での出力が多変量正規分布に従う、と仮定する。平均関数（通常ゼロ）と**カーネル関数** $k(x_i,x_j)$（2 点の出力がどれだけ相関するか＝関数の滑らかさ・スケールを決める）で完全に規定される。データを観測すると、事後分布も解析的に閉形式で求まり、**予測平均と不確実性（信頼区間）を厳密に**得られる。

> 最も直感寄りの「最初の一冊」は [[sources/2019-gp-not-for-dummies]]（Yuge Shi / The Gradient、Richard Turner 講演ノート）。「高次元ガウスからのサンプリングを index プロットで描く → 数点で条件付けると非線形回帰そのものに見える → RBF カーネルで実数 $x$ に拡張＝無限次元ガウス＝GP」という流れを図で腑に落とさせる。数式より絵で「GP が何をしているか」を掴む層。

> 入門としての体系的な解説は [[sources/2020-gp-regression-tutorial]]（Jie Wang のチュートリアル）を参照。正規分布 → 多変量正規分布（相関させて滑らかな関数に）→ カーネル → 無限次元 MVN ＝「関数上の分布」へと直感的に積み上げ、予測式 $\bar{\mathbf{f}}_*=\mathbf{K}_*^{\top}[\mathbf{K}+\sigma_n^2\mathbf{I}]^{-1}\mathbf{y}$、$\text{cov}(\mathbf{f}_*)=\mathbf{K}_{**}-\mathbf{K}_*^{\top}[\mathbf{K}+\sigma_n^2\mathbf{I}]^{-1}\mathbf{K}_*$ まで導く。RBF（二乗指数）カーネルの長さスケール $l$ が滑らかさを決め、ハイパラは対数周辺尤度の最大化で最適化する。標準 GP は計算量 $O(N^3)$・メモリ $O(N^2)$ で大規模に弱い（→ スパース GP）。

> 実践/応用寄りの短い入門は [[sources/2022-gpr-part1-basics]]（Kaixin Wang / Medium）。GPflow を使い、トイデータに線形・RBF・両者の和のカーネルを当てはめて「線形＝未学習／RBF 単体＝過学習／和＝バランス最良」を示し、長さスケール $l$ と分散 $\sigma^2$ の調整（過学習 vs 未学習）を図で見せる。「ライブラリで実際にカーネルを選び・調整する」感覚を補う層。

> ノンパラメトリック性と**ニューラルネットとの関係**を軸にした直感的入門は [[sources/2025-gp-intuitive-intro]]（Fan Pu Zeng）。事前/事後・能動学習（不確実性優先サンプリング）・株価予測のアニメで動機づけ、最後に**「ガウス重み初期化の NN は無限幅極限で GP に収束する（NNGP）」**を予告する。NN ↔ GP の橋渡しは、PFN（[[prior-data-fitted-networks]]）が NN でベイズ推論を償却近似できることの土壌として示唆的。

> より理論寄り・上級の解説は [[sources/2021-gp-models-intro]]（Thomas Beckers, 制御/システム同定の観点）を参照。同じ予測式を起点に、**カーネルトリック → 再生核ヒルベルト空間（RKHS）と RKHS ノルム → GPR の予測分散を使ったモデル誤差の保証付き上界（ロバスト／シナリオ／情報理論的）→ カーネル動物園 → GP 動的モデル（GP-SSM/GP-NOE）** まで踏み込む。特に「GP の不確実性は飾りでなく、**真の関数との誤差を確率付きで抑える数学的保証**になりうる」（情報利得 $\gamma_{\max}$ を含む Srinivas らの境界）点は、PFN/TFM が掲げる「較正のよい予測分布」の規範例として重要。

## なぜ PFN の文脈で重要か

このリポジトリでは GP は 2 つの役割で繰り返し登場する。

1. **PFN の正しさを測る「物差し」**（[[sources/2021-transformers-can-do-bayesian-inference]]）
   - 固定ハイパーパラメータの GP は PPD（[[bayesian-inference]] の事後予測分布）が**閉形式で厳密に計算できる**数少ない例。だから「PFN の近似がどれだけ正しいか」を厳密な真値と突き合わせて検証できる。
   - 原典論文は、GP 事前分布から合成データを引いて PFN を訓練し、その予測平均・95% 信頼区間が厳密 GP とほぼ区別がつかないことを示した（メタ訓練データセットを増やすほど一致）。これが「Transformer はベイズ推論ができる」ことの最も直接的な実証。
   - ハイパー事前分布（カーネルのスケールや長さスケール上の分布）を付けると GP の PPD はもはや閉形式で解けなくなる。PFN はこの**扱いにくい**ケースでも、MLE-II より 200 倍以上・NUTS より 1000〜8000 倍速く、真の PPD に最も近い近似を出した。

2. **PFN に与える事前分布の一つ**
   - GP 自体を「データ生成器」＝事前分布として使い、PFN を訓練できる。[[sources/2022-tabpfn]] では GP ベースの分類事前分布（PFN-GP）を用意し、ターゲットの中央値でクラス分けして二値分類化している。

3. **GP の最大の応用先＝ベイズ最適化のサロゲート**（[[bayesian-optimization]] / [[sources/2018-bayesian-optimization-tutorial]]）
   - ベイズ最適化（BayesOpt）は高コストなブラックボックス関数を少ない評価回数で最適化する手法で、**ほぼ必ず GP をサロゲート（代理モデル）に使う**。GP が各点で予測平均と不確実性を閉形式で与えるからこそ、「次にどこを評価すると得か」を測る獲得関数（期待改善など）が計算できる。
   - 逆に、GP の $O(N^3)$ コストと次元の制約は BayesOpt のボトルネックでもある。**この「GP の代わりに使える高速サロゲート」という空白を、PFN 系（[[tabular-foundation-model]]）が埋めうる**——高次元 BO の GIT-BO は TabPFNv2 をサロゲートに採用している。

## GP と PFN の関係（まとめ）

GP は「カーネルで関数の事前分布を表し、ベイズ推論で予測する」古典的枠組み。PFN は「事前分布から合成したデータで Transformer を訓練し、ベイズ推論を**償却（amortize）**する」新しい枠組み。PFN は GP を**模倣の対象**にも**事前分布の素材**にもできる。GP に似た挙動（観測点から遠いほど不確実性が増す滑らかな予測）は、TabPFN のトイデータ実験でも観察されている。

## 関連ページ

- [[bayesian-inference]] — GP は PPD が厳密に解ける代表例
- [[prior-data-fitted-networks]] — GP を模倣／事前分布に用いる
- [[sources/2019-gp-not-for-dummies]] — GP の最も直感寄りの入門（index プロットで条件付け＝回帰を可視化）
- [[sources/2020-gp-regression-tutorial]] — GP/GPR の入門チュートリアル（本概念の初級リファレンス）
- [[sources/2022-gpr-part1-basics]] — GPR の実践/応用入門（GPflow でカーネル選択・ハイパラ調整）
- [[sources/2022-gpr-part2-concrete]] — 上記の続編。GPR を実データ（コンクリート強度）に適用し ANN 並み＋不確実性、SHAP で解釈
- [[sources/2025-gp-intuitive-intro]] — ノンパラメトリック性と NN との関係（NNGP）を軸にした直感的入門
- [[sources/2021-gp-models-intro]] — GP の上級リファレンス（RKHS・モデル誤差境界・GP 動的モデル）
- [[sources/2021-transformers-can-do-bayesian-inference]] — GP をベンチマークに PFN を検証した原典
- [[sources/2022-tabpfn]] — GP ベースの分類事前分布（PFN-GP）
- [[sources/2025-tabicl]] / [[sources/2026-tabicl-v2]] — GP 関数を合成 prior に使う TFM
- [[bayesian-optimization]] — GP を最も多用する応用領域（GP をサロゲートに使う）
- [[sources/2018-bayesian-optimization-tutorial]] — GP をサロゲートに使う BayesOpt のチュートリアル
- [[questions/gaussian-process-intuitive-explainer]] — GP を数式控えめに図で説明する直感的解説
- [[questions/pfn-paper-and-gaussian-process]] — PFN 原典と GP の関係（物差し／事前分布／高速近似）
- [[questions/tabpfn-tabicl-versions-mechanism]] — 各バージョンの TabPFN/TabICL が GP をどう扱うか
