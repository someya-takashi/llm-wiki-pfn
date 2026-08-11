---
type: concept
aliases: [置換不変ニューラルネットワーク, permutation invariance, 置換不変性, Set Transformer, DeepSets, set pooling, 集合入力ネットワーク, set-input networks, ISAB, PMA, SAB, inducing points, 誘導点, 順序不変]
tags: [architecture, attention, meta-learning, set-structured-data]
related:
  - "[[prior-data-fitted-networks]]"
  - "[[in-context-learning]]"
  - "[[tabular-foundation-model]]"
  - "[[gaussian-process]]"
sources:
  - "[[sources/2018-set-transformer]]"
  - "[[sources/2021-transformers-can-do-bayesian-inference]]"
  - "[[sources/2022-tabpfn]]"
  - "[[sources/2025-tabicl]]"
  - "[[sources/2026-tabpfn-3]]"
  - "[[sources/2025-nanotabpfn]]"
updated: 2026-08-12
---

# Permutation-Invariant Neural Networks（置換不変ニューラルネットワーク）

## 一言で

**置換不変ニューラルネットワーク（permutation-invariant neural networks, 集合入力ネットワーク）** は、**「集合」を入力にとり、要素の並び順に依存しない出力を返す**ように設計されたニューラルネットの一族。集合には順序という概念がないので、(1) **置換不変性**（要素を並べ替えても出力が同じ）と (2) **可変サイズの入力**（要素数 $n$ が変わっても動く）が必須要件になる。通常のフィードフォワード NN は両方に違反し、RNN は順序に敏感なので、専用の設計が要る。この wiki の主題である [[prior-data-fitted-networks]]（PFN）は「**データセットそのもの**を1つの集合として Transformer に入れる」モデルであり、その足元はまるごとこの概念の上に建っている。

> なぜ PFN に必須か: 訓練データ $\{(x_1,y_1),\dots,(x_n,y_n)\}$ は**順序をもたない集合**である。行の並び順を変えただけで予測が変わるモデルは、ベイズ推論の近似として原理的におかしい（事後分布はデータの順序に依存しない）。だから PFN 系は必ず「位置エンコーディングを外した Transformer ＝ 置換不変な集合処理器」として作られる。

## 2つの基本要件と用語

- **置換不変（invariant）**: $f(\pi X)=f(X)$。並べ替えても**出力が変わらない**。集合全体への出力（ラベル・統計量・予測分布）に要求される。
- **置換同変（equivariant）**: $f(\pi X)=\pi f(X)$。並べ替えると**出力も同じ順で並べ替わる**。要素ごとの特徴を作る中間層に要求される。
- 定石は「**同変な層を積み重ねてから、最後に不変な集約（プーリング）を1回かける**」。同変×同変=同変、同変の後に不変=不変、なので全体が置換不変になる。

## 代表手法の系譜

### (1) set pooling / DeepSets — 「独立に埋め込んで、足す」

最初の体系的な解（Zaheer et al. 2017「DeepSets」、Edwards & Storkey「Neural Statistician」）。各要素 $x_i$ を**独立に**エンコーダ $\phi$ に通し、mean/sum/max などの**固定プーリング**で潰し、後処理 $\rho$ をかける:

$$\mathrm{net}(\{x_1,\ldots,x_n\})=\rho(\mathrm{pool}(\{\phi(x_1),\ldots,\phi(x_n)\}))$$

驚くべきことに「sum プーリング＋連続な $\rho,\phi$」で**すべての置換不変関数を表現できる**（普遍近似）。ただし理論上表現できることと学習しやすいことは別で、**各要素を独立に処理するため要素間の相互作用（比較・重複・説明の取り合い）が捨てられ**、クラスタリングのような相互作用が本質のタスクで過少適合する（[[sources/2018-set-transformer]] §5 が実証）。

### (2) Set Transformer — 「アテンションで相互作用ごと処理する」

[[sources/2018-set-transformer]]（Lee et al., ICML 2019）が、この系譜の決定版。3つの部品からなる:

- **SAB（Set Attention Block）** = 位置エンコーディングを除いた Transformer ブロックによる**集合内自己アテンション** MAB(X, X)。要素間のペアワイズ〜高次の相互作用をエンコード。計算量 $O(n^2)$。
- **ISAB（Induced Set Attention Block）** = **誘導点（inducing points）**——$m$ 個の学習可能なベクトル $I$——を介した2段アテンション。$I$ が集合を要約し（$H=\mathrm{MAB}(I,X)$）、集合が要約を参照する（$\mathrm{MAB}(X,H)$）。計算量を $O(nm)$（$n$ に線形）へ削減。名前はスパース [[gaussian-process]] の誘導点法（少数の擬似入力で $O(N^3)$ を回避する技法）に由来する。
- **PMA（Pooling by Multihead Attention）** = プーリング自体を学習可能に。$k$ 個の学習シードベクトルをクエリにして集約し、$k>1$ なら「相関した複数出力」（例: クラスタ中心の集合）も出せる。「効く要素を探してアテンションする」集約は、固定 mean/max より本質的に柔軟。

理論的にも全体が置換不変で、かつ**置換不変関数の普遍近似器**であることが証明されている。象徴的な実験が**償却クラスタリング**: EM（[[expectation-maximization]]）の反復で解くはずの [[gaussian-mixture-model]] のパラメータ推定を順伝播1回に**償却（amortize）**し、EM 1ステップ後にはオラクルすら上回った。「反復推論を学習で1回に畳む」——後の PFN の構図の、アーキテクチャ側からの先行例である。

### (3) PFN 系 — 「データセット＝集合」を本気で使う

[[prior-data-fitted-networks]]（PFN）と後続の表形式基盤モデル（[[tabular-foundation-model]]）は、この置換不変設計を「訓練データセットを文脈として丸ごと入れる」ために使う:

- **PFN 原典** [[sources/2021-transformers-can-do-bayesian-inference]] / **TabPFN** [[sources/2022-tabpfn]]: 位置エンコーディングなしの Transformer で $(x,y)$ ペアを1トークンずつ集合として入力。訓練点同士は相互に、クエリ点は訓練点にのみアテンション（非対称マスク）。Set Transformer の結論が示唆した「**ベイズ事後推論のメタ学習**」をまさに実現した形。
- **TabICL** [[sources/2025-tabicl]]: 特徴量埋め込みを「置換不変なセル値集合→埋め込み」という**集合入力問題**として再定式化し、**Set Transformer（ISAB, 誘導点128）をそのまま採用**。最も直接的な継承。
- **TabPFN-3** [[sources/2026-tabpfn-3]]: 全行への二次アテンションを避けるため**誘導点**（128個）で訓練集合を要約。ISAB の発想の大規模版。
- **nanoTabPFN** [[sources/2025-nanotabpfn]]: 「位置埋め込みを使わない」「テスト行同士を見せない非対称マスク」など、置換不変性を保つ最小実装を教育的に示す。
- 完全な不変性が破れる場合の実務対処: [[sources/2025-tabicl]] は RoPE 導入で列順序の不変性を破る代わりに**列置換アンサンブル**で近似回復。TabPFN v2 も特徴量順のシャッフル・アンサンブルを使う。「不変性は設計で保証するか、アンサンブルで近似回復するか」という実務パターン。

## なぜ PFN の文脈で重要か

1. **PFN の前提条件**: PFN が近似する事後予測分布（PPD, [[bayesian-inference]]）はデータの順序に依存しない量。アーキテクチャの置換不変性は「正しいベイズ近似器であること」の必要条件であり、PFN 原典のアブレーション（付録 E.3）でも順序不変設計の寄与が検証されている。
2. **スケーリングの鍵**: 文脈（データセット）が大きくなると $O(n^2)$ アテンションが律速になる。誘導点（ISAB）はこれを $O(nm)$ に落とす標準手段として TabICL・TabPFN-3 に受け継がれ、**表形式基盤モデルの大規模化を支える技術**になっている。
3. **償却推論の設計図**: 「反復アルゴリズム（EM・[[markov-chain-monte-carlo|MCMC]]）が解いてきた推論を、集合を読むネットワークの順伝播1回に置き換える」という PFN の中心思想は、Set Transformer の償却クラスタリングが最初に具体化したパターンの一般化と見なせる。

## 関連ページ

- [[sources/2018-set-transformer]] — 本概念の代表手法（SAB/ISAB/PMA、普遍近似、償却クラスタリング）
- [[prior-data-fitted-networks]] — 「データセット＝集合」入力でベイズ推論を償却する枠組み
- [[in-context-learning]] — 集合として与えた文脈から重み更新なしに予測する構図
- [[tabular-foundation-model]] — 置換不変設計＋誘導点で大規模化した表形式基盤モデル群
- [[gaussian-process]] — ISAB の誘導点の由来（スパース GP）
- [[expectation-maximization]] / [[gaussian-mixture-model]] — 償却クラスタリングが置き換える反復推論
- [[bayesian-inference]] — 順序に依存しない事後・PPD（置換不変性が必要な理由）
