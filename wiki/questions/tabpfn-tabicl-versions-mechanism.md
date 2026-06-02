---
type: question
asked: 2026-06-01
question: "TabPFN・TabICL の各バージョンは、PFN 原典のように『ガウス過程（固定カーネル）を事前分布として重みに焼き込む』方式を同じように用いているか？"
sources_used:
  - "[[sources/2021-transformers-can-do-bayesian-inference]]"
  - "[[sources/2022-tabpfn]]"
  - "[[sources/2025-tabpfn-v2]]"
  - "[[sources/2025-tabpfn-2-5]]"
  - "[[sources/2026-tabpfn-3]]"
  - "[[sources/2025-tabicl]]"
  - "[[sources/2026-tabicl-v2]]"
---

# TabPFN・TabICL の各バージョンは「ガウス過程」をどう扱うか

> 問い: 「PFN 原典が **GP（固定カーネル）を事前分布として重みに焼き込む** のは理解した。TabPFN・TabICL の各バージョンも、GP を同じように使っているのか？」への回答。前提の議論は [[questions/pfn-paper-and-gaussian-process]]、各世代の事前分布の中身は [[structural-causal-model]] を参照。

## 一言で

PFN の一般機構（事前分布 → 合成データ → 重みに焼き込み → in-context 推論）は全バージョン共通だが、**「ガウス過程（GP）を事前分布に使う度合い」は世代で大きく変わる**。

**GP を“固定カーネルの事前分布”として丸ごと重みに焼き込むのは、事実上 PFN 原典（[[sources/2021-transformers-can-do-bayesian-inference]]）のトイ例だけ**。表形式モデル（TabPFN／TabICL）では事前分布の主役は **[[structural-causal-model|SCM]]（構造的因果モデル）** に移り、GP の役割は世代ごとに「ベースライン／物差し」「多数の合成関数の中の一材料」「不使用」へと変化する。

つまり「各バージョンで GP の扱いは同じか？」への答えは **No（GP の使われ方は世代でかなり違う）**。共通なのは PFN という枠組みであって、「GP を事前分布にする」やり方ではない。

## なぜ GP が主役でなくなるのか

原典で GP を焼き込んだのは、**GP が「事後予測分布（PPD; posterior predictive distribution）を厳密に閉形式で計算できる」数少ない例**で、近似の正しさを真値と突き合わせて検証できる“都合のよいトイ”だったから（[[questions/pfn-paper-and-gaussian-process]] の「関係1：物差し」）。

しかし実データの表形式では、列どうしが**因果的に依存**する（例: 年齢 → 収入）。GP 単独のカーネル（「近い入力は似た出力」）ではこの構造を表しきれない。そこで TabPFN は「世の中の表データは何らかの SCM が生成している」という、より表現力の高い事前分布に乗り換えた（[[structural-causal-model]]）。GP は、その豊かな事前分布の中の**一材料**に格下げされる（使われる場合）か、まったく使われなくなる。

## GP の役割：バージョン別

| モデル | GP の役割 | 具体 |
|---|---|---|
| **PFN 原典 2021** | **事前分布そのもの＋検証の物差し** | 固定カーネル GP（RBF 等）から合成データを引いて焼き込み、厳密 GP 事後で近似精度を検証（図3）。ハイパー事前分布付き GP も対象 |
| **TabPFN v1 (2022)** | **ベースライン＋分類用 GP 事前分布（PFN-GP）変種**（主役ではない） | 主 prior は **SCM＋BNN の混合**。GP は比較対象、および分類用に作った GP 事前分布の一変種として登場。トイデータで「観測から遠いほど不確実性が増す」GP 様の挙動が観察される |
| **TabPFN v2 (2025)** | **事前分布に不使用** | prior は SCM ベース（DAG 成長・NN 辺・カテゴリ・木・ノイズ）。GP を prior 材料に使う記述はない |
| **TabPFN-2.5 (2025)** | **事前分布に不使用** | v2 系 SCM をスケール。GP は prior に登場しない |
| **TabPFN-3 (2026)** | **事前分布に不使用** | SCM を時間（動的 SCM）・空間・OOD まで拡張。GP は prior 材料ではない |
| **TabICL v1 (2025, Inria)** | **prior の“活性化関数の一つ”として GP 由来関数** | SCM prior の活性化を 4→19+ に多様化し、その中に**ガウス過程由来のランダム関数**を含む。さらに木ベース SCM を 30% 混合 |
| **TabICLv2 (2026)** | **prior の“8 種ランダム関数の 1 つ”が GP** | 合成 prior の関数型に **GP（多変量・滑らかさを理論証明）** を含む（他は MLP/Tree Ensemble/Discretize/Linear/Quadratic/EM/Product）。GP は 8 分の 1 の材料 |

## まとめ

- **「GP（固定カーネル）を事前分布として焼き込む」のは PFN 原典のトイ例だけ。** 表形式モデルではこの意味での「GP 焼き込み」は行われない。
- 表形式系での GP の役割は世代で次のように変化する:
  - **ベースライン／検証の物差し**（原典・TabPFN v1）、
  - **多数の合成関数の一材料**（TabICL v1 の活性化、TabICLv2 の 8 種の 1 つ）、
  - **不使用**（TabPFN v2 / 2.5 / 3＝SCM ベース）。
- したがって「各バージョンで GP の扱いは同じ」ではない。**共通なのは PFN の一般機構**（事前分布を焼き込み、推論は重み更新なしの 1 回の順伝播）であって、GP の使い方ではない。一般機構そのものの世代比較やアーキテクチャの違いは [[prior-data-fitted-networks]] を参照。

## 用語と略称

- **GP** = Gaussian Process（ガウス過程。「近い入力は似た出力」をカーネルで表す、関数上の確率モデル）→ [[gaussian-process]]
- **PFN** = Prior-Data Fitted Network（事前分布の合成データで一度訓練し、推論は in-context で行う枠組み）→ [[prior-data-fitted-networks]]
- **ICL** = In-Context Learning（文脈内学習。推論時に重み更新なしで文脈から予測）→ [[in-context-learning]]
- **PPD** = Posterior Predictive Distribution（事後予測分布）→ [[bayesian-inference]]
- **SCM** = Structural Causal Model（構造的因果モデル。表形式モデルの主たる事前分布）→ [[structural-causal-model]]
- **PFN-GP** = TabPFN 文脈で、GP を素材に作った（分類用の）事前分布変種
- **ハイパー事前分布（hyper-prior）** = カーネルのハイパーパラメータ自体に置く分布。原典 §5.2 の GP がこれ

## 関連ページ

- [[questions/pfn-paper-and-gaussian-process]] — 前提（PFN がどう GP を物差し／事前分布に使うか・出力の仕組み・カーネル/ハイパラの扱い）
- [[gaussian-process]] — GP 概念ハブ
- [[structural-causal-model]] — 表形式モデルの主たる事前分布（GP に代わって主役になったもの・各世代の中身）
- [[prior-data-fitted-networks]] — PFN の一般機構と代表手法の世代比較（GP に限らない違い）
- [[sources/2021-transformers-can-do-bayesian-inference]] — GP を焼き込み＋物差しにした原典
- [[sources/2022-tabpfn]] — SCM+BNN 主体（GP はベースライン/PFN-GP 変種）
- [[sources/2025-tabicl]] / [[sources/2026-tabicl-v2]] — prior の一材料として GP を含む（活性化／8 種関数の 1 つ）
