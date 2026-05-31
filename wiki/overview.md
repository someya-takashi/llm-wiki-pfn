---
type: overview
updated: 2026-05-30
---

# PFN（Prior-Data Fitted Networks）総括

> このページは PFN 領域全体の地図です。原典を ingest するたびに随時更新します。

## PFN とは

**Prior-Data Fitted Network（PFN, 事前分布からサンプリングした合成データセットで一度だけ訓練し、推論時には重み更新なしに in-context でベイズ推論を近似するニューラルネットワーク）**。「新しいデータが来たらモデルを当てはめる」という機械学習の基本動作を、「訓練済みネットワークにデータセットを丸ごと 1 回通す」へ置き換えるのが核心。詳細は [[prior-data-fitted-networks]]。

## 主要な系譜・パラダイム

現時点の地図（ingest が進むにつれ拡張）:

- **理論的基盤**: [[bayesian-inference]] の事後予測分布（PPD）を、ニューラルネットで **償却（amortize）** して近似する。この PPD 近似が PFN 全体の "解こうとしている問題"。PFN 原典（[[sources/2021-transformers-can-do-bayesian-inference]]）が、訓練損失最小化＝PPD への KL 最小化であることを証明した。
- **推論の枠組み**: [[in-context-learning]]。重み更新なし・順伝播一回で、文脈（訓練データ）から予測する。大規模言語モデルの ICL と同じ構図を、設計された事前分布のもとで意図的に行う。
- **原典**: Transformers Can Do Bayesian Inference（[[sources/2021-transformers-can-do-bayesian-inference]]）。PFN の枠組みと理論を最初に提示。固定 [[gaussian-process]] をほぼ完璧に模倣し、扱いにくい GP/BNN でも MCMC/SVI より 200〜10000 倍速い近似を達成。回帰用の予測ヘッド「リーマン分布」も導入。
- **系譜（代表手法）**: 原典(2021) → **TabPFN v1**（[[sources/2022-tabpfn]], 2022 ICLR）→ **TabPFN v2**（[[sources/2025-tabpfn-v2]], 2025 Nature）→ **TabPFN-2.5**（[[sources/2025-tabpfn-2-5]], 2025 arXiv）→ **TabPFN-3**（[[sources/2026-tabpfn-3]], 2026 arXiv）。v1 は表形式データ向けに [[structural-causal-model]]＋BNN 混合を事前分布として設計し小規模表データで GBDT を上回った。v2 は 10,000 サンプル・500 特徴量へ 50 倍スケールし [[tabular-foundation-model]] へ発展。2.5 は 50,000 サンプル・2,000 特徴量へスケールし TabArena 首位＋蒸留。3 は **100 万行**へスケールし **テスト時計算（Thinking mode）** を導入、多クラス・関係データ・時系列・テキスト・因果へ拡大した現最前線（アーキは v1 流の行 ICL へ回帰）。
- **別系統の兄弟 TFM（Inria 系）**: TabICL（[[sources/2025-tabicl]], 2025, 分類専用）→ **TabICLv2**（[[sources/2026-tabicl-v2]], 2026, 回帰対応・QASSMax・オープンウェイトの SOTA）。TabPFN 一族とは別グループ（Inria）が「列ごと埋め込み→行圧縮→行 ICL」の 3 段アーキで大規模対応を実現。**この TabICLv2 のアーキを TabPFN-3 が採用**しており（"TabICLv2 ベース"＝v1 流 ICL への回帰の源流）、Prior Labs 系（TabPFN）と Inria 系（TabICL）が相互に影響し合っていることを示す。
- **表形式基盤モデル化**: [[tabular-foundation-model]]。単一の事前訓練モデルが予測だけでなく生成・密度推定・埋め込み・関係/時系列/テキスト/因果まで担う、言語/画像に続く基盤モデルのパラダイムが表データに到来。TabPFN-3 で **学習規模に加え推論時計算（Thinking）も性能軸**になり、TFM が「構造化データ推論のコアエンジン」へ向かう。
- **実データへの適応（継続事前訓練）**: [[sources/2025-real-tabpfn]] は、合成事前訓練済み TabPFNv2 を厳選実データで継続事前訓練（continued pre-training）すると素直に強くなることを示した（Real-TabPFN）。その発展形 RealTabPFN-2.5 が、TabPFN-3・TabICLv2 が「無調整で超えるべき SOTA」として比較する基準になっている。合成 prior と実データの橋渡しは、蒸留やテスト時計算と並ぶ [[tabular-foundation-model]] の主要な適応戦略。
- **事前分布の設計**: PFN の性能は事前分布の設計に決定的に依存する。TabPFN では「単純な因果構造ほど高確率」（オッカムの剃刀）を [[structural-causal-model]] で表現したのが鍵。ベンチマークや事前分布の素材として [[gaussian-process]] が繰り返し登場する。
- **下流応用としてのベイズ最適化**: [[bayesian-optimization]]（高コストなブラックボックス関数を「サロゲート＋獲得関数」で少ない評価回数で大域最適化する手法）は、従来サロゲートに [[gaussian-process]] を使ってきた（[[sources/2018-bayesian-optimization-tutorial]]）。GP の $O(N^3)$ コストと次元制約がボトルネックで、ここに **TFM をサロゲートに使う道**が開ける（高次元 BO の GIT-BO が TabPFNv2 を採用）。PFN がベイズ推論を償却し高速・較正のよい予測分布を返す性質が、まさに獲得関数の入力に向く。

## オープンな課題

- **スケーラビリティ**: アテンションが入力サンプル数に二次（$O(n^2)$）。大規模データへの拡張（線形アテンション等）。
- **データ型の一般化**: カテゴリ特徴量・欠損値・無情報特徴量への頑健性。表形式以外への一般化。
- **事前分布設計**: データセット依存の事前分布選択、因果（介入・反事実）への拡張。
- **較正と信頼性**: 較正（calibration）、分布外ロバスト性、公平性、説明可能性。

## 関連ページ

- [[prior-data-fitted-networks]] ・ [[in-context-learning]] ・ [[bayesian-inference]] ・ [[structural-causal-model]] ・ [[gaussian-process]] ・ [[tabular-foundation-model]] ・ [[bayesian-optimization]]
- [[index]]
