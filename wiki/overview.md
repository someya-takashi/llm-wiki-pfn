---
type: overview
updated: 2026-06-03
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
- **時系列基盤モデルとの収斂**: 表形式 TFM の時系列展開（TabPFN-TS / TabPFN-TS-3）は、**時系列ネイティブの基盤モデル**と同じ土俵（fev-bench）で競合する。代表が Amazon の **Chronos-2**（[[sources/2025-chronos-2]]）で、**group attention＝時系列版の [[in-context-learning|ICL]]**（関連系列・多変量の変量・共変量を「グループ」として束ね文脈共有）と**合成データのみで汎用能力を獲得**という、TabPFN/PFN と共通の設計原理に立つ。Chronos-2 が fev-bench 首位、共変量つきタスクでは TabPFN-TS が2位——表形式（TabPFN 一族）と時系列（Chronos 一族）という別系統の基盤モデルが「ICL ＋合成データ事前訓練」へ収斂しつつあることを示す。
- **実データへの適応（継続事前訓練）**: [[sources/2025-real-tabpfn]] は、合成事前訓練済み TabPFNv2 を厳選実データで継続事前訓練（continued pre-training）すると素直に強くなることを示した（Real-TabPFN）。その発展形 RealTabPFN-2.5 が、TabPFN-3・TabICLv2 が「無調整で超えるべき SOTA」として比較する基準になっている。合成 prior と実データの橋渡しは、蒸留やテスト時計算と並ぶ [[tabular-foundation-model]] の主要な適応戦略。
- **事前分布の設計**: PFN の性能は事前分布の設計に決定的に依存する。TabPFN では「単純な因果構造ほど高確率」（オッカムの剃刀）を [[structural-causal-model]] で表現したのが鍵。ベンチマークや事前分布の素材として [[gaussian-process]] が繰り返し登場する。**Mitra（[[sources/2025-mitra]], Amazon）はこの「prior 設計」を定量原理に変えた**——良い prior を性能・多様性・独自性（汎化性行列 $\mathbf{G}$ ・性能ベクトル $\mathbf{P}$）で測り、SCM＋木ベース prior の混合を原理的に選定して TabPFNv2/TabICL を上回る。「どの prior をどう混ぜるか」の方法論が TFM 設計の新しい軸になった。
- **下流応用としてのベイズ最適化**: [[bayesian-optimization]]（高コストなブラックボックス関数を「サロゲート＋獲得関数」で少ない評価回数で大域最適化する手法）は、従来サロゲートに [[gaussian-process]] を使ってきた（[[sources/2018-bayesian-optimization-tutorial]]）。GP の $O(N^3)$ コストと次元制約がボトルネックで、ここに **PFN をサロゲートに使う道**が開ける。これを正面から実装したのが **PFNs4BO**（[[sources/2023-pfns4bo]]、PFN 原典と同じ著者グループ）で、PFN を GP の「ベイズ的ドロップイン置換」として BO に据え、GP では難しい拡張（ユーザー事前分布・無関係次元・非近視眼的獲得関数）まで「事前分布を書くだけ」で実現し HPO ベンチで GP ベース SOTA（HEBO）を上回った。**高次元（100〜500 次元）BO** では **GIT-BO**（[[sources/2025-git-bo]]）が TabPFN v2 をサロゲートに使い、順伝播の勾配で能動部分空間を毎反復同定して GP ベース手法を 2 桁速く凌駕した（低次元＝PFNs4BO、高次元＝GIT-BO）。PFN がベイズ推論を償却し高速・較正のよい予測分布を返す性質が、まさに獲得関数の入力に向く。
- **下流応用としての因果効果推定**: PFN の射程は予測・最適化にとどまらず、**因果推論**にも届く。2025 年にほぼ同時期、PFN を因果効果推定へ拡張した**双子の論文**が現れた。**CausalPFN**（[[sources/2025-causalpfn]]、U Toronto/Layer6）は **潜在結果フレームワーク（Rubin 流）＋無視可能性（ignorability＝無交絡＋正値性）** に立ち、ignorability を設計で焼き込んだ DGP 事前分布（OpenML 実表＋[[structural-causal-model|TabPFN v1 生成器]]）から CATE/ATE を推定、真値の条件付き期待潜在結果（CEPO）をラベルに訓練し IHDP/ACIC/Lalonde で専用推定器群を無調整で上回った。一方 **Do-PFN**（[[sources/2025-do-pfn]]、Freiburg/Tübingen, Hutter・Schölkopf）は **SCM／do 計算（Pearl 流）** に立ち、prior に**介入 $do(t)$ つきの SCM** を据えて条件付き介入分布 $p(y\mid do(t),x)$ を予測する。Do-PFN の際立つ点は、**因果グラフを与えず・無交絡を要求せず・同定不能性を不確実性として表現する**——伝統的因果推論の3前提を同時に緩めること。両者とも理論面で「損失最小化＝（CEPO-PPD / CID への）前向き KL 最小化」を証明し、原典の「損失最小化＝PPD 近似」を因果へ一般化している。「PFN の因果版」が Rubin 流（CausalPFN）と Pearl 流（Do-PFN）の両側から立ち上がったことになり、ベイズ最適化（PFNs4BO/GIT-BO）と並ぶ「PFN を新ドメインへ展開」する潮流の一つ。両者の対比は [[structural-causal-model]] にまとめた。

## オープンな課題

- **スケーラビリティ**: アテンションが入力サンプル数に二次（$O(n^2)$）。大規模データへの拡張（線形アテンション等）。
- **データ型の一般化**: カテゴリ特徴量・欠損値・無情報特徴量への頑健性。表形式以外への一般化。
- **事前分布設計**: データセット依存の事前分布選択、因果（介入・反事実）への拡張。
- **較正と信頼性**: 較正（calibration）、分布外ロバスト性、公平性、説明可能性。

## 関連ページ

- [[prior-data-fitted-networks]] ・ [[in-context-learning]] ・ [[bayesian-inference]] ・ [[structural-causal-model]] ・ [[gaussian-process]] ・ [[tabular-foundation-model]] ・ [[bayesian-optimization]]
- 掘り下げ（questions）: [[questions/gaussian-process-intuitive-explainer]] ・ [[questions/pfn-paper-and-gaussian-process]] ・ [[questions/tabpfn-tabicl-versions-mechanism]] ・ [[questions/riemann-distribution-buckets]] ・ [[questions/evaluating-pfn-gp-approximation]] ・ [[questions/pfn-and-bayesian-optimization]]
- [[index]]
