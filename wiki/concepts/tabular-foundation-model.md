---
type: concept
aliases: [Tabular Foundation Model, 表形式基盤モデル, tabular FM]
tags: [foundation-model, tabular, in-context-learning, generative-model]
related:
  - "[[prior-data-fitted-networks]]"
  - "[[in-context-learning]]"
  - "[[bayesian-inference]]"
  - "[[structural-causal-model]]"
  - "[[bayesian-optimization]]"
sources:
  - "[[sources/2025-tabpfn-v2]]"
  - "[[sources/2025-tabpfn-2-5]]"
  - "[[sources/2026-tabpfn-3]]"
  - "[[sources/2025-tabicl]]"
  - "[[sources/2026-tabicl-v2]]"
  - "[[sources/2025-real-tabpfn]]"
updated: 2026-06-01
---

# Tabular Foundation Model（表形式基盤モデル）

## 一言で

**表形式基盤モデル（tabular foundation model, 一度だけ大規模に事前訓練しておき、個別データへの追従なしに表形式データの多様なタスク——予測・生成・密度推定・埋め込み・ファインチューニング——を 1 つのモデルで担う基盤モデル）**。言語（GPT 等）や画像で確立した「基盤モデル」のパラダイムを、行と列のスプレッドシート型データに持ち込んだもの。代表例は TabPFN v2（[[sources/2025-tabpfn-v2]]）。

## なぜ「表」では難しかったか

言語や画像と違い、表形式データは**データセットごとに列の意味が全く異なる**（ある表の「列3」は血圧、別の表では株価）。この異質性のため、画像・言語のように「巨大な公開データで 1 つのモデルを事前訓練して使い回す」が成立しにくく、各データセットに個別のモデルを当てはめる（しかも GBDT が強い）のが常識だった。表形式の基盤モデルは長らく「欠けたピース」とされてきた。

## どう実現したか（PFN との関係）

表形式基盤モデルは、このリポジトリの中核 [[prior-data-fitted-networks]]（PFN）＋[[in-context-learning]] の発展形として実現された。鍵は **公開データの代わりに合成データを使う**こと:

- **合成データで事前訓練**: [[structural-causal-model]] ベースの生成器から約 1 億の多様な合成データセットを作り、Transformer に「データセットを文脈に取り保留点を当てる」訓練を一度だけ施す（プライバシー・著作権・テスト汚染を回避）。
- **推論時は ICL**: 実データの訓練部分＋テスト部分を丸ごと入力し、重み更新なし・1 順伝播で予測（近似ベイズ推論＝[[bayesian-inference]] の PPD）。
- **基盤モデルらしい多能性**: 同じ事前訓練モデルが、回帰・分類の予測に加えて、
  - **密度推定**（同時分布を $p(\mathbf{x},y)=\prod_j p(x_j\mid\mathbf{x}_{<j})\cdot p(y\mid\mathbf{x})$ と因数分解）→ 異常検知
  - **合成データ生成**（各特徴量を逐次サンプリング）→ データ拡張・プライバシー保護共有
  - **再利用可能な埋め込み**（最終層のターゲット列表現）→ 補完・クラスタリング
  - **ファインチューニング**（NN なので関連データセットへ適応。木ベースには不可）
  をこなす。

## 代表例

- **TabPFN v2**（Hollmann et al. 2025, Nature → [[sources/2025-tabpfn-v2]]）: 「表形式基盤モデル」を明示的に打ち出した代表例。最大 10,000 サンプル・500 特徴量で GBDT を上回り、上記の生成・密度推定・埋め込み・FT を備える。前身は [[sources/2022-tabpfn]]（TabPFN v1, 予測のみ）、理論的土台は [[sources/2021-transformers-can-do-bayesian-inference]]。
- **TabPFN-2.5**（Prior Labs 2025 → [[sources/2025-tabpfn-2-5]]）: 上記を **50,000 サンプル・2,000 特徴量**（セル 20×）へスケールし、業界標準ベンチ **TabArena で首位**。TFM が GBDT/AutoML を明確に凌ぐ段階を示した。下記の「蒸留」「因果推論」「エコシステム」を伴う。
- **TabPFN-3**（Prior Labs 2026 → [[sources/2026-tabpfn-3]]）: **100 万行**へスケールし、**テスト時計算（Thinking mode）** を TFM に導入（下記）。多クラス（最大160）・関係データ（RelBenchV1 SOTA）・テキスト表・時系列・因果へ適用拡大した現最前線。アーキテクチャは v1 流の行 ICL（3 段の行圧縮）へ回帰してスケールと両立。
- **TabICL（Inria 系）**（Qu et al. → [[sources/2025-tabicl]] v1 2025 / [[sources/2026-tabicl-v2]] v2 2026）: **TabPFN 一族とは別グループ（Inria）** の兄弟 TFM。「列ごと埋め込み → 行ごと相互作用で各行を 1 ベクトルに圧縮 → データセット ICL」の **3 段アーキテクチャ**。v1 は分類専用で最大 50 万サンプルへスケール。**v2（TabICLv2）は回帰（分位点分布）対応・QASSMax（長系列汎化）・新 prior・Muon でオープンウェイトの SOTA** となり、無調整で RealTabPFN-2.5 を上回る。この 3 段「行圧縮 ICL」設計を **TabPFN-3 が採用**（"TabICLv2 ベース"）。TFM が単一チームの産物でなく、Prior Labs 系（TabPFN）と Inria 系（TabICL）が相互に影響し合っていることを示す。

## 展開・蒸留（production への橋渡し）

TFM の弱点は、ICL 推論が重く（データセット全体を毎回処理）、レイテンシ・メモリ・規制制約のある本番に乗せにくいこと。TabPFN-2.5（[[sources/2025-tabpfn-2-5]]）は **蒸留エンジン** を導入し、訓練データを与えると TFM の予測を **コンパクトな MLP（as-MLP）や木アンサンブル（as-TreeEns）** に写し取る。蒸留先はデータセット固有・ICL なし・単一データ点入力で低レイテンシ、既存パイプラインに plug-and-play で載る。「強力だが重い基盤モデル」を「軽く速い実運用モデル」に変換する、TFM 実用化の鍵となる手法。

## 実データへの適応（継続事前訓練）

純粋に合成データで作る TFM に、**少量の厳選した実世界データで「継続事前訓練（continued pre-training）」** を足すと素直に性能が伸びる。Real-TabPFN（[[sources/2025-real-tabpfn]]）は、合成事前訓練済み TabPFNv2 を出発点に、OpenML/Kaggle の大きく厳選した 71 テーブルのみで追加学習（小さい学習率＋L2-SP 正則化で破滅的忘却を回避）し、OpenML AMLB の 29 データセットで TabPFNv2 を有意に上回った。「広いがノイジーで小規模なコーパス（CommonCrawl/GitTables）より、大きく厳選した少数の実テーブルが効く」のが知見。この発展形 **RealTabPFN-2.5** は、後続の TabPFN-3（[[sources/2026-tabpfn-3]]）や TabICLv2（[[sources/2026-tabicl-v2]]）が「無調整で超えるべき SOTA」として比較する基準になった。合成 prior（学習前）と実データ（継続事前訓練）の橋渡しは、蒸留（後段）と並ぶ TFM の主要な適応戦略である。

## テスト時計算（Thinking mode）

TabPFN-3（[[sources/2026-tabpfn-3]]）は、大規模言語モデルの「思考（reasoning）」に相当する **テスト時計算スケーリング** を TFM に持ち込んだ。推論時に追加の計算を費やして予測品質を上げる枠組みで、API 提供の **TabPFN-3-Plus（Thinking）** が実装。標準 TabArena で非 TabPFN モデルを 200+ Elo（最大データで 420 Elo）上回り、4 時間チューニングの AutoGluon 1.5 extreme を 1/10 時間で凌駕する。重要なのは、**LLM・実データ・インターネット検索・他モデルを一切使わず**、TabPFN 単体のテスト時計算だけでこれを達成する点。学習（事前訓練の合成データ規模）だけでなく**推論時の計算**も TFM の性能軸になったことを示す。

## 周辺への波及（基盤層としての TFM）

TFM は予測器を超えて、他タスクの**基盤層**になりつつある。TabPFN-3（[[sources/2026-tabpfn-3]]）では多くがコアモデルの直接の改善として SOTA に達する。
- **因果推論**: ベース予測性能の高さが [[structural-causal-model]] 系の CATE（条件付き平均処置効果）推定に転移。TabPFN を T/S/X-Learner として使うと専用手法を上回り、Do-PFN/CausalPFN 等は介入結果予測用に PFN を事前訓練する。
- **関係データ（RFM）**: TabPFN-3 ベースの TabPFN-REL が RelBenchV1（マルチテーブル関係 DB）で関係基盤モデルの新 SOTA。
- **時系列**: 合成のみ訓練の TabPFN-TS-3 が fev-bench で 2 位（実データ訓練勢のリーク問題と対照的に汚染ゼロ）。
- **テキスト表**: TabPFN-3-Plus が文字列列をネイティブにエンコードし TabSTAR で SOTA。
- **解釈可能性・グラフ・データストリーム・RL・[[bayesian-optimization|ベイズ最適化]]・マルチモーダル**: KV キャッシュ高速化で SHAP 計算が最大 120× 高速など、繰り返し順伝播に依存する拡張が直接加速。ベイズ最適化では、高コスト関数の代理に従来使われた GP（[[gaussian-process]]）の代わりに TFM を**サロゲート**として使える（高次元 BO の GIT-BO が TabPFNv2 を採用）。

## 限界・留意点

現状の TFM の主戦場は中〜大規模（TabPFN-3 で 〜100 万行・200 特徴量／〜2,000 特徴量。行数・特徴量数が共に巨大だと早期の特徴量圧縮がボトルネック）。リアルタイム推論やメモリ制約では蒸留先や従来手法（CatBoost 等）が有利な場合がある。Thinking mode やコア推論エンジンは非公開（API/エンタープライズのみ）でライセンスも商用は別途。データドリフト、さらなるスケール（数百万→）も今後の論点。

## 関連ページ

- [[prior-data-fitted-networks]] — 表形式基盤モデルを実現した枠組み
- [[in-context-learning]] — 重み更新なしの推論メカニズム
- [[bayesian-inference]] — 予測の理論的解釈（PPD 近似）
- [[structural-causal-model]] — 合成事前分布の中核／因果推論応用との接続
- [[sources/2025-tabpfn-v2]] — 代表例 TabPFN v2
- [[sources/2025-tabpfn-2-5]] — TabPFN-2.5（スケール・蒸留・エコシステム）
- [[sources/2026-tabpfn-3]] — 最新到達点 TabPFN-3（1M スケール・テスト時計算・関係/時系列/テキスト/因果）
- [[sources/2025-tabicl]] — 別系統（Inria）の兄弟 TFM v1。3 段アーキが TabPFN-3 に採用された
- [[sources/2026-tabicl-v2]] — Inria 系の最新（回帰対応・QASSMax・オープン）。TabPFN-3 が採用した本体
- [[sources/2025-real-tabpfn]] — 実データでの継続事前訓練（RealTabPFN-2.5 の前身）
- [[bayesian-optimization]] — TFM をサロゲートに使える下流応用（従来は GP）
- [[sources/2018-bayesian-optimization-tutorial]] — ベイズ最適化のチュートリアル（サロゲート＋獲得関数）
- [[questions/pfn-paper-and-gaussian-process]] — TFM が平均・分散（予測分布）を出力する仕組み
- [[questions/tabpfn-tabicl-versions-mechanism]] — 各バージョンの TabPFN/TabICL が GP をどう扱うか
