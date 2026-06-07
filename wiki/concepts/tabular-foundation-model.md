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
  - "[[sources/2025-mitra]]"
  - "[[sources/2025-git-bo]]"
  - "[[sources/2025-causalpfn]]"
  - "[[sources/2025-do-pfn]]"
  - "[[sources/2025-chronos-2]]"
  - "[[sources/2025-nanotabpfn]]"
  - "[[sources/2026-shappfn]]"
updated: 2026-06-06
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

## 教育・軽量化（アクセシビリティの向上）

TFM は強力な一方、公式実装（例: TabPFN v2 は 10,000 行超）が複雑で理解・改変が難しく、事前訓練にも大規模計算資源を要する。**nanoTabPFN（[[sources/2025-nanotabpfn]]、Pfefferle et al. 2025 / Freiburg・Prior Labs）** は、TabPFN v2 のコアアーキテクチャを 500 行未満に再実装した教育・研究向けの軽量版で、「**nanoGPT（Karpathy）の表形式基盤モデル版**」。シングル GPU 1分の事前訓練（TabPFN v2 比 160,000 倍速）で従来 ML ベースライン（kNN・決定木・ランダムフォレスト）に並ぶ性能を実現し、new prior / アーキ変更 / 訓練手順のアイデアを分単位で試せる出発点として TFM 研究のプロトタイピングを高速化する。

## 展開・蒸留（production への橋渡し）

TFM の弱点は、ICL 推論が重く（データセット全体を毎回処理）、レイテンシ・メモリ・規制制約のある本番に乗せにくいこと。TabPFN-2.5（[[sources/2025-tabpfn-2-5]]）は **蒸留エンジン** を導入し、訓練データを与えると TFM の予測を **コンパクトな MLP（as-MLP）や木アンサンブル（as-TreeEns）** に写し取る。蒸留先はデータセット固有・ICL なし・単一データ点入力で低レイテンシ、既存パイプラインに plug-and-play で載る。「強力だが重い基盤モデル」を「軽く速い実運用モデル」に変換する、TFM 実用化の鍵となる手法。

## 実データへの適応（継続事前訓練）

純粋に合成データで作る TFM に、**少量の厳選した実世界データで「継続事前訓練（continued pre-training）」** を足すと素直に性能が伸びる。Real-TabPFN（[[sources/2025-real-tabpfn]]）は、合成事前訓練済み TabPFNv2 を出発点に、OpenML/Kaggle の大きく厳選した 71 テーブルのみで追加学習（小さい学習率＋L2-SP 正則化で破滅的忘却を回避）し、OpenML AMLB の 29 データセットで TabPFNv2 を有意に上回った。「広いがノイジーで小規模なコーパス（CommonCrawl/GitTables）より、大きく厳選した少数の実テーブルが効く」のが知見。この発展形 **RealTabPFN-2.5** は、後続の TabPFN-3（[[sources/2026-tabpfn-3]]）や TabICLv2（[[sources/2026-tabicl-v2]]）が「無調整で超えるべき SOTA」として比較する基準になった。合成 prior（学習前）と実データ（継続事前訓練）の橋渡しは、蒸留（後段）と並ぶ TFM の主要な適応戦略である。

## テスト時計算（Thinking mode）

TabPFN-3（[[sources/2026-tabpfn-3]]）は、大規模言語モデルの「思考（reasoning）」に相当する **テスト時計算スケーリング** を TFM に持ち込んだ。推論時に追加の計算を費やして予測品質を上げる枠組みで、API 提供の **TabPFN-3-Plus（Thinking）** が実装。標準 TabArena で非 TabPFN モデルを 200+ Elo（最大データで 420 Elo）上回り、4 時間チューニングの AutoGluon 1.5 extreme を 1/10 時間で凌駕する。重要なのは、**LLM・実データ・インターネット検索・他モデルを一切使わず**、TabPFN 単体のテスト時計算だけでこれを達成する点。学習（事前訓練の合成データ規模）だけでなく**推論時の計算**も TFM の性能軸になったことを示す。

## 周辺への波及（基盤層としての TFM）

TFM は予測器を超えて、他タスクの**基盤層**になりつつある。TabPFN-3（[[sources/2026-tabpfn-3]]）では多くがコアモデルの直接の改善として SOTA に達する。
- **因果推論**: ベース予測性能の高さが [[structural-causal-model]] 系の CATE（条件付き平均処置効果）推定に転移。TabPFN を T/S/X-Learner として使うと専用手法を上回る。さらに **CausalPFN（[[sources/2025-causalpfn]]）** は因果効果推定を**専用に事前訓練**した TFM で、注目すべきはその**事前分布が TFM の素材を流用**している点——OpenML の実データ表＋**TabPFN v1 のランダム NN 合成生成器**から、無視可能性（ignorability）を満たす因果データセットを量産する。訓練も TFM 流の2段構成（[68] TabDPT を模した予測フェーズ→因果フェーズ）。**Do-PFN（[[sources/2025-do-pfn]]）** は同じく **TabPFN v2 とほぼ同一のアーキ**（730 万パラメータ＋先頭列＝処置の指示子）を流用し、SCM の prior に介入 $do(t)$ を加えて条件付き介入分布を予測する。表形式予測の基盤（合成 prior・ICL・ヒストグラム回帰ヘッド・TabPFN アーキ）がそのまま因果効果推定に転用できることを、潜在結果側（CausalPFN）と do 計算側（Do-PFN）の両方が示し、TFM が「構造化データ推論の汎用エンジン」へ向かう流れを補強する。
- **関係データ（RFM）**: TabPFN-3 ベースの TabPFN-REL が RelBenchV1（マルチテーブル関係 DB）で関係基盤モデルの新 SOTA。
- **時系列**: 合成のみ訓練の TabPFN-TS-3 が fev-bench で 2 位（実データ訓練勢のリーク問題と対照的に汚染ゼロ）。なお同じ fev-bench は**時系列ネイティブの基盤モデル**の主戦場でもあり、Amazon の **Chronos-2（[[sources/2025-chronos-2]]）** がそこで首位を取る。Chronos-2 は表形式 TFM ではないが、**group attention＝時系列版の [[in-context-learning|ICL]]**（関連系列・多変量・共変量をグループとして文脈共有）と**合成データのみで汎用能力**という、TFM と同じ設計原理に立つ「いとこ」。共変量つきタスクでは TabPFN を時系列に適応した **TabPFN-TS** が Chronos-2 に次ぐ2位を占め、表形式 TFM 一族と時系列 FM 一族が fev-bench で直接競合・収斂しつつあることを示す。
- **テキスト表**: TabPFN-3-Plus が文字列列をネイティブにエンコードし TabSTAR で SOTA。
- **解釈可能性・グラフ・データストリーム・RL・[[bayesian-optimization|ベイズ最適化]]・マルチモーダル**: KV キャッシュ高速化で SHAP 計算が最大 120× 高速など、繰り返し順伝播に依存する拡張が直接加速。ベイズ最適化では、高コスト関数の代理に従来使われた GP（[[gaussian-process]]）の代わりに TFM を**サロゲート**として使える（低次元は PFNs4BO＝[[sources/2023-pfns4bo]]、**高次元 100〜500 次元は GIT-BO**＝[[sources/2025-git-bo]] が TabPFN v2 を採用し、順伝播の勾配で能動部分空間を毎反復同定して GP ベース手法を 2 桁速く凌駕）。
- **説明可能性の内蔵（ShapPFN）**: 上の「SHAP を高速化する」よりさらに踏み込み、**説明をアーキテクチャに焼き込む**方向もある。**ShapPFN（[[sources/2026-shappfn]]）** は [[sources/2025-nanotabpfn|nanoTabPFN]] をベースに、予測を「ベース項＋特徴量ごとの加法的寄与」に分解する2デコーダーヘッドを追加し、Shapley 一致性損失（ViaSHAP 流）で訓練することで、**予測と Shapley 値説明を1回の順伝播で同時に**出力する。事後の KernelSHAP（約 610 秒）を 0.06 秒（1000 倍超速）で同品質（一致 R²=0.96）に置換し、「リアルタイム説明」を可能にする。TFM が「説明の計算すら事前訓練に償却する」段階を示す。

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
- [[sources/2025-mitra]] — 合成 prior の設計原理（性能・多様性・独自性）と SCM＋木 prior の混合（Amazon）
- [[bayesian-optimization]] — TFM をサロゲートに使える下流応用（従来は GP）
- [[sources/2018-bayesian-optimization-tutorial]] — ベイズ最適化のチュートリアル（サロゲート＋獲得関数）
- [[sources/2025-git-bo]] — TabPFNv2 を高次元 BO サロゲートに使った代表例（GIT-BO, MIT）
- [[sources/2025-causalpfn]] — TFM の prior（TabPFN v1 生成器＋OpenML）を流用して因果効果推定を専用事前訓練
- [[sources/2025-do-pfn]] — TabPFN v2 アーキを流用し、介入つき SCM prior で条件付き介入分布を予測
- [[sources/2025-chronos-2]] — 時系列ネイティブの基盤モデル（fev-bench 首位）。group attention＝時系列版 ICL・合成データ事前訓練で TFM と収斂、TabPFN-TS と競合
- [[sources/2025-nanotabpfn]] — TabPFN v2 の教育的軽量再実装（500行・1分訓練・nanoGPT 流。TFM の教育・民主化）
- [[sources/2026-shappfn]] — Shapley 値説明をアーキに内蔵した TFM（nanoTabPFN ベース・予測と説明を1順伝播で・KernelSHAP 1000倍速）
- [[questions/pfn-paper-and-gaussian-process]] — TFM が平均・分散（予測分布）を出力する仕組み
- [[questions/tabpfn-tabicl-versions-mechanism]] — 各バージョンの TabPFN/TabICL が GP をどう扱うか
