---
type: source
source_path: raw/articles/nanoTabPFN_ A Lightweight and Educational Reimplementation of TabPFN.md
source_kind: article
title: "nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN"
authors: [Alexander Pfefferle, Johannes Hog, Lennart Purucker, Frank Hutter]
year: 2025
venue: arXiv 2511.03634
ingested: 2026-06-06
tags: [pfn, tabpfn, tabular-foundation-model, educational, lightweight, reimplementation]
translation: "[[translations/2025-nanotabpfn]]"
---

# nanoTabPFN: TabPFN の軽量かつ教育的な再実装

> 原典: [[translations/2025-nanotabpfn]] ・ `raw/articles/nanoTabPFN_ ... .md`（ar5iv, arXiv:2511.03634）
> 著者・年・所属: Alexander Pfefferle, Johannes Hog, Lennart Purucker, Frank Hutter（ELLIS Tübingen / University of Freiburg / Prior Labs, 2025）
> コード: https://github.com/automl/nanoTabPFN

## 一言まとめ

**TabPFN v2 のための「nanoGPT」——500 行未満のコードで TabPFN v2 のコアアーキテクチャを再実装し、シングル GPU 1分の事前訓練で従来 ML ベースラインに並ぶ性能を実証した教育・研究向けの軽量実装。** 公式 TabPFN v2 の 10,000 行超のコードと比べ、研究のプロトタイピングや教育に参入する障壁を劇的に下げることを目指す。

## 背景と問題意識

**TabPFN v2（[[sources/2025-tabpfn-v2]]）は、[[prior-data-fitted-networks|PFN（Prior-Data Fitted Networks）]] を表形式データ予測の実用的な基盤モデルに成長させた里程標となる論文**だが、その公式実装は 10,000 行を超える Python コードで構成されており、次の問題がある。

1. **理解困難**: アーキテクチャのドキュメントが不足し、複雑なパイプラインの中で核心部分を見つけにくい
2. **実験困難**: 事前訓練に8枚の GPU を2週間かけて実施（TabPFN v2 の規模）、研究者・学生には再現が難しい
3. **改変困難**: 新しい prior や訓練手順を試す際の出発点として使いにくい

**参考: minGPT / nanoGPT（Karpathy）との対応**。大規模言語モデルでは、GPT-2 を数百行のコードで再実装した minGPT（2020）・nanoGPT（2022）が「LLM を理解したい人の教材」として広く使われた。nanoTabPFN はその表形式基盤モデル版——TabPFN v2 のアーキテクチャの骨子を抽出し、教育・プロトタイピング向けに整理したもの。

## 提案手法 / 主張

**nanoTabPFN のコアアーキテクチャ**（4構成要素）:

1. **FeatureEncoder**: 訓練セットの平均・標準偏差で正規化＋外れ値クリッピング→線形層で高次元埋め込みへ。位置埋め込みなし（データポイントと特徴量に対する**置換不変性**を保持するため）。
2. **TargetEncoder**: `y_train` を平均値で `y_test` 分まで埋めて連結→線形層で埋め込み（推論時の未知ターゲット初期化）。
3. **TransformerEncoderLayer（複数）**: **双方向注意（bi-attention）** を繰り返す。
   - *Feature attention*（特徴量間）→ *Datapoint attention*（データポイント間）の順に交互適用。
   - データポイント間注意では「訓練データは自分だけを見る、テストデータは訓練データだけを見る」という非対称マスクにより、テストデータ同士が互いに影響せず**置換不変**になる（[[in-context-learning|文脈内学習]] のキャッシュ的な性質を保つ）。
   - 各ブロックの後にスキップ接続・LayerNorm・セルごとの 2 層 MLP（スキップ接続＋LayerNorm）。
4. **Decoder**: `y_test` の埋め込みを受け取り 2 層 MLP で分類ロジットへ写像。

<figure>

![](../../raw/assets/2025-nanotabpfn/architecture.png)

<figcaption>図1（出典: 本論文）: nanoTabPFN のアーキテクチャ。FeatureEncoder・TargetEncoder がセルごとの埋め込みを作り、TransformerEncoderStack（特徴量間とデータポイント間の bi-attention）が埋め込みを洗練し、Decoder が予測を出力する。TabPFN v2 Fig.1 を元に簡略化。</figcaption>
</figure>

**TabPFN v2 から意図的に省いたもの**（簡潔さ優先）:
- 隣接特徴量ペアを組み合わせた特徴量埋め込み（コードを複雑にし置換不変性を損なう）
- データポイント識別用の行ハッシュ列
- カテゴリ特徴量・欠損値の内部処理
- 回帰・アンサンブル

**訓練の仕組み**: スケジューラーなし AdamW（重み減衰なし）で、ディスクから HDF5 形式の**事前生成済み合成データ**を読み込んで訓練。「生成のたびに prior を計算する」のではなくダンプを読むため、訓練ループが高速に回せる。ハイパーパラメータは 200 設定のランダムサーチで決定。**訓練データ生成には TabICL（[[sources/2025-tabicl]]）の prior 実装を流用**。

## 実験結果と知見

| 設定 | 値 |
|---|---|
| モデル規模 | 3 層・4ヘッド・埋め込み 96 次元・MLP 192 次元 |
| 訓練データ | 80,000 件の合成データセット（各 150 データ×5特徴×2クラス） |
| GPU | RTX 2080 Ti 1枚 |
| 訓練時間 | **60秒**（TabPFN v2 の 8 GPU×2週間の **160,000 倍速**） |
| 評価 | TabArena の二値分類データセット（欠損値なし・最大10特徴・200データにサブサンプル）・5分割 CV×20回繰り返し |

60 秒の事前訓練で nanoTabPFN は kNN・決定木・ランダムフォレスト（scikit-learn デフォルト）をすべて上回る ROC AUC を達成。さらに数秒後には「前処理なし・アンサンブルなし」の TabPFN v2 設定も上回った。

<figure>

![](../../raw/assets/2025-nanotabpfn/roc_auc.png)

<figcaption>図4（出典: 本論文）: 訓練時間 vs 平均 ROC AUC。60秒で従来 ML ベースライン全て（kNN・決定木・ランダムフォレスト）を上回り、数秒後に「前処理なし TabPFN v2」も超える。</figcaption>
</figure>

**重要な注意**: これはあくまで**小規模・教育向けの実験**（150データ・5特徴・2クラスに限定）であり、フルスケールの TabPFN v2 と同等の汎用性を主張するものではない。

## 限界・批判的視点

- **タスクの制限**: 分類のみ、欠損値なし、最大10特徴量、最大200データという制限下での評価。
- **prior が単純**: 150×5×2という固定形式の合成データで訓練——現実のデータの多様性を捉えられていない。**TabICL の prior 実装を流用**しているが、簡略化された prior の独自実装は含まない（将来課題）。
- **「匹敵する性能」の意味**: 従来 ML ベースラインに匹敵するが、フル TabPFN v2 よりは大幅に性能が低い。「教育目的での最初の一歩」という位置づけ。
- **欠如機能**: アンサンブル・前処理・回帰・カテゴリ特徴量・欠損値処理なし。

## 研究の意義

nanoTabPFN が狙うのは**性能の最先端ではなく、TabPFN の「仕組みを理解できる最小の実装」を提供すること**。これは [[tabular-foundation-model|表形式基盤モデル]] の教育・民主化という文脈で重要である。  
同時に、研究の観点からは「**新しい prior・アーキテクチャ変更・訓練手順のアイデアを 1 分で試せる出発点**」として機能する。大規模モデルの full re-training なしにアブレーション研究が回せることで、PFN 研究のサイクルを高速化しうる。  
なお、著者陣（Hutter・Purucker）は TabPFN v2・TabPFN-2.5 の著者でもあり、これは公式グループからのコミュニティ向け贈り物といえる。

## 用語と略称

- **PFN** = Prior-Data Fitted Network（合成データで一度訓練し推論は文脈内学習）→ [[prior-data-fitted-networks]]
- **TFM** = Tabular Foundation Model（表形式基盤モデル）→ [[tabular-foundation-model]]
- **bi-attention（双方向注意）** = 特徴量間の注意とデータポイント間の注意を交互に適用する、TabPFN 特有の注意機構
- **置換不変性（permutation invariance）** = データポイントや特徴量の順序を入れ替えても出力が変わらない性質
- **HDF5** = Hierarchical Data Format version 5。大規模数値データを効率的に保存するファイル形式
- **ROC AUC** = 二値分類の性能指標（受信者操作特性曲線の下面積）。高いほど良い
- **minGPT / nanoGPT** = Andrej Karpathy が GPT-2 を教育用に再実装した小規模コード（nanoTabPFN の着想源）
- **スケジューラーなし AdamW（scheduler-free AdamW）** = 学習率スケジューラーを持たない改良版 AdamW 最適化器

## 関連ページ

- [[prior-data-fitted-networks]] — PFN の中核概念。nanoTabPFN は TabPFN v2 の教育的再実装
- [[tabular-foundation-model]] — 表形式基盤モデルの概念ページ
- [[sources/2025-tabpfn-v2]] — nanoTabPFN が再実装するモデル（TabPFN v2）
- [[sources/2022-tabpfn]] — TabPFN v1（nanoTabPFN の究極の源流）
- [[sources/2025-tabicl]] — 訓練データ生成に prior 実装を流用
- [[in-context-learning]] — nanoTabPFN のコア（表形式データを文脈として与え重み更新なしに予測）
