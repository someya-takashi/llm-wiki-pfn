---
type: source
source_path: raw/articles/Real-TabPFN_ Improving Tabular Foundation Models via Continued Pre-training With Real-World Data.md
source_kind: paper
title: "Real-TabPFN: Improving Tabular Foundation Models via Continued Pre-training With Real-World Data"
authors: [Anurag Garg, Muhammad Ali, Noah Hollmann, Lennart Purucker, Samuel Müller, Frank Hutter]
year: 2025
venue: arXiv 2507.03971
tags: [tabular-foundation-model, continued-pretraining, real-data, tabpfn, in-context-learning]
ingested: 2026-05-31
translation: "[[translations/2025-real-tabpfn]]"
---

# Real-TabPFN: 実世界データでの継続事前訓練（arXiv 2025）

> 原典: [[translations/2025-real-tabpfn]] ・ `raw/articles/Real-TabPFN_ Improving Tabular Foundation Models via Continued Pre-training With Real-World Data.md`（arXiv:2507.03971）
> 著者・年: Anurag Garg, Muhammad Ali, Noah Hollmann, Lennart Purucker, Samuel Müller, Frank Hutter（Freiburg / Prior Labs）/ 2025

## 一言まとめ

合成データのみで事前訓練した [[sources/2025-tabpfn-v2]]（TabPFNv2）を、**厳選した実世界データで「継続事前訓練（continued pre-training）」する**だけで性能が大きく伸びることを示した短い論文。得られた **Real-TabPFN** は OpenML AutoML Benchmark の 29 データセットで TabPFNv2 を有意に上回る（平均正規化 ROC-AUC 0.954→0.976）。後続の **RealTabPFN-2.5**（TabPFN-2.5/TabICLv2 の比較対象に頻出する「実データ FT 済み SOTA」）の直接の前身。鍵は「広いがノイジーなコーパス（CommonCrawl/GitTables）より、大きく厳選した少数の実テーブル（OpenML/Kaggle 71 件）の方が効く」という知見。

<figure>

![](../../raw/assets/2025-real-tabpfn/x1.png)

<figcaption>図1（再掲）: OpenML AutoML Benchmark の 29 データセットでの TabPFN（default）対 Real-TabPFN のデータセットごと正規化 ROC 比較。Real-TabPFN が有意に上回る（ウィルコクソン検定）。［[[translations/2025-real-tabpfn]] 図1 より］</figcaption>
</figure>

## 背景と問題意識

TabPFN 系の TFM（[[tabular-foundation-model]]）は **完全合成データ**（[[structural-causal-model]] 由来の 1 億超の合成テーブル）のみで事前訓練され、小規模データで強い。しかし合成だけで作られた TabPFNv2 はさらなる改善が難しく、よく調整した木アンサンブルに負けるデータセットも残る。一方、言語モデルでは「continued pre-training（既存の事前訓練済みモデルを別ドメインのデータで追加学習）」が顕著に成功している。本論文は **この continued pre-training を表形式 TFM に持ち込み、合成と実データの gap を橋渡しできるか** を問う。TabDPT（実データのみで事前訓練）と純合成（TabPFN）の中間を狙う。

## 提案手法 / 主張

**2 段アプローチ（continued pre-training）**
- 段 1: 既存の合成事前訓練済み TabPFNv2 チェックポイントを出発点にする。
- 段 2: **厳選した実世界テーブルのみ**で事前訓練を継続する（合成と実を同時に混ぜる "mixed" 訓練とは対照的。強い合成ベースの上に直接構築できるため 2 段を採用）。

**データ設計（本論文の肝）**
- 評価は OpenML AutoML Benchmark の 29 データセット（≤10k サンプル・≤500 特徴量）。
- 継続事前訓練データは **OpenML/Kaggle から手作業で厳選した 71 件の大きいデータセット**。前処理は最小（OrdinalEncoder、10 クラス超は上位 9＋残りを 1 クラスに統合）。
- **データ汚染対策を徹底**（評価データは全て <10k なので訓練は >10k のみ採用、ID/名前/形状/行列ハッシュ照合＋メタデータ手動検査）。これが少数厳選にこだわる理由。

**訓練の工夫（破滅的忘却の回避）**
- 元の TabPFNv2 アーキテクチャを保持、**非常に小さい学習率 $3\times10^{-7}$**（AdamW＋warmup＋cosine）。
- **L2-SP 正則化**: 初期重み $\mathbf{w}^0$ からの逸脱を罰する $\Omega(\mathbf{w})=\frac{\alpha}{2}\lVert\mathbf{w}-\mathbf{w}^0\rVert_2^2$（$\alpha=0.003$）を交差エントロピーに加え、破滅的忘却（catastrophic forgetting）を緩和。
- バッチサイズ 1（＝1 データセット。可変特徴量数をパディングなしで扱え、文脈を最大 20,000 サンプルまで最大化）。60% 文脈・40% クエリ。RTX 2080 Ti 1 枚、20,000 ステップ、最大 40 万セル。

## 実験結果と知見

- **対 TabPFNv2**: 29 データセットで有意に勝ち（図1, ウィルコクソン）。平均正規化 ROC-AUC を **0.954 → 0.976** に改善し、4 時間調整した全ベースラインも平均で上回る（図3）。
- **文脈サイズの効果**: 継続事前訓練のデータセットサイズを 2,048→20,000 に増やすほど下流精度が上がる（図4）。「より多くの文脈」が効く。
- **データ源の比較（重要な知見）**: OpenML のみ +0.019、Kaggle のみ +0.015、両者結合が最強 +0.022（補完的）。一方 **CommonCrawl**（平均 ~100 行・7 特徴量と小さい）は**性能を下げ**、**GitTables**（~1000 行）は改善、**OpenML/Kaggle**（1 万〜10 万行）が最大改善（図5,6）。→ 「広いがノイジー・小規模」より「大きく厳選」が効く。

## 限界・批判的視点

- **評価が狭い**: 29 の OpenML 分類データセット（≤10k）のみ。回帰や大規模は未評価。
- **手作業の厳選 71 件**に依存し、汚染回避のため CommonCrawl/GitTables と併用しなかった（スケール可能性は別途）。
- **改善幅は小さめ**（+0.022 程度）だが、医療・信用など表データ支配領域では意味があると主張。
- **継続事前訓練の理論的理解は浅い**（経験的知見が中心）。
- 短いワークショップ規模の論文で、§2（関連研究）が ar5iv クリップに含まれていない。

## 意義（なぜ重要か）

「**完全合成 prior の TFM に、少量の厳選実データで継続事前訓練を足すと素直に強くなる**」ことを示し、合成 vs 実データの二項対立に橋を架けた。これは後続の **RealTabPFN-2.5**（[[sources/2025-tabpfn-2-5]] の実データ FT 版で、TabArena/TALENT で TabPFN-3・TabICLv2 の主要比較対象＝「無調整で超えるべき SOTA」）の直接の前身であり、TabICLv2（[[sources/2026-tabicl-v2]]）が比較した RealTabPFN-2.5 もこの系譜。[[tabular-foundation-model]] の「適応戦略」（ファインチューニング/継続事前訓練）の代表例。

## 用語と略称

- **Real-TabPFN** = 実データで継続事前訓練した TabPFNv2 → [[prior-data-fitted-networks]], [[tabular-foundation-model]]
- **continued pre-training（継続事前訓練）** = 既存の事前訓練済みモデルを別データで追加学習する適応戦略
- **L2-SP（L2-Starting-Point）正則化** = 初期重みからの距離を罰し破滅的忘却を防ぐ正則化
- **catastrophic forgetting（破滅的忘却）** = 追加学習で既習知識を失う現象
- **TabDPT** = 実データのみで事前訓練する対照的な TFM
- **mixed 訓練** = 合成と実データを同時に与える訓練（本論文の 2 段とは対照的）
- **CommonCrawl / GitTables / OpenML / Kaggle** = 比較した表データ源
- **AMLB（OpenML AutoML Benchmark）** = 評価ベンチ（29 分類データセット）
- **ROC-AUC** = 評価指標

## 関連ページ

- [[tabular-foundation-model]] — 適応戦略（継続事前訓練）の代表例
- [[prior-data-fitted-networks]] — ベースの TabPFNv2 が属する枠組み
- [[sources/2025-tabpfn-v2]] — 継続事前訓練の出発点（TabPFNv2）
- [[sources/2025-tabpfn-2-5]] / [[sources/2026-tabpfn-3]] / [[sources/2026-tabicl-v2]] — RealTabPFN-2.5（本手法の発展形）を比較対象にする後続
- [[structural-causal-model]] — 段 1 の合成 prior の中核
- [[translations/2025-real-tabpfn]] — 本文 §1〜6 の翻訳
