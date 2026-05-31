---
type: concept
aliases: [SCM, SCMs, Structural Causal Model, 構造的因果モデル]
tags: [causality, probabilistic-modeling, prior-design]
related:
  - "[[prior-data-fitted-networks]]"
  - "[[bayesian-inference]]"
sources:
  - "[[sources/2022-tabpfn]]"
  - "[[sources/2025-tabpfn-v2]]"
  - "[[sources/2025-tabpfn-2-5]]"
  - "[[sources/2026-tabpfn-3]]"
  - "[[sources/2025-tabicl]]"
  - "[[sources/2026-tabicl-v2]]"
updated: 2026-05-31
---

# Structural Causal Model（SCM, 構造的因果モデル）

## 一言で

**構造的因果モデル（SCM; Structural Causal Model, 変数間の因果関係を、有向非巡回グラフ（DAG）＋各ノードの生成式で表すモデル）**。各変数 $z_i$ は「親（直接の原因）の値とノイズ」から決まる:

$$
z_i = f_i\!\left(z_{\mathrm{PA}_{\mathcal{G}}(i)},\, \epsilon_i\right)
$$

ここで $\mathrm{PA}_{\mathcal{G}}(i)$ は DAG $\mathcal{G}$ におけるノード $i$ の親集合、$f_i$ は（非線形でもよい）決定論的関数、$\epsilon_i$ はノイズ変数。原因から結果へ向かう矢印で因果を表す。

## なぜ PFN の文脈で重要か

このリポジトリでは SCM は単独で深掘りする対象というより、**[[prior-data-fitted-networks]]（PFN）の "事前分布" を設計する道具** として登場する。表形式データは列どうしに因果的な依存（例: 年齢 → 収入）を持つことが多く、因果メカニズムは人間の推論でも強い事前分布として働く。そこで TabPFN（[[sources/2022-tabpfn]]）は、**「世の中の表データは何らかの SCM が生成している」** という仮定を事前分布に据えた。後継の TabPFN v2（[[sources/2025-tabpfn-v2]]）もこの SCM ベース事前分布を継承・発展させ、DAG を growing-network-with-redirection でサンプリングし、各辺の計算写像に小さな NN・カテゴリ離散化・決定木・ガウスノイズを組み込み、後処理で欠損・外れ値・量子化まで合成データに織り込んでいる。

## TabPFN での合成データ生成（事前分布としての使い方）

1. MLP 状の層構造グラフを作り、辺をランダムに**ドロップ**してランダムな DAG（＝SCM）にする。
2. 各辺に重み、各ノードにノイズ分布をサンプリングし、$f_i$ をアフィン写像＋活性化として具体化する（活性化は Tanh/LeakyReLU/ELU/Identity から）。
3. グラフのノードから特徴量ノード $z_X$ とターゲットノード $z_y$ を選ぶ。ノイズを流して伝播させ、$z_X, z_y$ の値を取り出せば 1 サンプル。
4. これを繰り返して 1 つの合成データセットができる。**特徴量とターゲットが因果グラフ経由で相関**し、ターゲットが特徴量の原因にも結果にもなり得る点が、単純な回帰データ生成と違う。

**単純性の事前分布**: ノード・パラメータが少ない（単純な）SCM ほど高確率になるよう設計され、オッカムの剃刀（簡潔な説明を選好）を体現する。これが TabPFN の「滑らかで過適合しにくい予測」の源泉。アブレーション（付録 B.4）では SCM 事前分布が BNN 単独より強いことが示された。

**世代を追うごとの拡張**: 後継モデルは SCM 事前分布を継続的に拡張している。TabPFN-3（[[sources/2026-tabpfn-3]]）では、グラフ生成の多様化・新しい combiner 機構・表現力あるカテゴリ・高周波振動・**空間活性化**・多クラス・**時間（離散時間 Dynamic SCM）**・**OOD/外挿 prior** を追加し、最終モデルは合成データ 8 兆トークン超で訓練された。事前分布の表現力がそのままモデルの適用範囲（時系列・空間・外挿など）を決める点が PFN 系の設計思想。別系統の TabICL（[[sources/2025-tabicl]]）も同様に SCM 事前分布を拡張し、活性化関数を多様化（ガウス過程由来のランダム関数を含む）したうえで、**木ベース SCM 事前分布**（各層を XGBoost で当てはめ）を 30% 混ぜて木モデルの帰納バイアスを合成データに注入している。その後継 TabICLv2（[[sources/2026-tabicl-v2]]）は prior をさらに大幅拡張し、**8 種のランダム関数**（MLP・Tree Ensemble〔CatBoost 風対称木〕・Discretize・GP〔多変量・滑らかさを理論証明〕・Linear・Quadratic・EM〔プラトー〕・Product）、木構造に限らない接続を作る **random Cauchy graph**、5 種のランダム行列、ランダム活性化、相関スカラーサンプリング、そして「単純な ExtraTrees が定数ベースラインを超えられない／$y$ が $x$ と独立な」データセットを除く **フィルタリング** を導入した。アブレーションでは prior の多様性が最大の性能要因であり、TabICLv2 のアーキテクチャは高い prior 多様性があって初めて汎化することが示された。

## "因果推論" そのものではない点に注意

TabPFN は SCM を事前分布に使うが、**因果グラフを推定したり介入・反事実を計算したりはしない**。Pearl の「因果のはしご」で言えば、第 1 段（連関）と第 2 段（介入）の間の "段 1.5"——SCM がデータをよく説明すると仮定したうえで、観測データ上の連関ベース予測を行うだけ。明示的なグラフ表現はスキップして [[bayesian-inference]] の事後予測分布（PPD）を直接近似する。

## 因果推論への応用（予測性能の転移）

一方で、TabPFN を**因果推論（causal inference）の道具として外側で使う**流れが育っている。TabPFN-2.5（[[sources/2025-tabpfn-2-5]]）は、非交絡設定での **CATE（条件付き平均処置効果）推定** を、TabPFN を傾向/結果モデルに据えた T/S/X-Learner（メタ学習器）として行い、RealCause ベンチで専用の因果フォレスト等を上回って上位を独占した。「表形式予測の強さが因果推論に転移する」ことを示す例で、Do-PFN/CausalPFN/CausalFM のように介入結果予測用に PFN を事前訓練する研究も現れている。SCM が *事前分布の素材* である本来の用法とは別に、因果という *応用領域* とも接続する点に注意。

## 関連ページ

- [[prior-data-fitted-networks]] — SCM を事前分布として使う枠組み
- [[bayesian-inference]] — SCM 上の事前分布から PPD を作る
- [[tabular-foundation-model]] — 因果推論を含む応用の基盤
- [[sources/2022-tabpfn]] — SCM 事前分布を導入した論文
- [[sources/2025-tabpfn-2-5]] — 因果推論（CATE/RealCause）応用
- [[sources/2026-tabpfn-3]] — SCM 事前分布の拡張（時間 DSCM・空間・OOD）と因果推論（QINI）
- [[sources/2025-tabicl]] — 木ベース SCM 事前分布＋活性化関数の多様化（別系統 TabICL）
