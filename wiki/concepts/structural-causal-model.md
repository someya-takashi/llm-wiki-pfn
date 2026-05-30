---
type: concept
aliases: [SCM, SCMs, Structural Causal Model, 構造的因果モデル]
tags: [causality, probabilistic-modeling, prior-design]
related: [[prior-data-fitted-networks]], [[bayesian-inference]]
sources: [[sources/2022-tabpfn]]
updated: 2026-05-30
---

# Structural Causal Model（SCM, 構造的因果モデル）

## 一言で

**構造的因果モデル（SCM; Structural Causal Model, 変数間の因果関係を、有向非巡回グラフ（DAG）＋各ノードの生成式で表すモデル）**。各変数 $z_i$ は「親（直接の原因）の値とノイズ」から決まる:

$$
z_i = f_i\!\left(z_{\mathrm{PA}_{\mathcal{G}}(i)},\, \epsilon_i\right)
$$

ここで $\mathrm{PA}_{\mathcal{G}}(i)$ は DAG $\mathcal{G}$ におけるノード $i$ の親集合、$f_i$ は（非線形でもよい）決定論的関数、$\epsilon_i$ はノイズ変数。原因から結果へ向かう矢印で因果を表す。

## なぜ PFN の文脈で重要か

このリポジトリでは SCM は単独で深掘りする対象というより、**[[prior-data-fitted-networks]]（PFN）の "事前分布" を設計する道具** として登場する。表形式データは列どうしに因果的な依存（例: 年齢 → 収入）を持つことが多く、因果メカニズムは人間の推論でも強い事前分布として働く。そこで TabPFN（[[sources/2022-tabpfn]]）は、**「世の中の表データは何らかの SCM が生成している」** という仮定を事前分布に据えた。

## TabPFN での合成データ生成（事前分布としての使い方）

1. MLP 状の層構造グラフを作り、辺をランダムに**ドロップ**してランダムな DAG（＝SCM）にする。
2. 各辺に重み、各ノードにノイズ分布をサンプリングし、$f_i$ をアフィン写像＋活性化として具体化する（活性化は Tanh/LeakyReLU/ELU/Identity から）。
3. グラフのノードから特徴量ノード $z_X$ とターゲットノード $z_y$ を選ぶ。ノイズを流して伝播させ、$z_X, z_y$ の値を取り出せば 1 サンプル。
4. これを繰り返して 1 つの合成データセットができる。**特徴量とターゲットが因果グラフ経由で相関**し、ターゲットが特徴量の原因にも結果にもなり得る点が、単純な回帰データ生成と違う。

**単純性の事前分布**: ノード・パラメータが少ない（単純な）SCM ほど高確率になるよう設計され、オッカムの剃刀（簡潔な説明を選好）を体現する。これが TabPFN の「滑らかで過適合しにくい予測」の源泉。アブレーション（付録 B.4）では SCM 事前分布が BNN 単独より強いことが示された。

## "因果推論" そのものではない点に注意

TabPFN は SCM を事前分布に使うが、**因果グラフを推定したり介入・反事実を計算したりはしない**。Pearl の「因果のはしご」で言えば、第 1 段（連関）と第 2 段（介入）の間の "段 1.5"——SCM がデータをよく説明すると仮定したうえで、観測データ上の連関ベース予測を行うだけ。明示的なグラフ表現はスキップして [[bayesian-inference]] の事後予測分布（PPD）を直接近似する。

## 関連ページ

- [[prior-data-fitted-networks]] — SCM を事前分布として使う枠組み
- [[bayesian-inference]] — SCM 上の事前分布から PPD を作る
- [[sources/2022-tabpfn]] — SCM 事前分布を導入した論文
