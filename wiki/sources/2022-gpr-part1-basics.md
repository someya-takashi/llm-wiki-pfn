---
type: source
source_path: "raw/articles/Introduction to Gaussian process regression, Part 1_ The basics.md"
source_kind: blog
title: "Introduction to Gaussian process regression, Part 1: The basics"
authors: [Kaixin Wang]
year: 2022
venue: Medium (Data Science at Microsoft)
tags: [gaussian-process, gpr, kernel, rbf, hyperparameter, gpflow, tutorial]
ingested: 2026-05-31
translation: "[[translations/2022-gpr-part1-basics]]"
---

# ガウス過程回帰入門 Part 1: 基礎（Medium 2022）

> 原典: [[translations/2022-gpr-part1-basics]] ・ `raw/articles/Introduction to Gaussian process regression, Part 1_ The basics.md`（medium.com/data-science-at-microsoft, 2022-10-04）
> 著者・媒体: Kaixin Wang（Data Science at Microsoft）/ Medium

## 一言まとめ

**ガウス過程回帰（GPR; [[gaussian-process]] を回帰に使う手法）の最も実践寄り・短尺の入門**。理論（GP 事前→事後、予測分布、ノイズの扱い）を式画像で手短に押さえたあと、**GPflow を使った実装視点**で「トイデータに線形・RBF・両者の和の3カーネルを当てはめると何が起こるか（線形＝未学習、RBF 単体＝ノイズに過学習、和＝内挿と外挿のバランス最良）」「長さスケール $l$ と分散 $\sigma^2$ をどう調整するか」を図で見せる。この wiki では [[gaussian-process]] の**4 本目のリファレンスで「実践/応用」の層**にあたり、既存の入門（直感の [[sources/2019-gp-not-for-dummies]]、体系的な [[sources/2020-gp-regression-tutorial]]）と上級理論（[[sources/2021-gp-models-intro]]）の間で、「実際にライブラリでカーネルを選び・調整する」感覚を補う。

## 背景と問題意識

GP/GPR の理論は他の入門が扱うが、「では実務でどうカーネルを選び、ハイパーパラメータを決めるのか」は初学者がつまずく所。本稿は **過学習（overfitting）と未学習（underfitting）のバランス**という実務の軸で、カーネル選択とハイパーパラメータ調整を具体的なトイデータと GPflow の例で示す。Part 1（基礎）で、続く Part 2 は実世界応用（コンクリート強度予測）。

## 内容（要点の再解釈）

1. **GPR の方法論（式画像）**: GP は関数上の分布を直接推論するノンパラメトリック・ベイズ手法。平均関数（通常 $m(x)=0$）と正定値の**カーネル（共分散）関数** $k$ で規定。観測値と予測の同時ガウスを $K$（訓練×訓練）/$K^*$（訓練×テスト）/$K^{**}$（テスト×テスト）に分割し、ガウスの性質から予測事後分布（平均・共分散）を閉形式で得る。ノイズ $y=f+\varepsilon$（i.i.d. 零平均ガウス）は**共分散行列の対角に加える**。
2. **実装の選択肢**: scikit-learn（簡単だが調整の選択肢少）/ GPyTorch（PyTorch、柔軟だが要前提知識）/ GPflow（TensorFlow、調整柔軟・構築素直）。本稿は **GPflow** を採用。
3. **カーネル選択（図2）**: **線形カーネル**（分散 $\sigma^2$ で規定。$\sigma^2=0$ で同次）と **RBF＝二乗指数カーネル**（長さスケール $l$＋分散 $\sigma^2$、無限回微分可能で滑らか）。トイデータで比較すると、**線形＝未学習（純粋に線形）、RBF 単体＝外挿が弱くノイズに過学習、線形＋RBF の和＝内挿と外挿のバランス最良**。各予測には 95% 信頼区間が付き、非線形カーネルの区間が真信号に触れる。
4. **ハイパーパラメータ最適化（図3）**: **長さスケール $l$ が滑らかさを支配**（小さい→過学習、大きい→滑らか。選定値 2.5）、**分散 $\sigma^2$ は広がりを制御**（影響は $l$ より小さい。選定値 0.1）。
5. **結論（図4）**: 最適化した線形＋RBF が真信号を正確に捉え、95% 信頼区間がノイズ度合いと整合。GPR の3利点＝**内挿・確率的（信頼区間）・多用途（カーネルの組み合わせ）**。

<figure>

![](../../raw/assets/2022-gpr-part1-basics/MKs9HBgmYeSbd8AA3n--Wg.png)

<figcaption>図2（再掲）: 線形（緑＝未学習）・RBF（赤＝過学習気味）・線形+RBF の和（紫＝バランス最良）の GPR 予測と 95% 信頼区間。［[[translations/2022-gpr-part1-basics]] より］</figcaption>
</figure>

## 限界・批判的視点

- **入門・短尺**で、理論の導出（条件付きガウスの式変形、RKHS など）には踏み込まない（そこは [[sources/2021-gp-models-intro]]）。数式は画像埋め込みのみ。
- 1 次元トイデータの例示が中心で、高次元・大規模での $O(n^3)$ スケール問題には触れない（[[gaussian-process]] / [[sources/2020-gp-regression-tutorial]] が補完）。
- カーネル選択は「線形・RBF・その和」に限定。周期カーネル等の比較は [[sources/2019-gp-not-for-dummies]] 側にある。

## 意義（なぜこの wiki に重要か）

1. **「カーネル選択＝事前分布の設計」を実務感覚で見せる**: PFN の性能が事前分布の設計に決定的に依存する（[[prior-data-fitted-networks]] / [[structural-causal-model]]）のと同様、GPR では**カーネルの選択と組み合わせ**が予測の質（過学習/未学習）を決める。本稿の「線形＋RBF の和が最良」という実例は、カーネル＝関数の事前知識という直感を実務レベルで補強する。
2. **較正された不確実性の実演**: 95% 信頼区間が真信号に触れる図は、PFN/TFM が掲げる「較正のよい予測分布」（[[bayesian-inference]] の PPD）の GPR 版の具体例。PFN はこの GPR の事後予測分布を Transformer の前向き計算で償却近似する。
3. **GP 四段リファレンスの「実践」層**: [[gaussian-process]] に 直感（2019）→ 体系的入門（2020）→ **実践/応用（本稿 2022）** → 上級理論（2021）が揃い、読者の目的（直感を掴む／式を追う／ライブラリで動かす／理論を厳密に）に応じて辿れる。

## 用語と略称

- **GPR** = Gaussian Process Regression（ガウス過程回帰）→ [[gaussian-process]]
- **カーネル（共分散）関数** = 2 点の関数値の相関を定める正定値関数。事前分布の性質を決める
- **線形カーネル** = $\sigma^2$ で規定される最も単純なカーネル（$\sigma^2=0$ で同次）
- **RBF / 二乗指数カーネル** = 長さスケール $l$＋分散 $\sigma^2$、無限回微分可能で滑らか
- **長さスケール $l$** = 相関が落ちる速さ＝滑らかさを支配（小→過学習、大→滑らか）
- **過学習 / 未学習（overfitting / underfitting）** = カーネル選択・ハイパラ調整のバランスの軸
- **GPflow / GPyTorch / scikit-learn** = GPR の代表的 Python 実装
- **PPD / 事後予測分布** = 観測を条件にした予測分布 → [[bayesian-inference]]

## 参照（原典内リンク）

- scikit-learn（Pedregosa et al. 2011）、Krasser のブログ "Gaussian processes"（2018）、GPyTorch（Gardner et al. 2018, arXiv:1809.11165）、GPflow（Matthews et al. 2017, JMLR）
- 続編: [[sources/2022-gpr-part2-concrete]]（同著者 Part 2: コンクリート強度予測への応用）

## 関連ページ

- [[gaussian-process]] — 本記事が解説する概念（この source は「実践/応用」リファレンス）
- [[sources/2019-gp-not-for-dummies]] — 最も直感寄りの入門
- [[sources/2020-gp-regression-tutorial]] — 体系的な入門（全式つき）
- [[sources/2021-gp-models-intro]] — 上級（RKHS・モデル誤差境界・GP 動的モデル）
- [[sources/2022-gpr-part2-concrete]] — 本記事の続編（実データ応用＋SHAP 解釈）
- [[bayesian-inference]] — GPR の事後予測分布／較正された不確実性
- [[structural-causal-model]] — カーネル＝事前知識の設計（合成 prior との接続）
- [[translations/2022-gpr-part1-basics]] — 本文の翻訳
