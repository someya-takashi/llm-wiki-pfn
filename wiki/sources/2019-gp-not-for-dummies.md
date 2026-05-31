---
type: source
source_path: raw/articles/Gaussian Processes, not quite for dummies.md
source_kind: blog
title: "Gaussian Processes, not quite for dummies"
authors: [Yuge Shi]
year: 2019
venue: The Gradient
tags: [gaussian-process, kernel, bayesian-inference, regression, tutorial, visual-intuition]
ingested: 2026-05-31
translation: "[[translations/2019-gp-not-for-dummies]]"
---

# ガウス過程、完全な初心者向けとまではいかないが（The Gradient 2019）

> 原典: [[translations/2019-gp-not-for-dummies]] ・ `raw/articles/Gaussian Processes, not quite for dummies.md`（thegradient.pub/gaussian-process-not-quite-for-dummies/）
> 著者・媒体・年: Yuge Shi（University of Oxford, Torr Vision Group）/ The Gradient / 2019

## 一言まとめ

**ガウス過程（[[gaussian-process]]）の最も直感寄り・図中心の入門ブログ**。Richard Turner 博士の講演ノートをもとに、「**高次元ガウスからのサンプリングを index プロット（横軸＝変数の番号、縦軸＝その値）として描く**」という独特の可視化で、多変量ガウスの条件付け → 20 次元ガウスを数点で条件付けると**そのまま非線形回帰に見える** → RBF カーネルで実数 $x$ に拡張すると無限次元ガウス＝GP、という流れを絵で腑に落とさせる。この wiki では [[gaussian-process]] の**3 本目のリファレンスで「最初の一冊」**にあたり、既存の [[sources/2020-gp-regression-tutorial]]（体系的入門）→ [[sources/2021-gp-models-intro]]（RKHS・誤差境界などの上級）の手前に置く最も平易な層。数式より図で「GP が何をしているか」を掴ませる点に価値がある。

## 背景と問題意識

著者は「GP は関数の集合上の確率分布を定義できる魔法のアルゴリズム」という決まり文句までは説明するチュートリアルが多いのに、その先（なぜ連続・無限の空間でモデル化できるのか）に踏み込むと急に逃げ腰になる、と問題提起する。一方 Rasmussen & Williams の教科書は数学的に重い。本稿は両者の**間の橋渡し**——「for dummies 以上、教科書未満」——を、Turner の講演図を使って埋めることを狙う。

## 内容（直感の積み上げ）

1. **動機**: 伝統的な非線形回帰は「最良の単一曲線」を返すが、同程度に当てはまる関数は複数あり得る。だから単一関数でなく**関数上の分布**が欲しい。
2. **多変量ガウスと共分散行列・条件付け**: 共分散行列の対角＝分散、非対角＝共分散。相関を下げると等高線が伸びた楕円→球になる。$y_1$ を固定して $y_2$ を条件付けると再びガウス。相関 0 では $y_1$ は $y_2$ について何も語らず、平均 0・分散大に。
3. **index プロットという可視化**: 等高線上の 1 点の $y_1,y_2$ を「index 1, 2 の高さ」として描き直す。複数サンプルすると相関の高い隣接点は一緒に上下する。これを 5D→20D に上げ、共分散をカラーマップ（暖色＝高相関、近い index ほど高相関）で表す。
4. **条件付け＝回帰の視覚的等価性**: 20D ガウスを 2 点（や 4 点）で条件付けると、観測に当てはまる**関数族**が生成され、その平均と分散が回帰結果になる。**観測に近い点は不確実性が小さく非ゼロ平均、遠い点は不確実性大・平均 0**（しかも実際は解析的で、サンプルは不要）。
5. **「現実」へ＝カーネルで連続化**: index は整数だが、共分散を **RBF（放射基底関数）カーネル** $K(x_1,x_2)=\sigma^2 e^{-\frac{1}{2l^2}(x_1-x_2)^2}$ で作れば、任意の**実数** $x$ を代入できる＝無限次元ガウス＝**ガウス過程**。$\Sigma = K + I\sigma_y^2$ でノイズを織り込む。
6. **GP の定義・ノンパラメトリック**: GP は平均関数 $m(x)$ とカーネル $K(x,x')$ で完全に規定。「**ノンパラメトリック＝無限個のパラメータ**」。パラメトリック関数の上に GP 事前分布を置く＝考える関数集合に事前分布を置く回帰。
7. **ハイパラ**: 縦スケール $\sigma$（縦の幅）と横スケール（長さスケール）$l$（相関が落ちる速さ＝滑らかさ）。周辺尤度が閉形式なので勾配法で最大化。
8. **計算の要は周辺化**: 無限×無限行列をどう扱うか →「関心ある有限個 $y_1$」と「気にしない無限個 $y_2$」に分け、**周辺化で $y_1$ だけ計算**できる（無限は忘れてよい）。予測はベイズ則で $p(y_1\mid y_2)$ を解析的に。ここで $C^{-1}$ の計算が $O(n^3)$（→ Cholesky）。データが増えるほど確信が増す。
9. **カーネル選択**: ラプラシアン（連続だが微分不可、平均はブラウン橋）、有理二次、周期。ベイズモデル比較は難しい積分で非現実的・事前分布に敏感。実務では**深層ガウス過程による自動カーネル設計**が一般的。

<figure>

![](../../raw/assets/2019-gp-not-for-dummies/61446482-1f098300-a947-11e9-8658-18c0b6e4d50d.png)

<figcaption>図（再掲）: 20D ガウスを観測点で条件付けると、当てはまる関数族から平均と分散（＝回帰の予測と不確実性）が得られる。観測に近いほど不確実性が小さい。［[[translations/2019-gp-not-for-dummies]] より］</figcaption>
</figure>

## 限界・批判的視点

- **ブログゆえ厳密さは省略**: 周辺化・条件付けの式や RKHS には踏み込まない（そこは [[sources/2021-gp-models-intro]] が担う）。図の多くは Turner 講演由来。
- **GP 自体の限界に言及**: 大規模データでの $O(n^3)$ スケール問題と、カーネル選択への高い感度。対処として deep GP / sparse GP を「今後の探求」として挙げるにとどまる。
- 回帰中心で分類は扱わない。
- （取り込み上の注記）原典は Web ブログで、Obsidian Web Clipper が本文中の図をほぼ取りこぼしていたため、**図 44 枚は原サイトから再取得**してローカル保存した。

## 意義（なぜこの wiki に重要か）

1. **PFN が近似する対象の「最も平易な絵」**: PFN/TFM は [[bayesian-inference]] の事後予測分布（PPD）を Transformer の 1 回の前向き計算で償却近似するが、その「再現したい振る舞い」——**観測に近いほど不確実性が小さく、遠いほど大きい滑らかな予測分布**——を本稿の index プロット＋条件付けの図が直感的に見せる。[[sources/2022-tabpfn]] のトイデータ実験で観察される TabPFN の挙動の視覚的原型。
2. **GP 三段リファレンスの入口**: [[gaussian-process]] には本稿（直感）→ [[sources/2020-gp-regression-tutorial]]（体系的入門・全式）→ [[sources/2021-gp-models-intro]]（RKHS・誤差境界・GP 動的モデル）の三段がそろった。PFN 系の理解の前提となる GP を、読者のレベルに応じて辿れる。
3. **カーネル＝滑らかさの直感**: RBF の長さスケールが滑らかさを決めるという核心は、[[structural-causal-model]] 系の合成 prior（TabICL/TabICLv2 が GP 関数を採用、カーネルの裾と滑らかさを関係づける）の土台。BayesOpt（[[bayesian-optimization]]）が GP をサロゲートに使う際の「不確実性で探索する」直感ともつながる。

## 用語と略称

- **GP** = Gaussian Process（ガウス過程）→ [[gaussian-process]]
- **index プロット** = 高次元ガウスのサンプルを「変数番号（横軸）× 値（縦軸）」で描く可視化（本稿独自の説明法）
- **RBF カーネル** = Radial Basis Function（放射基底関数）カーネル。$\sigma$（縦スケール）・$l$（長さスケール）を持ち、近い入力ほど高い共分散＝滑らかさを保証
- **ノンパラメトリック** = 実質無限個のパラメータを持つモデル
- **周辺化（marginalization）** = 無限次元の同時ガウスから関心ある有限部分だけを取り出す操作。GP の計算を可能にする鍵
- **ブラウン橋（Brownian bridge）** = ラプラシアンカーネルの全サンプル平均（データ点を結ぶ直線）
- **deep GP / sparse GP** = カーネル自動設計／大規模化のための発展手法
- **PPD / 事後予測分布** = 観測を条件にした予測分布 → [[bayesian-inference]]

## 参照（原典内リンク）

- Richard Turner 博士の講演 "Gaussian Processes: From the Basic to the State-of-the-Art"（YouTube `watch?v=92-98SYOdlY` ／ [スライド PDF](http://cbl.eng.cam.ac.uk/pub/Public/Turner/News/imperial-gp-tutorial.pdf)）— 本稿の図の出所
- Rasmussen & Williams, *Gaussian Processes for Machine Learning*（教科書）
- The Kernel Cookbook（カーネル選択の参考）

## 関連ページ

- [[gaussian-process]] — 本記事が解説する概念（この source は「最初の一冊」リファレンス）
- [[sources/2020-gp-regression-tutorial]] — 次に読む体系的入門（全式つき）
- [[sources/2021-gp-models-intro]] — さらに上級（RKHS・モデル誤差境界・GP 動的モデル）
- [[bayesian-inference]] — GP の事後予測分布／較正された不確実性
- [[structural-causal-model]] — GP 関数を含む合成 prior（カーネル＝滑らかさ）
- [[bayesian-optimization]] — GP をサロゲートに使う応用（不確実性で探索）
- [[translations/2019-gp-not-for-dummies]] — 本文の翻訳
