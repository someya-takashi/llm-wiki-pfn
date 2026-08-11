---
type: source
source_path: raw/articles/Set Transformer_ A Framework for Attention-based Permutation-Invariant Neural Networks.md
source_kind: paper
title: "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"
authors: [Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, Yee Whye Teh]
year: 2018
venue: ICML 2019
ingested: 2026-08-12
tags: [set-transformer, permutation-invariance, attention, inducing-points, amortized-inference, meta-learning, deep-sets]
translation: "[[translations/2018-set-transformer]]"
---

# Set Transformer: アテンションに基づく置換不変ニューラルネットワーク（Lee et al. 2019）

> 原典: [[translations/2018-set-transformer]] ・ `raw/articles/Set Transformer_ A Framework for Attention-based Permutation-Invariant Neural Networks.md`（arXiv:1810.00825）
> 著者・年・会議: Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, Yee Whye Teh / 2018（ICML 2019）

## 一言まとめ

**「データの集合を、順序に依存せず（置換不変, permutation invariant）、要素間の相互作用まで見て処理する Transformer」** を提案した論文。単純な平均/最大プーリングで集約する従来の set pooling（DeepSets 系）に対し、①エンコーダの**自己アテンション（SAB）**で要素間相互作用を捉え、②スパースガウス過程由来の**誘導点（inducing points）**で計算量を $O(n^2)\to O(nm)$ に落とし（**ISAB**）、③集約さえも学習可能なアテンション（**PMA**）にする——という3点で刷新した。これは後の [[prior-data-fitted-networks]]（PFN）系が「データセットを丸ごと集合として Transformer に入れる」ための**建築部品の系譜の原点**であり、[[sources/2025-tabicl]]（TabICL）では列埋め込みにこの Set Transformer/ISAB が**そのまま採用**されている。

## 背景と問題意識

多重インスタンス学習・3D 形状認識・few-shot 分類・そして**メタ学習**（1つのタスクの訓練データセット＝1つの入力集合）は、いずれも「**集合を入力にとる**」問題である。集合には順序がないので、モデルは (1) **置換不変**（要素を並べ替えても出力が変わらない）で、(2) **任意サイズの集合**を扱えなければならない。通常のフィードフォワード NN は両方に違反し、RNN は順序に敏感である。

先行する解は **set pooling（DeepSets; Zaheer et al. 2017 等）**: 各要素を独立にエンコード（$\phi$）→ mean/sum/max で**プーリング**→ 後処理（$\rho$）。この形は「sum プーリング＋連続な $\rho,\phi$」で**すべての置換不変関数を表現できる**（普遍近似）ことが証明されており、理論的には十分に見える。

しかし実際には決定的な弱点がある。**各要素を独立に処理するため、要素間の相互作用の情報が必然的に捨てられる**。象徴的なのが**償却クラスタリング（amortized clustering）**——「点の集合 → クラスタ中心」の写像を学習する問題（通常は EM のような反復アルゴリズムで解くところを、1回の順伝播に**償却**する。この「反復推論を学習で置き換える」構図はまさに [[bayesian-inference]] の償却推論であり、[[expectation-maximization]]／[[gaussian-mixture-model]] の文脈でも重要）。ここでは「どの点をどのクラスタに割り当てるか」を、クラスタ同士が**説明の取り合い（explaining away, 一方が説明した点を他方は説明しない）**をするように決めねばならず、要素間・出力間の相互作用が本質的。プーリング構造は空間の量子化を学ぶことしかできず、**量子化が集合の内容に適応できない**ため過少適合に陥る（§5 で実証）。

## 提案手法 / 主張

Set Transformer は「エンコーダ＝置換**同変**（equivariant, 並べ替えると出力も同じ順で並べ替わる）な層の積み重ね」＋「デコーダ＝置換**不変**な集約」という set pooling と同じ骨格を保ちつつ、各部品をアテンションに置き換える。

<figure>

![](../../raw/assets/2018-set-transformer/main.png)

<figcaption>図1(a)（再掲）: Set Transformer の全体像。入力集合 X → エンコーダ（SAB/ISAB の積み重ね、要素間で相互作用）→ 特徴集合 Z → デコーダ（PMA で集約 → SAB → rFF）→ 出力。</figcaption>
</figure>

**部品1: MAB / SAB（自己アテンションのブロック化）**。MAB（Multihead Attention Block）は Transformer のエンコーダブロックから**位置エンコーディングとドロップアウトを除いた**もの（位置を捨てる＝順序を捨てる、が肝）。SAB = MAB(X, X)（集合内の自己アテンション）で、積み重ねるほど高次の相互作用を符号化できる。

**部品2: ISAB（誘導点で $O(nm)$ に）**。SAB の $O(n^2)$ を避けるため、**学習可能な $m$ 個のベクトル $I$（誘導点）** を導入し、$H=\mathrm{MAB}(I,X)$（誘導点が集合を要約）→ $\mathrm{MAB}(X,H)$（集合が要約を参照）の2段にする。計算量は $O(nm)$。名前の由来は**スパースガウス過程の誘導点法**（[[gaussian-process]] の $O(N^3)$ を少数の擬似入力で近似する技法）で、「少数の学習された代表点を通して全要素を間接的に比較する」という同じ発想。

<figure>

![](../../raw/assets/2018-set-transformer/imab.png)

<figcaption>図1(d)（再掲）: ISAB の構造。学習可能な誘導点 I がまず入力集合 X にアテンションして要約 H を作り（1つ目の MAB, m×d）、次に X が H にアテンションして n×d の出力を作る（2つ目の MAB）。計算量は O(nm)。</figcaption>
</figure>

**部品3: PMA（プーリングも学習する）**。平均/最大の固定プーリングの代わりに、**$k$ 個の学習可能なシードベクトル $S$ をクエリにしたアテンション** $\mathrm{PMA}_k(Z)=\mathrm{MAB}(S,\mathrm{rFF}(Z))$ で集約する。$k=1$ なら1本のベクトルへの要約、$k>1$ なら「$k$ 個の相関した出力」（例: $k$ 個のクラスタ中心）を出し、その後の SAB で**出力間の相互作用（説明の取り合い）**までモデル化する。直感: 最大値回帰のように「1つの要素だけがターゲットを決める」問題では、その要素を**探してアテンションする**集約が固定平均より本質的に有利。

**理論**: SAB/ISAB は置換同変、PMA は置換不変なので、**Set Transformer 全体は置換不変**（命題1）。さらに DeepSets の普遍近似の結果の上に、**Set Transformer も置換不変関数の普遍近似器**であることを証明（命題2）。

## 実験結果と知見

一貫した比較設計:「エンコーダ＝rFF（独立処理）vs SAB/ISAB」×「デコーダ＝固定プーリング vs PMA」の全組み合わせ。

- **最大値回帰（トイ）**: mean/sum プーリングは大きく外す（MAE 1.9〜2.1）が、SAB+PMA は 0.21 と、答えを直接持つ max プーリング（0.14）に肉薄。**アテンションが「効く要素を探す」ことを学習できる**証拠。
- **ユニーク文字数え上げ（Omniglot）**: 要素間の「同じか違うか」の比較が本質のタスク。rFF 系 0.44〜0.46 に対し SAB+PMA **0.60**。誘導点は**1個でも** rFF 系を上回り、増やすほど単調に改善（図3）。
- **償却クラスタリング（合成 2D MoG / CIFAR-100）**: 本命タスク。ISAB(16)+PMA が LL/ARI とも最良で、**EM 1ステップ後にはオラクル（真のパラメータ／収束 EM）すら上回る**（ARI 0.9223 vs オラクル 0.9150）。小標本ではオラクルの真パラメータより「データに適応した」推定が勝てるため。さらに**16個の誘導点の ISAB が完全な SAB を上回る**——誘導点による知識転移・正則化の効果。改善幅は「SAB の寄与 < **PMA の寄与**」で、**アテンションによる集約（デコーダ）が最重要**という主張を支持。
- **集合内異常検出（CelebA）**: SAB+PMA が AUROC 0.594 で全手法を有意に上回る。
- **点群分類（ModelNet40）**: 小さい集合（100点）では ISAB+PMA が最良（0.845）。ただし大きい集合（5000点）では ISAB+Pooling（0.904）に逆転される——**集合が十分大きいと単純集約でも情報が足り、相互作用モデル化の利得が減る**という限界も正直に報告。

## 限界・批判的視点

- **大集合では PMA の優位が消える**ことがある（点群 5000 点）。相互作用モデリングの価値はタスクの「情報の濃さ」に依存する。
- **誘導点数 $m$ は新たなハイパーパラメータ**。本論文では小さい $m$（16程度）で十分と示すが、選び方の原理は与えていない。
- 実験はいずれも小〜中規模。**表形式データへの適用や大規模事前訓練は射程外**（それを実行したのが後の PFN 系）。
- 結論で「**ベイズモデルの事後推論をメタ学習する**のに Set Transformer を使うのは有望」と示唆するに留まる——この予言をまさに実現したのが [[sources/2021-transformers-can-do-bayesian-inference]]（PFN）である。

## PFN 系との関係（この wiki での位置づけ）

1. **設計思想の祖先**: PFN 原典 [[sources/2021-transformers-can-do-bayesian-inference]] と [[sources/2022-tabpfn]] は「データセット $D$ を**順序不変な集合**として Transformer に入れる（位置エンコーディングなし・集合値入力）」を採る。これは本論文が体系化した「アテンションによる置換不変な集合処理」そのもの。「メタ学習＝入力集合がタスクの訓練データセット」（§1）という見方も PFN の問題設定を先取りしている。
2. **償却推論の先行例**: §5.3 の償却クラスタリングは「反復アルゴリズム（EM）を1回の順伝播に償却」——PFN が「MCMC/VI による事後推論を順伝播1回に償却」するのと同じ構図の、アーキテクチャ側からの先行例（[[questions/evaluating-pfn-with-known-gmm]] の GMM 評価とも通じる）。
3. **ISAB/誘導点の直接採用**: [[sources/2025-tabicl]] は特徴量埋め込みを「集合入力問題」と再定式化し **Set Transformer（ISAB, 誘導点128）を明示的に使用**。[[sources/2026-tabpfn-3]] も全行への二次アテンションを避ける**誘導点**（128個）を採用。[[sources/2025-nanotabpfn]] は位置埋め込みを外して置換不変性を保つ実装を解説。

## 用語と略称

- **置換不変（permutation invariant）** = 入力集合の要素を並べ替えても出力が変わらない性質。**置換同変（equivariant）** = 並べ替えると出力も同じ順で並べ替わる性質（同変な層を積んで最後に不変な集約をすると全体が不変になる）
- **set pooling / DeepSets** = 各要素を独立にエンコード→ mean/sum/max で集約する先行アーキテクチャ（sum 版は普遍近似器）
- **MAB** = Multihead Attention Block（位置エンコーディングとドロップアウトを除いた Transformer エンコーダブロック）
- **SAB** = Set Attention Block = MAB(X, X)。集合内の自己アテンション、$O(n^2)$
- **ISAB** = Induced Set Attention Block。$m$ 個の学習可能な誘導点を介した2段 MAB、$O(nm)$
- **誘導点（inducing points）** = 集合を要約する少数の学習可能ベクトル。スパース [[gaussian-process]] の誘導点法・Nyström 法に由来
- **PMA** = Pooling by Multihead Attention。$k$ 個の学習可能シードベクトルをクエリにする集約（固定プーリングの置き換え）
- **rFF** = row-wise FeedForward（各要素を独立・同一に処理する層）
- **償却クラスタリング（amortized clustering）** = 「集合→クラスタパラメータ」の写像を学習し、EM の反復を順伝播1回に**償却（amortize）**すること
- **説明の取り合い（explaining away）** = ある出力（クラスタ）が説明した要素を他の出力が重複して説明しないようにする相互作用
- **MoG / ARI / AUROC / AUPR** = ガウス混合（[[gaussian-mixture-model]]）／調整ランド指数／ROC 曲線下面積／適合率-再現率曲線下面積

## 関連ページ

- [[permutation-invariant-neural-networks]] — 本論文が代表する「集合を順序不変に処理する NN」の概念ページ
- [[prior-data-fitted-networks]] — 「データセット＝集合」入力の設計を受け継ぎ、ベイズ推論の償却に使った枠組み
- [[in-context-learning]] — 集合として与えた文脈から重み更新なしに予測する構図
- [[sources/2021-transformers-can-do-bayesian-inference]] — 本論文が結論で示唆した「事後推論のメタ学習」を実現した PFN 原典
- [[sources/2025-tabicl]] — Set Transformer/ISAB を列埋め込みに直接採用した表形式基盤モデル
- [[sources/2026-tabpfn-3]] — 誘導点による要約で大規模化した TabPFN-3
- [[sources/2025-nanotabpfn]] — 置換不変性を保つ TabPFN 実装の教育的解説
- [[gaussian-process]] — ISAB の名の由来であるスパース GP の誘導点法
- [[expectation-maximization]] / [[gaussian-mixture-model]] — 償却クラスタリング実験で「置き換えられる側」の反復アルゴリズム
- [[translations/2018-set-transformer]] — 全文翻訳（Abstract〜§6、図11枚ローカル保存）
