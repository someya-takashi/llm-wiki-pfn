---
type: source
source_path: "raw/papers/PFNs4BO- In-Context Learning for Bayesian Optimization.pdf"
source_kind: paper
title: "PFNs4BO: In-Context Learning for Bayesian Optimization"
authors: [Samuel Müller, Matthias Feurer, Noah Hollmann, Frank Hutter]
year: 2023
venue: ICML 2023
ingested: 2026-06-03
tags: [pfn, bayesian-optimization, in-context-learning, surrogate, acquisition-function, hyperparameter-optimization, user-prior, knowledge-gradient]
translation: "[[translations/2023-pfns4bo]]"
---

# PFNs4BO: ベイズ最適化のための文脈内学習（ICML 2023）

> 原典: [[translations/2023-pfns4bo]] ・ `raw/papers/PFNs4BO- In-Context Learning for Bayesian Optimization.pdf`（arXiv:2305.17535）
> 著者・年・会議: Samuel Müller, Matthias Feurer, Noah Hollmann, Frank Hutter / 2023（ICML 2023）

## 一言まとめ

**PFN（[[prior-data-fitted-networks]]）を、ベイズ最適化（[[bayesian-optimization]]）の GP サロゲートの“ベイズ的ドロップイン置換”として使えることを示した論文**。PFN 原典（[[sources/2021-transformers-can-do-bayesian-inference]]）と同じ著者グループが、「事前分布から合成データを引いて一度だけ訓練し、推論は文脈内学習（[[in-context-learning]]）で 1 回の順伝播」という PFN を、BO のサロゲートに据える。単純な GP・高度な GP（HEBO 着想）・BNN の 3 つの事前分布で実装し、固定 GP の事後分布をほぼ厳密に再現（図1）。さらに **GP では難しい拡張**——ユーザー事前分布（最適解の位置のヒント）、無関係次元の無視、獲得関数（知識勾配）を直接学習する非近視眼的 BO——を可能にする。この wiki では、これまで [[bayesian-optimization]] と [[prior-data-fitted-networks]] が「GP の代わりに使える高速サロゲートを PFN 系が埋めうる」と予告していた、その**正準的な実体**にあたる。

## 背景と問題意識

BO は「サロゲート（代理モデル）＋獲得関数」で高コストなブラックボックス関数を少ない評価回数で最適化する（[[bayesian-optimization]] / [[sources/2018-bayesian-optimization-tutorial]]）。そのサロゲートは長らく **GP** が事実上の標準だった。だが GP には弱点がある——

- **有効なカーネルとしてエンコードできる事前分布しか表せない**（カテゴリ・階層構造が難しい）、同時ガウス分布の仮定が裾の長いデータと不整合
- **3 乗のスケーリング**（$O(n^3)$）で大規模に弱い
- **定常カーネル**ゆえ非定常・ヘテロスケダスティック関数が苦手
- **カーネルのハイパーパラメータを毎回当てはめる**（経験ベイズ＝最尤）必要があり、完全ベイズより原理的でない・次元の呪いに弱い

PFN 原典は「事前分布からサンプリングできさえすれば、Transformer が 1 回の順伝播で PPD（[[bayesian-inference]] の事後予測分布）を近似できる」ことを示した。本論文はこれを **BO のサロゲートに持ち込み、GP の上記制約を回避**する。

## 提案手法 / 主張

**(1) PFN を BO サロゲートに（§3）**
- アルゴリズム1 で「経験ベイズの GP」を「訓練済み PFN」に置き換える（標準＝紫、PFN＝緑）。GP は毎反復でハイパラを当てはめる**オンラインコスト**を負うが、PFN は**事前分布ごとに一度だけ訓練**すればよく、次元の異なるタスクにも同じ PFN を使い回せる（文脈としてデータを渡すだけ）。
- 出力は **リーマン分布（Riemann distribution, 出力範囲をビンに離散化した区分定数の予測分布）**。これにより EI・PI・UCB の効用を**厳密に**計算できる（彼らは単純な EI が頑健と報告）。
- PFN は素の Transformer なので**出力→入力に勾配が流れる**。獲得関数の最大化（ランダム探索＋勾配上昇）と入力ワーピング（Kumaraswamy CDF）を勾配で行う。

**(2) 3 つの事前分布で柔軟性を実証（§4）**
- **単純な GP**（固定ハイパラ・RBF・零平均）: 真値の事後を厳密比較できるトイ。図1 のとおり PFN は固定 GP 事後をほぼ厳密に再現。
- **HEBO 着想（HEBO / HEBO⁺）**: NeurIPS ブラックボックス競技優勝の SOTA BO 手法 HEBO を模倣。長さスケール（次元ごと）・出力スケール・ノイズを分布からサンプルし 3/2 Matérn で関数を生成。**HEBO⁺** は無関係次元 30% を足した拡張。
- **BNN**: ネットワークのアーキ（層数・幅・スパース性・ノイズ）と重みをサンプルして関数を生成。

**(3) GP では難しい事前分布の拡張（§5）— ここが本論文の核心的価値**
- **入力ワーピング**（事前分布に組み込む）
- **無関係次元（spurious dimensions）**: 出力に効かない特徴量を事前当てはめ時にランダムに混ぜる（GP/NN には渡さない）。HEBO⁺ で 30%。GP/経験ベイズには組み込みにくい。
- **ユーザー事前分布（user priors）**: ユーザーが「最適解はこの区間 $I$ にありそう、確信度 $\rho$」と指定でき、事前分布を $p(D|\rho,I)=\rho\,p(D|m\in I)+(1-\rho)\,p(D)$ に即座に適応（式3）。$\rho, I$ は追加トークン（スタイル埋め込み風の線形エンコーダ）で PFN に渡す。1 つの PFN が任意の区間・確信度を受け付ける。
- **非近視眼的（non-myopic）獲得関数**: **知識勾配（KG; Knowledge Gradient, 追加観測で事後平均の最大がどれだけ上がるかの期待。[[bayesian-optimization]] / [[sources/2018-bayesian-optimization-tutorial]] 参照）** を、推論時の最適化なしに PFN が**直接近似するよう学習**。少次元の 4 探索空間で標準 EI を上回る。

## 実験結果と知見

- **GP 事後の再現（§6）**: 経験ベイズなしで PFN が真の GP 事後に一致。ただし**事前分布サンプルが少ない（≤10 万）と緩く、十分（≥2000 万）で緊密**になる（訓練量が効く＝原典の知見と整合）。HEBO 事前分布でも PFN と経験ベイズは同程度に事後を近似（表8）。表1 では 1000 データセットで GP（経験ベイズ）と PFN が同等、ほとんど同じ最大値を発見（引き分けが多数）。
- **HPO ベンチマーク 3 種・105 タスク・19 探索空間（§7）**:
  - **HPO-B**（木ベース ML, 離散, 51 タスク）: **PFN(HEBO⁺) がベースライン（GP・DNGO・DGP・HEBO 等）を上回る**、PFN(BNN) は最良ベースライン並み（図4 上）。HPO-B の約 10% のハイパラに対数スケールが欠落していた（バグ）ため、推論時入力ワーピングの実演に使用。
  - **Bayesmark**（連続, 36 タスク）: PFN(BNN) は HEBO に劣るが他を上回る、HEBO⁺ は HEBO と同等〜やや劣。完全ベイズゆえ**初期設計が小さくてよく、単一の初期点（特に探索空間の隅）が全手法に優越**。
  - **PD1**（大規模 NN 調整, 18 タスク, 5 次元）: PFN(HEBO⁺) が HEBO を上回り、**一般的なユーザー事前分布を足すとさらに明確に改善**（図4 下）。ユーザー事前分布からの準ランダム探索は BO に勝てず、事前分布が問題を自明化していないことも確認。

## 限界・批判的視点

著者自身が挙げる限界:
- **事前分布で尤度が極端に低いデータに弱い**（GP は問題にしにくい）。
- **誤って対数スケールでない次元の自動回復はできなかった**（正しく指定された空間では強い）。
- **複数データ点の同時サンプルを引けない**ため、ノイズ付き EI（Letham 2018）のような獲得関数の素直な適応が難しい（[[gaussian-process]] の予測は同時共分散を出せるのと対照的——cf. [[questions/gaussian-process-intuitive-explainer]] の「帯は対角・サンプルは全共分散」）。
- **精度指標に偏り**、非精度タスク（べき変換で動特性が変わる）で弱い。
- **約 1000 データ点まで**でしか良い性能が示されていない（Hollmann 2023 時点。後の TabPFN v2/2.5/3 が大規模化）。
- 今後: 非定常・ヘテロスケダスティック・離散/カテゴリ・不連続・100〜1000 の無関係特徴量、student-t 過程など非ガウス出力。

## 意義（なぜ重要か）

ベイズ最適化の「サロゲート＝GP」という前提を、**PFN という学習されたサロゲートで置き換えられる**ことを大規模に実証した論文。GP の弱点（カーネル表現の制約・$O(n^3)$・定常性・ハイパラ当てはめ）を、「**事前分布を書いて一度訓練するだけ**」で回避し、しかも GP では組み込みにくい**ユーザー事前分布・無関係次元・非近視眼的獲得関数**を自然に扱える。これは [[bayesian-optimization]] の中核（[[sources/2018-bayesian-optimization-tutorial]] が解説する「サロゲート＋獲得関数」）に PFN を接ぎ木した到達点であり、[[tabular-foundation-model]] の文脈で言及される「高次元 BO のサロゲートに TabPFNv2 を使う（GIT-BO 等）」の直接の源流。PFN 原典（[[sources/2021-transformers-can-do-bayesian-inference]]）の応用編として、GP と PFN を結ぶ橋（[[questions/pfn-paper-and-gaussian-process]]）の“応用側”を担う。

## 用語と略称

- **PFN** = Prior-Data Fitted Network（事前分布の合成データで一度訓練し、推論は文脈内学習で行う枠組み）→ [[prior-data-fitted-networks]]
- **BO** = Bayesian Optimization（ベイズ最適化）→ [[bayesian-optimization]]
- **サロゲート（surrogate）** = 高コストな目的関数を代理する安価なモデル（従来は GP、本論文は PFN）
- **PPD** = Posterior Predictive Distribution（事後予測分布）→ [[bayesian-inference]]
- **ICL** = In-Context Learning（文脈内学習。推論時に重み更新なしで文脈から予測）→ [[in-context-learning]]
- **獲得関数（acquisition function）** = 次にどこを評価すると得かを測る関数。EI（期待改善）/ PI（改善確率）/ UCB（上側信頼限界）
- **リーマン分布（Riemann distribution）** = 出力範囲をビンに離散化した区分定数の予測分布。EI/PI/UCB を厳密計算できる
- **HEBO / HEBO⁺** = NeurIPS BBO 競技優勝の SOTA BO 手法／その PFN 向け拡張（無関係次元 30% 等）
- **経験ベイズ（Empirical Bayes）** = カーネルのハイパラを最尤で点推定する GP の標準当てはめ
- **ユーザー事前分布（user prior）** = 最適解の位置に関するユーザーの信念（区間 $I$ ＋確信度 $\rho$）を事前分布に組み込む仕組み
- **無関係次元（spurious dimensions）** = 出力に効かない特徴量。事前当てはめでランダムに混ぜて頑健化
- **非近視眼的（non-myopic）** = 1 ステップ先だけでなく将来を見据えた最適化
- **KG** = Knowledge Gradient（知識勾配。追加観測で事後平均の最大がどれだけ上がるかの期待）→ [[bayesian-optimization]]
- **HPO** = Hyperparameter Optimization（ハイパーパラメータ最適化）
- **BNN** = Bayesian Neural Network（重みを分布として扱うニューラルネット）

## 関連ページ

- [[bayesian-optimization]] — 本論文が PFN サロゲートを持ち込んだ対象（サロゲート＋獲得関数の枠組み）
- [[prior-data-fitted-networks]] — 本論文が BO に応用した枠組み（同著者グループ）
- [[sources/2021-transformers-can-do-bayesian-inference]] — PFN の原典（本論文はその BO 応用）
- [[gaussian-process]] — PFN が置き換える従来サロゲート（図1 で固定 GP 事後を再現）
- [[sources/2018-bayesian-optimization-tutorial]] — EI/KG など獲得関数・BO の基礎
- [[in-context-learning]] — PFN の推論メカニズム（タイトルの "In-Context Learning"）
- [[bayesian-inference]] — 近似対象である PPD
- [[questions/pfn-paper-and-gaussian-process]] — GP と PFN の関係（本論文はその応用側）
- [[tabular-foundation-model]] — TabPFN を高次元 BO のサロゲートに使う流れ（GIT-BO 等）の源流
- [[translations/2023-pfns4bo]] — 本文 §1〜8 ＋ 付録 A〜M の翻訳
