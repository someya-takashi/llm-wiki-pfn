# Index — PFN LLM Wiki カタログ

全ページのカタログ。1 行 = 1 ページ（`- [[<slug>]] — <一行の説明>`）。ingest / query で新規ページを作るたびに更新する。

## Overview

- [[overview]] — PFN（Prior-Data Fitted Networks）領域の総括

## Sources

### paper

- [[sources/2021-transformers-can-do-bayesian-inference]] — Transformers Can Do Bayesian Inference: PFN の枠組みを提示した原典（Müller et al. 2021, ICLR 2022）
- [[sources/2022-tabpfn]] — TabPFN v1: 小規模表形式分類を 1 秒で解く Transformer（Hollmann et al. 2022, ICLR 2023）
- [[sources/2025-tabpfn-v2]] — TabPFN v2: 表形式基盤モデルによる小規模データでの正確な予測（Hollmann et al. 2025, Nature）
- [[sources/2025-tabpfn-2-5]] — TabPFN-2.5: 表形式基盤モデルの最先端を前進（Prior Labs 2025, arXiv 2511.08667）
- [[sources/2026-tabpfn-3]] — TabPFN-3 Technical Report: 100万行スケール＋テスト時計算（Prior Labs 2026, arXiv 2605.13986）
- [[sources/2025-tabicl]] — TabICL: 大規模データ ICL の表形式基盤モデル（Qu et al. / Inria, ICML 2025, arXiv 2502.05564）
- [[sources/2026-tabicl-v2]] — TabICLv2: より良く・速く・スケーラブルでオープンな TFM（Qu et al. / Inria, ICML 2026, arXiv 2602.11139）
- [[sources/2025-real-tabpfn]] — Real-TabPFN: 実データでの継続事前訓練（Garg et al. / Freiburg, arXiv 2507.03971）
- [[sources/2023-pfns4bo]] — PFNs4BO: ベイズ最適化のための文脈内学習（Müller et al. 2023, ICML 2023, arXiv 2305.17535）。PFN を BO サロゲートに
- [[sources/2025-mitra]] — Mitra: 表形式基盤モデルを強化する混合合成事前分布（Zhang et al. 2025, Amazon, arXiv 2510.21204）。prior 設計の原理（性能・多様性・独自性）
- [[sources/2025-git-bo]] — GIT-BO: 表形式基盤モデルによる高次元ベイズ最適化（Yu et al. 2025, MIT, arXiv 2505.20685）。TabPFNv2 を高次元 BO サロゲートに（勾配誘導部分空間）

### article / tutorial

- [[sources/2019-gp-not-for-dummies]] — ガウス過程、完全な初心者向けとまではいかないが（Yuge Shi 2019, The Gradient）。[[gaussian-process]] の最も直感寄りの入門
- [[sources/2020-gp-regression-tutorial]] — ガウス過程回帰への直感的チュートリアル（Jie Wang 2020, arXiv 2009.10862）。[[gaussian-process]] のリファレンス
- [[sources/2022-gpr-part1-basics]] — ガウス過程回帰入門 Part 1: 基礎（Kaixin Wang 2022, Medium）。[[gaussian-process]] の実践/応用入門（GPflow・カーネル選択・ハイパラ調整）
- [[sources/2022-gpr-part2-concrete]] — ガウス過程回帰入門 Part 2: コンクリート強度予測への応用（Kaixin Wang 2022, Medium）。GPR の実データ応用＋SHAP 解釈（Part 1 の続編）
- [[sources/2025-gp-intuitive-intro]] — ガウス過程への直感的入門（Fan Pu Zeng 2025, fanpu.io）。ノンパラメトリック性と NN との関係（NNGP）を軸にした入門
- [[sources/2018-bayesian-optimization-tutorial]] — ベイズ最適化のチュートリアル（Peter I. Frazier 2018, arXiv 1807.02811）。[[bayesian-optimization]] のリファレンス
- [[sources/2021-gp-models-intro]] — ガウス過程モデル入門（Thomas Beckers 2021, arXiv 2102.05497）。[[gaussian-process]] の上級リファレンス（RKHS・モデル誤差境界・GP 動的モデル）
- [[sources/2025-causalpfn]] — CausalPFN: 文脈内学習による償却的な因果効果推定（Balazadeh et al. 2025, U Toronto/Layer6, arXiv 2506.07918）。PFN を因果効果推定（ATE/CATE）へ拡張（潜在結果＋ignorability）
- [[sources/2025-do-pfn]] — Do-PFN: 因果効果推定のための文脈内学習（Robertson et al. 2025, Freiburg/Tübingen, arXiv 2506.06039）。介入つき SCM prior で条件付き介入分布を予測（do 計算・グラフ不要）
- [[sources/2025-chronos-2]] — Chronos-2: 単変量から普遍的予測へ（Ansari et al. 2025, Amazon, arXiv 2510.15821）。時系列基盤モデル。group attention＝時系列版 ICL で多変量・共変量つき予測をゼロショット
- [[sources/2025-nanotabpfn]] — nanoTabPFN: TabPFN の軽量かつ教育的な再実装（Pfefferle et al. 2025, Freiburg/Prior Labs, arXiv 2511.03634）。500行・1分・単一 GPU で従来 ML ベースラインに並ぶ nanoGPT 流の教育ツール
- [[sources/2026-shappfn]] — ShapPFN: 表形式基盤モデルのためのリアルタイム説明（Sena & Azevedo 2026, Kunumi/UFMG, arXiv 2603.29946）。nanoTabPFN に Shapley 値説明を内蔵し予測と説明を1順伝播で（KernelSHAP 1000倍速）
- [[sources/2024-bayesian-inference-step-by-step]] — ベイズ推論ステップバイステップ入門（Rahuldhrh 2024, Medium）。MLEの限界→事前/事後→共役事前→逐次更新。[[bayesian-inference]] の基礎入門
- [[sources/2020-understanding-bayesian-inference]] — ベイズ推論を機械学習パラダイムとして説く入門（Jonty Sinai 2020）。予測分布(PPD)＋近似推論(MCMC/VI/EP)まで到達。[[bayesian-inference]] の入門
- [[sources/2021-mcmc-explained]] — MCMC の入門解説（Shivam Agrahari 2021, TDS）。モンテカルロ＋マルコフ連鎖→定常分布→詳細釣り合い→Metropolis–Hastings。[[bayesian-inference]] の近似推論を深掘り
- [[sources/2019-conceptual-intro-mcmc]] — MCMC の概念的入門（Joshua Speagle 2019, arXiv 1909.12313）。グリッド→重点サンプリング→MCMC、ESS・自己相関・次元の呪い・typical set まで。[[bayesian-inference]] 近似推論の厳密版
- [[sources/2018-mcmc-simple-introduction]] — MCMC の平易な入門（van Ravenzwaaij et al. 2018, Psychon Bull Rev）。Metropolis→Gibbs→DE、burn-in/R-hat/blocking と SDT 応用。[[markov-chain-monte-carlo]] の応用入門
- [[sources/2024-mcmc-basics-pymc3]] — MCMC の超入門＋PyMC3/NUTS 実コード（Sarowar Ahmed 2024, Medium）。身長データで μ・σ の事後を推定。[[markov-chain-monte-carlo]] の実践レシピ
- [[sources/2024-gmm-em-clustering]] — ガウス混合モデル（GMM）と EM の実装入門（Tejas Pawar 2024, Medium）。3次元データで E-step/M-step を反復しクラスタリング。[[gaussian-mixture-model]] の一次資料
- [[sources/2025-gmm-sklearn-guide]] — GMM の sklearn 実装ガイド（Lakhan Bukkawar 2025, GoPenAI）。soft/hard 比較・顧客セグメンテーション・BIC/AIC で成分数選択。[[gaussian-mixture-model]] の実践版
- [[sources/2020-gmm-overview]] — GMM の理論的概観（Patacchiola 2020, MML 第11章ベース）。潜在変数→責任=事後→非識別可能性→鶏と卵の EM 導出。[[gaussian-mixture-model]] の理論リファレンス
- [[sources/2019-em-algorithm-explained]] — EM アルゴリズムの解説（Chloe Bi 2019, Medium）。下界=ELBO・Jensen・KL・座標上昇で「なぜ収束するか」。[[expectation-maximization]] の一次資料
- [[sources/2014-no-u-turn-sampler]] — No-U-Turn Sampler の原典（Hoffman & Gelman 2014, JMLR）。HMC の L を U ターン基準＋双対平均化で自動化＝turnkey。Stan の中核、PFN 原典の速度比較相手。[[markov-chain-monte-carlo]] の HMC/NUTS 一次資料
- [[sources/2017-conceptual-intro-hmc]] — HMC の幾何学的基礎（Betancourt 2017, arXiv 1701.02434）。typical set から HMC を必然として導き、E-BFMI・発散で診断。NUTS 原典の Why。[[markov-chain-monte-carlo]] の理論的基礎
- [[sources/2022-em-by-examples]] — EM を3例で理解（Clarice Wang 2022, Medium）。K-Means・Two Coins（二項混合）・GMM を同じ EM として統一。[[expectation-maximization]] の実例集

## Translations

- [[translations/2021-transformers-can-do-bayesian-inference]] — PFN 原典の全文翻訳（本文＋付録 A〜H）
- [[translations/2022-tabpfn]] — TabPFN v1 論文の全文翻訳（本文＋付録 A〜F）
- [[translations/2025-tabpfn-v2]] — TabPFN v2 論文の翻訳（Main＋Methods）
- [[translations/2025-tabpfn-2-5]] — TabPFN-2.5 の翻訳（本文＋付録 A〜I、Appendix B は件数サマリ）
- [[translations/2026-tabpfn-3]] — TabPFN-3 の翻訳（本文 §1〜5、References・Appendix は対象外）
- [[translations/2025-tabicl]] — TabICL の翻訳（本文 §1〜6 ＋ 付録 A〜E）
- [[translations/2026-tabicl-v2]] — TabICLv2 の全訳（本文 §1〜9 ＋ 付録 A〜K、全 69 図）
- [[translations/2025-real-tabpfn]] — Real-TabPFN の翻訳（本文 §1〜6）
- [[translations/2023-pfns4bo]] — PFNs4BO の翻訳（本文 §1〜8 ＋ 付録 A〜M）
- [[translations/2025-mitra]] — Mitra の翻訳（本文 §1〜5 ＋ 付録 A〜D、図26枚）
- [[translations/2025-git-bo]] — GIT-BO の翻訳（本文 §1〜6、図2〜5）
- [[translations/2019-gp-not-for-dummies]] — 「GP, not quite for dummies」の翻訳（本文、図44枚を原サイト再取得）
- [[translations/2020-gp-regression-tutorial]] — GP 回帰チュートリアルの翻訳（本文）
- [[translations/2022-gpr-part1-basics]] — GPR 入門 Part 1 の翻訳（本文、図12枚）
- [[translations/2022-gpr-part2-concrete]] — GPR 入門 Part 2 の翻訳（本文、図9枚）
- [[translations/2025-gp-intuitive-intro]] — 「ガウス過程への直感的入門」の翻訳（本文、図7枚）
- [[translations/2018-bayesian-optimization-tutorial]] — ベイズ最適化チュートリアルの翻訳（本文 §1〜7）
- [[translations/2021-gp-models-intro]] — ガウス過程モデル入門の翻訳（本文 §1〜5）
- [[translations/2025-causalpfn]] — CausalPFN の翻訳（本文 §1〜7 ＋ 付録 A〜D、全14図）
- [[translations/2025-do-pfn]] — Do-PFN の翻訳（本文 §1〜5 ＋ 付録 A〜D、全19図）
- [[translations/2025-chronos-2]] — Chronos-2 の翻訳（本文 §1〜6 ＋ 付録 A〜B、全19図・主要テーブル）
- [[translations/2025-nanotabpfn]] — nanoTabPFN の翻訳（本文 §1・3〜5 ＋ 付録 A〜B、全4図）
- [[translations/2026-shappfn]] — ShapPFN の翻訳（本文 §1〜5 ＋ 付録 A、図1＝アーキ図）
- [[translations/2024-bayesian-inference-step-by-step]] — ベイズ推論入門記事の全文翻訳（本文、数式図24枚をローカル保存）
- [[translations/2020-understanding-bayesian-inference]] — 「ベイズ推論を理解する」の全文翻訳（本文、図解8枚ローカル保存）
- [[translations/2021-mcmc-explained]] — MCMC 解説記事の全文翻訳（本文＋Extras、図/数式21枚ローカル保存）
- [[translations/2019-conceptual-intro-mcmc]] — MCMC 概念的入門論文の全文翻訳（本文§1〜9＋演習、図解15枚ローカル保存）
- [[translations/2018-mcmc-simple-introduction]] — MCMC 平易入門の全文翻訳（本文＋用語集、図3枚ローカル保存）
- [[translations/2024-mcmc-basics-pymc3]] — MCMC 超入門の全文翻訳（本文＋PyMC3 コード、図1枚ローカル保存）
- [[translations/2024-gmm-em-clustering]] — GMM/EM クラスタリング記事の全文翻訳（本文＋Python 実装、図9枚ローカル保存）
- [[translations/2025-gmm-sklearn-guide]] — GMM sklearn ガイドの全文翻訳（本文＋sklearn コード、図5枚ローカル保存）
- [[translations/2020-gmm-overview]] — GMM 理論概観の全文翻訳（導出＋numpy 実装、図2枚ローカル保存）
- [[translations/2019-em-algorithm-explained]] — EM 解説記事の全文翻訳（下界導出、図8枚ローカル保存）
- [[translations/2014-no-u-turn-sampler]] — NUTS 原典の全文翻訳（本文§1〜5＋アルゴリズム、図解12枚ローカル保存）
- [[translations/2017-conceptual-intro-hmc]] — HMC 概念的入門の全文翻訳（序〜§7＋付録A、図43枚ローカル保存）
- [[translations/2022-em-by-examples]] — EM 実例集の全文翻訳（K-Means/Two Coins/GMM、図21枚ローカル保存）

## Concepts

- [[prior-data-fitted-networks]] — 合成データで一度訓練し推論時にベイズ推論を償却近似する中核概念
- [[in-context-learning]] — 重み更新なしに文脈から予測する枠組み
- [[bayesian-inference]] — 近似対象である事後予測分布（PPD）と償却推論
- [[markov-chain-monte-carlo]] — 事後分布から直接サンプルする近似推論の定番（PFN が償却で置換）
- [[structural-causal-model]] — TabPFN の表形式事前分布の中核（因果 DAG による合成データ生成）
- [[gaussian-process]] — PFN の検証ベンチマーク兼事前分布の素材となる古典的ベイズモデル
- [[gaussian-mixture-model]] — ガウスの混合による確率的クラスタリング／密度推定。EM で当てはめ、既知 GMM は PFN 評価の物差し
- [[expectation-maximization]] — 潜在変数モデルの MLE/MAP を E/M-step の反復で求める汎用アルゴリズム（下界=ELBO 最大化、VI の特別な場合）
- [[tabular-foundation-model]] — 予測＋生成＋密度推定＋埋め込みを単一モデルで担う表形式基盤モデル（TabPFN v2）
- [[bayesian-optimization]] — 高コスト関数を GP サロゲート＋獲得関数で大域最適化する手法。PFN/TFM の下流応用先

略称リダイレクト（正式名称ページに集約）:
- PFN → [[prior-data-fitted-networks]]
- ICL → [[in-context-learning]]
- PPD / 事後予測分布 → [[bayesian-inference]]
- MCMC / Metropolis-Hastings / HMC / NUTS / 詳細釣り合い / typical set / 典型集合 / 重点サンプリング / Gibbs サンプリング / Metropolis-within-Gibbs / R-hat / Gelman–Rubin / blocking / 差分進化 / DE-MCMC / PyMC / 確率的プログラミング → [[markov-chain-monte-carlo]]
- SCM → [[structural-causal-model]]
- GP → [[gaussian-process]]
- GMM / ガウス混合モデル / 混合モデル / ソフトクラスタリング / 責任 responsibility / BIC / AIC / 情報量規準 → [[gaussian-mixture-model]]
- EM / 期待値最大化 / Expectation-Maximization / E-step / M-step / ELBO / 変分下界 / 座標上昇法 / coordinate ascent → [[expectation-maximization]]
- BayesOpt / BO / EI / KG / 期待改善 / 知識勾配 / 獲得関数 → [[bayesian-optimization]]
- TabPFN（v1/v2/2.5/3）/ TabPFN-3-Plus / Thinking → [[prior-data-fitted-networks]]（個別ページは作らず source に記載）
- TFM / 表形式基盤モデル / テスト時計算 → [[tabular-foundation-model]]
- TabICL / TabICLv2 / QASSMax → [[sources/2025-tabicl]] ・ [[sources/2026-tabicl-v2]]（別系統 TFM。概念は [[tabular-foundation-model]] / [[prior-data-fitted-networks]] / [[in-context-learning]]）
- Real-TabPFN / RealTabPFN-2.5 / 継続事前訓練 → [[sources/2025-real-tabpfn]]（概念は [[tabular-foundation-model]]）
- Mitra / 混合合成事前分布 / 汎化性行列 / 多様性・独自性 → [[sources/2025-mitra]]（概念は [[structural-causal-model]] / [[tabular-foundation-model]] / [[prior-data-fitted-networks]]）
- GIT-BO / 勾配誘導部分空間 / 高次元ベイズ最適化 / TabPFN サロゲート → [[sources/2025-git-bo]]（概念は [[bayesian-optimization]] / [[tabular-foundation-model]]）
- CausalPFN / 因果効果推定 / 処置効果 / CEPO / ATE / CATE / 潜在結果 / ignorability → [[sources/2025-causalpfn]]（概念は [[structural-causal-model]] / [[prior-data-fitted-networks]] / [[in-context-learning]] / [[bayesian-inference]]）
- Do-PFN / 条件付き介入分布 / CID / do 計算 / 介入 / do(t) / 介入効果推定 → [[sources/2025-do-pfn]]（概念は [[structural-causal-model]] / [[prior-data-fitted-networks]] / [[in-context-learning]] / [[bayesian-inference]]）
- Chronos-2 / 時系列基盤モデル / time series foundation model / group attention / 多変量予測 / 共変量つき予測 / クロス学習 / fev-bench / TabPFN-TS → [[sources/2025-chronos-2]]（概念は [[in-context-learning]] / [[tabular-foundation-model]]）
- nanoTabPFN / 軽量実装 / 教育的再実装 / minGPT / nanoGPT / bi-attention → [[sources/2025-nanotabpfn]]（概念は [[prior-data-fitted-networks]] / [[tabular-foundation-model]]）
- ShapPFN / 説明可能性 / Shapley 値 / SHAP / KernelSHAP / ViaSHAP / 特徴量帰属 / リアルタイム説明 → [[sources/2026-shappfn]]（概念は [[prior-data-fitted-networks]] / [[tabular-foundation-model]]）

## Questions

- [[questions/gaussian-process-intuitive-explainer]] — ガウス過程を数式を極力使わず図で説明する直感的解説記事（GP 入門 6 本を統合）
- [[questions/pfn-paper-and-gaussian-process]] — PFN 原典とガウス過程の関係（物差し／事前分布の素材／高速近似の 3 つ＋GP vs PFN 比較表、出力の仕組み・カーネル/ハイパラの扱いの深掘りと概念図）
- [[questions/tabpfn-tabicl-versions-mechanism]] — TabPFN/TabICL 各バージョンは「ガウス過程」をどう扱うか（GP 焼き込みは原典トイのみ／SCM が主役／GP は一材料・物差し・不使用へ。バージョン別表）
- [[questions/riemann-distribution-buckets]] — リーマン分布のバケット／prior-data とは（回帰を等確率バケットの分類に変える PFN の回帰ヘッド・図解）
- [[questions/evaluating-pfn-gp-approximation]] — PFN の GP 近似を定量評価する指標（KL/平均・信頼幅RMSE/被覆率/CRPS）とプロトコル
- [[questions/pfn-and-bayesian-optimization]] — PFN とベイズ最適化の関係（BO のサロゲートを GP→PFN に差し替える）初心者向け解説・図解
- [[questions/pfn-bayesian-inference-evaluation-settings]] — PFN のベイズ推論精度を評価できる問題設定（GP以外の例＋共役/グリッド/SCM/MCMC で自前評価）
- [[questions/posterior-vs-posterior-predictive]] — 事後分布と事後予測分布の違い（パラメータの分布 vs 次データの分布、比較表＋コイン/温度の例＋PFN接続）
- [[questions/mcmc-intuitive-explainer]] — MCMC を数式の気持ち（マス目/乱点/散歩者/オレンジの皮）で直感解説。2019-conceptual-intro-mcmc の噛み砕き
- [[questions/hmc-nuts-intuitive-explainer]] — HMC と NUTS を数式の気持ち（惑星と衛星/転がるボール/Uターン/温度調節）で直感解説。2017-conceptual-intro-hmc と NUTS 原典の噛み砕き
- [[questions/evaluating-pfn-with-known-gmm]] — 既知 GMM で PFN を評価する具体プロトコル（責任=ベイズ最適事後を物差しに・指標・有限文脈の注意・密度版）
