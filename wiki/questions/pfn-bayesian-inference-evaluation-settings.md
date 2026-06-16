---
type: question
asked: 2026-06-16
question: "PFN のベイズ推論精度を評価できる問題設定は？ GP 回帰以外の例と、自前で組める設定。"
sources_used:
  - "[[sources/2021-transformers-can-do-bayesian-inference]]"
  - "[[sources/2025-do-pfn]]"
  - "[[sources/2025-causalpfn]]"
  - "[[sources/2023-pfns4bo]]"
  - "[[sources/2026-tabicl-v2]]"
  - "[[sources/2025-mitra]]"
---

# PFN のベイズ推論精度を評価できる問題設定（GP 以外の例＋自前で組める設定）

> 問い: PFN（Prior-Data Fitted Network, 事前分布から合成したデータで一度だけ訓練し、推論時は重み更新なしに文脈内学習でベイズ推論を近似するネットワーク, [[prior-data-fitted-networks]]）のベイズ推論精度を評価したい。原典 [[sources/2021-transformers-can-do-bayesian-inference]] はガウス過程回帰（GP; Gaussian Process, [[gaussian-process]]）で評価していたが、(a) GP 以外の評価例はあるか、(b) 自前で評価できる問題設定はあるか。
>
> なお**評価“指標”**（KL・平均/信頼幅 RMSE・被覆率・CRPS とプロトコル）は [[questions/evaluating-pfn-gp-approximation]] に詳しい。本ページはその補完として、**「どんな問題設定なら“正解の事後”が手に入るか」**という設定側を整理する。

## 大原則：「真の PPD が分かる問題」を物差しにする

PFN が近似しようとしているのは **事後予測分布（PPD; posterior predictive distribution, 訓練データ $\mathcal{D}$ を条件にしたテスト点 $x$ のラベル分布）**（[[bayesian-inference]]）。「PFN のベイズ推論がどれだけ正確か」を測るには、**真の PPD（または真の事後）が分かる問題**を選び、PFN の出力をそれに突き合わせる。正解の手に入れ方は2段構え:

1. **閉形式で厳密に解ける**（GP・共役ベイズモデルなど）→ 真値そのものと比較できる。**最も望ましい**。
2. **閉形式は無いが、収束させた MCMC（マルコフ連鎖モンテカルロ。事後分布の標準的サンプリング近似。代表が NUTS = No-U-Turn Sampler）を“ゴールドスタンダード＝真値の代理”にする** → 適用範囲は最も広い。

GP が原典で選ばれたのは、「**固定ハイパーパラメータの GP は PPD が閉形式のガウスで厳密に出る**数少ない例」だから（[[questions/pfn-paper-and-gaussian-process]] 関係1「物差し」）。この性質さえ満たせば、GP でなくても物差しになる。理論的にも、PFN の訓練損失（Prior-Data NLL）の最小化は真の PPD への**前向き KL ダイバージェンス（2分布の非対称な近さ）の最小化**に一致する（原典 洞察1・系1.1）ので、「真の事後との KL／NLL」が精度の最も自然な尺度になる。

## (a) wiki 内にある「GP 以外」の評価例

原典自体が、固定 GP のほかに次の系統で PFN のベイズ推論精度を評価している:

| 評価対象（prior） | 正解（物差し）の出し方 | 何で測ったか |
|---|---|---|
| 固定ハイパラ GP | 閉形式の厳密事後（ガウス） | 平均・95%信頼区間の重ね描き（図3）、Prior-Data NLL（図4a） |
| **ハイパー事前分布つき GP**（カーネルのハイパラに分布を置く＝解析的に解けない） | MLE-II / NUTS を真値の代理に | 真 PPD への近さ＋速度（NUTS比 1000〜8000倍速で同等以上） |
| **ベイズニューラルネット（BNN, 重みを分布として扱う NN）** | SVI（確率的変分推論）/ NUTS | 同上（SVI比1000倍・NUTS比10000倍速） |
| 表形式分類 | （厳密事後が無いので）保留ラベルへの当てはまり | ECE（Expected Calibration Error, 予測確率と実正解率のズレ）0.025 |

ここで重要なのは、**厳密事後が無い prior（GP-hyperprior・BNN）では「十分収束させた MCMC（NUTS）を真値の代理」にして比べている**こと。閉形式で解ける必要は必ずしもない（＝上の大原則の段構え2）。

さらに wiki には**因果推論を題材にした“正解の分かる”評価**がある:

- **Do-PFN [[sources/2025-do-pfn]] / CausalPFN [[sources/2025-causalpfn]]**: 合成した構造的因果モデル（SCM; Structural Causal Model, DAG＋構造方程式, [[structural-causal-model]]）からデータを生成するので、**真の条件付き介入分布 $p(y\mid do(t),x)$ や真の処置効果（CATE）が既知**。これを物差しに、真の因果グラフを使えるゴールドスタンダード（DoWhy）と比較する。理論面も「損失最小化＝真の分布への前向き KL 最小化」を証明（Do-PFN 命題1）しており、原典の評価哲学を因果へ一般化した形。
- **PFNs4BO [[sources/2023-pfns4bo]]**: 保留点の**尤度（log score）**で経験ベイズ GP と PFN を比較（付録 I・表8）、BO ステップ上で厳密 GP の平均/EI と PFN を重ねて評価（図11・12）。
- **TabICLv2 [[sources/2026-tabicl-v2]] / Mitra [[sources/2025-mitra]]**: 回帰の評価に **CRPS（Continuous Ranked Probability Score, 任意形の予測分布に使える適正スコア則）** を採用。厳密事後が無くても予測分布全体の質を測れる。

## (b) 自前で組める「正解の分かる」問題設定

厳密 PPD が出る族は GP だけではない。難易度順に挙げる（測り方は [[questions/evaluating-pfn-gp-approximation]] をそのまま流用できる）。

**(A) 共役ベイズモデル — 閉形式 PPD が出る、最もクリーンな物差し**

「事前分布と尤度が共役（conjugate, 事後が事前と同じ分布族になる組）」なら事後も PPD も手計算で出る。GP より実装が圧倒的に軽い。

- **ベイズ線形回帰（ガウス事前）— 第一推奨**。PPD が閉形式（観測ノイズ分散が既知ならガウス、未知なら Student-t 分布）。実は**線形カーネルの GP の特殊ケース**にあたり、しかも任意次元・任意サンプル数へ軽く拡張でき、数式も実装も数行。「GP の次に試す物差し」として最適。
- **Beta–Bernoulli（コイン投げ）／ Normal–Normal（既知分散での平均推定）**：事後・PPD が手計算で出る。低次元なので PFN の出力分布を**真の事後と直接重ねて**目視＋指標で検証できる。この2例の手計算は [[sources/2024-bayesian-inference-step-by-step]] が具体的（コイン投げ→Beta(8,4)、正規平均→𝒩(68.121, 0.869)）。
- **Gamma–Poisson（カウント）／ Dirichlet–Multinomial（カテゴリ）**：カウント・カテゴリ版の厳密事後。**分類タスクでの“ベイズ最適”物差し**を作れる。

**(B) 分類の「ベイズ最適事後」を作る**

既知パラメータの**生成モデル**（クラス条件をガウス混合や線形判別で固定）からデータを作ると、各点の**ベイズ最適クラス事後確率 $p(y\mid x)$ が解析的に計算できる**。PFN のクラス確率をこの最適値に突き合わせれば、回帰における GP と同じ「厳密な物差し」が分類で得られる。原典が表形式分類では ECE（較正）止まりだった部分を、**厳密な事後確率比較**にできるのが利点。

**(C) 低次元で数値積分（グリッド）→ “準厳密”**

共役でなくても、1〜2 次元の**ベイズロジスティック回帰**や小さな混合モデルなら、パラメータ空間をグリッドで密に積分して事後・PPD を実質厳密に出せる。「閉形式は無いが次元が低い」問題で物差しを作る常套手段。

**(D) 合成 SCM で真の効果を既知にする（因果版・Do-PFN 流）**

自分で SCM を書けば真の介入分布・CATE が分かるので、PFN（やその因果拡張）の**因果推定の正確さ**を厳密評価できる。観測 PPD とは別物（介入効果）を測りたいときの設定。

**(E) 収束 NUTS をゴールドスタンダードに（原典の GP-hyperprior/BNN 流）**

閉形式が一切無い任意の prior（BNN など）でも、**十分収束させた NUTS** を真値の代理にして KL/CRPS/NLL で比較できる。厳密ではないが適用範囲が最も広い、最後の手段。MCMC（マルコフ連鎖モンテカルロ）と NUTS の仕組み自体は [[sources/2021-mcmc-explained]]、その収束・一貫性・有効サンプルサイズ（＝「どれだけ回せば物差しとして信頼できるか」）の厳密な議論は [[sources/2019-conceptual-intro-mcmc]] を参照。

## おすすめの次の一歩

- **回帰**: まず **(A) ベイズ線形回帰**。GP と同じ「閉形式の厳密事後」が得られて実装が最も軽く、原典のプロトコル（データ点数 $n$ を変えながら平均 RMSE・σ の RMSE・点ごと KL・被覆率を見る）をそっくり適用できる。
- **分類**: **(B) 既知パラメータのガウス混合のベイズ最適事後**。クラス確率を厳密な物差しと比較でき、ECE より踏み込んだ評価になる。
- いずれも「厳密事後が無い保険」として **保留点 NLL** と **CRPS** を併記しておくと、リーマン分布（PFN の回帰出力＝バケット化ヒストグラム, [[questions/riemann-distribution-buckets]]）の非ガウス性まで拾える。

## 用語と略称

- **PPD** = Posterior Predictive Distribution（事後予測分布。PFN の近似対象）→ [[bayesian-inference]]
- **共役（conjugate）** = 事前分布と尤度の組で、事後が事前と同じ分布族になる性質。事後・PPD が閉形式で出る
- **ベイズ線形回帰** = 線形モデルの係数にガウス事前を置いたベイズ回帰。PPD が閉形式（線形カーネル GP の特殊ケース）
- **ベイズ最適事後** = 真の生成モデルが既知のとき解析的に出る最適なクラス事後確率 $p(y\mid x)$
- **NUTS** = No-U-Turn Sampler（MCMC の代表。収束させれば真の事後の代理になる）
- **CRPS** = Continuous Ranked Probability Score（任意形の予測分布に使える適正スコア則）
- **CATE** = Conditional Average Treatment Effect（条件付き平均処置効果）→ [[sources/2025-do-pfn]]
- **ECE** = Expected Calibration Error（予測確率と実正解率のズレ）
- **前向き KL** = 真の分布 $p$ から近似 $q$ への KL$(p\Vert q)$。PFN の損失最小化が一致する量（系1.1）

## 関連ページ

- [[questions/evaluating-pfn-gp-approximation]] — 本ページの対になる「評価指標とプロトコル」（KL・RMSE・被覆率・CRPS）
- [[sources/2021-transformers-can-do-bayesian-inference]] — 固定 GP・GP-hyperprior・BNN・分類で PFN を評価した原典
- [[questions/pfn-paper-and-gaussian-process]] — GP が「物差し」になる仕組み・PFN の出力（平均/分散）の出し方
- [[gaussian-process]] — 厳密事後が閉形式で出る代表的な物差し
- [[bayesian-inference]] — 近似対象である PPD と償却推論
- [[sources/2025-do-pfn]] / [[sources/2025-causalpfn]] — 合成 SCM の真の介入分布/CATE を物差しにした因果版評価
- [[sources/2023-pfns4bo]] — 保留点尤度で経験ベイズ GP と比較
- [[questions/riemann-distribution-buckets]] — PFN の回帰出力（リーマン分布）から平均・分散・分位点を得る
