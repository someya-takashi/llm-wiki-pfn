---
type: question
asked: 2026-06-17
question: "既知ガウス混合モデル（GMM）を使って PFN のベイズ推論精度をどう評価するか、具体的な手順を知りたい。"
sources_used:
  - "[[sources/2021-transformers-can-do-bayesian-inference]]"
  - "[[sources/2020-gmm-overview]]"
  - "[[sources/2024-gmm-em-clustering]]"
---

# 既知 GMM で PFN を評価する具体プロトコル

> 問い: [[questions/pfn-bayesian-inference-evaluation-settings]] の「(B) 分類のベイズ最適事後（既知 GMM）」を、実際にどう実行するのか具体的に知りたい。
>
> 要点: **パラメータを自分で決めたガウス混合モデル（GMM; [[gaussian-mixture-model]]）は、各点の正解クラス事後が数式で書ける**ので、PFN（[[prior-data-fitted-networks]]）の出力クラス確率を「厳密な正解」と突き合わせられる。GP 回帰で「固定 GP の閉形式ガウス事後」を物差しにした（[[questions/evaluating-pfn-gp-approximation]]）のと同じことを、**分類**で行う。

## なぜ GMM が「正解の物差し」になるのか

成分 $k$（混合重み $\pi_k$・平均 $\mu_k$・共分散 $\Sigma_k$ がすべて既知）からなる GMM を考え、**「その点 $x$ を生成した成分の番号 $y=k$」をクラスラベル**とみなす。するとベイズ則から、各点の**最適なクラス事後**が閉形式で出る:

$$
p(y=k\mid x)=\frac{\pi_k\,\mathcal N(x\mid\mu_k,\Sigma_k)}{\sum_{j}\pi_j\,\mathcal N(x\mid\mu_j,\Sigma_j)}.
$$

これは [[gaussian-mixture-model]] / [[expectation-maximization]] の **E-step が出す「責任 $r_{ik}$」そのもの**であり、しかも**ベイズ最適**（どんな分類器もこれ以上正確な確率は出せない）。真のパラメータを自分で決めたのだから、この事後は**厳密・閉形式**で手に入る。GP が回帰で果たした「物差し」の役割を、分類で果たす。

## 具体プロトコル（7手順）

1. **既知 GMM を1つ決める**。例: 2次元で $K=3$ 成分、$\pi,\mu,\Sigma$ を指定。**わざと重なり（overlap）を作る**のが肝心——よく分離していると正解事後が 0/1 に潰れて退屈で、確率較正をテストできない。重なる領域でこそ「0.7 対 0.3」のような中間の事後が出て、PFN の較正の良し悪しが見える。
2. **ラベル付きデータを生成**。点 $x_n$ を GMM からサンプルし、**どの成分から出たか $y_n$ を記録**（祖先サンプリング: まず $\pi$ で成分を選び、次にそのガウスから引く、[[sources/2020-gmm-overview]]）。これで「正解の事後が全点で分かる分類データセット」ができる。
3. **文脈（train）とクエリ（test）に分割**。
4. **物差しを計算**。各クエリ $x_*$ で、上の式に**真のパラメータ**を入れてベイズ最適事後ベクトル $[p(y{=}1\mid x_*),\dots,p(y{=}K\mid x_*)]$ を出す。
5. **PFN を走らせる**。TabPFN に「ラベル付き文脈＋クエリ $x_*$」を入れ、予測クラス確率ベクトル $\hat p(y\mid x_*)$ を読む。
6. **突き合わせる**（クエリ平均）:
   - **点ごとの KL$(\text{ベイズ最適}\,\|\,\text{PFN})$** … 2つのカテゴリ分布のズレを1数値で。**主指標**（[[questions/evaluating-pfn-gp-approximation]] の点ごと KL の分類版）。理論的にも PFN の損失＝真の事後予測分布（PPD）への**前向き KL 最小化**（[[sources/2021-transformers-can-do-bayesian-inference]] 系1.1）なので、KL が最も自然な尺度。
   - **全変動距離（TV）/ L1** … 確率ベクトル同士の差。KL の補完。
   - **較正（reliability diagram / ECE）** … 通常の ECE が「予測確率 vs 実ラベル(0/1)」なのに対し、ここは**予測確率 vs 真の事後**と比べられる＝より強い検証。
   - **Brier スコア / NLL と、ベイズ誤り率（Bayes error）を“床”に** … ベイズ誤り $\int\big(1-\max_k p(y{=}k\mid x)\big)\,p(x)\,dx$ は原理的に到達できない最小誤差。PFN の精度をこの床と比べる。
7. **条件を振る**: 成分の**重なり具合**、**文脈サイズ $n$**（増やすほどベイズ最適に近づくか）、$K$、共分散の形（球状／楕円／相関あり、[[sources/2024-gmm-em-clustering]] の「適応的共分散」）。

## 2つの評価モード

| モード | 何をするか | 何が分かるか |
|---|---|---|
| **① 既製 TabPFN の較正チェック**（手軽・第一歩）| 市販の TabPFN に GMM 生成データを食わせ、出力確率を上の物差しと比較 | 汎用の表形式基盤モデル（[[tabular-foundation-model]]）のクラス事後が「正解が分かる問題」でどれだけ較正されているか |
| **② GMM 事前分布で PFN を訓練**（統制実験・原典の GP 実験に最も近い）| ランダムな GMM を prior（データ生成器）にして PFN を訓練→未知 GMM で評価 | 「PFN が GMM のベイズ推論を学べるか」を直接検証。原典が GP 事前分布で PFN を訓練したのと同じ枠組み |

## 重要な注意（正直なところ）

- **「既知パラメータの責任」は n→∞ のベイズ最適**である。一方 PFN は**有限文脈**しか見ないので、本来近似しているのは「**有限データ＋事前分布のもとでの事後予測**」で、これは普通**ベイズ最適より控えめ（自信過小）**で、文脈を増やすほどベイズ最適に近づく。だから実務では:
  - PFN に**大きめの文脈**を与えて $\hat p\approx$ ベイズ最適にしてから比較する、または
  - **傾向**を見る（$n$ を増やすと PFN がベイズ最適へ寄り、ベイズ誤りの床は割らない）。
  - 厳密に「有限データの正解」を出すには、GMM パラメータの事後にわたって積分した事後予測が要るが、それ自体が扱いにくい（EM/MCMC が近似する当の対象）。だから**閉形式で確実に手に入る物差しは “既知パラメータのベイズ最適” の方**。
- **GP との違い**（混同注意）: GP は「有限データ $D$ を条件にした**厳密な回帰 PPD**（閉形式ガウス）」を出すので、**有限データの物差し**として完璧。GMM が出すのは「**既知パラメータのベイズ最適な分類事後**」で、これは**n→∞／ベイズ最適の物差し**。両方とも有効だが、**GP は「PFN は厳密な事後予測に一致するか」、GMM は「PFN のクラス事後はベイズ最適へ向かって較正されているか」**と、少し違う問いに答える。

## ボーナス: 密度推定版の物差し（教師なし）

TabPFN v2 の生成・密度推定機能（[[tabular-foundation-model]]）を使うなら、**教師なし**でも GMM を物差しにできる。真の密度 $p(x)=\sum_k\pi_k\,\mathcal N(x\mid\mu_k,\Sigma_k)$ は数式で書けるので、PFN の推定密度 $\hat p(x)$ と**保留点の対数尤度や KL** で比べられる。つまり **1つの既知 GMM から、クラス事後（教師あり）＋密度（教師なし）の2種類の物差し**が取れる。

## なぜ KL を主指標にするか（理論的根拠）

PFN 原典（[[sources/2021-transformers-can-do-bayesian-inference]]）は、PFN の訓練損失（Prior-Data NLL）の最小化が**真の事後予測分布への前向き KL ダイバージェンスの最小化**に一致することを示した（系1.1）。つまり PFN は「KL の意味で真の事後（予測）に寄る」よう作られている。だから評価でも、PFN 出力と既知 GMM のベイズ最適事後の **KL** を測るのが、PFN の設計目標と整合した最も自然な指標になる。

## 用語と略称

- **GMM** = Gaussian Mixture Model（ガウス混合モデル）→ [[gaussian-mixture-model]]
- **責任（responsibility）$r_{ik}$** = 点 $x_i$ が成分 $k$ から来た事後確率＝E-step の出力。既知パラメータならベイズ最適クラス事後
- **ベイズ最適事後／ベイズ誤り率** = 真の生成モデルが既知のときの最良のクラス事後／その分類器でも避けられない最小誤差（物差しの“床”）
- **較正（calibration）/ ECE** = 予測確率と実際の正しさの整合 / その期待誤差。ここでは「真の事後」と比べる強い版
- **KL / 前向き KL** = 2分布の非対称な近さ。PFN 損失が最小化する量（真 PPD への前向き KL）
- **TV（全変動距離）** = 2つの確率分布の最大の差
- **PPD** = Posterior Predictive Distribution（事後予測分布。PFN の近似対象）→ [[bayesian-inference]]
- **祖先サンプリング** = 親（成分 $\pi$）→子（ガウス）の順にサンプル（GMM からのデータ生成）

## 関連ページ

- [[questions/pfn-bayesian-inference-evaluation-settings]] — 親ページ。本ページはその (B) の実行可能な詳細版
- [[questions/evaluating-pfn-gp-approximation]] — 評価指標とプロトコル（KL・RMSE・被覆率・CRPS）。指標はここを流用
- [[gaussian-mixture-model]] — 責任＝ベイズ最適クラス事後が閉形式で出る根拠
- [[expectation-maximization]] — 責任は E-step の出力（潜在変数の事後）
- [[bayesian-inference]] — 近似対象の事後予測分布（PPD）
- [[prior-data-fitted-networks]] — 評価対象の PFN
- [[sources/2021-transformers-can-do-bayesian-inference]] — 物差し評価の原典（損失＝前向き KL）。GP 事前分布で PFN を訓練した枠組み
- [[tabular-foundation-model]] — 密度推定機能（教師なしの GMM 物差し）
- [[sources/2020-gmm-overview]] / [[sources/2024-gmm-em-clustering]] — GMM の責任・祖先サンプリング・共分散
