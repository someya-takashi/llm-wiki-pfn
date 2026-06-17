---
type: source
source_path: raw/articles/A Conceptual Introduction to Markov Chain Monte Carlo Methods.md
source_kind: paper
title: "A Conceptual Introduction to Markov Chain Monte Carlo Methods"
authors: [Joshua S. Speagle]
year: 2019
venue: "arXiv 1909.12313"
ingested: 2026-06-16
tags: [mcmc, monte-carlo, importance-sampling, metropolis-hastings, effective-sample-size, curse-of-dimensionality, typical-set, bayesian-inference, posterior-predictive-distribution]
translation: "[[translations/2019-conceptual-intro-mcmc]]"
---

# A Conceptual Introduction to Markov Chain Monte Carlo Methods（MCMC の概念的入門）

> 原典: [[translations/2019-conceptual-intro-mcmc]] ・ `raw/articles/A Conceptual Introduction to Markov Chain Monte Carlo Methods.md`（arXiv 1909.12313, ar5iv 版）
> 著者・年・媒体: Joshua S. Speagle（Harvard）/ 2019（arXiv 1909.12313）

## 一言まとめ

**MCMC（Markov Chain Monte Carlo, マルコフ連鎖モンテカルロ）を、「ベイズ推論で本当に欲しいのは事後分布そのものでなく事後分布上の積分（期待値）だ」という視点から、グリッド近似 → モンテカルロ → 重点サンプリング → MCMC という一本の連続体として導出する、厳密かつ包括的な入門論文。** 直前に取り込んだ [[sources/2021-mcmc-explained]]（informal なブログ）の理論的な兄貴分にあたり、有効サンプルサイズ（ESS）・自己相関時間・**次元の呪い・typical set（典型集合）**・アンサンブル法（emcee の stretch move まで）を扱う。PFN（[[prior-data-fitted-networks]]）の文脈では、これは **PFN が償却で置き換える「近似推論」の最も精密な解剖図**であり、「なぜデータセットごとに MCMC を回すのが高コストなのか」を原理から説明する。著者 Speagle は入れ子サンプリング実装 `dynesty` の作者。

## 背景と問題意識：欲しいのは「事後そのもの」でなく「事後上の積分」

ベイズ推論は事後分布 $\mathcal{P}(\boldsymbol\Theta)=\mathcal{L}(\boldsymbol\Theta)\pi(\boldsymbol\Theta)/\mathcal{Z}$（尤度×事前÷証拠）を与えるが、本稿の核心命題は **「実用上ほぼ常に欲しいのは事後分布そのものではなく、事後分布上の積分＝期待値 $\mathbb E_{\mathcal P}[f(\boldsymbol\Theta)]$ だ」**（§3）という点。点推定・信用区間・予測・モデル比較のすべてが、事後上の積分に帰着する:

- **点推定**＝損失関数の期待損失最小化。**二乗損失→平均、絶対損失→中央値、破滅的（catastrophic）損失→最頻値**、と損失の選び方が推定量を決める（§3.1）。
- **信用区間（credible interval）**＝事後の X% を含む領域（閾値版／パーセンタイル版）（§3.2）。
- **事後予測（posterior predictive）** $P(\tilde{\mathbf D}\mid\mathbf D)=\int P(\tilde{\mathbf D}\mid\boldsymbol\Theta)P(\boldsymbol\Theta\mid\mathbf D)\,d\boldsymbol\Theta=\mathbb E_{\mathcal P}[\tilde{\mathcal L}(\boldsymbol\Theta)]$（§3.3）——**これが PFN の近似ターゲット PPD そのもの**。
- **モデル比較**＝証拠の比＝ベイズ因子（§3.4）。証拠 $\mathcal Z=\int\tilde{\mathcal P}(\boldsymbol\Theta)\,d\boldsymbol\Theta$ も事後上の積分。

「$\mathcal P(\boldsymbol\Theta)$ の推定が極めて貧弱でも $\mathbb E_{\mathcal P}[f]$ の優れた推定は得られることが多い」——この区別が本稿全体の指針になる。

## 提案手法 / 主張：グリッド → モンテカルロ → 重点サンプリング → MCMC

**①グリッド近似と次元の呪い（§4）.** 事後上の積分をリーマン和（離散グリッド）で近似する。だがグリッド点数は次元 $d$ に対し $k^d$ で**指数増（次元の呪い, curse of dimensionality）**。さらにグリッド点の置き方で**有効サンプルサイズ（ESS; effective sample size, $n_{\rm eff}=(\sum w_i)^2/\sum w_i^2$）**が大きく変わり、誤差は $n$ でなく $n_{\rm eff}$ で決まる。グリッドは**収束（convergence, ある値に収束）するが一貫（consistency, 真値に収束）しない**ことがある（範囲外にある事後質量を見落とす）。

**②グリッド→モンテカルロ→重点サンプリング（§5）.** グリッド間隔を密度 $\mathcal Q(\boldsymbol\Theta)\propto 1/\Delta\boldsymbol\Theta$ とみなし無限解像度の極限を取ると、グリッドは連続分布 $\mathcal Q$（**提案分布, proposal distribution**）に等価になる。これにより期待値が $\mathcal Q$ からの乱数サンプルで推定できる＝**重点サンプリング（importance sampling）**。重み $\tilde w_i=\tilde{\mathcal P}(\boldsymbol\Theta_i)/\mathcal Q(\boldsymbol\Theta_i)$。**提案を事前分布にすると重み＝尤度**、**提案を事後分布にすると重み＝定数 $\mathcal Z$（＝ESS 最大 $n_{\rm eff}=n$ で最適）**。この「事後そのものからサンプルしたい」という結論が MCMC の動機。

**③MCMC（§6）.** MCMC は**重みを一定にする＝事後分布 $\mathcal P$ に比例した密度でサンプルを生成する**手法。サンプル密度 $\rho(\boldsymbol\Theta)\to\mathcal P(\boldsymbol\Theta)$ なので、期待値は単純な標本平均、証拠 $\mathcal Z$ も推定可能。**詳細釣り合い（detailed balance）** $P(\boldsymbol\Theta_{i+1}\mid\boldsymbol\Theta_i)\mathcal P(\boldsymbol\Theta_i)=P(\boldsymbol\Theta_i\mid\boldsymbol\Theta_{i+1})\mathcal P(\boldsymbol\Theta_{i+1})$ を「提案 $\mathcal Q$ ＋受容 $T$」に分け、**Metropolis 基準** $T=\min[1,\frac{\mathcal P(\boldsymbol\Theta_{i+1})}{\mathcal P(\boldsymbol\Theta_i)}\frac{\mathcal Q(\boldsymbol\Theta_i\mid\boldsymbol\Theta_{i+1})}{\mathcal Q(\boldsymbol\Theta_{i+1}\mid\boldsymbol\Theta_i)}]$ がこれを満たす（**Metropolis–Hastings アルゴリズム**）。

> **本稿が正す2つの誤解（§6）**: (1)「MCMC では正規化定数 $\mathcal Z$ を推定できない」は誤り——$\rho$ を介して一貫推定できる（収束は遅いが）。(2)「MCMC の主目的は事後の"近似/探索"」も誤り——重点サンプリングの系譜から見れば**主目的は期待値推定**であり、事後の近似は $\mathcal Z$ 推定にしか効かない。

**④ESS と自己相関時間（§6.2）.** MCMC のサンプルは隣と相関するので、**自己相関時間 $\tau$** を入れた $n'_{\rm eff}=n/(1+\tau)$ が効く。相関が強い（受容率が低い）ほど ESS が落ち、iid な重点サンプリングに負けることすらある。

**⑤次元の呪いと typical set（§7）.** 事後を「近似」するには領域分割で $k^d$ サンプル要＝再び次元の呪い。さらに **typical set（典型集合）**: 高次元では**事後質量＝密度×体積が、半径 $r_{\rm peak}=\sqrt{d-1}\,\sigma$（$\approx\sqrt d\,\sigma$）の薄い殻に集中**する。サンプルの大半は密度ピークでなくこの殻に居るため、(a) MCMC はピーク領域の特徴づけに非効率、(b) 高次元では提案が殻から外れて**受容率が指数的に低下**（提案幅を $\gamma\propto1/\sqrt d$ で縮める必要）。

**⑥トイ問題とアンサンブル法（§8）.** $d$ 次元ガウスで上記を実証。ガウス提案は $\gamma=\delta/\sqrt d$ で受容率を一定化（$\tau\approx 3d$ の線形に改善）。共分散を自動調整する**アンサンブル法**: ensemble Gaussian・**DE-MCMC**（3粒子の差分）・**affine-invariant stretch move**（=emcee, 2粒子）。stretch move は $\gamma^{d-1}$ 項のため高次元で苦戦する。

## 実験結果と知見

トイの $d$ 次元ガウスで、(i) 固定提案 $\gamma=\sqrt2$ は次元増で急速に行き詰まる一方、適応提案 $\gamma=2.5/\sqrt d$（受容率 25% 狙い）は安定（図13）、(ii) アンサンブル3手法のうち提案幅を縮められる ensemble Gaussian と DE-MCMC は高次元でも効率的、stretch move は指数的に劣化（図15）。理論予測（受容率・$\tau$・ESS の次元依存）が数値と一致することを示す。

## 限界・批判的視点

- **概念理解に焦点**を明言し、HMC（ハミルトニアンモンテカルロ）/ NUTS・入れ子サンプリング・収束診断・$\hat\tau$ 推定法などの**具体アルゴリズムの詳細は範囲外**（文献参照に留める）。
- astro-stats（天体物理）の文脈で書かれ、機械学習・PFN への言及はない（接続は本 wiki 側で行う）。
- 多くの結果は $d$ 次元ガウスというトイに基づく近似則で、一般の事後への外挿には注意。

## PFN との接続（なぜこの wiki に置くか）

- **事後予測 §3.3 ＝ PFN の近似ターゲット PPD**。本稿の $P(\tilde{\mathbf D}\mid\mathbf D)=\int P(\tilde{\mathbf D}\mid\boldsymbol\Theta)P(\boldsymbol\Theta\mid\mathbf D)d\boldsymbol\Theta$ は [[bayesian-inference]] の事後予測分布そのもの。[[prior-data-fitted-networks]]（PFN）は、これを MCMC で1データセットずつ積分する代わりに、**Transformer の順伝播1回で出力**する。
- **「欲しいのは事後上の積分（期待値）」＝ PFN の出力の正体**。本稿の核心命題は、PFN が事後パラメータを陽に持たず**期待値（PPD）を直接出す**設計と完全に符合する。PFN は「事後を近似してから積分」ではなく「積分（期待値）を償却して直接出す」。
- **次元の呪い・typical set・自己相関＝ per-dataset サンプリングが高コストな理由**。本稿が精密に示す「高次元で受容率が指数低下し、収束に $k^d$ サンプル要る」コストこそ、PFN が**事前訓練に一度だけ償却**して回避する相手。PFN は推論時に近似計算ゼロ。
- **PFN 原典の速度比較相手の理論的裏付け**。[[sources/2021-transformers-can-do-bayesian-inference]] は扱いにくい事後を **NUTS（HMC ベースの MCMC、本稿の MH を高次元向けに洗練したもの）** で近似する従来法と比べ、PFN が NUTS 比 **1000〜10000 倍速**で同等以上を達成すると示した。本稿はその「遅い基準」の内部メカニズム（なぜ遅いか）を与える。
- **[[sources/2021-mcmc-explained]] の厳密版**であり、[[questions/pfn-bayesian-inference-evaluation-settings]] の「(E) 収束 NUTS をゴールドスタンダードに」の理論的根拠（収束・一貫性・ESS）を提供する。

## 用語と略称

- **MCMC** = Markov Chain Monte Carlo（マルコフ連鎖モンテカルロ）
- **モンテカルロ推定** = 乱数サンプルの平均で積分・期待値を近似
- **重点サンプリング（importance sampling）** = 提案分布 $\mathcal Q$ から引いて重み $\tilde{\mathcal P}/\mathcal Q$ で補正
- **提案分布（proposal distribution）$\mathcal Q$** = サンプルを引く（または次候補を提案する）分布
- **詳細釣り合い（detailed balance）** = 連鎖の定常分布を事後 $\mathcal P$ に一致させる可逆性条件
- **Metropolis–Hastings / Metropolis 基準** = 提案＋受容確率 $\min[1,\cdot]$ で詳細釣り合いを満たす MCMC
- **ESS（有効サンプルサイズ）** = $(\sum w_i)^2/\sum w_i^2$。相関を入れると $n/(1+\tau)$
- **自己相関時間 $\tau$** = サンプルが無相関になるまでの実効的な遅れ
- **証拠 / 周辺尤度 $\mathcal Z$** = $P(\mathbf D\mid M)=\int\mathcal L\pi\,d\boldsymbol\Theta$。ベイズ因子の素
- **事後予測（posterior predictive）/ PPD** = $\int P(\tilde{\mathbf D}\mid\boldsymbol\Theta)P(\boldsymbol\Theta\mid\mathbf D)d\boldsymbol\Theta$ → [[bayesian-inference]]
- **次元の呪い（curse of dimensionality）** = グリッド/領域数が $k^d$ で指数増
- **typical set（典型集合）** = 事後質量（密度×体積）が集中する半径 $\approx\sqrt d\,\sigma$ の薄い殻
- **burn-in** = 定常状態に達する前の初期サンプルを捨てる区間
- **HMC / NUTS** = Hamiltonian Monte Carlo / No-U-Turn Sampler（高次元向けの MCMC。PFN 原典の比較相手）
- **DE-MCMC / stretch move** = アンサンブル MCMC（後者は emcee の中核）
- **amortize（償却）** = 事前の一括計算で個別推論を安価にすること（PFN の鍵）

## 関連ページ

- [[markov-chain-monte-carlo]] — MCMC の概念ページ（本稿はその厳密・包括版の一次資料）
- [[bayesian-inference]] — 事後・PPD・近似推論の概念ハブ。本稿は「事後上の積分」視点を精密化
- [[prior-data-fitted-networks]] — MCMC の反復サンプリングを順伝播1回に償却する枠組み
- [[sources/2021-mcmc-explained]] — 同じ MCMC の informal な入門。本稿はその厳密・包括版
- [[sources/2018-mcmc-simple-introduction]] — MCMC の応用・実務寄り入門（Gibbs・DE・blocking・R-hat、SDT 応用）
- [[sources/2020-understanding-bayesian-inference]] — 近似推論（MCMC/VI）を高レベルで導入した入門
- [[sources/2021-transformers-can-do-bayesian-inference]] — PFN 原典。NUTS（MCMC）比 1000〜10000 倍速を実証
- [[questions/pfn-bayesian-inference-evaluation-settings]] — 「収束 NUTS を物差しに」＝本稿の収束・一貫性・ESS が裏付け
- [[translations/2019-conceptual-intro-mcmc]] — 本論文の全文翻訳（本文§1〜9＋演習）
