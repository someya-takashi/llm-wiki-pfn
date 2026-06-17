---
type: source
source_path: raw/articles/A simple introduction to Markov Chain Monte–Carlo sampling.md
source_kind: paper
title: "A simple introduction to Markov Chain Monte–Carlo sampling"
authors: [Don van Ravenzwaaij, Pete Cassey, Scott D. Brown]
year: 2018
venue: "Psychonomic Bulletin & Review (2018)"
ingested: 2026-06-17
tags: [mcmc, metropolis-hastings, gibbs-sampling, differential-evolution, burn-in, convergence, bayesian-inference, tutorial]
translation: "[[translations/2018-mcmc-simple-introduction]]"
---

# A simple introduction to Markov Chain Monte–Carlo sampling（MCMC の平易な入門）

> 原典: [[translations/2018-mcmc-simple-introduction]] ・ `raw/articles/A simple introduction to Markov Chain Monte–Carlo sampling.md`（Psychonomic Bulletin & Review 2018, PMC5862921）
> 著者・年・媒体: Don van Ravenzwaaij, Pete Cassey, Scott D. Brown / 2018（Psychonomic Bulletin & Review）

## 一言まとめ

**MCMC（Markov Chain Monte–Carlo, マルコフ連鎖モンテカルロ）を、認知科学者・初学者向けに「手を動かせば数行のコードで動く」レベルまで噛み砕いたチュートリアル。** Metropolis アルゴリズムを1観測の母平均推定で step-by-step に示し、**実務上つまずく点（提案分布のチューニング・棄却率・局所最大・収束/burn-in の判断・R-hat）** をていねいに扱う。MCMC 三部作の3本目として、既存の [[sources/2021-mcmc-explained]]（informal）・[[sources/2019-conceptual-intro-mcmc]]（厳密）に無い **ギブスサンプリング（Metropolis-within-Gibbs）・blocking・差分進化（DE-MCMC）の直感図・収束診断** を補う。[[markov-chain-monte-carlo]] の「応用＆実務」面のリファレンス。

## 背景と問題意識

ベイズ推論（[[bayesian-inference]]）では、データ $D$ から関心パラメータ $\mu$ の事後分布 $p(\mu\mid D)\propto p(D\mid\mu)\,p(\mu)$（尤度×事前）を得たい。だが多くの場合、事後分布の解析的表現が手に入らない。そこで MCMC で**事後分布から一連のサンプルを引き、その平均・範囲を調べる**。MCMC の強みは「**密度（尤度×事前の比）さえ計算できれば、正規化されていない事後からでもサンプルできる**」こと。名前どおり「**モンテカルロ**（乱数サンプルの平均で量を近似）＋**マルコフ連鎖**（次のサンプルが直前のサンプルだけに依存する逐次生成）」の合成。心理学では記憶・信号検出・意思決定など広範に応用されてきた。

## 提案手法 / 主張（MCMC の道具立て）

**① Metropolis アルゴリズム（提案＋受容）.** 例「1人の得点100、母標準偏差15から母平均を推定」（事後は $\mathcal N(100,15)$）。初期値110から、(1) 対称な提案分布 $\mathcal N(0,5)$ でノイズを足して候補を作り、(2) 事後の高さを現在値と比べ、**候補の方が高ければ受容／低ければ「高さの比」の確率で受容**（例: 1/5の高さなら20%で受容）。これを繰り返すと500サンプルで真の分布の本質（標準偏差16.96 ≈ 真値15）を捉える。

**② 限界と実務.** (a) 提案分布は**対称**でなければならない（非対称なら受容ステップを直す＝**Metropolis–Hastings**）。(b) 提案幅は**チューニングパラメータ**: 広すぎ（σ=50）ると棄却率が跳ね上がり、狭すぎ（σ=1）ると収束が遅く**局所最大**に詰まる。自動調整アルゴリズムで緩和できる。(c) **収束と burn-in**: 悪い初期値（250 や 650）からだと、真の事後に達するまで 80〜300 反復かかる（図1）。**収束前のサンプルは目標分布ではないので破棄**。判断は**事後的（post-hoc）**——サンプリング後に連鎖を見て決める。保守的に多めに捨てるのが安全。客観的指標として **R-hat（Gelman–Rubin 統計量）** や**複数チェーン**を使う。

**③ 相関パラメータと SDT 応用.** 信号検出理論（SDT）の感度 $d'$・基準 $C$ の**二変量事後**を MCMC で推定（図2右で $d'$ と $C$ は相関）。相関は収束を著しく遅くする。単純な対策が **blocking**（相関するパラメータ群を分けて提案・受容・棄却する）。

**④ ギブスサンプリング（Metropolis-within-Gibbs）.** 多変量事後を、**各パラメータの条件付き分布**（他パラメータを固定したときの分布）から**パラメータごとに**サンプルして分解する。$d'$ を $C$ 固定で更新→$C$ を $d'$ 固定で更新、を交互に。多変量の提案を考えずに済むので相関に強い。

**⑤ 差分進化（DE-MCMC）.** 無相関の提案分布は相関した事後と噛み合わず、提案の約半分が棄却される（図3左の白い領域）。**複数チェーンを並走させ、他の2チェーンの差ベクトル $\gamma(\theta_m-\theta_n)$ ＋微小ノイズ**で提案を作ると、相関方向を自然に拾える（図3右）。チューニングパラメータが実質2個（$\gamma$ とノイズ幅）で、デフォルトが広く効くため**ほぼ自動調整**。応答時間モデルなど相関の強いモデルで有効。

## 実験結果と知見

図1が核心の可視化: **同じターゲットでも開始値しだいで burn-in が 0／約80／約300 反復**と変わり、burn-in を切らないと密度推定が歪む（下段）。図2は Metropolis-within-Gibbs が $d'$・$C$ の相関した同時事後を捉える様子、図3は DE のクロスオーバーが相関方向に提案を向ける直感。定量ベンチではなく「動かして腑に落とす」教育的実証。

## 限界・批判的視点

- **応用は認知科学・低次元（2 パラメータ）中心**。高次元での次元の呪い・typical set・有効サンプルサイズの理論は扱わず（→ [[sources/2019-conceptual-intro-mcmc]] に譲る）。
- HMC/NUTS（勾配を使う高次元向け MCMC）には触れない。
- R コード（付録 A/B/C）は本翻訳では割愛（付録扱い）。$\gamma$ の推奨式は元クリップで欠落（標準的には $\gamma=2.38/\sqrt{2K}$、$K$＝パラメータ数。本文では値が欠けている）。

## PFN との接続

- **MCMC ＝ PFN が「償却」で置き換える近似推論**。[[prior-data-fitted-networks]]（PFN）は、データセットごとに連鎖を回す代わりに、事前分布から合成したデータで一度だけ事前訓練し、推論時は近似計算ゼロで事後予測分布（PPD, [[bayesian-inference]]）を順伝播1回で返す。
- **本稿の価値は「per-dataset の運用負荷」を生々しく見せること**。提案分布の幅チューニング・棄却率の調整・burn-in の事後的判断・収束診断（R-hat・複数チェーン）・相関対策（blocking・Gibbs・DE）——これらは**新しいデータセットごとに人手で繰り返す手作業**であり、PFN はそれを**順伝播1回で消す**。MCMC の「速いか遅いか」だけでなく「**運用が煩雑**」という側面が、PFN の「即・無調整」の意義を際立たせる。
- **Gibbs/DE が苦労する「相関した高次元事後」こそ PFN の償却が効く難所**。MCMC が相関や高次元で受容率に苦しむのに対し、PFN は学習済みの写像でその構造を内在化する。
- **MCMC 三部作の補完**: [[sources/2021-mcmc-explained]]（informal な仕組み）↔ [[sources/2019-conceptual-intro-mcmc]]（厳密な理論）↔ 本稿（応用・実務・診断）。PFN 原典 [[sources/2021-transformers-can-do-bayesian-inference]] の速度比較相手 NUTS（HMC ベース MCMC）の「手前」にある基本手法群を押さえられる。
- 参考: Metropolis–Hastings と Gibbs の動くアニメ可視化 → twiecki, "MCMC sampling for dummies"（http://twiecki.github.io/blog/2014/01/02/visualizing-mcmc/ ）。

## 用語と略称

- **MCMC** = Markov Chain Monte–Carlo（マルコフ連鎖モンテカルロ）
- **Metropolis / Metropolis–Hastings** = 提案＋受容で事後からサンプルする基本 MCMC（後者は非対称提案に対応）
- **ギブスサンプリング（Gibbs sampling）/ Metropolis-within-Gibbs** = 条件付き分布からパラメータごとにサンプルする MCMC
- **差分進化（DE-MCMC）** = 複数チェーンの差ベクトルで提案を作る、ほぼ自動調整の MCMC
- **提案分布（proposal distribution）/ チューニングパラメータ** = 候補を生む分布 / その幅などサンプラーの挙動を左右する非モデルパラメータ
- **burn-in** = 収束前の初期サンプルを破棄する区間（判断は事後的）
- **収束（convergence）/ R-hat（Gelman–Rubin）** = 連鎖が定常に達した状態 / その客観的診断統計量
- **blocking** = 相関するパラメータ群を分けてサンプルする工夫
- **棄却率（rejection rate）/ 局所最大（local maxima）** = 提案が捨てられる割合 / 連鎖が詰まる罠
- **SDT（Signal Detection Theory）** = 信号検出理論。感度 d′・基準 C を持つ認知モデル（本稿の応用例）
- **事後分布 / 事後予測分布（PPD）** = パラメータの分布 / 次データの分布 → [[questions/posterior-vs-posterior-predictive]]
- **amortize（償却）** = 事前の一括計算で個別推論を安価にすること（PFN の鍵）

## 関連ページ

- [[markov-chain-monte-carlo]] — MCMC 概念ページ（本稿は応用・実務・Gibbs/DE/診断を補う）
- [[bayesian-inference]] — MCMC が近似する事後・PPD と、それを償却する PFN
- [[prior-data-fitted-networks]] — per-dataset の MCMC 運用を順伝播1回に償却する枠組み
- [[sources/2021-mcmc-explained]] — MCMC の informal な仕組み（詳細釣り合い・Metropolis–Hastings）
- [[sources/2019-conceptual-intro-mcmc]] — MCMC の厳密・包括版（重点サンプリング・ESS・次元の呪い・typical set）
- [[sources/2024-mcmc-basics-pymc3]] — PyMC3/NUTS で実際に回す最小コード例
- [[sources/2021-transformers-can-do-bayesian-inference]] — PFN 原典。NUTS（MCMC）比 1000〜10000 倍速
- [[questions/posterior-vs-posterior-predictive]] — 事後分布と事後予測分布の違い
- [[translations/2018-mcmc-simple-introduction]] — 本チュートリアルの全文翻訳（本文＋用語集）
