---
type: concept
aliases: [MCMC, Markov Chain Monte Carlo, マルコフ連鎖モンテカルロ, Metropolis-Hastings, メトロポリス・ヘイスティングス, HMC, Hamiltonian Monte Carlo, ハミルトニアンモンテカルロ, NUTS, No-U-Turn Sampler, leapfrog, リープフロッグ, dual averaging, 双対平均化, Stan, 詳細釣り合い, typical set, 典型集合, Gibbs sampling, ギブスサンプリング, R-hat, blocking]
tags: [bayesian-inference, approximate-inference, monte-carlo, sampling]
related:
  - "[[bayesian-inference]]"
  - "[[prior-data-fitted-networks]]"
  - "[[gaussian-process]]"
  - "[[expectation-maximization]]"
sources:
  - "[[sources/2021-mcmc-explained]]"
  - "[[sources/2019-conceptual-intro-mcmc]]"
  - "[[sources/2018-mcmc-simple-introduction]]"
  - "[[sources/2024-mcmc-basics-pymc3]]"
  - "[[sources/2020-understanding-bayesian-inference]]"
  - "[[sources/2014-no-u-turn-sampler]]"
  - "[[sources/2017-conceptual-intro-hmc]]"
  - "[[sources/2021-transformers-can-do-bayesian-inference]]"
updated: 2026-06-17
---

# Markov Chain Monte Carlo（MCMC, マルコフ連鎖モンテカルロ）

## 一言で

**MCMC（Markov Chain Monte Carlo, マルコフ連鎖モンテカルロ。マルコフ連鎖を使ってモンテカルロ推定を行うアルゴリズムの一群）** は、**直接サンプリングできない確率分布——とりわけベイズ推論の扱いにくい事後分布——から、近似的にサンプルを生成する**手法。生成したサンプルの平均で、解析的に解けない積分（期待値・事後予測・証拠）を近似する。最大の勘所は、**ベイズ則の分母にある正規化定数（証拠 $\mathcal{Z}$）を一切計算せず、「事後 $\propto$ 尤度 × 事前」という比例関係だけでサンプルできる**こと。[[bayesian-inference]] の事後予測分布（PPD）を計算する従来手段の代表であり、[[prior-data-fitted-networks]]（PFN）が順伝播1回に**償却（amortize）**して置き換えようとしている当の計算でもある。

> 直感重視の最初の一冊は [[sources/2021-mcmc-explained]]（Shivam Agrahari / Towards Data Science）。「モンテカルロ（乱数で量を近似）＋マルコフ連鎖（現在状態だけで次が決まる遷移）→ 定常分布 → 詳細釣り合い → Metropolis–Hastings（提案確率 g ＋受容確率 A）」を、3状態の数値例や濡れたソファ風の例で平易に追う。「正規化定数 Z が打ち消えて消える」というトリックを具体的に見せる層。

> 厳密・包括的な解説は [[sources/2019-conceptual-intro-mcmc]]（Joshua Speagle, arXiv 1909.12313, dynesty 作者）。「ベイズ推論で本当に欲しいのは事後そのものでなく**事後上の積分（期待値）**だ」という視点から、**グリッド近似 → 重点サンプリング → MCMC** を一本の連続体として導出し、有効サンプルサイズ（ESS）・自己相関時間・**次元の呪い・typical set（典型集合）**・アンサンブル法（emcee の stretch move まで）を扱う。「なぜ高次元で MCMC が遅くなるのか」を原理から理解したい層。

> 応用・実務寄りの平易なチュートリアルは [[sources/2018-mcmc-simple-introduction]]（van Ravenzwaaij et al., Psychon Bull Rev 2018）。Metropolis を1観測の母平均推定で手を動かして示し、**提案幅のチューニング・棄却率・局所最大・burn-in/収束診断（R-hat・複数チェーン）**といった実務のつまずきを丁寧に扱う。さらに **ギブスサンプリング（Metropolis-within-Gibbs）・blocking・差分進化（DE-MCMC）**を信号検出理論の相関した二変量事後で直感的に見せる。「実際に回すと何が大変か」を掴む層。

## 仕組み（4つの部品）

MCMC は名前どおり「モンテカルロ」＋「マルコフ連鎖」の合成で、次の部品から成る。

1. **モンテカルロ推定**: 期待値 $\mathbb{E}_{\mathcal{P}}[f(\Theta)]=\int f(\Theta)\mathcal{P}(\Theta)\,d\Theta$ を、分布からのサンプル平均 $\frac1n\sum_i f(\Theta_i)$ で近似する（乱数で決定的な量を当てる）。問題は「事後分布 $\mathcal{P}$ から直接サンプルできないこと」。
2. **マルコフ連鎖**: 次の状態が**現在の状態だけ**に依存し過去の履歴に依存しない（マルコフ性）状態列 $\{\Theta_1\to\dots\to\Theta_n\}$。
3. **定常分布（stationary distribution）**: 遷移を繰り返すと分布が変化しなくなる不動点。MCMC の核心は、**サンプルしたい事後分布が、ある巧妙に設計したマルコフ連鎖の定常分布になる**ように作ること。
4. **詳細釣り合い（detailed balance）**: 定常分布を目的の事後 $\mathcal{P}$ に一致させる十分条件 $\mathcal{P}(\Theta_i)\,P(\Theta_{i+1}\!\mid\!\Theta_i)=\mathcal{P}(\Theta_{i+1})\,P(\Theta_i\!\mid\!\Theta_{i+1})$。これにより、**事後の比 $\mathcal{P}(\Theta_{i+1})/\mathcal{P}(\Theta_i)$ さえ計算できればよく、正規化定数 $\mathcal{Z}$ は約分されて消える**。

**代表アルゴリズム＝Metropolis–Hastings（メトロポリス・ヘイスティングス, MH）**: 各ステップを「提案」と「受容」に分ける。(1) 提案分布 $\mathcal{Q}(\Theta'\!\mid\!\Theta)$ から次候補 $\Theta'$ を引き、(2) **受容確率 $T=\min\!\left[1,\ \frac{\mathcal{P}(\Theta')}{\mathcal{P}(\Theta)}\frac{\mathcal{Q}(\Theta\mid\Theta')}{\mathcal{Q}(\Theta'\mid\Theta)}\right]$** でその候補を採るか（採らなければ現在地に留まる）を、$[0,1]$ の乱数とのコイン投げで決める。これを繰り返すと、サンプルの密度が事後 $\mathcal{P}$ に比例していく。

**ギブスサンプリング（Gibbs sampling）/ Metropolis-within-Gibbs**: 多変量の事後を、各パラメータの**条件付き分布**（他のパラメータを現在値に固定したときの分布）から**パラメータごとに**順に更新する MCMC（[[sources/2018-mcmc-simple-introduction]]）。多変量の提案を設計せずに済むので、パラメータが相関しているとき MH より扱いやすい。

## なぜ必要か（PPD・期待値の計算困難性）

ベイズ推論で欲しい量——事後予測分布（PPD）$\int p(y\mid x,\Theta)\,p(\Theta\mid D)\,d\Theta$、パラメータの期待値、モデル比較の証拠 $\mathcal{Z}$——は、いずれも**事後分布上の積分**である（[[bayesian-inference]]）。共役な特殊例（[[sources/2024-bayesian-inference-step-by-step]] のコイン投げ・正規平均）を除けば、この積分は解析的に解けず、分母の証拠 $\mathcal{Z}=\int\mathcal{L}(\Theta)\pi(\Theta)\,d\Theta$ の計算が特に困難。MCMC は「事後からサンプルを引いて積分を平均で近似」しつつ、詳細釣り合いのおかげで $\mathcal{Z}$ を回避するので、この困難を実用的に切り抜けられる。変分推論（VI）と並ぶ**近似推論（approximate inference）の二大手法**の一方である。

## 限界・実務上の注意

- **burn-in**: 連鎖が定常状態に達する前の初期サンプルは偏っているので捨てる。
- **自己相関と有効サンプルサイズ（ESS）**: 連続するサンプルは相関するため、$n$ 個あっても情報量は $n_{\rm eff}=n/(1+\tau)$（$\tau$＝自己相関時間）に減る。受容率が低いほど $\tau$ が増え効率が落ち、iid な重点サンプリングに負けることすらある（[[sources/2019-conceptual-intro-mcmc]] §6.2）。
- **多峰（multi-modal）分布に弱い**: モードの一つに連鎖が詰まり、偏ったサンプルになる。
- **収束診断は事後的（post-hoc）**: 連鎖が定常に達したかは、走らせた後に判断する。客観的指標として **R-hat（Gelman–Rubin 統計量）** や**複数チェーンの比較**を使う（[[sources/2018-mcmc-simple-introduction]]）。提案幅は**チューニングパラメータ**で、広すぎると棄却率が跳ね上がり狭すぎると収束が遅れる。
- **相関したパラメータ**: 事後の相関は収束を著しく遅くする。**blocking**（相関するパラメータ群を分けて提案・受容）、**ギブスサンプリング**、**差分進化（DE-MCMC）**（複数チェーンの差ベクトルで提案を相関方向に向ける）などで対処する。
- **次元の呪いと typical set（典型集合）**: 高次元では**事後質量＝密度×体積が、半径 $\approx\sqrt{d}\,\sigma$ の薄い殻に集中**する。サンプルの大半は密度ピークでなくこの殻に居るため、(a) ピーク領域の特徴づけに非効率、(b) 提案が殻から外れて**受容率が指数的に低下**する。提案幅を $\propto 1/\sqrt{d}$ で縮める、共分散を自動調整するアンサンブル法（DE-MCMC・emcee の stretch move）などで対処する。
- **改良版**: 勾配を使って遠くまで効率よく動く **HMC（Hamiltonian Monte Carlo）** と、その軌跡長を自動調整する **NUTS（No-U-Turn Sampler）** が高次元で主流（下節）。
- **実務では確率的プログラミング言語（PyMC・Stan 等）に任せる**: 詳細釣り合いや提案分布を自分で書かず、事前×尤度のモデルを*宣言*すれば NUTS 等のサンプラーが事後を出す（最小例は [[sources/2024-mcmc-basics-pymc3]] の PyMC3 コード）。PFN はこの「モデルを書く→サンプラーが事後を出す」を「事前＝データ生成器を書く→順伝播が事後予測を出す」へ償却した、いわば**償却版の確率的プログラミング**と見なせる。

## HMC と NUTS（勾配を使う高次元向け MCMC）

高次元・強相関の事後で Metropolis–Hastings やギブスがランダムウォークに陥る問題を、**HMC（Hamiltonian Monte Carlo, ハミルトニアンモンテカルロ）** は勾配で解く（一次資料 [[sources/2014-no-u-turn-sampler]]）。各パラメータ $\theta$ に補助の**運動量 $r$** を足し、$p(\theta,r)\propto\exp\{\mathcal L(\theta)-\frac12 r\cdot r\}$ を「位置エネルギー＋運動エネルギー」の物理系とみなす。**対数事後の勾配で導かれる軌跡（リープフロッグ積分）**を走らせるので、前のサンプルから**遠くまで一気に**移動でき、独立サンプル1つあたり $O(D^{5/4})$（ランダムウォークは $O(D^2)$）。代償は (1) 勾配が要る（自動微分で緩和）、(2) ステップサイズ $\epsilon$ とステップ数 $L$ の手調整が要る（$L$ が小→ランダムウォーク、大→**U ターンして足跡をたどり無駄**）。

**NUTS（No-U-Turn Sampler）** は $L$ の手調整を消す: 軌跡が出発点へ戻り始めた（$(\theta^+-\theta^-)\cdot r^{\pm}<0$）かを見て、**前後に $1\to2\to4\to\dots$ ステップ伸ばす再帰的な二分木（doubling）**を、いずれかの平衡部分木が U ターンしたら停止する。スライス変数と慎重な候補集合の設計で詳細釣り合いを保つ。さらに **双対平均化（dual averaging）** で $\epsilon$ を目標受容率（$\delta\approx0.6$）に合わせて自動調整。結果、**ユーザーは $\delta$ を1つ指定するだけの「ターンキー」サンプラー**になり、よく調整した HMC と同等以上、ランダムウォーク Metropolis やギブスより桁違いに効率的。**確率的プログラミング言語 Stan の中核**であり、**PFN 原典の速度比較相手そのもの**（PFN は NUTS 比 1000〜10000 倍速）。詳細・実験は [[sources/2014-no-u-turn-sampler]]。

## HMC の幾何学的基礎（なぜ効くか・いつ失敗するか）

NUTS 原典が「HMC の作り方（How）」なら、Betancourt の概念的総説 [[sources/2017-conceptual-intro-hmc]] は「なぜ効くか（Why）」を、微分幾何学の数式を避けた**幾何学的直観**だけで与える。鍵は前出の **typical set（典型集合, 密度×体積＝確率質量が集中する細い殻）** という単一の視点に集約される。

- **HMC を必然として導く**: 高次元では体積が裾に集中するため、ランダムウォーク的な遷移（拡散的＝その場でうろつく）は典型集合を渡りきれない。等高線に**沿って滑走する coherent（首尾一貫した）遷移**が要る。対数事後の**勾配**だけでは（パラメータ化依存で）モードに落ちて典型集合から外れるが、補助**運動量 $p$** を足して**相空間（phase space）**へ拡張し、$H(q,p)=-\log\pi(q,p)$（**ハミルトニアン**＝エネルギー、$K$ 運動＋$V$ ポテンシャルに分解）の定める軌道を走らせると、勾配が典型集合に整列するようねじれる。物理アナロジー（モード＝惑星、勾配＝重力、典型集合＝周回軌道、運動量を与えて安定軌道に乗せる）で一貫して説明される。
- **チューニングの幾何学**: 相空間はエネルギーの**等位集合（level set）**に葉層化し、連鎖は「等位集合上の決定論的滑走」＋「等位集合間のランダムウォーク」に分離する。ここから (1) **質量行列 $M^{-1}$ を事後共分散に合わせる**と等位集合が一様化して探索が速い、(2) **最適な積分時間はエネルギーごとに異なる**（裾が重いほど長い）ので動的に決める＝**No-U-Turn 基準の動機**、という指針が出る。
- **実装はシンプレクティック積分器**: 普通の数値ソルバーは誤差累積で**ドリフト**するが、**シンプレクティック積分器（リープフロッグ）**は相空間体積を保存しエネルギー等位集合の近くで振動するだけ。残差はメトロポリス補正（運動量反転で可逆化）で厳密に除ける。
- **HMC 固有の診断（いつ失敗するか）**: ① 運動エネルギー不適合 → **E-BFMI**（energy Bayesian fraction of missing information）が **0.3 未満で危険**。② 高曲率領域（階層モデルの漏斗など）→ シンプレクティック積分器が**発散（divergence）**するので、見えにくい統計的病理の鋭敏な検出器になる（対処：ステップサイズ縮小／中心化⇄非中心化パラメータ化の切替）。これらは必要条件にすぎず、複数チェーンの **分割 $\hat R$** と併用する。

> typical set 自体の一般 MCMC 文脈での議論（次元の呪い・ESS・重点サンプリングとの連続性）は [[sources/2019-conceptual-intro-mcmc]] と重なる。両者を合わせると「なぜ高次元で素朴 MCMC が破綻し、勾配ベースの HMC/NUTS が要るのか」が原理から腑に落ちる。

## なぜ PFN の文脈で重要か

- **MCMC ＝ PFN が「償却」で置き換える近似推論の代表**。MCMC は**データセットごとに連鎖を回し直す**必要があり、高次元では受容率の指数低下・長い自己相関で高コスト。[[prior-data-fitted-networks]]（PFN）はこの反復サンプリングを、事前分布から合成したデータで**一度だけ事前訓練**することに償却し、推論時は近似計算ゼロで事後予測分布を順伝播1回で返す（[[bayesian-inference]] の「償却ベイズ推論」）。
- **PFN 原典の速度比較相手そのもの**。[[sources/2021-transformers-can-do-bayesian-inference]] は、扱いにくいハイパー事前分布付き [[gaussian-process]] や BNN の事後を **NUTS（HMC ベースの MCMC）** で近似する従来法と比べ、PFN が同等以上の精度を **NUTS 比 1000〜8000 倍（GP）／10000 倍（BNN）速く**達成すると示した。MCMC の内部メカニズム（なぜ遅いか）は、PFN の速さの意義を理解する前提になる。
- **PFN 精度評価の「物差し」**。閉形式の事後がない設定でも、**十分収束させた MCMC（NUTS）を真値の代理（ゴールドスタンダード）**にして、PFN の予測分布の正しさを KL・CRPS・NLL で測れる（[[questions/pfn-bayesian-inference-evaluation-settings]] の「(E) 収束 NUTS を物差しに」）。その「どれだけ回せば信頼できるか」の理論的裏付け（収束・一貫性・ESS）が [[sources/2019-conceptual-intro-mcmc]]。

## 関連ページ

- [[bayesian-inference]] — MCMC が近似しようとする事後・PPD・証拠と、それを償却する PFN
- [[prior-data-fitted-networks]] — MCMC の反復サンプリングを順伝播1回に償却する枠組み
- [[gaussian-process]] — ハイパー事前分布付き GP の事後を NUTS で近似する従来法（PFN の比較対象）
- [[sources/2021-mcmc-explained]] — MCMC の informal な入門（定常分布・詳細釣り合い・Metropolis–Hastings）
- [[sources/2019-conceptual-intro-mcmc]] — MCMC の厳密・包括版（重点サンプリング・ESS・次元の呪い・typical set・アンサンブル法）
- [[sources/2018-mcmc-simple-introduction]] — 応用・実務寄りの平易な入門（Gibbs・DE・blocking・R-hat・burn-in と SDT 応用）
- [[sources/2024-mcmc-basics-pymc3]] — PyMC3/NUTS で実際に回す最小コード例（確率的プログラミング）
- [[sources/2014-no-u-turn-sampler]] — HMC/NUTS の原典（Hoffman & Gelman）。U ターン基準＋双対平均化で turnkey 化、Stan の中核、PFN 原典の速度比較相手
- [[sources/2017-conceptual-intro-hmc]] — HMC の幾何学的基礎（Betancourt）。typical set から HMC を必然として導き、E-BFMI・発散の診断まで。NUTS 原典の Why に当たる総説
- [[sources/2020-understanding-bayesian-inference]] — 近似推論（MCMC/VI）を高レベルで導入した入門
- [[sources/2021-transformers-can-do-bayesian-inference]] — PFN 原典。NUTS（MCMC）比 1000〜10000 倍速を実証
- [[questions/pfn-bayesian-inference-evaluation-settings]] — 「収束 NUTS をゴールドスタンダードに」＝MCMC を PFN 精度の物差しに
