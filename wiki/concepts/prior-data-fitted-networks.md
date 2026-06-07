---
type: concept
aliases: [PFN, PFNs, Prior-Data Fitted Network, Prior-Data Fitted Networks, prior-fitted network]
tags: [meta-learning, bayesian-inference, in-context-learning, amortized-inference]
related:
  - "[[in-context-learning]]"
  - "[[bayesian-inference]]"
  - "[[structural-causal-model]]"
  - "[[tabular-foundation-model]]"
sources:
  - "[[sources/2021-transformers-can-do-bayesian-inference]]"
  - "[[sources/2022-tabpfn]]"
  - "[[sources/2025-tabpfn-v2]]"
  - "[[sources/2025-tabpfn-2-5]]"
  - "[[sources/2026-tabpfn-3]]"
  - "[[sources/2025-tabicl]]"
  - "[[sources/2026-tabicl-v2]]"
  - "[[sources/2025-real-tabpfn]]"
  - "[[sources/2023-pfns4bo]]"
  - "[[sources/2025-mitra]]"
  - "[[sources/2025-causalpfn]]"
  - "[[sources/2025-do-pfn]]"
  - "[[sources/2025-nanotabpfn]]"
  - "[[sources/2026-shappfn]]"
updated: 2026-06-06
---

# Prior-Data Fitted Networks（PFN）

> このリポジトリの中心概念。個別手法（TabPFN など）はページを分けず、ここに「代表手法」としてまとめる。

## 一言で

**Prior-Data Fitted Network（PFN, 事前分布から人工生成した合成データセットで一度だけ訓練しておき、推論時には重み更新なしに 1 回の順伝播でベイズ推論を近似するニューラルネットワーク）**。ふつうの機械学習が「新しいデータが来たらモデルをそのデータに当てはめる（fit）」のに対し、PFN は **当てはめ・予測アルゴリズムそのものを事前に学習** してしまう。新しいデータセットは「入力」として丸ごと与えるだけで、追加学習なしに予測が返る。

## 何を解こうとしているか — 近似対象は PPD

PFN が近似するのは **事後予測分布（PPD; Posterior Predictive Distribution）**。ベイズの教師あり学習では、テスト点 $x$ のラベル分布は「あり得るデータ生成メカニズム（仮説）$\phi$ 全体」について積分した量になる:

$$
p(y\mid x, D)\propto\int_{\Phi}p(y\mid x,\phi)\,p(D\mid\phi)\,p(\phi)\,d\phi.
$$

各仮説はその事前確率 $p(\phi)$ と、観測データに対する尤度 $p(D\mid\phi)$ で重み付けされる。この積分はふつう解析的に解けず、計算困難。詳細は [[bayesian-inference]]。

## どう近似するか — 合成データでの事前当てはめ（prior-fitting）

仕組みは驚くほど単純:

1. **事前分布を「データセット生成器」として用意する**。$\phi\sim p(\phi)$ で生成メカニズムを引き、$D\sim p(D\mid\phi)$ で合成データセットを作る。
2. 合成データセットの一部を訓練、残りをテストとし、「訓練部分を見てテスト部分のラベルを当てる」**交差エントロピー損失**で Transformer を訓練する。
3. これを膨大な数の合成データセットで繰り返す。

理論的に、この損失の最小化が**真の PPD を近似する**ことが示されている（[[sources/2021-transformers-can-do-bayesian-inference]] の洞察1・系1.1。Prior-Data NLL = PPD との交差エントロピー ＝ 加法定数を除き KL ダイバージェンスの期待値）。したがって訓練済み PFN の 1 回の順伝播 ≒ 近似ベイズ推論になる。重要なのは、この事前当てはめは**事前分布ごとに一度だけ**行えばよい点（アルゴリズム開発時の一括コスト）。なお、PFN は過適合の概念を持たない——訓練損失の改善は常に「厳密な事後分布への近さ」の改善を意味する。

## アーキテクチャ上の特徴

- データセットを **集合（set）** として入力する。各 (特徴量, ラベル) ペアを 1 トークンにし、トークン同士がアテンションで注意を向け合う。サンプルの順序に依存しない（順序不変, permutation invariant）。
- テスト点は訓練点に注意を向け、テスト集合全体の予測を **1 回の順伝播で同時に** 出す。勾配ベースの学習を推論時に一切行わない（ガウス過程の予測に近い構図）。
- この「重み更新なしに文脈から学ぶ」挙動は、大規模言語モデルで観測される [[in-context-learning]]（文脈内学習）と同じ枠組みであり、PFN はそれを **設計された事前分布のもとで意図的に行う** 点が特徴。

<figure>

![](../../raw/assets/2022-tabpfn/fig1b.svg)

<figcaption>図（再掲）: PFN のアテンション構造。各 (特徴量, ラベル) を 1 トークンにし、訓練サンプル同士は互いに注意を向け合い、テストサンプルは訓練サンプルにのみ注意を向ける（集合として順序不変に扱う）。［[[sources/2022-tabpfn]] 図1(b) より］</figcaption>
</figure>

## なぜ強力か — 事前分布を「設計」できる

通常の NN や GBDT では帰納バイアス（モデルの前提・選好）が「効率的に実装できる形」（$L_2$ 正則化、ドロップアウト、木の深さ制限など）に縛られる。PFN では **サンプリングできる事前分布を書くだけ** で任意の帰納バイアスを注入できる。これは学習アルゴリズムの設計方法を根本的に変える。さらにハイパーパラメータを点ではなく**分布**で与えれば、PPD がその空間まで積分してくれるので、ユーザ側のチューニングが要らなくなる。

## 代表手法

- **Transformers Can Do Bayesian Inference**（Müller et al. 2021 → [[sources/2021-transformers-can-do-bayesian-inference]]）: PFN の枠組みと理論を最初に提示した**原典**。損失最小化が PPD 近似に一致することを証明し（洞察1・定理）、固定 GP をほぼ完璧に模倣、扱いにくい GP/BNN でも MCMC/SVI より 200〜10000 倍速い近似を達成。回帰用の予測ヘッド「リーマン分布」も導入。表形式分類では既に XGBoost 等を上回った（[[sources/2022-tabpfn]] の前身）。
- **TabPFN（v1）**（Hollmann et al. 2022 → [[sources/2022-tabpfn]]）: 上記を表形式データ向けに特化。[[structural-causal-model]] と BNN の混合を事前分布にし、小規模表データ（〜1000 サンプル）で GBDT を上回り 1 時間級 AutoML に 1 秒未満で並ぶ。PFN を実用レベルに引き上げた最初の例。
- **TabPFN v2**（Hollmann et al. 2025, Nature → [[sources/2025-tabpfn-v2]]）: v1 を 50 倍スケール（〜10,000 サンプル・500 特徴量）し、カテゴリ・欠損・外れ値・回帰をネイティブ対応。各セルを 1 トークンとする 2 次元アテンションを採用。さらに生成・密度推定・埋め込み・ファインチューニングを備えた **表形式基盤モデル（[[tabular-foundation-model]]）** へ発展。
- **TabPFN-2.5**（Prior Labs 2025 → [[sources/2025-tabpfn-2-5]]）: v2 を 〜50,000 サンプル・2,000 特徴量（セル 20×）へスケールし、業界標準ベンチ TabArena で首位。深い層＋"thinking rows"、実データ微調整版 Real-TabPFN-2.5、本番展開用の蒸留（as-MLP/TreeEns）、因果推論応用を追加。
- **TabPFN-3**（Prior Labs 2026 → [[sources/2026-tabpfn-3]]）: **100 万行**へスケールし、**テスト時計算（Thinking mode）** を導入。多クラス（最大160）・関係データ・時系列・テキストへ拡大。アーキテクチャは v2.x の行×特徴量交互アテンションを捨て、**v1 流の「行全体に対する ICL」へ回帰**（TabICLv2 系の 3 段で各行を 1 ベクトルに圧縮→系列長が行数のみに比例し大規模に効率的）。PFN 系の現最前線。
- **TabICL（Inria 系）**（Qu et al. → [[sources/2025-tabicl]] v1 / [[sources/2026-tabicl-v2]] v2）: TabPFN 一族とは**別グループ**だが、SCM 合成事前分布＋ICL という PFN 流の枠組みを共有する兄弟。「列ごと埋め込み→行圧縮→行 ICL」の **3 段アーキテクチャ** で大規模（最大 50 万サンプル）対応を実現。**v2（TabICLv2）は回帰・QASSMax・新 prior・Muon でオープンな SOTA** となり、この設計を TabPFN-3 が採用した（上記の「v1 流 ICL への回帰」の源流＝"TabICLv2 ベース"）。
- **Mitra**（Zhang et al. 2025, Amazon → [[sources/2025-mitra]]）: 「PFN の性能は事前分布の設計で決まる」を**定量的な設計原理**に変えた代表例。良い prior を**性能・多様性・独自性**（汎化性行列 $\mathbf{G}$ と性能ベクトル $\mathbf{P}$）で測り、[[structural-causal-model|SCM]]＋木ベース prior の混合を選定。行 1D・要素 2D の両アーキで効く**モデル非依存**性を示し、TabPFNv2/TabICL を分類・回帰で上回る。
- **PFNs4BO**（Müller et al. 2023 → [[sources/2023-pfns4bo]]）: PFN を**ベイズ最適化（[[bayesian-optimization]]）のサロゲート**に応用した代表例。表形式分類とは別系統の応用で、PFN を「GP のベイズ的ドロップイン置換」として使う。GP/高度GP/BNN を事前分布に持ち、固定 GP 事後を再現しつつ、GP では難しい拡張（ユーザー事前分布＝最適解位置のヒント、無関係次元の無視、獲得関数＝知識勾配を直接学習する非近視眼的 BO）を「事前分布を書くだけ」で実現。HPO ベンチで GP ベース SOTA（HEBO）を上回る。
- **CausalPFN**（Balazadeh et al. 2025 → [[sources/2025-causalpfn]]）: PFN を**因果効果推定（処置効果 ATE/CATE）**へ拡張した代表例。予測（回帰・分類）でも最適化でもなく、「観測データから介入の効果を推定する」という新ドメインに PFN を持ち込む。鍵は**無視可能性（ignorability）を設計で焼き込んだ DGP 事前分布**——任意の表から、処置を共変量だけで生成して因果データセットを量産し、真値の条件付き期待潜在結果（CEPO）をラベルに、PFN の data-prior 損失を因果版（CEPO への前向き KL 最小化）にして訓練。予測ヘッドは原典のリーマン分布と同型の 1024 ビンのヒストグラム（CEPO-PPD）。IHDP/ACIC/Lalonde で専用の因果推定器群を無調整で上回り、PFN/ICL の射程が因果へ届くことを示した。潜在結果フレームワーク（Rubin 流）に立ち、[[structural-causal-model|SCM]] を prior 素材に使う TabPFN とは因果の定式化が異なる点に注意（→ [[structural-causal-model]]）。
- **Do-PFN**（Robertson et al. 2025 → [[sources/2025-do-pfn]]）: 同じく PFN を因果効果推定へ拡張した代表例だが、CausalPFN とは逆に **SCM／do 計算（Pearl 流）** に立つ。prior に**介入 $do(t)$ つきの SCM** を据え、観測データから条件付き介入分布 $p(y\mid do(t),x)$ を予測（負の対数尤度最小化＝CID への前向き KL 最小化）。**因果グラフを与えず・無交絡を要求せず・同定不能性を不確実性として表現**する点が新しい。アーキは TabPFN とほぼ同じ（730 万パラメータ）。CausalPFN（潜在結果・ignorability 要求）と Do-PFN（SCM・do 計算・グラフ不要）で、「PFN の因果版」が Rubin 流と Pearl 流の両側から立ち上がった（→ [[structural-causal-model]] に両者の対比）。
- **nanoTabPFN**（Pfefferle et al. 2025 → [[sources/2025-nanotabpfn]]）: TabPFN v2 の**教育・研究向けの軽量再実装**。Karpathy の minGPT/nanoGPT（GPT の教育的再実装）の表形式基盤モデル版。500 行未満のコードで TabPFN v2 の4構成要素（FeatureEncoder・TargetEncoder・TransformerEncoderLayer の bi-attention・Decoder）を実装し、シングル GPU 1分の事前訓練で従来 ML ベースライン（kNN・決定木・ランダムフォレスト）に並ぶ性能を実現（TabPFN v2 事前訓練より **160,000 倍速**）。「新しい prior・アーキ変更・訓練手順のアイデアを分単位で試せる出発点」として、TFM 研究のプロトタイピングを高速化する。著者は Frank Hutter（TabPFN シリーズと共通）。
- **ShapPFN**（Sena & Azevedo 2026 → [[sources/2026-shappfn]]）: PFN に**説明可能性（Shapley 値）をアーキテクチャごと内蔵**した代表例。[[sources/2025-nanotabpfn|nanoTabPFN]] をベースに、予測を「ベース項＋特徴量ごとの加法的寄与 $\phi_f$」に分解する2つのデコーダーヘッド（BaseDecoder・ShapDecoder）を追加し、$f_\theta(x)=\mathrm{base}+\sum_f\phi_f(x)$ という加法的予測にする。ViaSHAP 流の Shapley 一致性損失で訓練すると、**予測と説明を1回の順伝播で同時に**出せ、事後計算が重い KernelSHAP（約 610 秒）を **0.06 秒（1000 倍超速）** で同品質（一致 R²=0.96）に置き換える。「ベイズ推論を事前訓練に償却する PFN」の発想を**説明計算の償却**へ応用した形。nanoTabPFN を実研究の踏み台に使った最初の公開例でもある。

## 限界

- アテンションが入力サンプル数に対して二次（$O(n^2)$）のため、素朴には大規模データに弱い（線形アテンション系の導入が課題）。
- 性能は**事前分布の設計**に決定的に依存する。事前分布が想定しない状況（TabPFN なら無情報特徴量・カテゴリ・欠損）では劣化する。

## 関連ページ

- [[in-context-learning]] — PFN の推論挙動の一般枠組み
- [[bayesian-inference]] — 近似対象（PPD）の理論的背景
- [[structural-causal-model]] — TabPFN の事前分布の中核
- [[sources/2022-tabpfn]] — TabPFN 論文
- [[questions/pfn-paper-and-gaussian-process]] — PFN 原典と GP の関係・出力の仕組み・カーネル/ハイパラの扱い
- [[questions/tabpfn-tabicl-versions-mechanism]] — 各バージョンの TabPFN/TabICL は GP をどう扱うか
- [[bayesian-optimization]] — PFN を BO サロゲートに応用（[[sources/2023-pfns4bo]]）
- [[sources/2023-pfns4bo]] — PFN を BO に応用した代表例（GP のドロップイン置換）
- [[sources/2025-mitra]] — 合成 prior の設計原理（性能・多様性・独自性）と混合（Amazon）
- [[sources/2025-causalpfn]] — PFN を因果効果推定に拡張（ignorability prior・CEPO-PPD）
- [[sources/2025-do-pfn]] — PFN を介入効果推定に拡張（介入つき SCM prior・CID・do 計算）
- [[sources/2025-nanotabpfn]] — TabPFN v2 の教育的軽量再実装（500行・1分訓練・nanoGPT の表形式版）
- [[sources/2026-shappfn]] — PFN に Shapley 値説明を内蔵（nanoTabPFN ベース・予測と説明を1順伝播で・KernelSHAP 1000倍速）
