---
type: concept
aliases: [ICL, In-Context Learning, 文脈内学習]
tags: [meta-learning, transformer, amortized-inference]
related:
  - "[[prior-data-fitted-networks]]"
  - "[[bayesian-inference]]"
sources:
  - "[[sources/2021-transformers-can-do-bayesian-inference]]"
  - "[[sources/2022-tabpfn]]"
  - "[[sources/2025-tabpfn-v2]]"
  - "[[sources/2025-tabpfn-2-5]]"
  - "[[sources/2026-tabpfn-3]]"
  - "[[sources/2025-tabicl]]"
  - "[[sources/2026-tabicl-v2]]"
  - "[[sources/2025-causalpfn]]"
  - "[[sources/2025-do-pfn]]"
  - "[[sources/2025-chronos-2]]"
updated: 2026-06-03
---

# In-Context Learning（ICL, 文脈内学習）

## 一言で

**文脈内学習（ICL; In-Context Learning, モデルの重みを一切更新せず、入力＝文脈として与えたラベル付き例の系列から、その場でタスクを解く能力）**。学習＝勾配でパラメータを更新すること、という常識に対し、ICL では予測に必要な「学習」が**順伝播の内部**で起こる。大規模言語モデル（LLM）でプロンプトに数例を入れると新しいタスクを解けてしまう現象として広く知られるようになった。

## PFN との関係（このリポジトリでの位置づけ）

[[prior-data-fitted-networks]]（PFN）は、この ICL を **偶発的な創発ではなく、設計された事前分布のもとで意図的に行う** 枠組みと見なせる。
- LLM の ICL: 大量の自然言語で次トークン予測を訓練した副産物として、文脈内の例から学ぶ能力が「創発」する。
- PFN の ICL: 「データセットを文脈として与え、保留点を当てる」タスクで明示的に訓練する。入力は (x, f(x)) のラベル付き例の系列（または集合）で、出力はテスト点の予測。

どちらも共通して、**重み更新なし・順伝播一回**で「訓練データを見て予測する」を実現する。この構図と理論は PFN 原典（[[sources/2021-transformers-can-do-bayesian-inference]]）が確立し、TabPFN（[[sources/2022-tabpfn]]）はそれを表形式データに持ち込んだ代表例で、論文要旨でも自手法を「ICL を行う」と明言している。後継の TabPFN v2（[[sources/2025-tabpfn-v2]]）は、この ICL を「例示ベースの宣言的プログラミング（望ましい挙動を示す合成データを生成して学習させる）」と位置づけ、予測に加え生成・密度推定・埋め込みまで担う [[tabular-foundation-model]] へと押し広げた。さらに TabPFN-2.5（[[sources/2025-tabpfn-2-5]]）は ICL を 〜5 万サンプル・2,000 特徴量へスケールし、計算容量を足す「思考（thinking）行」を入力に加えるなど、ICL 自体の実用的な拡張も進めている。TabPFN-3（[[sources/2026-tabpfn-3]]）は ICL を **100 万行**へスケール（各行を 1 ベクトルに圧縮し系列長を行数のみに依存させる＋QASSMax で長さ汎化）し、さらに推論時に計算量を増やす **テスト時計算（Thinking mode）** を ICL に持ち込んだ（[[tabular-foundation-model]] 参照）。

**長文脈での ICL の落とし穴（attention fading と QASSMax）**: ソフトマックス注意は文脈長 $n$（＝訓練サンプル数）が増えると分母が増大して注意分布が平坦化し、関連トークンに鋭く集中できなくなる（attention fading）。短い系列で事前訓練したモデルが大規模データで識別的な注意を保てない、という長さ汎化の壁である。TabICLv2（[[sources/2026-tabicl-v2]]）はこれに **QASSMax（query-aware scalable softmax）** で対処した。注意クエリを $\log n$ に応じて動的に再スケールして注意を鋭く保つ手法で、長系列での事前訓練なしに大規模データへの ICL 汎化を改善する。この QASSMax を TabPFN-3 も採用している（同論文の "TabICLv2 ベース"）。

**ICL は時系列予測へ — group attention**: ICL の「関連データを文脈として束ね、その場で活用する」構図は、表形式だけでなく**時系列予測**にも現れる。**Chronos-2（[[sources/2025-chronos-2]]、Amazon）** は、バッチ内の関連系列・多変量の変量・ターゲットと共変量を「グループ」として束ねる **group attention** 機構を導入し、これを ICL の中核に据えた。グループ ID を指定するだけで、単変量・多変量・共変量つき予測を1モデル・アーキ変更なしで解く（連結で文脈を伸ばさずバッチ軸で共有するので変量数に線形）。短い過去しかない系列でも、関連系列から情報を借りる**クロス学習（few-shot）**で予測が改善する。TabPFN が表形式で行う ICL を、時系列の多変量・共変量へ一般化した例といえる（合成データのみで能力を獲得する点も TabPFN 流）。

**ICL の射程は予測の外へ — 因果効果推定**: 文脈＝ラベル付き例の系列から重み更新なしに解く、という ICL の構図は、回帰・分類に限らない。**CausalPFN（[[sources/2025-causalpfn]]）** は、観測データ $\{(t,x,y)\}$ を文脈トークンとして与えると、クエリ $(t,x)$ に対する**因果効果**（条件付き期待潜在結果 CEPO、そこから ATE/CATE）の予測分布を 1 回の順伝播で返す。同時期の **Do-PFN（[[sources/2025-do-pfn]]）** も同じく観測データを文脈に取るが、Pearl の **do 計算**側から条件付き介入分布 $p(y\mid do(t),x)$ を ICL で予測する（因果グラフ不要・無交絡不要）。いずれも「データセットを文脈に入れてその場で解く」という ICL を、予測タスクから**介入効果の推定タスク**へ移植した例であり、ICL ＝償却ベイズ推論という見方（次節）の因果版にあたる。

## なぜ "学習" が順伝播で起きるのか

理論的な見方の一つは **償却ベイズ推論（amortized Bayesian inference）**。訓練を通じてネットワークは「文脈（訓練データ）→ 事後予測分布」という写像そのものを学ぶ。一度学んでしまえば、新しい文脈に対しては写像を 1 回評価するだけでよい（個別の最適化を「償却」する）。この見方では ICL ≒ 近似ベイズ推論であり、[[bayesian-inference]] と [[prior-data-fitted-networks]] を結ぶ橋になっている。

## 関連ページ

- [[prior-data-fitted-networks]] — ICL を事前分布設計のもとで行う枠組み
- [[bayesian-inference]] — ICL を償却ベイズ推論として解釈する視点
- [[sources/2026-tabicl-v2]] — attention fading と QASSMax（長文脈 ICL 汎化）の出典
- [[sources/2022-tabpfn]] — 表形式データへの ICL 適用例
- [[sources/2025-causalpfn]] — ICL を因果効果推定へ移植した例（文脈＝観測データ→因果効果の予測分布）
- [[sources/2025-do-pfn]] — ICL を介入効果推定へ移植（文脈＝観測データ→条件付き介入分布 CID）
- [[sources/2025-chronos-2]] — ICL を時系列予測へ（group attention・クロス学習・共変量つき予測）
- [[questions/pfn-paper-and-gaussian-process]] — 文脈＋クエリ→1 回の順伝播で予測分布を出す仕組み
