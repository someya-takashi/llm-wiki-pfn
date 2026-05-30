---
type: concept
aliases: [GP, GPs, Gaussian Process, ガウス過程]
tags: [probabilistic-modeling, bayesian-inference, kernel-method]
related: [[bayesian-inference]], [[prior-data-fitted-networks]]
sources: [[sources/2021-transformers-can-do-bayesian-inference]], [[sources/2022-tabpfn]]
updated: 2026-05-30
---

# Gaussian Process（GP, ガウス過程）

## 一言で

**ガウス過程（GP; Gaussian Process, 関数そのものの上に確率分布を置くノンパラメトリックなベイズモデル）**。任意の有限個の入力点での出力が多変量正規分布に従う、と仮定する。平均関数（通常ゼロ）と**カーネル関数** $k(x_i,x_j)$（2 点の出力がどれだけ相関するか＝関数の滑らかさ・スケールを決める）で完全に規定される。データを観測すると、事後分布も解析的に閉形式で求まり、**予測平均と不確実性（信頼区間）を厳密に**得られる。

## なぜ PFN の文脈で重要か

このリポジトリでは GP は 2 つの役割で繰り返し登場する。

1. **PFN の正しさを測る「物差し」**（[[sources/2021-transformers-can-do-bayesian-inference]]）
   - 固定ハイパーパラメータの GP は PPD（[[bayesian-inference]] の事後予測分布）が**閉形式で厳密に計算できる**数少ない例。だから「PFN の近似がどれだけ正しいか」を厳密な真値と突き合わせて検証できる。
   - 原典論文は、GP 事前分布から合成データを引いて PFN を訓練し、その予測平均・95% 信頼区間が厳密 GP とほぼ区別がつかないことを示した（メタ訓練データセットを増やすほど一致）。これが「Transformer はベイズ推論ができる」ことの最も直接的な実証。
   - ハイパー事前分布（カーネルのスケールや長さスケール上の分布）を付けると GP の PPD はもはや閉形式で解けなくなる。PFN はこの**扱いにくい**ケースでも、MLE-II より 200 倍以上・NUTS より 1000〜8000 倍速く、真の PPD に最も近い近似を出した。

2. **PFN に与える事前分布の一つ**
   - GP 自体を「データ生成器」＝事前分布として使い、PFN を訓練できる。[[sources/2022-tabpfn]] では GP ベースの分類事前分布（PFN-GP）を用意し、ターゲットの中央値でクラス分けして二値分類化している。

## GP と PFN の関係（まとめ）

GP は「カーネルで関数の事前分布を表し、ベイズ推論で予測する」古典的枠組み。PFN は「事前分布から合成したデータで Transformer を訓練し、ベイズ推論を**償却（amortize）**する」新しい枠組み。PFN は GP を**模倣の対象**にも**事前分布の素材**にもできる。GP に似た挙動（観測点から遠いほど不確実性が増す滑らかな予測）は、TabPFN のトイデータ実験でも観察されている。

## 関連ページ

- [[bayesian-inference]] — GP は PPD が厳密に解ける代表例
- [[prior-data-fitted-networks]] — GP を模倣／事前分布に用いる
- [[sources/2021-transformers-can-do-bayesian-inference]] — GP をベンチマークに PFN を検証した原典
- [[sources/2022-tabpfn]] — GP ベースの分類事前分布（PFN-GP）
