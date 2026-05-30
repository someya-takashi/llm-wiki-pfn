# Index — PFN LLM Wiki カタログ

全ページのカタログ。1 行 = 1 ページ（`- [[<slug>]] — <一行の説明>`）。ingest / query で新規ページを作るたびに更新する。

## Overview

- [[overview]] — PFN（Prior-Data Fitted Networks）領域の総括

## Sources

### paper

- [[sources/2021-transformers-can-do-bayesian-inference]] — Transformers Can Do Bayesian Inference: PFN の枠組みを提示した原典（Müller et al. 2021, ICLR 2022）
- [[sources/2022-tabpfn]] — TabPFN: 小規模表形式分類を 1 秒で解く Transformer（Hollmann et al. 2022）

## Translations

- [[translations/2021-transformers-can-do-bayesian-inference]] — PFN 原典の全文翻訳（本文＋付録 A〜H）
- [[translations/2022-tabpfn]] — TabPFN 論文の全文翻訳（本文＋付録 A〜F）

## Concepts

- [[prior-data-fitted-networks]] — 合成データで一度訓練し推論時にベイズ推論を償却近似する中核概念
- [[in-context-learning]] — 重み更新なしに文脈から予測する枠組み
- [[bayesian-inference]] — 近似対象である事後予測分布（PPD）と償却推論
- [[structural-causal-model]] — TabPFN の表形式事前分布の中核（因果 DAG による合成データ生成）
- [[gaussian-process]] — PFN の検証ベンチマーク兼事前分布の素材となる古典的ベイズモデル

略称リダイレクト（正式名称ページに集約）:
- PFN → [[prior-data-fitted-networks]]
- ICL → [[in-context-learning]]
- PPD / 事後予測分布 → [[bayesian-inference]]
- SCM → [[structural-causal-model]]
- GP → [[gaussian-process]]

## Questions

（まだありません）
