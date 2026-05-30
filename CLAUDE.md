# PFN（Prior-Data Fitted Networks）LLM Wiki — スキーマ

このリポジトリは Andrej Karpathy が提唱した「LLM Wiki」パターンに基づく、**PFN（Prior-Data Fitted Networks; 事前分布からサンプリングした合成データセットで一度だけ訓練し、推論時には重み更新なしに in-context でベイズ推論を近似するニューラルネットワーク）** 領域のパーソナル・ナレッジベースです。TabPFN に代表される「データセットを丸ごと文脈として与えると即座に予測を返す」系のモデル・理論・周辺手法を扱います。LLM（あなた）が wiki を**読み・書き・更新する側**、ユーザーは**情報源のキュレーションと質問**を担当します。ユーザーは wiki を直接編集することはほぼありません。

このファイルは「あなた（LLM）」のための運用ルール書（スキーマ）です。ingest / query / lint の各オペレーションは skill として切り出してあり（§3）、それらの skill は本ファイルのスキーマ規約に従って実行してください。

---

## 1. ディレクトリ構成

```
PFN/
├── CLAUDE.md                ← このファイル（スキーマ）
├── .claude/skills/          ← オペレーション skill（ingest / query / lint）
├── raw/                     ← 原典（immutable, LLM は読むだけ）
│   ├── papers/              ← 論文 PDF（arXiv 等）
│   ├── articles/            ← ブログ記事・Webクリップ等の markdown
│   ├── images/              ← 記事に紐づかない単独画像
│   └── assets/              ← 記事内画像のローカル保存先
└── wiki/                    ← LLM が完全所有する markdown 群
    ├── index.md             ← カタログ（全ページの一覧）
    ├── log.md               ← 時系列ログ（append-only）
    ├── overview.md          ← PFN 全体の総括ページ（随時更新）
    ├── sources/             ← 原典 1 件につき 1 ページの要約・解説
    ├── translations/        ← 原典の全文翻訳（要約せず一文ずつ正確に）
    ├── concepts/            ← 概念・カテゴリ（抽象レベル）の解説ページ
    └── questions/           ← query で得た成果物（比較表・分析等）を保存
```

### 命名規約

- ファイル名は `kebab-case.md`（例: `prior-data-fitted-networks.md`）
- 原典の wiki ページ（sources/translations）は元ファイル名にあわせる（例: `raw/papers/2022-tabpfn.pdf` → `wiki/sources/2022-tabpfn.md`, `wiki/translations/2022-tabpfn.md`）
- **概念ページは「抽象的な概念・カテゴリ」を単位にする**。特定の手法・個別モデルそのものをページにせず、その手法が属する上位概念でまとめる。スラグは概念の英語正式名称（略称ではなくフルネーム）。
  - 例: TabPFN は prior-data-fitted-networks（PFN）/ in-context-learning に属する一手法 → `tabpfn` という個別ページは作らず、`prior-data-fitted-networks` や `in-context-learning` の概念ページ内に「代表手法」として記述する。
- **個別のモデル・データセット・人物・組織の専用ページ（旧 entities）は作らない**。それらは関連する概念ページ、または該当する `sources/` ページ内で言及する。
- 略称（PFN, ICL, PPD 等）は専用ページを作らず、対応する正式名称ページ（prior-data-fitted-networks, in-context-learning, posterior-predictive-distribution）にリダイレクトする位置づけで `index.md` に併記する。

### 言語ポリシー

- **wiki 内の解説文（sources / concepts / questions / overview / index / log）は日本語**で書く
- **translations のみ、原典が英語であれば日本語訳**を作成する（原典が日本語であれば翻訳は作成しない）
- 固有名詞・術語は無理に和訳せず、初出時に「Prior-Data Fitted Networks（PFN, 事前分布からの合成データで訓練し推論時に in-context でベイズ推論を近似するネットワーク）」のように原語＋略称＋短い注釈を添える

### リンク規約

- 内部リンクは Obsidian の `[[wikilink]]` 記法を使う（`[[prior-data-fitted-networks]]` のように slug を直接書く）
- まだ存在しないページへのリンク（dangling link）も許容する。`lint` 時に未作成ページとして検出する
- 原典への参照は `[[sources/2022-tabpfn]]` のようにディレクトリ込みで書く

---

## 2. Frontmatter 規約

すべての wiki ページに YAML frontmatter を付与する。Obsidian Dataview で集計できるようにする。

### sources/*.md

```yaml
---
type: source
source_path: raw/papers/2022-tabpfn.pdf
source_kind: paper            # paper | article | blog | video | podcast
title: "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"
authors: [Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter]
year: 2022
venue: ICLR 2023
ingested: 2026-05-30
tags: [pfn, tabpfn, in-context-learning, tabular, bayesian-inference]
translation: [[translations/2022-tabpfn]]
---
```

### translations/*.md

```yaml
---
type: translation
source_path: raw/papers/2022-tabpfn.pdf
source_page: [[sources/2022-tabpfn]]
original_language: en
translated_to: ja
translated_at: 2026-05-30
---
```

### concepts/*.md

```yaml
---
type: concept
aliases: [PFN, Prior-Data Fitted Networks]
tags: [meta-learning, bayesian-inference, in-context-learning]
related: [[in-context-learning]], [[bayesian-inference]]
sources: [[sources/2021-transformers-can-do-bayesian-inference]], [[sources/2022-tabpfn]]
updated: 2026-05-30
---
```

### questions/*.md

```yaml
---
type: question
asked: 2026-05-30
question: "TabPFN と勾配ブースティング（GBDT）は小規模表データでどちらが強いか？"
sources_used: [[sources/2022-tabpfn]]
---
```

---

## 3. オペレーション（ingest / query / lint）

主要オペレーションは **skill として切り出してある**。実行時は対応する skill の手順に従うこと（手順の本体は各 SKILL.md にあり、本ファイルには重複させない）。

- **ingest（原典の取り込み）** — `.claude/skills/ingest/SKILL.md`
  raw/ に置かれた論文・記事を読み、翻訳ファイル（translations）・要約ページ（sources）・関連する概念ページ（concepts）を作成／更新し、index/log を更新する。**翻訳・要約・画像の具体テンプレと書式（旧 §4/§5/§7）もこの skill 内に置いてある**。
- **query（質問への回答）** — `.claude/skills/query/SKILL.md`
  index から関連ページを辿り、`[[wikilink]]` 引用付きで回答する。必要なら成果物を `questions/` として保存する。
- **lint（健康診断）** — `.claude/skills/lint/SKILL.md`
  矛盾・古い記述・孤立ページ・dangling link・欠落クロスリファレンス・データギャップを点検し、一覧で提示する。

共通方針：

- ingest は基本 **1 件ずつ**、ユーザーと対話しながら進める（1 件の ingest で 5〜15 ページが触られるのが普通）。
- 下記 §4（「機械的なまとめ」にしないルール）は、`translations/` を除くすべてのページ生成、および query の回答に **常時適用**する。

---

## 4. 「機械的なまとめ」にしない（最重要ルール）

`wiki/translations/` 以外のすべてのページ（sources / concepts / questions / overview）および query の回答で守ること：

1. **略称は必ず初出時に展開する**。`ICL` → `ICL（In-Context Learning, 推論時に文脈として与えた例から重み更新なしにタスクを解く枠組み）` のように、**展開＋短い意味付け**をセットで書く。
2. **難概念は補足を入れる**。専門用語（例: posterior predictive distribution, prior, amortized inference, meta-learning, well-calibrated）が出てきたら、その場で 1 文の補足説明を付ける。リンクで概念ページに飛べる場合でも、文脈で必要なら補足を残す。
3. **初学者の読者を想定する**。学部高学年〜修士 1 年程度の機械学習初学者が読んで「何のことを言っているか分からない」段落を作らない。逆に、自明な内容を冗長に説明するのも避ける。
4. **原典の章立てをそのままコピーしない**。要約ページの構造は ingest skill の要約テンプレートに従い、原典の構造を「再解釈」した形にする。
5. **「研究の意義」を自分の言葉で説明する**。原典の Abstract をなぞるだけにしない。「なぜこの結果が PFN／機械学習分野にとって重要なのか」を 1〜2 文加える。
6. **既存 wiki との接続を明示する**。「これは [[bayesian-inference]] を Transformer の forward 一回で近似（amortize）する最初の本格的枠組みであり、[[in-context-learning]] を表形式データに持ち込んだ例である」のように、既存知識と結びつける一文を必ず入れる。

逆に、`wiki/translations/` では上記ルールは **適用しない**。翻訳ファイルでは補足・解釈・接続づけは一切せず、原典に忠実に訳すことに専念する。

---

## 5. index.md と log.md の運用

### index.md

カテゴリ別の全ページカタログ。1 行 = 1 ページ。フォーマット：

```markdown
- [[<slug>]] — <一行の説明>
```

セクション：
- Overview
- Sources（さらに paper / article 等に分けてよい）
- Translations
- Concepts
- Questions

略称リダイレクト（例: `PFN → [[prior-data-fitted-networks]]`）は Concepts セクション内に併記してよい。ingest / query で新規ページを作るたびに必ず更新する。

### log.md

時系列の append-only ログ。

```markdown
## [YYYY-MM-DD] ingest | <タイトル>

- 取り込み: `raw/papers/2022-tabpfn.pdf`
- 作成: [[sources/2022-tabpfn]], [[translations/2022-tabpfn]], [[concepts/prior-data-fitted-networks]]
- 更新: [[concepts/in-context-learning]], [[overview]], [[index]]
- メモ: ...
```

`grep "^## \[" log.md | tail -10` で直近の動きを追えるよう、必ずこのプレフィックス形式を守る。スキーマ変更は `## [YYYY-MM-DD] schema-update | <要点>` で記録する（§7）。

---

## 6. ツール

- **Obsidian**：wiki の閲覧・グラフビュー確認。ユーザーが裏側で開いている。
- **Obsidian Web Clipper**：記事を markdown 化して `raw/articles/` に保存。
- **Marp**：スライド出力が必要な質問への回答に使う（任意）。
- **Dataview**：frontmatter ベースの動的集計（任意）。

検索は規模が小さいうちは index.md ベースで十分。ページ数が増えてきたら `qmd` 等の導入を検討する。

---

## 7. このスキーマ自体について

このスキーマはユーザーと共進化する。運用していく中で「このカテゴリが必要」「このルールは緩めたい」となったら、ユーザーに提案してこの CLAUDE.md（およびオペレーション手順を持つ `.claude/skills/` 配下の SKILL.md）を更新する。スキーマ変更は log.md にも `## [YYYY-MM-DD] schema-update | <要点>` として記録する。
