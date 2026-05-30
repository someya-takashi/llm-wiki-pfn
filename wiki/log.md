# Log — 時系列ログ（append-only）

`## [YYYY-MM-DD] <kind> | <要点>` 形式で追記する（kind = ingest | query | schema-update）。
`grep "^## \[" log.md | tail -10` で直近の動きを追える。

## [2026-05-30] schema-update | CV→PFN 化・entities 廃止・概念の抽象化・ingest/query/lint を skill 化

- 対象: `CLAUDE.md`, `.claude/skills/{ingest,query,lint}/SKILL.md`, `wiki/{index,log,overview}.md`
- 変更:
  - ドメインを Computer Vision から **PFN（Prior-Data Fitted Networks）** に変更。
  - `wiki/entities/` を廃止（モデル・データセット・人物・組織の専用ページは作らない）。
  - `wiki/concepts/` は**抽象的な概念・カテゴリ単位**でまとめる方針に変更（個別手法は上位概念ページに統合。例: TabPFN → prior-data-fitted-networks / in-context-learning）。
  - ingest / query / lint を `.claude/skills/` 配下の skill として切り出し。翻訳・要約・画像の具体テンプレ（旧 §4/§5/§7）は ingest skill へ移動。横断品質ルール（旧 §6）は CLAUDE.md に残置（§4）。
  - wiki/ raw/ の骨組みと index/log/overview スタブを作成。

## [2026-05-30] schema-update | ingest skill に appendix 取り込みの例外ルールを追加

- 対象: `.claude/skills/ingest/SKILL.md`
- 変更: 翻訳の「何を翻訳するか」に例外を追加。デフォルトは appendix を除外するが、ユーザーが明示的に「appendix も含めて」と指示した場合は付録も一文ずつ翻訳する（references・acknowledgments は除外のまま）。appendix を含めたかは log にメモする。

## [2026-05-30] ingest | TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second

- 取り込み: `raw/articles/TabPFN_ A Transformer That Solves Small Tabular Classification Problems in a Second.md`（arXiv:2207.01848）
- 作成: [[sources/2022-tabpfn]], [[translations/2022-tabpfn]], [[concepts/prior-data-fitted-networks]], [[concepts/in-context-learning]], [[concepts/bayesian-inference]], [[concepts/structural-causal-model]]
- 更新: [[overview]], [[index]]
- メモ:
  - **付録（Appendix A〜F）も含めて翻訳**（ユーザー指示）。references・acknowledgments は除外。
  - 原典の Abstract と一部の図は SVG／base64 画像として埋め込まれていたが、Abstract は実際にはテキスト行で全文翻訳済み。図は SVG/外部 PNG のため本文に再掲せず、キャプションのみ訳出。画像のローカル保存（raw/assets）は今回は実施せず。
  - 概念は抽象レベルで作成（TabPFN 個別ページは作らず prior-data-fitted-networks に代表手法として統合）。
  - dangling link: Müller et al. 2021「Transformers Can Do Bayesian Inference」（PFN 原典）は未 ingest。次の ingest 候補。

## [2026-05-30] schema-update | ingest skill の画像ローカル保存を必須化

- 対象: `.claude/skills/ingest/SKILL.md`「画像の扱い」
- 変更: 原典に画像リンク（外部 URL・data URI・SVG）があれば、`raw/assets/<source-slug>/figN.*` へ**必ずダウンロード／デコードして保存**し、ローカルパスで参照することを必須化（従来は「可能なら」の任意）。取得失敗した画像のみ例外的に元 URL を使い、その旨を log にメモする。
- 影響: [[sources/2022-tabpfn]] / [[translations/2022-tabpfn]] は新ルール導入前の ingest のため図を未保存（SVG/base64/外部 PNG）。要バックフィル（下記で実施）。

## [2026-05-30] ingest-update | TabPFN 図のバックフィル（画像ローカル保存）

- 対象: [[translations/2022-tabpfn]], [[sources/2022-tabpfn]], [[concepts/prior-data-fitted-networks]]
- 保存: `raw/assets/2022-tabpfn/` に図1〜21 を計 32 ファイル保存。
  - 外部 PNG/JPG 14 枚を ar5iv からダウンロード（fig3,4,8,11〜21）。
  - インライン SVG 18 枚を原典 md から抽出（fig1a/1b, fig2a/b/c, fig5_1〜4, fig6, fig7_1〜4, fig9_1〜3, fig10）。
- 差し込み: translations に図1〜21 を `<figure>` + `<figcaption>` 形式で全挿入（21 ブロック）。sources に図1・図5 を再掲、concepts/prior-data-fitted-networks にアテンション図(図1b)を再掲。すべてローカルパス参照、元 URL は残していない。
- メモ: 表 1/2/5/6 は原典では HTML `<table>`（画像ではなくデータ表）。今回の画像バックフィル対象外（本文ではキャプションのみ訳出済み、データ表の完全移植は別途課題）。取得失敗画像なし。

## [2026-05-30] schema-update | ingest skill に PDF 原典の画像取り込み方針を追加

- 対象: `.claude/skills/ingest/SKILL.md`「画像の扱い」
- 変更: 原典の取得経路（ar5iv→Web Clipper で markdown、変換不可なら PDF）を背景として明記し、画像の扱いを 2 ケースに分割。
  - ケース A（markdown 原典 `raw/articles/`）: 画像リンクは必ずローカル保存（従来通り）。
  - ケース B（PDF 原典 `raw/papers/`）: 画像はプログラム的に取得不可なので LLM 側からの取り込みは行わない。ただしユーザーが手動保存した画像のファイルパスを指示した場合は、markdown 同様に `raw/assets/<source-slug>/` にフォルダを作って移動し、ローカルパスで参照する。

## [2026-05-30] ingest | Transformers Can Do Bayesian Inference（PFN 原典, PDF, 付録込み）

- 取り込み: `raw/papers/TRANSFORMERS CAN DO BAYESIAN INFERENCE.pdf`（Müller et al., ICLR 2022, 全 23 ページ）
- 作成: [[sources/2021-transformers-can-do-bayesian-inference]], [[translations/2021-transformers-can-do-bayesian-inference]], [[concepts/gaussian-process]]
- 更新: [[concepts/prior-data-fitted-networks]]（原典への dangling link を解消＝代表手法に追記、sources 追加）, [[concepts/bayesian-inference]], [[concepts/in-context-learning]]（sources 追加・本文に原典の言及）, [[overview]], [[index]]
- メモ:
  - **付録 A〜H も含めて翻訳**（ユーザー指示）。references・acknowledgments は除外。
  - **PDF 原典のため画像は取り込まず**（新ルール ケース B）。図1〜12 はキャプションのテキスト訳のみ。表 1〜7 は主要なもの（表1=表形式集計, 表2=Omniglot）を markdown 表として訳出、他はキャプション要約。
  - 概念は抽象レベルを維持。新規 concept は gaussian-process のみ（PFN の検証ベンチマーク兼事前分布素材として複数 source が参照するため）。リーマン分布は原典固有なので独立ページにせず source/translation 内で説明。
  - これにより前回 ingest（TabPFN）で dangling だった PFN 原典が解消。
