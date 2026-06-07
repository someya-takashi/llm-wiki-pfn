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

## [2026-05-30] ingest | TabPFN v2: Accurate predictions on small data with a tabular foundation model（Nature 2025）

- 取り込み: `raw/articles/Accurate predictions on small data with a tabular foundation model.md`（Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6）
- 作成: [[sources/2025-tabpfn-v2]], [[translations/2025-tabpfn-v2]], [[concepts/tabular-foundation-model]]
- 更新: [[concepts/prior-data-fitted-networks]]（代表手法に v2 追記、v1→v2 系譜、sources/related 追加）, [[concepts/structural-causal-model]], [[concepts/in-context-learning]]（sources 追加・本文に v2 言及）, [[overview]], [[index]]
- 図保存: markdown 原典（ケース A）のため **Fig.1〜6 を `raw/assets/2025-tabpfn-v2/fig1〜6.png` にローカル保存**（Springer 外部 URL から、webp ではなく PNG で取得）。translations に 6 図を `<figure>` で全挿入、sources に図1・図6 を再掲。
- メモ:
  - **翻訳スコープは Main＋Methods のみ**（ユーザー指示、appendix 明示なし）。Extended Data 図表・Supplementary・References・Acks・Ethics・Peer review・Rights は対象外。
  - スラグは既存 `2022-tabpfn`（v1）と区別して `2025-tabpfn-v2`。
  - 新規 concept は tabular-foundation-model のみ（本論文が中心的に打ち出す抽象概念で、予測＋生成＋密度推定＋埋め込みを単一モデルで担う枠組み）。TabPFN 個別ページは作らず prior-data-fitted-networks / source に集約（抽象化方針）。
  - 系譜が原典(2021)→v1(2022)→v2(2025) で揃った。

## [2026-05-30] ingest | TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models（arXiv 2025）

- 取り込み: `raw/articles/TabPFN-2.5 Advancing the State of the Art in Tabular Foundation Models.md`（Prior Labs, arXiv:2511.08667, 2025）
- 作成: [[sources/2025-tabpfn-2-5]], [[translations/2025-tabpfn-2-5]]
- 更新: [[concepts/tabular-foundation-model]]（蒸留・スケール・エコシステム・因果応用を追記、sources/related 追加）, [[concepts/prior-data-fitted-networks]]（代表手法に 2.5 追記）, [[concepts/in-context-learning]]（sources＋thinking rows/スケール言及）, [[concepts/structural-causal-model]]（因果推論応用の節を追加）, [[overview]], [[index]]
- 図保存: markdown 原典（ケース A）のため **20 枚を `raw/assets/2025-tabpfn-2-5/` にローカル保存**（ar5iv 外部 PNG、元アセット名 x1〜x29/prior-logo/realcause_pretty/TabPFN_workflow を維持＝図番号非連続のため取り違え防止）。translations に 20 図を `<figure>` で挿入、sources に図1・図9 を再掲。
- メモ:
  - **翻訳は本文 §1〜7 ＋ 付録 A〜I を含む**（ユーザー指示で appendix 込み）。ただし **Appendix B（公開100件の応用事例カタログ）は件数サマリのみ**（Healthcare 51/Finance 3/Energy 15/Manufacturing 12/Other 19）。Appendix A（貢献者）は氏名そのまま・役職見出しのみ訳。References/Acknowledgements 節は原典に無し。
  - スラグは既存 `2025-tabpfn-v2`（Nature 版）と区別し `2025-tabpfn-2-5`（ドット不使用）。
  - 新規 concept は作らず既存に集約（蒸留・因果推論は source ＋ tabular-foundation-model 内で説明）。
  - 系譜が 原典(2021)→v1(2022)→v2(2025 Nature)→2.5(2025 arXiv) に伸びた。

## [2026-05-30] ingest | TabPFN-3 Technical Report（arXiv 2026, PDF）

- 取り込み: `raw/papers/TabPFN-3- Technical Report.pdf`（Prior Labs Team, arXiv:2605.13986, 2026-05-12, 全82ページ）
- 作成: [[sources/2026-tabpfn-3]], [[translations/2026-tabpfn-3]]
- 更新: [[concepts/tabular-foundation-model]]（テスト時計算 Thinking・1M スケール・関係/時系列/テキスト/因果の波及を追記）, [[concepts/prior-data-fitted-networks]]（代表手法に 3 追記＋アーキが v1 流 ICL へ回帰した点）, [[concepts/in-context-learning]]（1M スケール・QASSMax・テスト時計算）, [[concepts/structural-causal-model]]（SCM prior 拡張＝時間DSCM/空間/OOD、QINI 因果）, [[overview]], [[index]]
- 図保存: PDF 原典＋**ユーザー提供画像（ケース B）**。`raw/images/fig1〜22.png`（22枚＝本文 Figure 1〜22 に 1:1 対応）を `raw/assets/2026-tabpfn-3/` へ**移動**。translations に 22 図を `<figure>` で挿入、sources に図1・図5 を再掲。`raw/images/` は .gitkeep のみに復帰。
- メモ:
  - **翻訳は本文 §1〜5 のみ**（ユーザー appendix 指定なし＋提供図が本文図と一致）。References（p25〜）・Appendix（p45〜82, 図23以降は未提供）は対象外。
  - スラグは原典日付（2026-05-12）に基づき `2026-tabpfn-3`（2025 系と区別）。
  - 新規 concept は作らず既存へ集約（テスト時計算 Thinking / 関係 RFM は source ＋ tabular-foundation-model で説明。将来 source 増で昇格検討）。
  - 系譜が 原典(2021)→v1(2022)→v2(2025 Nature)→2.5(2025)→**3(2026)** に到達。

## [2026-05-30] ingest | TabICL: A Tabular Foundation Model for In-Context Learning on Large Data（ICML 2025）

- 取り込み: `raw/articles/TabICL_ A Tabular Foundation Model for In-Context Learning on Large Data.md`（Qu, Holzmüller, Varoquaux, Le Morvan / Inria Soda, arXiv:2502.05564, ICML 2025）
- 作成: [[sources/2025-tabicl]], [[translations/2025-tabicl]]
- 更新: [[concepts/tabular-foundation-model]]（別系統の兄弟 TFM として TabICL を代表例に追加、3 段アーキが TabPFN-3 に採用された点）, [[concepts/prior-data-fitted-networks]]（PFN 流の兄弟として TabICL を追記）, [[concepts/in-context-learning]]（sources 追加）, [[concepts/structural-causal-model]]（木ベース SCM 事前分布＋活性化多様化を追記）, [[overview]], [[index]]
- 図保存: markdown 原典（ケース A）。**23 枚を `raw/assets/2025-tabicl/` にローカル保存**（ar5iv 外部 PNG、元アセット名 x1〜x37 を維持＝図番号非連続のため取り違え防止）。translations に 23 図を `<figure>` で挿入、sources に図1・図4 を再掲。
- メモ:
  - **本文 §1〜6 ＋ 付録 A〜E を訳出**（ユーザー指示で appendix 含む）。Acknowledgements・References は対象外。
  - スラグは arXiv 2502（2025年2月）に基づき `2025-tabicl`。
  - **TabPFN とは別グループ（Inria）の兄弟 TFM**。3 段「行圧縮 ICL」アーキテクチャを提案し、後に TabPFN-3（[[sources/2026-tabpfn-3]]）が "TabICLv2 ベース" として採用 → 既存ページの「TabPFN-3 が v1 流 ICL へ回帰」記述と接続。
  - 新規 concept は作らず既存へ集約（Set Transformer・RoPE・カリキュラム学習は source/translation 内で説明）。

## [2026-05-31] ingest | TabICLv2: A better, faster, scalable, and open tabular foundation model（ICML 2026）

- 取り込み: `raw/articles/TabICLv2_ A better, faster, scalable, and open tabular foundation model.md`（Qu, Holzmüller, Varoquaux, Le Morvan / Inria, arXiv:2602.11139, ICML 2026）
- 作成: [[sources/2026-tabicl-v2]], [[translations/2026-tabicl-v2]]
- 更新: [[concepts/tabular-foundation-model]]（TabICL を v1/v2 系譜に拡張＋関連リンク）, [[concepts/prior-data-fitted-networks]]（同）, [[concepts/in-context-learning]]（attention fading と QASSMax の節を追加）, [[concepts/structural-causal-model]]（TabICLv2 の 8 種ランダム関数・random Cauchy graph・フィルタリングを追記）, [[overview]], [[index]]
- 図保存: markdown 原典（ケース A）。**69 枚を `raw/assets/2026-tabicl-v2/` にローカル保存**（ar5iv 外部 PNG、元アセット名 x1〜x96 を維持＝図番号非連続のため取り違え防止）。
- メモ:
  - **本文 §1〜9 ＋ 付録 A〜K を全訳（全 69 図埋め込み）**（ユーザー指示の最大忠実度オプション。J/K のベンチ結果ギャラリーも全掲載）。Acknowledgements・Contribution/Impact Statement・References は対象外。
  - スラグは arXiv 2602（2026年2月）に基づき `2026-tabicl-v2`（v1 の 2025-tabicl と区別）。
  - **これは [[sources/2025-tabicl]] の後継であり、[[sources/2026-tabpfn-3]] が「TabICLv2 ベース」と明言したアーキテクチャ本体**。PFN 系 2 系統（Prior Labs / Inria）の交差点を解消。
  - 3 本柱（新 prior・アーキ革新〔repeated feature grouping/target-aware embedding/QASSMax/mixed-radix/分位点回帰〕・事前訓練〔Muon〕）、回帰対応、オープンウェイト。
  - 新規 concept は作らず既存へ集約（QASSMax/Muon/分位点回帰/mixed-radix は source/translation で説明）。

## [2026-05-31] ingest | Real-TabPFN: Improving Tabular Foundation Models via Continued Pre-training With Real-World Data（arXiv 2025）

- 取り込み: `raw/articles/Real-TabPFN_ Improving Tabular Foundation Models via Continued Pre-training With Real-World Data.md`（Garg, Ali, Hollmann, Purucker, Müller, Hutter / Freiburg・Prior Labs, arXiv:2507.03971）
- 作成: [[sources/2025-real-tabpfn]], [[translations/2025-real-tabpfn]]
- 更新: [[concepts/tabular-foundation-model]]（「実データへの適応（継続事前訓練）」節を追加、sources）, [[concepts/prior-data-fitted-networks]]（sources）, [[overview]], [[index]]
- 図保存: markdown 原典（ケース A）。**6 枚を `raw/assets/2025-real-tabpfn/` にローカル保存**（x1〜x5 ＋ all_data_sizes_720p、元アセット名維持）。translations に 6 図を `<figure>` で挿入、sources に図1 を再掲。
- メモ:
  - **本文（§1, §3〜§6）のみ訳出**（ユーザー appendix 指定なし）。Acknowledgements・References・Appendix A〜C は対象外。原典 ar5iv クリップに §2 は含まれず。
  - スラグは arXiv 2507（2025年7月）に基づき `2025-real-tabpfn`。
  - **合成事前訓練済み TabPFNv2 を厳選実データで継続事前訓練 → Real-TabPFN**。L2-SP 正則化で破滅的忘却を回避。発展形 RealTabPFN-2.5 は TabPFN-3/TabICLv2 の主要比較対象（既出ページの記述と接続）。
  - 新規 concept は作らず既存に集約（continued pre-training は tabular-foundation-model の適応戦略として記述）。

## [2026-05-31] schema-update | frontmatter の wikilink を Obsidian 有効形式に修正

- 原因: frontmatter の `related`/`sources`/`sources_used`/`translation`/`source_page` を裸の `[[..]]`（カンマ区切り）で書いていた。YAML では入れ子配列と解釈され Obsidian が「無効なプロパティ」と表示していた（concepts の閲覧時に発覚）。
- 修正: 全 wiki md（concepts 6・sources 8・translations 8 ＝計 22 ファイル）の frontmatter を、リンク複数フィールドは YAML ブロックリスト `  - "[[..]]"`、単一リンクは `key: "[[..]]"` に変換（/tmp/fix_frontmatter.py で一括）。in-context-learning の sources 重複（2025-tabicl 2 回）も解消。
- 再発防止: **CLAUDE.md §2** に「frontmatter 内 wikilink は必ず引用符で囲む（裸 `[[..]]` は無効プロパティ）」ルールを追記し、4 つの frontmatter 例を修正。ingest skill の翻訳テンプレも修正。
- 本文（frontmatter 外）の `[[wikilink]]` は従来どおり引用符なしで変更なし。

## [2026-05-31] ingest | An Intuitive Tutorial to Gaussian Process Regression（arXiv 2020）

- 取り込み: `raw/articles/An Intuitive Tutorial to Gaussian Process Regression.md`（Jie Wang / University of Waterloo, arXiv:2009.10862）
- 作成: [[sources/2020-gp-regression-tutorial]], [[translations/2020-gp-regression-tutorial]]
- 更新: [[concepts/gaussian-process]]（本チュートリアルをリファレンスとして導入文・予測式・関連リンクに追記、sources 拡充）, [[index]]
- 図保存: markdown 原典（ケース A）。**12 枚を `raw/assets/2020-gp-regression-tutorial/` にローカル保存**（x1,x3,x4,x5,x7,x9,x10,x12,x13,x16 ＋ 200d_gaussian_kernel_prior, predictions）。translations に 12 図を `<figure>` で挿入、sources に図10 を再掲。
- メモ:
  - **本文（MATHEMATICAL BASICS〜CONCLUSION）のみ訳出**（ユーザー appendix 指定なし）。ACKNOWLEDGMENTS・REFERENCES は対象外。
  - スラグは arXiv 2009（2020年9月）に基づき `2020-gp-regression-tutorial`。これは TFM 論文でなく **[[gaussian-process]] 概念のリファレンス**（GP は PFN が近似する対象・合成 prior の素材として頻出）。
  - 新規 concept は作らず既存の gaussian-process を充実。index は Sources に "article / tutorial" サブセクションを新設。
  - 作業中に index の翻訳一覧で real-tabpfn 行が重複したため除去（catalog は一意化）。

## [2026-05-31] ingest | A Tutorial on Bayesian Optimization（arXiv 2018）

- 取り込み: `raw/articles/A Tutorial on Bayesian Optimization.md`（Peter I. Frazier / Cornell University, arXiv:1807.02811）
- 作成: [[sources/2018-bayesian-optimization-tutorial]], [[translations/2018-bayesian-optimization-tutorial]], [[concepts/bayesian-optimization]]（新規概念）
- 更新: [[concepts/gaussian-process]]（GP の最大応用＝BayesOpt サロゲートを追記、related/sources 拡充）, [[concepts/tabular-foundation-model]]（ベイズ最適化を下流応用としてリンク、GIT-BO で TabPFNv2 をサロゲートに、related/sources 追加）, [[overview]], [[index]]
- 図保存: markdown 原典（ケース A）。**3 枚を `raw/assets/2018-bayesian-optimization-tutorial/` にローカル保存**（Animation1, vary_hypers, EIcontour）。translations に 3 図を `<figure>` で挿入。
- メモ:
  - **本文 §1〜7 のみ訳出**（ユーザー appendix 指定なし）。Acknowledgments・References は対象外。スラグは arXiv 1807（2018年7月）→ `2018-bayesian-optimization-tutorial`。
  - **新規 concept [[bayesian-optimization]] を作成**（抽象的方法論で既存概念と区別される。GP チュートリアルが [[gaussian-process]] に集約できたのと異なり、BO は「サロゲート＋獲得関数」という独立した枠組み）。EI/KG/ES/PES・エキゾチック拡張・探索 vs 活用を解説。
  - PFN 文脈での位置づけ: BayesOpt は TFM の**下流応用先**（GP サロゲートの $O(N^3)$・次元制約を TabPFN が埋めうる／GIT-BO が TabPFNv2 採用）。§3 GP 回帰は [[sources/2020-gp-regression-tutorial]] と対をなす。
  - QASSMax 等と同様、EI/KG/ES/PES・多忠実度などは独立ページを作らず concept/source 内で説明（過剰生成回避）。

## [2026-05-31] ingest | An Introduction to Gaussian Process Models（arXiv 2021）

- 取り込み: `raw/articles/An Introduction to Gaussian Process Models.md`（Thomas Beckers / TU München, Chair of Information-oriented Control, arXiv:2102.05497）
- 作成: [[sources/2021-gp-models-intro]], [[translations/2021-gp-models-intro]]
- 更新: [[concepts/gaussian-process]]（上級リファレンスとして導入文・関連リンク追記、RKHS/モデル誤差境界/GP動的モデルを明示、sources 拡充）, [[index]]
- 図保存: markdown 原典（ケース A）。**5 枚を `raw/assets/2021-gp-models-intro/` にローカル保存**（x1=図1, x2=図2, x4=図4, x9=図9, x10=図10）。translations に 5 図を `<figure>` で挿入、sources に図4 を再掲。
- メモ:
  - **本文 §1〜5 のみ訳出**（ユーザー appendix 指定なし）。Appendix A（Conditional Distribution）・References は対象外。スラグは arXiv 2102（2021年2月）→ `2021-gp-models-intro`（既存 `2021-transformers-can-do-bayesian-inference` と非衝突）。
  - **新規 concept は作らず既存 [[gaussian-process]] を充実**（2 本目の GP リファレンス）。[[sources/2020-gp-regression-tutorial]]＝初級（直感の積み上げ）に対し、本稿＝上級（カーネルトリック→RKHS→モデル誤差の保証付き上界→カーネル動物園→GP-SSM/GP-NOE）と棲み分け。
  - PFN 文脈での意義: §2.5 の「GP 予測分散＝真の関数との誤差を確率付きで抑える保証」（情報利得 $\gamma_{\max}$ を含む Srinivas らの境界）が、PFN/TFM の「較正のよい予測分布」（[[bayesian-inference]]）の規範例。情報理論的境界は [[bayesian-optimization]] の GP-UCB と同系譜なので source に相互リンク。
  - 図 3・5・6・7・8 は原典 markdown に画像が含まれない（クリッパー未取得）ため、translation ではキャプションのみ訳出。RKHS/カーネル/モデル誤差は独立ページを作らず source/translation 内で説明（過剰生成回避）。

## [2026-05-31] ingest | Gaussian Processes, not quite for dummies（The Gradient 2019）

- 取り込み: `raw/articles/Gaussian Processes, not quite for dummies.md`（Yuge Shi / University of Oxford, The Gradient, thegradient.pub/gaussian-process-not-quite-for-dummies/）
- 作成: [[sources/2019-gp-not-for-dummies]], [[translations/2019-gp-not-for-dummies]]
- 更新: [[concepts/gaussian-process]]（3 本目＝最も直感寄りの「最初の一冊」リファレンスとして導入文・関連リンク・sources に追記）, [[index]]
- 図保存: **原サイト thegradient.pub から再取得**して 44 枚を `raw/assets/2019-gp-not-for-dummies/` にローカル保存（PNG 37・アニメ GIF 7）。translations に全 44 図を `<figure>` で document 順に配置、sources に代表図1枚を再掲。
- メモ:
  - **arXiv でなく Web ブログ**。Obsidian Web Clipper が本文中の図をほぼ取りこぼし（クリップ済み md には実画像1枚＋YouTube リンクのみ）。ユーザー確認のうえ**原ページ HTML（user-images.githubusercontent.com/18204038/*）から content 図のみを document 順で抽出して再取得**。サイト chrome（logo/icon/tracking/関連記事サムネ/Pinterest/feature 画像）は除外。配置は各図の直前テキストを HTML から抽出して原文脈にマッピング。
  - YouTube（Turner 講演 `watch?v=92-98SYOdlY`）は画像でなく動画なので DL せず参照リンク扱い。
  - 本ケースは**今回限りの対応**。ユーザー指示により `.claude/skills/ingest/SKILL.md` は変更せず据え置き（ブログ画像方針はスキルに追記しない）。
  - **本文のみ訳出**（Author Bio/Acknowledgments/Citation/購読案内は対象外）。スラグは The Gradient 2019 → `2019-gp-not-for-dummies`。
  - **新規 concept は作らず既存 [[gaussian-process]] を充実**（3 本目）。直感（2019）→ 体系的入門（2020）→ 上級/RKHS（2021）の三段リファレンスが揃った。index プロット＝条件付けで回帰を可視化する点が独自価値。PFN が近似する「観測に近いほど不確実性が小さい PPD」の視覚的原型として [[bayesian-inference]] に接続。

## [2026-05-31] ingest | Introduction to Gaussian process regression, Part 1: The basics（Medium 2022）

- 取り込み: `raw/articles/Introduction to Gaussian process regression, Part 1_ The basics.md`（Kaixin Wang / Data Science at Microsoft, Medium, 2022-10-04）
- 作成: [[sources/2022-gpr-part1-basics]], [[translations/2022-gpr-part1-basics]]
- 更新: [[concepts/gaussian-process]]（4 本目＝「実践/応用」リファレンスとして導入文・関連リンク・sources に追記）, [[index]]
- 図保存: Medium（ケース A）。**12 枚を `raw/assets/2022-gpr-part1-basics/` にローカル保存**（式画像 8・プロット 4＝図1〜4）。miro CDN の `format:webp` を外して原 PNG を取得（全数 PNG 確認）。translations に全 12 図を `<figure>` で配置、sources に図2 を再掲。
- メモ:
  - **Medium 記事**。クリップ済み md に画像 URL（miro.medium.com）が inline で残っていたため取得は容易（ユーザー指示「画像は柔軟に対応」に沿い `format:webp` を外して原 PNG をローカル保存）。
  - **本文のみ訳出**。装飾的カバー写真（Unsplash）、本文途中の購読ウィジェット「Get Kaixin Wang's stories in your inbox」、末尾 LinkedIn/次稿誘導/References は対象外。数式は原典が画像埋め込みのため式画像をローカル保存し訳注で式番号・内容を付した。
  - スラグは Medium 2022・Part 1 → `2022-gpr-part1-basics`。続編 Part 2（コンクリート強度予測）は未取り込み。
  - **新規 concept は作らず既存 [[gaussian-process]] を充実**（4 本目＝実践/応用層）。直感（2019）→ 体系的入門（2020）→ 実践/応用（2022）→ 上級理論（2021）の四段リファレンスが揃った。独自価値＝GPflow でのカーネル選択（線形/RBF/和）と過学習 vs 未学習のハイパラ調整。「カーネル選択＝事前分布の設計」「較正された信頼区間」を実務感覚で見せ、[[prior-data-fitted-networks]] / [[bayesian-inference]] に接続。

## [2026-05-31] schema-update | ingest skill に Web 記事の画像取り込み方針（ケース C）を追加

- 変更ファイル: `.claude/skills/ingest/SKILL.md`
- 背景: 非 arXiv の Web 記事（Medium / The Gradient / 個人ブログ等）では Obsidian Web Clipper が画像をうまく取り込めないことが多い（図の欠落、動画リンク化、装飾/UI 画像の混入、CDN 変換 URL）。直近 3 件（[[sources/2019-gp-not-for-dummies]]＝原ページ HTML から図44枚を再取得、[[sources/2022-gpr-part1-basics]]＝miro CDN の format:webp を外して原 PNG 取得、[[sources/2018-bayesian-optimization-tutorial]] 他）で都度対応していたものを恒久ルール化。
- 追加・更新内容:
  - 「原典の取得経路（背景）」を 2 ケース → **3 ケース**（A: arXiv/ar5iv markdown、B: PDF、**C: Web 記事・ブログ**）に拡張。Web Clipper の画像取りこぼしを前提とする旨を明記。
  - **ケース C を新設**: クリップ済み md の画像を点検 → 欠落時は原ページ（frontmatter `source:` URL）から document 順で再取得＋直前テキストで位置マッピング、コンテンツ図と chrome（ロゴ/トラッキング/関連記事サムネ/SNS共有/購読ウィジェット/装飾カバー）の区別と除外、CDN 変換 URL の原画像化（Medium `format:webp` を外す等）、動画埋め込みは参照リンク扱い、数式画像はローカル保存＋訳注、log への記録。ユーザー指示が優先。
  - ケース A を「arXiv（ar5iv）由来」と明示し、`?as=webp` を外す・元アセット名保持・bash 3.2 の連想配列非対応（一時ファイル＋while ループ）等の実務ノウハウを追記。
  - 翻訳の「除外（デフォルト）」に **Web 記事/ブログ特有の定型要素**（購読/フォロー誘導、Author Bio、Citation/BibTeX、関連記事/次稿誘導、SNS 共有/フッター）を追加。

## [2026-05-31] ingest | Introduction to Gaussian process regression, Part 2: Application to predicting concrete strength（Medium 2022）

- 取り込み: `raw/articles/Introduction to Gaussian process regression, Part 2_ Application to predicting concrete strength.md`（Kaixin Wang / Data Science at Microsoft, Medium, 2022-10-11）。[[sources/2022-gpr-part1-basics]] の続編
- 作成: [[sources/2022-gpr-part2-concrete]], [[translations/2022-gpr-part2-concrete]]
- 更新: [[concepts/gaussian-process]]（応用編として sources・関連リンクに追記）, [[sources/2022-gpr-part1-basics]]（続編への相互リンク追加）, [[index]]
- 図保存: Medium（ケース C）。**9 枚を `raw/assets/2022-gpr-part2-concrete/` にローカル保存**（表1・標準化式・図1〜7）。miro CDN の `format:webp` を外して原 PNG を取得（全数 PNG 確認）。translations に全 9 図を `<figure>` で配置、sources に図5 を再掲。
- メモ:
  - 新スキル**ケース C**（Web 記事）の手順に沿って取り込み。クリップ済み md に画像 URL が inline で残っていたため再取得は不要だったが、装飾カバー写真（Unsplash）・購読ウィジェット・References は除外、`format:webp` 除去で原 PNG 化、と方針どおり処理。
  - **本文のみ訳出**。スラグは Medium 2022・Part 2 → `2022-gpr-part2-concrete`。
  - 内容: UCI コンクリート圧縮強度（約1000試料・化学組成7特徴量→MPa）に線形＋RBF カーネルの GPR を GPflow で適用。テスト R²≈0.90/RMSE≈5.4 で **ANN（テスト R²≈0.92）に匹敵＋95% 信頼区間**。SHAP（kernel explainer）でセメント最重要・材齢二次・水セメント比の効きを解釈。
  - **新規 concept は作らず既存 [[gaussian-process]] に「応用」事例として集約**（GP リファレンスは入門3＋上級1＋応用1）。PFN 文脈の接続: コンクリート強度は**表形式回帰の定番ベンチ＝TabPFN の土俵**で、「GPR が ANN 並み＋較正された不確実性」は TabPFN の価値命題の GPR 版。SHAP 解釈は [[tabular-foundation-model]] の「TabPFN-3 が KV キャッシュで SHAP を 120× 高速化」と同主題（source 側からリンク）。

## [2026-05-31] ingest | An Intuitive Introduction to Gaussian Processes（fanpu.io 2025）

- 取り込み: `raw/articles/An Intuitive Introduction to Gaussian Processes.md`（Fan Pu Zeng / 個人ブログ fanpu.io, 2025-01-21）
- 作成: [[sources/2025-gp-intuitive-intro]], [[translations/2025-gp-intuitive-intro]]
- 更新: [[concepts/gaussian-process]]（6 本目＝「NN 接続つき直感」リファレンスとして導入文・関連リンク・sources に追記）, [[index]]
- 図保存: 個人ブログ（ケース C）。**7 枚を `raw/assets/2025-gp-intuitive-intro/` にローカル保存**（webp 4＝gp-prior/gp-posterior/coin-flip/covariance_viz、アニメ GIF 3＝gp_random_sampling/gp_most_uncertain_sampling/gp_nvda）。クリップ済み md に画像 URL（fanpu.io 直リンク）が inline で残っていたためそのまま取得（CDN 変換なし）。translations に全 7 図を `<figure>` で配置、sources に図4 を再掲。
- メモ:
  - ケース C（Web 記事）方針で処理。クリッパーの取りこぼしはなく、原ページ再取得は不要だった。コードの `<iframe>`（Jupyter ノートブック埋め込み）は画像でないため DL せず、source に参照リンクとして記載（本文には訳注で明示）。
  - **本文のみ訳出**。スラグは fanpu.io 2025 → `2025-gp-intuitive-intro`。
  - **新規 concept は作らず既存 [[gaussian-process]] を充実**（6 本目）。GP リファレンスは 直感3（2019/2020/2025）＋実践応用2（2022 part1/part2）＋上級理論1（2021）の構成に。
  - 独自価値＝**ノンパラメトリック性**の強調と、**「無限幅 NN ＝ GP（NNGP）」**の予告。後者は PFN（[[prior-data-fitted-networks]]）が NN でベイズ推論を償却近似できることの土壌として示唆的なので、source で [[prior-data-fitted-networks]] / [[bayesian-inference]] に接続。図4 の不確実性優先サンプリングは能動学習＝獲得関数の直感として [[bayesian-optimization]] にも接続。

## [2026-05-31] query | ガウス過程の直感的説明記事（数式を極力使わない）

- 問い: 「ガウス過程を数式を極力使わずに直感的に説明すると？」
- 参照: GP 入門 6 本（[[sources/2019-gp-not-for-dummies]], [[sources/2020-gp-regression-tutorial]], [[sources/2021-gp-models-intro]], [[sources/2022-gpr-part1-basics]], [[sources/2022-gpr-part2-concrete]], [[sources/2025-gp-intuitive-intro]]）＋概念 [[gaussian-process]]。外部資料として Distill「A Visual Exploration of Gaussian Processes」(Görtler et al. 2019) を発想の下敷きに参照（図は引用せず自作）。
- 作成: [[questions/gaussian-process-intuitive-explainer]]（このリポジトリ初の questions ページ）
- 更新: [[index]]（Questions セクションを新設・1 行追記）
- 図: **新規作成 5 枚**を `raw/assets/gaussian-process-intuitive-explainer/` に保存（f1-many-curves / f2-prior-samples / f3-lengthscale / f4-prior-to-posterior / f5-uncertainty）。1 次元 RBF カーネルの素朴な GP を numpy で実装し matplotlib で描画（Hiragino Sans で和文表示）。**再掲 5 枚**は既存 source のアセットを「図（再掲・出典）」つきで引用（2019 index プロット / 2025 covariance_viz・active-learning gif / 2022 part1 カーネル比較・part2 95%CI）。
- メモ:
  - ユーザー確定方針に沿って実施: 形式＝Markdown 記事、図＝再掲＋新規作成の併用、**PFN/TabPFN への接続は入れず GP の説明に徹する**、Web も積極参照。
  - 数式は本文で極力排し、図と言葉で 7 ステップ（あり得る関数の束→関数上の分布→点で見れば多変量正規→カーネル＝事前の信念→事前から事後への絞り込み→較正された不確実性→使いどころと限界）に再構成。厳密な式は各 source へのリンクに委譲。
  - 環境メモ: Homebrew Python は PEP 668 で外部管理のため、`/tmp/gpvenv` に venv を作り numpy/matplotlib を導入して作図（システム環境は変更せず）。和文フォントは `/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc`。
  - 生成図は原典でなく本記事用の生成物だが、画像参照規約に合わせ `raw/assets/<slug>/` 配下に保存した。

## [2026-05-31] query | GP サンプリング（コレスキー法）の補足を explainer に追加

- 問い: [[questions/gaussian-process-intuitive-explainer]] の「関数を引く（サンプリングする）」をもっと詳しく。
- 更新: [[questions/gaussian-process-intuitive-explainer]] に §3 直後の無番号セクション「補足：関数を“引く”とは具体的にどうするか（サンプリングの仕組み）」を追加。グリッド→共分散行列 K→コレスキー分解 K=LLᵀ→素のノイズ z→f=Lz（事後も同手順）の 5 ステップと、「L が独立ノイズを相関させて滑らかにする」直感を説明。実務メモ（ジッタ・O(n³)）も追記。§4〜§8 の番号は維持（無番号セクションで挿入）。
- 図: **新規作成 1 枚** `raw/assets/gaussian-process-intuitive-explainer/f6-cholesky-sampling.png`（左＝独立な z、右＝Lz で成形、左右で色対応）。venv `/tmp/gpvenv` の numpy/matplotlib＋Hiragino で作図。
- 参照: [[sources/2020-gp-regression-tutorial]]（コレスキー標準アルゴリズム）, [[sources/2021-gp-models-intro]]（ジッタ・スケール）, [[sources/2019-gp-not-for-dummies]]（index プロット）。
- メモ: 元記事は数式を極力排す方針だが、本補足はユーザーの「サンプリングを詳しく」という要望に応じ inline 数式（K=LLᵀ, f=Lz）を最小限使用。PFN への接続は引き続き入れない。

## [2026-05-31] query | PFN 原典とガウス過程の関係

- 問い: 「PFN の論文とガウス過程の関係を教えてください」
- 参照: [[sources/2021-transformers-can-do-bayesian-inference]]（PFN 原典）, [[gaussian-process]]（「なぜ PFN の文脈で重要か」節）, [[prior-data-fitted-networks]], [[bayesian-inference]], [[sources/2022-tabpfn]]（PFN-GP）。
- 作成: [[questions/pfn-paper-and-gaussian-process]]（2 件目の questions ページ）
- 更新: [[index]]（Questions に 1 行追記）
- 内容: 関係を 3 本柱で整理——(1) 厳密 GP を PFN の検証「物差し」に（図3、予測平均・95%CI が厳密 GP とほぼ一致）、(2) GP を PFN の事前分布＝データ生成器に、(3) ハイパー事前分布付き GP で PPD が解けなくなる場面を PFN が高速近似（MLE-II 比 200x・NUTS 比 1000〜8000x、BNN は SVI 比 1000x・NUTS 比 10000x）。GP vs PFN の比較表（分布の対象/推論方法/事前分布の表現/不確実性/学習/計算コスト/適用範囲）付き。
- メモ: ユーザー指示どおり「3 つの関係＋比較表」構成。新規作図はせず、GP 側の直感は [[questions/gaussian-process-intuitive-explainer]] へリンク。数値は原典 source と一致を確認。

## [2026-05-31] query | GP 予測の平均・分散の計算（条件付け）と帯の描き方の補足を explainer に追加

- 問い: データ点が与えられたとき未観測点の平均・分散はどう計算するか（条件付け？周辺化？）／図7 の 95% 帯は等間隔グリッドで点ごとに計算しているのか。
- 更新: [[questions/gaussian-process-intuitive-explainer]] の §6 直後に無番号セクション「補足：予測の平均と分散はどう計算するか（条件付け）と、帯の描き方」を追加。§7・§8 の番号は維持。
- 内容: (1) 周辺化で無限を有限の同時分布（K / K* / K**）に落とす土台、(2) 観測値で条件付けして閉形式の μ*=K*ᵀ(K+σ²I)⁻¹y・Σ*=K**−K*ᵀ(K+σ²I)⁻¹K* を得る本体、という「両方使う」整理。(3) 帯は等間隔グリッド（実スクリプトは linspace(-5,5,300)、間隔≈0.033）の各点で μ±1.96σ を計算する“点ごと（周辺）95% 区間”で同時帯ではない、帯は対角分散だけ・関数サンプルはフル共分散（コレスキー）という区別も明記。分散が y に非依存＝能動学習の根拠にも言及。
- 参照: [[sources/2025-gp-intuitive-intro]]（式3〜6・MAP=事後平均）, [[sources/2021-gp-models-intro]]（§2.1・付録A 条件付き分布）, [[sources/2020-gp-regression-tutorial]], [[gaussian-process]]。
- メモ: explainer 本文は数式控えめ方針だが、本補足はユーザーの精密な計算質問に応じて inline 数式を使用（display $$ ブロックは引き続き 0、新規作図なし）。

## [2026-05-31] query | TFM/PFN による GP 近似の仕組み（出力・カーネル/ハイパラの扱い）

- 問い: (1) TFM は文脈データから未観測点の平均・分散を出力できるか。(2) PFN 原典 §5.1 GP 近似でカーネル・ハイパラはどう扱われるか（PFN は事前分布を学習するのに GP の事前分布＝カーネル＋ハイパラは？）。
- 更新: [[questions/pfn-paper-and-gaussian-process]] に「関係3」後・「比較表」前へ 2 節を追加——「深掘り1: TFM/PFN は平均・分散をどう出力するのか」（文脈＋クエリ→1 forward→予測分布＝リーマン分布→平均・分散・95%。ICL。TabPFN v2 も同様）と「深掘り2: GP のカーネルとハイパラはどこへ？」（事前分布は設計者が与える生成レシピで、PFN が学習するのは推論。§5.1 固定カーネルは重みにコンパイル／§5.2 ハイパー事前分布は学習で周辺化）。frontmatter sources_used に [[sources/2025-tabpfn-v2]] 追加、用語に「リーマン分布」「ハイパー事前分布」追記。
- 図: **新規作成 1 枚** `raw/assets/pfn-paper-and-gaussian-process/fig-pfn-gp-pipeline.png`（①事前分布を選ぶ→②合成データ生成→③PFN 事前訓練→④推論で予測分布、の 4 ボックス模式図＋「カーネル/ハイパラは②に埋め込まれ③で重みにコンパイル・④では見ない」注記）。venv `/tmp/gpvenv` の matplotlib＋Hiragino。
- 参照: [[sources/2021-transformers-can-do-bayesian-inference]]（§3 訓練・§5.1 固定 GP・§5.2 ハイパー事前分布・付録F の具体値 RBF 0.6 / Matérn ν=2.5＋Gamma）, [[prior-data-fitted-networks]], [[in-context-learning]], [[tabular-foundation-model]], [[sources/2025-tabpfn-v2]]。
- 核心メッセージ: 「PFN は事前分布を学習しない。事前分布（カーネル＋ハイパー事前分布）は設計＝合成データ生成器で、PFN が学習するのは推論（データ→予測分布）」。数値（200x/1000〜8000x）・設定は原典訳と一致を確認。

## [2026-06-01] query | TabPFN/TabICL 各バージョンで方式は同じか

- 問い: PFN が固定カーネル/ハイパラを重みに内在するのは理解した。TabPFN・TabICL の各バージョンでもこの方式は同じか。
- 作成: [[questions/tabpfn-tabicl-versions-mechanism]]（3 件目の questions ページ）
- 更新: [[index]]（Questions に 1 行追記）
- 参照: [[prior-data-fitted-networks]]（共通の骨格・代表手法一覧）, [[structural-causal-model]]（各世代の事前分布の中身）, source 群（2021 原典・2022-tabpfn・2025-tabpfn-v2・2025-tabpfn-2-5・2026-tabpfn-3・2025-tabicl・2026-tabicl-v2・2025-real-tabpfn）。
- 結論: 骨格（事前分布を重みに焼き込む＋ICL 推論＋推論時にカーネル/ハイパラを推定しない）は全バージョン共通。差分は (a) 事前分布の中身（GP トイ→SCM→多様なランダム関数混合、GP は TabICLv2 では一材料）、(b) アーキ（2D セルアテンション⇔3 段行 ICL、TabPFN-3 は TabICLv2 系へ回帰）、(c) 回帰出力（リーマン分布／分位点分布）、(d) スケール（〜1000→100 万行）、(e) 任意拡張（継続事前訓練/微調整/蒸留/テスト時計算）。バージョン別比較表付き。
- メモ: 前回 [[questions/pfn-paper-and-gaussian-process]] の続き。「カーネル/ハイパラを内在」は「(より複雑な) 事前分布を内在」と一般化して読み替える、を明示。新規作図なし。

## [2026-06-01] query | （修正）questions ページを「ガウス過程限定」に作り替え

- 経緯: ユーザーの本来の問いは「各バージョンが **GP を** 同じように扱うか」という GP 限定の質問だった。直前に作った [[questions/tabpfn-tabicl-versions-mechanism]] は一般機構（SCM・アーキ・回帰出力・例外）まで広げていたため、**GP 限定の回答に全面改稿**（同一ファイルを書き換え。slug は据え置き＝log の過去エントリとの整合のため）。
- 更新: [[questions/tabpfn-tabicl-versions-mechanism]] を改稿（タイトル「各バージョンはガウス過程をどう扱うか」、frontmatter の question を GP 限定に、sources_used から Real-TabPFN を除外）。一般機構の「深掘り出力／2D⇔3 段アーキ／重み固定の例外」など GP 非関連の節を削除し、[[prior-data-fitted-networks]] / [[questions/pfn-paper-and-gaussian-process]] へ誘導。[[index]] の説明も GP 限定に更新。
- 結論（GP 限定）: 「GP（固定カーネル）を事前分布として焼き込む」のは **PFN 原典のトイ例のみ**。表形式系では主役が SCM に移り、GP の役割は ①ベースライン/物差し（原典・TabPFN v1）②多数の合成関数の一材料（TabICL v1 の活性化・TabICLv2 の 8 種の 1 つ）③不使用（TabPFN v2/2.5/3）へ世代変化。共通なのは PFN の一般機構であって GP の使い方ではない。
- 参照: 各 source の GP 言及を確認（2022-tabpfn＝GP 様挙動/PFN-GP・2025-tabicl＝活性化に GP 由来・2026-tabicl-v2＝8 種の 1 つが GP・v2/2.5/3 は GP を prior 材料にせず）。

## [2026-06-01] lint-fix | questions の逆リンク補修＋updated 日付の更新

- 経緯: lint で (1) questions 3 ページが概念/overview から逆リンクされていない（index からのみ到達）、(2) 一部 concept/overview の `updated` 日付が編集後も 2026-05-30 のまま、を検出。両方を修正。
- 逆リンク追加（各ページの「関連ページ」へ）:
  - [[gaussian-process]] → questions 3 件（explainer / pfn-paper / versions）
  - [[prior-data-fitted-networks]] → pfn-paper, versions
  - [[structural-causal-model]] → versions
  - [[bayesian-inference]] → pfn-paper
  - [[in-context-learning]] → pfn-paper
  - [[tabular-foundation-model]] → pfn-paper, versions
  - [[overview]] → questions 3 件を「掘り下げ」として追記
- `updated` を 2026-06-01 に更新: overview / gaussian-process / prior-data-fitted-networks / structural-causal-model / bayesian-inference / in-context-learning / tabular-foundation-model。
- 検出のみで未対応のまま残した項目: なし（lint で挙げた課題 1・2 を解消。課題 3 データギャップは設計上問題なしと判断）。
- メモ: 矛盾・古い内容・dangling・孤立・概念粒度は lint 時点でクリーン。今回はクロスリファレンスとメタデータのみの補修。

## [2026-06-03] ingest | PFNs4BO: In-Context Learning for Bayesian Optimization（ICML 2023）

- 取り込み: `raw/papers/PFNs4BO- In-Context Learning for Bayesian Optimization.pdf`（Samuel Müller, Matthias Feurer, Noah Hollmann, Frank Hutter / ICML 2023, arXiv:2305.17535）
- 作成: [[sources/2023-pfns4bo]], [[translations/2023-pfns4bo]]
- 更新: [[concepts/bayesian-optimization]]（PFNs4BO を「PFN サロゲートの正準的実体」として本文・sources・related に追記、updated→06-03）, [[concepts/prior-data-fitted-networks]]（代表手法に PFNs4BO＝BO 応用を追加、sources/related、updated→06-03）, [[concepts/gaussian-process]]（固定 GP 事後の再現＝図1、sources/related、updated→06-03）, [[overview]]（BO 節に PFNs4BO を反映、updated→06-03）, [[index]]
- 画像: **PDF（ケース B）のため画像は取り込まず**、図表（図1〜5・表1）はキャプションのテキスト訳のみ translation に残した。
- メモ:
  - **本文 §1〜8 のみ訳出**（ユーザー appendix 指定なし）。Acknowledgements・References・付録 A〜L は対象外。スラグは ICML 2023 → `2023-pfns4bo`。
  - 内容: PFN を **BO の GP サロゲートのベイズ的ドロップイン置換**に。3 事前分布（単純 GP・HEBO 着想 HEBO/HEBO⁺・BNN）。獲得関数 EI/PI/UCB をリーマン分布から厳密計算、勾配で獲得関数最適化＋入力ワーピング。**GP では難しい拡張**＝ユーザー事前分布 $p(D|\rho,I)$・無関係次元 30%・**非近視眼的 KG を直接学習**。HPO-B/Bayesmark/PD1 の 105 タスクで評価し HEBO（GP ベース SOTA）を上回る場面を示す。限界＝事前分布で低尤度のデータに弱い・同時サンプル不可・〜1000 点まで。
  - PFN 文脈での位置づけ: 原典（[[sources/2021-transformers-can-do-bayesian-inference]]）と同著者グループの **BO 応用編**。これまで [[bayesian-optimization]]/[[prior-data-fitted-networks]] が予告していた「GP の代わりの高速サロゲート」の正準実体で、[[questions/pfn-paper-and-gaussian-process]] の応用側、TabPFNv2 を BO に使う GIT-BO 流れの源流。新規 concept は作らず既存 2 概念に統合。

## [2026-06-03] ingest(追記) | PFNs4BO の付録 A〜M を翻訳追加

- ユーザー指示により [[translations/2023-pfns4bo]] に **付録 A〜M を一文ずつ全訳**して追記（PDF の page 14〜27）。冒頭注記を「本文 §1〜8 ＋ 付録 A〜M」に更新。References・Acknowledgements は引き続き対象外。
- 付録の内容: A 探索空間分割 / B 事前分布詳細（B.1 HEBO 着想の Gamma ハイパー事前分布、B.2 BNN＝tanh MLP 8〜15層・重みドロップ等）/ C 獲得関数最適化（10万ランダム＋L-BFGS-B）/ D BO トリック（べき変換・単位超立方体化）/ E 訓練詳細（次元 1〜18・埋め込み512・6層・8×2080Ti 24h, E.1 リーマン分布）/ F リーマン分布の PI 閉形式（式5〜8）/ G ユーザー事前分布の構築（式9〜14）/ H 非近視眼 KG の 2 段近似（式15・16, H.1 少次元で KG×EI 混合）/ I 事前データ上の近似品質（HEBO 尤度比較・図8〜12）/ J PD1 のユーザー事前分布（表2）/ K 他の BO サロゲート（DNGO/BOHAMIANN/Deep GP/木）/ L アブレーション（HEBO 変種・無関係特徴量＝表3・獲得関数＝表4・図13〜18）/ M ソフトウェアバージョン。
- 図表は PDF（ケース B）のため画像取り込みなし、図1〜18・表1〜4 はキャプションのテキスト訳のみ。index/source の翻訳範囲表記も「§1〜8 ＋ 付録 A〜M」に更新。

## [2026-06-03] ingest | Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models（arXiv 2025）

- 取り込み: `raw/articles/Mitra_ Mixed Synthetic Priors for Enhancing Tabular Foundation Models.md`（Xiyuan Zhang ら / Amazon, arXiv:2510.21204）
- 作成: [[sources/2025-mitra]], [[translations/2025-mitra]]
- 更新: [[concepts/structural-causal-model]]（「良い prior を測る原理（性能・多様性・独自性）」節を新設、sources/related、updated→06-03）, [[concepts/tabular-foundation-model]]（sources/related、updated→06-03）, [[concepts/prior-data-fitted-networks]]（代表手法に Mitra 追加、sources/related、updated→06-03）, [[overview]]（事前分布の設計の節に Mitra を反映）, [[index]]（sources/translations/略称リダイレクト）
- 図保存: ar5iv 由来 markdown（ケース A）。**26 枚を `raw/assets/2025-mitra/` にローカル保存**（x1〜x26、素 PNG）。translations に全 26 図を `<figure>` で配置、sources に図1 を再掲。
- メモ:
  - **本文 §1〜5 ＋ 付録 A〜D を全訳**（ユーザー指示「appendix も含めて」）。References は対象外。スラグは arXiv 2510（2025年10月）→ `2025-mitra`。
  - 内容: TFM の性能は合成 prior の設計で決まるという主張を、**「良い prior」を性能（実データ）・多様性（汎化性行列 $\mathbf{G}$ の対角が低い）・独自性（非対角が低い）で測る原理**に体系化。Mitra は **SCM＋木ベース prior（DT/ET/GB/RF＋直接サンプリング DSRF）の 50/50 混合**を選定。行 1D・要素 2D の両アーキで効く**モデル非依存**性を示し、TabPFNv2/TabICL を分類・回帰で上回る。最大 16 特徴量・640 行でしか事前訓練しないのに汎化。バギング・TabArena・スケーリング則・決定境界（GP が一部で強い→今後 GP prior 追加）も付録に。
  - 大きな結果表（表2〜18）はキャプションを訳出し、主要表（2/3/4/7/17）を markdown 化、巨大な per-dataset/leaderboard 表は要点（首位 Elo 等）を注記。アルゴリズム1〜5 は引用ブロックで訳出。
  - PFN 文脈: [[structural-causal-model]] の「prior 多様性が最大の性能要因」（TabICLv2）を **prior の選び方の方法論**へ一般化。[[tabular-foundation-model]] の中核命題に定量的設計指針を与える。新規 concept は作らず既存 3 概念に統合。

## [2026-06-03] query | リーマン分布のバケット／prior-data とは

- 問い: [[sources/2021-transformers-can-do-bayesian-inference]] の「リーマン分布は prior-data 上で等確率になるようバケットに離散化」が分からない。prior-data・バケットとは。
- 作成: [[questions/riemann-distribution-buckets]]（4 件目の questions ページ）
- 図: **新規作成 1 枚** `raw/assets/riemann-distribution-buckets/f1-buckets.png`（左＝prior-data の歪んだ y 分布を分位点で等確率10バケットに区切る／右＝推論時の固定バケット上のヒストグラム型出力）。venv `/tmp/gpvenv` の matplotlib＋Hiragino。
- 更新: [[index]]（Questions に 1 行）
- 参照: 原典付録 E.2（定義 $p(y)=p_{b(y)}/w(b(y))$・定理1・半正規の裾）, [[sources/2025-mitra]] 付録 E.1（$p(y\in b)=1/|\mathbf{B}|$）, [[sources/2023-pfns4bo]] §3.4/付録F（リーマン分布から EI/PI/UCB を厳密計算）。
- 要点: バケット＝y 軸の区間（ヒストグラムのビン）、prior-data＝事前分布から作った合成訓練データ、等確率＝prior-data の分位点で境界を置き各バケットの確率を 1/|B| に揃える（密な所は狭く・疎な所は広く）。これで回帰を分類に変え任意の形の予測分布を表せる。前回の「予測分布から平均・分散を読む」（[[questions/pfn-paper-and-gaussian-process]]）と接続。

## [2026-06-03] query | PFN の GP 近似を定量評価する指標

- 問い: 原典のように PFN が GP 回帰（平均・信頼幅）をどれだけ近似できているか定量評価したい。おすすめ指標は。
- 作成: [[questions/evaluating-pfn-gp-approximation]]（5 件目の questions ページ。ユーザー希望で表・プロトコルのみ、コード/図なし）
- 更新: [[index]]（Questions に 1 行）
- 参照: [[sources/2021-transformers-can-do-bayesian-inference]]（Prior-Data NLL＝図4a・図3 の平均/95%CI 可視化・ECE 0.025・損失＝KL+定数の系1.1）, [[sources/2023-pfns4bo]]（保留尤度比較・図11/12 で厳密 GP と重ねる）, [[sources/2026-tabicl-v2]]（回帰で CRPS）。
- 結論: (A) 固定ハイパラ GP の**厳密事後を物差し**に → 点ごと KL(厳密GP‖PFN, ガウス閉形式)・平均RMSE・信頼幅(σ/95%CI半幅)RMSE・2-Wasserstein。(B) 厳密 GP が無い場合 → 保留 NLL・CRPS・被覆率/分位点較正。最小セット＝KL＋平均/σ RMSE＋被覆率を $n$ を変えて測る。実装注記: PFN 出力はリーマン分布なので (a) 平均・分散抽出で比較 / (b) CRPS・数値KL で分布ごと比較（多峰・歪みは後者でしか見えない）。

## [2026-06-03] ingest | GIT-BO: High-Dimensional Bayesian Optimization with Tabular Foundation Models（arXiv 2025）

- 取り込み: `raw/articles/GIT-BO_ High-Dimensional Bayesian Optimization with Tabular Foundation Models.md`（Rosen Ting-Ying Yu, Cyril Picard, Faez Ahmed / MIT, arXiv:2505.20685）
- 作成: [[sources/2025-git-bo]], [[translations/2025-git-bo]]
- 更新: [[concepts/bayesian-optimization]]（GIT-BO を高次元 BO サロゲートとして本文・sources・related に追記）, [[concepts/tabular-foundation-model]]（sources/related・本文 BO 応用にリンク）, [[concepts/gaussian-process]]（GIT-BO 言及にリンク）, [[overview]], [[index]]（sources/translations/略称リダイレクト）。既存の各所「GIT-BO」言及を [[sources/2025-git-bo]] にリンク。
- 図保存: ar5iv 由来 markdown（ケース A）。**本文の図 4 枚を `raw/assets/2025-git-bo/` に保存**（x2=図2・x3=図3・x4=図4・x5=図5）。**図1（アーキ概要 x1）は ar5iv 側が "NO IMAGE AVAILABLE" のプレースホルダで画像変換失敗 → DL せずキャプションのみ訳出**（取得失敗扱い）。translations に図2〜5 を `<figure>`、sources に図5 を再掲。
- メモ:
  - **本文 §1〜6 のみ訳出**（ユーザー appendix 指定なし）。付録 A〜C・References は対象外。スラグは arXiv 2505（2025年5月）→ `2025-git-bo`。
  - 内容: **TabPFN v2 を高次元 BO（100〜500 次元）のサロゲート**に。固定基盤モデルは GP と違いカーネルを適応できないが、GIT-BO は **TabPFN 予測平均の順伝播勾配 ∇μ から診断行列 H=E[∇μ∇μᵀ] を作り上位 r=15 固有ベクトルで“勾配誘導能動部分空間”を毎反復構築**、その低次元部分空間内で Thompson Sampling。GP の O(n³) 再訓練不要。23 ベンチ・63 変種で SOTA（首位ランク1.97、SAASBO 等より 2 桁速い）。アブレーションで GI 部分空間なしのバニラ TabPFNv2 は大失敗（8.6 倍悪いリグレット）。限界＝GPU 必須・500 次元上限・訓練分布外（車設計）で順位低下。
  - PFN 文脈での位置づけ: wiki が各所で予告していた「高次元 BO のサロゲートに TabPFNv2 を使う」の**本体**。[[sources/2023-pfns4bo]]（低次元 BO）の高次元版。新規 concept は作らず既存概念に統合。

## [2026-06-03] query | PFN とベイズ最適化の関係（初心者向け解説）

- 問い: 「PFN とベイズ最適化の関係を初心者にもわかりやすく解説した記事を作成して」。
- 作成: [[questions/pfn-and-bayesian-optimization]]（6 件目の questions ページ）
- 図: **新規作成 1 枚** `raw/assets/pfn-and-bayesian-optimization/fig-bo-loop.png`（BO の 4 ノード円環ループ＋②サロゲート＝GP or PFN の吹き出し）。venv `/tmp/gpvenv` の matplotlib＋Hiragino。**再掲 4 枚**（Frazier の BO 例・GP の予測帯・PFN パイプライン図・GIT-BO アブレーション、各出典つき）。
- 更新: [[index]]（Questions に 1 行）
- 参照: [[sources/2018-bayesian-optimization-tutorial]]（BO 基礎）, [[sources/2021-transformers-can-do-bayesian-inference]]（PFN が GP を近似する根拠）, [[sources/2023-pfns4bo]]（低次元 BO 代表）, [[sources/2025-git-bo]]（高次元 BO 代表）, [[bayesian-optimization]] / [[prior-data-fitted-networks]]。
- 構成: BO とは→サロゲート＋獲得関数→GP の弱点→PFN とは→**サロゲートを GP→PFN に差し替える（核心）**→代表例（PFNs4BO 低次元 / GIT-BO 高次元、簡潔）→GP vs PFN 比較表→まとめ。初心者向けに略称初出展開・数式は最小限（詳細はリンク委譲）。既存 [[questions/pfn-paper-and-gaussian-process]]（GP↔PFN）とは焦点を分け相互リンク。

## [2026-06-03] ingest | CausalPFN: Amortized Causal Effect Estimation via In-Context Learning

- 取り込み: `raw/articles/CausalPFN_ Amortized Causal Effect Estimation via In-Context Learning.md`（ar5iv, arXiv:2506.07918, Balazadeh et al. / U Toronto・Vector・Layer6 AI 2025）。ケース A。
- 作成: [[sources/2025-causalpfn]], [[translations/2025-causalpfn]]
- 更新: [[structural-causal-model]], [[prior-data-fitted-networks]], [[in-context-learning]], [[bayesian-inference]], [[tabular-foundation-model]], [[overview]], [[index]]
- 翻訳範囲: **ユーザー指示により付録も含めて全訳**。本文 §1〜7 ＋ 付録 A〜D（A 記法/定義/仮定・B 一致性証明〔Doob 系〕・C data-prior 損失=前向き KL の証明・D 実験詳細 D.1〜D.5）。References・Appendix Contents スタブ・Acknowledgments は除外。Algorithm 1 は引用ブロックで訳出。
- 画像: 全14図（x1〜x14）を ar5iv から取得し `raw/assets/2025-causalpfn/` に保存（`file` で妥当画像と確認、参照=保存=14枚一致）。本文 Fig1〜7＝x1〜x7、付録 D.5 較正曲線 Fig8〜14＝x8〜x14。取得失敗なし。
- 概念粒度: 新規 concept ページは作らず既存5概念に統合（CLAUDE.md §1）。
- **設計判断（重要）**: CausalPFN は SCM/DAG ではなく**潜在結果フレームワーク（Rubin）＋強い ignorability** に立つため、[[structural-causal-model]] には「SCM とは別形式」と明記したうえで小節を追加して接続（prior 素材としての SCM 用法とは区別）。これにより既存の「TabPFN は SCM を prior に使う」記述との矛盾を回避。
- メモ: PFN/ICL の射程が予測（TabPFN）・最適化（PFNs4BO/GIT-BO）に続き**因果効果推定**へ届くことを示す論文。prior は OpenML 実表＋TabPFN v1 生成器を流用し ignorability を設計で保証、CEPO-PPD は原典リーマン分布と同型の 1024 ビンヒストグラム。data-prior 損失=前向き KL は [[bayesian-inference]] の償却の因果版。limitations: ignorability 検証不能・有限標本理論なし・大規模表で文脈長制約・二値処置のみ。

## [2026-06-03] ingest | Do-PFN: In-Context Learning for Causal Effect Estimation

- 取り込み: `raw/articles/Do-PFN_ In-Context Learning for Causal Effect Estimation.md`（ar5iv, arXiv:2506.06039, Robertson・Reuter・Guo・Hollmann・Hutter・Schölkopf / ELLIS Tübingen・MPI-IS・Cambridge・Freiburg・Prior Labs 2025）。ケース A。
- 作成: [[sources/2025-do-pfn]], [[translations/2025-do-pfn]]
- 更新: [[structural-causal-model]], [[prior-data-fitted-networks]], [[in-context-learning]], [[bayesian-inference]], [[tabular-foundation-model]], [[overview]], [[index]]
- 翻訳範囲: **ユーザー指示により付録も含めて全訳**。本文 §1〜5 ＋ 付録 A〜D（A 命題証明〔前向き KL〕・B prior-fitting 詳細・C 実験詳細〔ケーススタディ構造方程式・評価指標・ベースライン〕・D 補足結果）。References は除外。Algorithm 1 は引用ブロックで訳出、表1は要約注記。
- 画像: 全19図を ar5iv から取得し `raw/assets/2025-do-pfn/` に保存（`figures/` サブパスは `figures__` 前置でフラット化、`file` で妥当画像と確認、参照=保存=19枚一致）。本文 Fig1-7＝x1,x2,cid_rmse_groups_int,cate_rmse_groups,sales_cid,cate_rmse_sufficiency,treatment_weight。付録 Fig8-19＝x3,sales_cid_scatter,laws_cid_scatter,scatter_noise,cid_rmse_groups_dowhy,cate_rmse_groups_dowhy,cid_rmse_groups_obs,cid_rmse_groups_var,x4,entropy,Calibration__Observed_Confounder,bias_cid_top。取得失敗なし。
- 概念粒度: 新規 concept ページは作らず既存5概念に統合（CLAUDE.md §1）。
- **設計判断（重要・前回 CausalPFN との対比）**: Do-PFN は CausalPFN と**ほぼ同時期の姉妹研究**だが、立つ因果の枠組みが逆。Do-PFN＝**SCM／do 計算（Pearl）・因果グラフ不要・無交絡を要求しない・条件付き介入分布（CID）$p(y|do(t),x)$ を予測**。CausalPFN＝潜在結果（Rubin）・ignorability 要求・CATE/ATE。Do-PFN は**介入つき SCM を prior に据える**ため [[structural-causal-model]] に最も自然に収まる（CausalPFN より中心的）。同ページに両者の対比小節を追加。
- メモ: アーキは TabPFN v2 とほぼ同一（730 万パラメータ＋先頭列＝処置の指示子）で軽量（RTX 2080 ×48h）。Algorithm 1 の NLL 最小化＝CID への前向き KL 最小化（命題1）。同定不能ケース（未観測交絡）でも「因果的痕跡」からもっともらしい解集合＋高エントロピーで不確実性を正しく表現。CATE はバイアス相殺で CID 単体より強い。limitations: prior 依存・同定可能性理論未整備・二値介入のみ・解釈可能性。著者は Hutter（TabPFN）＋Schölkopf（因果）。

## [2026-06-03] ingest | Chronos-2: From Univariate to Universal Forecasting

- 取り込み: `raw/articles/Chronos-2_ From Univariate to Universal Forecasting.md`（ar5iv, arXiv:2510.15821, Ansari ら / Amazon Web Services・Amazon・Freiburg・JKU Linz ほか 2025）。ケース A。
- 作成: [[sources/2025-chronos-2]], [[translations/2025-chronos-2]]
- 更新: [[in-context-learning]], [[tabular-foundation-model]], [[overview]], [[index]]
- 翻訳範囲: **ユーザー指示により付録も含めて全訳**。本文 §1〜6 ＋ 付録 A（訓練データ表）・B（追加結果＝MASE/WQL/WAPE 表＋ペアワイズ比較図）。References・Acknowledgements は除外。能力比較表（表1）・結果表（表3-5）は訳出、長いデータセット羅列表（表6/10/11）と反復的な結果表（表7-9）は代表値＋原典参照の注記で要点保持。
- 画像: 全19図を ar5iv から取得し `raw/assets/2025-chronos-2/`（x1,x2,x4,x7,x9,x11,x12,x13,x16,x19,x21,x23-x30）。`file` で妥当画像と確認、参照=保存=19枚一致。図の対応を Read で確認: x16/x19/x21 は x4/x7/x9（単変量/多変量/エネルギー共変量サブセット）の MASE 版。取得失敗なし。
- **概念粒度の判断（重要）**: Chronos-2 は**時系列基盤モデル**で PFN そのものではない。新規 concept ページ（例: `time-series-foundation-model`）は**作らず**、接点の強い [[in-context-learning]]（group attention＝時系列版 ICL・クロス学習）と [[tabular-foundation-model]]（fev-bench で TabPFN-TS と競合・合成データ事前訓練の共通路線）に統合。**今後 TS 基盤モデルのソースが増えたら専用 concept ページ化を再検討**（→ ユーザーに提案候補）。
- メモ: 中核は group attention（バッチ軸でグループ内系列の情報共有、$O(V)$ スケール）＝多変量・共変量つき予測をゼロショットで1モデル化。能力は合成データ（multivariatizer）由来で、合成のみ版が実データ混合版にほぼ並ぶ＝TabPFN 流「合成 prior」と共鳴。fev-bench/GIFT-Eval/Chronos Benchmark II で SOTA、共変量タスクで TabPFN-TS が2位。分位点回帰（21分位点・pinball）で確率予測、ベイズ的 PPD 枠組みは前面に出さない。表形式 TFM と時系列 FM が「ICL＋合成データ」へ収斂する好例として overview に追記。

## [2026-06-04] schema-update | lint 修正（孤立ページ解消・overview 掘り下げ更新）

- 点検: 58ページ全体を走査（dangling 0・frontmatter 裸 [[ 0・translation/source 対称 21↔21・概念粒度逸脱なし・日付整合）。
- **孤立ページ解消**: [[questions/evaluating-pfn-gp-approximation]] → [[bayesian-inference]] と [[gaussian-process]] の関連ページに追加。[[questions/pfn-and-bayesian-optimization]] → [[bayesian-optimization]] の関連ページに追加。
- **overview 掘り下げ更新**: 掘り下げ行に [[questions/riemann-distribution-buckets]]・[[questions/evaluating-pfn-gp-approximation]]・[[questions/pfn-and-bayesian-optimization]] の 3 件を追加（それまで 3 件のみだったところを 6 件に更新）。
- 孤立ページ再チェック: 0 件（全 questions が1件以上の概念/sources から被リンク）。

## [2026-06-06] ingest | nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN

- 取り込み: `raw/articles/nanoTabPFN_ A Lightweight and Educational Reimplementation of TabPFN.md`（ar5iv, arXiv:2511.03634, Pfefferle・Hog・Purucker・Hutter / ELLIS Tübingen・Freiburg・Prior Labs 2025）。ケース A。
- 作成: [[sources/2025-nanotabpfn]], [[translations/2025-nanotabpfn]]
- 更新: [[prior-data-fitted-networks]], [[tabular-foundation-model]], [[index]]
- 翻訳範囲: **ユーザー指示により付録も含めて全訳**。本文 §1・§3〜§5 ＋ 付録 A（詳細実験設定・HPO 探索空間）・付録 B（データセットごとの結果）。References・Disclaimer・Acknowledgement は除外。Figure 3（コード例）は data URI のダウンロードリンクなので画像非取得、引用ブロックで要旨を記載。
- 画像: 4図（architecture.png / x1.png / roc_auc.png / summary.png）を ar5iv から取得し `raw/assets/2025-nanotabpfn/` に保存。全て妥当画像と確認。
- **概念粒度**: nanoTabPFN は個別モデルなので専用 concept ページは作らず、[[prior-data-fitted-networks]] と [[tabular-foundation-model]] の「代表手法」節に統合（CLAUDE.md §1 規約）。
- **設計判断**: tabular-foundation-model に「教育・軽量化（アクセシビリティの向上）」節を新設し nanoTabPFN を配置。蒸留（production 向け）と並ぶ「TFM を小さく・使いやすくする」軸として位置づけ。
- メモ: 「nanoGPT の表形式版」というポジショニングが明快。500行コード・1分訓練・160,000倍速・単一 GPU。性能はあくまで小規模・教育向け（150データ×5特徴×2クラス）。訓練データ生成に TabICL の prior 実装を流用（[[sources/2025-tabicl]] との接点）。TabPFN 官製グループ（Hutter・Purucker）による TFM 民主化の贈り物。

## [2026-06-06] ingest | ShapPFN: Real-Time Explanations for Tabular Foundation Models

- 取り込み: `raw/articles/Real-Time Explanations for Tabular Foundation Models.md`（ar5iv, arXiv:2603.29946, Sena & Azevedo / Kunumi Institute・UFMG, Brazil 2026）。ケース A。当初 arxiv.org HTML 版（画像なし）だったが、ユーザーが ar5iv 版に差し替え。
- 作成: [[sources/2026-shappfn]], [[translations/2026-shappfn]]
- 更新: [[prior-data-fitted-networks]], [[tabular-foundation-model]], [[index]]
- 翻訳範囲: **ユーザー指示により付録も含めて全訳**。本文 §1〜5 ＋ 付録 A（ハイパーパラメータの影響・逐次 HPO）。References・Disclaimer・Acknowledgement は除外。表1/2/3 は詳細数値を原典参照の注記＋要点で保持（巨大な per-dataset テーブル）。
- **画像（特記）**: 図1（アーキ図）は ar5iv で **inline SVG（TikZ ベクター）**。ar5iv の assets PNG は「NO IMAGE AVAILABLE」プレースホルダーのみ（GIT-BO x1 と同型、3 URL とも同一 20498B）。→ inline SVG を抽出し `raw/assets/2026-shappfn/architecture.svg` に保存（xmlns 補完）。ただし SVG のラベルは foreignObject（HTML）で、Obsidian の `![]()` img 埋め込みでは描画されない。→ **Chrome ヘッドレス（`--headless=new --screenshot`、Google Chrome.app）で foreignObject 込みでラスタライズ**し `architecture.png`（1891×405）を生成。フォント縮小でラベルのはみ出しを軽減。cairosvg/rsvg/inkscape は未インストール＋foreignObject 非対応のため不可、ヘッドレスブラウザが唯一の解だった。
- 概念粒度: ShapPFN は個別モデルなので専用 concept は作らず [[prior-data-fitted-networks]]（代表手法）と [[tabular-foundation-model]]（「説明可能性の内蔵」節）に統合。
- メモ: **nanoTabPFN（[[sources/2025-nanotabpfn]]）を実研究の踏み台に使った最初の公開例**。ViaSHAP 流の Shapley 値回帰を PFN に統合し、予測を base+Σφ の加法形にして予測と説明を1順伝播で同時出力。KernelSHAP 610s→0.06s（1000倍超速）で一致 R²=0.96。SHAP 損失なしだと説明品質崩壊（R²=0.179）＝損失が鍵。「ベイズ推論を事前訓練に償却する PFN」の発想を**説明計算の償却**へ応用。予測性能は TabPFN v2（0.872）に及ばず（0.848）、二値分類・小特徴量に限定。
