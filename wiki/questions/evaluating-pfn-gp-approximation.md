---
type: question
asked: 2026-06-03
question: "PFN がガウス過程回帰（平均・信頼幅）をどれだけよく近似できているかを定量評価するには、どの指標がおすすめか？"
sources_used:
  - "[[sources/2021-transformers-can-do-bayesian-inference]]"
  - "[[sources/2023-pfns4bo]]"
  - "[[sources/2026-tabicl-v2]]"
---

# PFN の GP 近似をどう定量評価するか（評価指標とプロトコル）

> 問い: PFN 原典（[[sources/2021-transformers-can-do-bayesian-inference]]）のように、PFN がガウス過程回帰（GP; [[gaussian-process]]）を**どれだけよく近似できているか**（平均・信頼幅など）を定量評価したい。おすすめの指標は？

## 大方針：何と比べるかで指標が変わる

GP を近似対象にする最大の利点は、**固定ハイパーパラメータの GP なら事後予測分布（PPD; posterior predictive distribution、[[bayesian-inference]]）が閉形式の正規分布で厳密に出る**こと（[[questions/pfn-paper-and-gaussian-process]] 関係1「物差し」）。評価は 2 系統に分かれる。

- **(A) 厳密 GP 事後と直接比べる**（固定ハイパラ GP を使う＝原典の設定）。**最もおすすめ**。平均も信頼幅も“真値”があるので分布同士の近さを厳密に測れる。
- **(B) 真の保留点 $y$ に対する予測の良さで測る**（厳密 GP が無い／ハイパー事前分布つき GP の場合）。proper scoring rule（適正スコア則）と較正で測る。

原典自身は (A)(B) の両面を使い、**原理的な指標として「Prior-Data NLL（保留点の負の対数尤度）」**（図4(a)）を主軸に、**平均・95% 信頼区間を厳密 GP と重ねて可視化**（図3、「ほぼ区別がつかない」）し、分類では **ECE 0.025**（較正）も報告した。理論的には**損失＝PPD との交差エントロピー＝KL ダイバージェンス＋定数**（洞察1・系1.1）なので、**NLL/KL が“近似の正しさ”の最も自然な尺度**である。

## おすすめ指標（用途別）

| 指標 | 何を測るか | 平均/信頼幅/全体 | 厳密GP要否 | 備考 |
|---|---|---|---|---|
| **点ごとの KL(厳密GP ‖ PFN)** | 予測分布全体の近さ | 全体（平均＋分散同時） | 要 | ガウス同士は閉形式。**第一推奨**。論文の発想と一致 |
| **平均関数の RMSE/MAE** | 予測平均 μ のズレ | 平均 | 要 | 図3 的な可視化と相性良い |
| **不確実性の RMSE**（σ または 95%CI 半幅の誤差） | **信頼幅**の再現度 | 信頼幅 | 要 | 「信頼幅をどれだけ再現したか」に直答 |
| **2-Wasserstein 距離** | 予測分布の近さ | 全体 | 要 | 1D ガウスは閉形式。KL の代替/補完 |
| **保留点 NLL（log score）** | 真値 y への当てはまり | 全体 | 不要 | = 訓練損失。小さいほど真 PPD に近い（系1.1） |
| **CRPS** | 予測分布全体（平均＋裾） | 全体 | 不要 | 適正スコア則。**リーマン分布（非ガウス）でも使える**。[[sources/2026-tabicl-v2]]/Mitra も回帰で採用 |
| **被覆率 / PICP** | X% 区間が実際に X% 当たるか | 較正 | 不要 | 「信頼幅が正直か」の素朴で強力な検証 |
| **分位点較正曲線 / 回帰ECE** | 全分位点での較正 | 較正 | 不要 | reliability diagram。系統的な過信/過小評価を可視化 |

### 最小おすすめセット

1. **点ごとの KL(厳密GP ‖ PFN)**（平均＋分散を 1 数値で）
2. **平均の RMSE** と **信頼幅（σ か 95%CI 半幅）の RMSE**（質問の「平均」「信頼幅」を分けて見る）
3. **被覆率（95% 区間の経験被覆）** と **分位点較正曲線**（信頼幅が正直か）
4. 厳密 GP が無い場面の保険として **保留点 NLL** と **CRPS**

## 具体的な評価プロトコル（原典に倣う）

1. **固定ハイパラの GP を 1 つ決める**（例：RBF、長さスケール固定）。これで各クエリ点 $x_*$ の厳密事後 $\mathcal{N}(\mu_{GP}(x_*),\sigma_{GP}^2(x_*))$ が出る。
2. その GP 事前分布から**多数のデータセットを合成**（PFN 訓練と同じ prior）。各データセットでサポート点を文脈に与え、**多数のクエリ点**で PFN と厳密 GP の両方を評価。
3. **各クエリ点で指標を計算 → 全クエリ・全データセットで平均**し、平均±信頼区間で報告（バラつきも見る）。
4. **データ点数 $n$ を変えて**プロット（原典の知見：訓練量・サンプル数を増やすほど近似が締まる）。ベースライン（厳密 GP 自身＝下限、Attentive Neural Process 等）と並べる。

## 実装上の注意（重要）

- **PFN の出力はリーマン分布（バケット化ヒストグラム、[[questions/riemann-distribution-buckets]]）**で、厳密 GP 事後はガウス。比較は 2 通り:
  - **(a) 平均・分散を抽出して比較**: リーマン分布から平均 $\mu_{PFN}$・分散 $\sigma_{PFN}^2$ を計算し、「平均/信頼幅 RMSE」「ガウス近似での KL/Wasserstein」に使う（手軽）。
  - **(b) 分布のまま比較**: **CRPS** や、ヒストグラム対ガウスの KL を数値積分で。GP 事後は単峰ガウスなので、PFN が**余計な多峰性・歪み**を出していれば (a) では見えず (b)/CRPS で見える。
- **ガウス同士の KL の閉形式**:

$$
\mathrm{KL}(\mathcal{N}(\mu_1,\sigma_1^2)\,\Vert\,\mathcal{N}(\mu_2,\sigma_2^2))=\log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac12.
$$

「厳密GP=1、PFN=2」で点ごとに計算して平均する。

- **2-Wasserstein（1D ガウス）**も閉形式: $W_2^2=(\mu_1-\mu_2)^2+(\sigma_1-\sigma_2)^2$。
- **被覆率**は、各クエリで PFN の 95% 区間が真値 $y$（または厳密 GP からのサンプル）を含むかを 0/1 で記録し平均。理想は 0.95。**分位点較正曲線**は、名目分位点 $\tau$ に対し「PFN の $\tau$ 分位点以下に真値が入る割合」をプロットし、対角線（$y=x$）に乗るかを見る。

## 関連研究での使われ方（参考）

- **原典 [[sources/2021-transformers-can-do-bayesian-inference]]**: 固定 GP で**厳密事後を物差し**に、Prior-Data NLL（図4a）・平均/95%CI の可視化（図3）で評価。損失＝KL＋定数（洞察1）が指標選択の理論的根拠。
- **[[sources/2023-pfns4bo]]**: 保留出力の**尤度**で PFN と経験ベイズ GP を比較（付録 I・表8）、厳密 GP の平均/EI と PFN を BO ステップ上で重ねて評価（図11・12）。
- **[[sources/2026-tabicl-v2]] / [[sources/2025-mitra]]**: 回帰で **CRPS** を採用（分布全体の質）。

## まとめの一言

**「平均」「信頼幅」だけでなく“予測分布全体の近さ”を 1 つ見るなら KL（または CRPS）、人間が解釈しやすい分解として 平均RMSE＋信頼幅RMSE、信頼幅が正直かは 被覆率＋分位点較正。** 固定ハイパラ GP を使えば厳密事後が物差しになるので、まず **KL・平均/σ の RMSE・被覆率** の 3 点を、データ点数を変えながら測るのがおすすめ。

## 用語と略称

- **PPD** = Posterior Predictive Distribution（事後予測分布。近似対象）→ [[bayesian-inference]]
- **NLL** = Negative Log-Likelihood（負の対数尤度。log score。= PFN の訓練損失）
- **KL ダイバージェンス** = 2 分布の非対称な「近さ」。損失最小化＝PPD への KL 最小化（系1.1）
- **CRPS** = Continuous Ranked Probability Score（連続値の適正スコア則。任意形の予測分布に使える）
- **被覆率 / PICP** = Prediction Interval Coverage Probability（区間が真値を含む経験割合）
- **較正（calibration）/ ECE** = 予測確率（信頼幅）と実際の的中率の整合 / その期待誤差
- **2-Wasserstein** = 分布間の輸送距離（1D ガウスは閉形式）
- **リーマン分布** = PFN の回帰出力（バケット化ヒストグラム）→ [[questions/riemann-distribution-buckets]]

## 関連ページ

- [[sources/2021-transformers-can-do-bayesian-inference]] — 厳密 GP を物差しに PFN を評価した原典
- [[questions/pfn-paper-and-gaussian-process]] — PFN と GP の関係（物差し・出力の仕組み）
- [[questions/riemann-distribution-buckets]] — PFN の予測分布（リーマン分布）から平均・分散・分位点を得る
- [[questions/gaussian-process-intuitive-explainer]] — GP の平均・信頼幅の計算（条件付け）
- [[gaussian-process]] — 厳密事後が閉形式で出る近似対象
- [[bayesian-inference]] — PPD と較正の枠組み
