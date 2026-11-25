# 方法細節

本文對照程式說明最終提交的每個步驟。超參數都可在 [`player_identification/config/final.yaml`](../player_identification/config/final.yaml) 與 `player_identification/src/` 中找到。

## 1. 官方題池

[`src/gallery.py`](../player_identification/src/gallery.py) 讀入四組官方題目與對應的候選庫。

- 候選庫 CSV 的棋局依出現順序編號，同一位玩家的棋局組成一個候選人。
- 查詢題的棋局接在候選庫之後編號。
- 每盤棋記錄執色，供同色／異色比較使用。

推論前，[`src/prepare_stage.py`](../player_identification/src/prepare_stage.py) 會核對 9 個官方檔案的 SHA-256。讀取專家矩陣時，也會核對題號、候選人代碼與盤數是否和官方題池一致。

## 2. 整盤棋局表徵（E4）

程式：[`src/encoders.py`](../player_identification/src/encoders.py)、[`src/encoder_workflow.py`](../player_identification/src/encoder_workflow.py)。

| 項目 | 設定 |
|---|---|
| 快照 | 目標玩家每盤的落子中均勻取 64 手，各取落子前的盤面 |
| 每個快照的平面 | 自己的棋子、對手的棋子、這一手的落點 |
| 輸入 | 64 × 3 個平面，加上 2 個座標平面與 1 個執色平面，共 195 個通道，19 × 19 |
| 骨幹 | 3×3 卷積 stem、10 個帶 SE 的殘差 block，寬度 128 或 192 |
| 輸出 | 3 × 3 平均池化後線性投影成 256 維，再做 L2 正規化 |
| 訓練目標 | 單盤 CosFace 玩家分類，scale 30、margin 0.2 |
| 訓練設定 | 官方訓練集全部 100 萬盤，batch 512、23,438 步；AdamW 學習率 0.002、weight decay 0.05，1,000 步暖身後餘弦衰減；BF16 |

四個編碼器的寬度與種子如下，串接後就是 E4 嵌入（共 1,024 維）。

| 名稱 | 寬度 | 種子 |
|---|---:|---:|
| `seq64` | 128 | 20261113 |
| `wide` | 192 | 20261115 |
| `seq64_s2` | 128 | 20261116 |
| `wide_s2` | 192 | 20261117 |

和先前只看前 50 手、24 個快照的小模型相比，這一步把本地新題組分數從 0.396 提高到 0.597，Public 從 0.668 提高到 0.805。

## 3. 舊題（`pub_`、`priv_`）

程式：[`src/predict_stage.py`](../player_identification/src/predict_stage.py) 的 `old_pool`、[`src/gallery.py`](../player_identification/src/gallery.py) 的 `components`。

1. 對每位候選人與每題，分別計算兩種相似度：
   - **原型餘弦**：查詢棋局平均嵌入與候選人平均嵌入的餘弦。
   - **逐盤最佳兩局**：每盤查詢棋局取與該候選人最相似的兩盤，再平均。
2. 兩者各自做題內標準化後，以 `1 : 0.5` 相加（`old_recipe`）。
3. 在候選池內做 Sinkhorn 正規化（τ = 0.05，60 次迭代），取前五名。

舊題的候選人各有 100 盤參考棋譜，本地分數接近滿分。最終提交沿用 37 專家融合版的這 800 個答案。

## 4. 新題的專家融合（`pub2_`、`priv2_`）

程式：[`src/common.py`](../player_identification/src/common.py) 的 `blend`、`transform`。

每題的分數矩陣是 400 題 × 600 位候選人。融合公式：

```text
S = baseline + Σ_b  w_b(n) · M_b
n ≤ 2 盤用第一組權重，3–5 盤用第二組
```

- 每個專家矩陣 `M_b` 都已經在題內標準化。
- 名稱帶 `anchor_scaled:` 的分支，會先乘上一組固定參考融合分數的題內標準差，讓它和其他分支的尺度一致。
- 名稱帶 `engine_only:` 的分支，在沒有引擎特徵的歷史題池上停用；官方新題組有引擎特徵，所以全部啟用。

融合後做一次池內軟校正：

```text
T(S) = z(S) + 2 · z(log Sinkhorn(z(S) / 0.2))
```

`z` 是題內標準化。每題依 `T(S)` 保留前 20 名候選，交給下一步重排。

### 37 個專家與權重

分類依分支名稱與實驗紀錄整理。權重欄是「1–2 盤／3–5 盤」。

| 類別 | 分支 | 權重 |
|---|---|---|
| 基準 | `baseline`：E4 嵌入相似度（Sinkhorn 前） | 1／1 |
| 棋局集合比對 | `setmatch/e4` | 0.274／0.280 |
| | `episodic/joint` | 0.642／0.719 |
| | `facets/native` | 0.322／0.120 |
| | `facets/unseen` | 1.215／1.249 |
| | `fresh_match` | 0.013／0.001 |
| | `plda40`（雙色 Gaussian PLDA） | 0.178／0.123 |
| | `anchor_scaled:collective/cohort` | 0.435／0.419 |
| | `support_adapt/moves` | 0.006／0.006 |
| 逐手決策 | `decisions` | 0.147／0.068 |
| | `replies` | 0.242／0.206 |
| | `moves_distill` | 0.287／0.224 |
| | `stacker_moves` | 0.620／0.496 |
| | `style_policy` | 0.001／0.001 |
| | `neuralmatch/moves` | 0.001／0.001 |
| | `neuralmatch/moves_board` | 0.000／0.000 |
| | `neuralmatch/moves_deep` | 0.345／0.298 |
| | `engine_only:neuralmatch/moves_patches` | 0.234／0.259 |
| Fox 預訓練 | `moves_fox` | 0.111／0.061 |
| | `moves_fox_episode` | 0.374／0.331 |
| | `neuralmatch/moves_fox` | 0.128／0.119 |
| | `engine_only:neuralmatch/late_residual_fox` | 1.278／1.189 |
| | `human_trunk/fox_joint_views` | 0.800／0.566 |
| | `engine_only:engine_search/fox_set` | 0.251／0.068 |
| KataGo／HumanSL | `engine_only:engine_moves/human` | 0.113／0.179 |
| | `engine_only:engine_search` | 0.164／0.001 |
| | `engine_only:engine_sequence` | 0.028／0.018 |
| | `engine_only:engine_sequence/long` | 0.043／0 |
| | `engine_only:strength` | 0.001／0 |
| | `human_trunk/d18` | 0.252／0.191 |
| | `human_trunk/d18_views` | 0.426／0.322 |
| | `human_trunk/turn_match` | 0.023／0 |
| 開局 | `opening_book` | 0.700／0.612 |
| | `domain_adapt/opening_anchor` | 2.050／1.620 |
| 其他 | `metadata` | 0.582／0.512 |
| | `opponents/strength`（對手一方的訊號） | 0.024／0.007 |
| | `end_profiles` | 0／0.002 |

權重的完整精度在 `config/final.yaml`。這組權重在比賽中用跨池盲測選定，Public 為 0.8685411。

## 5. 短名單特徵

程式：[`src/pairs.py`](../player_identification/src/pairs.py)、[`src/experts.py`](../player_identification/src/experts.py)、[`src/reranking.py`](../player_identification/src/reranking.py)。

每一題的每位短名單候選人有三組特徵。

**逐盤比對特徵（13 項）**
- E4 四個子嵌入分開計算後平均，共 9 項：
  - 原型餘弦；
  - 逐盤最大相似度，以及前 2、前 3 名的平均；
  - 單一最佳配對；
  - soft-max 相似度（溫度 0.05）；
  - 同色與異色的最大相似度；
  - 候選棋局中和查詢同色的比例。
- HumanSL 逐手描述向量（每盤 12 手、每手 96 維，先減去候選池平均），共 4 項：
  - 逐手最大相似度；
  - 前 3 名平均；
  - 整體平均向量的餘弦；
  - 是否有描述向量。

**專家特徵（37 項）**：37 個專家分數在 20 位候選人之間做標準化。

**設計矩陣**
- 融合分數的題內標準化值，以及和第一名的差距。
- 每個特徵的原值、短名單內標準化值、和「其他候選人最大值」的差。
- 候選人在短名單中的位置。
- 1–2 盤與 3–5 盤的題目分開放在兩組欄位，各自學一組係數。

## 6. 列表式重排器

| 項目 | 設定 |
|---|---|
| 模型 | 線性打分 `score = w·x + b`，20 位候選人做 softmax |
| 損失 | 正解位置的交叉熵，加上權重的 L2（0.01） |
| 優化 | Adam，學習率 0.01，全批次 300 個 epoch，特徵先標準化 |
| 訓練資料 | 八個本地標註題池共 3,200 題，只用正解進入前 20 名的題目 |
| 集成 | 種子 0、1、2，推論時平均 logits |

推論時依平均 logits 排序，取前五名。

## 7. 輸出與驗證

- 四組題目各自在自己的候選庫內排序，依官方範例的題號順序寫出 1,600 列的 `question_id,top_1,…,top_5`。
- [`verify_submission.py`](../verify_submission.py) 依序檢查：
  - 表頭與題號順序；
  - 每題五個候選人互不重複，且屬於對應的候選庫；
  - 與比賽提交原檔逐列比對，SHA-256 一致。

## 本地驗證結果

八個本地題池輪流留一池評估，新題型的平均分數：

| 方法 | 主要保留組 | 引擎保留組 | 平均 |
|---|---:|---:|---:|
| 37 專家融合 | 0.76918 | 0.78491 | 0.77704 |
| 最終版（前 20 名重排） | 0.77983 | 0.79626 | 0.78805 |

最終版相對 37 專家融合 **+0.0097**，95% 信賴區間 [+0.003, +0.016]，8 個題池中 7 個為正。Public 實際提升 +0.0168，換算新題組約 +0.0336，是本地估計的 3 倍左右。
