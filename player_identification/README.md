# player_identification

玩家辨識的程式、設定與比賽產物。最終版本為 **B_top20_scalefree_pairs**：Private 0.8985554（第 11 名／136 隊），Public 0.8853318。方法說明見 [docs/method.md](../docs/method.md)。

## 各階段指令

從 repo 根目錄執行。`acceptance.py` 會依序呼叫推論與驗證；需要逐步執行時，可以直接用以下指令：

```shell
python -B player_identification/scripts/prepare.py
python -B player_identification/scripts/train.py
python -B player_identification/scripts/predict.py --trained
python -B player_identification/scripts/verify.py
```

| 指令 | 輸入 | 輸出 |
|---|---|---|
| `prepare.py` | 官方 CSV、專家矩陣、E4 嵌入、HumanSL 描述向量 | `outputs/layout/`、`outputs/features/`（十個題池的融合分數、短名單特徵） |
| `train.py` | 八個本地題池的特徵（缺少時自動呼叫 prepare） | `outputs/weights/rerank_{0,1,2}.pt` |
| `predict.py` | 官方 CSV、`pub2`／`priv2` 的特徵、重排權重 | `outputs/submission_player.csv` |
| `verify.py` | `outputs/submission_player.csv`、`submission/submission_player.csv` | `outputs/verification.json` |

- `predict.py` 會自己重建新題組的特徵，所以重現提交檔不需要先執行 `prepare.py`。
- `predict.py` 預設使用比賽權重 `artifacts/weights/`；加 `--trained` 則使用 `train.py` 產生的 `outputs/weights/`。
- `config/final.yaml` 記錄 37 個專家的融合權重、舊題設定、短名單大小 20、L2 0.01、種子 0–2、300 個 epoch、八個訓練題池，以及官方輸入與提交原檔的 SHA-256。

### 從官方棋譜重建 E4（選用，需 CUDA）

```shell
python -B player_identification/scripts/prepare.py --encode-sgf
python -B player_identification/scripts/predict.py --rebuilt-embeddings
python -B player_identification/scripts/verify.py
```

- `prepare.py --encode-sgf` 用 `artifacts/encoders/` 的四個比賽權重，從官方 SGF 重新產生四個題池的 E4 嵌入，寫入 `outputs/embeddings/`。
- `predict.py --rebuilt-embeddings` 改用這些重建的嵌入；其他專家分數與 HumanSL 描述向量仍使用保存的產物。

### 重訓 E4 編碼器（選用，需 CUDA）

```shell
python -B player_identification/scripts/train.py --encoders
python -B player_identification/scripts/prepare.py --encode-sgf --trained-encoders
```

- 需要官方訓練集（見 [data/README.md](../data/README.md)），另外約 12 GB 的訓練快取。
- 四個編碼器各訓練 23,438 步，寫入 `outputs/encoder_training/`。

## 程式

| 檔案 | 內容 |
|---|---|
| `src/gallery.py` | 讀取官方題池；舊題的原型、逐盤相似度與 Sinkhorn |
| `src/common.py` | 路徑、專家矩陣讀取與核對、37 專家融合、池內軟校正 |
| `src/prepare_stage.py` | 核對官方輸入，建立題池索引、融合分數與短名單特徵 |
| `src/pairs.py` | 短名單的 E4 逐盤比對與 HumanSL 逐手比對特徵 |
| `src/experts.py` | 短名單候選人的 37 個專家分數 |
| `src/reranking.py` | 設計矩陣、線性列表式 softmax 的訓練與資料集 |
| `src/train_stage.py` | 訓練三個種子的重排器；`--encoders` 訓練 E4 |
| `src/predict_stage.py` | 舊題計分、新題重排、依官方題號寫出 CSV |
| `src/encoders.py`、`src/encoder_workflow.py` | SGF 重播、64 快照編碼、E4 訓練與編碼 |
| `src/verify_stage.py` | 呼叫根目錄的 `verify_submission.py` |

提交原檔的 SHA-256 為 `027f7ad8ff2483a96ee051ae0537241101846de90f96ed1683e7d3de86b91459`。
