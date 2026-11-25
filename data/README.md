# 資料

## 官方競賽資料（必要）

競賽資料由 AI CUP 2026 主辦單位提供，本 repo 不重新散布。請從競賽平台取得後，放到 `data/official/player_identification/`：

| 檔案 | SHA-256 |
|---|---|
| `player_identification_candidates_public.csv` | `e662242cb0f0b2be2009c129abd291c9c852af7fae25d8f1e1c00208785ee24d` |
| `player_identification_candidates_private.csv` | `db2cfe33be9c1e36355fac5664b758909ef80bce1be257e55bae056fba40d963` |
| `player_identification_test_public.csv` | `fbf6f14de57bc37bbec6e3d732143a2561887d37b1f36271223ad1a14810ac74` |
| `player_identification_test_private.csv` | `eedd2630731070633dfa264897ac739aec03879f90ac57149bba033d9bd53dee` |
| `player_identification_candidates2_public.csv` | `ac3d7d20d1a19ee7e670638b556f5c3cc021904ca4e1644bc0d0bcec4ff6616d` |
| `player_identification_candidates2_private.csv` | `8e80f94689284722bd21a3f3306c476ee78aead60a4061e74a0f8aaaede36812` |
| `player_identification_test2_public.csv` | `33485b58da29a43d53526f9f83d92ebdc4610948cb0f5249e07bba5cbdd53d6a` |
| `player_identification_test2_private.csv` | `52e01bcc508e3e350f47b6430a7c841531976ac4a9d294fc2328b4570ca8a527` |
| `sample_submission_player_v2.csv` | `70bf0e3cd31ce8aa0ad438c576f77b3f8c0ca3a49a8089bd026108ad3ba26c9c` |

- 帶 `2` 的檔案是 11 月 8 日更新版新增的題組（`pub2_`、`priv2_`）。
- 程式執行前會核對上表的雜湊，檔案不同時會停止。

## 官方訓練集（只有重訓 E4 編碼器才需要）

`python player_identification/scripts/train.py --encoders` 會讀取官方訓練集（v1.1 版，含 `HD[n]` 屬性）。請放到 `data/official/training/`：

| 檔案 | 大小（bytes） | SHA-256 |
|---|---:|---|
| `train_D.csv` | 141,401,228 | `4f2fab08929bae7fd6a142044a2ba98d093df17f1e7ca7834ab52ec721f3df78` |
| `train_C.csv` | 141,054,668 | `4e10a420fae0b0e03f1574e3c7cd0e63731ddf2719e5f135889ff81a95b53e6d` |
| `train_B.csv` | 140,652,464 | `77f8042c60e6e193364f0dab99c816bf65fa8b3d478cda6cf4d55ae304c485a6` |
| `train_A.csv` | 139,029,626 | `4f95c02cac49fc913b113a5250d2722c242e54145c9432d7d82f50cc9d8df47d` |
| `train_1D.csv` | 139,173,714 | `33c405a0a51b1e57dcb25a597585141dfccb5508c2e4635ac654174729fed1f8` |
| `train_2D.csv` | 137,766,630 | `950864405342dd5e58107e7daa4aeb3cec17e3aa821034d83fc1668410e6dca9` |
| `train_3D.csv` | 135,736,356 | `6f832c44d04a167b1458ab3c9c3a0625adcf37d656ca540b4e758039d7057a4d` |
| `train_4D.csv` | 135,321,612 | `2c18fbaceb6fc0bb971d99942a1379020224f3924de744edb50e26e004653372` |
| `train_5D.csv` | 135,913,572 | `3efd1ade4d93f8b344b5cf67517ca83cf3e22f8c6efce2ffe472cf10d00aadd5` |
| `train_6D.csv` | 132,914,598 | `a90e226f22319cf37a305668e668914ab99ff9bf5a7d8b16b17a0892e474963d` |

重現提交檔、重訓重排器都不需要訓練集。

## 模型產物（GitHub Release）

由 [`download_artifacts.py`](../download_artifacts.py) 下載到 `player_identification/artifacts/`。每個檔案的大小與 SHA-256 都列在 [`manifest.json`](../player_identification/artifacts/manifest.json)。

| 資料夾 | 內容 | 所屬壓縮檔 |
|---|---|---|
| `embeddings/pub`、`priv`、`pub2`、`priv2` | 四個官方題池每盤棋的四組 E4 嵌入 | inference |
| `experts/<分支>/pub2.npz`、`priv2.npz` | 37 個專家在新題組的分數矩陣（400 題 × 600 位候選人） | inference |
| `human_trunk/pub2_*`、`priv2_*` | 新題組每盤 12 手的 HumanSL 描述向量 | inference |
| `experts/<分支>/` 其餘 8 個檔案 | 37 個專家在八個本地題池的分數矩陣 | training |
| `embeddings/emb_main_E4.npy`、`emb_engine_E4.npy` | 本地題池的 E4 嵌入 | training |
| `human_trunk/train_*` | 本地題池的 HumanSL 描述向量 | training |
| `pools/` | 本地題池定義、訓練集中繼資料（玩家代碼、執色、手數、段位） | training |
| `encoders/` | 四個 E4 編碼器的比賽權重 | encoders |

這些檔案是模型輸出與中繼資料，不含任何棋譜內容。

## 外部資源

本 repo 不包含 KataGo 程式、權重或 Fox 棋譜。比賽期間使用的版本與授權列在[首頁的外部資源表](../README.md#資料與外部資源)。
