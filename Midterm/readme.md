# CNN 時尚 MNIST 圖像分類
# 期中作業

## 專案概述

本專案實作一個卷積神經網路（CNN），用於分類 Fashion MNIST 數據集中的圖像。該數據集包含 70,000 張 28x28 像素的灰階服飾圖像，分為 10 個類別（如襯衫、鞋子、包包等）。

## 本專案如何完成

- 使用 TensorFlow 的 datasets API 直接載入 Fashion MNIST 數據集。
- 將圖像像素值標準化至 0-1 範圍，以提升神經網路的訓練效果。
- 將資料重新調整形狀以增加通道維度，使其符合 CNN 模型輸入需求。
- 建立 CNN 模型，包含兩層卷積層、最大池化層、展平層以及兩層全連接層（Dense Layers）。
- 使用 Adam 優化器及 sparse categorical cross-entropy 損失函數進行編譯。
- 訓練模型共 10 個 epoch，並設置 10% 驗證資料，以監控過擬合情況。
- 模型訓練後，在測試集上進行評估，準確率約為 90%。
- 顯示一筆樣本預測結果以展示模型輸出。

## 原理說明

### 卷積神經網路（CNN）

- **卷積層** 會應用濾波器來偵測圖像的空間特徵（如邊緣、形狀、紋理）。
- **最大池化層（MaxPooling）** 會對特徵圖進行下採樣，減少計算成本並保留主要特徵。
- **全連接層（Dense Layer）** 則根據這些特徵進行最終分類。
- **ReLU 激活函數** 引入非線性，使模型能擬合複雜模式。
- **Softmax 激活**（在損失函數計算時隱式套用）會將模型輸出轉換為機率分佈。
- 模型透過 **反向傳播（Backpropagation）** 和 **Adam 優化器** 進行訓練，最小化分類誤差。

### 數據集說明

- Fashion MNIST 是影像分類任務中常用的標準資料集。
- 像素值正規化有助於神經網路更快收斂。
- 驗證集可幫助觀察模型是否過擬合。

## 執行方式

1. 安裝所需套件：
pip install tensorflow matplotlib

2. 執行主程式：
python cnn_fashion_mnist.py
3. 程式會訓練 CNN 模型、評估測試集準確率，並顯示一筆預測圖像。

## 參考資料

- TensorFlow Fashion MNIST 教學：https://www.tensorflow.org/tutorials/keras/classification  
- DeepLearning.AI 課程內容  
- TensorFlow 與 Keras 官方文件  

## 原創說明

本專案由我自行構思與實作，並使用 ChatGPT 作為輔助工具。專案內容受到 TensorFlow 教學啟發，但程式結構、變數命名與說明文字皆為原創，未直接複製任何程式碼。所列參考資料僅用於理解原理。

## 補充說明

- 我已準備一段簡短影片，包含專案講解、程式碼操作與即時展示，預計於現場進行報告。
- 未來可考慮加入資料擴增、Dropout 層或更深層模型以進一步提升準確率。

---

謝謝！
