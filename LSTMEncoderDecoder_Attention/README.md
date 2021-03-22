# LSTM Encder-Decoder Attention

## データセット 
### 訓練時
||入力(Encoder)|入力(Decoder)|出力|
|:--|:--|:--|:--|
|1|hi, how are you doing?|<PAD\>i'm fine. how about yourself?|i'm fine. how about yourself?\<CLS\>|
|2|i'm fine. how about yourself?|\<PAD\>i'm pretty good. thanks for asking.|i'm pretty good. thanks for asking.\<CLS\>|
|3|:|:|:|

### 推論時
||入力(Encoder)|入力(Decoder)|出力|
|:--|:--|:--|:--|
|1|hi, how are you doing?|<PAD\>|i'm fine. how about yourself?\<CLS\>|
|2|i'm fine. how about yourself?|\<PAD\>|i'm pretty good. thanks for asking.\<CLS\>|
|3|:|:|:|



## モデル構築
<img src="https://github.com/Jumpei-Fujita/chatbot/blob/main/LSTMEncoderDecoder_Attention/LSTM_Attention_chatbot/model.png" width="70%"><br>
LSTMを用いたEncoder-Decoder-Attentionモデルを構築した。質問文をEncoderに入力し内部表現を獲得する。その後、その内部表現に対し、Decoderで得た内部表現とAttentionをかける事で、トークンを出力するたびに入力のどの部分に注目するかを学習する。Attentionは内積を用いている。
アーキテクチャの概略は上の通りである。
訓練時にはターゲットとなる文章を入力し、それぞれ次の単語を予測するように学習した。誤差関数はクロスエントロピー誤差関数を用いた。
最適化手法はAdamを選択し、ハイパーパラメータとしては以下がある。
|ハイパーパラメータ| |
|:--|:--|
|``` d_model```|dimension、embedding size|
|```lr```|学習率|
|```epochs```|学習エポック数|
|```batch_size```|バッチサイズ|


## テスト結果
### 1.テスト用データに対する応答
<img src="https://github.com/Jumpei-Fujita/chatbot/blob/main/LSTMEncoderDecoder_Attention/LSTM_Attention_chatbot/attention.png" width="30%">

### 2.モデル自身で会話
<img src="https://github.com/Jumpei-Fujita/chatbot/blob/main/LSTMEncoderDecoder_Attention/LSTM_Attention_chatbot/dialog.png" width="80%">

### 3.学習の様子
![graph](https://github.com/Jumpei-Fujita/chatbot/blob/main/LSTMEncoderDecoder_Attention/LSTM_Attention_chatbot/graph.png)

## コードの実行手順
LSTMEncoderDecoderAttention.ipynbを上から順に実行していく



