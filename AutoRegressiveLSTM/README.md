# Auto Regressive LSTM 

## データセット 
### 訓練時
||入力|出力|
|:--|:--|:--|
|1|hi, how are you doing?\<PAD\>i'm fine. how about yourself?|, how are you doing?\<PAD\>i'm fine. how about yourself?\<CLS\>|
|2|i'm fine. how about yourself?\<PAD\>i'm pretty good. thanks for asking.|'m fine. how about yourself?\<PAD\>i'm pretty good. thanks for asking.\<CLS\>|
|3|:|:|

### 推論時
||入力|出力|
|:--|:--|:--|
|1|hi, how are you doing?\<PAD\>|i'm fine. how about yourself?\<CLS\>|
|2|i'm fine. how about yourself?\<PAD\>|i'm pretty good. thanks for asking.\<CLS\>|
|3|:|:|



## モデル構築
![model](https://github.com/Jumpei-Fujita/kadai1/blob/master/dentsu_cnn.png)<br>
LSTMを用いた自己回帰モデルを構築した。そのため、EcnoderDecoderフレームワークのDecoder部分を用いた。
アーキテクチャの概略は以下の通りである。
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
<img src="https://github.com/Jumpei-Fujita/chatbot/blob/main/AutoRegressiveLSTM/ARLSTM/test.png" width="70%">

### 2.モデル自身で会話
<img src="https://github.com/Jumpei-Fujita/chatbot/blob/main/AutoRegressiveLSTM/ARLSTM/dialog.png" width="50%">

### 3.学習の様子
![graph](https://github.com/Jumpei-Fujita/chatbot/blob/main/AutoRegressiveLSTM/ARLSTM/graph.png)

## コードの実行手順
AutoRegressiveLSTM.ipynbを上から順に実行していく


