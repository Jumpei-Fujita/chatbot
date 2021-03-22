# Auto Regressive LSTM 


## モデル構築
![model](https://github.com/Jumpei-Fujita/kadai1/blob/master/dentsu_cnn.png)<br>
LSTMを用いた自己回帰モデルを構築した。そのため、EcnoderDecoderフレームワークのDecoder部分を用いた。
アーキテクチャの概略は以下の通りである。
訓練時にはターゲットとなる文章を入力し、それぞれ次の単語を予測するように学習した。誤差関数はクロスエントロピー誤差関数を用いた。
最適化手法はAdamを選択し、ハイパーパラメータとしては以下がある。
'''
a
'''

## テスト結果
### 1.テスト用データに対するPrecision, Recall, F-score
|label|0|1|2|3|4|5|6|7|8|9|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|Precision|0.96|0.99|0.96|0.99|0.96|0.96|0.99|0.98|0.97|0.98|
|Recall|0.81|0.82|0.83|0.80|0.80|0.80|0.78|0.80|0.79|0.79|
|F-score|0.88|0.90|0.89|0.88|0.87|0.87|0.87|0.88|0.87|0.88|

### 学習の様子
![model](https://github.com/Jumpei-Fujita/kadai1/blob/master/graph_loss.png)
### 検証用データに対する正解率の推移
![model](https://github.com/Jumpei-Fujita/kadai1/blob/master/graph.png)

## コードの実行手順
mnist.ipynbを上から順に実行していく


