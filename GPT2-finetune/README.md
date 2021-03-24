# GPT2-finetune

## データセット 
### 訓練時
||入力|出力|
|:--|:--|:--|
|1|hi, how are you doing?\<i'm fine. how about yourself?|, how are you doing?\<PAD\>i'm fine. how about yourself?\>|
|2|i'm fine. how about yourself?\<i'm pretty good. thanks for asking.|'m fine. how about yourself?\<i'm pretty good. thanks for asking.\>|
|3|:|:|

### 推論時
||入力|出力|
|:--|:--|:--|
|1|hi, how are you doing?\<|i'm fine. how about yourself?\>|
|2|i'm fine. how about yourself?\<|i'm pretty good. thanks for asking.\>|
|3|:|:|



## モデル構築
<img src="https://github.com/Jumpei-Fujita/chatbot/blob/main/GPT2-finetune/GPT2-finetune/model.png" width="50%"><br>
事前学習済モデルであるGPT-2を用いた自己回帰モデルを構築した。GPT-2はtransformerのDecoder部分の積み重ねによる構造になっており、様々なテキストを用いて事前に学習されたパラメータが設定されている。
アーキテクチャの概略は上の通りである。
訓練時にはターゲットとなる文章を入力し、それぞれ次の単語を予測するように学習した。誤差関数はクロスエントロピー誤差関数を用いた。
最適化手法はAdamを選択し、ハイパーパラメータとしては以下がある。
|ハイパーパラメータ| |
|:--|:--|
|```lr```|学習率|
|```epochs```|学習エポック数|
|```batch_size```|バッチサイズ|


## テスト結果
### 1.テスト用データに対する応答
<img src="https://github.com/Jumpei-Fujita/chatbot/blob/main/GPT2-finetune/GPT2-finetune/test.png" width="70%">

### 2.モデル自身で会話
<img src="https://github.com/Jumpei-Fujita/chatbot/blob/main/GPT2-finetune/GPT2-finetune/dialog.png" width="50%">

### 3.学習の様子
![graph](https://github.com/Jumpei-Fujita/chatbot/blob/main/GPT2-finetune/GPT2-finetune/graph.png)

## コードの実行手順
gpt2.ipynbを上から順に実行していく



