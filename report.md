### 互联网数据挖掘大作业

#### 数据预处理部分

​	我们对训练数据进行了简单的统计，发现训练集中总共有264415个三元组，其中13580个是正例，250836个是负例，正例和负例比例接近1:18。同时，一个问题可能对应0～10个正确答案和0～30个错误答案。

​	为了解决数据比例严重不均衡的情况，我们使用上采样和下采样相结合的方式。我们训练时只采样正例$N_S$倍的负例，剩余负例丢弃。具体细节说明如下：我们构建一个正例池和一个负例池，每次从正例中采样一个正例的时候，从负例池中随机采样$N_S$个负例返回。

​	我们使用了两种loss function，一种是交叉熵：$Loss =-[y \cdot \log (score)+(1-y) \cdot \log (1-score)]$，另一种是MarginLoss：$Loss = \sum \max({score}_{negative} - {score}_{positive}+margin, 0)$。其中score代表模型给问题、答案对(Question, Answer)的打分，代表其匹配程度。如果选用MarginLoss，还需要对正例进行上采样，保证正负比是1:1。另外在计算MarginLoss的时候，我们考虑`self-adversarial negative sampling`的方式，其本质上是我们按照如下分布来进行负采样：$P\left(q_{j}^{\prime}, a_{j}^{\prime} |\left\{\left(q_{i}, a_{i}\right)\right\}\right)=\frac{\exp \alpha f\left(\mathbf{q}_{j}^{\prime}, \mathbf{a}_{j}^{\prime}\right)}{\sum_{i} \exp \alpha f\left(\mathbf{q}_{i}^{\prime}, \mathbf{a}_{i}^{\prime}\right)}$ ，即按照更高的可能性去采样更加难学的负例。$\alpha$为超参数，当其为0是意味着等概率采样，具体代码实现时，$Loss$函数如下：$Loss = \sum P(negative) \max({score}_{negative} - {score}_{positive}+margin, 0)$  

​	关于预训练部分，我们引用了以百度百科作为文库、使用skip-gram预训练好的[中文词向量集合]((https://github.com/Embedding/Chinese-Word-Vectors))，[百度网盘地址](https://pan.baidu.com/s/1Gndr0fReIq_oJ3R34CxlPg)在这里。这里词向量解压之后得到一个文本文件，将其命名为baidubaike放置于`data`目录下。该文件的格式是`word num1 ... num300`。我们首先将其中所有的词汇提取出来的到一个`pretrained_word.txt`文件放置于`data`目录下，以这个文件作为自定义词典，使用jieba对问题和答案进行分词。分词后总共得到301836个词汇，其中编号0、1、2分别和`[UNKNOWN]`，`[START]`，`[PAD]`对应，剩下的词汇对应各个具体的词汇。这样就可以把问题和答案都转化为一个编号组成的序列，可以作为embedding层的输入。

#### 模型部分

##### RNN

​	这里使用GRU模型作为baseline。使用一层GRU分别对代表问题的词向量序列和代表答案的词向量序列进行编码，将编码得到的新的词向量序列通过最大池化之后得到问题向量和答案向量。计算他们的余弦相似度即可得到问题和答案的相似度。

​	值得一提的是，我们并不希望GRU训练的时候RNN会把padding部分也进行编码，只希望它考虑前面有效的单词。这里我们使用了pytorch的 **pack_padded_sequence**函数，它保证padding不会真正进入到 GRU 中影响效果，但是需要我们事先把 input_seqs 先按长度从大到小排列一下，然后把排序后每个序列的真正长度 input_lengths 传进来，然后包装好放进 GRU 里， GRU 运行完了再用 **pad_packed_sequence** 这个函数解包一下。与此相比，tensorflow提供了dynamic_rnn函数，显得更加方便。

##### CNN

​	我们从字面思考本问答问题，可以将其视为一个文本匹配任务。我们发现如果文本和问句中具有相同或者表达差不多意思的词，那个该文本为该问句的可能性会增加。RNN可以捕捉句子语义层度的信息，但是无法捕捉字面层度的信息，因为词汇在经过不同GRU单元的编码之后会丢失基础信息而带上上下文信息。为此，我们引入CNN来编码相关信息。

​	首先，我们构造一个相似度矩阵(similarity matrix)，其第$i$行第$j$列代表着问句中第$i$个词和文本中第$j$个词的相似度，可以通过如下公式计算：$M_{i j}=u_{i} \otimes v_{j}$。$u_i$和$v_j$分别代表问句中第$i$个词的词向量和文本中第$j$个词的词向量，$\otimes$代表相似度计算，这里我们采用余弦相似度。

​	然后，我们将这个矩阵通过一个卷积层和一个最大池化层。通过最大池化层时，我们考虑两个方向的最大池化，分别对应着问题层面的最大池化和文本层面的最大池化，即对于问题中的每个词，我们找到文本句子中和它最匹配的那个词；以及对于文本中的每个词，找到句子中和它最匹配的那个词。最后将两个向量分别通过一个全连接层即可得到两个值。具体公式如下：
$$
\begin{array}{l}{
g_{i, j}^{k}=\sigma\left(\sum_{s=0}^{r_{k}-1} \sum_{t=0}^{r_{k}-1} w_{s, t}^{k} \cdot M_{i+s, j+t}+b^{k}\right) \\
y_{i}^{(1, k)}=\max _{0 \leq a x} g_{i, t}^{k}} \\ 
{y_{j}^{(2, k)}=\max _{0 \leq t<d_{2}} g_{t, j}^{k} \\
z_{1}=W_{2} \sigma\left(W_{1}\left[y^{(1,0)} ; y^{(1, K)}\right]+b_{1}\right)+b_{2}} \\ 
{z_{2}=W_{2} \sigma\left(W_{1}\left[y^{(2,0)} ; y^{(2, K)}\right]+b_{1}\right)+b_{2}}\end{array}
$$
其中$d_1$和$d_2$分别代表着相似度矩阵的长和宽。最后可以用如下公式计算出最终相似度：
$$
S\left(P, r^{k}\right)=\operatorname{Sigmoid}\left(W^{T}\left[z_{1} ; z_{2} ; z_{3}\right]+b\right)
$$
$z_3$是通过RNN计算出的相似度。

#### 实验结果

| model   | Recall | Precision | F1   | MAP  | MRR  |
| ------- | ------ | --------- | ---- | ---- | ---- |
| RNN     |        |           |      |      |      |
| RNN&CNN |        |           |      |      |      |
|         |        |           |      |      |      |

