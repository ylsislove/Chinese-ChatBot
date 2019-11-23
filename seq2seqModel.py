# 导入依赖包
import tensorflow as tf
import getConfig

# 初始化一个字典，用于存放从配置文件中读取的超参数
gConfig = {}
gConfig=getConfig.get_config(config_file='seq2seq.ini')

# 定义一个Encoder类，实现Encoder-Decoder结构中的Encoder部分
class Encoder(tf.keras.Model):
  # 定义初始化函数，将形参初始化
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,return_sequences=True,return_state=True,
                                   recurrent_initializer='glorot_uniform')

  # 定义执行函数，所有的算法逻辑执行都在call函数中完成
  def call(self, x, hidden):
    # 对输入的序列进行embedding
    x = self.embedding(x)
    # 将embedding的结果输入gru神经网络层，得到输出结果和神经元状态
    output, state = self.gru(x, initial_state = hidden)
    # 返回输出结果和神经元状态
    return output, state

  # 进入隐藏层初始化函数
  def initialize_hidden_state(self):
    # 使用全零矩阵进行初始化
    return tf.zeros((self.batch_sz, self.enc_units))
 
# BahdanauAttention是attention机制的一种变形实现，我们定义一个BahdanauAttention来完成attention机制的实现
class BahdanauAttention(tf.keras.Model):
  # 定义初始化函数，将形参初始化
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    # 分别使用神经网络全连接层初始化W1 W2 V，作为计算Q K V的算法
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  # 定义执行函数，所有的算法逻辑执行都在call函数中完成
  def call(self, query, values):
    # 将query序列增加一个维度
    hidden_with_time_axis = tf.expand_dims(query, 1)
    # 使用W1 W2 V计算attention值，也就是score。在计算的过程中，将W1和W2d的计算结果进行了一次非线性变换
    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
    # 使用softmax将score中的元素值按行转换成概率分布作为attention的权重值
    attention_weights = tf.nn.softmax(score, axis=1)
    # 将获得attention权重与输入的序列相乘得到语境向量
    context_vector = attention_weights * values
    # 将语境向量按行求和，得到最后的语境向量
    context_vector = tf.reduce_sum(context_vector, axis=1)
    # 最后返回语境向量和attention权重值
    return context_vector, attention_weights

# 定义一个Decoder类，实现Encoder-Decoder结构中的Decoder部分
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    # 定义初始化函数，将形参初始化
    super(Decoder, self).__init__()
    # 初始化batch_size
    self.batch_sz = batch_sz
    # 初始化神经元数量
    self.dec_units = dec_units
    # 初始化embedding层
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    # 初始化gru
    self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True,
                                   recurrent_initializer='glorot_uniform')
    # 初始化全连接层
    self.fc = tf.keras.layers.Dense(vocab_size)

    # 实例化一个BahdanauAttention
    self.attention = BahdanauAttention(self.dec_units)

  # 定义执行函数，所有的算法逻辑执行都在call函数中完成
  def call(self, x, hidden, enc_output):
    # 首先使用BahdanauAttention对encode的输出和隐藏状态进行attention计算，输出语境向量和attention权重
    context_vector, attention_weights = self.attention(hidden, enc_output)
    # 对decode的输入序列进行embedding计算
    x = self.embedding(x)
    # 将语境向量增加一个维度后与embedding的结果拼接在一起
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    # 将拼接后的结构输入到gru神经网络层，然后返回输出结果和神经元状态
    output, state = self.gru(x)
    # 将输出结果进行维度变换
    output = tf.reshape(output, (-1, output.shape[2]))
    # 将维度变换后的结果输入到输出层，也就是一个全连接神经网络层
    output_x = self.fc(output)
    # 最后返回输出结果，神经元状态和attention的权重
    return output_x, state, attention_weights

# 以下是对超参数的赋值，分别是对输入、输出的vocab_size进行赋值
vocab_inp_size = gConfig['enc_vocab_size']
vocab_tar_size = gConfig['dec_vocab_size']
# 对embedding的维度（或者说长度）进行赋值
embedding_dim=gConfig['embedding_dim']
# 对神经网络的层级进行赋值
units=gConfig['layer_size']
# 对批量的大小进行赋值
BATCH_SIZE=gConfig['batch_size']

# 实例化一个Encoder，并将超参数传入进去
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# 实例化一个BahdanauAttention，head头为10
attention_layer = BahdanauAttention(10)

# 实例化一个Decoder，并将超参数传入进去
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

# 实例化一个优化器，使用Adam优化器
optimizer = tf.keras.optimizers.Adam()

# 实例化一个loss object，使用SparseCategoricalCrossentropy作为loss函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 定义一个loss function，将计算每次训练的结果的loss，以便求解梯度并进行梯度下降的策略优化参数
def loss_function(real, pred):
  # 因为我们在训练前对序列进行了padding操作，所以在我们进行求解loss的时候同样需要把padding带来的噪声给mask掉
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  # 调用loss_object求解loss
  loss_ = loss_object(real, pred)
  # 对mask中的元素进行数据类型转换，与loss_中的数据类型保持一致
  mask = tf.cast(mask, dtype=loss_.dtype)
  # 去除loss中padding带来的噪声
  loss_ *= mask
  # 返回平均loss
  return tf.reduce_mean(loss_)

# 实例化一个模型保存器，以便在训练完成时保存模型的参数
checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)

# 定义一个训练函数，以完成对训练集数据的循环训练
def train_step(inp, targ, targ_lang,enc_hidden):
  # 初始化loss
  loss = 0
  # 当我们使用tf.keras.model进行构造模型时，一般采用tf.GradientTape进行半手工计算梯度，然后将梯度给
  # 优化器进行参数优化。with xxx as : 代表以下的操作都是在同一个spacename下进行的
  with tf.GradientTape() as tape:
    # 首先将训练集输入输出序列和enc_hidden的初始化作为encoder的输入
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    # 然后dec_hidden共享enc_hidden的值
    dec_hidden = enc_hidden
    # 使用字典中的start索引构建一个decoder的输入，也就是意味着第一个词的开始
    dec_input = tf.expand_dims([targ_lang.word_index['start']] * BATCH_SIZE, 1)

    # 然后接着开始进行强制循环，把输出的词作为下一个循环的decoder的输入
    for t in range(1, targ.shape[1]):
      # 把输出的词作为下一个循环的decoder的输入
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      # 调用loss function计算训练的loss
      loss += loss_function(targ[:, t], predictions)
      # 使用标注数据来构建decoder的输入
      dec_input = tf.expand_dims(targ[:, t], 1)

  # 计算批量训练的平均loss
  batch_loss = (loss / int(targ.shape[1]))
  # 构造encoder和decoder中可以被优化的参数
  variables = encoder.trainable_variables + decoder.trainable_variables
  # 计算梯度
  gradients = tape.gradient(loss, variables)
  # 使用梯度优化可以被优化的参数
  optimizer.apply_gradients(zip(gradients, variables))
  # 返回批量loss
  return batch_loss






