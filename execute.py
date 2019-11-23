# -*- coding:utf-8 -*-
# 导入依赖包
import os
import sys
import time
import tensorflow as tf
import seq2seqModel
import getConfig
import io

# 初始化一个字典，用于存放从配置文件中读取的超参数
gConfig = {}
gConfig=getConfig.get_config(config_file='seq2seq.ini')
# 赋值字典的长度，embedding的维度，神经网络的层级，批处理的大小
vocab_inp_size = gConfig['enc_vocab_size']
vocab_tar_size = gConfig['dec_vocab_size']
embedding_dim=gConfig['embedding_dim']
units=gConfig['layer_size']
BATCH_SIZE=gConfig['batch_size']
# 赋值训练数据中的最大值，就是语句的最大长度，如果语句小于这个长度会被padding，如果大于这个长度则会被截断
max_length_inp, max_length_tar=20,20

# 定义一个训练处理函数，就是在训练的前后分别加上start和end
def preprocess_sentence(w):
    w ='start '+ w + ' end'
    #print(w)
    return w

# 读取训练集的数据
def create_dataset(path, num_examples):
    # 打开数据集文件，按行读取，并去除换行符
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    # 循环为每行读取的数据减去start和end
    word_pairs = [[preprocess_sentence(w)for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)

# 定义一个最大长度求取函数，用于求取最长语句的长度
def max_length(tensor):
    return max(len(t) for t in tensor)

# 定义read_data函数，读取训练集的数据，并对数据进行tokenize处理
def read_data(path, num_examples):
    # 将数据进行拆分，也就是拆分成输入集和输出集
    input_lang,target_lang = create_dataset(path,num_examples)
    # 分别对输入数据集和输出数据集进行字符转数字的处理，返回处理后的数字向量和词典
    input_tensor, input_token = tokenize(input_lang)
    target_tensor,target_token = tokenize(target_lang)
    # 最后返回处理后的数字向量和词典
    return input_tensor,input_token,target_tensor,target_token

# 定义字符转换函数，其作用是将字符转换为在字典中对应的索引数字
def tokenize(lang):
    # 实例化一个tokenizer
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=gConfig['enc_vocab_size'], oov_token=3)
    # 使用fit_on_texts的方法在目标数据集上训练，其实就是构建一个数据集的词典
    lang_tokenizer.fit_on_texts(lang)
    # 使用texts_to_sequences对目标数据集进行字符到数字的转换
    tensor = lang_tokenizer.texts_to_sequences(lang)
    # 最后对转换后的数字向量进行padding
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_length_inp,padding='post')
    # 返回数字向量和词典对象
    return tensor, lang_tokenizer

# 调用read_data函数读取数据集数据，并返回相应的结果
input_tensor,input_token,target_tensor,target_token= read_data(gConfig['seq_data'], gConfig['max_train_data_size'])

# 定义训练函数，对训练集数据进行循环训练
def train():
    print("Preparing data in %s" % gConfig['train_data'])
    # 计算将全部训练集数据训练完一遍所需的步数
    steps_per_epoch = len(input_tensor) // gConfig['batch_size']
    print(steps_per_epoch)
    # 对encoder的隐藏层状态进行初始化
    enc_hidden = seq2seqModel.encoder.initialize_hidden_state()
    # 读取模型保存在文件夹中的数据
    checkpoint_dir = gConfig['model_data']
    ckpt = tf.io.gfile.listdir(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # 判断是否存在已经训练好的数据模型，如果存在就加载已有的模型并继续进行训练
    if ckpt:
        print("reload pretrained model")
        seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # 赋值需要进行随机打乱的数据数量，我们这里使用全局全打乱
    BUFFER_SIZE = len(input_tensor)
    # 使用tf.data.Dataset对训练数据进行一系列的处理，包括数据的随机打乱
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor,target_tensor)).shuffle(BUFFER_SIZE)
    # 使用batch方法，将数据按照batch_size的大小进行切割，将余数丢掉
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    # 获取训练开始时间
    start_time = time.time()

    # 开始进行循环训练，设置100次
    for i in range(100):
        # 获取每个epoch开始的时间
        start_time_epoch = time.time()
        total_loss = 0
        # 使用enumerate方法穷举遍历所有的批数据，并调用seq2seqModel中的train_step函数进行训练
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = seq2seqModel.train_step(inp, targ, target_token, enc_hidden)
            total_loss += batch_loss
            # print(batch_loss.numpy())
        # 计算每一步消耗的时间
        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        # 计算每一步的平均loss
        step_loss = total_loss / steps_per_epoch
        # 计算当前训练的步数
        current_steps += steps_per_epoch
        # 计算全部训练每步的平均时间
        step_time_total = (time.time() - start_time) / current_steps
        # 打印出相关的参数
        print('训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss {:.4f}'.format(
                                current_steps, step_time_total, step_time_epoch, step_loss.numpy()))
        # 保存训练完成的模型
        seq2seqModel.checkpoint.save(file_prefix=checkpoint_prefix)
        # 刷新输出屏幕
        sys.stdout.flush()

# 定义预测函数
def predict(sentence):
    # 加载已经完成训练的模型
    checkpoint_dir = gConfig['model_data']
    seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # 对输入的序列进行处理
    sentence = preprocess_sentence(sentence)
    # 将输入序列进行字符转数字的处理
    inputs = [input_token.word_index.get(i,3) for i in sentence.split(' ')]
    # 对输入序列进行padding处理
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],maxlen=max_length_inp,padding='post')
    # 将处理完成的数据转换为tensor
    inputs = tf.convert_to_tensor(inputs)
    # 初始化result字符串
    result = ''
    # 使用全零作为模型的隐藏层初始化向量
    hidden = [tf.zeros((1, units))]
    # 对输入语句进行encoder编码
    enc_out, enc_hidden = seq2seqModel.encoder(inputs, hidden)

    dec_hidden = enc_hidden
    # 构造decoder的输入
    dec_input = tf.expand_dims([target_token.word_index['start']], 0)
    # 开始循环预测输出的词语，最大长度为max_length_tar
    for t in range(max_length_tar):
        # 使用decoder对encoder的输出数据进行处理，得到预测结构、隐藏层状态和attention的权重参数
        predictions, dec_hidden, attention_weights = seq2seqModel.decoder(dec_input, dec_hidden, enc_out)
        # 使用argmax求的预测结果中的最大概率的元素
        predicted_id = tf.argmax(predictions[0]).numpy()
        # 如果预测的结果是end，则停止循环预测
        if target_token.index_word[predicted_id] == 'end':
            break
        # 否则继续进行循环预测，直到达到最大值
        result += target_token.index_word[predicted_id] + ' '
        # 预测输出的结果是作为下一个词预测的输入
        dec_input = tf.expand_dims([predicted_id], 0)
    # 最后返回预测的结果
    return result


if __name__ == '__main__':
    # 从命令行中读取配置文件
    if len(sys.argv) - 1:
        gConfig = getConfig.get_config(sys.argv[1])
    # 默认读取seq2seq.ini
    else:
        gConfig = getConfig.get_config()

    print('\n>> Mode : %s\n' %(gConfig['mode']))

    if gConfig['mode'] == 'train':
        train()
    elif gConfig['mode'] == 'serve':
        print('Serve Usage : >> python3 app.py')
