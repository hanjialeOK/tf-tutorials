import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
from datetime import datetime

from model import (
    Encoder,
    Decoder,
    BahdanauAttention,
    LuongAttention,
    Decoder2,
    Decoder3,
)
from bleu import compute_bleu

# path_to_zip = tf.keras.utils.get_file(
#     "spa-eng.zip",
#     origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
#     extract=True,
# )


def print_total_parameters(variables):
    total_parameters = 0
    for variable in variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        print(
            "%s  dim=%i shape=%s params=%i"
            % (
                variable.name,
                len(shape),
                shape,
                variable_parameters,
            )
        )
        total_parameters += variable_parameters
    print(
        f"total_parameters = {total_parameters}, about {total_parameters / 1e6:.1f} Million, at least need {total_parameters*4 / 1e6:.1f} MB size"
    )


physical_devices = tf.config.list_physical_devices("GPU")
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

path_to_file = "./spa-eng/spa.txt"


# 将 unicode 文件转换为 ascii
def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    # 参考：https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    # 除去最前面和最后面的空格
    w = w.rstrip().strip()

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    w = "<start> " + w + " <end>"
    return w


en_sentence = "May I borrow this book?"
sp_sentence = "¿Puedo tomar prestado este libro?"
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(sp_sentence).encode("utf-8"))


# 1. 去除重音符号
# 2. 清理句子
# 3. 返回这样格式的单词对：[ENGLISH, SPANISH]
def create_dataset(path, num_examples):
    print(os.getcwd())
    lines = io.open(path, encoding="UTF-8").read().strip().split("\n")

    word_pairs = [
        [preprocess_sentence(w) for w in l.split("\t")] for l in lines[:num_examples]
    ]

    return zip(*word_pairs)


en, sp = create_dataset(path_to_file, None)
print(en[-1])
print(sp[-1])


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # 创建清理过的输入输出对
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


# 尝试实验不同大小的数据集
num_examples = None
# num_examples = 80000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(
    path_to_file, num_examples
)

# 计算目标张量的最大长度 （max_length）
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

# 采用 80 - 20 的比例切分训练集和验证集
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = (
    train_test_split(input_tensor, target_tensor, test_size=0.2)
)

# 显示长度
print(
    len(input_tensor_train),
    len(target_tensor_train),
    len(input_tensor_val),
    len(target_tensor_val),
)


def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


print("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print()
print("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[0])


def tensor2sentence(inp, targ, pred, verbose=False):
    """
    输入tensor，转为句子。
        targ: [batch, len]
        pred: [batch, len]
    """
    inp_result = inp_lang.sequences_to_texts(inp)
    targ_result = targ_lang.sequences_to_texts(targ)
    pred_result = targ_lang.sequences_to_texts(pred)

    if verbose:
        for x1, x2, x3 in zip(inp_result, targ_result, pred_result):
            print(f"input: {x1}")
            print(f"targ: {x2}")
            print(f"pred: {x3}")

    return inp_result, targ_result, pred_result


# tensor2sentence(
#     input_tensor_train[:10], target_tensor_train[:10], target_tensor_train[:10]
# )

BATCH_SIZE = 128
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
steps_eval_per_epoch = len(input_tensor_val) // BATCH_SIZE
embedding_dim = 1024
enc_num_layers = 2
dec_num_layers = 4
units = 1024
dropout = 0.2
lr = 1e-3
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)
).shuffle(input_tensor_train.shape[0])
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

valset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
valset = valset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

encoder = Encoder(
    vocab_inp_size, embedding_dim, enc_num_layers, units, BATCH_SIZE, dropout
)
# 样本输入
sample_output, sample_hidden = encoder(example_input_batch)
print(
    "Encoder output shape: (batch size, sequence length, units) {}".format(
        sample_output.shape
    )
)
print("Encoder Hidden state shape: (batch size, units) {}".format(sample_hidden.shape))

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(
    tf.expand_dims(sample_hidden, 1), sample_output
)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print(
    "Attention weights shape: (batch_size, sequence_length, 1) {}".format(
        attention_weights.shape
    )
)

Luong_layer = LuongAttention(units)
Luong_result, Luong_weights = Luong_layer(
    tf.expand_dims(sample_hidden, 1), sample_output
)

print("Luong result shape: (batch size, units) {}".format(Luong_result.shape))
print(
    "Luong weights shape: (batch_size, sequence_length, 1) {}".format(
        Luong_weights.shape
    )
)

decoder = Decoder2(
    vocab_tar_size, embedding_dim, dec_num_layers, units, BATCH_SIZE, dropout
)
dec_states = decoder.get_initial_state(batch_size=BATCH_SIZE)
sample_decoder_output, _, _, _ = decoder(
    tf.random.uniform((BATCH_SIZE, 1)), dec_states, sample_output, sample_hidden
)
print(
    "Decoder output shape: (batch_size, vocab size) {}".format(
        sample_decoder_output.shape
    )
)


optimizer = tf.keras.optimizers.Adam(lr)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / (tf.reduce_sum(mask) + 1e-6)


def masked_acc(y_true, y_pred):
    mask = tf.cast(y_true != 0, tf.float32)
    # y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    match *= mask

    return (
        tf.reduce_sum(match).numpy(),
        tf.reduce_sum(mask).numpy(),
        tf.reduce_sum(match).numpy() / (tf.reduce_sum(mask).numpy() + 1e-6),
    )


def split(x):
    x = x[len("<start>") :]
    x, _, _ = x.partition("<end>")
    return x.rstrip().strip().split()


def get_bleu(targ, pred):
    targ_split = list(map(split, targ))
    targ_split = list(map(lambda x: [x], targ_split))
    pred_split = list(map(split, pred))
    bleu = compute_bleu(reference_corpus=targ_split, translation_corpus=pred_split)
    return 100 * bleu[0]


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_dir = f"./training_checkpoints_{current_time}"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


@tf.function
def train_step(inp, targ):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index["<start>"]] * BATCH_SIZE, 1)

        # dec_hidden = None
        dec_states = decoder.get_initial_state(batch_size=inp.shape[0])
        # 教师强制 - 将目标词作为下一个输入
        for t in range(1, targ.shape[1]):
            # 将编码器输出 （enc_output） 传送至解码器
            logits, dec_states, dec_hidden, _ = decoder(
                dec_input, dec_states, enc_output, dec_hidden
            )

            loss += loss_function(targ[:, t], logits)

            # 使用教师强制
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = loss / int(targ.shape[1])
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


@tf.function
def evaluate(inp, targ):
    enc_out, enc_hidden = encoder(inp, training=False)

    # [batch,]
    dec_input = tf.constant(
        value=targ_lang.word_index["<start>"], shape=(inp.shape[0],), dtype=tf.int64
    )
    # dec_input *= targ_lang.word_index["<start>"]

    result = []
    # 把 <start> 转为 [batch,] 的形状后加入
    result.append(tf.cast(dec_input, tf.int64))

    dec_states = decoder.get_initial_state(batch_size=inp.shape[0])
    dec_hidden = enc_hidden
    # [batch, 1]
    dec_input = tf.expand_dims(dec_input, -1)
    for t in range(max_length_targ - 1):
        logits, dec_states, dec_hidden, attention_weights = decoder(
            dec_input, dec_states, enc_out, dec_hidden, training=False
        )
        # [batch,]
        predicted_id = tf.argmax(logits, axis=-1)
        # 把新产生的词语加入
        result.append(predicted_id)
        # 预测的 ID 被输送回模型
        # [batch, 1]
        dec_input = tf.expand_dims(predicted_id, -1)

    pred = tf.stack(result, axis=1)
    return targ, pred


def translate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_inp, padding="post"
    )
    inputs = tf.convert_to_tensor(inputs)

    result = ""

    enc_out, enc_hidden = encoder(inputs, training=False)

    dec_input = tf.expand_dims([targ_lang.word_index["<start>"]], 0)
    dec_states = decoder.get_initial_state(batch_size=dec_input.shape[0])

    dec_hidden = enc_hidden
    for t in range(max_length_targ - 1):
        logits, dec_states, dec_hidden, attention_weights = decoder(
            dec_input, dec_states, enc_out, dec_hidden, training=False
        )

        # 存储注意力权重以便后面制图
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(logits[0]).numpy()

        result += targ_lang.index_word[predicted_id] + " "

        if targ_lang.index_word[predicted_id] == "<end>":
            # 删除最后的空格
            result = result.rstrip()
            return result, sentence, attention_plot

        # 预测的 ID 被输送回模型
        dec_input = tf.expand_dims([predicted_id], 0)

    # 删除最后的空格
    result = result.rstrip()
    return result, sentence, attention_plot


# 注意力权重制图函数
def plot_attention(attention, sentence, predicted_sentence, prefix=""):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap="viridis")

    fontdict = {"fontsize": 14}

    ax.set_xticklabels([""] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([""] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.join(checkpoint_dir, f"{prefix}plot_{current_time}.png"))


def plot(sentence, prefix=""):
    result, sentence, attention_plot = translate(sentence)

    print(f"attention shape: {attention_plot.shape}")
    print("Input: %s" % (sentence))
    print("Predicted translation: {}".format(result))

    attention_plot = attention_plot[
        : len(result.split(" ")), : len(sentence.split(" "))
    ]
    plot_attention(
        attention_plot, sentence.split(" "), result.split(" "), prefix=prefix
    )


print("params:")
variables = encoder.trainable_variables + decoder.trainable_variables
print_total_parameters(variables)
print()

# 测试一下长句子的效果
print("testing:")
# size = 5
# inp = tf.convert_to_tensor(input_tensor_val[-size:])
# targ = tf.convert_to_tensor(target_tensor_val[-size:])
# targ, pred = evaluate(inp, targ)
# _, targ_text, pred_text = tensor2sentence(
#     inp.numpy(), targ.numpy(), pred.numpy(), verbose=True
# )
# bleu_score = get_bleu(targ_text, pred_text)
print()

EPOCHS = 12
train_summary_writer = tf.summary.create_file_writer(
    os.path.join(checkpoint_dir, "logs")
)

for epoch in range(1, EPOCHS + 1):
    start = time.time()

    total_loss = []
    total_lens = []

    acc = 0

    # halving learning rate
    if epoch > 8:
        lr /= 2.0
        optimizer.lr.assign(lr)
        print(f"halving learning rate, lr: {optimizer.lr.numpy()}")

    for batch, (inp, targ) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ)
        total_loss.append(batch_loss.numpy())
        total_lens.append(max(inp.shape[1], targ.shape[1]))

        if batch % 100 == 0:
            # print(total_lens)
            print(
                "Epoch {}, Batch {}, Loss {:.4f}".format(
                    epoch, batch, batch_loss.numpy()
                )
            )
            total_lens = []

    print("evaluating...")
    total_acc = []
    total_bleu = []
    infos = []
    total_match = 0
    total_mask = 0
    for batch, (inp, targ) in enumerate(valset.take(steps_eval_per_epoch)):
        targ, pred = evaluate(inp, targ)
        # 计算准确率时去掉 <start>
        match, mask, acc = masked_acc(targ[:, 1:], pred[:, 1:])
        _, targ_text, pred_text = tensor2sentence(
            inp.numpy(), targ.numpy(), pred.numpy()
        )
        bleu_score = get_bleu(targ_text, pred_text)
        total_match += match
        total_mask += mask
        infos.append((match, mask))
        total_acc.append(acc)
        total_bleu.append(bleu_score)

    avg_loss = np.mean(total_loss)
    avg_acc = total_match / (total_mask + 1e-6)
    avg_bleu = np.mean(total_bleu)
    print(
        "Epoch {}, Loss {:.4f}, Test Accuracy {:.4f}, BLEU: {:.4f}".format(
            epoch,
            avg_loss,
            avg_acc,
            avg_bleu,
        )
    )
    with train_summary_writer.as_default():
        tf.summary.scalar("train/learning_rate", optimizer.lr.numpy(), step=epoch)
        tf.summary.scalar("train/loss", avg_loss, step=epoch)
        tf.summary.scalar("test/accuracy", avg_acc, step=epoch)
        tf.summary.scalar("test/bleu", avg_bleu, step=epoch)

    # print(infos)
    # print(total_acc)
    # print(total_bleu)
    print("Time taken for 1 epoch {} sec".format(time.time() - start))

    # 每 2 个周期（epoch），保存（检查点）一次模型
    if (epoch) % 2 == 0:
        print("plotting:")
        checkpoint.save(file_prefix=os.path.join(checkpoint_dir, f"ckpt-epoch-{epoch}"))
        plot("¿todavia estan en casa?", prefix=f"epoch-{epoch}-index-1")
        time.sleep(1)
        plot("Esta es mi vida.", prefix=f"epoch-{epoch}-index-2")
        time.sleep(1)
        plot(
            "Tienes que levantar la mano si quieres hablar en la reunión.",
            prefix=f"epoch-{epoch}-index-3",
        )

    # 测试一下长句子的效果
    print("testing:")
    # size = 5
    # inp = tf.convert_to_tensor(input_tensor_val[-size:])
    # targ = tf.convert_to_tensor(target_tensor_val[-size:])
    # targ, pred = evaluate(inp, targ)
    # tensor2sentence(inp.numpy(), targ.numpy(), pred.numpy(), verbose=True)

    print()

# 恢复检查点目录 （checkpoint_dir） 中最新的检查点
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# plot("¿todavia estan en casa?")
