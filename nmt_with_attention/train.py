import tensorflow as tf
import tensorflow_text as tf_text
from sklearn.model_selection import train_test_split

from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import unicodedata
import re
import numpy as np
import os
import io
import time
from datetime import datetime
import pathlib

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
    w = "[START] " + w + " [END]"
    return w


# 1. 去除重音符号
# 2. 清理句子
# 3. 返回这样格式的单词对：[ENGLISH, SPANISH]
def load_data(path, num_examples):
    print(os.getcwd())
    lines = io.open(path, encoding="UTF-8").read().strip().split("\n")

    word_pairs = [
        [preprocess_sentence(w) for w in l.split("\t")] for l in lines[:num_examples]
    ]

    return zip(*word_pairs)


# # 返回这样格式的单词对：[ENGLISH, SPANISH]
# def load_data(path, num_examples):
#     print(os.getcwd())
#     text = path.read_text(encoding="utf-8")

#     lines = text.splitlines()
#     pairs = [line.split("\t") for line in lines]
#     pairs = pairs[:num_examples]

#     context = np.array([context for target, context in pairs])
#     target = np.array([target for target, context in pairs])

#     return target, context

en_sentence = "If you need any money, I'll lend you some."
sp_sentence = "¿Todavía está en casa?"
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(sp_sentence).encode("utf-8"))
print()

# 尝试实验不同大小的数据集
# num_examples = None
num_examples = 80000
target_raw, context_raw = load_data(path_to_file, num_examples)
print(target_raw[-1])
print(context_raw[-1])
print()

# # 采用 80 - 20 的比例切分训练集和验证集
# context_raw_train, context_raw_val, target_raw_train, target_raw_val = train_test_split(
#     context_raw, target_raw, test_size=0.2
# )

# total_size = len(context_raw)
# train_size = len(context_raw_train)
# val_size = len(context_raw_val)

# train_raw = tf.data.Dataset.from_tensor_slices((context_raw_train, target_raw_train))
# train_raw = train_raw.shuffle(train_size)
# val_raw = tf.data.Dataset.from_tensor_slices((context_raw_val, target_raw_val))

# print(f"total_size: {total_size}")
# print(f"train_size: {train_size}")
# print(f"val_size: {val_size}")
# print()

# example_context_strings, example_target_strings = next(iter(train_raw.batch(64)))
# print(example_context_strings[:3])
# print(example_target_strings[:3])
# print()


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters="", lower=False, oov_token="[UNK]"
    )
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")
    # https://github.com/tensorflow/tensorflow/issues/47853
    # tensor = tf.RaggedTensor.from_tensor(tensor, padding=0)

    return tensor, lang_tokenizer


input_tensor, inp_lang_tokenizer = tokenize(context_raw)
target_tensor, targ_lang_tokenizer = tokenize(target_raw)
print(input_tensor[:5])
print(target_tensor[:5])
print()

total_size = input_tensor.shape[0]
train_size = int(total_size * 0.8) // 128 * 128
val_size = total_size - train_size

# test1 = tf.data.Dataset.from_tensor_slices((input_ragged_tensor, target_ragged_tensor))
input_tensor_train, input_tensor_val = tf.split(
    tf.random.shuffle(input_tensor), [train_size, val_size], axis=0
)
input_tensor_train = tf.RaggedTensor.from_tensor(input_tensor_train, padding=0)
input_tensor_val = tf.RaggedTensor.from_tensor(input_tensor_val, padding=0)
target_tensor_train, target_tensor_val = tf.split(
    tf.random.shuffle(target_tensor), [train_size, val_size], axis=0
)
target_tensor_train = tf.RaggedTensor.from_tensor(target_tensor_train, padding=0)
target_tensor_val = tf.RaggedTensor.from_tensor(target_tensor_val, padding=0)
# input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = (
#     train_test_split(input_ragged_tensor, target_ragged_tensor, test_size=0.2)
# )

train_ds = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)
).shuffle(train_size)
# dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
val_ds = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
# valset = valset.batch(BATCH_SIZE, drop_remainder=True)
print("done")

# def tf_lower_and_split_punct(text):
#     # Split accented characters.
#     text = tf_text.normalize_utf8(text, "NFKD")
#     text = tf.strings.lower(text)
#     # Keep space, a to z, and select punctuation.
#     text = tf.strings.regex_replace(text, "[^ a-z.?!,¿]", "")
#     # Add spaces around punctuation.
#     text = tf.strings.regex_replace(text, "[.?!,¿]", r" \0 ")

#     # 在单词与跟在其后的标点符号之间插入一个空格
#     # 例如： "he is a boy." => "he is a boy ."
#     # 参考：https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
#     text = tf.strings.regex_replace(text, "([?.!,¿])", r" \1 ")
#     text = tf.strings.regex_replace(text, '[" "]+', " ")
#     # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
#     text = tf.strings.regex_replace(text, "[^a-zA-Z?.!,¿]+", " ")

#     # Strip whitespace.
#     text = tf.strings.strip(text)

#     text = tf.strings.join(["[START]", text, "[END]"], separator=" ")
#     return text

# example_text = tf.constant("¿Todavía está en casa?")
# print(example_text.numpy().decode())
# print(preprocess_sentence(example_text).decode())
# example_text = tf.constant("If you need any money, I'll lend you some.")
# print(example_text.numpy().decode())
# print(tf_lower_and_split_punct(example_text))
# print()

# context_text_processor = tf.keras.layers.TextVectorization(
#     standardize=None, max_tokens=None, ragged=True
# )
# context_text_processor.adapt(train_raw.map(lambda context, target: context))
# # Here are the first 10 words from the vocabulary:
# print(context_text_processor.get_vocabulary()[:10])

# target_text_processor = tf.keras.layers.TextVectorization(
#     standardize=None, max_tokens=None, ragged=True
# )
# target_text_processor.adapt(train_raw.map(lambda context, target: target))
# print(target_text_processor.get_vocabulary()[:10])

# example_context_tokens = context_text_processor(example_context_strings)
# example_target_tokens = target_text_processor(example_target_strings)
# # print(example_context_tokens)

# context_vocab = np.array(context_text_processor.get_vocabulary())
# context_words = context_vocab[example_context_tokens[0].numpy()]
# print(" ".join(context_words))

# target_vocab = np.array(target_text_processor.get_vocabulary())
# target_words = target_vocab[example_target_tokens[0].numpy()]
# print(" ".join(target_words))
# print()

# context_start_id = tf.constant(context_vocab.tolist().index("[START]"), dtype=tf.int64)
# context_end_id = tf.constant(context_vocab.tolist().index("[END]"), dtype=tf.int64)
# target_start_id = tf.constant(target_vocab.tolist().index("[START]"), dtype=tf.int64)
# target_end_id = tf.constant(target_vocab.tolist().index("[END]"), dtype=tf.int64)

context_start_id = tf.constant(inp_lang_tokenizer.word_index["[START]"], dtype=tf.int64)
context_end_id = tf.constant(inp_lang_tokenizer.word_index["[END]"], dtype=tf.int64)
target_start_id = tf.constant(targ_lang_tokenizer.word_index["[START]"], dtype=tf.int64)
target_end_id = tf.constant(targ_lang_tokenizer.word_index["[END]"], dtype=tf.int64)

# plt.subplot(1, 2, 1)
# # to_tensor 会把 ragged tensor 按最大长度补零对齐为 tensor
# plt.pcolormesh(example_context_tokens.to_tensor())
# plt.title("Token IDs")

# plt.subplot(1, 2, 2)
# # to_tensor 会把 ragged tensor 补零对齐为 tensor
# plt.pcolormesh(example_context_tokens.to_tensor() != 0)
# plt.title("Mask")
# plt.savefig("mask.png")


def process_text(context, target):
    """
    这里要注意，如果 dataset 已经使用了 batch 函数，那么输入是 [batch_size, len]
    否则是 [len]
    """
    # context = context_text_processor(context)
    # target = target_text_processor(target)
    context = tf.cast(context, tf.int64)
    target = tf.cast(target, tf.int64)
    targ_in = target[:-1]
    targ_out = target[1:]
    # context = tf.cast(context, tf.int64)
    # targ_in = tf.cast(targ_in, tf.int64)
    # targ_out = tf.cast(targ_out, tf.int64)
    return context, targ_in, targ_out


# tf.data.AUTOTUNE 会根据可用的 CPU 动态设置并行处理数据的数量
train_ds = train_ds.map(process_text, tf.data.AUTOTUNE)
val_ds = val_ds.map(process_text, tf.data.AUTOTUNE)

ex_context_tok, ex_tar_in, ex_tar_out = next(iter(train_ds))
print(ex_context_tok.numpy())
print(ex_tar_in.numpy())
print(ex_tar_out.numpy())
print()

max_inp_len = 0
max_targ_len = 0
for ex_context_tok, ex_tar_in, ex_tar_out in train_ds:
    max_inp_len = max(tf.size(ex_context_tok).numpy(), max_inp_len)
    max_targ_len = max(tf.size(ex_tar_in).numpy(), max_targ_len)
print(f"max_inp_len: {max_inp_len}")
print(f"max_targ_len: {max_targ_len}")
print()


max_length_targ = 80
num_buckets = 4
src_max_len = None
BATCH_SIZE = 128
steps_per_epoch = train_size // BATCH_SIZE
steps_eval_per_epoch = val_size // BATCH_SIZE
embedding_dim = 1024
enc_num_layers = 2
dec_num_layers = 4
units = 1024
dropout = 0.2
lr = 1e-3
grad_clip = 5.0
vocab_inp_size = len(inp_lang_tokenizer.word_index) + 1
vocab_tar_size = len(targ_lang_tokenizer.word_index) + 1


# Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
def batching_func(x):
    return x.padded_batch(
        BATCH_SIZE,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # targ_in
            tf.TensorShape([None]),  # targ_out
        ),
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            tf.cast(0, tf.int64),  # src
            tf.cast(0, tf.int64),  # targ_in
            tf.cast(0, tf.int64),  # targ_out
        ),
    )  # tgt_len -- unused


if num_buckets > 1:

    def key_func(src, targ_in, targ_out):
        # Calculate bucket_width by maximum source sequence length.
        # Pairs with length [0, bucket_width) go to bucket 0, length
        # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
        # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
        if src_max_len:
            bucket_width = (src_max_len + num_buckets - 1) // num_buckets
        else:
            bucket_width = 5

        # Bucket sentence pairs by the length of their source sentence and target
        # sentence.
        bucket_id = tf.maximum(
            tf.size(src) // bucket_width, tf.size(targ_in) // bucket_width
        )
        return tf.cast(tf.minimum(num_buckets, bucket_id), tf.int64)

    def reduce_func(unused_key, windowed_data):
        return batching_func(windowed_data)

    # train_ds = train_ds.group_by_window(
    #     key_func=key_func, reduce_func=reduce_func, window_size=BATCH_SIZE
    # )
    # val_ds = val_ds.group_by_window(
    #     key_func=key_func, reduce_func=reduce_func, window_size=BATCH_SIZE
    # )

    def patch_broken_batch(dataset):
        # 这个 filter 无法工作，因为获取的 shape[0] 为 noneType
        batched_ds = dataset.filter(
            lambda src, targ_in, targ_out: src.shape[0] == BATCH_SIZE
        )
        broken_ds = dataset.filter(
            lambda src, targ_in, targ_out: src.shape[0] != BATCH_SIZE
        )
        broken_ds = broken_ds.unbatch()
        broken_ds = broken_ds.padded_batch(BATCH_SIZE, drop_remainder=True)
        return batched_ds.concatenate(broken_ds)

    def length_func(src, targ_in, targ_out):
        max_len = tf.maximum(tf.size(src), tf.size(targ_in))
        return max_len

    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#bucket_by_sequence_length
    bucket_boundaries = [10, 15, 20]
    bucket_batch_sizes = [BATCH_SIZE for _ in range(len(bucket_boundaries) + 1)]
    train_ds = train_ds.bucket_by_sequence_length(
        element_length_func=length_func,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
        drop_remainder=True,
    )
    val_ds = val_ds.bucket_by_sequence_length(
        element_length_func=length_func,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
        drop_remainder=True,
    )
    # train_ds = patch_broken_batch(train_ds)
    # val_ds = patch_broken_batch(val_ds)

else:
    train_ds = batching_func(train_ds)
    val_ds = batching_func(val_ds)


example_input_batch, example_target_in_batch, example_target_out_batch = next(
    iter(train_ds)
)
print(example_input_batch.shape)
print(example_target_in_batch.shape)
print()

max_inp_len = 0
max_targ_len = 0
batch_size_dict = defaultdict(int)
bad_batches = []
for ex_context_tok, ex_tar_in, ex_tar_out in train_ds:
    max_inp_len = max(ex_context_tok.shape[1], max_inp_len)
    max_targ_len = max(ex_tar_in.shape[1], max_targ_len)
    batch_size_dict[ex_context_tok.shape[0]] += 1
    if ex_context_tok.shape[0] < BATCH_SIZE:
        bad_batches.append(ex_context_tok.shape)
print(f"after bucket, max_inp_len: {max_inp_len}")
print(f"after bucket, max_targ_len: {max_targ_len}")
print(f"after bucket, batches: {batch_size_dict}")
print(f"after bucket, bad batches: {bad_batches}")
print()

# train_ds = train_ds.cache()
# train_ds = train_ds.prefetch(1)
# val_ds = val_ds.cache()
# val_ds = val_ds.prefetch(1)
# train_ds = train_ds.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
# val_ds = val_ds.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))

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
    # x = x[len("[START]") :]
    x, _, _ = x.partition("[END]")
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


# 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地
# 执行。该函数专用于参数张量的精确形状。为了避免由于可变序列长度或可变
# 批次大小（最后一批次较小）导致的再追踪，使用 input_signature 指定
# 更多的通用形状。
# 但是这里用 input_signature 并不方便，因为 dec_states 是一个 tuple，很难指定其形状 :)
@tf.function
def train_step(inp, targ_in, targ_out):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp)

        dec_hidden = enc_hidden

        # dec_hidden = None
        dec_states = decoder.get_initial_state(batch_size=inp.shape[0])
        # 教师强制 - 将目标词作为下一个输入
        for t in range(targ_in.shape[1]):
            # 使用教师强制，第一个词为 [START]
            dec_input = tf.expand_dims(targ_in[:, t], 1)

            # 将编码器输出 （enc_output） 传送至解码器
            logits, dec_states, dec_hidden, _ = decoder(
                dec_input, dec_states, enc_output, dec_hidden
            )

            loss += loss_function(targ_out[:, t], logits)

    batch_loss = loss / int(targ_in.shape[1])
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    if grad_clip is not None:
        gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


@tf.function
def evaluate(inp, targ_in):
    enc_out, enc_hidden = encoder(inp, training=False)

    result = []

    dec_hidden = enc_hidden
    dec_states = decoder.get_initial_state(batch_size=inp.shape[0])
    # [batch_size, 1]，第一个词是 [START]
    dec_input = tf.constant(
        value=target_start_id, shape=(inp.shape[0], 1), dtype=tf.int64
    )
    # TODO: 这里的长度应该是目标句子长度嘛？
    for t in range(targ_in.shape[1]):
        logits, dec_states, dec_hidden, attention_weights = decoder(
            dec_input, dec_states, enc_out, dec_hidden, training=False
        )
        # [batch,]
        predicted_id = tf.argmax(logits, axis=-1)
        # 记录新产生的词语
        result.append(predicted_id)
        # 预测的 ID 被输送回模型
        # [batch, 1]
        dec_input = tf.expand_dims(predicted_id, -1)

    pred = tf.stack(result, axis=1)
    return pred


# @tf.function
def inference(inp):
    enc_out, enc_hidden = encoder(inp, training=False)

    pred_id_list = []
    weights_list = []

    dec_hidden = enc_hidden

    dec_states = decoder.get_initial_state(batch_size=inp.shape[0])
    # [batch_size, 1]，第一个词是 [START]
    dec_input = tf.constant(value=target_start_id, shape=(1, 1), dtype=tf.int64)
    for t in range(max_length_targ):
        logits, dec_states, dec_hidden, attention_weights = decoder(
            dec_input, dec_states, enc_out, dec_hidden, training=False
        )
        # [batch,]
        predicted_id = tf.argmax(logits, axis=-1)
        # 记录新产生的词语
        pred_id_list.append(predicted_id)
        # [inp_len]
        attention_weights = tf.reshape(attention_weights, (-1,))
        weights_list.append(attention_weights)

        if tf.equal(
            predicted_id[0],
            target_end_id,
        ):
            break

        # 预测的 ID 被输送回模型
        # [batch, 1]
        dec_input = tf.expand_dims(predicted_id, -1)

    # [1, max_length_targ]
    pred = tf.stack(pred_id_list, axis=1)
    # [max_length_targ]
    # pred = tf.reshape(pred, (-1,))
    # [max_length_targ, inp_len]
    weights = tf.stack(weights_list, axis=0)
    return pred, weights


def translate(sentence):
    sentence = preprocess_sentence(sentence)
    tokens = inp_lang_tokenizer.texts_to_sequences([sentence])
    # print(tokens)
    inputs = tf.constant(tokens)
    # inputs = tf.expand_dims(inputs, axis=0)

    pred, attention_plot = inference(inputs)

    # inputs = tf.squeeze(inputs, axis=0)
    # context_tokens = context_vocab[inputs]
    # context = inp_lang_tokenizer.sequences_to_texts(inputs)
    # pred = tf.squeeze(pred, axis=0)
    pred = targ_lang_tokenizer.sequences_to_texts(pred.numpy())
    pred = pred[0]
    return pred, sentence, attention_plot


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
    pred, context, attention_plot = translate(sentence)

    print(f"attention shape: {attention_plot.shape}")
    print("Input: %s" % (sentence))
    print("Predicted translation: {}".format(pred))

    attention_plot = attention_plot[: len(pred.split(" ")), : len(context.split(" "))]
    plot_attention(attention_plot, context.split(" "), pred.split(" "), prefix=prefix)


print("params:")
variables = encoder.trainable_variables + decoder.trainable_variables
print_total_parameters(variables)
print()

# 测试一下长句子的效果
print("testing:")
size = 5
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
    bad_batches = []

    # halving learning rate
    if epoch > 8:
        lr /= 2.0
        optimizer.lr.assign(lr)
        print(f"halving learning rate, lr: {optimizer.lr.numpy()}")

    for batch, (inp, targ_in, targ_out) in enumerate(train_ds):
        batch_loss = train_step(inp, targ_in, targ_out)
        total_loss.append(batch_loss.numpy())

        total_lens.append(max(inp.shape[1], targ_in.shape[1]))
        if batch % 100 == 0:
            print(total_lens)
            # print(inp[:3])
            print(
                "Epoch {}, Batch {}, Batch length: avg={}, max={}, Loss {:.4f}".format(
                    epoch,
                    batch,
                    np.mean(total_lens),
                    np.max(total_lens),
                    batch_loss.numpy(),
                )
            )
            total_lens = []

        if inp.shape[0] < BATCH_SIZE:
            bad_batches.append(inp.shape)
    print(f"Epoch {epoch}, bad batches: {bad_batches}")

    print("evaluating...")
    total_acc = []
    total_bleu = []
    infos = []
    total_match = 0
    total_mask = 0
    for batch, (inp, targ_in, targ_out) in enumerate(val_ds):
        pred = evaluate(inp, targ_in)
        match, mask, acc = masked_acc(targ_out, pred)
        # targ_text = target_vocab[targ_out.numpy()]
        # pred_text = target_vocab[pred.numpy()]
        targ_text = targ_lang_tokenizer.sequences_to_texts(targ_out.numpy())
        pred_text = targ_lang_tokenizer.sequences_to_texts(pred.numpy())
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
    size = 5
    # inp = tf.convert_to_tensor(input_tensor_val[-size:])
    # targ = tf.convert_to_tensor(target_tensor_val[-size:])
    # pred = evaluate(inp, targ)
    # tensor2sentence(inp.numpy(), targ.numpy(), pred.numpy(), verbose=True)

    print()

# 恢复检查点目录 （checkpoint_dir） 中最新的检查点
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# plot("¿todavia estan en casa?")
