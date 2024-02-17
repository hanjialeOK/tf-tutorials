import tensorflow as tf
import tensorflow_addons as tfa


class Encoder(tf.keras.Model):
    def __init__(
        self, vocab_size, embedding_dim, num_layers, enc_units, batch_sz, dropout
    ):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=False
        )
        self.rnn_cells = tf.keras.layers.StackedRNNCells(
            [
                tf.keras.layers.LSTMCell(
                    enc_units,
                    # recurrent_initializer="glorot_uniform",
                    dropout=dropout,
                    # recurrent_dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # self.rnn = tf.keras.layers.RNN(
        #     self.rnn_cells,
        #     return_sequences=True,
        #     return_state=True,
        # )
        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode="ave",
            layer=tf.keras.layers.RNN(
                self.rnn_cells,
                return_sequences=True,
                return_state=True,
            ),
        )
        # self.projection = tf.keras.layers.Dense(enc_units)

    def call(self, inputs, training=True):
        inputs = self.embedding(inputs)
        result_list = self.rnn(inputs, training=training)
        output = result_list[0]
        # 只取最后一个作为隐层输出
        state = output[:, -1, :]
        # output = self.projection(output)
        return output, state

    def initialize_hidden_state(self, batch_sz):
        # return tf.zeros((self.batch_sz, self.enc_units))
        return self.rnn_cells.get_initial_state(batch_size=batch_sz, dtype=tf.float32)


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        """
        input:
            query: [batch, 1, depth]
            values: [batch, len_v, depth]
        output:
            attention_weights: [batch, 1, len_v]
            context: [batch, 1, depth]
        """
        # [batch, 1, depth] + [batch, len_v, depth]
        # score: [batch, len_v, 1]
        score = self.V(tf.nn.tanh(self.W1(query) + self.W2(values)))

        # [batch, len_v, 1]
        attention_weights = tf.nn.softmax(score, axis=1)

        # [batch, len_v, 1] * [batch, len_v, depth] = [batch, len_v, depth]
        context = attention_weights * values
        # [batch, 1, depth]
        context = tf.reduce_sum(context, axis=1, keepdims=True)

        # [batch, len_v]
        attention_weights = tf.squeeze(attention_weights, axis=-1)
        # [batch, 1, len_v]
        attention_weights = tf.expand_dims(attention_weights, 1)

        return context, attention_weights


class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        # self.wq = tf.keras.layers.Dense(units)
        # self.wk = tf.keras.layers.Dense(units)

    def call(self, query, values):
        """
        input:
            query: [batch, len_q, depth]
            values: [batch, len_v, depth]
        output:
            attention_weights: [batch, len_q, len_v]
            context: [batch, len_q, depth]
        """
        # [batch, len_q, depth] . [batch, depth, len_v] = [batch, len_q, len_v]
        score = tf.matmul(query, values, transpose_b=True)

        # [batch, len_q, len_v]
        attention_weights = tf.nn.softmax(score, axis=-1)

        # [batch, len_q, len_v] . [batch, len_v, depth] = [batch, len_q, depth]
        context = tf.matmul(attention_weights, values)

        return context, attention_weights


class Decoder(tf.keras.Model):
    def __init__(
        self, vocab_size, embedding_dim, num_layers, dec_units, batch_sz, dropout
    ):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=False
        )
        self.rnn_cells = tf.keras.layers.StackedRNNCells(
            [
                tf.keras.layers.LSTMCell(
                    dec_units,
                    # recurrent_initializer="glorot_uniform",
                    # dropout=dropout,
                    # recurrent_dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.rnn = tf.keras.layers.RNN(
            self.rnn_cells,
            return_sequences=True,
            return_state=True,
        )
        self.fc = tf.keras.layers.Dense(vocab_size)

        # 用于注意力
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, inputs, hidden, enc_output, training=True):
        """
        input:
            inputs: [batch, 1]
            hidden: [batch, depth]
            enc_output: [batch, seq_len, depth]
        output:
            context: [batch, 1, depth]
            attention_weights: [batch, 1, seq_len]
        """
        # 把隐藏层加在一起
        # context: [batch, depth]
        # attention_weights: [batch, seq_len, 1]
        context, attention_weights = self.attention(
            tf.expand_dims(hidden, 1), enc_output
        )

        # [batch, 1, embedding_dim]
        inputs = self.embedding(inputs)

        # [batch, 1, embedding_dim + depth]
        inputs = tf.concat([context, inputs], axis=-1)

        # 将合并后的向量传送到 GRU
        # output: [batch, 1, depth]
        # state: [batch, depth]
        result_list = self.rnn(inputs, training=training)
        output = result_list[0]
        # 只取最后一个作为隐层输出
        state = output[:, -1, :]

        # [batch*1, depth]
        output = tf.reshape(output, (-1, output.shape[2]))

        # [batch, vocab]
        inputs = self.fc(output)

        return inputs, state, attention_weights


class Decoder2(tf.keras.Model):
    def __init__(
        self, vocab_size, embedding_dim, num_layers, dec_units, batch_sz, dropout
    ):
        super(Decoder2, self).__init__()
        self.num_layers = num_layers
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.dropout_rate = dropout
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=False
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.cells = [
            tf.keras.layers.LSTMCell(
                dec_units,
            )
            for _ in range(num_layers)
        ]
        self.wc = tf.keras.layers.Dense(dec_units)
        self.projection = tf.keras.layers.Dense(vocab_size)

        # 用于注意力
        self.attention = LuongAttention(self.dec_units)
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def get_initial_state(self, batch_size):
        return tuple(
            cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
            for cell in self.cells
        )

    def call(self, inputs, states, enc_output, hidden=None, training=True):
        """
        input:
            inputs: [batch, 1]
            states: tuple ([batch, depth], ...)
            enc_output: [batch, seq_len, depth]
            hidden: [batch, depth]
        output:
            context: [batch, 1, depth]
            attention_weights: [batch, 1, seq_len]
        """
        # [batch, 1, embedding_dim]
        inputs = self.embedding(inputs)
        # [batch, embedding_dim]
        inputs = tf.squeeze(inputs, axis=1)

        # feed input
        # [batch, embedding_dim + depth]
        inputs = self.add([inputs, hidden])

        new_states = []
        # output: [batch, depth]
        # state: [batch, depth]
        # states = self.get_initial_state(batch_size=inputs.shape[0])
        last_outputs, state_0 = self.cells[0](inputs, states[0])
        new_states.append(state_0)

        # context: [batch, 1, depth]
        # attention_weights: [batch, seq_len, 1]
        context, attention_weights = self.attention(
            tf.expand_dims(last_outputs, 1), enc_output
        )
        context = tf.squeeze(context, axis=1)

        for i in range(1, len(self.cells)):
            merged_inputs = tf.concat([last_outputs, context], axis=-1)
            outputs, state_i = self.cells[i](merged_inputs, states[i])
            outputs = self.dropout(outputs, training=training)
            new_states.append(state_i)
            if i >= 2:
                outputs += last_outputs
            last_outputs = outputs

        # [batch, 1, depth + depth]
        new_hidden = tf.nn.tanh(self.wc(tf.concat([last_outputs, context], axis=-1)))
        # [batch, depth]
        # new_hidden = self.layernorm(self.add([last_outputs, context]))

        # [batch, vocab]
        final = self.projection(new_hidden)

        return final, tuple(new_states), new_hidden, attention_weights


class Decoder3(tf.keras.Model):
    def __init__(
        self, vocab_size, embedding_dim, num_layers, dec_units, batch_sz, dropout
    ):
        super(Decoder3, self).__init__()
        self.num_layers = num_layers
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.dropout = dropout
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=False
        )
        self.stacked_cell = tf.keras.layers.StackedRNNCells(
            [
                tf.keras.layers.LSTMCell(
                    dec_units,
                    # recurrent_initializer="glorot_uniform",
                    # dropout=dropout,
                    # recurrent_dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.rnn = tf.keras.layers.RNN(
            self.stacked_cell,
            return_sequences=True,
            return_state=True,
        )
        self.projection = tf.keras.layers.Dense(vocab_size)

        # 用于注意力
        self.attention = LuongAttention(self.dec_units)
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def get_initial_state(self, batch_size):
        return self.stacked_cell.get_initial_state(
            batch_size=batch_size, dtype=tf.float32
        )

    def call(self, inputs, states, enc_output, hidden=None, training=True):
        """
        input:
            inputs: [batch, 1]
            states: tuple ([batch, depth], ...)
            enc_output: [batch, seq_len, depth]
            hidden: [batch, depth]
        output:
            context: [batch, 1, depth]
            attention_weights: [batch, 1, seq_len]
        """
        # [batch, 1, embedding_dim]
        inputs = self.embedding(inputs)

        # feed input
        # [batch, 1, embedding_dim + depth]
        inputs = self.add([inputs, tf.expand_dims(hidden, 1)])

        # output: [batch, depth]
        # state: [batch, depth]
        rnn_outputs = self.rnn(inputs, initial_state=None, training=training)
        outputs = rnn_outputs[0]
        new_states = rnn_outputs[1:]

        # context: [batch, 1, depth]
        # attention_weights: [batch, seq_len, 1]
        context, attention_weights = self.attention(outputs, enc_output)

        # [batch, 1, depth + depth]
        # output = tf.nn.tanh(self.wc(tf.concat([outputs, context], axis=-1)))
        # [batch, 1, depth]
        outputs = self.layernorm(self.add([context, outputs]))
        new_hidden = tf.squeeze(outputs, axis=1)

        # [batch, vocab]
        final = self.projection(new_hidden)

        return final, tuple(new_states), new_hidden, attention_weights
