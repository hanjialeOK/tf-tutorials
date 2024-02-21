# nmt with attention

原文链接为 [https://www.tensorflow.org/tutorials/text/nmt_with_attention?hl=zh-cn](https://www.tensorflow.org/tutorials/text/nmt_with_attention?hl=zh-cn)

## 下载数据集

```c
wget  http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
unzip spa-eng.zip
```

## 运行

```c
workon py38
pip install tensorflow-addons==0.19.0
pip install tensorflow-text==2.8.2
pip install scikit-learn
python train.py
```

## 思考

- 使用 bucket 技术，也就是不同长度的句子放在不同的的batch中可以明显提高训练速度。然而，bucket 之后会留下一些碎片 batch，如下面这样（标准的 batch_size=128），这些碎片会导致 tf2 出现[重追踪问题](https://www.tensorflow.org/guide/function#tracing) 。如何处理这些碎片呢？能否通过 concat 形成完整的 batch 呢，我在 patch_broken_batch 函数中进行了尝试没有成功。

    ```c
    bad batches: [TensorShape([110, 4]), TensorShape([23, 9]), TensorShape([17, 13]), TensorShape([121, 19]), TensorShape([52, 41])]
    ```

- [tf2 教程](https://www.tensorflow.org/text/tutorials/nmt_with_attention)中的 tf.keras.layers.TextVectorization 在 adapt 时非常慢，我不得不换为 tf.keras.preprocessing.text.Tokenizer 进行处理。
- 模型参考 [https://github.com/OpenNMT/OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf) 中的 RNMT+，文章为 [https://arxiv.org/pdf/1804.09849.pdf](https://arxiv.org/pdf/1804.09849.pdf)。
