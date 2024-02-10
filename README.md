# tf-tutorials

来自 tensorflow 的官方教程 [https://www.tensorflow.org/tutorials/quickstart/advanced?hl=zh-cn](https://www.tensorflow.org/tutorials/quickstart/advanced?hl=zh-cn)

## 要求

Python 3.8+, Flax 0.4.1+, PyTorch 1.11+, and TensorFlow 2.6+

## Eager Execution

tf2 默认使用 eager 模式（即时计算，类似 pytorch，取消了计算图），但是需要 tf.GradientTape （梯度磁带）跟踪运算，然后调用 tape.gradient 反向播放条带计算梯度。

虽然 Eager Execution 增强了开发和调试的交互性，但 TensorFlow 1.x 样式的计算图执行在分布式训练、性能优化和生产部署方面具有优势。为了弥补这一差距，您可以使用 tf.function 将程序转换为计算图。

建议**先在 Eager 模式下调试，然后使用 @tf.function 进行装饰**。

## vscode 配置

需要插件：`python` 和 `markdownlint`

配置 workspace settings.json

```c
{
    "[markdown]": {
        "editor.formatOnSave": true,
        "editor.formatOnPaste": true
    },
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true
    },
}
```
