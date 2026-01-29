### -

此分支的readme仅记录一些琐碎的改动, 使用方法与原项目一致

#### change

重写webui, 仅保留推理功能

训练中途ctrl+c后保存当前步数的模型, 两次ctrl+c立即终止

添加[whisper-large-v3](https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt)编码器,  [底模链接](https://huggingface.co/byzp/so-vits-svc-4.1-whisper-large-pretrain)

移除了whisper-large-v3 encoder以外的层, 默认推理精度改为int4, vits的默认推理精度改为fp16, 推理仅需3GB VRAM

添加训练参数grad_accumulate, 当batch_size受显存容量限制并且梯度震荡严重时可以尝试增大此参数, 它与batch_size的乘积近似于实际的batch_size
