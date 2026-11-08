### -

此分支的readme仅记录一些琐碎的改动, 使用方法与原项目一致

```
wget https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip
unzip rmvpe.zip -d pretrain
mv pretrain/model.pt pretrain/rmvpe.pt

wget https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt -O pretrain/large-v3.pt

#wget https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip
#unzip nsf_hifigan_20221211.zip -d pretrain/nsf_hifigan

python resample.py
python preprocess_flist_config.py
python preprocess_hubert_f0.py --f0_predictor pm
python train.py -c configs/config.json -m 44k
```

#### change

训练中途ctrl+c后保存当前步数的模型, 两次ctrl+c立即终止

添加[whisper-large-v3](https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt)编码器,  [底模链接](https://huggingface.co/byzp/so-vits-svc-4.1-whisper-large-pretrain), 其他组件已被移除

移除了 whisper-large-v3 encoder 以外的层。CUDA 默认使用 INT4 编码，Pascal 及更早架构默认使用 FP16；VITS 默认使用 FP16。显存占用因设备、音频长度和其他进程而异，可通过 `--encoder_precision` 手动选择精度。

配置中的 `nsf-hifigan` 是模型内置的 F0 条件解码器，实现在 `vdecoder/hifigan/models.py`，训练和推理均会使用。`vdecoder/nsf_hifigan` 则是独立的可选后处理增强器，推理时可通过 `-eh` 启用、通过 `-eak` 调整音高适配；其权重和配置位于 `pretrain/nsf_hifigan/`。

添加训练参数grad_accumulate, 当batch_size受显存容量限制并且梯度震荡严重时可以尝试增大此参数, 它与batch_size的乘积近似于实际的batch_size
