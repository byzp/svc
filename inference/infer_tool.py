import gc
import hashlib
import io
import json
import logging
import os
import pickle
import time
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional, Literal, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import traceback

import maad
import librosa
import numpy as np
import soundfile
import torch
import torchaudio
import bitsandbytes as bnb

import cluster
import utils
from diffusion.unit2mel import load_model_vocoder
from inference import slicer
from models import SynthesizerTrn

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def read_temp(file_name):
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(json.dumps({"info": "temp_dict"}))
        return {}
    else:
        try:
            with open(file_name, "r") as f:
                data = f.read()
            data_dict = json.loads(data)
            if os.path.getsize(file_name) > 50 * 1024 * 1024:
                f_name = file_name.replace("\\", "/").split("/")[-1]
                print(f"clean {f_name}")
                for wav_hash in list(data_dict.keys()):
                    if int(time.time()) - int(data_dict[wav_hash]["time"]) > 14 * 24 * 3600:
                        del data_dict[wav_hash]
        except Exception as e:
            print(e)
            print(f"{file_name} error,auto rebuild file")
            data_dict = {"info": "temp_dict"}
        return data_dict


def write_temp(file_name, data):
    with open(file_name, "w") as f:
        f.write(json.dumps(data))


def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print("executing '%s' costed %.3fs" % (func.__name__, time.time() - t))
        return res
    return run


def format_wav(audio_path):
    if Path(audio_path).suffix == ".wav":
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != "."]
        dirs[:] = [d for d in dirs if d[0] != "."]
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def get_md5(content):
    return hashlib.new("md5", content).hexdigest()


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])


def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


def pad_array(arr, target_length):
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    else:
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_arr = np.pad(arr, (pad_left, pad_right), "constant", constant_values=(0, 0))
        return padded_arr


def split_list_by_n(list_collection, n, pre=0):
    for i in range(0, len(list_collection), n):
        yield list_collection[i - pre if i - pre >= 0 else i: i + n]


class F0FilterException(Exception):
    pass


@dataclass
class EncoderOutput:
    """数据类用于在encoder和vits之间传递数据"""
    c: torch.Tensor
    f0: torch.Tensor
    uv: torch.Tensor
    sid: torch.Tensor
    vol: Optional[torch.Tensor]
    wav: np.ndarray
    n_frames: int
    segment_info: dict  # 用于存储额外的段信息


def replace_linear_with_bnb(model, quantization: str = "int8", exclude_modules: list = None):
    """
    将模型中的Linear层替换为bitsandbytes量化层
    
    Args:
        model: PyTorch模型
        quantization: "int8" 或 "int4"
        exclude_modules: 要排除的模块名称列表
    """
    
    exclude_modules = exclude_modules or []
    
    for name, module in model.named_children():
        if name in exclude_modules:
            continue
            
        if isinstance(module, torch.nn.Linear):
            # 获取原始参数
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            
            # 创建量化层
            if quantization == "int8":
                new_module = bnb.nn.Linear8bitLt(
                    in_features, 
                    out_features, 
                    bias=bias,
                    has_fp16_weights=False,
                    threshold=6.0
                )
            elif quantization == "int4":
                new_module = bnb.nn.Linear4bit(
                    in_features,
                    out_features,
                    bias=bias,
                    compute_dtype=torch.float16,
                    quant_type="nf4"
                )
            else:
                raise ValueError(f"Unknown quantization type: {quantization}")
            
            # 复制权重
            new_module.weight = bnb.nn.Params4bit(
                module.weight.data,
                requires_grad=False,
                quant_type="nf4" if quantization == "int4" else None
            ) if quantization == "int4" else module.weight
            
            if bias:
                new_module.bias = module.bias
                
            setattr(model, name, new_module)
        else:
            # 递归处理子模块
            replace_linear_with_bnb(module, quantization, exclude_modules)
    
    return model


class Svc(object):
    def __init__(
        self,
        net_g_path,
        config_path,
        device=None,
        cluster_model_path="logs/44k/kmeans_10000.pt",
        nsf_hifigan_enhance=False,
        diffusion_model_path="logs/44k/diffusion/model_0.pt",
        diffusion_config_path="configs/diffusion.yaml",
        shallow_diffusion=False,
        only_diffusion=False,
        spk_mix_enable=False,
        feature_retrieval=False,
        precision: Literal["fp32", "fp16", "int8", "int4"] = "fp16",
        encoder_device: Optional[str] = "cuda:0",
        vits_device: Optional[str] = "cuda:0",
        enable_pipeline: bool = True,
        pipeline_queue_size: int = 1024,
    ):
        """
        初始化SVC模型
        
        Args:
            net_g_path: VITS模型路径
            config_path: 配置文件路径
            device: 默认设备 (当encoder_device和vits_device未指定时使用)
            cluster_model_path: 聚类模型路径
            nsf_hifigan_enhance: 是否使用NSF-HiFiGAN增强
            diffusion_model_path: 扩散模型路径
            diffusion_config_path: 扩散模型配置路径
            shallow_diffusion: 是否使用浅扩散
            only_diffusion: 是否仅使用扩散
            spk_mix_enable: 是否启用说话人混合
            feature_retrieval: 是否启用特征检索
            precision: 推理精度 - "fp32", "fp16", "int8", "int4"
            encoder_device: Encoder模型的设备 (如 "cuda:0")
            vits_device: VITS模型的设备 (如 "cuda:1")
            enable_pipeline: 是否启用管道并行推理
            pipeline_queue_size: 管道队列大小
        """
        self.net_g_path = net_g_path
        self.only_diffusion = only_diffusion
        self.shallow_diffusion = shallow_diffusion
        self.feature_retrieval = feature_retrieval
        self.precision = precision
        self.enable_pipeline = enable_pipeline
        self.pipeline_queue_size = pipeline_queue_size
        
        # 设置设备
        if device is None:
            default_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            default_device = device
            
        self.dev = torch.device(default_device)
        self.encoder_device = torch.device(encoder_device if encoder_device else default_device)
        self.vits_device = torch.device(vits_device if vits_device else default_device)
        
        print(f"Encoder device: {self.encoder_device}")
        print(f"VITS device: {self.vits_device}")
        print(f"Precision: {self.precision}")
        
        # 确定数据类型
        if precision == "fp16":
            self.dtype = torch.float16
        elif precision in ["int8", "int4"]:
            self.dtype = torch.float16  # 量化模型通常使用fp16计算
        else:
            self.dtype = torch.float32
            
        self.net_g_ms = None
        if not self.only_diffusion:
            self.hps_ms = utils.get_hparams_from_file(config_path, True)
            self.target_sample = self.hps_ms.data.sampling_rate
            self.hop_size = self.hps_ms.data.hop_length
            self.spk2id = self.hps_ms.spk
            self.unit_interpolate_mode = self.hps_ms.data.unit_interpolate_mode if self.hps_ms.data.unit_interpolate_mode is not None else "left"
            self.vol_embedding = self.hps_ms.model.vol_embedding if self.hps_ms.model.vol_embedding is not None else False
            self.speech_encoder = self.hps_ms.model.speech_encoder if self.hps_ms.model.speech_encoder is not None else "vec768l12"

        self.nsf_hifigan_enhance = nsf_hifigan_enhance
        if self.shallow_diffusion or self.only_diffusion:
            if os.path.exists(diffusion_model_path) and os.path.exists(diffusion_model_path):
                self.diffusion_model, self.vocoder, self.diffusion_args = load_model_vocoder(
                    diffusion_model_path, self.vits_device, config_path=diffusion_config_path
                )
                if self.only_diffusion:
                    self.target_sample = self.diffusion_args.data.sampling_rate
                    self.hop_size = self.diffusion_args.data.block_size
                    self.spk2id = self.diffusion_args.spk
                    self.speech_encoder = self.diffusion_args.data.encoder
                    self.unit_interpolate_mode = self.diffusion_args.data.unit_interpolate_mode if self.diffusion_args.data.unit_interpolate_mode is not None else "left"
                if spk_mix_enable:
                    self.diffusion_model.init_spkmix(len(self.spk2id))
            else:
                print("No diffusion model or config found. Shallow diffusion mode will False")
                self.shallow_diffusion = self.only_diffusion = False

        # load hubert and model
        if not self.only_diffusion:
            self.load_model(spk_mix_enable)
            self.hubert_model = utils.get_speech_encoder(self.speech_encoder, device=self.encoder_device)
            self.volume_extractor = utils.Volume_Extractor(self.hop_size)
        else:
            self.hubert_model = utils.get_speech_encoder(self.diffusion_args.data.encoder, device=self.encoder_device)
            self.volume_extractor = utils.Volume_Extractor(self.diffusion_args.data.block_size)

        # 应用精度设置到hubert模型
        self._apply_precision_to_model(self.hubert_model.model, self.encoder_device, is_encoder=True)

        if os.path.exists(cluster_model_path):
            if self.feature_retrieval:
                with open(cluster_model_path, "rb") as f:
                    self.cluster_model = pickle.load(f)
                self.big_npy = None
                self.now_spk_id = -1
            else:
                self.cluster_model = cluster.get_cluster_model(cluster_model_path)
        else:
            self.feature_retrieval = False

        if self.shallow_diffusion:
            self.nsf_hifigan_enhance = False
        if self.nsf_hifigan_enhance:
            from modules.enhancer import Enhancer
            self.enhancer = Enhancer("nsf-hifigan", "pretrain/nsf_hifigan/model", device=self.vits_device)

        # 初始化管道
        if self.enable_pipeline:
            self._init_pipeline()

    def _apply_precision_to_model(self, model, device, is_encoder=False):
        """应用精度设置到模型"""
        if self.precision == "fp16":
            model = model.half().to(device)
        elif self.precision == "int8":
            model = model.to(device)
            model = replace_linear_with_bnb(model, "int8")
        elif self.precision == "int4":
            model = model.to(device)
            model = replace_linear_with_bnb(model, "int4")
        else:
            model = model.to(device)
        
        model.eval()
        return model

    def load_model(self, spk_mix_enable=False):
        """加载VITS模型"""
        # get model configuration
        self.net_g_ms = SynthesizerTrn(
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            **self.hps_ms.model
        )
        _ = utils.load_checkpoint(self.net_g_path, self.net_g_ms, None)
        
        # 应用精度设置
        if self.precision == "fp16":
            print("Loading VITS model in FP16 precision")
            self.net_g_ms = self.net_g_ms.half()
        elif self.precision == "int8":
                print("Loading VITS model with INT8 quantization")
                self.net_g_ms = replace_linear_with_bnb(self.net_g_ms, "int8")
        elif self.precision == "int4":
                print("Loading VITS model with INT4 quantization")
                self.net_g_ms = replace_linear_with_bnb(self.net_g_ms, "int4")
        else:
            print("Loading VITS model in FP32 precision")
            
        self.net_g_ms = self.net_g_ms.eval().to(self.vits_device)
        
        # 更新dtype
        self.dtype = next(self.net_g_ms.parameters()).dtype
        
        # 删除encoder部分以节省显存
        if hasattr(self.net_g_ms, 'enc_q'):
            del self.net_g_ms.enc_q
            
        if spk_mix_enable:
            self.net_g_ms.EnableCharacterMix(len(self.spk2id), self.vits_device)

    def _init_pipeline(self):
        """初始化管道并行处理"""
        self.encoder_queue = Queue(maxsize=self.pipeline_queue_size)
        self.vits_queue = Queue(maxsize=self.pipeline_queue_size)
        self.result_queue = Queue(maxsize=self.pipeline_queue_size)
        
        self._pipeline_running = False
        self._encoder_thread = None
        self._vits_thread = None
        
        # CUDA流用于异步操作
        if self.encoder_device.type == "cuda":
            self.encoder_stream = torch.cuda.Stream(device=self.encoder_device)
        else:
            self.encoder_stream = None
            
        if self.vits_device.type == "cuda":
            self.vits_stream = torch.cuda.Stream(device=self.vits_device)
        else:
            self.vits_stream = None

    def _encoder_worker(self, f0_predictor, cluster_infer_ratio, f0_filter, cr_threshold, tran, speaker):
        """Encoder工作线程"""
        while self._pipeline_running:
            try:
                item = self.encoder_queue.get()
                if item is None:  # 结束信号
                    self.vits_queue.put(None)
                    break
                    
                wav, segment_info = item
                
                # 在encoder设备上执行
                if self.encoder_stream is not None:
                    with torch.cuda.stream(self.encoder_stream):
                        encoder_output = self._encode_segment(
                            wav, tran, cluster_infer_ratio, speaker, 
                            f0_filter, f0_predictor, cr_threshold, segment_info
                        )
                    self.encoder_stream.synchronize()
                else:
                    encoder_output = self._encode_segment(
                        wav, tran, cluster_infer_ratio, speaker,
                        f0_filter, f0_predictor, cr_threshold, segment_info
                    )
                
                self.vits_queue.put(encoder_output)
                
            except Exception as e:
                if self._pipeline_running:
                    print(f"Encoder worker error: {e}")
                    traceback.print_exc()

    def _vits_worker(self, auto_predict_f0, noice_scale, enhancer_adaptive_key, 
                     k_step, second_encoding, loudness_envelope_adjustment):
        """VITS工作线程"""
        while self._pipeline_running:
            try:
                encoder_output = self.vits_queue.get()
                if encoder_output is None:  # 结束信号
                    self.result_queue.put(None)
                    break
                
                # 在VITS设备上执行
                if self.vits_stream is not None:
                    with torch.cuda.stream(self.vits_stream):
                        result = self._vits_decode(
                            encoder_output, auto_predict_f0, noice_scale,
                            enhancer_adaptive_key, k_step, second_encoding,
                            loudness_envelope_adjustment
                        )
                    self.vits_stream.synchronize()
                else:
                    result = self._vits_decode(
                        encoder_output, auto_predict_f0, noice_scale,
                        enhancer_adaptive_key, k_step, second_encoding,
                        loudness_envelope_adjustment
                    )
                
                self.result_queue.put((result, encoder_output.segment_info))
                
            except Exception as e:
                if self._pipeline_running:
                    print(f"VITS worker error: {e}")
                    traceback.print_exc()

    def _encode_segment(self, wav, tran, cluster_infer_ratio, speaker, 
                        f0_filter, f0_predictor, cr_threshold, segment_info):
        """编码单个音频段"""
        with torch.no_grad():
            # F0预测
            if not hasattr(self, "f0_predictor_object") or self.f0_predictor_object is None or f0_predictor != self.f0_predictor_object.name:
                self.f0_predictor_object = utils.get_f0_predictor(
                    f0_predictor, hop_length=self.hop_size, 
                    sampling_rate=self.target_sample, device=self.encoder_device, 
                    threshold=cr_threshold
                )
            f0, uv = self.f0_predictor_object.compute_f0_uv(wav)

            if f0_filter and sum(f0) == 0:
                raise F0FilterException("No voice detected")
                
            f0 = torch.FloatTensor(f0).to(self.encoder_device)
            uv = torch.FloatTensor(uv).to(self.encoder_device)

            f0 = f0 * 2 ** (tran / 12)
            f0 = f0.unsqueeze(0)
            uv = uv.unsqueeze(0)

            wav_tensor = torch.from_numpy(wav).to(self.encoder_device).to(self.dtype)
            
            if not hasattr(self, "audio16k_resample_transform_encoder"):
                self.audio16k_resample_transform_encoder = torchaudio.transforms.Resample(
                    self.target_sample, 16000
                ).to(self.dtype).to(self.encoder_device)
            wav16k = self.audio16k_resample_transform_encoder(wav_tensor[None, :])[0]

            c = self.hubert_model.encoder(wav16k)
            c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1], self.unit_interpolate_mode)

            if cluster_infer_ratio != 0:
                if self.feature_retrieval:
                    speaker_id = self.spk2id.get(speaker)
                    if not speaker_id and type(speaker) is int:
                        if len(self.spk2id.__dict__) >= speaker:
                            speaker_id = speaker
                    if speaker_id is None:
                        raise RuntimeError("The name you entered is not in the speaker list!")
                    feature_index = self.cluster_model[speaker_id]
                    feat_np = np.ascontiguousarray(c.transpose(0, 1).cpu().numpy())
                    if self.big_npy is None or self.now_spk_id != speaker_id:
                        self.big_npy = feature_index.reconstruct_n(0, feature_index.ntotal)
                        self.now_spk_id = speaker_id
                    score, ix = feature_index.search(feat_np, k=8)
                    weight = np.square(1 / score)
                    weight /= weight.sum(axis=1, keepdims=True)
                    npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                    c = cluster_infer_ratio * npy + (1 - cluster_infer_ratio) * feat_np
                    c = torch.FloatTensor(c).to(self.encoder_device).transpose(0, 1)
                else:
                    cluster_c = cluster.get_cluster_center_result(
                        self.cluster_model, c.cpu().numpy().T, speaker
                    ).T
                    cluster_c = torch.FloatTensor(cluster_c).to(self.encoder_device)
                    c = cluster_infer_ratio * cluster_c + (1 - cluster_infer_ratio) * c

            c = c.unsqueeze(0)
            
            # 获取speaker id
            speaker_id = self.spk2id.get(speaker)
            if not speaker_id and type(speaker) is int:
                if len(self.spk2id.__dict__) >= speaker:
                    speaker_id = speaker
            if speaker_id is None:
                raise RuntimeError("The name you entered is not in the speaker list!")
            sid = torch.LongTensor([int(speaker_id)]).to(self.encoder_device).unsqueeze(0)
            
            # 提取音量
            vol = None
            if self.vol_embedding:
                vol = self.volume_extractor.extract(
                    torch.FloatTensor(wav).to(self.encoder_device)[None, :]
                )[None, :].to(self.encoder_device)
            
            n_frames = f0.size(1)
            
            return EncoderOutput(
                c=c, f0=f0, uv=uv, sid=sid, vol=vol, 
                wav=wav, n_frames=n_frames, segment_info=segment_info
            )

    def _vits_decode(self, encoder_output: EncoderOutput, auto_predict_f0, noice_scale,
                     enhancer_adaptive_key, k_step, second_encoding, loudness_envelope_adjustment):
        """VITS解码"""
        with torch.no_grad():
            # 将数据移动到VITS设备
            c = encoder_output.c.to(self.vits_device).to(self.dtype)
            f0 = encoder_output.f0.to(self.vits_device).to(self.dtype)
            uv = encoder_output.uv.to(self.vits_device).to(self.dtype)
            sid = encoder_output.sid.to(self.vits_device)
            vol = encoder_output.vol.to(self.vits_device).to(self.dtype) if encoder_output.vol is not None else None
            wav = encoder_output.wav
            
            if not self.only_diffusion:
                audio, f0 = self.net_g_ms.infer(
                    c, f0=f0, g=sid, uv=uv, 
                    predict_f0=auto_predict_f0, noice_scale=noice_scale, vol=vol
                )
                audio = audio[0, 0].data.float()
                audio_mel = self.vocoder.extract(audio[None, :], self.target_sample) if self.shallow_diffusion else None
            else:
                audio = torch.FloatTensor(wav).to(self.vits_device)
                audio_mel = None
                
            if self.dtype != torch.float32:
                c = c.to(torch.float32)
                f0 = f0.to(torch.float32)
                uv = uv.to(torch.float32)
                
            if self.only_diffusion or self.shallow_diffusion:
                vol = self.volume_extractor.extract(audio[None, :])[None, :, None].to(self.vits_device) if vol is None else vol[:, :, None]
                if self.shallow_diffusion and second_encoding:
                    if not hasattr(self, "audio16k_resample_transform_vits"):
                        self.audio16k_resample_transform_vits = torchaudio.transforms.Resample(
                            self.target_sample, 16000
                        ).to(self.vits_device)
                    audio16k = self.audio16k_resample_transform_vits(audio[None, :])[0]
                    # 注意: 这里可能需要将hubert移到vits设备或进行设备间通信
                    c = self.hubert_model.encoder(audio16k.to(self.encoder_device)).to(self.vits_device)
                    c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1], self.unit_interpolate_mode)
                f0 = f0[:, :, None]
                c = c.transpose(-1, -2)
                audio_mel = self.diffusion_model(
                    c, f0, vol, spk_id=sid, spk_mix_dict=None, gt_spec=audio_mel,
                    infer=True, infer_speedup=self.diffusion_args.infer.speedup,
                    method=self.diffusion_args.infer.method, k_step=k_step, use_tqdm=False
                )
                audio = self.vocoder.infer(audio_mel, f0).squeeze()
                
            if self.nsf_hifigan_enhance:
                audio, _ = self.enhancer.enhance(
                    audio[None, :], self.target_sample, f0[:, :, None],
                    self.hps_ms.data.hop_length, adaptive_key=enhancer_adaptive_key
                )
            if loudness_envelope_adjustment != 1:
                audio = utils.change_rms(
                    wav, self.target_sample, audio, self.target_sample, 
                    loudness_envelope_adjustment
                )
            
            return audio

    def get_unit_f0(self, wav, tran, cluster_infer_ratio, speaker, f0_filter, f0_predictor, cr_threshold=0.05):
        """获取单元和F0特征"""
        if not hasattr(self, "f0_predictor_object") or self.f0_predictor_object is None or f0_predictor != self.f0_predictor_object.name:
            self.f0_predictor_object = utils.get_f0_predictor(
                f0_predictor, hop_length=self.hop_size, 
                sampling_rate=self.target_sample, device=self.encoder_device, 
                threshold=cr_threshold
            )
        f0, uv = self.f0_predictor_object.compute_f0_uv(wav)

        if f0_filter and sum(f0) == 0:
            raise F0FilterException("No voice detected")
        f0 = torch.FloatTensor(f0).to(self.encoder_device)
        uv = torch.FloatTensor(uv).to(self.encoder_device)

        f0 = f0 * 2 ** (tran / 12)
        f0 = f0.unsqueeze(0)
        uv = uv.unsqueeze(0)

        wav = torch.from_numpy(wav).to(self.encoder_device).to(self.dtype)
        if not hasattr(self, "audio16k_resample_transform"):
            self.audio16k_resample_transform = torchaudio.transforms.Resample(
                self.target_sample, 16000
            ).to(self.dtype).to(self.encoder_device)
        wav16k = self.audio16k_resample_transform(wav[None, :])[0]

        c = self.hubert_model.encoder(wav16k)
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1], self.unit_interpolate_mode)

        if cluster_infer_ratio != 0:
            if self.feature_retrieval:
                speaker_id = self.spk2id.get(speaker)
                if not speaker_id and type(speaker) is int:
                    if len(self.spk2id.__dict__) >= speaker:
                        speaker_id = speaker
                if speaker_id is None:
                    raise RuntimeError("The name you entered is not in the speaker list!")
                feature_index = self.cluster_model[speaker_id]
                feat_np = np.ascontiguousarray(c.transpose(0, 1).cpu().numpy())
                if self.big_npy is None or self.now_spk_id != speaker_id:
                    self.big_npy = feature_index.reconstruct_n(0, feature_index.ntotal)
                    self.now_spk_id = speaker_id
                print("starting feature retrieval...")
                score, ix = feature_index.search(feat_np, k=8)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                c = cluster_infer_ratio * npy + (1 - cluster_infer_ratio) * feat_np
                c = torch.FloatTensor(c).to(self.encoder_device).transpose(0, 1)
                print("end feature retrieval...")
            else:
                cluster_c = cluster.get_cluster_center_result(self.cluster_model, c.cpu().numpy().T, speaker).T
                cluster_c = torch.FloatTensor(cluster_c).to(self.encoder_device)
                c = cluster_infer_ratio * cluster_c + (1 - cluster_infer_ratio) * c

        c = c.unsqueeze(0)
        return c, f0, uv

    def infer(
        self, speaker, tran, raw_path, cluster_infer_ratio=0, auto_predict_f0=False,
        noice_scale=0.4, f0_filter=False, f0_predictor="pm", enhancer_adaptive_key=0,
        cr_threshold=0.05, k_step=100, frame=0, spk_mix=False, second_encoding=False,
        loudness_envelope_adjustment=1, vol=None, start_frame=None, use_tqdm=True
    ):
        """执行推理"""
        if isinstance(raw_path, str) or isinstance(raw_path, io.BytesIO):
            wav, sr = torchaudio.load(raw_path)
            if not hasattr(self, "audio_resample_transform") or self.audio_resample_transform.orig_freq != sr:
                self.audio_resample_transform = torchaudio.transforms.Resample(sr, self.target_sample)
            wav = self.audio_resample_transform(wav).numpy()[0]
        else:
            wav = raw_path
            
        if spk_mix:
            c, f0, uv = self.get_unit_f0(wav, tran, 0, None, f0_filter, f0_predictor, cr_threshold=cr_threshold)
            n_frames = f0.size(1)
            sid = speaker[:, frame: frame + n_frames].transpose(0, 1)
        else:
            speaker_id = self.spk2id.get(speaker)
            if not speaker_id and type(speaker) is int:
                if len(self.spk2id.__dict__) >= speaker:
                    speaker_id = speaker
            if speaker_id is None:
                raise RuntimeError("The name you entered is not in the speaker list!")
            sid = torch.LongTensor([int(speaker_id)]).to(self.vits_device).unsqueeze(0)
            c, f0, uv = self.get_unit_f0(wav, tran, cluster_infer_ratio, speaker, f0_filter, f0_predictor, cr_threshold=cr_threshold)
            n_frames = f0.size(1)
            
        # 将encoder输出移动到vits设备
        c = c.to(self.vits_device).to(self.dtype)
        f0 = f0.to(self.vits_device).to(self.dtype)
        uv = uv.to(self.vits_device).to(self.dtype)
        
        if start_frame is not None:
            c = c[:, :, start_frame:]
            f0 = f0[:, start_frame:]
            uv = uv[:, start_frame:]
            
        with torch.no_grad():
            start = time.time()
            if not self.only_diffusion:
                vol = self.volume_extractor.extract(
                    torch.FloatTensor(wav).to(self.vits_device)[None, :]
                )[None, :].to(self.vits_device) if self.vol_embedding and vol is None else vol
                vol = vol.to(self.dtype) if vol is not None else vol
                audio, f0 = self.net_g_ms.infer(
                    c, f0=f0, g=sid, uv=uv, predict_f0=auto_predict_f0, 
                    noice_scale=noice_scale, vol=vol
                )
                audio = audio[0, 0].data.float()
                audio_mel = self.vocoder.extract(audio[None, :], self.target_sample) if self.shallow_diffusion else None
            else:
                audio = torch.FloatTensor(wav).to(self.vits_device)
                audio_mel = None
                
            if self.dtype != torch.float32:
                c = c.to(torch.float32)
                f0 = f0.to(torch.float32)
                uv = uv.to(torch.float32)
                
            if self.only_diffusion or self.shallow_diffusion:
                vol = self.volume_extractor.extract(audio[None, :])[None, :, None].to(self.vits_device) if vol is None else vol[:, :, None]
                if self.shallow_diffusion and second_encoding:
                    if not hasattr(self, "audio16k_resample_transform"):
                        self.audio16k_resample_transform = torchaudio.transforms.Resample(self.target_sample, 16000).to(self.vits_device)
                    audio16k = self.audio16k_resample_transform(audio[None, :])[0]
                    c = self.hubert_model.encoder(audio16k.to(self.encoder_device)).to(self.vits_device)
                    c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1], self.unit_interpolate_mode)
                f0 = f0[:, :, None]
                c = c.transpose(-1, -2)
                audio_mel = self.diffusion_model(
                    c, f0, vol, spk_id=sid, spk_mix_dict=None, gt_spec=audio_mel,
                    infer=True, infer_speedup=self.diffusion_args.infer.speedup,
                    method=self.diffusion_args.infer.method, k_step=k_step, use_tqdm=use_tqdm
                )
                audio = self.vocoder.infer(audio_mel, f0).squeeze()
            if self.nsf_hifigan_enhance:
                audio, _ = self.enhancer.enhance(
                    audio[None, :], self.target_sample, f0[:, :, None],
                    self.hps_ms.data.hop_length, adaptive_key=enhancer_adaptive_key
                )
            if loudness_envelope_adjustment != 1:
                audio = utils.change_rms(wav, self.target_sample, audio, self.target_sample, loudness_envelope_adjustment)
            use_time = time.time() - start
            print("vits use time:{}".format(use_time))
        return audio, audio.shape[-1], n_frames

    def clear_empty(self):
        """清理VRAM"""
        torch.cuda.empty_cache()

    def unload_model(self):
        """卸载模型"""
        if self.net_g_ms is not None:
            self.net_g_ms = self.net_g_ms.to("cpu")
            del self.net_g_ms
        if hasattr(self, "enhancer"):
            self.enhancer.enhancer = self.enhancer.enhancer.to("cpu")
            del self.enhancer.enhancer
            del self.enhancer
        if hasattr(self, "hubert_model"):
            self.hubert_model.model = self.hubert_model.model.to("cpu")
            del self.hubert_model
        gc.collect()
        torch.cuda.empty_cache()

    def slice_inference(
        self, raw_audio_path, spk, tran, slice_db, cluster_infer_ratio,
        auto_predict_f0, noice_scale, pad_seconds=0.5, clip_seconds=0,
        lg_num=0, lgr_num=0.75, f0_predictor="pm", enhancer_adaptive_key=0,
        cr_threshold=0.05, k_step=100, use_spk_mix=False, second_encoding=False,
        loudness_envelope_adjustment=1
    ):
        """分片推理"""
        if use_spk_mix:
            if len(self.spk2id) == 1:
                spk = self.spk2id.keys()[0]
                use_spk_mix = False
                
        wav_path = Path(raw_audio_path).with_suffix(".wav")
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)
        per_size = int(clip_seconds * audio_sr)
        lg_size = int(lg_num * audio_sr)
        lg_size_r = int(lg_size * lgr_num)
        lg_size_c_l = (lg_size - lg_size_r) // 2
        lg_size_c_r = lg_size - lg_size_r - lg_size_c_l
        lg = np.linspace(0, 1, lg_size_r) if lg_size != 0 else 0

        if use_spk_mix:
            # ... (保持原有的spk_mix逻辑)
            assert len(self.spk2id) == len(spk)
            audio_length = 0
            for slice_tag, data in audio_data:
                aud_length = int(np.ceil(len(data) / audio_sr * self.target_sample))
                if slice_tag:
                    audio_length += aud_length // self.hop_size
                    continue
                if per_size != 0:
                    datas = split_list_by_n(data, per_size, lg_size)
                else:
                    datas = [data]
                for k, dat in enumerate(datas):
                    pad_len = int(audio_sr * pad_seconds)
                    per_length = int(np.ceil(len(dat) / audio_sr * self.target_sample))
                    a_length = per_length + 2 * pad_len
                    audio_length += a_length // self.hop_size
            audio_length += len(audio_data)
            spk_mix_tensor = torch.zeros(size=(len(spk), audio_length)).to(self.dev)
            for i in range(len(spk)):
                last_end = None
                for mix in spk[i]:
                    if mix[3] < 0.0 or mix[2] < 0.0:
                        raise RuntimeError("mix value must higher Than zero!")
                    begin = int(audio_length * mix[0])
                    end = int(audio_length * mix[1])
                    length = end - begin
                    if length <= 0:
                        raise RuntimeError("begin Must lower Than end!")
                    step = (mix[3] - mix[2]) / length
                    if last_end is not None:
                        if last_end != begin:
                            raise RuntimeError("[i]EndTime Must Equal [i+1]BeginTime!")
                    last_end = end
                    if step == 0.0:
                        spk_mix_data = torch.zeros(length).to(self.dev) + mix[2]
                    else:
                        spk_mix_data = torch.arange(mix[2], mix[3], step).to(self.dev)
                    if len(spk_mix_data) < length:
                        num_pad = length - len(spk_mix_data)
                        spk_mix_data = torch.nn.functional.pad(spk_mix_data, [0, num_pad], mode="reflect").to(self.dev)
                    spk_mix_tensor[i][begin:end] = spk_mix_data[:length]

            spk_mix_ten = torch.sum(spk_mix_tensor, dim=0).unsqueeze(0).to(self.dev)
            for i, x in enumerate(spk_mix_ten[0]):
                if x == 0.0:
                    spk_mix_ten[0][i] = 1.0
                    spk_mix_tensor[:, i] = 1.0 / len(spk)
            spk_mix_tensor = spk_mix_tensor / spk_mix_ten
            if not ((torch.sum(spk_mix_tensor, dim=0) - 1.0) < 0.0001).all():
                raise RuntimeError("sum(spk_mix_tensor) not equal 1")
            spk = spk_mix_tensor

        # 选择推理模式
        if self.enable_pipeline and not use_spk_mix:
            return self._pipeline_slice_inference(
                audio_data, audio_sr, spk, tran, cluster_infer_ratio,
                auto_predict_f0, noice_scale, pad_seconds, per_size, lg_size,
                lg_size_r, lg_size_c_l, lg_size_c_r, lg, f0_predictor,
                enhancer_adaptive_key, cr_threshold, k_step, second_encoding,
                loudness_envelope_adjustment
            )
        else:
            return self._sequential_slice_inference(
                audio_data, audio_sr, spk, tran, cluster_infer_ratio,
                auto_predict_f0, noice_scale, pad_seconds, per_size, lg_size,
                lg_size_r, lg_size_c_l, lg_size_c_r, lg, f0_predictor,
                enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix,
                second_encoding, loudness_envelope_adjustment
            )

    def _pipeline_slice_inference(
        self, audio_data, audio_sr, spk, tran, cluster_infer_ratio,
        auto_predict_f0, noice_scale, pad_seconds, per_size, lg_size,
        lg_size_r, lg_size_c_l, lg_size_c_r, lg, f0_predictor,
        enhancer_adaptive_key, cr_threshold, k_step, second_encoding,
        loudness_envelope_adjustment
    ):
        """使用管道并行的分片推理"""
        print("Using pipeline parallel inference...")
        
        # 准备所有段
        segments = []
        global_frame = 0
        
        for slice_idx, (slice_tag, data) in enumerate(audio_data):
            length = int(np.ceil(len(data) / audio_sr * self.target_sample))
            if slice_tag:
                segments.append({
                    'type': 'empty',
                    'length': length,
                    'slice_idx': slice_idx
                })
                global_frame += length // self.hop_size
                continue
                
            if per_size != 0:
                datas = list(split_list_by_n(data, per_size, lg_size))
            else:
                datas = [data]
                
            for k, dat in enumerate(datas):
                per_length = int(np.ceil(len(dat) / audio_sr * self.target_sample)) if per_size != 0 else length
                pad_len = int(audio_sr * pad_seconds)
                dat_padded = np.concatenate([np.zeros([pad_len]), dat, np.zeros([pad_len])])
                
                segments.append({
                    'type': 'audio',
                    'data': dat_padded,
                    'per_length': per_length,
                    'pad_seconds': pad_seconds,
                    'slice_idx': slice_idx,
                    'clip_idx': k,
                    'global_frame': global_frame,
                    'audio_sr': audio_sr
                })
                
                per_length_padded = int(np.ceil(len(dat_padded) / audio_sr * self.target_sample))
                global_frame += per_length_padded // self.hop_size
        
        # 启动管道
        self._pipeline_running = True
        
        # 启动工作线程
        self._encoder_thread = Thread(
            target=self._encoder_worker,
            args=(f0_predictor, cluster_infer_ratio, False, cr_threshold, tran, spk)
        )
        self._vits_thread = Thread(
            target=self._vits_worker,
            args=(auto_predict_f0, noice_scale, enhancer_adaptive_key,
                  k_step, second_encoding, loudness_envelope_adjustment)
        )
        
        self._encoder_thread.start()
        self._vits_thread.start()
        
        # 提交任务到encoder队列
        for seg in segments:
            if seg['type'] == 'audio':
                # 重采样
                raw_path = io.BytesIO()
                soundfile.write(raw_path, seg['data'], seg['audio_sr'], format="wav")
                raw_path.seek(0)
                wav, sr = torchaudio.load(raw_path)
                if not hasattr(self, "audio_resample_transform") or self.audio_resample_transform.orig_freq != sr:
                    self.audio_resample_transform = torchaudio.transforms.Resample(sr, self.target_sample)
                wav = self.audio_resample_transform(wav).numpy()[0]
                
                self.encoder_queue.put((wav, seg))
            else:
                # 空段直接放入结果
                self.result_queue.put((None, seg))
        
        # 发送结束信号
        self.encoder_queue.put(None)
        
        # 收集结果
        results = {}
        completed = 0
        total_audio_segments = sum(1 for s in segments if s['type'] == 'audio')
        total_segments = len(segments)
        
        while completed < total_segments:
            result = self.result_queue.get()
            if result is None:
                break
            audio_out, seg_info = result
            results[f"{seg_info['slice_idx']}_{seg_info.get('clip_idx', 0)}"] = (audio_out, seg_info)
            completed += 1
        
        # 等待线程结束
        self._pipeline_running = False
        self._encoder_thread.join()
        self._vits_thread.join()
        
        # 组装最终音频
        audio = []
        for seg in segments:
            key = f"{seg['slice_idx']}_{seg.get('clip_idx', 0)}"
            if key not in results:
                continue
                
            audio_out, seg_info = results[key]
            
            if seg_info['type'] == 'empty':
                _audio = np.zeros(seg_info['length'])
                audio.extend(list(pad_array(_audio, seg_info['length'])))
            else:
                _audio = audio_out.cpu().numpy()
                pad_len = int(self.target_sample * seg_info['pad_seconds'])
                _audio = _audio[pad_len:-pad_len]
                _audio = pad_array(_audio, seg_info['per_length'])
                
                # 交叉淡入淡出
                if lg_size != 0 and seg_info['clip_idx'] != 0:
                    lg1 = audio[-(lg_size_r + lg_size_c_r): -lg_size_c_r] if lg_size_c_r != 0 else audio[-lg_size_r:]
                    lg2 = _audio[lg_size_c_l: lg_size_c_l + lg_size_r]
                    lg_pre = lg1 * (1 - lg) + lg2 * lg
                    audio = audio[0: -(lg_size_r + lg_size_c_r)] if lg_size_c_r != 0 else audio[0:-lg_size_r]
                    audio.extend(lg_pre)
                    _audio = _audio[lg_size_c_l + lg_size_r:]
                    
                audio.extend(list(_audio))
        
        return np.array(audio)

    def _sequential_slice_inference(
        self, audio_data, audio_sr, spk, tran, cluster_infer_ratio,
        auto_predict_f0, noice_scale, pad_seconds, per_size, lg_size,
        lg_size_r, lg_size_c_l, lg_size_c_r, lg, f0_predictor,
        enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix,
        second_encoding, loudness_envelope_adjustment
    ):
        """顺序分片推理 (原始实现)"""
        global_frame = 0
        audio = []
        
        for slice_tag, data in audio_data:
            print(f"#=====segment start, {round(len(data) / audio_sr, 3)}s======")
            length = int(np.ceil(len(data) / audio_sr * self.target_sample))
            if slice_tag:
                print("jump empty segment")
                _audio = np.zeros(length)
                audio.extend(list(pad_array(_audio, length)))
                global_frame += length // self.hop_size
                continue
            if per_size != 0:
                datas = split_list_by_n(data, per_size, lg_size)
            else:
                datas = [data]
            for k, dat in enumerate(datas):
                per_length = int(np.ceil(len(dat) / audio_sr * self.target_sample)) if per_size != 0 else length
                if per_size != 0:
                    print(f"###=====segment clip start, {round(len(dat) / audio_sr, 3)}s======")
                pad_len = int(audio_sr * pad_seconds)
                dat = np.concatenate([np.zeros([pad_len]), dat, np.zeros([pad_len])])
                raw_path = io.BytesIO()
                soundfile.write(raw_path, dat, audio_sr, format="wav")
                raw_path.seek(0)
                out_audio, out_sr, out_frame = self.infer(
                    spk, tran, raw_path, cluster_infer_ratio=cluster_infer_ratio,
                    auto_predict_f0=auto_predict_f0, noice_scale=noice_scale,
                    f0_predictor=f0_predictor, enhancer_adaptive_key=enhancer_adaptive_key,
                    cr_threshold=cr_threshold, k_step=k_step, frame=global_frame,
                    spk_mix=use_spk_mix, second_encoding=second_encoding,
                    loudness_envelope_adjustment=loudness_envelope_adjustment
                )
                global_frame += out_frame
                _audio = out_audio.cpu().numpy()
                pad_len = int(self.target_sample * pad_seconds)
                _audio = _audio[pad_len:-pad_len]
                _audio = pad_array(_audio, per_length)
                if lg_size != 0 and k != 0:
                    lg1 = audio[-(lg_size_r + lg_size_c_r): -lg_size_c_r] if lg_size_c_r != 0 else audio[-lg_size:]
                    lg2 = _audio[lg_size_c_l: lg_size_c_l + lg_size_r] if lg_size_c_r != 0 else _audio[0:lg_size]
                    lg_pre = lg1 * (1 - lg) + lg2 * lg
                    audio = audio[0: -(lg_size_r + lg_size_c_r)] if lg_size_c_r != 0 else audio[0:-lg_size]
                    audio.extend(lg_pre)
                    _audio = _audio[lg_size_c_l + lg_size_r:] if lg_size_c_r != 0 else _audio[lg_size:]
                audio.extend(list(_audio))
        return np.array(audio)


class RealTimeVC:
    def __init__(self):
        self.last_chunk = None
        self.last_o = None
        self.chunk_len = 16000
        self.pre_len = 3840

    def process(
        self, svc_model, speaker_id, f_pitch_change, input_wav_path,
        cluster_infer_ratio=0, auto_predict_f0=False, noice_scale=0.4, f0_filter=False
    ):

        audio, sr = torchaudio.load(input_wav_path)
        audio = audio.cpu().numpy()[0]
        temp_wav = io.BytesIO()
        if self.last_chunk is None:
            input_wav_path.seek(0)

            audio, sr = svc_model.infer(
                speaker_id, f_pitch_change, input_wav_path,
                cluster_infer_ratio=cluster_infer_ratio,
                auto_predict_f0=auto_predict_f0, noice_scale=noice_scale,
                f0_filter=f0_filter
            )

            audio = audio.cpu().numpy()
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            return audio[-self.chunk_len:]
        else:
            audio = np.concatenate([self.last_chunk, audio])
            soundfile.write(temp_wav, audio, sr, format="wav")
            temp_wav.seek(0)

            audio, sr = svc_model.infer(
                speaker_id, f_pitch_change, temp_wav,
                cluster_infer_ratio=cluster_infer_ratio,
                auto_predict_f0=auto_predict_f0, noice_scale=noice_scale,
                f0_filter=f0_filter
            )

            audio = audio.cpu().numpy()
            ret = maad.util.crossfade(self.last_o, audio, self.pre_len)
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            return ret[self.chunk_len: 2 * self.chunk_len]
