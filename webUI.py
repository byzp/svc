import gradio as gr
import os
from inference.infer_tool import Svc, format_wav
from spkmix import spk_mix_map
import soundfile
import tempfile

def save_uploaded(file_obj):
    os.makedirs("raw", exist_ok=True)
    with open(file_obj.name, "rb") as src:
        content = src.read()
    filename = os.path.basename(file_obj.name)
    save_path = os.path.join("raw", filename)
    with open(save_path, "wb") as dst:
        dst.write(content)
    return filename

def infer(
    model_path, config_path, uploaded_files, trans, spk_list,
    clip, auto_predict_f0, cluster_model_path, cluster_infer_ratio,
    linear_gradient, f0_predictor, enhance, shallow_diffusion, use_spk_mix,
    loudness_envelope_adjustment, feature_retrieval, diffusion_model_path,
    diffusion_config_path, k_step, second_encoding, only_diffusion,
    slice_db, device, noice_scale, pad_seconds, wav_format,
    linear_gradient_retain, enhancer_adaptive_key, f0_filter_threshold
):
    clean_names = [save_uploaded(f) for f in uploaded_files]
    trans = list(map(int, trans.strip().split())) if isinstance(trans, str) else trans
    spk_list = spk_list.strip().split()

    svc_model = Svc(model_path, config_path, device, cluster_model_path,
                    enhance, diffusion_model_path, diffusion_config_path,
                    shallow_diffusion, only_diffusion, use_spk_mix, feature_retrieval)

    os.makedirs("results", exist_ok=True)

    if len(spk_mix_map) <= 1:
        use_spk_mix = False
    if use_spk_mix:
        spk_list = [spk_mix_map]

    outputs = []
    for name, tran in zip(clean_names, trans):
        raw_audio_path = f"raw/{name}"
        """
        if not raw_audio_path.endswith(".wav"):
            raw_audio_path += ".wav"
        """
        format_wav(raw_audio_path)
        for spk in spk_list:
            audio = svc_model.slice_inference(
                raw_audio_path=raw_audio_path,
                spk=spk,
                tran=tran,
                slice_db=slice_db,
                cluster_infer_ratio=cluster_infer_ratio,
                auto_predict_f0=auto_predict_f0,
                noice_scale=noice_scale,
                pad_seconds=pad_seconds,
                clip_seconds=clip,
                lg_num=linear_gradient,
                lgr_num=linear_gradient_retain,
                f0_predictor=f0_predictor,
                enhancer_adaptive_key=enhancer_adaptive_key,
                cr_threshold=f0_filter_threshold,
                k_step=k_step,
                use_spk_mix=use_spk_mix,
                second_encoding=second_encoding,
                loudness_envelope_adjustment=loudness_envelope_adjustment
            )
            key = "auto" if auto_predict_f0 else f"{tran}key"
            cluster_tag = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
            mode = "sovits"
            if shallow_diffusion: mode = "sovdiff"
            if only_diffusion: mode = "diff"
            if use_spk_mix: spk = "spk_mix"

            out_path = f"results/{name}_{key}_{spk}{cluster_tag}_{mode}_{f0_predictor}.{wav_format}"
            soundfile.write(out_path, audio, svc_model.target_sample, format=wav_format)
            svc_model.clear_empty()
            outputs.append(out_path)
    return outputs

with gr.Blocks() as demo:
    gr.Markdown("# So-VITS-SVC WebUI")

    with gr.Row():
        model_path = gr.Textbox("logs/44k/G_0.pth", label="模型路径")
        config_path = gr.Textbox("logs/44k/config.json", label="配置路径")
        cluster_model_path = gr.Textbox("", label="聚类模型路径（可为空）")
        shallow_diffusion = gr.Checkbox(False, label="使用浅层扩散")
        diffusion_model_path = gr.Textbox("logs/44k/diffusion/model_0.pt", label="扩散模型路径")
        diffusion_config_path = gr.Textbox("logs/44k/diffusion/config.yaml", label="扩散配置")

    with gr.Row():
        uploaded_files = gr.File(file_count="multiple", label="上传音频文件")
        trans = gr.Textbox("0", label="音高变化（半音） 空格分隔，对应每个文件")
        spk_list = gr.Textbox("cabbage", label="目标说话人 空格分隔")

    with gr.Row():
        clip = gr.Number(0, label="强制切片/s（0为自动）")
        cluster_infer_ratio = gr.Slider(0, 1, step=0.01, value=0, label="聚类混合比例")
        linear_gradient = gr.Number(0, label="淡入长度/s")
        linear_gradient_retain = gr.Slider(0, 1, value=0.75, label="交叉保留比例")
        loudness_envelope_adjustment = gr.Slider(0, 1, value=1, label="响度包络融合比例")

    with gr.Row():
        f0_predictor = gr.Dropdown(["pm", "crepe", "dio", "harvest", "rmvpe", "fcpe"], value="rmvpe", label="F0预测器")
        wav_format = gr.Dropdown(["flac", "wav"], value="flac", label="输出格式")
        device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="推理设备")
        k_step = gr.Number(100, label="扩散步数")

    with gr.Row():
        slice_db = gr.Number(-40, label="切片分贝阈值")
        noice_scale = gr.Number(0.4, label="噪音等级")
        pad_seconds = gr.Number(0.5, label="pad秒数")
        enhancer_adaptive_key = gr.Number(0, label="增强器音域调整")
        f0_filter_threshold = gr.Slider(0, 1, value=0.05, label="F0过滤阈值")

    with gr.Row():
        auto_predict_f0 = gr.Checkbox(False, label="自动预测F0")
        enhance = gr.Checkbox(False, label="使用增强器")
        only_diffusion = gr.Checkbox(False, label="纯扩散模式")
        use_spk_mix = gr.Checkbox(False, label="角色融合")
        second_encoding = gr.Checkbox(False, label="二次编码")
        feature_retrieval = gr.Checkbox(False, label="特征检索")

    btn = gr.Button("开始推理")
    output = gr.Files(label="输出结果")

    btn.click(fn=infer, inputs=[
        model_path, config_path, uploaded_files, trans, spk_list,
        clip, auto_predict_f0, cluster_model_path, cluster_infer_ratio,
        linear_gradient, f0_predictor, enhance, shallow_diffusion, use_spk_mix,
        loudness_envelope_adjustment, feature_retrieval, diffusion_model_path,
        diffusion_config_path, k_step, second_encoding, only_diffusion,
        slice_db, device, noice_scale, pad_seconds, wav_format,
        linear_gradient_retain, enhancer_adaptive_key, f0_filter_threshold
    ], outputs=output)

if __name__ == '__main__':
    os.system("start http://127.0.0.1:7860")
    demo.launch()
