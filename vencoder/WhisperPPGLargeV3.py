import torch
import torchaudio

from vencoder.encoder import SpeechEncoder
from vencoder.whisper.audio import log_mel_spectrogram, pad_or_trim
from vencoder.whisper.model import ModelDimensions
from vencoder.whisper.model import AudioEncoder


class WhisperPPGLargeV3(SpeechEncoder):
    def __init__(self, vec_path="pretrain/large-v3.pt", device=None):
        super().__init__()

        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)

        checkpoint = torch.load(vec_path, map_location="cpu")
        dims = ModelDimensions(**checkpoint["dims"])  # n_mels=128
        self.model = AudioEncoder(
            n_mels=dims.n_mels,
            n_ctx=dims.n_audio_ctx,
            n_state=dims.n_audio_state,
            n_head=dims.n_audio_head,
            n_layer=dims.n_audio_layer,
        )
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
            if k.startswith("encoder.")
        }
        self.model.load_state_dict(encoder_state, strict=True)
        self.model.eval()
        self.model.to(self.dev)
        self.hidden_dim = dims.n_audio_state
        if self.dev.type == "cuda":
            self.model = self.model.half()


    def encoder(self, wav):
        audio = wav
        audln = audio.shape[0]
        ppgln = audln // 320
        audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio, 128).to(self.dev).float()  # uses n_mels=128 internally
        with torch.no_grad(), torch.amp.autocast("cuda",enabled=True):
            # FP16，自动混合精度
            ppg = self.model(mel.unsqueeze(0)).squeeze()
        ppg = ppg.data.cpu().float().numpy()
        ppg = torch.FloatTensor(ppg[:ppgln]).to(self.dev)
        return ppg[None, :, :].transpose(1, 2).float()
