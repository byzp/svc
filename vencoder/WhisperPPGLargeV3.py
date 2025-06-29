import torch
import torchaudio

from vencoder.encoder import SpeechEncoder
from vencoder.whisper.audio import log_mel_spectrogram, pad_or_trim
from vencoder.whisper.model import ModelDimensions, Whisper


class WhisperPPGLargeV3(SpeechEncoder):
    def __init__(self, vec_path="pretrain/large-v3.pt", device=None):
        super().__init__()
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)

        checkpoint = torch.load(vec_path, map_location=self.dev)
        dims = ModelDimensions(**checkpoint["dims"])  # Make sure n_mels=128
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.hidden_dim = dims
        if self.dev.type=="cuda":
            print("WhisperPPGLargeV3.__init__ half()")
            self.model = model.half().to(self.dev)
        else:
            self.model = model.to(self.dev)
    

    def encoder(self, wav):
        audio = wav
        audln = audio.shape[0]
        ppgln = audln // 320
        audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio, 128).to(self.dev).float()  # uses n_mels=128 internally
        with torch.no_grad(), torch.amp.autocast("cuda",enabled=True):
            # FP16，自动混合精度
            ppg = self.model.encoder(mel.unsqueeze(0)).squeeze()
        ppg = ppg.data.cpu().float().numpy()
        ppg = torch.FloatTensor(ppg[:ppgln]).to(self.dev)
        return ppg[None, :, :].transpose(1, 2).float()
    
    