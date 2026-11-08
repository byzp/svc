class SpeechEncoder:
    """16 kHz waveform -> [batch, channels, frames] content features."""

    def encoder(self, wav):
        raise NotImplementedError
