import os
import json
import torch
import logging
import concurrent.futures
import librosa, soundfile
import numpy as np
from scipy import signal
import torch.distributed as dist
from typing import Collection
import torch
import torchaudio
from torch import nn
import random
import re
from funasr.tokenizer.cleaner import TextCleaner
from funasr.register import tables
 

@tables.register("preprocessor_classes", "SpeechPreprocessSpeedPerturb")
class SpeechPreprocessSpeedPerturb(nn.Module):
    def __init__(self, speed_perturb: list = None, **kwargs):
        super().__init__()
        self.speed_perturb = speed_perturb

    def forward(self, waveform, fs, **kwargs):
        if self.speed_perturb is None:
            return waveform
        speed = random.choice(self.speed_perturb)
        if speed != 1.0:
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform)
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform.view(1, -1), fs, [["speed", str(speed)], ["rate", str(fs)]]
            )
            waveform = waveform.view(-1)

        return waveform

def get_random_chunk(data, chunk_len):
    """ Get random chunk

        Args:
            data: torch.Tensor (random len)
            chunk_len: chunk length

        Returns:
            torch.Tensor (exactly chunk_len)
    """
    data_len = len(data)
    data_shape = data.shape
    # random chunk
    if data_len >= chunk_len:
        chunk_start = random.randint(0, data_len - chunk_len)
        data = data[chunk_start:chunk_start + chunk_len]
        # re-clone the data to avoid memory leakage
        if type(data) == torch.Tensor:
            data = data.clone()
        else:  # np.array
            data = data.copy()
    else:
        # padding
        repeat_factor = chunk_len // data_len + 1
        repeat_shape = repeat_factor if len(data_shape) == 1 else (
            repeat_factor, 1)
        if type(data) == torch.Tensor:
            data = data.repeat(repeat_shape)
        else:  # np.array
            data = np.tile(data, repeat_shape)
        data = data[:chunk_len]

    return data

@tables.register("preprocessor_classes", "SpeechPreprocessAddNoiseReverb")
class SpeechPreprocessAddNoiseReverb(nn.Module):

    def __init__(self, noise_path: str = None, reverb_path: str = None, 
            noise_prob: float=0.8, reverb_prob: float=0.8, **kwargs):
        super().__init__()
        self.noise_prob = noise_prob
        self.reverb_prob = reverb_prob 
        with open(noise_path ) as f:
            self.noise_list = f.readlines()
            self.noise_list = [x.strip().split()[1] for x in self.noise_list]
        with open(reverb_path) as f:
            self.reverb_list = f.readlines()
            self.reverb_list = [x.strip().split()[1] for x in self.reverb_list]
        

    def forward(self, audio, fs, **kwargs):
        audio = audio.numpy()   # tensor -> numpy
        save_audio=False
        if save_audio:
            soundfile.write("./debug_orig.wav", audio, fs)
            
        # calculate the audio energy first, because we may add reverb
        audio_db = 10 * np.log10(np.mean(audio**2) + 1e-4)
        if self.reverb_prob > random.random():  
            # add reverberation
            audio_len = audio.shape[0]
            rir_path = random.choice(self.reverb_list)  
            rir_audio, rir_sr = librosa.load(rir_path, sr=fs, mono=True)
            rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2))
            audio = signal.convolve(audio, rir_audio, mode='full')[:audio_len]
        
        if self.noise_prob > random.random():
            # add additive noise
            audio_len = audio.shape[0]

            noise_path = random.choice(self.noise_list)
            key = os.path.basename(noise_path).split('.')[0]
            if key.startswith('noise'):
                snr_range = [0, 15]
            elif key.startswith('speech'):
                snr_range = [10, 30]
            elif key.startswith('music'):
                snr_range = [5, 15]
            else:
                snr_range = [0, 15]
            noise_audio, noise_sr = librosa.load(noise_path, sr=fs, mono=True)
            noise_audio = get_random_chunk(noise_audio, audio_len)
            noise_snr = random.uniform(snr_range[0], snr_range[1])
            noise_db = 10 * np.log10(np.mean(noise_audio**2) + 1e-4)
            noise_audio = np.sqrt(10**(
                (audio_db - noise_db - noise_snr) / 10)) * noise_audio
            out_audio = audio + noise_audio

            # normalize into [-1, 1]
            out_audio = out_audio / (np.max(np.abs(out_audio)) + 1e-4)
            audio = out_audio
        
        if save_audio:
            soundfile.write("./debug_noise.wav", audio, fs)

        return torch.from_numpy(audio)


@tables.register("preprocessor_classes", "SpeechPreprocessDenoise")
class SpeechPreprocessDenoise(nn.Module):

    def __init__(self, denoise_prob: float = 0.5, ans_model: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
                 onnx_providers: list = None, **kwargs):
        super().__init__()
        self.denoise_prob = denoise_prob
        import onnxruntime
        from modelscope.utils.file_utils import get_modelscope_cache_dir

        cache_dir = get_modelscope_cache_dir()
        onnx_path = os.path.join(cache_dir, f"hub/{ans_model}/onnx_model.onnx")
        if not os.path.exists(onnx_path):
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            pipeline(Tasks.acoustic_noise_suppression, model=ans_model)

        providers = onnx_providers or ["CPUExecutionProvider"]
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        from modelscope.models.audio.ans.zipenhancer import mag_pha_stft, mag_pha_istft
        self.mag_pha_stft = mag_pha_stft
        self.mag_pha_istft = mag_pha_istft
        self._denoise_count = 0
        self._skip_count = 0
        logging.info(f"SpeechPreprocessDenoise loaded: ans_model={ans_model}, onnx_path={onnx_path}, "
                     f"denoise_prob={denoise_prob}, providers={providers}") 

    def _denoise(self, wav_np, fs):
        from modelscope.utils.audio.audio_utils import audio_norm
        wav = audio_norm(wav_np).astype(np.float32)
        noisy_wav = torch.from_numpy(np.reshape(wav, [1, wav.shape[0]]))
        n_fft, hop_size, win_size = 400, 100, 400

        norm_factor = torch.sqrt(noisy_wav.shape[1] / torch.sum(noisy_wav ** 2.0))
        noisy_audio = noisy_wav * norm_factor
        noisy_amp, noisy_pha, _ = self.mag_pha_stft(
            noisy_audio, n_fft, hop_size, win_size, compress_factor=0.3, center=True)

        def to_numpy(t):
            return t.detach().cpu().numpy() if t.requires_grad else t.cpu().numpy()

        ort_inputs = {
            self.onnx_session.get_inputs()[0].name: to_numpy(noisy_amp),
            self.onnx_session.get_inputs()[1].name: to_numpy(noisy_pha),
        }
        ort_outs = self.onnx_session.run(None, ort_inputs)

        amp_g = torch.from_numpy(ort_outs[0])
        pha_g = torch.from_numpy(ort_outs[1])
        enhanced = self.mag_pha_istft(
            amp_g, pha_g, n_fft, hop_size, win_size, compress_factor=0.3, center=True)
        enhanced = enhanced / norm_factor
        return to_numpy(enhanced[0])

    def forward(self, audio, fs, **kwargs):
        if self.denoise_prob <= 0 or random.random() >= self.denoise_prob:
            self._skip_count += 1
            if self._skip_count % 1000 == 0:
                logging.info(f"SpeechPreprocessDenoise stats: denoised={self._denoise_count}, skipped={self._skip_count}")
            return audio
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else np.array(audio)
        try:
            enhanced = self._denoise(audio_np, fs)
            self._denoise_count += 1
            if self._denoise_count == 1:
                logging.info(f"SpeechPreprocessDenoise: first audio denoised successfully, shape={enhanced.shape}")
            if self._denoise_count % 1000 == 0:
                logging.info(f"SpeechPreprocessDenoise stats: denoised={self._denoise_count}, skipped={self._skip_count}")
            return torch.from_numpy(enhanced.astype(np.float32))
        except Exception as e:
            logging.warning(f"Denoise failed, using original audio: {e}")
            return audio if isinstance(audio, torch.Tensor) else torch.from_numpy(audio)


@tables.register("preprocessor_classes", "TextPreprocessSegDict")
class TextPreprocessSegDict(nn.Module):
    def __init__(
        self,
        seg_dict: str = None,
        text_cleaner: Collection[str] = None,
        split_with_space: bool = False,
        **kwargs
    ):
        super().__init__()

        self.text_cleaner = TextCleaner(text_cleaner)

    def forward(self, text, **kwargs):
        text = self.text_cleaner(text)

        return text
