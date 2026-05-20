import os
import json
import torch
import logging
import threading
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
from funasr.utils.load_utils import load_audio_text_image_video
 

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
    _init_lock = threading.Lock()

    def __init__(self, denoise_prob: float = 0.5,
                 ans_model: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
                 denoise_gpu: int = None, **kwargs):
        super().__init__()
        self.denoise_prob = denoise_prob
        self.ans_model = ans_model
        self.denoise_gpu = denoise_gpu
        self._ans_pipeline = None
        self._denoise_count = 0
        self._skip_count = 0
        logging.info(f"SpeechPreprocessDenoise configured: ans_model={ans_model}, "
                     f"denoise_prob={denoise_prob}, denoise_gpu={denoise_gpu}")

    def _ensure_pipeline(self):
        if self._ans_pipeline is not None:
            return
        with self._init_lock:
            if self._ans_pipeline is not None:
                return
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            if self.denoise_gpu is not None and self.denoise_gpu >= 0:
                device = f"cuda:{self.denoise_gpu}"
            else:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                device = f"cuda:{local_rank}"
            self._ans_pipeline = pipeline(
                Tasks.acoustic_noise_suppression,
                model=self.ans_model,
                device=device)
            logging.info(f"SpeechPreprocessDenoise pipeline loaded on device={device}, local_rank={local_rank}")

    def _to_numpy_audio(self, data):
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        if isinstance(data, bytes):
            pcm = np.frombuffer(data, dtype=np.int16)
            return pcm.astype(np.float32) / 32768.0
        if isinstance(data, str):
            wav, _ = soundfile.read(data)
            return wav.astype(np.float32)
        raise ValueError(f"Cannot convert {type(data)} to numpy audio")

    def _extract_result(self, result):
        if isinstance(result, dict):
            for key in ('output_pcm', 'output', 'wav'):
                if key in result:
                    return self._to_numpy_audio(result[key])
            for v in result.values():
                if isinstance(v, (np.ndarray, bytes)):
                    return self._to_numpy_audio(v)
            raise ValueError(f"Cannot find audio in pipeline result: {list(result.keys())}")
        if isinstance(result, (np.ndarray, bytes)):
            return self._to_numpy_audio(result)
        if isinstance(result, (list, tuple)) and len(result) > 0:
            item = result[0]
            if isinstance(item, dict):
                for key in ('output_pcm', 'output', 'wav'):
                    if key in item:
                        return self._to_numpy_audio(item[key])
            return self._to_numpy_audio(item)
        raise ValueError(f"Unexpected pipeline result type: {type(result)}")

    def forward(self, audio, fs, source=None, **kwargs):
        do_denoise = self.denoise_prob > 0 and random.random() < self.denoise_prob
        if not do_denoise:
            self._skip_count += 1
            if self._skip_count % 1000 == 0:
                logging.info(f"SpeechPreprocessDenoise stats: denoised={self._denoise_count}, skipped={self._skip_count}")
            if audio is not None:
                return audio
            if source is not None:
                return load_audio_text_image_video(source, fs=fs)
            raise ValueError("Either audio or source must be provided")

        try:
            self._ensure_pipeline()
            import sys
            import io as _io
            _prev_stdout = sys.stdout
            sys.stdout = _io.StringIO()
            try:
                if source is not None:
                    result = self._ans_pipeline(source)
                else:
                    import tempfile
                    if not hasattr(self, '_tmp_wav') or self._tmp_wav is None:
                        fd, self._tmp_wav = tempfile.mkstemp(suffix='.wav')
                        os.close(fd)
                    audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else np.array(audio)
                    soundfile.write(self._tmp_wav, audio_np.astype(np.float32), fs)
                    result = self._ans_pipeline(self._tmp_wav)
            finally:
                sys.stdout = _prev_stdout

            enhanced = self._extract_result(result)
            self._denoise_count += 1
            if self._denoise_count == 1:
                debug_dir = os.path.dirname(os.path.abspath(__file__))
                orig_path = os.path.join(debug_dir, "debug_denoise_orig.wav")
                enh_path = os.path.join(debug_dir, "debug_denoise_enhanced.wav")
                if source is not None:
                    orig_wav, orig_sr = soundfile.read(source)
                    soundfile.write(orig_path, orig_wav, orig_sr)
                elif audio is not None:
                    audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else np.array(audio)
                    soundfile.write(orig_path, audio_np.astype(np.float32), fs)
                soundfile.write(enh_path, enhanced, fs)
                logging.info(f"SpeechPreprocessDenoise: debug audio saved to {orig_path} and {enh_path}, shape={enhanced.shape}")
            if self._denoise_count % 1000 == 0:
                logging.info(f"SpeechPreprocessDenoise stats: denoised={self._denoise_count}, skipped={self._skip_count}")
            return torch.from_numpy(enhanced.astype(np.float32))
        except Exception as e:
            logging.warning(f"Denoise failed, using original audio: {e}")
            if audio is not None:
                return audio
            if source is not None:
                return load_audio_text_image_video(source, fs=fs)
            raise


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
