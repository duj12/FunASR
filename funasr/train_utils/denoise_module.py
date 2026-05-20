import os
import sys
import io
import random
import logging
import numpy as np
import torch
import soundfile
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


class DenoiseModule:
    def __init__(self, denoise_prob=0.5,
                 ans_model="iic/speech_zipenhancer_ans_multiloss_16k_base"):
        self.denoise_prob = denoise_prob
        self.ans_model = ans_model
        self._pipeline = None
        self._denoise_count = 0
        self._skip_count = 0
        logging.info(f"DenoiseModule configured: ans_model={ans_model}, denoise_prob={denoise_prob}")

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        self._pipeline = pipeline(
            Tasks.acoustic_noise_suppression,
            model=self.ans_model,
            device=device)
        logging.info(f"DenoiseModule pipeline loaded on {device}")

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

    def denoise_audio(self, audio_path):
        self._ensure_pipeline()
        prev_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = self._pipeline(audio_path)
        finally:
            sys.stdout = prev_stdout
        return self._extract_result(result)

    def denoise_batch(self, batch, frontend):
        sources = batch.get("sources", None)
        if self._denoise_count == 0 and self._skip_count == 0:
            logging.info(f"DenoiseModule.denoise_batch called: "
                         f"sources={'present' if sources is not None else 'MISSING'}, "
                         f"sources_type={type(sources).__name__}, "
                         f"frontend={'present' if frontend is not None else 'MISSING'}, "
                         f"batch_keys={list(batch.keys())}")
            if sources is not None:
                logging.info(f"DenoiseModule: sources len={len(sources)}, "
                             f"first_item_type={type(sources[0]).__name__ if len(sources) > 0 else 'empty'}, "
                             f"first_item={repr(sources[0])[:100] if len(sources) > 0 else 'N/A'}")
        if sources is None:
            return batch

        speech = batch["speech"]
        speech_lengths = batch["speech_lengths"]
        if speech_lengths.dim() > 1:
            lengths = speech_lengths[:, 0]
        else:
            lengths = speech_lengths

        batch_max_len = speech.shape[1]  # batch中的最大长度
        num_segments = len(sources)
        
        for i in range(num_segments):
            if random.random() >= self.denoise_prob:
                self._skip_count += 1
                continue

            audio_path = sources[i]
            # Handle nested list format from collator: [[p1], [p2], ...]
            if isinstance(audio_path, (list, tuple)):
                audio_path = audio_path[0] if len(audio_path) > 0 else None
            if not isinstance(audio_path, str) or not audio_path:
                self._skip_count += 1
                continue

            try:
                enhanced = self.denoise_audio(audio_path)
                enhanced_tensor = torch.from_numpy(enhanced.astype(np.float32))
                new_speech, new_speech_len = extract_fbank(
                    enhanced_tensor, data_type="sound", frontend=frontend, is_final=True
                )

                orig_valid_len = lengths[i].item()  # 原始音频的有效长度
                new_valid_len = new_speech_len[0].item()  # 降噪后的有效长度
                
                # 日志记录长度变化
                if self._denoise_count == 0:
                    logging.info(f"DenoiseModule length check: sample {i}, "
                                f"orig_valid_len={orig_valid_len}, new_valid_len={new_valid_len}, "
                                f"batch_max_len={batch_max_len}, speech_shape={speech.shape}")

                # 根据有效长度进行调整（截断或填充到有效长度）
                if new_valid_len > orig_valid_len:
                    # 降噪后更长，截断到原始有效长度
                    new_speech_valid = new_speech[0, :orig_valid_len, :]
                    new_valid_len = orig_valid_len
                elif new_valid_len < orig_valid_len:
                    # 降噪后更短，填充到原始有效长度
                    new_speech_valid = new_speech[0, :, :]
                    pad_len = orig_valid_len - new_valid_len
                    pad = torch.zeros(pad_len, new_speech.shape[-1], dtype=new_speech.dtype)
                    new_speech_valid = torch.cat([new_speech_valid, pad], dim=0)
                    new_valid_len = orig_valid_len
                else:
                    # 长度恰好相同
                    new_speech_valid = new_speech[0, :, :]

                # 再填充到batch最大长度
                if new_valid_len < batch_max_len:
                    pad_len = batch_max_len - new_valid_len
                    pad = torch.zeros(pad_len, new_speech_valid.shape[-1], 
                                      dtype=new_speech_valid.dtype)
                    new_speech_padded = torch.cat([new_speech_valid, pad], dim=0)
                else:
                    new_speech_padded = new_speech_valid

                # 更新speech（使用copy_避免形状不匹配）
                speech[i].copy_(new_speech_padded.to(speech.device))
                self._denoise_count += 1

                if self._denoise_count == 1:
                    debug_dir = os.path.dirname(os.path.abspath(__file__))
                    soundfile.write(os.path.join(debug_dir, "debug_denoise_orig.wav"),
                                    enhanced, 16000)
                    # also save original for comparison
                    try:
                        orig_wav, orig_sr = soundfile.read(audio_path)
                        soundfile.write(os.path.join(debug_dir, "debug_denoise_input.wav"),
                                        orig_wav, orig_sr)
                    except Exception:
                        pass
                    logging.info(f"DenoiseModule: first audio denoised, shape={enhanced.shape}")

                if self._denoise_count % 1000 == 0:
                    logging.info(f"DenoiseModule stats: denoised={self._denoise_count}, "
                                f"skipped={self._skip_count}")
            except Exception as e:
                logging.warning(f"Denoise failed for {audio_path}: {e}")
                self._skip_count += 1

        batch["speech"] = speech
        return batch
