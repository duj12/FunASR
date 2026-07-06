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
    """降噪模块，支持多个模型后端（ModelScope / HuggingFace / 本地路径）。

    可通过 ``model_backend`` 选择加载方式：
    - ``"modelscope"`` （默认）: 通过 ModelScope pipeline 加载
    - ``"huggingface"``: 从 HuggingFace Hub 或本地路径加载 PyTorch checkpoint
    - ``"auto"``: 先尝试 modelscope，失败后自动回退到 huggingface

    不同模型的输出 key 通过 ``output_key`` 配置：
    - ``"wav_l2"`` （默认）: ZipEnhancer 系列
    - ``"output_pcm"``: 部分 ANS pipeline 的默认输出
    - ``"wav"`` / ``"output"``: 其他常见命名
    """

    def __init__(self, denoise_prob=0.5,
                 ans_model="iic/speech_zipenhancer_ans_multiloss_16k_base",
                 model_backend="modelscope",
                 output_key="wav_l2"):
        self.denoise_prob = denoise_prob
        self.ans_model = ans_model
        self.model_backend = model_backend
        self.output_key = output_key
        self._pipeline = None
        self._model = None
        self._device = None
        self._denoise_count = 0
        self._skip_count = 0
        logging.info(f"DenoiseModule configured: ans_model={ans_model}, "
                     f"backend={model_backend}, output_key={output_key}, "
                     f"denoise_prob={denoise_prob}")

    def _ensure_pipeline(self):
        if self._model is not None:
            return

        backends_to_try = ["modelscope", "huggingface"] if self.model_backend == "auto" else [self.model_backend]

        last_exc = None
        for backend in backends_to_try:
            try:
                if backend == "modelscope":
                    self._load_modelscope()
                elif backend == "huggingface":
                    self._load_huggingface()
                else:
                    raise ValueError(f"Unknown model_backend: {backend}")
                return
            except Exception as e:
                last_exc = e
                logging.warning(f"Denoise backend '{backend}' failed for "
                                f"'{self.ans_model}': {e}")

        raise RuntimeError(
            f"All denoise backends failed for model '{self.ans_model}'"
        ) from last_exc

    def _load_modelscope(self):
        """通过 ModelScope pipeline 加载模型。"""
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        self._pipeline = pipeline(
            Tasks.acoustic_noise_suppression,
            model=self.ans_model,
            device=device)
        self._pipeline.prepare_model()
        self._model = self._pipeline.model
        self._model.eval()
        self._device = next(self._model.parameters()).device
        logging.info(f"DenoiseModule (modelscope) loaded on {device}, "
                     f"model device={self._device}")

    def _load_huggingface(self):
        """从 HuggingFace Hub 或本地路径加载 PyTorch 模型。

        支持格式（按优先级）:
        1. TorchScript 模型  (model.ts / model.pt / model_jit.pt)
        2. 常规 PyTorch checkpoint (pytorch_model.bin / model.pth)
        """
        import os as _os

        # --- resolve model path ---
        if _os.path.isdir(self.ans_model):
            model_dir = self.ans_model
        else:
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(self.ans_model)

        local_rank = int(_os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        import torch as _torch

        # try TorchScript first
        for ts_name in ("model.ts", "model_jit.pt", "model.pt"):
            ts_path = _os.path.join(model_dir, ts_name)
            if _os.path.exists(ts_path):
                self._model = _torch.jit.load(ts_path, map_location="cpu")
                self._model.eval()
                self._model.to(device)
                self._device = next(self._model.parameters()).device
                logging.info(f"DenoiseModule (huggingface) loaded TorchScript "
                             f"from {ts_path}, device={self._device}")
                return

        # fallback: try regular checkpoint
        for ckpt_name in ("pytorch_model.bin", "model.pth", "generator.pt"):
            ckpt_path = _os.path.join(model_dir, ckpt_name)
            if _os.path.exists(ckpt_path):
                state = _torch.load(ckpt_path, map_location="cpu", weights_only=True)
                # Try to infer architecture from config
                cfg_path = _os.path.join(model_dir, "configuration.json")
                if _os.path.exists(cfg_path):
                    import json
                    with open(cfg_path) as f:
                        cfg = json.load(f)
                    model_cls = self._infer_model_class(cfg)
                    if model_cls is not None:
                        self._model = model_cls(**cfg.get("model_conf", {}))
                        if isinstance(state, dict) and "state_dict" in state:
                            self._model.load_state_dict(state["state_dict"])
                        elif isinstance(state, dict) and "generator" in state:
                            self._model.load_state_dict(state["generator"])
                        else:
                            self._model.load_state_dict(state)
                    else:
                        raise ValueError(
                            f"Cannot infer model class from config: {cfg_path}")
                else:
                    raise ValueError(
                        f"No configuration.json found in {model_dir}, "
                        f"cannot determine model architecture")

                self._model.eval()
                self._model.to(device)
                self._device = next(self._model.parameters()).device
                logging.info(f"DenoiseModule (huggingface) loaded checkpoint "
                             f"from {ckpt_path}, device={self._device}")
                return

        raise FileNotFoundError(
            f"No supported model file found in {model_dir}. "
            f"Looked for: model.ts, model_jit.pt, model.pt, "
            f"pytorch_model.bin, model.pth, generator.pt"
        )

    @staticmethod
    def _infer_model_class(cfg):
        """根据 configuration.json 推断模型类（仅 modelscope 注册的模型）。"""
        try:
            from modelscope.utils.constant import Tasks
            from modelscope.models.builder import MODELS
            task = cfg.get("task", Tasks.acoustic_noise_suppression)
            model_name = cfg.get("model", {}).get("model_name") or cfg.get("name")
            if model_name:
                return MODELS.get(task, model_name)
        except Exception:
            pass
        return None

    @staticmethod
    def _audio_norm(x):
        """audio_norm from modelscope.utils.audio.audio_utils — match training-time normalization."""
        rms = (x ** 2).mean() ** 0.5
        scalar = 10 ** (-25 / 20) / (rms + 1e-8)
        x = x * scalar
        pow_x = x ** 2
        avg_pow_x = pow_x.mean()
        rmsx = pow_x[pow_x > avg_pow_x].mean() ** 0.5
        scalarx = 10 ** (-25 / 20) / (rmsx + 1e-8)
        x = x * scalarx
        return x

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
        """从 pipeline 输出中提取音频波形（仅 modelscope backend 的 fallback 路径使用）。"""
        if isinstance(result, dict):
            for key in (self.output_key, 'output_pcm', 'output', 'wav'):
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
                for key in (self.output_key, 'output_pcm', 'output', 'wav'):
                    if key in item:
                        return self._to_numpy_audio(item[key])
            return self._to_numpy_audio(item)
        raise ValueError(f"Unexpected pipeline result type: {type(result)}")

    def denoise_audio(self, audio_path):
        """逐条通过 pipeline 降噪（仅 modelscope backend 的 fallback）。"""
        self._ensure_pipeline()
        if self._pipeline is None:
            raise RuntimeError("denoise_audio requires pipeline (modelscope backend)")
        prev_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = self._pipeline(audio_path)
        finally:
            sys.stdout = prev_stdout
        return self._extract_result(result)

    def _denoise_audio_sub_batch(self, audio_paths):
        """Process a sub-batch of audio through the model (single forward).

        Args:
            audio_paths: list of audio file paths (str), small enough for one forward.
        Returns:
            list of float32 numpy arrays (denoised waveforms).
        """
        sample_rate = 16000
        window = sample_rate * 2  # 2s, matching ANSZipEnhancerPipeline

        # Phase 1: load all audios
        waveforms = []
        orig_lengths = []
        for path in audio_paths:
            wav, sr = soundfile.read(path)
            if len(wav.shape) > 1:
                wav = wav[:, 0]  # mono
            if sr != sample_rate:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
            wav = wav.astype(np.float32)
            orig_lengths.append(len(wav))
            waveforms.append(torch.from_numpy(wav))

        # Phase 2: normalize (matching pipeline preprocess)
        for i in range(len(waveforms)):
            waveforms[i] = torch.from_numpy(self._audio_norm(waveforms[i].numpy()))

        # Phase 3: pad to sub-batch max length (at least window)
        max_len = max(max(orig_lengths), window)
        batch_list = []
        for wav in waveforms:
            pad_len = max_len - len(wav)
            if pad_len > 0:
                wav = torch.nn.functional.pad(wav, (0, pad_len))
            batch_list.append(wav.unsqueeze(0))  # [1, T]
        batch_tensor = torch.cat(batch_list, dim=0)  # [N, max_len]

        # Phase 4: single model forward
        with torch.no_grad():
            out = self._model(dict(noisy=batch_tensor.to(self._device)))
            # 尝试多个可能的输出 key
            if isinstance(out, dict):
                for k in (self.output_key, 'wav_l2', 'output_pcm', 'wav', 'output'):
                    if k in out:
                        output = out[k]
                        break
                else:
                    output = list(out.values())[0]
            else:
                output = out
            # output shape: [N, max_len]

        # Phase 5: trim to original lengths
        results = []
        for i, length in enumerate(orig_lengths):
            enhanced = output[i, :length].cpu().numpy().astype(np.float32)
            results.append(enhanced)

        return results

    def denoise_audio_batch(self, audio_paths, max_batch_size=8):
        """Batch inference with automatic sub-batching to limit GPU memory.

        N audios are split into chunks of ``max_batch_size``, each chunk
        processed as one model forward.  This prevents OOM when the training
        batch (from dynamic token batching) is very large.

        Args:
            audio_paths: list of audio file paths (str).
            max_batch_size: max number of audios per single forward.
        Returns:
            list of float32 numpy arrays (denoised waveforms).
        """
        self._ensure_pipeline()

        results = []
        for start in range(0, len(audio_paths), max_batch_size):
            sub_paths = audio_paths[start:start + max_batch_size]
            sub_results = self._denoise_audio_sub_batch(sub_paths)
            results.extend(sub_results)
        return results

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

        batch_max_len = speech.shape[1]
        num_segments = len(sources)

        # === Phase 1: collect paths & indices of samples to denoise ===
        indices = []
        paths = []
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

            indices.append(i)
            paths.append(audio_path)

        if not paths:
            return batch

        # === Phase 2: batched model inference (N audios → 1 forward) ===
        try:
            enhanced_list = self.denoise_audio_batch(paths)
        except Exception as e:
            logging.warning(f"Batch denoise failed, falling back to per-sample: {e}")
            enhanced_list = []
            fallback_paths = []
            fallback_indices = []
            for i, path in zip(indices, paths):
                try:
                    enhanced = self.denoise_audio(path)
                    enhanced_list.append(enhanced)
                    fallback_indices.append(i)
                    fallback_paths.append(path)
                except Exception as e2:
                    logging.warning(f"Denoise fallback failed for {path}: {e2}")
                    self._skip_count += 1
            indices = fallback_indices
            paths = fallback_paths

        # === Phase 3: per-sample fbank extraction + copy back ===
        for batch_idx, path, enhanced in zip(indices, paths, enhanced_list):
            try:
                enhanced_tensor = torch.from_numpy(enhanced.astype(np.float32))
                new_speech, new_speech_len = extract_fbank(
                    enhanced_tensor, data_type="sound", frontend=frontend, is_final=True
                )

                orig_valid_len = lengths[batch_idx].item()
                new_valid_len = new_speech_len[0].item()

                if self._denoise_count == 0:
                    logging.info(f"DenoiseModule length check: sample {batch_idx}, "
                                f"orig_valid_len={orig_valid_len}, new_valid_len={new_valid_len}, "
                                f"batch_max_len={batch_max_len}, speech_shape={speech.shape}")

                # 根据有效长度进行调整（截断或填充到有效长度）
                if new_valid_len > orig_valid_len:
                    new_speech_valid = new_speech[0, :orig_valid_len, :]
                    new_valid_len = orig_valid_len
                elif new_valid_len < orig_valid_len:
                    new_speech_valid = new_speech[0, :, :]
                    pad_len = orig_valid_len - new_valid_len
                    pad = torch.zeros(pad_len, new_speech.shape[-1], dtype=new_speech.dtype)
                    new_speech_valid = torch.cat([new_speech_valid, pad], dim=0)
                    new_valid_len = orig_valid_len
                else:
                    new_speech_valid = new_speech[0, :, :]

                # 再填充到batch最大长度
                if new_valid_len < batch_max_len:
                    pad_len = batch_max_len - new_valid_len
                    pad = torch.zeros(pad_len, new_speech_valid.shape[-1],
                                      dtype=new_speech_valid.dtype)
                    new_speech_padded = torch.cat([new_speech_valid, pad], dim=0)
                else:
                    new_speech_padded = new_speech_valid

                speech[batch_idx].copy_(new_speech_padded.to(speech.device))
                self._denoise_count += 1

                if self._denoise_count == 1:
                    debug_dir = os.path.dirname(os.path.abspath(__file__))
                    soundfile.write(os.path.join(debug_dir, "debug_denoise_orig.wav"),
                                    enhanced, 16000)
                    try:
                        orig_wav, orig_sr = soundfile.read(paths[0])
                        soundfile.write(os.path.join(debug_dir, "debug_denoise_input.wav"),
                                        orig_wav, orig_sr)
                    except Exception:
                        pass
                    logging.info(f"DenoiseModule: first audio denoised via batch, shape={enhanced.shape}")

                if self._denoise_count % 1000 == 0:
                    logging.info(f"DenoiseModule stats: denoised={self._denoise_count}, "
                                f"skipped={self._skip_count}")
            except Exception as e:
                logging.warning(f"Denoise post-process failed for {path}: {e}")
                self._skip_count += 1

        batch["speech"] = speech
        return batch
