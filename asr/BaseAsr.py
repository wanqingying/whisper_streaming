#!/usr/bin/env python3
import sys
import numpy as np
from functools import lru_cache
import logging
import io
import soundfile as sf
import math
from typing import Optional, List, Tuple, Dict, Any

# 定义全局变量
SAMPLE_RATE = 16000 
# SAMPLE_RATE = 44100 

logger = logging.getLogger(__name__)

class ASRBase:
    sep: str = " "  # join transcribe words with this character (" " for whisper_timestamped,
                    # "" for faster-whisper because it emits the spaces when needed)

    def __init__(self, lan: str, modelsize: Optional[str] = None, cache_dir: Optional[str] = None, model_dir: Optional[str] = None, logfile: Any = sys.stderr) -> None:
        self.logfile = logfile
        self.transcribe_kargs: Dict[str, Any] = {}
        self.original_language: Optional[str] = None if lan == "auto" else lan
        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize: Optional[str], cache_dir: Optional[str], model_dir: Optional[str] = None) -> Any:
        raise NotImplementedError("must be implemented in the child class")

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> Any:
        raise NotImplementedError("must be implemented in the child class")

    def use_vad(self) -> None:
        raise NotImplementedError("must be implemented in the child class")

class WhisperTimestampedASR(ASRBase):
    """Uses whisper_timestamped library as the backend. Initially, we tested the code on this backend. It worked, but slower than faster-whisper.
    On the other hand, the installation for GPU could be easier.
    """

    sep: str = " "

    def load_model(self, modelsize: Optional[str] = None, cache_dir: Optional[str] = None, model_dir: Optional[str] = None) -> Any:
        import whisper
        import whisper_timestamped
        from whisper_timestamped import transcribe_timestamped
        self.transcribe_timestamped = transcribe_timestamped
        if model_dir is not None:
            logger.debug("ignoring model_dir, not implemented")
        return whisper.load_model(modelsize, download_root=cache_dir)

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> Dict[str, Any]:
        result = self.transcribe_timestamped(self.model,
                                             audio, language=self.original_language,
                                             initial_prompt=init_prompt, verbose=None,
                                             condition_on_previous_text=True, **self.transcribe_kargs)
        return result

    def ts_words(self, r: Dict[str, Any]) -> List[Tuple[float, float, str]]:
        o: List[Tuple[float, float, str]] = []
        for s in r["segments"]:
            for w in s["words"]:
                t = (w["start"], w["end"], w["text"])
                o.append(t)
        return o

    def segments_end_ts(self, res: Dict[str, Any]) -> List[float]:
        return [s["end"] for s in res["segments"]]

    def use_vad(self) -> None:
        self.transcribe_kargs["vad"] = True

    def set_translate_task(self) -> None:
        self.transcribe_kargs["task"] = "translate"

class FasterWhisperASR(ASRBase):
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version.
    """

    sep: str = ""

    def load_model(self, modelsize: Optional[str] = None, cache_dir: Optional[str] = None, model_dir: Optional[str] = None) -> Any:
        from faster_whisper import WhisperModel
        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        model = WhisperModel(model_size_or_path, device="cpu", compute_type="int8")
        return model

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> List[Any]:
        segments, info = self.model.transcribe(audio, language=self.original_language, initial_prompt=init_prompt, beam_size=5, word_timestamps=True, condition_on_previous_text=True, **self.transcribe_kargs)
        return list(segments)

    def ts_words(self, segments: List[Any]) -> List[Tuple[float, float, str]]:
        o: List[Tuple[float, float, str]] = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res: List[Any]) -> List[float]:
        return [s.end for s in res]

    def use_vad(self) -> None:
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self) -> None:
        self.transcribe_kargs["task"] = "translate"

class OpenaiApiASR(ASRBase):
    """Uses OpenAI's Whisper API for audio transcription."""

    def __init__(self, lan: Optional[str] = None, temperature: float = 0, logfile: Any = sys.stderr) -> None:
        self.logfile = logfile
        self.modelname: str = "whisper-1"
        self.original_language: Optional[str] = None if lan == "auto" else lan
        self.response_format: str = "verbose_json"
        self.temperature: float = temperature
        self.load_model()
        self.use_vad_opt: bool = False
        self.task: str = "transcribe"
        self.transcribed_seconds: int = 0

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        from openai import OpenAI
        self.client = OpenAI()

    def ts_words(self, segments: Any) -> List[Tuple[Optional[float], Optional[float], str]]:
        no_speech_segments: List[Tuple[Optional[float], Optional[float]]] = []
        if self.use_vad_opt:
            for segment in segments.segments:
                if segment["no_speech_prob"] > 0.8:
                    no_speech_segments.append((segment.get("start"), segment.get("end")))

        o: List[Tuple[Optional[float], Optional[float], str]] = []
        for word in segments.words:
            start = word.get("start")
            end = word.get("end")
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                continue
            o.append((start, end, word.get("word")))
        return o

    def segments_end_ts(self, res: Any) -> List[Optional[float]]:
        return [s["end"] for s in res.words]

    def transcribe(self, audio_data: np.ndarray, prompt: Optional[str] = None, *args: Any, **kwargs: Any) -> Any:
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format='WAV', subtype='PCM_16')
        buffer.seek(0)

        self.transcribed_seconds += math.ceil(len(audio_data) / 16000)

        params: Dict[str, Any] = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"]
        }
        if self.task != "translate" and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt

        proc = self.client.audio.translations if self.task == "translate" else self.client.audio.transcriptions
        transcript = proc.create(**params)
        logger.debug(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds")

        return transcript

    def use_vad(self) -> None:
        self.use_vad_opt = True

    def set_translate_task(self) -> None:
        self.task = "translate"