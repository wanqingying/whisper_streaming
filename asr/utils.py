#!/usr/bin/env python3
import sys
import numpy as np
import librosa
from functools import lru_cache
import time
import logging
from argparse import ArgumentParser, Namespace
from typing import Optional, List, Tuple, Any
from asr.OnlineAsr import OnlineASRProcessor, VACOnlineASRProcessor
from asr.BaseAsr import OpenaiApiASR, FasterWhisperASR, WhisperTimestampedASR

# 定义全局变量
SAMPLE_RATE = 16000 

logger = logging.getLogger(__name__)

@lru_cache
def load_audio(fname: str) -> np.ndarray:
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname: str, beg: float, end: float) -> np.ndarray:
    audio = load_audio(fname)
    beg_s = int(beg * SAMPLE_RATE)
    end_s = int(end * SAMPLE_RATE)
    return audio[beg_s:end_s]

WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(",")

class UkrainianTokenizer:
    def split(self, text: str) -> List[str]:
        import tokenize_uk
        return tokenize_uk.tokenize_sents(text)

class WtPtok:
    def __init__(self, lan: Optional[str]) -> None:
        from wtpsplit import WtP
        self.wtp = WtP("wtp-canine-s-12l-no-adapters")
        self.lan = lan

    def split(self, sent: str) -> List[str]:
        return self.wtp.split(sent, lang_code=self.lan)

def create_tokenizer(lan: str) -> Any:
    """returns an object that has split function that works like the one of MosesTokenizer"""

    assert lan in WHISPER_LANG_CODES, "language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)

    if lan == "uk":
        return UkrainianTokenizer()

    # supported by fast-mosestokenizer
    if lan in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split():
        from mosestokenizer import MosesTokenizer
        return MosesTokenizer(lan)

    # the following languages are in Whisper, but not in wtpsplit:
    if lan in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split():
        logger.debug(f"{lan} code is not supported by wtpsplit. Going to use None lang_code option.")
        lan = None

    return WtPtok(lan)

def add_shared_args(parser: ArgumentParser) -> None:
    """shared args for simulation (this entry point) and server
    parser: argparse.ArgumentParser object
    """
    parser.add_argument('--min-chunk-size', type=float, default=1.0, help='Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.')
    parser.add_argument('--model', type=str, default='base', choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large".split(","),help="Name size of the Whisper model to use (default: large-v2). The model is automatically downloaded from the model hub if not present in model cache dir.")
    parser.add_argument('--model_cache_dir', type=str, default=None, help="Overriding the default model cache dir where models downloaded from the hub are saved")
    parser.add_argument('--model_dir', type=str, default=None, help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.")
    parser.add_argument('--lan', '--language', type=str, default='auto', help="Source language code, e.g. en,de,cs, or 'auto' for language detection.")
    parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe","translate"],help="Transcribe or translate.")
    parser.add_argument('--backend', type=str, default="faster-whisper", choices=["faster-whisper", "whisper_timestamped", "openai-api"],help='Load only this backend for Whisper processing.')
    parser.add_argument('--vac', action="store_true", default=False, help='Use VAC = voice activity controller. Recommended. Requires torch.')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04, help='VAC sample size in seconds.')
    parser.add_argument('--vad', action="store_true", default=False, help='Use VAD = voice activity detection, with the default parameters.')
    parser.add_argument('--buffer_trimming', type=str, default="segment", choices=["sentence", "segment"],help='Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.')
    parser.add_argument('--buffer_trimming_sec', type=float, default=15, help='Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.')
    parser.add_argument("-l", "--log-level", dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the log level", default='DEBUG')

def asr_factory(args: Namespace, logfile: Any = sys.stderr) -> Tuple[Any, Any]:
    """
    Creates and configures an ASR and ASR Online instance based on the specified backend and arguments.
    """
    backend = args.backend
    if backend == "openai-api":
        logger.debug("Using OpenAI API.")
        asr = OpenaiApiASR(lan=args.lan)
    else:
        if backend == "faster-whisper":
            asr_cls = FasterWhisperASR
        else:
            asr_cls = WhisperTimestampedASR

        # Only for FasterWhisperASR and WhisperTimestampedASR
        size = args.model
        t = time.time()
        logger.info(f"Loading Whisper {size} model for {args.lan}...")
        asr = asr_cls(modelsize=size, lan=args.lan, cache_dir=args.model_cache_dir, model_dir=args.model_dir)
        e = time.time()
        logger.info(f"done. It took {round(e-t,2)} seconds.")

    # Apply common configurations
    if getattr(args, 'vad', False):  # Checks if VAD argument is present and True
        logger.info("Setting VAD filter")
        asr.use_vad()

    language = args.lan
    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "en"  # Whisper translates into English
    else:
        tgt_language = language  # Whisper transcribes in this language

    # Create the tokenizer
    if args.buffer_trimming == "sentence":
        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None

    # Create the OnlineASRProcessor
    if args.vac:
        online = VACOnlineASRProcessor(args.min_chunk_size, asr, tokenizer, logfile=logfile, buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))
    else:
        online = OnlineASRProcessor(asr, tokenizer, logfile=logfile, buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))

    return asr, online

def set_logging(args: Namespace, logger: logging.Logger, other: str = "_server") -> None:
    logging.basicConfig(format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)
    logging.getLogger("whisper_online" + other).setLevel(args.log_level)