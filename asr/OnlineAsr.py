#!/usr/bin/env python3
import sys
import numpy as np
import librosa
from functools import lru_cache
import time
import logging
import io
import soundfile as sf
import math
from typing import Optional, List, Tuple, Dict, Any

# 定义全局变量
# SAMPLE_RATE = 16000 
SAMPLE_RATE = 44100 

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

class HypothesisBuffer:
    def __init__(self, logfile: Any = sys.stderr) -> None:
        self.commited_in_buffer: List[Tuple[float, float, str]] = []
        self.buffer: List[Tuple[float, float, str]] = []
        self.new: List[Tuple[float, float, str]] = []

        self.last_commited_time: float = 0
        self.last_commited_word: Optional[str] = None

        self.logfile = logfile

    def insert(self, new: List[Tuple[float, float, str]], offset: float) -> None:
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):  # 5 is the maximum 
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1])
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = " ".join(words)
                            logger.debug(f"removing last {i} words: {words_msg}")
                            break

    def flush(self) -> List[Tuple[float, float, str]]:
        commit: List[Tuple[float, float, str]] = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time: float) -> None:
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self) -> List[Tuple[float, float, str]]:
        return self.buffer

class OnlineASRProcessor:
    SAMPLING_RATE: int = 16000

    def __init__(self, asr: Any, tokenizer: Optional[Any] = None, buffer_trimming: Tuple[str, int] = ("segment", 15), logfile: Any = sys.stderr) -> None:
        """asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language. 
        Must have a method *split* that behaves like the one of MosesTokenizer. 
        It can be None, if "segment" buffer trimming option is used, then tokenizer is not used at all.
        ("segment", 15)
        buffer_trimming: a pair of (option, seconds), 
        where option is either "sentence" or "segment", and seconds is a number. 
        Buffer is trimmed if it is longer than "seconds" threshold. Default is the most recommended option.
        logfile: where to store the log. 
        """
        self.asr = asr
        self.tokenizer = tokenizer
        self.logfile = logfile

        self.init()

        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self, offset: Optional[float] = None) -> None:
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset: float = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited: List[Tuple[float, float, str]] = []

    def insert_audio_chunk(self, audio: np.ndarray) -> None:
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self) -> Tuple[str, str]:
        """Returns a tuple: (prompt, context), 
        where "prompt" is a 200-character suffix of commited text 
        that is inside of the scrolled away part of audio buffer. 
        "context" is the commited text that is inside the audio buffer. 
        It is transcribed again and skipped. It is returned only for debugging and logging reasons.
        """
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1

        p = self.commited[:k]
        p = [t for _, _, t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x) + 1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(t for _, _, t in non_prompt)

    def process_iter(self) -> Tuple[Optional[float], Optional[float], str]:
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, ""). 
        The non-emty text is confirmed (committed) partial transcript.
        """
        print("process_iter")
        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT: {prompt}")
        logger.debug(f"CONTEXT: {non_prompt}")
        logger.debug(f"transcribing {len(self.audio_buffer) / self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}")
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

        # transform to [(beg,end,"word1"), ...]
        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)
        completed = self.to_flush(o)
        logger.debug(f">>>>COMPLETE NOW: {completed}")
        the_rest = self.to_flush(self.transcript_buffer.complete())
        logger.debug(f"INCOMPLETE: {the_rest}")

        # there is a newly confirmed text

        if o and self.buffer_trimming_way == "sentence":  # trim the completed sentences
            if len(self.audio_buffer) / self.SAMPLING_RATE > self.buffer_trimming_sec:  # longer than this
                self.chunk_completed_sentence()

        if self.buffer_trimming_way == "segment":
            s = self.buffer_trimming_sec  # trim the completed segments longer than s,
        else:
            s = 30  # if the audio buffer is longer than 30s, trim it

        if len(self.audio_buffer) / self.SAMPLING_RATE > s:
            self.chunk_completed_segment(res)
            logger.debug("chunking segment")

        logger.debug(f"len of buffer now: {len(self.audio_buffer) / self.SAMPLING_RATE:2.2f}")
        return self.to_flush(o)

    def chunk_completed_sentence(self) -> None:
        if self.commited == []:
            return
        logger.debug(self.commited)
        sents = self.words_to_sentences(self.commited)
        for s in sents:
            logger.debug(f"\t\tSENT: {s}")
        if len(sents) < 2:
            return
        while len(sents) > 2:
            sents.pop(0)
        chunk_at = sents[-2][1]
        logger.debug(f"--- sentence chunked at {chunk_at:2.2f}")
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res: Any) -> None:
        if self.commited == []:
            return

        ends = self.asr.segments_end_ts(res)
        t = self.commited[-1][1]

        if len(ends) > 1:
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= t:
                logger.debug(f"--- segment chunked at {e:2.2f}")
                self.chunk_at(e)
            else:
                logger.debug(f"--- last segment not within commited area")
        else:
            logger.debug(f"--- not enough segments to chunk")

    def chunk_at(self, time: float) -> None:
        """trims the hypothesis and audio buffer at "time" """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.SAMPLING_RATE):]
        self.buffer_time_offset = time

    def words_to_sentences(self, words: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
        """Uses self.tokenizer for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """
        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out: List[Tuple[float, float, str]] = []
        while s:
            beg: Optional[float] = None
            end: Optional[float] = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b, e, w = cwords.pop(0)
                w = w.strip()
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg, end, fsent))
                    break
                sent = sent[len(w):].strip()
        return out

    def finish(self) -> Tuple[Optional[float], Optional[float], str]:
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logger.debug(f"last, noncommited: {f}")
        self.buffer_time_offset += len(self.audio_buffer) / 16000
        return f

    def to_flush(self, sents: List[Tuple[float, float, str]], sep: Optional[str] = None, offset: float = 0) -> Tuple[Optional[float], Optional[float], str]:
        """concatenates the timestamped words or sentences into one sequence that is flushed in one line
        sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        """
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b, e, t)

class VACOnlineASRProcessor(OnlineASRProcessor):
    '''Wraps OnlineASRProcessor with VAC (Voice Activity Controller). 

    It works the same way as OnlineASRProcessor: it receives chunks of audio (e.g. 0.04 seconds), 
    it runs VAD and continuously detects whether there is speech or not. 
    When it detects end of speech (non-voice for 500ms), it makes OnlineASRProcessor to end the utterance immediately.
    '''

    def __init__(self, online_chunk_size: int, *a: Any, **kw: Any) -> None:
        self.online_chunk_size = online_chunk_size

        self.online = OnlineASRProcessor(*a, **kw)

        # VAC:
        import torch
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad:v4.0',
            model='silero_vad'
        )
        from silero_vad import VADIterator
        self.vac = VADIterator(model)  # we use all the default options: 500ms silence, etc.  

        self.logfile = self.online.logfile
        self.init()

    def init(self) -> None:
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size: int = 0

        self.is_currently_final: bool = False

        self.status: Optional[str] = None  # or "voice" or "nonvoice"
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset: int = 0  # in frames

    def clear_buffer(self) -> None:
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([], dtype=np.float32)

    def insert_audio_chunk(self, audio: np.ndarray) -> None:
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            frame = list(res.values())[0]
            if 'start' in res and 'end' not in res:
                self.status = 'voice'
                send_audio = self.audio_buffer[frame - self.buffer_offset:]
                self.online.init(offset=frame / self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.clear_buffer()
            elif 'end' in res and 'start' not in res:
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[:frame - self.buffer_offset]
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
            else:
                raise NotImplementedError("both start and end of voice in one chunk!!!")
        else:
            if self.status == 'voice':
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                self.buffer_offset += max(0, len(self.audio_buffer) - self.SAMPLING_RATE)
                self.audio_buffer = self.audio_buffer[-self.SAMPLING_RATE:]

    def process_iter(self) -> Tuple[Optional[float], Optional[float], str]:
        print("VAC process_iter")
        if self.is_currently_final:
            return self.finish()
        elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter()
            return ret
        else:
            print("no online update, only VAD", self.status, file=self.logfile)
            return (None, None, "")

    def finish(self) -> Tuple[Optional[float], Optional[float], str]:
        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return ret