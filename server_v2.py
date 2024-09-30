import eventlet
import socketio
import sys
import os
import logging
import numpy as np
import io
import soundfile
import librosa
import argparse
from typing import Optional, Dict, Any, Tuple, List
from asr.OnlineAsr import OnlineASRProcessor, VACOnlineASRProcessor
from asr.BaseAsr import OpenaiApiASR, FasterWhisperASR, WhisperTimestampedASR
from asr.utils import load_audio, load_audio_chunk, asr_factory, add_shared_args, set_logging

# 允许所有域名跨域请求
sio = socketio.Server(cors_allowed_origins='*', namespace='/socket.io/')
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000

class Connection:
    '''it wraps conn object'''
    PACKET_SIZE = 32000 * 5 * 60  # 5 minutes # was: 65536

    def __init__(self, sid: str) -> None:
        self.sid = sid
        self.last_line = ""

    def send(self, line: str) -> None:
        '''it doesn't send the same line twice, because it was problematic in online-text-flow-events'''
        if line == self.last_line:
            return
        sio.emit('transcript', line, room=self.sid)
        self.last_line = line

class ServerProcessor:
    min_chunk: int
    def __init__(self, online_asr_proc: OnlineASRProcessor) -> None:
        self.connection: Optional[Connection] = None
        self.online_asr_proc = online_asr_proc
        self.online_asr_proc.init()
        self.min_chunk = 5

        self.last_end: Optional[float] = None
        self.is_first = True
        self.buffer = bytearray()

    def set_connection(self, sid: str) -> None:
        self.connection = Connection(sid)

    def receive_audio_chunk(self, raw_bytes: bytes) -> Optional[np.ndarray]:
        self.buffer.extend(raw_bytes)  # 将接收到的raw_bytes添加到缓冲区中

        minlimit = self.min_chunk * SAMPLING_RATE
        if len(self.buffer) < minlimit:
            print("Buffer size is too small", len(self.buffer))
            return None  # 如果缓冲区中的数据不足，返回None
        out = []
        sf = soundfile.SoundFile(io.BytesIO(self.buffer), channels=2, endian="LITTLE", samplerate=SAMPLING_RATE, subtype="PCM_16", format="RAW")
        audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)
        out.append(audio)
        if not out:
            print("No audio in this segment")
            return None
        conc = np.concatenate(out)
        if self.is_first and len(conc) < 50:
            print("The first chunk is too short", len(conc))
            return None
        self.is_first = False
        self.buffer = bytearray()  # 清空缓冲区
        return np.concatenate(out)

    def format_output_transcript(self, o: Tuple[Optional[float], Optional[float], str]) -> Optional[str]:
        if o[0] is not None:
            beg, end = o[0] * 1000, o[1] * 1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            print("%1.0f %1.0f %s" % (beg, end, o[2]), flush=True, file=sys.stderr)
            return "%1.0f %1.0f %s" % (beg, end, o[2])
        else:
            logger.debug("No text in this segment")
            return None

    def send_result(self, o: Tuple[Optional[float], Optional[float], str]) -> None:
        msg = self.format_output_transcript(o)
        if msg is not None:
            self.connection.send(msg)

    def process(self, raw_bytes: bytes) -> None:
        a = self.receive_audio_chunk(raw_bytes)
        if a is None:
            print("No audio in this segment")
            return
        print("process a size ", len(a))
        self.online_asr_proc.insert_audio_chunk(a)
        o = self.online_asr_proc.process_iter()
        print('online.process_iter', o)
        self.send_result(o)

# 存储每个连接的 ServerProcessor 实例
connections: Dict[str, ServerProcessor] = {}

@sio.event
def connect(sid: str, environ: Dict[str, Any]) -> None:
    print('connect ', sid)
    # 为每个连接创建一个新的 ServerProcessor 实例
    connections[sid] = ServerProcessor(online)

@sio.event
def stream(sid: str, data: bytes) -> None:
    if sid in connections:
        # 检测音频流数据大小
        print('stream ', sid, len(data))
        server_processor = connections[sid]
        server_processor.set_connection(sid)
        server_processor.process(data)

@sio.event
def disconnect(sid: str) -> None:
    print('disconnect ', sid)
    # 断开连接时删除对应的 ServerProcessor 实例
    if sid in connections:
        del connections[sid]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=43007)
    parser.add_argument("--warmup-file", type=str, dest="warmup_file", 
            help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")
    add_shared_args(parser)
    args = parser.parse_args()
    args.model = "base"
    args.backend = "faster-whisper"

    set_logging(args, logger, other="")

    size = args.model
    language = args.lan
    asr, online = asr_factory(args)
    min_chunk = 100

    msg = "Whisper is not warmed up. The first chunk processing may take longer."
    if args.warmup_file:
        if os.path.isfile(args.warmup_file):
            a = load_audio_chunk(args.warmup_file, 0, 1)
            asr.transcribe(a)
            logger.info("Whisper is warmed up.")
        else:
            logger.critical("The warm up file is not available. " + msg)
            sys.exit(1)
    else:
        logger.warning(msg)

    eventlet.wsgi.server(eventlet.listen((args.host, args.port)), app)