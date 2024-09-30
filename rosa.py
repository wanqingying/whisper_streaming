import librosa
import soundfile as sf

# 读取音频文件
audio_path = './recording-4.wav'
y, sr = librosa.load(audio_path, sr=None)

# 获取音频文件的信息
with sf.SoundFile(audio_path) as f:
    sample_rate = f.samplerate
    channels = f.channels
    subtype = f.subtype

# 输出音频文件的信息
print(f"采样率: {sample_rate} Hz")
print(f"通道数: {channels}")
print(f"位深度: {subtype}")