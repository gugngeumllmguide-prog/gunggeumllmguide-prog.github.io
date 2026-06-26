# import pyaudio
# p = pyaudio.PyAudio()
# for i in range(p.get_device_count()):
#     info = p.get_device_info_by_index(i)
#     print(i, info['name'], '| 입력채널:', info['maxInputChannels'])
# p.terminate()
import pyaudio
p = pyaudio.PyAudio()
for idx in [16, 18]:
    try:
        s = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                   frames_per_buffer=320, input_device_index=idx)
        s.close()
        print(f"인덱스 {idx} 성공")
    except Exception as e:
        print(f"인덱스 {idx} 실패: {e}")
p.terminate()