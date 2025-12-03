#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CosyVoice 
@File    ：self_test.py
@Author  ：Neil
@Date    ：2025/4/13 13:33 
"""

import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import time

start = time.time()
cosyvoice = CosyVoice2('iic/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# # zero_shot usage
prompt_speech_16k = load_wav('BaseVoice.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# #
# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# 在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。
# In the middle of telling the <strong>fantastic</strong> story, he suddenly [laughter] stopped because he was laughing himself [laughter].
for i, j in enumerate(cosyvoice.inference_cross_lingual('你好，欢迎使用语音测试功能，我是Neil，希望你能喜欢这个功能', prompt_speech_16k, stream=False)):
    torchaudio.save('outputs/example.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# # # # instruct usage
# for i, j in enumerate(cosyvoice.inference_instruct2('В середине рассказа этой фантастической истории он внезапно остановился, потому что сам смеялся.', '用俄语说这句话', prompt_speech_16k, stream=False)):
#     torchaudio.save('outputs/instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

print(time.time() - start)

