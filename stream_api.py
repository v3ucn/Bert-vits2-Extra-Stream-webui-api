import argparse
import os,io,wave
import sys
import re_matching

now_dir = os.getcwd()

from scipy.io.wavfile import write


def audio_array_to_bytes(sample_rate, audio_data):
    # 将音频数据转换为 bytes
    audio_data = np.array(audio_data, dtype=np.int16)
    
    # 创建一个 BytesIO 对象
    byte_io = io.BytesIO()
    
    # 将音频数据写入 BytesIO 对象
    write(byte_io, sample_rate, audio_data)
    
    # 从 BytesIO 对象中读取二进制数据
    result_bytes = byte_io.getvalue()
    
    return result_bytes


from starlette.middleware.cors import CORSMiddleware  #引入 CORS中间件模块

#设置允许访问的域名
origins = ["*"]  #"*"，即为所有。

import signal
from time import time as ttime
import torch
import librosa
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

import numpy as np

from io import BytesIO

import utils
from infer import infer_stream, latest_version, get_net_g, infer_multilang

from config import config
from tools.translate import translate
import gradio as gr

net_g = None

device = config.webui_config.device
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


hps = utils.get_hparams_from_file(config.webui_config.config_path)
# 若config.json中未指定版本则默认为最新版本
version = hps.version if hasattr(hps, "version") else latest_version
net_g = get_net_g(
    model_path=config.webui_config.model, version=version, device=device, hps=hps
)
speaker_ids = hps.data.spk2id
speakers = list(speaker_ids.keys())

# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=44100):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()

def generate_audio(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    style_text,
    style_weight,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            audio = infer(
                piece,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                style_text=style_text,
                style_weight=style_weight,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            # audio_list.append(silence)  # 将静音添加到列表中
    return audio_list


def generate_audio_multilang(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    style_text,
    style_weight,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            audio = infer_multilang(
                piece,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language[idx],
                hps=hps,
                net_g=net_g,
                device=device,
                style_text=style_text,
                style_weight=style_weight,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            # audio_list.append(silence)  # 将静音添加到列表中
    return audio_list


def tts_split_stream(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    cut_by_sent,
    interval_between_para,
    interval_between_sent,
    reference_audio,
    emotion,
    style_text,
    style_weight,
):
    if style_text == "":
        style_text = None
    if language == "mix":
        return ("invalid", None)
    while text.find("\n\n") != -1:
        text = text.replace("\n\n", "\n")
    para_list = re_matching.cut_para(text)
    

    cut_by_sent = True

    for idx, p in enumerate(para_list):
        skip_start = idx != 0
        skip_end = idx != len(para_list) - 1
        
        sent_list = re_matching.cut_sent(p)
        for idx, s in enumerate(sent_list):

            audio_list_sent = []
            audio_list = []

            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(sent_list) - 1) and skip_end
            chunks = infer_stream(
                s,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                style_text=style_text,
                style_weight=style_weight,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            
            for audio in chunks:
            
                audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
                audio_list.append(audio16bit)
                #silence = np.zeros((int)(44100 * interval_between_para), dtype=np.int16)
                #audio_list.append(silence)
                print(s)
                audio_concat = np.concatenate(audio_list)
                byt = audio_array_to_bytes(44100,audio_concat)
                yield wave_header_chunk()
                # wav = BytesIO()
                # sf.write(wav, audio_concat, 44100, format="wav")
                # wav.seek(0)
                yield byt
            
            #     silence = np.zeros((int)(44100 * interval_between_sent))
            #     audio_list_sent.append(silence)
            # if (interval_between_para - interval_between_sent) > 0:
            #     silence = np.zeros(
            #         (int)(44100 * (interval_between_para - interval_between_sent))
            #     )
            #     audio_list_sent.append(silence)
            # audio16bit = gr.processing_utils.convert_to_16_bit_wav(
            #     np.concatenate(audio_list_sent)
            # )  # 对完整句子做音量归一
            # audio_list.append(audio16bit)




app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,  #设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  #允许跨域的headers，可以用来鉴别来源等作用。



@app.get("/")
async def handle(text: str,speaker=speakers[0],sdp_ratio=0.2,noise_scale=0.6,noise_scale_w=0.8,length_scale=1,language="ZH",cut_by_sent=True,interval_between_para=1,interval_between_sent=0.2,reference_audio=None,emotion="Happy",style_text="",style_weight=0.7,stream = False):
    
    if text == "" or text is None:
        
        return JSONResponse({"code":400,"message":"推理文本不能为空"}, status_code=400)

    if speaker == "" or speaker is None:
        
        return JSONResponse({"code":400,"message":"角色名称不能为空"}, status_code=400)

    

    return StreamingResponse(
        tts_split_stream(text,speaker,sdp_ratio,noise_scale,noise_scale_w,length_scale,language,cut_by_sent,interval_between_para,interval_between_sent,reference_audio,emotion,style_text,style_weight),
        media_type="audio/x-wav",
    )


@app.post("/")
async def tts_endpoint(request: Request):
    json_post_raw = await request.json()

    text = json_post_raw.get("text")

    speaker = json_post_raw.get("speaker",speakers[0])

    sdp_ratio = json_post_raw.get("sdp_ratio",0.2)

    noise_scale = json_post_raw.get("noise_scale",0.6)

    noise_scale_w = json_post_raw.get("noise_scale_w",0.8)

    length_scale = json_post_raw.get("length_scale",1)

    language = json_post_raw.get("language","ZH")

    cut_by_sent = json_post_raw.get("cut_by_sent",True)

    interval_between_para = json_post_raw.get("interval_between_para",1)

    interval_between_sent = json_post_raw.get("interval_between_sent",0.2)

    reference_audio = json_post_raw.get("reference_audio",None)

    emotion = json_post_raw.get("emotion","Happy")

    style_text = json_post_raw.get("style_text","")

    style_weight = json_post_raw.get("style_weight",0.7)

    stream = json_post_raw.get("stream",False)


    if text == "" or text is None:
        
        return JSONResponse({"code":400,"message":"推理文本不能为空"}, status_code=400)

    if speaker == "" or speaker is None:
        
        return JSONResponse({"code":400,"message":"角色名称不能为空"}, status_code=400)


    with torch.no_grad():
        gen = tts_split(text,speaker,sdp_ratio,noise_scale,noise_scale_w,length_scale,language,cut_by_sent,interval_between_para,interval_between_sent,reference_audio,emotion,style_text,style_weight)
        sampling_rate, audio_data = gen

    wav = BytesIO()
    sf.write(wav, audio_data, sampling_rate, format="wav")
    wav.seek(0)

    torch.cuda.empty_cache()
    if device == "mps":
        print('executed torch.mps.empty_cache()')
        torch.mps.empty_cache()
    return StreamingResponse(wav, media_type="audio/wav")
    
    


@app.post("/set_tts_settings")
async def set_tts_settings(request: Request):
    json_post_raw = await request.json()
    return JSONResponse(["female_calm","female","male"], status_code=200)



def speaker_handle():

    return JSONResponse(["female_calm","female","male"], status_code=200)


@app.get("/speakers")
async def speakers_endpoint():
    return JSONResponse([{"name":"default","vid":1}], status_code=200)


@app.get("/speakers_list")
async def speakerlist_endpoint():
    return speaker_handle()



@app.post("/tts_to_audio/")
async def tts_to_audio(request: Request):

    json_post_raw = await request.json()

    text = json_post_raw.get("text")

    speaker = json_post_raw.get("speaker",speakers[0])

    sdp_ratio = json_post_raw.get("sdp_ratio",0.2)

    noise_scale = json_post_raw.get("noise_scale",0.6)

    noise_scale_w = json_post_raw.get("noise_scale_w",0.8)

    length_scale = json_post_raw.get("length_scale",1)

    language = json_post_raw.get("language","ZH")

    cut_by_sent = json_post_raw.get("cut_by_sent",True)

    interval_between_para = json_post_raw.get("interval_between_para",1)

    interval_between_sent = json_post_raw.get("interval_between_sent",0.2)

    reference_audio = json_post_raw.get("reference_audio",None)

    emotion = json_post_raw.get("emotion","Happy")

    style_text = json_post_raw.get("style_text","")

    style_weight = json_post_raw.get("style_weight",0.7)

    stream = json_post_raw.get("stream",False)


    if text == "" or text is None:
        
        return JSONResponse({"code":400,"message":"推理文本不能为空"}, status_code=400)

    if speaker == "" or speaker is None:
        
        return JSONResponse({"code":400,"message":"角色名称不能为空"}, status_code=400)


    with torch.no_grad():
        gen = tts_split(text,speaker,sdp_ratio,noise_scale,noise_scale_w,length_scale,language,cut_by_sent,interval_between_para,interval_between_sent,reference_audio,emotion,style_text,style_weight)
        sampling_rate, audio_data = gen

    wav = BytesIO()
    sf.write(wav, audio_data, sampling_rate, format="wav")
    wav.seek(0)

    torch.cuda.empty_cache()
    if device == "mps":
        print('executed torch.mps.empty_cache()')
        torch.mps.empty_cache()
    return StreamingResponse(wav, media_type="audio/wav")


if __name__ == "__main__":
    

    languages = ["ZH", "JP", "EN", "mix", "auto"]

    uvicorn.run(app, host="0.0.0.0", port=9886, workers=1)
