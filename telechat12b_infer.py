"""
@Time ： 2024/10/17 14:57
@Auth ： opprash
@File ：telechat12b_infer.py
@IDE ：PyCharm
"""
import os
import torch
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from modeling_telechat_gptq import TelechatGPTQForCausalLM
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
PATH = '/home/base_models/telechat-12b-chat'


#model_path = '/home/base_models/baichuan2-13b-chat'
model_path = '/home/base_models/Telechat-52b'
#model_path = '/home/base_models/Qwen-14B-Chat'
#model_path = '/home/llm_models/Qwen1.5-14B-Chat-AWQ'

do_sample = False
max_new_tokens = 512

r_telechat_prefix = re.compile(r"<_user>.+<_bot>")


class QwenAdaptor:
    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        #model = AutoModel.from_pretrained(qa_lora_model_path, trust_remote_code=True).quantize(quant_num).cuda()
        #model = AutoModelForCausalLM.from_pretrained(base_model_data_path, torch_dtype=torch.float16, trust_remote_code=True).quantize(quant_num).cuda()
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
        )
        #self.model = AutoModelForCausalLM.from_pretrained(self.model_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
        #self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
             self.model_path,
             device_map="auto",
             trust_remote_code=True,
             use_cache_quantization=True,
             use_cache_kernel=True,
             use_flash_attn=False
        )
        #self.model.generation_config = GenerationConfig.from_pretrained(self.model_path)
        #self.model.generation_config.max_new_tokens = max_new_tokens
        #self.model.generation_config.repetition_penalty = 1.0
        #self.model.generation_config.top_p = 1.0
        #self.model.generation_config.top_k = 1
        self.model.eval()
        self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        self.generation_config.max_new_tokens = max_new_tokens
        self.generation_config.repetition_penalty = 1.0
        self.generation_config.top_p = 1.0
        self.generation_config.top_k = 1

    def predict(self, uid, question, history, stream=False):
        # 可以这么调用传入history
        #history = [
        #    {"role": "user", "content": "你是谁"},
        #    {"role": "bot", "content": "我是telechat"},
        #]

        print("提问:", question)
        answer, history = self.model.chat(self.tokenizer,
                                        query=question,
                                        history=history,
                                        generation_config=self.generation_config)
                                        #stream=False)
        torch.cuda.empty_cache()
        return answer

class TelechatAdaptor:
    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        #model = AutoModel.from_pretrained(qa_lora_model_path, trust_remote_code=True).quantize(quant_num).cuda()
        #model = AutoModelForCausalLM.from_pretrained(base_model_data_path, torch_dtype=torch.float16, trust_remote_code=True).quantize(quant_num).cuda()
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
        )
        #self.model = AutoModelForCausalLM.from_pretrained(self.model_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        #self.model.generation_config = GenerationConfig.from_pretrained(self.model_path)
        #self.model.generation_config.max_new_tokens = max_new_tokens
        #self.model.generation_config.repetition_penalty = 1.0
        #self.model.generation_config.top_p = 1.0
        #self.model.generation_config.top_k = 1
        self.model.eval()
        self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        self.generation_config.max_new_tokens = max_new_tokens
        self.generation_config.repetition_penalty = 1.0
        self.generation_config.top_p = 1.0
        self.generation_config.top_k = 1

    def predict(self, uid, question, history, stream=False):
        # 可以这么调用传入history
        #history = [
        #    {"role": "user", "content": "你是谁"},
        #    {"role": "bot", "content": "我是telechat"},
        #]

        print("提问:", question)
        answer, history = self.model.chat(self.tokenizer, question=question, history=[], generation_config=self.generation_config,
                         stream=False)
        torch.cuda.empty_cache()
        return answer


class TelechatGPTQAdaptor:

    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        #logger.info("loading model...")
        self.model = TelechatGPTQForCausalLM.from_quantized(self.model_path, device_map="auto", inject_fused_mlp=False,
                                                   inject_fused_attention=False, trust_remote_code=True)
        #logger.info("load finish.")
        self.generate_config = GenerationConfig.from_pretrained(self.model_path)
        self.model.eval()

    def predict(self, uid, question, history, stream=True):
        messages = []
        for item in history:
            messages.append({"role": "user", 'content':item['question']})
            messages.append({"role": "bot", 'content':item['answer']})
        if stream:
            res = self.model.chat(tokenizer=self.tokenizer,
                        question=question,
                        history=messages,
                        generation_config=self.generate_config,
                        stream=True)
        else:
            res, his = self.model.chat(tokenizer=self.tokenizer,
                        question=question,
                        history=messages,
                        generation_config=self.generate_config,
                        stream=False)
            #gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        #for answer, history in gen:
        return res






#vllm

import aiohttp
import json
import traceback
import asyncio
import os
import configparser
from aiohttp import web
import socket
import logging
import time
import re
import json
from typing import AsyncGenerator

from vllm.utils import random_uuid
from vllm.sampling_params import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from transformers import AutoTokenizer
import torch

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores
class VLLM:
    def __init__(self, config):
        self.config = config
        #os.environ['CUDA_VISIBLE_DEVICES'] = self.config.device
        #self.logger = self.create_logger()
        self.max_input_len = int(self.config.max_input_len)
        self.max_new_token = int(self.config.max_new_token)

        self.temperature = self.config.temperature
        self.top_p = self.config.top_p
        self.repetition_penalty = self.config.repetition_penalty

        logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        request_dict = {
            "n": 1,
            "temperature": self.temperature,
            "repetition_penalty":self.repetition_penalty,
            #"top_p": self.top_p,
            "max_tokens": self.max_new_token,
            "logits_processors": logits_processor
        }

        self.sampling_params = SamplingParams(**request_dict)
        print(self.sampling_params)
        self.engine = self.init_engine()
        #self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir)

    def build_rsp(self, answer, req_id,  tokens, cost_t, speed):
        elements = []
        for a in answer.split('\n\n'):
            a = a.split("：")
            elements.append({"tag":a[0],"value":a[1]})

        resp = {
            "id": req_id,
            "jsonrpc": "2.0",
            "ret": 0,
            "result": {
                "chatInfo": {
                    "answer": answer,
                    "elements": elements
                },
                "tokens": tokens,
                "cost_time": str(cost_t) + " ms",
                "speed": str(speed) + " tokens/s"
            }
        }
        return resp

    def init_engine(self):
        self.model_dir = self.config.model_dir

        engine_args = AsyncEngineArgs(model=self.model_dir, trust_remote_code=True, disable_log_requests=True)
        engine_args.max_model_len = self.config.max_model_len
        #self.logger.info(engine_args)
        engine_args.gpu_memory_utilization = self.config.gpu_memory_utilization
        engine_args.tensor_parallel_size = self.config.tensor_parallel_size
        if hasattr(self.config, "dtype"):
            engine_args.dtype = self.config.dtype
        print(engine_args)
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        return engine


class VLLMQwen(VLLM):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir)

    async def predict(self, text_id, content, history=[], system=None, stream=True):
        if not system:
            system = 'You are a helpful assistant.'
        messages = [
            {"role": "system", "content": system},
        ]
        session_id = text_id
        for item in history:
            messages.append({'role': 'user', 'content': item['question']})
            messages.append({'role': 'assistant', 'content': item['answer']})
        messages.append({"role": "user", "content": content})

        query = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"query_len:{len(query)}")
        #model_inputs = tokenizer([text], return_tensors="pt").to(device)
        start = time.time()
        request_id = random_uuid()
        if stream:
            results = self.engine.generate(prompt=query,sampling_params=self.sampling_params, request_id=request_id)
            #results = self.engine.generate(inputs=query,sampling_params=self.sampling_params, request_id=request_id)
        return results

class VLLMTele12B(VLLM):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir, trust_remote_code=True)

    async def predict(self, text_id, content, history=[], system=None, stream=True):
        if system:
            query = system+'\n'
        else:
            query = ""
        for i in range(len(history)):
            query += "<_user>" + history[i]['question'] + "<_bot>" + history[i]['answer'] + "<_end>"
        query += "<_user>" + content + "<_bot>"
        request_id = random_uuid()
        if stream:
            #results = self.engine.generate(prompt=query,sampling_params=self.sampling_params, request_id=request_id)
            results = self.engine.generate(inputs=query,sampling_params=self.sampling_params, request_id=request_id)
        return results





def main():
    # 加载模型相关
    tokenizer = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, device_map="auto",
                                                 torch_dtype=torch.float16)
    generate_config = GenerationConfig.from_pretrained(PATH)
    model.eval()
    print("*" * 10 + "流式输入演示" + "*" * 10)
    question = "你是谁？"
    print("提问:", question)
    gen = model.chat(tokenizer, question=question, history=[], generation_config=generate_config,
                     stream=True)
    for answer, history in gen:
        print("回答是:", answer)
        print("截至目前的聊天记录是:", history)




if __name__ == '__main__':
    main()