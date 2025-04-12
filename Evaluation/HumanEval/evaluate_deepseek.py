# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import re
from torch.nn import DataParallel
import transformers
import torch
import fire
import pandas as pd
import json
import time
import numpy as np
import random
import os
# from peft import PeftModel
from utils import *

def build_deepseekcoder_instruction(languge: str, question: str):
    return '''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
'''.strip().format(languge.lower(), question.strip())

# 为保持结果的一致性，设置随机数种子
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(
    ckpt_dir: str =None ,
    peft_path:str = None,
    tokenizer_path: str = 'deepseek-ai/deepseek-coder-1.3b-base',
    temperature: float = 0.2,
    # top_k: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    language:str = 'java',
    human_eval_path = "/home/baoxuanlin/graduation/ssf/DeepSeek-Coder/Evaluation/HumanEval/data/humaneval-java.jsonl",
    output_path = "/home/baoxuanlin/graduation/ssf/DeepSeek-Coder/Evaluation/HumanEval/output/test.jsonl",

):

    do_sample = False
    if not do_sample:
        temperature=1
        top_p=1
    seed=0
    set_random_seed(seed)
    data_name = "human_eval"
    do_filter = 0 # 0 is our
    max_batch_size = 25
    max_seq_len = 1024
    # ck_ite_list = [i for i in range(0,10)]
    # ck_ite_list = [21]

    learn_way = "deepseek-coder-6.7b"  # seq kd minillm distillcodecraft
    ck_ite_list = [i for i in range(0,1)]


    if ck_ite_list is not None or len(ck_ite_list) != 0:
        for ck_ite in ck_ite_list:
            # ckpt_dir = f"/home/baoxuanlin/graduation/magicoder/data/finetune_ssf/{learn_way}/checkpoint-epoch-{ck_ite}"
            # ckpt_dir = f"deepseek-ai/deepseek-coder-1.3b-base"
            ckpt_dir = f"/home/baoxuanlin/llms/deepseek-ai"
            

            if ck_ite is None:
                ck_ite = "tiny"
            
            output_path = f"/home/baoxuanlin/graduation/ssf/DeepSeek-Coder/Evaluation/HumanEval/output/{learn_way}/{ck_ite}_{data_name}_{max_seq_len}_{max_batch_size}_temp{temperature}_tp{top_p}_{seed}_dofil{do_filter}"
            if peft_path is not None:
                output_path = output_path + f"_peft_{peft_ite}"
            
            if do_filter is not None:
                output_path = output_path + "_filter"

            output_path = output_path + ".jsonl"

            directory = os.path.dirname(output_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if not os.path.exists(output_path):
                with open(output_path, 'w'):
                    pass  
            model_path = ckpt_dir
            tokenizer_path = ckpt_dir if tokenizer_path is None else tokenizer_path

            
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,trust_remote_code=False,code_revision=None)
            tokenizer.padding_side='left'
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto"
            )
            if peft_path is not None:
                model = PeftModel.from_pretrained(model, peft_path)
            # 需要对文本开头添加bos，末尾不需要添加eos
            tokenizer.pad_token = tokenizer.eos_token
            # inputs = ["for i in range(len(numbers)):","for i in range(len(numbers)): has_close_elements(numbers: List[float]"]
            # tokenizer.pad_token_id = -1

            dataframe = pd.read_json(human_eval_path,lines=True)
            prompts = dataframe['prompt'].to_list()
            task_ids = dataframe['task_id'].to_list()
            for i in range(0, len(prompts), max_batch_size):
                # 从原始列表中取出4个数据作为批次
                task_batch = task_ids[i:i + max_batch_size]
                prompts_batch = prompts[i:i + max_batch_size]
                prompts_batch = [build_deepseekcoder_instruction(language,prompt) for prompt in prompts_batch] 
                model_inputs = tokenizer(prompts_batch, return_tensors="pt", padding=True).to("cuda")
                
                with torch.no_grad():
                    generated_ids = model.generate(**model_inputs,
                                                do_sample = do_sample,
                                                max_new_tokens=max_seq_len,
                                                # top_k=top_k,
                                                # top_p=top_p,
                                                # temperature=temperature,
                                                )
                # output = tokenizer.decode(generated_ids,skip_special_tokens=True)
                output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)# skip_special_tokens指跳过特殊令牌
                if do_filter == 0:
                    output = [gen_text for _, gen_text in enumerate(output)]        # model_inputs = tokenizer(prompts_batch, return_tensors="pt", padding=False).to("cuda")
                elif do_filter ==1:
                    output = [gen_text[len(prompts_batch[idx]):] for idx, gen_text in enumerate(output)]


                # combined_batch = [{"task_id": task_id, "completion": generation} for task_id, generation in zip(task_batch, output)]
                combined_batch = [{"task_id": task_id, "completion": filter_data(generation)} for task_id, generation in zip(task_batch, output)]
                with open(output_path, "a") as file:
                    for item in combined_batch:
                        json.dump(item, file)  # 将字典转换为 JSON 格式
                        file.write("\n")  # 写入换行符，以便每个记录都在单独的行上

def filter_data(completion:str):
  if '```java' in completion: 
    def_line = completion.index('```java')
    completion = completion[def_line:].strip()
    completion = completion.replace('```java', '')
    # print(completion)
    try:
        next_line = completion.index('```')
        completion = completion[:next_line].strip()
    except:
        print(completion)
        return completion
  return completion


if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    end_time = time.time()
    print("耗时: {:.2f}秒".format(end_time - start_time))

