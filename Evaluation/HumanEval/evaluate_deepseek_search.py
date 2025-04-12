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

def build_deepseekcoder_instruction(language: str, question: str, task_id: str, search_reference: int = 0) -> str:
    # 读取codematcher的检索结果
    codematcher_file = "/home/baoxuanlin/code/codematcher-demo/humaneval-java-merge_search_results.jsonl"
    
    prompt1 = '''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. I will provide you with the function to complete and may also include some similar code examples for reference.
'''
    # 默认提示词
    prompt = '''
Here is the given code to do completion:
```{}
{}
'''.strip().format(language.lower(), question.strip())
    
    # 只有在search_reference > 0时才读取参考代码
    if search_reference > 0:
        try:
            with open(codematcher_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data["task_id"] == task_id:
                        # 找到匹配的task_id
                        if "codematcher_results" in data and len(data["codematcher_results"]) > 0:
                            # 获取指定数量的检索结果，但不超过实际可用数量
                            examples_count = min(search_reference, len(data["codematcher_results"]))
                            reference_examples = data["codematcher_results"][:examples_count]
                            
                            # 在提示词中添加检索到的代码片段
                            if examples_count == 1:
                                prompt1 += "\n\nHere is a similar implementation that may help you with this task:\n```java\n{}\n```".format(reference_examples[0])
                            else:
                                prompt1 += "\n\nHere are some similar implementations that may help you with this task:"
                                for i, example in enumerate(reference_examples):
                                    prompt1 += "\n\nExample {}:\n```java\n{}\n```".format(i+1, example)
                        break
        except Exception as e:
            print(f"Error reading codematcher results: {e}")
    
    return prompt1 + prompt

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

    learn_way = "oss_evol_instruct"  # seq kd minillm distillcodecraft
    ck_ite_list = [i for i in range(7,8)]
    search_reference = 1


    if ck_ite_list is not None or len(ck_ite_list) != 0:
        for ck_ite in ck_ite_list:
            ckpt_dir = f"/home/baoxuanlin/graduation/magicoder/data/finetune_ssf/{learn_way}/checkpoint-epoch-{ck_ite}"
            # ckpt_dir = f"deepseek-ai/deepseek-coder-1.3b-base"

            if ck_ite is None:
                ck_ite = "tiny"
            
            output_path = f"/home/baoxuanlin/graduation/ssf/DeepSeek-Coder/Evaluation/HumanEval/output/{learn_way}/{ck_ite}_{data_name}_{max_seq_len}_{max_batch_size}_temp{temperature}_tp{top_p}_{seed}_dofil{do_filter}_filter_ref{search_reference}.jsonl"
            # if peft_path is not None:
            #     output_path = output_path + f"_peft_{peft_ite}"
            
            # if do_filter is not None:
            #     output_path = output_path + "_filter"


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
                prompts_batch = [build_deepseekcoder_instruction(language, prompt, task_id,search_reference) for prompt, task_id in zip(prompts_batch, task_batch)] 
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
    def_line = completion.rfind('```java')
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

