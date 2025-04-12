import os
import sys
import fire
import json
import gzip
import regex
import numpy as np
import pandas as pd
import itertools

from typing import *
from tqdm.auto import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from human_eval.data import stream_jsonl
from human_eval.execution import check_correctness
import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9507))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "go"    : [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
    ],
    "cpp"   : [
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<math.h>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
        "#include<cassert>"
    ],
    "cs": ["using System.Numerics;", "using System.Diagnostics;", "using System.Collections.Generic;", "using System.Linq;", "using System.Text;", "using System.Security.Cryptography;", "using System.Collections.Generic;"]
}


LANGUAGE_NAME = {
    "cpp"   : "CPP",
    "go"    : "Go",
    "java"  : "Java",
    "js"    : "JavaScript",
    "python": "Python",
}


def read_dataset(
    data_file: str = None,
    dataset_type: str = "humaneval",
    num_shot=None,
) -> Dict:
    """
    Reads a dataset and returns a dictionary of tasks.
    """
    if num_shot is not None:
        print(f"{num_shot}-shot setting...")
    if "humaneval" in dataset_type.lower():
        if data_file is None:
            current_path = os.path.dirname(os.path.abspath(__file__))
            data_file = os.path.join(current_path, "..", "humaneval-x", "python", "data", "humaneval_python.jsonl.gz")
        dataset = {task["task_id"]: task for task in stream_jsonl(data_file)}
    else:
        raise f"Dataset: {dataset_type} not supported."

    return dataset

def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def process_humaneval_test(sample, problems, example_test=False, is_mbpp=False, language="python"):
    """
    Processes a sample for evaluation.
    """
    task_id = sample["task_id"]
    if is_mbpp:
        return sample["generation"] + "\n" + "\n".join(problems[task_id]["test"])

    prompt = sample["prompt"]
    if example_test and "example_test" in problems[task_id] and problems[task_id]["example_test"] != "":
        test = problems[task_id]["example_test"]
    else:
        test = problems[task_id]["test"]
    code = remove_last_brace(sample["generation"])
    test = remove_first_brace(test)

    # Pre-process for different languages
    if language == "python":
        test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
        test_string = test_setup + code + "\n" + test + "\n"
    elif language == "cpp":
        test_set_up = ""
        for s in IMPORT_HELPER["cpp"]:
            if s not in prompt:
                test_set_up += s + "\n"
        test_string = test_set_up + "\n" + code + "\n" + test
    elif language == "java":
        test_string = code + "\n" + test
    elif language == "cs":
        test_set_up = ""
        for s in IMPORT_HELPER["cs"]:
            test_set_up += s + "\n"
        test_string = test_set_up + "\n" + code + "\n" + test
    elif language in ["js", "javascript", "ts", "sh", "go"]:
        test_string = code + "\n" + test
    elif language == "go232":
        import_string = problems[task_id]["import"]
        prompt = prompt.replace(import_string, "")
        if example_test and "example_test" in problems[task_id]:
            test = problems[task_id]["example_test"]
        else:
            test = problems[task_id]["test"]
        test_setup = problems[task_id]["test_setup"]
        other_pkgs = []
        for pkg in IMPORT_HELPER["go"]:
            if pkg not in test_setup:
                p = pkg.split("/")[-1]
                if p + "." in code:
                    other_pkgs.append(f"\"{pkg}\"")
        if other_pkgs:
            import_other_pkgs = "import (\n" + "    ".join([p + "\n" for p in other_pkgs]) + ")"
            test_string = test_setup + "\n" + import_other_pkgs + "\n" + prompt + code + "\n" + test
        else:
            test_string = test_setup + "\n" + prompt + code + "\n" + test
    elif language == "rust":
        main = "\nfn main(){ \n } \n"
        declaration = problems[task_id]["declaration"]
        test_string = main + declaration + prompt + code + test
    elif language == "php":
        if code[:5] != "<?php":
            code = "<?php\n" + code
        test_string = code + "\n" + test + "?>"
    return test_string


def stream_jsonl_all(filename: str) -> Iterable[Dict]:
    """
    Streams a JSONL file.
    """
    results = []
    if filename.endswith(".gz"):
        fp = gzip.open(open(filename, "rb"), "rt")
    else:
        fp = open(filename, "r")
    for line in fp:
        if any(not x.isspace() for x in line):
            results.append(json.loads(line))
    fp.close()

    return results


def evaluate_functional_correctness(
        org_file: str = None,
        generate_file:str = None,
        tmp_dir: str = "./",
        n_workers: int = 32,
        timeout: float = 10.0,
        problem_file: str = "../data/humaneval_python.jsonl.gz",
        out_dir: str = None,
        k: List[int] = [1, 10, 100],
        test_groundtruth: bool = False,
        example_test: bool = False,
        is_mbpp: bool = False,
        language: str = "java",
):
    """
    Evaluates the functional correctness of a model.
    """
    if example_test:
        print("Example test...")

    problems = read_dataset(problem_file,
                            dataset_type="humaneval")
    sample_jsonl = stream_jsonl_all(org_file)
    generate_jsonl = stream_jsonl_all(generate_file)
    df1 = pd.DataFrame(sample_jsonl)
    df2 = pd.DataFrame(generate_jsonl)
    df2 = df2.rename(columns={"completion": "generation"})
    merged_df = df1.merge(df2, on="task_id", how="left")
    merged_data = merged_df.to_dict(orient="records")
    sample_jsonl = merged_data



    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        if test_groundtruth:
            print("Testing ground truth...")
            for sample in tqdm(problems.values()):
                task_id = sample["task_id"]
                lang = task_id.split("/")[0].lower()
                if lang == "javascript":
                    lang = "js"
                tmp_dir_ = os.path.join(tmp_dir, lang, "evaluation")
                sample["generation"] = sample["canonical_solution"]
                sample["test_code"] = process_humaneval_test(sample, problems, example_test, language)
                if sample["test_code"] is None:
                    continue
                args = (task_id, sample, lang, timeout, tmp_dir_, completion_id[task_id])
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1
        else:
            print("Reading samples...")
            for sample in tqdm(sample_jsonl):
                task_id = sample["task_id"]
                if not is_mbpp:
                    lang = language
                if not is_mbpp and lang == "javascript":
                    lang = "js"
                if is_mbpp:
                    lang = "python"
                tmp_dir_ = os.path.join(tmp_dir, lang, "evaluation")
                sample["task_id"] = task_id
                sample["test_code"] = process_humaneval_test(sample, problems, example_test, is_mbpp, language)
                if sample["test_code"] is None:
                    continue
                if "completion_id" in sample:
                    completion_id_ = sample["completion_id"]
                else:
                    completion_id_ = completion_id[task_id]
                args = (task_id, sample, lang, timeout, tmp_dir_, completion_id_)
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1

        if len(completion_id) == len(problems):
            evaluate_pass_at_k = True
        else:
            evaluate_pass_at_k = False

        print("Running test suites...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)
    if evaluate_pass_at_k:
        ks = k
        pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                     for k in ks if (total >= k).all()}
        print(pass_at_k)
    else:
        print("Total:", np.sum(total))
        print("Correct:", np.sum(correct))
    return pass_at_k


def remove_last_brace(text):
    last_index = text.rfind("}")  # 找到最后一个 } 的索引
    if last_index != -1:
        return text[:last_index] + text[last_index + 1:]  # 去掉该位置的 }
    return text  # 如果没有 }，返回原字符串

def remove_first_brace(text):
    first_index = text.find("}")  # 找到第一个 } 的索引
    if first_index != -1:
        return text[:first_index] + text[first_index + 1:]  # 去掉该位置的 }
    return text  # 如果没有 }，返回原字符串


def main(org_file = '/home/baoxuanlin/graduation/ssf/DeepSeek-Coder/Evaluation/HumanEval/data/humaneval-java.jsonl',
         generate_file='/home/baoxuanlin/graduation/ssf/DeepSeek-Coder/Evaluation/HumanEval/output/oss-instruction/2_human_eval_1024_25_temp1_tp1_0_dofil0_filter.jsonl',
        tmp_dir='./',
        n_workers=1,
        timeout=8.0,
        problem_file='/home/baoxuanlin/graduation/ssf/DeepSeek-Coder/Evaluation/HumanEval/data/humaneval-java.jsonl',
        language='java'
        ):
    
    ck_its = [i for i in range(7,8)]
    learn_type = 'oss_evol_instruct'
    search_reference = 1
    # ck_its = [21]
    for ck in ck_its:
        generate_file = f'/home/baoxuanlin/graduation/ssf/DeepSeek-Coder/Evaluation/HumanEval/output/{learn_type}/{ck}_human_eval_1024_25_temp1_tp1_0_dofil0_filter_ref{search_reference}.jsonl'
        result = evaluate_functional_correctness(
        org_file = org_file,
        generate_file=generate_file,
        tmp_dir=tmp_dir,
        n_workers=n_workers,
        timeout=timeout,
        problem_file=problem_file,
        language=language
        )
        print(f'ck_its{ck}:{result}')

if __name__=='__main__':
    fire.Fire(main)

