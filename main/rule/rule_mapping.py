import json
from typing import List

import pandas as pd
import os
from tqdm import tqdm


def read_json(js):
    f = open(js)
    keyword = json.load(f)
    return keyword


def read_jl(jl_file):
    with open(jl_file, encoding="utf-8") as file:
        data = list(file)
    lst_jl = [json.loads(json_str) for json_str in data]
    return lst_jl


def mapping(string, keywords):
    result = [key for key, value in keywords.items() if any(val in string for val in value)]
    if not result:
        result.append("unk")
    return result[0]


def rule_predict_batch(input: List, batch_size, keywords):
    preds = []
    if len(input) >= batch_size:
        for i in range(0, len(input), batch_size):
            pred = [mapping(j.lower(), keywords=keywords) for j in input[i:batch_size]]
            preds.extend(pred)
    else:
        preds = [mapping(j.lower(), keywords=keywords) for j in input]
    return preds


if __name__ == '__main__':
    jl = read_json("main/rule/data-bin/keyword_lv1.json")
    test = ["gạo nếp 100kg", "gạo tẻ"]
    print(mapping(test[0], jl))