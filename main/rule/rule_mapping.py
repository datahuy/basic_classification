import json
from typing import List
from tqdm import tqdm
import re


def read_json(js):
    with open(js, 'rb') as f:
        keyword = json.load(f)
    return keyword

def reverse_dict(d):
    reversed_dict = {}
    for key, value in d.items():
        for v in value:
            reversed_dict[v] = key
    return reversed_dict

def build_regex(keywords):
    keyword_list = sorted(keywords.keys(), key=lambda x: len(x), reverse=True)
    pattern = f'(?<=\\b)({"|".join(keyword_list)})(?=\\b)'
    pattern = re.compile(pattern)
    return pattern

def remove_accent(input_str: str) -> str:
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s

def mapping(string, keywords, pattern):
    string = string.lower()
    match = pattern.search(string)
    if match:
        return keywords[match.group()]
    else:
        return 'Không xác định'
        

def rule_predict_batch(input: List, batch_size, keywords, pattern):
    output = []
    for i in tqdm(range(0, len(input), batch_size)):
        input_batch = input[i:i + batch_size]
        input_batch = [str(x).lower() for x in input_batch]
        output_batch = [mapping(x, keywords, pattern) for x in input_batch]
        output.extend(output_batch)
    return output


if __name__ == '__main__':
    jl = read_json("main/rule/data-bin/keyword_lv1.json")
    reversed_jl = reverse_dict(jl)
    pattern = build_regex(reversed_jl)
    test = ["sữa đặc", "sữa chua vinamilk không đường", "sữa rửa mặt nivea", "thuốc lá thăng long", "srm adg"]
    for product in test:
        print(mapping(product, reversed_jl, pattern))