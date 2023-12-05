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

# function to build regex pattern using 2 lists of different priority
def build_priority_regex_pattern(priority_list, normal_list):
    '''
    priority_list has higher priority than normal_list
    '''
    priority_list = sorted(priority_list, key=len, reverse=True) # sort list by length, longest first
    if len(priority_list) == 0: # if priority_list is empty, add a dummy token to avoid error
        priority_list.append('unk_token')
    priority_list_compound = [f".*({word})" for word in priority_list] # wrap each word with .* to match any character before the word
    priority_list_compound = '|'.join(priority_list_compound)

    normal_list = sorted(normal_list, key=len, reverse=True) # sort list by length, longest first
    if len(normal_list) == 0: # if normal_list is empty, add a dummy token to avoid error
        normal_list.append('unk_token')
    normal_list_compound = [f"{word}" for word in normal_list]
    normal_list_compound = "|".join(normal_list_compound)

    pattern = rf"(?<=\b){priority_list_compound}(?=\b)|(?<=\b)({normal_list_compound})(?=\b)"
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

def mapping_with_priority(string, keywords, pattern):
    string = string.lower()
    match = pattern.findall(string)
    if match:
        match = [m for m in match[0] if m != ''][0]
        return keywords[match]
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

def rule_predict_batch_with_priority(input: List, batch_size, keywords, pattern):
    output = []
    for i in tqdm(range(0, len(input), batch_size)):
        input_batch = input[i:i + batch_size]
        input_batch = [str(x).lower() for x in input_batch]
        output_batch = [mapping_with_priority(x, keywords, pattern) for x in input_batch]
        output.extend(output_batch)
    return output


if __name__ == '__main__':
    jl = read_json("main/rule/data-bin/keyword_lv1.json")
    reversed_jl = reverse_dict(jl)
    pattern = build_regex(reversed_jl)
    test = ["sữa đặc", "sữa chua vinamilk không đường", "sữa rửa mặt nivea", "thuốc lá thăng long", "srm adg"]
    for product in test:
        print(mapping(product, reversed_jl, pattern))