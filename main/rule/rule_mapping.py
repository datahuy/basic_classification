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


def mapping(string, keywords):
    '''
    string: input string
    keywords: dictionary in the form of {keyword: category}

    this function will build a regex pattern from the list of keywords, then match the string with the pattern.
    The regex pattern is from all category.
    The regex is sorted by the length of the keyword in descending order.
    If there is a match, the function will return the category of the string.
    '''
    string = string.lower()
    keyword_list = sorted(keywords.keys(), key=lambda x: len(x), reverse=True)
    pattern = f'(?<=\\b)({"|".join(keyword_list)})(?=\\b)'
    pattern = re.compile(pattern)
    match = pattern.search(string)
    if match:
        return keywords[match.group()]
    else:
        return 'unk'
        

def rule_predict_batch(input: List, batch_size, keywords):
    output = []
    for i in tqdm(range(0, len(input), batch_size)):
        input_batch = input[i:i + batch_size]
        input_batch = [x.lower() for x in input_batch]
        output_batch = [mapping(x, keywords) for x in input_batch]
        output.extend(output_batch)
    return output


if __name__ == '__main__':
    jl = read_json("main/rule/data-bin/keyword_lv1.json")
    reversed_jl = reverse_dict(jl)
    test = ["gạo nếp 100kg", "gạo tẻ"]
    for product in test:
        print(mapping(product, reversed_jl))