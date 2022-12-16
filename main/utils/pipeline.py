import pandas as pd
from typing import List
from main.rule.rule_mapping import rule_predict_batch


def merge_output(model_output: List, rule_output: List):
    indexes = [idx for idx, j in enumerate(rule_output) if j != "unk"]
    for (index, r_out) in zip(indexes, rule_output):
        model_output[index] = r_out
    return model_output


"""def pipeline(input: List, keywords: List, batch_size: int):
    rule_output = rule_predict_batch(input, batch_size, keywords)
    model_output = model.predict(input, batch_size)
    merged_output = merge_output(model_output, rule_output)
    return merged_output"""
