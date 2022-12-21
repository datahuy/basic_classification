from typing import List
from main.rule.rule_mapping import rule_predict_batch


def merge_output(model_output: List, rule_output: List):
    # if rule_output is unk, use model output, else use rule output
    merged = []
    for model, rule in zip(model_output, rule_output):
        if rule == "unk":
            merged.append(model)
        else:
            merged.append(rule)
    return merged