from main.classifier.base_classifier import BaseClassifier
from main.rule.rule_mapping import read_json, rule_predict_batch, reverse_dict, build_regex, remove_accent, build_priority_regex_pattern, rule_predict_batch_with_priority
from main.utils.preprocess_text.preproces_industry_cls import clean_text
import logging
from typing import List, Tuple

class FMCGl1ModelClassifier(BaseClassifier):
    def __init__(self, model_path, batch_size=128, default_class='Không xác định'):
        name = 'FMCG l1 Classifier'
        super(FMCGl1ModelClassifier, self).__init__(model_path, batch_size, name)
        self.default_class = default_class

    def preprocess(self, input):
        input = list(map(clean_text, input))
        return input

    def postprocess(self, preds, probs, threshold):
        output = [preds[i] if probs[i] > threshold else self.default_class for i in range(len(preds))]
        return output


class FMCGl1RuleClassifier():
    def __init__(self, json_path, batch_size=128):
        self.keywords = read_json(json_path)

        # for each category, we have a list of keywords. Now remove accent of the keywords (for those that have length >= 6) and add them to the dict
        for category, keywords in self.keywords.items():
            keywords_no_accent = []
            for keyword in keywords:
                if len(keyword) >= 6:
                    keywords_no_accent.append(remove_accent(keyword))
            self.keywords[category] += keywords_no_accent

        self.keywords = reverse_dict(self.keywords)
        self.pattern = build_regex(self.keywords)
        self.batch_size = batch_size
        logging.info('Finish loading FMCG l1 rule classifier!')

    def predict(self, input):
        output = rule_predict_batch(input, keywords=self.keywords, pattern=self.pattern, batch_size=self.batch_size)
        return output

class FMCGl1RuleClassiferWithPriority():
    def __init__(self, priority_json_path, normal_json_path, batch_size=128):
        self.priority_keywords = read_json(priority_json_path)
        self.normal_keywords = read_json(normal_json_path)

        # for each category, we have a list of keywords. Now remove accent of the keywords (for those that have length >= 6) and add them to the dict
        for category, keywords in self.priority_keywords.items():
            keywords_no_accent = []
            for keyword in keywords:
                if len(keyword) >= 6:
                    keywords_no_accent.append(remove_accent(keyword))
            self.priority_keywords[category] += keywords_no_accent

        for category, keywords in self.normal_keywords.items():
            keywords_no_accent = []
            for keyword in keywords:
                if len(keyword) >= 6:
                    keywords_no_accent.append(remove_accent(keyword))
            self.normal_keywords[category] += keywords_no_accent
        
        self.priority_keywords = reverse_dict(self.priority_keywords)
        self.normal_keywords = reverse_dict(self.normal_keywords)
        self.pattern = build_priority_regex_pattern(priority_list=self.priority_keywords, normal_list=self.normal_keywords)
        self.batch_size = batch_size
        logging.info('Finish loading FMCG l1 rule classifier with priority!')

    def predict(self, input):
        output = rule_predict_batch_with_priority(input, batch_size=self.batch_size, keywords=self.normal_keywords, pattern=self.pattern)
        return output



class FMCGl1Classifier():
    def __init__(self, model_path, priority_rule_json_path, normal_rule_json_path, category_json_path, batch_size=128, default_class='Không xác định'):
        self.model = FMCGl1ModelClassifier(model_path, batch_size, default_class)
        # self.rule = FMCGl1RuleClassifier(rule_json_path, batch_size)
        self.rule = FMCGl1RuleClassiferWithPriority(priority_rule_json_path, normal_rule_json_path, batch_size)
        self.category = FMCGl1RuleClassifier(category_json_path, batch_size)
        logging.info('Finish loading FMCG l1 classifier!')

    def predict(self, name_input:List, self_category_input, model_threshold=0.9) -> Tuple[List, List]:
        '''
        get result of rule classifier on name_input and their index
        for those whose rule result is 'không xác định', get result of model classifier on name_input
        for those whose result is still 'không xác định', get result of category classifier on self_category_input
        merge the result back by their index
        '''
        # concat self_category_input with name_input
        name_input_with_category = [f"{name} {category}" for name, category in zip(name_input, self_category_input)]

        # run rule classifier on name_input_with_category
        rule_preds = self.rule.predict(name_input_with_category)
        pred_method = ['rule'] * len(rule_preds)

        # run category classifier on self_category_input
        index_to_run_category = [i for i, x in enumerate(rule_preds) if x == 'Không xác định']
        if len(index_to_run_category) > 0:
            category_preds = self.category.predict([self_category_input[i] for i in index_to_run_category])
            for i, pred in zip(index_to_run_category, category_preds):
                rule_preds[i] = pred
                pred_method[i] = 'category'
        
        # run model classifier on name_input
        index_to_run_model = [i for i, x in enumerate(rule_preds) if x == 'Không xác định']
        if len(index_to_run_model) > 0:
            model_preds = self.model.predict([name_input[i] for i in index_to_run_model], threshold=model_threshold)[0]
            for i, pred in zip(index_to_run_model, model_preds):
                rule_preds[i] = pred
                pred_method[i] = 'model'


        # set method to 'Không xác định' for those whose result is 'Không xác định'
        for i, pred in enumerate(rule_preds):
            if pred == 'Không xác định':
                pred_method[i] = 'Không xác định'

        return rule_preds, pred_method