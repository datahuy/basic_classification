from main.classifier.base_classifier import BaseClassifier
from main.classifier.FMCG_l1_classifier import FMCGl1Classifier
from main.rule.rule_mapping import read_json, rule_predict_batch, reverse_dict, build_regex, remove_accent
from main.utils.preprocess_text.preproces_industry_cls import clean_text
import logging
import sys 
from typing import List, Tuple

class FMCGl2ModelClassifier(BaseClassifier):
    def __init__(self, model_path, batch_size=128, default_class='Không xác định'):
        name = 'FMCG l2 Classifier'
        super(FMCGl2ModelClassifier, self).__init__(model_path, batch_size, name)
        self.default_class = default_class

    def preprocess(self, input):
        input = list(map(clean_text, input))
        return input

    def postprocess(self, preds, probs, threshold):
        output = [preds[i] if probs[i] > threshold else self.default_class for i in range(len(preds))]
        return output


class FMCGl2RuleClassifier():
    def __init__(self, json_path, batch_size=128):
        self.keywords = read_json(json_path)
        logging.info(f'Finish loading {len(self.keywords)} categories!')
        logging.info(f'First 10 categories: {list(self.keywords.keys())[:10]}')

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

    def predict(self, input):
        output = rule_predict_batch(input, keywords=self.keywords, pattern=self.pattern, batch_size=self.batch_size)
        return output


class FMCGl2Classifier():
    def __init__(self, 
        l2_model_path, l1_model_path, 
        l2_rule_json_path, l1_rule_json_path, 
        l2_category_json_path, l1_category_json_path, 
        l1l2_mapping_json_path,
        batch_size=128, 
        default_class='Không xác định'):

        self.l2_model = FMCGl2ModelClassifier(l2_model_path, batch_size, default_class)
        self.l1_model = FMCGl1Classifier(
            model_path=l1_model_path, 
            rule_json_path=l1_rule_json_path,
            category_json_path=l1_category_json_path,
            batch_size=batch_size,
            default_class=default_class
            )
        self.rule = FMCGl2RuleClassifier(l2_rule_json_path, batch_size)
        self.category = FMCGl2RuleClassifier(l2_category_json_path, batch_size)
        self.l1l2_mapping = read_json(l1l2_mapping_json_path)

    @staticmethod
    def _process_unknown_milk_product(product_name):
        '''
        This function is to determine if a milk product is 'Sữa bột' or 'Sữa tươi' or 'Sữa và sản phẩm từ sữa khác' using regex
        'Sữa bột' products contains '[number]g' or '[number] g' in their name
        'Sữa tươi' products contains '[number]ml' or '[number] ml' in their name
        'Sữa và sản phẩm từ sữa khác' products do not contain any of the above        
        '''
        import re
        if re.search(r'\b\d+g\b', product_name) or re.search(r'\b\d+ g\b', product_name):
            return 'Sữa bột'
        if re.search(r'\b\d+ml\b', product_name) or re.search(r'\b\d+ ml\b', product_name):
            return 'Sữa tươi'
        return 'Sữa và sản phẩm từ sữa khác'

        
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
        logging.info(f"rule_preds: {rule_preds}")

        # run category classifier on self_category_input
        index_to_run_category = [i for i, x in enumerate(rule_preds) if x == 'Không xác định']
        if len(index_to_run_category) > 0:
            category_preds = self.category.predict([self_category_input[i] for i in index_to_run_category])
            for i, pred in zip(index_to_run_category, category_preds):
                rule_preds[i] = pred
                pred_method[i] = 'category'
        logging.info(f"rule_preds after category: {rule_preds}")

        # for those whose result is 'Sữa và sản phẩm từ sữa khác', use regex to determine if it is 'Sữa bột' or 'Sữa tươi'
        for i, pred in enumerate(rule_preds):
            if pred == 'Sữa và sản phẩm từ sữa khác':
                rule_preds[i] = self._process_unknown_milk_product(name_input[i])
                pred_method[i] = 'rule'
        logging.info(f"rule_preds after regex: {rule_preds}")

        # run model classifier on name_input
        # for each name_input, get l1_prediction and l2_prediction
        # if l2_prediction belongs to l1l2_mapping[l1_prediction].available_l2, then use l2_prediction
        # otherwise, use use l1l2_mapping[l1_prediction].default_l2
        index_to_run_model = [i for i, x in enumerate(rule_preds) if x == 'Không xác định']
        if len(index_to_run_model) > 0:
            l2_model_preds = self.l2_model.predict([name_input[i] for i in index_to_run_model], threshold=model_threshold)[0]
            l1_model_preds = self.l1_model.predict(
                name_input=[name_input[i] for i in index_to_run_model],
                self_category_input=[self_category_input[i] for i in index_to_run_model],
                model_threshold=model_threshold)[0]
            model_preds = []
            for l1_pred, l2_pred in zip(l1_model_preds, l2_model_preds):
                if l2_pred in self.l1l2_mapping[l1_pred]['available_l2']:
                    model_preds.append(l2_pred)
                else:
                    model_preds.append(self.l1l2_mapping[l1_pred]['default_l2'])

            for i, pred in zip(index_to_run_model, model_preds):
                rule_preds[i] = pred
                pred_method[i] = 'model'
        logging.info(f"rule_preds after model: {rule_preds}")

        # set method to 'Không xác định' for those whose result is 'Không xác định'
        for i, pred in enumerate(rule_preds):
            if pred == 'Không xác định':
                pred_method[i] = 'Không xác định'

        return rule_preds, pred_method