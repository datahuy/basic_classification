from main.classifier.base_classifier import BaseClassifier
from main.classifier.FMCG_l1_classifier import FMCGl1Classifier
from main.rule.rule_mapping import read_json, rule_predict_batch, reverse_dict, build_regex, remove_accent
from main.utils.preprocess_text.preproces_industry_cls import clean_text
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
        l2_rule_json_path, l1_priority_rule_json_path,  l1_normal_rule_json_path,
        l2_category_json_path, l1_category_json_path, 
        l1l2_mapping_json_path,
        batch_size=128, 
        default_class='Không xác định'):

        self.l2_model = FMCGl2ModelClassifier(l2_model_path, batch_size, default_class)
        self.l1_clf = FMCGl1Classifier(
            model_path=l1_model_path, 
            category_json_path=l1_category_json_path,
            batch_size=batch_size,
            default_class=default_class,
            priority_rule_json_path=l1_priority_rule_json_path,
            normal_rule_json_path=l1_normal_rule_json_path
            )
        self.rule = FMCGl2RuleClassifier(l2_rule_json_path, batch_size)
        self.category = FMCGl2RuleClassifier(l2_category_json_path, batch_size)
        self.l1l2_mapping = read_json(l1l2_mapping_json_path)

    @staticmethod
    def _process_unknown_milk_product(product_name):
        '''
        This function is to determine if a milk product is 'Sữa bột' or 'Sữa tươi' or 'Sữa và sản phẩm từ sữa khác' using regex
        'Sữa bột' products contains '[number]g' or '[number] g or [number]gr' or '[number] gr' in their name
        'Sữa tươi' products contains '[number]ml' or '[number] ml' in their name
        'Sữa và sản phẩm từ sữa khác' products do not contain any of the above        
        '''
        import re
        product_name = product_name.lower()
        if re.search(r'\b\d+g\b', product_name) or re.search(r'\b\d+ g\b', product_name) or re.search(r'\b\d+gr\b', product_name) or re.search(r'\b\d+ gr\b', product_name):
            return 'Sữa bột'
        if re.search(r'\b\d+ml\b', product_name) or re.search(r'\b\d+ ml\b', product_name):
            return 'Sữa tươi'
        return 'Sữa và sản phẩm từ sữa khác'

        
    def predict(self, name_input:List, self_category_input, model_l1_threshold=0.9, model_l2_threshold=0.9) -> Tuple[List, List, List]:
        '''
        for l1: use l1 classifier
        for l2:
            get result of rule classifier on name_input and their index
            for those whose rule result is 'không xác định', get result of model classifier on name_input
            for those whose result is still 'không xác định', get result of category classifier on self_category_input
            merge the result back by their index
        '''

        # get l1 result
        l1_preds, _ = self.l1_clf.predict(name_input, self_category_input, model_l1_threshold)

        # concat self_category_input with name_input
        name_input_with_category = [f"{name} {category}" for name, category in zip(name_input, self_category_input)]

        # run rule classifier on name_input_with_category
        l2_preds = self.rule.predict(name_input_with_category)
        pred_method = ['rule'] * len(l2_preds)

        # run category classifier on self_category_input
        index_to_run_category = [i for i, x in enumerate(l2_preds) if x == 'Không xác định']
        if len(index_to_run_category) > 0:
            category_preds = self.category.predict([self_category_input[i] for i in index_to_run_category])
            for i, pred in zip(index_to_run_category, category_preds):
                l2_preds[i] = pred
                pred_method[i] = 'category'

        # for those whose l1 result is "Sữa và sản phẩm từ sữa", if their l2 result is 'Không xác định' or 'Sữa và sản phẩm từ sữa khác', \
        # use regex to determine if it is 'Sữa bột' or 'Sữa tươi'
        for i, pred in enumerate(l2_preds):
            if l1_preds[i] == 'Sữa và sản phẩm từ sữa' and (pred == 'Không xác định' or pred == 'Sữa và sản phẩm từ sữa khác'):
                l2_preds[i] = self._process_unknown_milk_product(name_input[i])
                pred_method[i] = 'rule_2'

        # run model classifier on name_input
        index_to_run_model = [i for i, x in enumerate(l2_preds) if x == 'Không xác định']
        l2_model_preds = self.l2_model.predict([name_input[i] for i in index_to_run_model], threshold=model_l2_threshold)[0]
        for i, pred in zip(index_to_run_model, l2_model_preds):
            l2_preds[i] = pred
            pred_method[i] = 'model'

       
        # set method to 'Không xác định' for those whose result is 'Không xác định'
        for i, pred in enumerate(l2_preds):
            if pred == 'Không xác định':
                pred_method[i] = 'Không xác định'

        # ensure consistency between l1 and l2 by using l1l2_mapping
        for i, l1_pred in enumerate(l1_preds):
            if l2_preds[i] not in self.l1l2_mapping[l1_pred]['available_l2']:
                l2_preds[i] = self.l1l2_mapping[l1_pred]['default_l2']
                pred_method[i] = 'rule_0'

        return l1_preds, l2_preds, pred_method