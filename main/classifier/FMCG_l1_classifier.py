from main.classifier.base_classifier import BaseClassifier
from main.rule.rule_mapping import read_json, rule_predict_batch, reverse_dict, build_regex
from main.utils.preprocess_text.preproces_industry_cls import clean_text
import logging
import sys 
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - {%(pathname)s:%(lineno)d} - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


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
        self.keywords = reverse_dict(read_json(json_path))
        self.pattern = build_regex(self.keywords)
        self.batch_size = batch_size
        logging.info('Finish loading FMCG l1 rule classifier!')

    def predict(self, input):
        output = rule_predict_batch(input, keywords=self.keywords, pattern=self.pattern, batch_size=self.batch_size)
        return output


class FMCGl1Classifier():
    def __init__(self, model_path, json_path, batch_size=128, default_class='Không xác định'):
        self.model = FMCGl1ModelClassifier(model_path, batch_size, default_class)
        self.rule = FMCGl1RuleClassifier(json_path, batch_size)
        logging.info('Finish loading FMCG l1 classifier!')

    def predict(self, input:List, model_threshold=0.9) -> List:
        '''
        get result of rule classifier and their index
        for those whose rule result is 'unk', get result of model classifier
        merge the result back by their index
        '''
        rule_preds = self.rule.predict(input)
        index_to_run_model = [i for i, x in enumerate(rule_preds) if x == 'unk']
        if len(index_to_run_model) > 0:
            model_preds = self.model.predict([input[i] for i in index_to_run_model], threshold=model_threshold)[0]
            for i, pred in zip(index_to_run_model, model_preds):
                rule_preds[i] = pred
        return rule_preds

