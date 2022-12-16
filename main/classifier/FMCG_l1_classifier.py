from main.classifier.base_classifier import BaseClassifier
from main.rule.rule_mapping import read_json, rule_predict_batch, reverse_dict, build_regex
import logging
import sys 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - {%(pathname)s:%(lineno)d} - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)



class FMCGl1Classifier(BaseClassifier):
    def __init__(self, model_path, batch_size=128):
        name = 'FMCG l1 Classifier'
        super(FMCGl1Classifier, self).__init__(model_path, batch_size, name)

    def predict(self, input):
        preds, probs = self.model.predict(input, batch_size=self.batch_size)
        #output = self.postprocess(preds, probs, threshold)
        return preds


class FMCGl1RuleClassifier():
    def __init__(self, json_path, batch_size=128):
        self.keywords = reverse_dict(read_json(json_path))
        self.pattern = build_regex(self.keywords)
        self.batch_size = batch_size
        logging.info('Finish loading FMCG l1 rule classifier')

    def predict(self, input):
        output = rule_predict_batch(input, keywords=self.keywords, pattern=self.pattern, batch_size=self.batch_size)
        return output