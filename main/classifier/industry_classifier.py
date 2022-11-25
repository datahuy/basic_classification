from main.classifier.base_classifier import BaseClassifier
from main.utils.preprocess_text.preproces_industry_cls import clean_text

class IndustryClassifier(BaseClassifier):
    def __init__(self, model_path, threshold=0.5):
        super(IndustryClassifier, self).__init__(model_path, threshold)

    def preprocess(self, input):
        input = list(map(clean_text, input))

        return input