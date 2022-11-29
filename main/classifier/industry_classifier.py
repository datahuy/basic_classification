from main.classifier.base_classifier import BaseClassifier
from main.utils.preprocess_text.preproces_industry_cls import clean_text

class IndustryClassifier(BaseClassifier):
    def __init__(self, model_path, batch_size):
        super(IndustryClassifier, self).__init__(model_path, batch_size)

    def preprocess(self, input):
        input = list(map(clean_text, input))

        return input