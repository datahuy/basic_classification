from main.utils.model_utils import load_model

class BaseClassifier():
    def __init__(self, model_path, threshold):
        self.model = load_model(model_path)
        self.threshold = threshold

    def preprocess(self, input):
        return input

    def postprocess(self, preds, probs):
        output = [preds[i] if probs[i]>self.threshold else self.model.index2label[0] for i in range(len(preds))]
        return output

    def predict(self, input):
        input = self.preprocess(input)
        preds, probs = self.model.predict(input)
        output = self.postprocess(preds, probs)

        return output, probs

