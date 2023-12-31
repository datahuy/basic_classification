from main.utils.model_utils import load_model
import logging


class BaseClassifier():
    def __init__(self, model_path, batch_size, name):
        logging.info(f"Finish loading {name}!")
        self.model = load_model(model_path)
        self.model.encoder.eval()
        self.batch_size = batch_size

    def preprocess(self, input):
        return input

    def postprocess(self, preds, probs, threshold):
        output = [preds[i] if probs[i] > threshold else self.model.index2label[0] for i in range(len(preds))]
        return output

    def predict(self, input, threshold):
        input = self.preprocess(input)
        preds, probs = self.model.predict(input, self.batch_size)
        output = self.postprocess(preds, probs, threshold)
        return output, probs

    def encode(self, input):
        '''
        to extract the penultimate layer of the model
        '''
        input = self.preprocess(input)
        return self.model.encode(input, self.batch_size)
