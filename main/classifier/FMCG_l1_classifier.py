from main.classifier.base_classifier import BaseClassifier


class FMCGl1Classifier(BaseClassifier):
    def __init__(self, model_path, batch_size=128):
        name = 'FMCG l1 Classifier'
        super(FMCGl1Classifier, self).__init__(model_path, batch_size, name)

    def predict(self, input):
        preds, probs = self.model.predict(input, batch_size=self.batch_size)
        #output = self.postprocess(preds, probs, threshold)
        return preds