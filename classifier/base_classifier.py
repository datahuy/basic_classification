class BaseClassifier():
    def __init__(self):
        self.model = None

    def preprocess(self):
        pass

    def postprocess(self):
        pass

    def predict(self):
        self.preprocess()
        self.model.predict()
        self.postprocess()

