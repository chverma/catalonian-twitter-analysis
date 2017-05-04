from models.StatModel import StatModel
import utils.defaults as defaults
import cv2
import numpy
class SVM(StatModel):
    def __init__(self, C=None, gamma=None):
        self.model = cv2.ml.SVM_create()
        self.model.setKernel(cv2.ml.SVM_LINEAR)
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setC(2.67)
        self.model.setGamma(5.383)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        # SVM.predict(samples)[1].ravel()
        return numpy.float32( [self.model.predict(s) for s in samples]) #last
    def evaluate(self, samples, labels,resp=None):
        if resp==None:
            resp =  numpy.float32( [self.model.predict(s) for s in samples])
        #resp = self.model.predict(samples)
        err = (labels != resp).mean()
        print('error: %.2f %%') % (err*100)

        confusion = numpy.zeros((defaults.CLASS_N, defaults.CLASS_N), numpy.int32)
        for i, j in zip(labels, resp):
            confusion[i, j] += 1
        print('confusion matrix:')
        print (confusion)
        print()
        return( confusion)
