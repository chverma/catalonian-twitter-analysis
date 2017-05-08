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

    def load(self, fn):
        self.model = cv2.ml.SVM_load(fn)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()

    def evaluate(self, samples, labels, resp=None):
        if resp is None:
            resp = self.predict(samples)

        err = (labels != resp).mean()
        print('error:', (err*100))

        confusion = numpy.zeros((defaults.CLASS_N, defaults.CLASS_N), numpy.float32)
        for i, j in zip(labels, numpy.int8(resp)):
            confusion[i, j] += 1.

        print('confusion matrix:')
        print (confusion)
        print()
        precision = [0.]*defaults.CLASS_N
        recall = [0.]*defaults.CLASS_N
        f1 = [0.]*defaults.CLASS_N
        print("class\tprec\trecall\tf1-score")
        for i in range(defaults.CLASS_N):
            precision[i] = confusion[i, i] / sum(confusion[i, :])
            recall[i] = confusion[i, i] / sum(confusion[:, i])
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            print("%i\t%.3f\t%.3f\t%.3f" % (i, precision[i], recall[i], f1[i]))
        print("avg\t%.3f\t%.3f\t%.3f" % (sum(precision[i] for i in [0, 2])/2, sum(recall[i] for i in [0, 2])/2, sum(f1[i] for i in [0, 2])/2))
