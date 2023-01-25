import numpy as np

class MetricEvaluation():
    def __init__(self, cm):
        super(MetricEvaluation, self).__init__()
        self.cm = cm
        
    def calculate(self):
        total = np.sum(self.cm)
        HDPE_total = np.sum(self.cm[0])
        LDPE_total = np.sum(self.cm[1])
        PET_total = np.sum(self.cm[2])
        PP_total = np.sum(self.cm[3])

        HDPE_TP = self.cm[0][0]
        HDPE_FP = self.cm[1][0] + self.cm[2][0] + self.cm[3][0]
        HDPE_FN = self.cm[0][1] + self.cm[0][2] + self.cm[0][3]

        LDPE_TP = self.cm[1][1]
        LDPE_FP = self.cm[0][1] + self.cm[2][1] + self.cm[3][1]
        LDPE_FN = self.cm[1][0] + self.cm[1][2] + self.cm[1][3]

        PET_TP = self.cm[2][2]
        PET_FP = self.cm[0][2] + self.cm[1][2] + self.cm[3][2]
        PET_FN = self.cm[2][0] + self.cm[2][1] + self.cm[2][3]
        
        PP_TP = self.cm[3][3]
        PP_FP = self.cm[0][3] + self.cm[1][3] + self.cm[2][3]
        PP_FN = self.cm[3][0] + self.cm[3][1] + self.cm[3][2]
        
        HDPE_precision = HDPE_TP/(HDPE_TP+HDPE_FP)
        HDPE_recall = HDPE_TP/HDPE_total
        
        LDPE_precision = LDPE_TP/(LDPE_TP+LDPE_FP)
        LDPE_recall = LDPE_TP/LDPE_total
        
        PET_precision = PET_TP/(PET_TP+PET_FP)
        PET_recall = PET_TP/PET_total
        
        PP_precision = PP_TP/(PP_TP+PP_FP)
        PP_recall = PP_TP/PP_total
        
        accuracy = (HDPE_TP+LDPE_TP+PET_TP+PP_TP)/total
        precision = (HDPE_precision*HDPE_total + LDPE_precision*LDPE_total + PET_precision*PET_total + PP_precision*PP_total)/total
        recall = (HDPE_recall*HDPE_total + LDPE_recall*LDPE_total + PET_recall*PET_total + PP_recall*PP_total)/total
        
        return accuracy, precision, recall