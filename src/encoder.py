import numpy as np

class Encoder(object):
    
    def __init__(self):
        self.num_labels = 88
        pass
    
    def onehot_encode(self, label, multilabel=True):
        encoded_label = np.zeros(self.num_labels, dtype=np.int8)
        if multilabel:
            for i in range(len(label)):
                encoded_label[int(label[i][1:]) - 1 + (44 if label[i][0] == 'J' else 0)] = 1
                
        return encoded_label