''' 
    Author: Vu Quoc Hien
    Date created: 2022-03-14
'''
import matplotlib.pyplot as plt
from matplotlib.style import available
import torch as T
import numpy as np
import sklearn.metrics as metrics
import seaborn as sns
import os


class Evaluator(object):
    
    def __init__(self, model=None) -> None:
        self.model = model
        pass
    
    def precision(self, y_preds, y_trues):
        """
        Calculates the precision
        """
        return metrics.precision_score(y_trues, y_preds, average='macro')

    def recall(self, y_preds, y_trues):
        """
        Calculates the recall
        """
        return metrics.recall_score(y_trues, y_preds, average='macro')

    def accuracy(self, y_preds, y_trues):
        """
        Calculates the accuracy
        """
        return metrics.accuracy_score(y_trues, y_preds)

    def f1(self, y_preds, y_trues):
        """
        Calculates the accuracy
        """
        return metrics.f1_score(y_trues, y_preds)

    def roc_auc(self, y_preds, y_trues):
        """
        Calculates the area under the ROC curve for multi class classification
        """
        return metrics.roc_auc_score(y_trues, y_preds, average='macro')

    def confusion_matrix(self, y_preds, y_trues):
        return metrics.confusion_matrix(y_trues, y_preds)

    '''
    *** Multi-class classification ***
    Input sample:
    y_true = np.array( [[0,1,0],
                        [0,1,1],
                        [1,0,1],
                        [0,0,1]])

    y_pred = np.array( [[0,1,1],
                        [0,1,1],
                        [0,1,0],
                        [0,0,0]])
                        
    Exact Match Ratio, EMR = 1/n * sum(y_true == y_pred)
    One-Zero Loss, OZL = 1/n * sum(y_true != y_pred)
    Accuracy = 1/n * sum(|y_true * y_pred| / |y_true + y_pred|)
    Hamming Loss, HL = 1/nL * sum(y_true != y_pred)
    Precision = 1/n * sum(|y_true * y_pred| / |y_pred|)
    '''
    
    def multilabel_EMR(self, y_preds, y_trues):
        """
        Calculates the exact match ratio
        """
        return 1/len(y_trues) * np.sum(y_trues == y_preds)
    
    def multilabel_one_zero_loss(self, y_preds, y_trues):
        """
        Calculates the one-zero loss
        """
        return 1/len(y_trues) * np.sum(y_trues != y_preds)
    
    def multilabel_accuracy(self, y_preds, y_trues):
        """
        Calculates the accuracy
        """
        temp = 0
        for i in range(y_trues.shape[0]):
            temp += sum(np.logical_and(y_trues[i], y_preds[i])) / sum(np.logical_or(y_trues[i], y_preds[i]))
        return temp / y_trues.shape[0]
    
    def multilabel_hamming_loss(self, y_trues, y_preds):
        '''
        Calculates the Hamming Loss
        '''
        temp=0
        for i in range(y_trues.shape[0]):
            temp += np.size(y_trues[i] == y_preds[i]) - np.count_nonzero(y_trues[i] == y_preds[i])
        return temp/(y_trues.shape[0] * y_trues.shape[1])
    
    def multilabel_precision(self, y_trues, y_preds):
        '''
        Calculates the precision
        '''
        temp = 0
        for i in range(y_trues.shape[0]):
            if sum(y_trues[i]) == 0:
                continue
            temp+= sum(np.logical_and(y_trues[i], y_preds[i]))/ sum(y_trues[i])
        return temp/ y_trues.shape[0]

    def multilabel_recall(self, y_trues, y_preds):
        '''
        Calculates the recall
        '''
        temp = 0
        for i in range(y_trues.shape[0]):
            if sum(y_trues[i]) == 0:
                continue
            temp+= sum(np.logical_and(y_trues[i], y_preds[i]))/ sum(y_preds[i])
        return temp/ y_trues.shape[0]
    
    def multilabel_f1_score(self, y_trues, y_preds):
        '''
        Calculates the f1 score
        '''
        temp = 0
        for i in range(y_trues.shape[0]):
            if sum(y_trues[i]) == 0:
                continue
            temp+= 2 * sum(np.logical_and(y_trues[i], y_preds[i]))/ (sum(y_trues[i]) + sum(y_preds[i]))
        return temp/ y_trues.shape[0]
    
    def multilabel_auc(self, y_preds, y_trues):
        """
        Calculates the area under the ROC curve for multi class classification
        """
        return metrics.roc_auc_score(y_trues, y_preds, average='macro')
    
    def multilabel_r2_score(self, y_trues, y_preds):
        '''
        Calculates the r2 score
        '''
        return metrics.r2_score(y_trues, y_preds, multioutput='uniform_average')
    
    def multilabel_confusion_matrix(self, y_trues, y_preds):
        '''
        Calculates the confusion matrix
        '''
        return metrics.multilabel_confusion_matrix(y_trues, y_preds)
    
    def get_all_metrics(self, y_preds, y_trues, metric_names='all', use_multilabel=False):
        metrics = {}
        if use_multilabel:
            if 'all' in metric_names:
                metrics['EMR'] = self.multilabel_EMR(y_preds, y_trues)
                metrics['OZL'] = self.multilabel_one_zero_loss(y_preds, y_trues)
                metrics['Accuracy'] = self.multilabel_accuracy(y_preds, y_trues)
                metrics['HL'] = self.multilabel_hamming_loss(y_trues, y_preds)
                metrics['Precision'] = self.multilabel_precision(y_trues, y_preds)
                metrics['Recall'] = self.multilabel_recall(y_trues, y_preds)
                metrics['F1'] = self.multilabel_f1_score(y_trues, y_preds)
                metrics['r2'] = self.multilabel_r2_score(y_trues, y_preds)
            else:
                for metric in metric_names:
                    if metric == 'EMR':
                        metrics['EMR'] = self.multilabel_EMR(y_preds, y_trues)
                    elif metric == 'OZL':
                        metrics['OZL'] = self.multilabel_one_zero_loss(y_preds, y_trues)
                    elif metric == 'Accuracy':
                        metrics['Accuracy'] = self.multilabel_accuracy(y_preds, y_trues)
                    elif metric == 'HL':
                        metrics['HL'] = self.multilabel_hamming_loss(y_trues, y_preds)
                    elif metric == 'Precision':
                        metrics['Precision'] = self.multilabel_precision(y_trues, y_preds)
                    elif metric == 'Recall':
                        metrics['Recall'] = self.multilabel_recall(y_trues, y_preds)
                    elif metric == 'F1':
                        metrics['F1'] = self.multilabel_f1_score(y_trues, y_preds)
                    elif metric == 'AUC':
                        metrics['AUC'] = self.multilabel_auc(y_preds, y_trues)
                    elif metric == 'r2':
                        metrics['r2'] = self.multilabel_r2_score(y_trues, y_preds)
                    elif metric == 'confusion_matrix':
                        metrics['confusion_matrix'] = self.multilabel_confusion_matrix(y_trues, y_preds)
                        sns.heatmap(metrics['confusion_matrix'],cmap='coolwarm',annot=True,)
                        if not os.path.exists('output/evaluation'):
                            os.makedirs('output/evaluation')
                        plt.savefig('output/evaluation/confusion_matrix.png')
                        plt.close()
                        metrics['confusion_matrix'] = None
                    else:
                        raise ValueError('Metric name not recognized')
        else:
            if 'all' in metric_names:
                metrics['precision'] = self.precision(y_preds, y_trues)
                metrics['recall'] = self.recall(y_preds, y_trues)
                metrics['accuracy'] = self.accuracy(y_preds, y_trues)
            else:
                for metric in metric_names:
                    if metric == 'precision':
                        metrics[metric] = self.precision(y_preds, y_trues)
                    elif metric == 'recall':
                        metrics[metric] = self.recall(y_preds, y_trues)
                    elif metric == 'accuracy':
                        metrics[metric] = self.accuracy(y_preds, y_trues)
                    elif metric == 'roc_auc':
                        metrics[metric] = self.roc_auc(y_preds, y_trues)
                    elif metric == 'confusion_matrix':
                        metrics[metric] = self.confusion_matrix(y_preds,y_trues)
                        sns.heatmap(metrics[metric],cmap='coolwarm',annot=True,)
                        if not os.path.exists('output/evaluation'):
                            os.makedirs('output/evaluation')
                        plt.savefig('output/evaluation/confusion_matrix.png')
                        plt.close()
        return metrics
    
    def evaluate(self, dataset: dict,  metric_names='all', use_multilabel=True, accept_threshold=0.51) -> dict:
        y_trues = {}
        y_preds = {}
        self.model._eval()
        metrics = {}
        
        for name in self.model.HLA_layers:	
            y_trues[name] = []
            y_preds[name] = []
            metrics[name] = {}
        
        with T.no_grad():		# Disable gradient calculation
            for _iter, (input, target) in enumerate(zip(dataset[0], dataset[1])):		# Lấy số vòng lặp, data và target lần lượt trong tqdm
                output = self.model(input).flatten(0)
                presize = 0
                for name in self.model.HLA_layers:
                    output_size = self.model.HLA_output_size[name]
                    if use_multilabel is False:
                        allele_out = output[presize:presize + output_size].cpu().numpy().argsort()[-1]
                        allele_target = target[presize:presize + output_size].cpu().numpy().argsort()[-1]
                        y_trues[name].append(allele_target)
                        y_preds[name].append(allele_out)
                    else:
                        allele_targets = target[presize:presize + output_size].cpu().numpy().argsort()[-2:][::-1]
                        allele_outs = output[presize:presize + output_size].cpu().numpy().argsort()[-2:][::-1]
                        if target[presize + allele_targets[1]] == 0:
                            allele_targets[1] = allele_targets[0]
                            if output[presize + allele_outs[1]] < accept_threshold:
                                allele_outs[1] = allele_outs[0]
                        else:
                            if output[presize + allele_outs[1]] < accept_threshold:
                                allele_outs[1] = allele_outs[0]
                        y_true_vec = np.zeros(output_size)
                        y_true_vec[allele_targets[0]] = 1
                        y_true_vec[allele_targets[1]] = 1
                        y_pred_vec = np.zeros(output_size)
                        y_pred_vec[allele_outs[0]] = 1
                        y_pred_vec[allele_outs[1]] = 1
                        y_trues[name].append(y_true_vec)
                        y_preds[name].append(y_pred_vec)
                            
                    presize += output_size
                    
        for name in self.model.HLA_layers:	
            y_true = np.array(y_trues[name])
            y_pred = np.array(y_preds[name])
            metrics[name] = self.get_all_metrics(y_pred, y_true, 
                                                 metric_names=metric_names, 
                                                 use_multilabel=use_multilabel)
        
        return metrics
        