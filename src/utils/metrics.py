from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
def basic_scores(y_true, y_prob, threshold=0.5):
    import numpy as np
    y_pred = (y_prob >= threshold).astype(int)
    p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {'precision':p,'recall':r,'f1':f1,'roc_auc':roc_auc_score(y_true,y_prob),'pr_auc':average_precision_score(y_true,y_prob)}
