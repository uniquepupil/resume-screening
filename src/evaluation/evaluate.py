from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support

def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return precision, recall, f1

def evaluate_model(y_true, similarity_scores, threshold=0.5):
    y_pred = (similarity_scores >= threshold).astype(int)
    precision, recall, f1 = calculate_metrics(y_true, y_pred)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
