from evaluation.evaluate import evaluate_model

y_true = [1, 0, 1]  # Example true labels
similarity_scores = [0.8, 0.3, 0.7]  # Example similarity scores

metrics = evaluate_model(y_true, similarity_scores)
print("Evaluation Metrics:", metrics)
