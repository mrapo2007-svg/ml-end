# Summary table
summary = []
for name, model in trained.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    summary.append({
        'Model':              name,
        'Accuracy':           round(acc, 4),
        'Precision (macro)':  round(report['macro avg']['precision'], 4),
        'Recall (macro)':     round(report['macro avg']['recall'], 4),
        'F1 (macro)':         round(report['macro avg']['f1-score'], 4),
        'F1 (weighted)':      round(report['weighted avg']['f1-score'], 4),
    })

df_summary = pd.DataFrame(summary).set_index('Model')
print(df_summary)