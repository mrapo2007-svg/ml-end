import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, model) in zip(axes, trained.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
    ax.set_title(name, fontweight='bold')
plt.suptitle('Confusion Matrices', fontweight='bold')
plt.tight_layout()
plt.show()

# Classification report
for name, model in trained.items():
    y_pred = model.predict(X_test)
    print(f'\n{name}:')
    print(classification_report(y_test, y_pred))

# ROC curves
y_bin = label_binarize(y_test, classes=[0, 1, 2])
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, model) in zip(axes, trained.items()):
    y_score = model.predict_proba(X_test)
    for i, color in zip(range(3), ['navy', 'darkorange', 'green']):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f'Class {i} (AUC={auc(fpr,tpr):.2f})')
    ax.plot([0,1],[0,1],'k--')
    ax.set_title(name, fontweight='bold')
    ax.legend(fontsize=7)
plt.suptitle('ROC Curves', fontweight='bold')
plt.tight_layout()
plt.show()

# Cross-validation
print('\n5-Fold Cross-Validation (F1 weighted):')
for name, model in trained.items():
    scores = cross_val_score(model, X_tv, y_tv, cv=5, scoring='f1_weighted')
    print(f'  {name:<20} {scores.mean():.4f} ± {scores.std():.4f}')