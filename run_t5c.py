"""
T5 Step C — Model comparison bar chart (reads resnet50_metrics.txt, does not run training)
"""
import os, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
sns.set_theme(style='darkgrid')

# Read ResNet50 metrics saved by run_t5b.py
metrics_path = os.path.join(OUTPUT_DIR, 'resnet50_metrics.txt')
if not os.path.exists(metrics_path):
    print("ERROR: resnet50_metrics.txt not found! Run run_t5b.py first.", flush=True)
    sys.exit(1)
with open(metrics_path) as f:
    test_acc_r = float(f.readline().strip())
    test_loss_r = float(f.readline().strip())

# Custom CNN metrics (from latest evaluation — check the T5_custom_confusion_matrix output)
# We already know from run_t5a.py output: acc=0.7256, loss=1.0878
test_acc_c, test_loss_c = 0.7256, 1.0878

print("[*] Saving T5_model_comparison.png ...", flush=True)
print(f"    Custom CNN: acc={test_acc_c:.4f}  loss={test_loss_c:.4f}", flush=True)
print(f"    ResNet-50:  acc={test_acc_r:.4f}  loss={test_loss_r:.4f}", flush=True)

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
names = ['Custom CNN\n(3 conv blocks)', 'ResNet-50\n(Frozen transfer)']

# Accuracy subplot
bc = a.bar(names, [test_acc_c, test_acc_r],
           color=['#4472C4', '#ED7D31'], width=0.5, edgecolor='black')
a.set_ylim([0, 1.0])
for bar, val in zip(bc, [test_acc_c, test_acc_r]):
    a.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
           f'{val:.4f}', ha='center', fontweight='bold', fontsize=11)
a.set_title('Test Accuracy — CIFAR-10', fontweight='bold')
a.set_ylabel('Accuracy'); a.grid(axis='y', alpha=0.4)

# Loss subplot
bc2 = b.bar(names, [test_loss_c, test_loss_r],
            color=['#4472C4', '#ED7D31'], width=0.5, edgecolor='black')
for bar, val in zip(bc2, [test_loss_c, test_loss_r]):
    b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
           f'{val:.4f}', ha='center', fontweight='bold', fontsize=11)
b.set_title('Test Loss — CIFAR-10', fontweight='bold')
b.set_ylabel('Categorical Cross-Entropy'); b.grid(axis='y', alpha=0.4)
plt.suptitle('Task 5 — CNN Model Comparison: Custom vs ResNet-50 Transfer Learning',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'T5_model_comparison.png'), dpi=150)
plt.close()
print("[*] Saved T5_model_comparison.png", flush=True)
print("[*] DONE.", flush=True)
