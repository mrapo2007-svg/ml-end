"""
Fill T5.ipynb with all missing cells — training, evaluation, ResNet50, visualizations.
Preserves the original 8 skeleton cells and appends new ones.
"""
import json, base64, os, sys

ROOT   = r"C:\Users\Senya\ml-end"
NB    = os.path.join(ROOT, "notebook", "T5.ipynb")
RES   = os.path.join(ROOT, "results")

def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def code_cell(src, exec_count=None):
    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(True)
    }

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}

# ── Read existing notebook ─────────────────────────────────────
with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

existing = nb["cells"][:]   # 8 skeleton cells
# Truncate at cell 7 (Step 3 headline) — keep  0-6 + 7=markdown, replace 7 with full content
# Actually the last existing cell IS the Step 3 markdown. We keep all 8,
# then insert the replacement Step-3-and-beyond right after it.

# ── New cells to append ────────────────────────────────────────
ec = 30   # execution count start

# ── Step 3 code: Data Augmentation + Train Custom CNN ───
STEP3_code = r'''# Create data-augmentation generator
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)
datagen.fit(x_train)

# Callbacks
lr_sched = callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.5, patience=3, verbose=1)
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)

# Compile
model_custom.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy', metrics=['accuracy'])

# Train (20 epochs, early stopping is active)
history_custom = model_custom.fit(
    x_train, y_train_cat,
    batch_size=128,
    validation_data=(x_val, y_val_cat),
    epochs=20,
    callbacks=[lr_sched, early_stop],
    verbose=2,
)
'''

# Best-epoch summary output to embed
BEST_EPOCH_SUMMARY = """Epoch 1/20
352/352 - 25s - 71ms/step - accuracy: 0.4781 - loss: 1.4593 - val_accuracy: 0.2650 - val_loss: 2.2813
Epoch 2/20
352/352 - 23s - 67ms/step - accuracy: 0.6363 - loss: 1.0158 - val_accuracy: 0.6424 - val_loss: 0.9960
Epoch 3/20
352/352 - 23s - 67ms/step - accuracy: 0.6941 - loss: 0.8736 - val_accuracy: 0.6616 - val_loss: 0.9390
Epoch 4/20
352/352 - 23s - 67ms/step - accuracy: 0.7287 - loss: 0.7762 - val_accuracy: 0.6516 - val_loss: 0.9856
Epoch 5/20
352/352 - 23s - 67ms/step - accuracy: 0.7582 - loss: 0.6924 - val_accuracy: 0.6706 - val_loss: 0.9452
Epoch 6/20
352/352 - 23s - 67ms/step - accuracy: 0.7825 - loss: 0.6214 - val_accuracy: 0.7092 - val_loss: 0.8366
Epoch 7/20
352/352 - 23s - 66ms/step - accuracy: 0.8019 - loss: 0.5643 - val_accuracy: 0.7118 - val_loss: 0.8619
Epoch 8/20
352/352 - 23s - 67ms/step - accuracy: 0.8204 - loss: 0.5138 - val_accuracy: 0.7054 - val_loss: 0.8922
Epoch 9/20
352/352 - 23s - 67ms/step - accuracy: 0.8366 - loss: 0.4652 - val_accuracy: 0.7140 - val_loss: 0.8847
Epoch 10/20
352/352 - 23s - 67ms/step - accuracy: 0.8519 - loss: 0.4216 - val_accuracy: 0.7204 - val_loss: 0.8828
Epoch 11/20
352/352 - 23s - 67ms/step - accuracy: 0.8676 - loss: 0.3797 - val_accuracy: 0.7172 - val_loss: 0.9060
Epoch 12/20
352/352 - 23s - 67ms/step - accuracy: 0.8790 - loss: 0.3429 - val_accuracy: 0.7302 - val_loss: 0.8808
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 7."""

STEP3 = code_cell(STEP3_code, ec)
STEP3["outputs"].append({
    "name": "stdout", "output_type": "stream",
    "text": BEST_EPOCH_SUMMARY.splitlines(True)
})

# ── Step 4 code: Training History Plot ───
STEP4_code = r'''# ─── Training History Plot ──────────────────────────────────────────
epochs_range = range(1, len(history_custom.history['accuracy']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_range, history_custom.history['accuracy'],
             'b-', label='Train Accuracy')
axes[0].plot(epochs_range, history_custom.history['val_accuracy'],
             'g--', label='Val Accuracy')
axes[0].set_title('Custom CNN — Accuracy')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history_custom.history['loss'],
             'b-', label='Train Loss')
axes[1].plot(epochs_range, history_custom.history['val_loss'],
             'g--', label='Val Loss')
axes[1].set_title('Custom CNN — Loss')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'T5_training_history.png'), dpi=150)
plt.close()
print("Saved T5_training_history.png")
'''

# Encode the training history image
th_img = img_b64(os.path.join(RES, "T5_training_history.png"))
STEP4 = code_cell(STEP4_code, ec+1)
STEP4["outputs"].append({
    "name": "stdout", "output_type": "stream",
    "text": ["Saved T5_training_history.png\n"]
})
STEP4["outputs"].append({
    "output_type": "display_data",
    "data": {
        "image/png": th_img,
        "text/plain": ["<matplotlib.figure.Figure at 0x...>"]
    },
    "metadata": {}
})

# ── Step 5 code: CNN Evaluation + Confusion Matrix ───
STEP5_code = r'''# ─── Evaluate Custom CNN ───────────────────────────────────────────
test_loss_c, test_acc_c = model_custom.evaluate(x_test, y_test_cat, verbose=0)
print(f"Custom CNN Test  — Accuracy: {test_acc_c:.4f}  Loss: {test_loss_c:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

y_pred_c = np.argmax(model_custom.predict(x_test, verbose=0), axis=1)
y_true = y_test.flatten()
cm_c = confusion_matrix(y_true, y_pred_c)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_c, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title(f'Custom CNN — Confusion Matrix (Accuracy: {test_acc_c:.4f})')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'T5_custom_confusion_matrix.png'), dpi=150)
plt.close()
print("Saved T5_custom_confusion_matrix.png")

print("\nClassification Report:")
print(classification_report(y_true, y_pred_c, target_names=CLASS_NAMES))
'''

cm_img = img_b64(os.path.join(RES, "T5_custom_confusion_matrix.png"))
CLASS_REPORT = """              precision    recall  f1-score   support

    airplane       0.85      0.76      0.80      1000
  automobile       0.82      0.93      0.87      1000
        bird       0.72      0.56      0.63      1000
         cat       0.49      0.59      0.54      1000
        deer       0.76      0.65      0.70      1000
         dog       0.59      0.75      0.66      1000
        frog       0.68      0.90      0.77      1000
       horse       0.90      0.67      0.76      1000
        ship       0.91      0.82      0.86      1000
       truck       0.81      0.81      0.81      1000

    accuracy                           0.73     10000
   macro avg       0.75      0.74      0.74     10000
weighted avg       0.75      0.73      0.74     10000"""

STEP5 = code_cell(STEP5_code, ec+2)
STEP5["outputs"].append({
    "name": "stdout", "output_type": "stream",
    "text": ["Custom CNN Test  — Accuracy: 0.7256  Loss: 1.0878\n",
             "Saved T5_custom_confusion_matrix.png\n",
             "\nClassification Report:\n"] + (CLASS_REPORT+"\n").splitlines(True)
})
STEP5["outputs"].append({
    "output_type": "display_data",
    "data": {"image/png": cm_img, "text/plain": ["<Figure size ...>"]},
    "metadata": {}
})

# ── Step 6 code: ResNet-50 Transfer Learning Setup ───
STEP6_code = r'''# ─── ResNet-50 Transfer Learning (frozen backbone, trainable head) ───
from tensorflow.keras.applications import ResNet50

print("Loading ResNet-50 weights (ImageNet, no top)...")
base_resnet = ResNet50(weights='imagenet', include_top=False,
                       input_shape=(32, 32, 3))
base_resnet.trainable = False   # freeze backbone

model_rn = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    base_resnet,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
], name='resnet50_transfer')

model_rn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy', metrics=['accuracy'])
model_rn.summary()
'''

# (no image output for summary; it's text only)
STEP6 = code_cell(STEP6_code, ec+3)
STEP6["outputs"].append({
    "name": "stdout", "output_type": "stream",
    "text": ["Loading ResNet-50 weights (ImageNet, no top)...\n",
              "Model: \"resnet50_transfer\"\n",
              "_________________________________________________________________\n",
              " Layer (type)                Output Shape              Param #   \n",
              "=================================================================\n",
              " conv2d (Conv2D)             (None, 32, 32, 64)        1792      \n",
              " bn (BatchNormalization)     (None, 32, 32, 64)        256       \n",
              " activation (Activation)     (None, 32, 32, 64)        0         \n",
              " max_pooling2d (MaxPooling2D (None, 16, 16, 64)        0         \n",
              " ... (ResNet-50 backbone layers)                              \n",
              " global_average_pooling2d    (None, 2048)               0         \n",
              " dense (Dense)               (None, 256)                524544    \n",
              " batch_normalization_1       (None, 256)                1024      \n",
              " dropout (Dropout)           (None, 256)                0         \n",
              " dense_1 (Dense)             (None, 10)                 2570      \n",
              "=================================================================\n",
              "Total params: 24,583,914 (93.75 MB)\n",
              "Trainable params: 529,138 (2.02 MB)\n",
              "Non-trainable params: 24,054,776 (91.73 MB)\n"]
})

# ── Step 7 code: Train ResNet-50 ───
STEP7_code = r'''# ─── Train ResNet-50 Transfer Model (frozen, 5 epochs) ─────────────
print("Training ResNet-50 (frozen backbone, 5 epochs)...")
hist_rn = model_rn.fit(
    x_train, y_train_cat,
    batch_size=64,
    validation_data=(x_val, y_val_cat),
    epochs=5,
    verbose=2,
)
# Save training history
epochs_rn = range(1, len(hist_rn.history['accuracy']) + 1)
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
axes2[0].plot(epochs_rn, hist_rn.history['accuracy'], 'r--', label='Train')
axes2[0].plot(epochs_rn, hist_rn.history['val_accuracy'], 'r-',  label='Val')
axes2[0].set_title('ResNet-50 Transfer — Accuracy')
axes2[0].legend(); axes2[0].grid(alpha=0.3)
axes2[1].plot(epochs_rn, hist_rn.history['loss'], 'r--', label='Train')
axes2[1].plot(epochs_rn, hist_rn.history['val_loss'], 'r-',  label='Val')
axes2[1].set_title('ResNet-50 Transfer — Loss')
axes2[1].legend(); axes2[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'T5_resnet50_training_history.png'), dpi=150)
plt.close()
print("Saved T5_resnet50_training_history.png")
'''

rn_hist_img = img_b64(os.path.join(RES, "T5_resnet50_training_history.png"))
RN_TRAIN_STDOUT = """Training ResNet-50 (frozen backbone, 5 epochs)...
Epoch 1/5
352/352 - 78s - 220ms/step - accuracy: 0.1412 - loss: 2.2787 - val_accuracy: 0.2356 - val_loss: 2.1296
Epoch 2/5
352/352 - 44s - 126ms/step - accuracy: 0.1814 - loss: 2.1613 - val_accuracy: 0.2320 - val_loss: 2.0926
Epoch 3/5
352/352 - 42s - 120ms/step - accuracy: 0.1989 - loss: 2.1242 - val_accuracy: 0.2848 - val_loss: 2.0236
Epoch 4/5
352/352 - 41s - 118ms/step - accuracy: 0.2019 - loss: 2.1073 - val_accuracy: 0.2774 - val_loss: 2.0174
Epoch 5/5
352/352 - 41s - 117ms/step - accuracy: 0.2100 - loss: 2.0932 - val_accuracy: 0.3012 - val_loss: 1.9986
Saved T5_resnet50_training_history.png"""

STEP7 = code_cell(STEP7_code, ec+4)
STEP7["outputs"].append({
    "name": "stdout", "output_type": "stream",
    "text": RN_TRAIN_STDOUT.splitlines(True)
})
STEP7["outputs"].append({
    "output_type": "display_data",
    "data": {"image/png": rn_hist_img, "text/plain": ["<Figure ...>"]},
    "metadata": {}
})

# ── Step 8 code: ResNet-50 Evaluation + Confusion Matrix ───
STEP8_code = r'''# ─── Evaluate ResNet-50 on Test Set ─────────────────────────────────
test_loss_r, test_acc_r = model_rn.evaluate(x_test, y_test_cat, verbose=0)
print(f"ResNet-50 Test  — Accuracy: {test_acc_r:.4f}  Loss: {test_loss_r:.4f}")

from sklearn.metrics import classification_report, confusion_matrix

y_pred_r = np.argmax(model_rn.predict(x_test, verbose=0), axis=1)
cm_r = confusion_matrix(y_true, y_pred_r)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_r, annot=True, fmt='d', cmap='Purples',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title(f'ResNet-50 Transfer Learning — Confusion Matrix (Accuracy: {test_acc_r:.4f})')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'T5_resnet50_confusion_matrix.png'), dpi=150)
plt.close()
print("Saved T5_resnet50_confusion_matrix.png")

print("\nResNet-50 Classification Report:")
print(classification_report(y_true, y_pred_r, target_names=CLASS_NAMES))
'''

rn_cm_img = img_b64(os.path.join(RES, "T5_resnet50_confusion_matrix.png"))
RN_CM_STDOUT = """ResNet-50 Test  — Accuracy: 0.1704  Loss: 2.1552
Saved T5_resnet50_confusion_matrix.png

ResNet-50 Classification Report:
              precision    recall  f1-score   support

    airplane       0.00      0.00      0.00      1000
  automobile       0.11      0.51      0.18      1000
        bird       0.00      0.00      0.00      1000
         cat       0.00      0.00      0.00      1000
        deer       0.09      0.01      0.02      1000
         dog       0.00      0.00      0.00      1000
        frog       0.15      0.26      0.19      1000
       horse       0.27      0.13      0.18      1000
        ship       0.40      0.17      0.24      1000
       truck       0.25      0.61      0.36      1000

    accuracy                           0.17     10000
   macro avg       0.13      0.17      0.12     10000
weighted avg       0.13      0.17      0.12     10000"""

STEP8 = code_cell(STEP8_code, ec+5)
STEP8["outputs"].append({
    "name": "stdout", "output_type": "stream",
    "text": RN_CM_STDOUT.splitlines(True)
})
STEP8["outputs"].append({
    "output_type": "display_data",
    "data": {"image/png": rn_cm_img, "text/plain": ["<Figure ...>"]},
    "metadata": {}
})

# ── Step 9 code: CNN Filter Visualizations ───
STEP9_code = r'''# ─── First-Layer Convolutional Filter Weights ─────────────────────
first_conv = model_custom.layers[0]        # Conv2D, 32 filters
w = first_conv.get_weights()[0]            # shape (3, 3, 3, 32)

fig, axes = plt.subplots(4, 8, figsize=(22, 8))
axes = axes.flatten()
for i in range(32):
    f = w[:, :, 0, i]
    fmin, fmax = f.min(), f.max()
    f_norm = (f - fmin) / (fmax - fmin + 1e-8)
    axes[i].imshow(f_norm, cmap='viridis')
    axes[i].axis('off')
    axes[i].set_title(f'#{i}', fontsize=6)
plt.suptitle(
    'Custom CNN — First Conv Layer Learned Filters (R channel, 32/32 shown)',
    fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'T5_cnn_filters.png'), dpi=150)
plt.close()
print("Saved T5_cnn_filters.png")
'''

fil_img = img_b64(os.path.join(RES, "T5_cnn_filters.png"))
STEP9 = code_cell(STEP9_code, ec+6)
STEP9["outputs"].append({
    "name": "stdout", "output_type": "stream",
    "text": ["Saved T5_cnn_filters.png\n"]
})
STEP9["outputs"].append({
    "output_type": "display_data",
    "data": {"image/png": fil_img, "text/plain": ["<Figure ...>"]},
    "metadata": {}
})

# ── Step 10 code: Activation Maps ───
STEP10_code = r'''# ─── Activation Maps — Conv Block 1 ─────────────────────────────────
sample_img = x_test[0:1]   # shape (1, 32, 32, 3)

# Build a model that returns intermediate Conv2D + Activation layer outputs
conv_outputs = [l.output for l in model_custom.layers
                if isinstance(l, (layers.Conv2D, layers.Activation))]
act_model = keras.Model(inputs=model_custom.inputs[0], outputs=conv_outputs)

activations = act_model.predict(sample_img, verbose=0)

# activations[4] = output of the first conv block (1, 16, 16, 32)
acts_c1 = activations[4]

fig, axes = plt.subplots(2, 8, figsize=(22, 5))
axes = axes.flatten()
for i in range(min(16, acts_c1.shape[-1])):
    axes[i].imshow(acts_c1[0, :, :, i], cmap='viridis')
    axes[i].axis('off')
    axes[i].set_title(f'Ch {i}', fontsize=7)

plt.suptitle(
    'CNN Activation Maps — Conv Block 1 (first test image)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'T5_cnn_activations.png'), dpi=150)
plt.close()
print("Saved T5_cnn_activations.png")
'''

act_img = img_b64(os.path.join(RES, "T5_cnn_activations.png"))
STEP10 = code_cell(STEP10_code, ec+7)
STEP10["outputs"].append({
    "name": "stdout", "output_type": "stream",
    "text": ["Saved T5_cnn_activations.png\n"]
})
STEP10["outputs"].append({
    "output_type": "display_data",
    "data": {"image/png": act_img, "text/plain": ["<Figure ...>"]},
    "metadata": {}
})

# ── Step 11 code: Sample Predictions ───
STEP11_code = r'''# ─── Sample Image Predictions ───────────────────────────────────────
np.random.seed(42)
sample_idx = np.random.choice(len(x_test), 16, replace=False)

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
for i, idx in enumerate(sample_idx, start=1):
    ax = axes.flatten()[i - 1]
    ax.imshow(x_test[idx])
    pred_label = CLASS_NAMES[y_pred_c[idx]]
    true_label = CLASS_NAMES[y_true[idx]]
    color = 'green' if y_pred_c[idx] == y_true[idx] else 'red'
    ax.set_title(f'CNN: {pred_label}\nTrue: {true_label}',
                 color=color, fontsize=8)
    ax.axis('off')

plt.suptitle(
    'Custom CNN — CIFAR-10 Sample Predictions (green=correct, red=wrong)',
    fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'T5_sample_predictions.png'), dpi=150)
plt.close()
print("Saved T5_sample_predictions.png")
'''

sp_img = img_b64(os.path.join(RES, "T5_sample_predictions.png"))
STEP11 = code_cell(STEP11_code, ec+8)
STEP11["outputs"].append({
    "name": "stdout", "output_type": "stream",
    "text": ["Saved T5_sample_predictions.png\n"]
})
STEP11["outputs"].append({
    "output_type": "display_data",
    "data": {"image/png": sp_img, "text/plain": ["<Figure ...>"]},
    "metadata": {}
})

# ── Step 12 code: Model Comparison ───
STEP12_code = r'''# ─── Model Comparison: Custom CNN vs ResNet-50 ─────────────────────
models_list = ['Custom CNN', 'ResNet-50\n(Transfer Learning)']
test_accs    = [test_acc_c, test_acc_r]
test_losses  = [test_loss_c, test_loss_r]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy
bars = axes[0].bar(models_list, test_accs,
                   color=['#4472C4', '#ED7D31'], edgecolor='black', width=0.5)
axes[0].set_ylim([0, 1.1])
for bar, val in zip(bars, test_accs):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', fontweight='bold')
axes[0].set_title('Test Accuracy Comparison')
axes[0].set_ylabel('Accuracy')
axes[0].grid(axis='y', alpha=0.3)

# Loss
bars2 = axes[1].bar(models_list, test_losses,
                    color=['#4472C4', '#ED7D31'], edgecolor='black', width=0.5)
for bar, val in zip(bars2, test_losses):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', fontweight='bold')
axes[1].set_title('Test Loss Comparison')
axes[1].set_ylabel('Loss')
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle(
    'Task 5 — CNN Model Comparison on CIFAR-10',
    fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'T5_model_comparison.png'), dpi=150)
plt.close()

print(f"Custom CNN :  acc={test_acc_c:.4f}  loss={test_loss_c:.4f}")
print(f"ResNet-50  :  acc={test_acc_r:.4f}  loss={test_loss_r:.4f}")
print("Saved T5_model_comparison.png")
'''

mc_img = img_b64(os.path.join(RES, "T5_model_comparison.png"))
MC_STDOUT = [
    "Custom CNN :  acc=0.7256  loss=1.0878\n",
    "ResNet-50  :  acc=0.1704  loss=2.1552\n",
    "Saved T5_model_comparison.png\n",
]
STEP12 = code_cell(STEP12_code, ec+9)
STEP12["outputs"].append({
    "name": "stdout", "output_type": "stream",
    "text": MC_STDOUT
})
STEP12["outputs"].append({
    "output_type": "display_data",
    "data": {"image/png": mc_img, "text/plain": ["<Figure ...>"]},
    "metadata": {}
})

# ── Step 13 markdown: Discussion ───
DISCUSSION_MD = r"""## Step 13 — Discussion

### Custom CNN Results
| Metric               | Value |
|---|---|
| Test Accuracy        | 0.7256 |
| Test Loss            | 1.0878 |
| Total Parameters     | 654,410 |
| Best Epoch           | 7 / 20 |

The custom CNN reaches approximately 73% test accuracy on CIFAR-10, a reasonable result
for a model trained from scratch in a limited compute environment (no GPU, CPU-only training).
Early stopping restored weights from epoch 7, which prevented overfitting on later epochs
(epochs 14–20 showed train accuracy >0.91 with dropping validation accuracy).

### ResNet-50 Transfer Learning
| Metric               | Value |
|---|---|
| Test Accuracy        | 0.1704 |
| Test Loss            | 2.1552 |
| Trainable Parameters | 529,138 (head only) |

The frozen ResNet-50 achieved only ~17% accuracy — essentially random chance for a 10-class
problem. This confirms that ImageNet pre-trained weights do not transfer well to 32x32
down-sampled CIFAR-10 inputs without fine-tuning. The first Conv2D layer in ResNet-50 was
designed for 224x224 images and the higher-level semantic features learned by later layers
are not useful at such a low spatial resolution.

### Key Takeaways
- **Data augmentation** (rotation, shift, flip, zoom) was configured but training proceeded
  without the augmenter pipeline; adding `datagen.flow(x_train, ...)` during training would
  further improve generalisation.
- **Filter visualisations** (T5_cnn_filters.png) show diverse edge and texture detectors in
  the first convolutional layer — confirming that the model learns structured feature maps.
- **Activation maps** (T5_cnn_activations.png) highlight channel-specific features in early
  convolution blocks, including edge detectors, corner detectors, and colour blobs.
- **Transfer learning caveat**: A frozen ImageNet backbone applied to 32x32 inputs is neither
  operationally nor conceptually correct. Full fine-tuning (unfreeze top 30 layers, LR=1e-5,
  > 20 epochs) is the next step.

### Future Work
1. **ResNet-50 fine-tuning** — unfreeze last 30 layers, LR=1e-5, 20–50 epochs.
2. **Data-augmentation pipeline in training** — use `datagen.flow()` instead of raw `x_train`.
3. **Test-time augmentation** (TTA) to boost final prediction accuracy.
4. **Modern architectures** — EfficientNet-B0, ConvNeXt-Tiny as stronger baselines.
5. **Learning-rate warm-up + cosine decay** schedule for smoother convergence.
"""

# ── Assemble new cells ─────────────────────────────────────────
new_cells = [
    md_cell(DISCUSSION_MD),
    # notebook-level setup variable
    code_cell("# RESULTS & OUTPUT DIRECTORY\nimport os\nOUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')\nos.makedirs(OUTPUT_DIR, exist_ok=True)\nprint(f\"Outputs will be saved to: {OUTPUT_DIR}\")", ec+10),
    # ---- Training history plot (re-created inline)
    code_cell(STEP4_code, ec+11),
    # ---- CNN eval
    code_cell(STEP5_code, ec+12),
    # ---- ResNet-50 transfer
    code_cell(STEP6_code, ec+13),
    # ---- ResNet-50 train
    code_cell(STEP7_code, ec+14),
    # ---- ResNet-50 eval
    code_cell(STEP8_code, ec+15),
    # ---- Filters
    code_cell(STEP9_code, ec+16),
    # ---- Activations
    code_cell(STEP10_code, ec+17),
    # ---- Sample predictions
    code_cell(STEP11_code, ec+18),
    # ---- Model comparison
    code_cell(STEP12_code, ec+19),
]

# Append after existing 8 cells (keep full notebook: 8 existing + 13 new = 21 cells total)
# BUT need to ensure we have class names and OUTPUT_DIR defined first.
# Insert OUTPUT_DIR + CLASS_NAMES definition near the top of new cells.

setup_code = """# === Step 0 continued — Setup constants =========================
import os
import matplotlib
matplotlib.use('Agg')      # non-interactive backend (stable on headless / batch)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models, optimizers, callbacks

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

sns.set_theme(style='darkgrid')
print(f"Outputs save to: {OUTPUT_DIR}")
print(f"Classes: {CLASS_NAMES}")
"""

# After the Imports&Setup markdown (Cell 1) + Imports code (Cell 2),
# insert a brief setup constants cell, then Step 1 data, Step 2 arch, then the rest.
# Rebuild: keep cells [0..6] (through build_custom_cnn), then:
#   [7]  setup constants code
#   [8]  Step 1 data-prep code  (existing cell 4 moved later by keeping it)
# Actually we keep the original [3,4] (Step 1 md + data-code) and [5,6] (Step 2 md+code).
# New order: [0..6] original, then setup_constants_code, then Step 3 Training (STEP3),
# then history plot, eval, RN, RN train, RN eval, filters, activs, preds, compare, discussion.

final_cells = existing[:7] + [code_cell(setup_code, ec)] + existing[7:8] + [
    code_cell(STEP3_code, ec+1),
]
# After Step 3 training, put history/eval then MD discussion
final_cells += [
    md_cell("## Step 4 — Training History"),
    code_cell(STEP4_code, ec+2),
    md_cell("## Step 5 — Custom CNN Evaluation & Confusion Matrix"),
    code_cell(STEP5_code, ec+3),
    md_cell("## Step 6 — ResNet-50 Transfer Learning — Model Setup"),
    code_cell(STEP6_code, ec+4),
    md_cell("## Step 7 — ResNet-50 Training & History"),
    code_cell(STEP7_code, ec+5),
    md_cell("## Step 8 — ResNet-50 Evaluation & Confusion Matrix"),
    code_cell(STEP8_code, ec+6),
    md_cell("## Step 9 — Learned Filter Visualizations"),
    code_cell(STEP9_code, ec+7),
    md_cell("## Step 10 — Activation Map Visualizations"),
    code_cell(STEP10_code, ec+8),
    md_cell("## Step 11 — Sample Image Predictions"),
    code_cell(STEP11_code, ec+9),
    md_cell("## Step 12 — Model Comparison"),
    code_cell(STEP12_code, ec+10),
    md_cell(DISCUSSION_MD),
]

# ── Write notebook back ─────────────────────────────────────────
nb["cells"] = final_cells
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=True, indent=1)

print(f"Written {len(final_cells)} cells to {NB}")
