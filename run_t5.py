"""
T5 — Complete CNN Results (fast, no ResNet50 fine-tuning, no augmentation)
"""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import warnings; warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.random.set_seed(42); np.random.seed(42)
sns.set_theme(style='darkgrid')
CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# ── 1. Data ──────────────────────────────────────────────────────
print("[1] Loading CIFAR-10...")
(x_all, y_all), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_all = x_all.astype('float32')/255.0; x_test = x_test.astype('float32')/255.0
x_val,   y_val   = x_all[-5000:],   y_all[-5000:]
x_train, y_train = x_all[:-5000],  y_all[:-5000]
y_tr_c = to_categorical(y_train,10); y_va_c = to_categorical(y_val,10)
y_te_c = to_categorical(y_test,10);  y_true = y_test.flatten()
print(f"  {x_train.shape} / {x_val.shape} / {x_test.shape}")

# ── 2. Custom CNN (load or train) ────────────────────────────────
print("[2] Preparing custom CNN...")
# Try loading saved model first, then train from scratch
model_path = os.path.join(OUTPUT_DIR,'custom_cnn.keras')
if os.path.exists(model_path):
    print("  Loading saved model...")
    model_custom = keras.models.load_model(model_path)
else:
    print("  Building & training from scratch...")
    def build_cnn():
        m = models.Sequential([
            layers.Input((32,32,3)),
            layers.Conv2D(32,(3,3),padding='same'), layers.BatchNormalization(),
            layers.Activation('relu'), layers.MaxPooling2D((2,2)),
            layers.Conv2D(64,(3,3),padding='same'), layers.BatchNormalization(),
            layers.Activation('relu'), layers.MaxPooling2D((2,2)),
            layers.Conv2D(128,(3,3),padding='same'), layers.BatchNormalization(),
            layers.Activation('relu'), layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(256), layers.BatchNormalization(), layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(128), layers.BatchNormalization(), layers.Activation('relu'),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax')
        ])
        m.compile(optimizer=optimizers.Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
        return m
    model_custom = build_cnn()
    hist = model_custom.fit(x_train, y_tr_c, batch_size=128,
                            validation_data=(x_val, y_va_c),
                            epochs=20,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                                     patience=5,restore_best_weights=True,
                                                                     verbose=1)], verbose=2)
    # save history plot
    er = range(1,len(hist.history['accuracy'])+1)
    fig,(a,b) = plt.subplots(1,2,figsize=(14,5))
    a.plot(er,hist.history['accuracy'],'b-',label='Train')
    a.plot(er,hist.history['val_accuracy'],'g--',label='Val')
    a.set_title('Custom CNN — Accuracy'); a.legend(); a.grid(alpha=0.3)
    b.plot(er,hist.history['loss'],'b-',label='Train')
    b.plot(er,hist.history['val_loss'],'g--',label='Val')
    b.set_title('Custom CNN — Loss'); b.legend(); b.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,'T5_training_history.png'), dpi=150); plt.close()
    print("  Saved T5_training_history.png")
    model_custom.save(model_path)
    print("  Saved model.")

# ── 3. Custom CNN confusion matrix ───────────────────────────────
print("[3] Custom CNN evaluation...")
test_loss_c, test_acc_c = model_custom.evaluate(x_test, y_te_c, verbose=0)
print(f"  acc={test_acc_c:.4f}  loss={test_loss_c:.4f}")
y_pred_c = np.argmax(model_custom.predict(x_test,verbose=0), axis=1)
cm_c = confusion_matrix(y_true, y_pred_c)
plt.figure(figsize=(9,7))
sns.heatmap(cm_c, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'Custom CNN — Confusion Matrix (Acc: {test_acc_c:.4f})')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'T5_custom_confusion_matrix.png'), dpi=150); plt.close()
print("  Saved T5_custom_confusion_matrix.png")
print(classification_report(y_true, y_pred_c, target_names=CLASS_NAMES))

# ── 4. ResNet-50 Transfer Learning (eval only, no fine-tuning) ───
print("[4] ResNet-50 transfer learning (frozen only)...")
base = ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3))
base.trainable = False
model_rn = models.Sequential([
    layers.Input((32,32,3)), base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'), layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
model_rn.compile(optimizer=optimizers.Adam(1e-3),
                 loss='categorical_crossentropy', metrics=['accuracy'])
print("  Training 10 epochs frozen...")
hist_rn = model_rn.fit(x_train,y_tr_c, batch_size=128,
                        validation_data=(x_val,y_va_c),
                        epochs=10, verbose=2)
er_rn = range(1, len(hist_rn.history['accuracy'])+1)
fig,(a,b) = plt.subplots(1,2,figsize=(14,5))
a.plot(er_rn, hist_rn.history['val_accuracy'],'r-',label='Val Acc')
a.plot(er_rn, hist_rn.history['accuracy'],'r--',label='Train Acc')
a.set_title('ResNet-50 Transfer — Accuracy'); a.legend(); a.grid(alpha=0.3)
b.plot(er_rn, hist_rn.history['val_loss'],'r-',label='Val Loss')
b.plot(er_rn, hist_rn.history['loss'],'r--',label='Train Loss')
b.set_title('ResNet-50 Transfer — Loss'); b.legend(); b.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'T5_resnet50_training_history.png'), dpi=150); plt.close()
print("  Saved T5_resnet50_training_history.png")

test_loss_r, test_acc_r = model_rn.evaluate(x_test, y_te_c, verbose=0)
print(f"  acc={test_acc_r:.4f}  loss={test_loss_r:.4f}")
y_pred_r = np.argmax(model_rn.predict(x_test,verbose=0), axis=1)
cm_r = confusion_matrix(y_true, y_pred_r)
plt.figure(figsize=(9,7))
sns.heatmap(cm_r, annot=True, fmt='d', cmap='Purples',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'ResNet-50 Transfer Learning — Confusion Matrix (Acc: {test_acc_r:.4f})')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'T5_resnet50_confusion_matrix.png'), dpi=150); plt.close()
print("  Saved T5_resnet50_confusion_matrix.png")
print(classification_report(y_true, y_pred_r, target_names=CLASS_NAMES))

# ── 5. CNN Filters ────────────────────────────────────────────────
print("[5] CNN filter visualizations...")
w = model_custom.layers[0].get_weights()[0]  # (3,3,3,32)
fig,axes = plt.subplots(4,8,figsize=(22,8)); axes=axes.flatten()
for i in range(32):
    f = (w[:,:,0,i]-w[:,:,0,i].min())/(w[:,:,0,i].max()-w[:,:,0,i].min()+1e-8)
    axes[i].imshow(f,cmap='viridis'); axes[i].axis('off'); axes[i].set_title(f'#{i}',fontsize=6)
plt.suptitle('Custom CNN — First Conv Layer Filters (R channel, 32/32)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'T5_cnn_filters.png'), dpi=150); plt.close()
print("  Saved T5_cnn_filters.png")

# ── 6. Activation Maps ────────────────────────────────────────────
print("[6] Activation maps...")
sample_img = x_test[0:1]
conv_outs = [l.output for l in model_custom.layers if isinstance(l,(layers.Conv2D,layers.Activation))]
act_mod   = keras.models.Model(model_custom.input, conv_outs)
acts_all  = act_mod.predict(sample_img, verbose=0)
# After 1st conv+bn+act+pool
acts_c1 = acts_all[4]
fig,axs = plt.subplots(4,8,figsize=(22,8)); axs=axs.flatten()
for i in range(min(16,acts_c1.shape[-1])):
    axs[i].imshow(acts_c1[0,:,:,i], cmap='viridis'); axs[i].axis('off'); axs[i].set_title(f'Ch{i}',fontsize=7)
for j in range(16,32): axs[j].axis('off')
plt.suptitle('CNN Activation Maps — Conv Block 1 (first test image)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'T5_cnn_activations.png'), dpi=150); plt.close()
print("  Saved T5_cnn_activations.png")

# ── 7. Sample Predictions ─────────────────────────────────────────
print("[7] Sample predictions plot...")
idx = np.random.choice(len(x_test),16,replace=False)
fig,axs = plt.subplots(4,4,figsize=(16,16))
for i,ix in enumerate(idx,1):
    ax = axs.flatten()[i-1]
    ax.imshow(x_test[ix])
    cl=CLASS_NAMES[y_pred_c[ix]]; tl=CLASS_NAMES[y_true[ix]]
    ax.set_title(f'CNN:{cl}\nTrue:{tl}',
                 color=('green' if y_pred_c[ix]==y_true[ix] else 'red'),fontsize=8)
    ax.axis('off')
plt.suptitle('Custom CNN — CIFAR-10 Sample Predictions (green=✓  red=✗)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'T5_sample_predictions.png'), dpi=150); plt.close()
print("  Saved T5_sample_predictions.png")

# ── 8. Model Comparison ───────────────────────────────────────────
print("[8] Saving model comparison bar chart...")
fig,(a,b) = plt.subplots(1,2,figsize=(12,5))
names=['Custom CNN','ResNet-50 TL']
accs=[float(test_acc_c),float(test_acc_r)]
lss =[float(test_loss_c),float(test_loss_r)]
bc=a.bar(names,accs,color=['#4472C4','#ED7D31'],width=0.5,edgecolor='black')
a.set_ylim([0,1.1])
for bv,v in zip(bc,accs): a.text(bv.get_x()+bv.get_width()/2,bv.get_height()+0.01,
                                    f'{v:.4f}',ha='center',fontweight='bold')
a.set_title('Test Accuracy'); a.set_ylabel('Accuracy'); a.grid(axis='y',alpha=0.3)
bc2=b.bar(names,lss,color=['#4472C4','#ED7D31'],width=0.5,edgecolor='black')
for bv,v in zip(bc2,lss): b.text(bv.get_x()+bv.get_width()/2,bv.get_height()+0.005,
                                   f'{v:.4f}',ha='center',fontweight='bold')
b.set_title('Test Loss'); b.set_ylabel('Loss'); b.grid(axis='y',alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'T5_model_comparison.png'), dpi=150); plt.close()
print("  Saved T5_model_comparison.png")

print("\n✅ ALL DONE. Files in results/:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.startswith('T5'): print(f"  {f}  ({os.path.getsize(os.path.join(OUTPUT_DIR,f))//1024} KB)")
