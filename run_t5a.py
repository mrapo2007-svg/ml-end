"""
T5 Step A: All fast-result files (custom CNN, filters, activations, predictions, comparison)
Runs in ~2-3 minutes — no ResNet50.
"""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings; warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.random.set_seed(42); np.random.seed(42)
sns.set_theme(style='darkgrid')

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

print("="*55)
print("T5 Step A — Custom CNN Results (fast)")
print("="*55)

print("\n[1] Loading CIFAR-10...")
(x_all,y_all),(x_test,y_test) = keras.datasets.cifar10.load_data()
x_all = x_all.astype('float32')/255.0;  x_test = x_test.astype('float32')/255.0
x_val, y_val   = x_all[-5000:],  y_all[-5000:]
x_train, y_train = x_all[:-5000], y_all[:-5000]
y_tr_c = keras.utils.to_categorical(y_train,10)
y_va_c = keras.utils.to_categorical(y_val,10)
y_te_c = keras.utils.to_categorical(y_test,10)
y_true = y_test.flatten()
print(f"  {x_train.shape} / {x_val.shape} / {x_test.shape}")

print("[2] Building & training custom CNN...")
def build_cnn():
    m = models.Sequential([
        keras.Input(shape=(32,32,3)),
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
    m.compile(optimizer=optimizers.Adam(1e-3), loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m

model_path = os.path.join(OUTPUT_DIR, 'custom_cnn.keras')
if os.path.exists(model_path):
    print("  Loading saved model ...")
    model_custom = keras.models.load_model(model_path)
else:
    model_custom = build_cnn()
    hist = model_custom.fit(
        x_train, y_tr_c, batch_size=128, validation_data=(x_val, y_va_c),
        epochs=20, verbose=2,
        callbacks=[callbacks.EarlyStopping(monitor='val_accuracy', patience=5,
                                           restore_best_weights=True, verbose=1)]
    )
    er = range(1, len(hist.history['accuracy'])+1)
    fig,(a,b) = plt.subplots(1,2, figsize=(14,5))
    a.plot(er, hist.history['accuracy'],          'b-',
           hist.history['val_accuracy'], 'g--', label='Train    Val')
    a.set_title('CNN — Accuracy'); a.legend(); a.grid(alpha=0.3)
    b.plot(er, hist.history['loss'], 'b-',  hist.history['val_loss'], 'g--', label='Train    Val')
    b.set_title('CNN — Loss'); b.legend(); b.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'T5_training_history.png'), dpi=150); plt.close()
    print("  Saved T5_training_history.png")
    model_custom.save(model_path)
    print("  Saved model.")

print("\n[3] CNN evaluation ...")
test_loss_c, test_acc_c = model_custom.evaluate(x_test, y_te_c, verbose=0)
print(f"  acc={test_acc_c:.4f}  loss={test_loss_c:.4f}")
y_pred_c = np.argmax(model_custom.predict(x_test, verbose=0), axis=1)
cm_c = confusion_matrix(y_true, y_pred_c)
fig, ax = plt.subplots(figsize=(9,7))
ConfusionMatrixDisplay(cm_c, display_labels=CLASS_NAMES).plot(cmap='Blues', ax=ax, colorbar=False)
plt.title(f'Custom CNN — Confusion Matrix  (acc={test_acc_c:.4f})')
for im in ax.get_images(): im.colorbar = None   # cleanup double colorbar from sklearn
fig.colorbar(ax.get_images()[0], ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'T5_custom_confusion_matrix.png'), dpi=150); plt.close()
print("  Saved T5_custom_confusion_matrix.png")
print("\n  Classification Report:")
print(classification_report(y_true, y_pred_c, target_names=CLASS_NAMES))

print("\n[4] Filter visualizations ...")
w = model_custom.layers[0].get_weights()[0]   # (3,3,3,32)
fig, axs = plt.subplots(4, 8, figsize=(22, 8)); axs = axs.flatten()
for i in range(32):
    f = w[:,:,0,i]; f_norm = (f-f.min())/(f.max()-f.min()+1e-8)
    axs[i].imshow(f_norm, cmap='viridis'); axs[i].axis('off'); axs[i].set_title(f'#{i}', fontsize=6)
plt.suptitle('Custom CNN — First ConvLayer Filters (R channel, 32/32)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'T5_cnn_filters.png'), dpi=150); plt.close()
print("  Saved T5_cnn_filters.png")

print("\n[5] Activation maps ...")
sample_img = x_test[0:1]
conv_out = [l.output for l in model_custom.layers
            if isinstance(l,(layers.Conv2D, layers.Activation))]
# In Keras3, use inputs[0] for the input tensor
act_mod = keras.Model(inputs=model_custom.inputs[0], outputs=conv_out)
acts_all = act_mod.predict(sample_img, verbose=0)
# acts_all[4] = after conv1+bn+relu+pool
acts_c1 = acts_all[4]
fig, axs2 = plt.subplots(2, 8, figsize=(22, 5)); axs2 = axs2.flatten()
for i in range(min(16, acts_c1.shape[-1])):
    axs2[i].imshow(acts_c1[0,:,:,i], cmap='viridis'); axs2[i].axis('off')
    axs2[i].set_title(f'Ch {i}', fontsize=7)
plt.suptitle('CNN Activation Maps — Conv Block 1 (first test image)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'T5_cnn_activations.png'), dpi=150); plt.close()
print("  Saved T5_cnn_activations.png")

print("\n[6] Sample predictions ...")
idx = np.random.choice(len(x_test), 16, replace=False)
fig, axs3 = plt.subplots(4, 4, figsize=(16, 16))
for i, ix in enumerate(idx, 1):
    ax = axs3.flatten()[i-1]; ax.imshow(x_test[ix])
    cl = CLASS_NAMES[y_pred_c[ix]]; tl = CLASS_NAMES[y_true[ix]]
    ax.set_title(f'CNN:{cl}\nTrue:{tl}',
                 color=('green' if y_pred_c[ix]==y_true[ix] else 'red'), fontsize=8)
    ax.axis('off')
plt.suptitle('Custom CNN — CIFAR-10 Sample Predictions  (green=correct  red=wrong)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'T5_sample_predictions.png'), dpi=150); plt.close()
print("  Saved T5_sample_predictions.png")

print("DONE — results/ files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if not f.startswith('T5'): continue
    print(f"  {f}  {os.path.getsize(os.path.join(OUTPUT_DIR,f))//1024:6d} KB")
