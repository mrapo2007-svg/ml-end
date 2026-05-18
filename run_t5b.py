"""
T5 Step B — ResNet50 Transfer Learning (FROZEN ONLY, 5 epochs)
NOTE: ResNet50 training on CPU is 400-800s/epoch. This script is <5 mins total.
Confirms only the frozen trainable-head ResNet50 path.
"""
import os, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings; warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.random.set_seed(42); np.random.seed(42)
sns.set_theme(style='darkgrid')
CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
sys.stdout.reconfigure(line_buffering=True)

print("[*] Load CIFAR-10 ...", flush=True)
(x_all,y_all),(x_test,y_test) = keras.datasets.cifar10.load_data()
x_all=x_all.astype('float32')/255.0; x_test=x_test.astype('float32')/255.0
x_val,y_val=x_all[-5000:],y_all[-5000:]
x_train,y_train=x_all[:-5000],y_all[:-5000]
y_tr_c=keras.utils.to_categorical(y_train,10); y_va_c=keras.utils.to_categorical(y_val,10); y_te_c=keras.utils.to_categorical(y_test,10)
y_true=y_test.flatten()
print(f"    train/{x_train.shape}  val/{x_val.shape}  test/{x_test.shape}", flush=True)

print("[*] Build ResNet50 frozen + classification head ...", flush=True)
base = ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3))
base.trainable = False
inputs = keras.Input(shape=(32,32,3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(10, activation='softmax')(x)
model_rn = keras.Model(inputs=inputs, outputs=outputs, name='rn50_frozen')
model_rn.compile(optimizer=keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

EPOCHS = 5   # frozen, no fine-tuning — fast
print(f"[*] Train {EPOCHS} epochs (bs=128, frozen base) ...", flush=True)
print(f"    Steps per epoch: {len(x_train)//128}  (+{len(x_train)%128} remainder)",
      flush=True)
hist = model_rn.fit(
    x_train, y_tr_c, batch_size=128, shuffle=True,
    validation_data=(x_val, y_va_c),
    epochs=EPOCHS, verbose=1
)
print("[*] Save history plot ...", flush=True)
er = range(1, len(hist.history['accuracy'])+1)
fig,(a,b)=plt.subplots(1,2,figsize=(14,5))
a.plot(er,hist.history['accuracy'], 'r--',label='Train'); a.plot(er,hist.history['val_accuracy'],'r-',label='Val')
a.set_title('ResNet-50 Transfer — Accuracy'); a.legend(); a.grid(alpha=0.3)
b.plot(er,hist.history['loss'],     'r--',label='Train'); b.plot(er,hist.history['val_loss'],    'r-',label='Val')
b.set_title('ResNet-50 Transfer — Loss');   b.legend(); b.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'T5_resnet50_training_history.png'),dpi=150); plt.close()
print("  Saved T5_resnet50_training_history.png", flush=True)

print("[*] Test eval ...", flush=True)
test_loss_r, test_acc_r = model_rn.evaluate(x_test, y_te_c, verbose=0)
print(f"    ResNet-50 acc={test_acc_r:.4f}  loss={test_loss_r:.4f}", flush=True)
y_pred_r = np.argmax(model_rn.predict(x_test, verbose=0), axis=1)
cm_r = confusion_matrix(y_true, y_pred_r)
fig, ax = plt.subplots(figsize=(9,7))
ConfusionMatrixDisplay(cm_r, display_labels=CLASS_NAMES).plot(cmap='Purples', ax=ax)
for im in ax.get_images(): im.colorbar = None
fig.colorbar(ax.get_images()[0], ax=ax)
plt.title(f'ResNet-50 Frozen Transfer — Confusion Matrix  (acc={test_acc_r:.4f})')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'T5_resnet50_confusion_matrix.png'),dpi=150); plt.close()
print("  Saved T5_resnet50_confusion_matrix.png", flush=True)
print(classification_report(y_true, y_pred_r, target_names=CLASS_NAMES))

with open(os.path.join(OUTPUT_DIR,'resnet50_metrics.txt'),'w') as f:
    f.write(f'{test_acc_r}\n{test_loss_r}\n')
print("[*] Saved resnet50_metrics.txt", flush=True)
print("[*] DONE.", flush=True)
