# 🏥 Proiect Computer Vision - Diagnostic Medical Pneumonie

Sistem de detectare automată a pneumoniei din imagini cu raze X pulmonare folosind Deep Learning.

## 📊 Dataset

**Sursa:** Kaggle - Chest X-Ray Images (Pneumonia)

**Link:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Statistici Dataset:
- **Total imagini:** 5,856
- **Training:** 5,216 imagini
- **Validation:** 16 imagini
- **Test:** 624 imagini
- **Clase:** NORMAL (1,583 imagini) și PNEUMONIA (4,273 imagini)

## 🚀 Instalare și Setup

### Pasul 1: Instalează dependențele

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow
```

### Pasul 2: Descarcă dataset-ul

#### Opțiunea A: Manual (Recomandat)
1. Accesează: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Apasă pe butonul "Download" (necesită cont Kaggle)
3. Extrage arhiva `archive.zip` în directorul proiectului
4. Asigură-te că există folderul `chest_xray`

#### Opțiunea B: Folosind Kaggle API
```bash
# Instalează kaggle API
pip install kaggle

# Configurează API key (pune kaggle.json în ~/.kaggle/)
# Descarcă de aici: https://www.kaggle.com/settings/account

# Descarcă dataset-ul
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Extrage arhiva
unzip chest-xray-pneumonia.zip
```

### Structura Directoarelor Așteptată:

```
proiect/
│
├── pneumonia_detection.py      # Script principal de antrenament
├── predict.py                  # Script pentru predicții
├── README.md                   # Acest fișier
│
└── chest_xray/                 # Dataset descărcat
    ├── train/
    │   ├── NORMAL/             # 1,341 imagini
    │   └── PNEUMONIA/          # 3,875 imagini
    ├── val/
    │   ├── NORMAL/             # 8 imagini
    │   └── PNEUMONIA/          # 8 imagini
    └── test/
        ├── NORMAL/             # 234 imagini
        └── PNEUMONIA/          # 390 imagini
```

## 🎯 Utilizare

### 1. Antrenarea Modelului

```bash
python pneumonia_detection.py
```

Acest script va:
- Încărca și preprocesa datele
- Construi modelul (VGG16 cu Transfer Learning)
- Antrena modelul cu Early Stopping
- Evalua performanța pe setul de test
- Salva modelul și graficele

**Parametri ajustabili în cod:**
- `img_size`: Dimensiunea imaginilor (default: 224x224)
- `batch_size`: Mărimea batch-ului (default: 32)
- `epochs`: Număr maxim de epoci (default: 25)
- `learning_rate`: Rata de învățare (default: 0.0001)
- `use_pretrained`: True pentru VGG16, False pentru model custom

### 2. Predicții pe Imagini Noi

```bash
python predict.py path/to/xray_image.jpg
```

## 🏗️ Arhitectura Modelului

### Model 1: Transfer Learning cu VGG16 (Recomandat)
```
VGG16 (pre-antrenat pe ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, ReLU) + Dropout(0.5)
    ↓
Dense(256, ReLU) + Dropout(0.3)
    ↓
Dense(1, Sigmoid) → Probabilitate pneumonie
```

### Model 2: CNN Custom (Alternativă)
```
4 Blocuri Convoluționale:
- Block 1: 2x Conv2D(32) + MaxPool + Dropout(0.25)
- Block 2: 2x Conv2D(64) + MaxPool + Dropout(0.25)
- Block 3: 2x Conv2D(128) + MaxPool + Dropout(0.25)
- Block 4: 2x Conv2D(256) + MaxPool + Dropout(0.25)

Clasificare:
- Flatten
- Dense(512, ReLU) + Dropout(0.5)
- Dense(256, ReLU) + Dropout(0.3)
- Dense(1, Sigmoid)
```

## 📈 Rezultate Așteptate

Pe baza arhitecturii și dataset-ului, te poți aștepta la:

- **Accuracy:** ~92-95%
- **Precision:** ~90-93%
- **Recall:** ~95-97%
- **F1-Score:** ~92-95%
- **AUC-ROC:** ~96-98%

**Note:** Recall ridicat este esențial în aplicații medicale pentru a minimiza false negative (cazuri de pneumonie nedetectate).

## 🔧 Tehnici Utilizate

### Data Augmentation (doar pentru training)
- Rotație: ±20°
- Deplasare: ±20%
- Flip orizontal
- Zoom: ±20%
- Shear: ±20%

### Regularizare
- Dropout (0.3-0.5)
- Early Stopping (patience=5)
- L2 Regularization (implicit în VGG16)

### Optimizare
- Optimizer: Adam
- Learning Rate: 0.0001
- ReduceLROnPlateau (factor=0.5, patience=3)

## 📊 Metrici de Evaluare

Proiectul calculează următoarele metrici:

1. **Accuracy:** Proporția predicțiilor corecte
2. **Precision:** Din toate predicțiile pozitive, câte sunt corecte
3. **Recall (Sensitivity):** Din toate cazurile pozitive reale, câte sunt detectate
4. **F1-Score:** Media armonică între Precision și Recall
5. **AUC-ROC:** Aria sub curba ROC
6. **Confusion Matrix:** Matricea de confuzie detaliată

## 🎨 Output-uri Generate

După antrenament, vei găsi:

1. **best_pneumonia_model.keras** - Cel mai bun model (AUC maxim)
2. **pneumonia_detector_final.keras** - Model final
3. **training_history.png** - Grafice cu evoluția antrenamentului
4. **confusion_matrix.png** - Matricea de confuzie

## 💡 Sfaturi pentru Îmbunătățire

### 1. Îmbunătățirea Performanței
```python
# Încearcă modele mai puternice
from tensorflow.keras.applications import ResNet50, EfficientNetB0

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

### 2. Class Imbalance
Dataset-ul are mai multe cazuri de pneumonie. Pentru echilibrare:
```python
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))

# Adaugă în model.fit()
model.fit(..., class_weight=class_weight_dict)
```

### 3. Ensemble Learning
Combină predicții de la mai multe modele pentru robustețe crescută.

## ⚠️ Considerații Medicale Importante

**DISCLAIMER:** Acest sistem este doar în scop educațional și de cercetare. 

- ❌ NU trebuie folosit pentru diagnostic medical real
- ❌ NU înlocuiește diagnosticul unui medic calificat
- ✅ Poate fi folosit ca instrument de asistență/screening
- ✅ Necesită validare clinică înainte de utilizare în practică

## 📚 Referințe

- **Dataset:** Kermany et al. (2018) - "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"
- **Kaggle:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Paper:** http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

## 🤝 Contribuții

Pentru îmbunătățiri:
1. Testează pe mai multe dataset-uri externe
2. Implementează explicabilitate (Grad-CAM, LIME)
3. Adaugă detectarea altor boli pulmonare
4. Optimizează pentru deployment pe dispozitive mobile

## 📞 Contact & Suport

Pentru întrebări sau probleme:
- Verifică că ai descărcat corect dataset-ul
- Asigură-te că ai instalat toate dependențele
- Verifică că GPU-ul este disponibil pentru TensorFlow (opțional, dar recomandat)

---

**Dezvoltat pentru proiecte educaționale de Computer Vision în domeniul medical** 🏥🤖
