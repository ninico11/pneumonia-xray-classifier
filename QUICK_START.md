# 🚀 GHID RAPID DE START

## Proiect Computer Vision - Diagnostic Medical Pneumonie

---

## 📦 CONȚINUTUL ARHIVEI

```
pneumonia_detection_project/
│
├── pneumonia_detection.py      # Script principal de antrenament
├── predict.py                  # Script pentru predicții
├── download_dataset.py         # Script pentru descărcarea dataset-ului
├── pneumonia_detection.ipynb   # Jupyter Notebook (alternativă)
├── requirements.txt            # Dependențe Python
├── README.md                   # Documentație completă
└── QUICK_START.md             # Acest fișier
```

---

```bash
cd Model
```

## ⚡ START RAPID (3 PAȘI)

### 1️⃣ Instalează Dependențele

```bash
pip install -r requirements.txt
```

**Dependențe principale:**
- TensorFlow >= 2.15.0
- NumPy, Matplotlib, Seaborn
- Scikit-learn, Pillow
- Kaggle (pentru descărcarea dataset-ului)

---

### 2️⃣ Descarcă Dataset-ul

#### Opțiunea A: Automată (cu Kaggle API)

```bash
# Configurează Kaggle API (vezi instrucțiuni mai jos)
python download_dataset.py
```

**Configurare Kaggle API:**
1. Creează cont pe [Kaggle](https://www.kaggle.com)
2. Accesează: https://www.kaggle.com/settings/account
3. Descarcă `kaggle.json` (secțiunea API)
4. Plasează în `~/.kaggle/kaggle.json`
5. Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`

#### Opțiunea B: Manuală (Recomandată)

1. Accesează: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Apasă "Download" (necesită cont Kaggle)
3. Extrage `chest-xray-pneumonia.zip` în folderul proiectului
4. Verifică că există `chest_xray/` cu subdirectoare `train/`, `val/`, `test/`

---

### 3️⃣ Antrenează Modelul

```bash
python pneumonia_detection.py
```

**Durata estimată:** 15-30 minute (depinde de hardware)

**Hardware recomandat:**
- ✅ GPU NVIDIA (CUDA): ~15-20 minute
- ⚠️ CPU: ~30-60 minute (mai lent, dar funcțional)

---

## 📊 REZULTATE AȘTEPTATE

După antrenare, vei avea:

### Fișiere Generate:
- ✅ `best_pneumonia_model.keras` - Cel mai bun model (AUC maxim)
- ✅ `pneumonia_detector_final.keras` - Model final
- ✅ `training_history.png` - Grafice de antrenament
- ✅ `confusion_matrix.png` - Matricea de confuzie

### Metrici de Performanță:
- **Accuracy:** ~92-95%
- **Precision:** ~90-93%
- **Recall:** ~95-97%
- **F1-Score:** ~92-95%
- **AUC-ROC:** ~96-98%

---

## 🔮 UTILIZAREA MODELULUI

### Predicție pe o Singură Imagine

```bash
python predict.py chest_xray/test/PNEUMONIA/person1_virus_6.jpeg
```

### Predicție pe Mai Multe Imagini (Batch)

```bash
python predict.py image1.jpg image2.jpg image3.jpg
```

**Output:** Fișier CSV cu toate predicțiile (`batch_predictions.csv`)

---

```bash
cd App
```

```bash
streamlit run app.py
```

## 💻 ALTERNATIVĂ: JUPYTER NOTEBOOK

Dacă preferi să lucrezi în Jupyter:

```bash
jupyter notebook pneumonia_detection.ipynb
```

Apoi rulează celulele pas cu pas pentru a vedea procesul interactiv.

---

## 🎯 STRUCTURA DATASET-ULUI

```
chest_xray/
├── train/
│   ├── NORMAL/       (1,341 imagini)
│   └── PNEUMONIA/    (3,875 imagini)
├── val/
│   ├── NORMAL/       (8 imagini)
│   └── PNEUMONIA/    (8 imagini)
└── test/
    ├── NORMAL/       (234 imagini)
    └── PNEUMONIA/    (390 imagini)

Total: 5,856 imagini
```

---

## 🛠️ PERSONALIZARE PARAMETRI

Editează parametrii în `pneumonia_detection.py`:

```python
IMG_SIZE = (224, 224)      # Dimensiune imagini
BATCH_SIZE = 32            # Mărime batch
EPOCHS = 25                # Epoci maxime
LEARNING_RATE = 0.0001     # Rată învățare
```

**Pentru model custom** (fără Transfer Learning):
```python
detector.build_model(use_pretrained=False)
```

---

## ⚠️ TROUBLESHOOTING

### Eroare: "Dataset not found"
```bash
# Verifică structura
python download_dataset.py
```

### Eroare: "Out of Memory"
```python
# Reduce batch_size în cod
BATCH_SIZE = 16  # sau 8
```

### Eroare: "TensorFlow GPU not found"
```bash
# Verifică CUDA
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Antrenamentul este prea lent
- Reduce dimensiunea imaginilor: `IMG_SIZE = (128, 128)`
- Reduce numărul de epoci: `EPOCHS = 10`
- Folosește un model mai simplu: `use_pretrained=False`

---

## 🎓 ÎMBUNĂTĂȚIRI POSIBILE

### 1. Modele Mai Avansate
```python
from tensorflow.keras.applications import EfficientNetB0, ResNet50

base_model = EfficientNetB0(weights='imagenet', include_top=False)
```

### 2. Class Balancing
```python
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
```

### 3. Explicabilitate (Grad-CAM)
Implementează vizualizări pentru a vedea ce zone ale plămânilor influențează predicția.

---

## 📚 RESURSE UTILE

- **Dataset Kaggle:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Paper Original:** http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
- **TensorFlow Docs:** https://www.tensorflow.org/tutorials
- **Transfer Learning:** https://www.tensorflow.org/tutorials/images/transfer_learning

---

## ⚕️ DISCLAIMER MEDICAL

**IMPORTANT:** Acest proiect este doar în scop educațional!

- ❌ NU folosiți pentru diagnostic medical real
- ❌ NU înlocuiește un medic calificat
- ✅ Poate fi folosit ca instrument de asistență/cercetare
- ✅ Necesită validare clinică pentru utilizare reală

---

## 🆘 SUPORT

### Întrebări Frecvente

**Q: Pot rula pe CPU?**
A: Da, dar va dura mai mult. Reduce `BATCH_SIZE` și `EPOCHS` pentru viteze mai bune.

**Q: Cât spațiu pe disk necesită?**
A: ~3 GB pentru dataset + ~500 MB pentru modele

**Q: Care este acuratețea așteptată?**
A: 92-95% pe setul de test, dar poate varia

**Q: Pot folosi propriile mele imagini?**
A: Da! Folosește `predict.py` cu orice imagine X-ray pulmonară

---

## 📧 CONTACT

Pentru probleme sau îmbunătățiri:
1. Verifică README.md pentru documentație completă
2. Testează cu diferite configurații
3. Citește erorile cu atenție - de obicei indică problema

---

**Succes cu proiectul tău de Computer Vision! 🚀🏥**

---

### Checklist Final ✅

- [ ] Am instalat dependențele (`pip install -r requirements.txt`)
- [ ] Am descărcat dataset-ul (chest_xray/)
- [ ] Am verificat structura dataset-ului
- [ ] Am rulat antrenamentul (`python pneumonia_detection.py`)
- [ ] Am testat predicțiile (`python predict.py <image>`)
- [ ] Am analizat rezultatele (grafice, metrici)

### Următorii Pași 🎯

1. Experimentează cu parametri diferiți
2. Încearcă modele alternative (ResNet, EfficientNet)
3. Testează pe imagini din surse externe
4. Implementează explicabilitate (Grad-CAM)
5. Consideră deployment (Flask API, Streamlit app)

**Mult succes! 🎉**
