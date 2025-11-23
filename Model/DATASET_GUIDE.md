# 📥 GHID COMPLET PENTRU DESCĂRCAREA DATASET-ULUI

## Dataset: Chest X-Ray Images (Pneumonia)

---

## 📊 INFORMAȚII DESPRE DATASET

**Sursa:** Kaggle  
**Autor:** Paul Mooney  
**Link Direct:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia  
**Dimensiune:** ~2.3 GB (arhivă comprimată)  
**Format:** Imagini JPEG  
**Licență:** CC BY 4.0

### Statistici Complete:
```
Total Imagini: 5,856

Training Set: 5,216 imagini
  ├── NORMAL: 1,341 imagini
  └── PNEUMONIA: 3,875 imagini

Validation Set: 16 imagini
  ├── NORMAL: 8 imagini
  └── PNEUMONIA: 8 imagini

Test Set: 624 imagini
  ├── NORMAL: 234 imagini
  └── PNEUMONIA: 390 imagini
```

---

## 🚀 METODĂ 1: DESCĂRCARE MANUALĂ (RECOMANDATĂ)

### Pas 1: Creează Cont Kaggle (dacă nu ai deja)
1. Accesează: https://www.kaggle.com/account/login
2. Înregistrează-te gratuit cu email sau Google

### Pas 2: Descarcă Dataset-ul
1. **Accesează link-ul dataset-ului:**  
   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

2. **Apasă pe butonul albastru "Download" (în colțul din dreapta sus)**

3. **Se va descărca fișierul:** `chest-xray-pneumonia.zip` (~570 MB)

### Pas 3: Extrage Arhiva
**Windows:**
```cmd
# Folosește Windows Explorer
Click dreapta pe fișier → Extract All
SAU folosește 7-Zip/WinRAR
```

**Linux/Mac:**
```bash
cd /calea/catre/proiect
unzip chest-xray-pneumonia.zip
```

### Pas 4: Verifică Structura
După extragere, ar trebui să ai:
```
proiect/
├── chest_xray/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
└── [alte fișiere proiect]
```

---

## 🤖 METODĂ 2: DESCĂRCARE AUTOMATĂ (CU KAGGLE API)

### Pas 1: Instalează Kaggle API
```bash
pip install kaggle
```

### Pas 2: Configurează API Key

#### A. Obține API Key
1. Loghează-te pe Kaggle: https://www.kaggle.com
2. Accesează: https://www.kaggle.com/settings/account
3. Scroll până la secțiunea **"API"**
4. Apasă **"Create New Token"**
5. Se va descărca `kaggle.json`

#### B. Instalează API Key

**Linux/Mac:**
```bash
# Creează directorul
mkdir -p ~/.kaggle

# Mută fișierul
mv ~/Downloads/kaggle.json ~/.kaggle/

# Setează permisiuni (important!)
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**
```cmd
# Creează directorul
mkdir %USERPROFILE%\.kaggle

# Mută fișierul
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\

# Nu e nevoie de chmod pe Windows
```

### Pas 3: Descarcă Dataset-ul
```bash
# Navighează în folderul proiectului
cd /calea/catre/proiect

# Descarcă și extrage automat
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia --unzip
```

SAU folosește scriptul nostru:
```bash
python download_dataset.py
```

---

## 🛠️ METODĂ 3: DESCĂRCARE CU SCRIPTUL PROIECTULUI

### Rulează Script-ul de Descărcare
```bash
python download_dataset.py
```

Scriptul va:
1. ✅ Verifica dacă Kaggle API este configurat
2. ✅ Descărca automat dataset-ul
3. ✅ Extrage arhiva
4. ✅ Verifica structura folderelor
5. ✅ Afișa statistici despre imagini

---

## ✅ VERIFICAREA INSTALĂRII

### Scriptul Python
```bash
python download_dataset.py
# Selectează opțiunea pentru verificare
```

### Manual
```bash
# Linux/Mac
ls -R chest_xray/

# Windows
dir /s chest_xray\
```

### Verificare Rapidă
Ar trebui să vezi:
- ✅ Folderul `chest_xray` există
- ✅ 6 subdirectoare (train/val/test × NORMAL/PNEUMONIA)
- ✅ 5,856 imagini în total

---

## 🔧 REZOLVAREA PROBLEMELOR

### Problema 1: "kaggle.json not found"
**Soluție:** Asigură-te că ai urmat pașii de configurare API și că `kaggle.json` este în locația corectă:
- Linux/Mac: `~/.kaggle/kaggle.json`
- Windows: `%USERPROFILE%\.kaggle\kaggle.json`

### Problema 2: "403 Forbidden" sau "401 Unauthorized"
**Soluție:** 
1. Verifică că ai un cont Kaggle valid
2. Re-generează API key-ul din setări
3. Înlocuiește vechiul `kaggle.json` cu cel nou

### Problema 3: Descărcarea este foarte lentă
**Soluție:**
- Folosește descărcarea manuală (poate fi mai rapidă)
- Verifică conexiunea la internet
- Încearcă într-o altă perioadă a zilei

### Problema 4: "Out of disk space"
**Soluție:**
- Dataset-ul necesită ~3 GB spațiu liber
- Eliberează spațiu pe disk
- Verifică: `df -h` (Linux/Mac) sau `Properties` pe drive (Windows)

### Problema 5: Structura folderelor este greșită
**Soluție:**
```bash
# Asigură-te că extragi în locația corectă
# Structura trebuie să fie:
# proiect/chest_xray/train/NORMAL/...
# NU proiect/train/NORMAL/...
```

---

## 📂 STRUCTURA FINALĂ AȘTEPTATĂ

```
proiectul_tau/
│
├── pneumonia_detection.py          # Script antrenament
├── predict.py                      # Script predicții
├── download_dataset.py             # Script descărcare
├── requirements.txt                # Dependențe
├── README.md                       # Documentație
│
└── chest_xray/                     # DATASET
    │
    ├── train/                      # 5,216 imagini
    │   ├── NORMAL/                 # 1,341 imagini
    │   │   ├── IM-0001-0001.jpeg
    │   │   ├── IM-0002-0001.jpeg
    │   │   └── ...
    │   └── PNEUMONIA/              # 3,875 imagini
    │       ├── person1_bacteria_1.jpeg
    │       ├── person1_virus_2.jpeg
    │       └── ...
    │
    ├── val/                        # 16 imagini
    │   ├── NORMAL/                 # 8 imagini
    │   └── PNEUMONIA/              # 8 imagini
    │
    └── test/                       # 624 imagini
        ├── NORMAL/                 # 234 imagini
        └── PNEUMONIA/              # 390 imagini
```

---

## 🎯 DUPĂ DESCĂRCARE

### Verifică Instalarea
```bash
python download_dataset.py
# Selectează opțiunea de verificare
```

### Începe Antrenamentul
```bash
python pneumonia_detection.py
```

---

## 📊 INFORMAȚII DESPRE IMAGINI

### Format
- **Tip:** JPEG
- **Dimensiuni:** Variate (vor fi redimensionate automat la 224×224)
- **Canale:** RGB (3 canale)
- **Calitate:** Variată (imagini medicale reale)

### Tipuri de Pneumonie în Dataset
- **Bacteriană:** Pneumonie cauzată de bacterii
- **Virală:** Pneumonie cauzată de virusuri
- **Normal:** Plămâni sănătoși (fără pneumonie)

### Sursă Imagini
Imagini provin de la:
- Copii de 1-5 ani
- Guangzhou Women and Children's Medical Center
- Colectate între 2013-2018
- Validate de experți medicali

---

## 📚 REFERINȚE

**Paper Original:**
- Kermany et al. (2018)
- "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"
- Cell, Volume 172, Issue 5
- DOI: 10.1016/j.cell.2018.02.010
- Link: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

**Dataset Kaggle:**
- https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## ⏱️ TIMP ESTIMAT

### Descărcare
- **Internet rapid (50+ Mbps):** 5-10 minute
- **Internet mediu (10-50 Mbps):** 10-20 minute
- **Internet lent (<10 Mbps):** 20-40 minute

### Extragere
- **SSD:** 2-5 minute
- **HDD:** 5-10 minute

### Total
- **15-30 minute** pentru întregul proces

---

## 💡 SFATURI

1. **Verifică spațiul pe disk înainte** (~3 GB necesar)
2. **Folosește conexiune stabilă** (Wi-Fi sau Ethernet)
3. **Nu întrerupe descărcarea** (poate corupe fișierul)
4. **Verifică integritatea** după descărcare
5. **Păstrează arhiva originală** (backup)

---

## 🆘 AJUTOR SUPLIMENTAR

### Dacă întâmpini probleme:

1. **Citește erorile cu atenție** - de obicei indică problema
2. **Verifică README.md** pentru mai multe detalii
3. **Încearcă descărcarea manuală** dacă automata eșuează
4. **Verifică spațiul pe disk** și permisiunile
5. **Re-descarcă** dacă fișierul pare corupt

---

## ✅ CHECKLIST FINAL

Înainte de a începe antrenamentul, asigură-te că:

- [ ] Dataset-ul este descărcat complet
- [ ] Arhiva este extrasă corect
- [ ] Folderul `chest_xray` există în proiect
- [ ] Există 6 subdirectoare (train/val/test × NORMAL/PNEUMONIA)
- [ ] Fiecare subdirector conține imagini JPEG
- [ ] Total ~5,856 imagini sunt prezente
- [ ] Scriptul de verificare rulează fără erori

---

**Gata! Acum poți începe antrenamentul! 🚀**

```bash
python pneumonia_detection.py
```

---

*Pentru mai multe detalii, consultă README.md sau QUICK_START.md*
