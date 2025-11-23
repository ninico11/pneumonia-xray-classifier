"""
Script pentru descărcarea automată a dataset-ului de pe Kaggle
"""

import os
import sys
import zipfile
import shutil

def check_kaggle_setup():
    """
    Verifică dacă Kaggle API este configurat
    """
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_json):
        print("="*60)
        print("⚠️  CONFIGURARE KAGGLE API NECESARĂ")
        print("="*60)
        print("\nPași pentru configurare:")
        print("\n1. Accesează: https://www.kaggle.com/settings/account")
        print("2. Scroll până la secțiunea 'API'")
        print("3. Apasă 'Create New Token'")
        print("4. Se va descărca fișierul 'kaggle.json'")
        print(f"\n5. Mută 'kaggle.json' în: {kaggle_dir}")
        print("\nComenzi (Linux/Mac):")
        print(f"  mkdir -p {kaggle_dir}")
        print(f"  mv ~/Downloads/kaggle.json {kaggle_dir}")
        print(f"  chmod 600 {kaggle_json}")
        print("\nComenzi (Windows):")
        print(f"  mkdir {kaggle_dir}")
        print(f"  move Downloads\\kaggle.json {kaggle_dir}")
        print("="*60)
        return False
    
    return True

def download_dataset():
    """
    Descarcă dataset-ul de pe Kaggle
    """
    print("\n" + "="*60)
    print("DESCĂRCARE DATASET")
    print("="*60)
    
    # Verifică Kaggle API
    if not check_kaggle_setup():
        return False
    
    try:
        import kaggle
    except ImportError:
        print("\n❌ Kaggle nu este instalat!")
        print("\nInstalează cu:")
        print("  pip install kaggle")
        return False
    
    dataset_name = "paultimothymooney/chest-xray-pneumonia"
    
    print(f"\n📥 Descărcare dataset: {dataset_name}")
    print("Aceasta poate dura câteva minute (dataset ~2.3 GB)...")
    
    try:
        # Descarcă dataset-ul
        kaggle.api.dataset_download_files(
            dataset_name,
            path='.',
            unzip=True
        )
        print("\n✓ Dataset descărcat cu succes!")
        return True
        
    except Exception as e:
        print(f"\n❌ Eroare la descărcare: {e}")
        print("\nDacă întâmpini probleme:")
        print("1. Verifică că ai cont Kaggle și ești autentificat")
        print("2. Verifică conexiunea la internet")
        print("3. Descarcă manual de la:")
        print(f"   https://www.kaggle.com/datasets/{dataset_name}")
        return False

def verify_dataset_structure():
    """
    Verifică structura dataset-ului
    """
    print("\n" + "="*60)
    print("VERIFICARE STRUCTURĂ DATASET")
    print("="*60)
    
    expected_dirs = [
        'chest_xray/train/NORMAL',
        'chest_xray/train/PNEUMONIA',
        'chest_xray/val/NORMAL',
        'chest_xray/val/PNEUMONIA',
        'chest_xray/test/NORMAL',
        'chest_xray/test/PNEUMONIA'
    ]
    
    all_exist = True
    
    for dir_path in expected_dirs:
        exists = os.path.exists(dir_path)
        status = "✓" if exists else "✗"
        
        if exists:
            num_files = len([f for f in os.listdir(dir_path) 
                           if f.endswith(('.jpeg', '.jpg', '.png'))])
            print(f"{status} {dir_path:<40} ({num_files} imagini)")
        else:
            print(f"{status} {dir_path:<40} (LIPSĂ)")
            all_exist = False
    
    if all_exist:
        print("\n✓ Toate directoarele sunt prezente!")
        
        # Calculează total imagini
        total_images = 0
        for dir_path in expected_dirs:
            total_images += len([f for f in os.listdir(dir_path) 
                               if f.endswith(('.jpeg', '.jpg', '.png'))])
        
        print(f"\nTotal imagini: {total_images}")
        print("="*60)
        return True
    else:
        print("\n❌ Structura dataset-ului este incompletă!")
        return False

def main():
    """
    Funcția principală
    """
    print("="*60)
    print("SETUP DATASET PNEUMONIE")
    print("="*60)
    
    # Verifică dacă dataset-ul există deja
    if os.path.exists('chest_xray'):
        print("\n✓ Dataset-ul 'chest_xray' există deja!")
        
        response = input("\nDorești să verifici structura? (y/n): ").lower()
        if response == 'y':
            verify_dataset_structure()
        
        print("\nPoți începe antrenamentul cu:")
        print("  python pneumonia_detection.py")
        return
    
    print("\n📋 Opțiuni descărcare:")
    print("1. Descărcare automată (necesită Kaggle API)")
    print("2. Instrucțiuni pentru descărcare manuală")
    print("3. Ieșire")
    
    choice = input("\nAlege opțiunea (1/2/3): ").strip()
    
    if choice == '1':
        if download_dataset():
            if verify_dataset_structure():
                print("\n" + "="*60)
                print("✅ SETUP COMPLET!")
                print("="*60)
                print("\nPoți începe antrenamentul cu:")
                print("  python pneumonia_detection.py")
            else:
                print("\n⚠️  Dataset descărcat, dar structura pare incompletă.")
                print("Verifică manual folderul 'chest_xray'")
        
    elif choice == '2':
        print("\n" + "="*60)
        print("DESCĂRCARE MANUALĂ")
        print("="*60)
        print("\nPași:")
        print("\n1. Accesează:")
        print("   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("\n2. Apasă pe butonul 'Download' (necesită cont Kaggle)")
        print("\n3. Extrage arhiva 'chest-xray-pneumonia.zip' în directorul curent")
        print("\n4. Asigură-te că există folderul 'chest_xray' cu structura corectă")
        print("\n5. Verifică structura cu:")
        print("   python download_dataset.py")
        print("\n6. Începe antrenamentul cu:")
        print("   python pneumonia_detection.py")
        print("="*60)
        
    elif choice == '3':
        print("\nLa revedere!")
        
    else:
        print("\n❌ Opțiune invalidă!")

if __name__ == "__main__":
    main()
