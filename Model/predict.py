"""
Script pentru predicții pe imagini noi cu raze X
Utilizare: python predict.py <cale_imagine>
"""

import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image

class PneumoniaPredictor:
    def __init__(self, model_path='best_pneumonia_model.keras', img_size=(224, 224)):
        """
        Inițializare predictor
        
        Args:
            model_path: Calea către modelul salvat
            img_size: Dimensiunea imaginilor
        """
        self.img_size = img_size
        self.model = None
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """
        Încarcă modelul salvat
        """
        if not os.path.exists(model_path):
            print(f"❌ Eroare: Modelul '{model_path}' nu a fost găsit!")
            print("\nAsigură-te că ai antrenat modelul mai întâi:")
            print("  python pneumonia_detection.py")
            sys.exit(1)
            
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✓ Model încărcat cu succes din '{model_path}'")
        except Exception as e:
            print(f"❌ Eroare la încărcarea modelului: {e}")
            sys.exit(1)
    
    def preprocess_image(self, image_path):
        """
        Preprocesează imaginea pentru predicție
        """
        if not os.path.exists(image_path):
            print(f"❌ Eroare: Imaginea '{image_path}' nu există!")
            sys.exit(1)
        
        try:
            # Încarcă și redimensionează imaginea
            img = keras.preprocessing.image.load_img(
                image_path, 
                target_size=self.img_size
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalizare
            
            return img, img_array
        except Exception as e:
            print(f"❌ Eroare la procesarea imaginii: {e}")
            sys.exit(1)
    
    def predict(self, image_path, visualize=True):
        """
        Face predicție pe o imagine
        
        Args:
            image_path: Calea către imaginea X-ray
            visualize: Dacă True, afișează imaginea cu predicția
        """
        print("\n" + "="*60)
        print("ANALIZĂ IMAGINE RAZE X")
        print("="*60)
        print(f"Imagine: {image_path}")
        
        # Preprocesare
        img, img_array = self.preprocess_image(image_path)
        
        # Predicție
        print("\nSe face predicția...")
        prediction_proba = self.model.predict(img_array, verbose=0)[0][0]
        
        # Interpretare rezultat
        is_pneumonia = prediction_proba > 0.5
        confidence = prediction_proba if is_pneumonia else 1 - prediction_proba
        
        predicted_class = "PNEUMONIE" if is_pneumonia else "NORMAL"
        
        # Afișare rezultate
        print("\n" + "="*60)
        print("REZULTAT DIAGNOSTIC")
        print("="*60)
        print(f"Predicție: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Probabilitate pneumonie: {prediction_proba:.4f}")
        print("="*60)
        
        # Interpretare clinică
        self._interpret_results(predicted_class, confidence)
        
        # Vizualizare
        if visualize:
            self.visualize_prediction(img, predicted_class, confidence, image_path)
        
        return predicted_class, confidence, prediction_proba
    
    def _interpret_results(self, predicted_class, confidence):
        """
        Oferă interpretare clinică a rezultatelor
        """
        print("\n📋 INTERPRETARE:")
        
        if predicted_class == "PNEUMONIE":
            if confidence >= 0.9:
                print("  🔴 Risc RIDICAT de pneumonie")
                print("  → Recomandare: Consultație medicală urgentă")
            elif confidence >= 0.7:
                print("  🟠 Risc MODERAT de pneumonie")
                print("  → Recomandare: Consultație medicală în cel mai scurt timp")
            else:
                print("  🟡 Risc SCĂZUT de pneumonie")
                print("  → Recomandare: Evaluare medicală pentru confirmare")
        else:
            if confidence >= 0.9:
                print("  🟢 Probabilitate FOARTE SCĂZUTĂ de pneumonie")
                print("  → Plămânii par normali")
            elif confidence >= 0.7:
                print("  🟢 Probabilitate SCĂZUTĂ de pneumonie")
                print("  → Rezultat probabil negativ")
            else:
                print("  🟡 Rezultat INCERT")
                print("  → Recomandare: Evaluare medicală pentru clarificare")
        
        print("\n⚠️  ATENȚIE: Acest sistem este doar un instrument de asistență.")
        print("    Diagnosticul final trebuie stabilit de un medic calificat.")
    
    def visualize_prediction(self, img, predicted_class, confidence, image_path):
        """
        Vizualizează imaginea cu predicția
        """
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')
        
        # Culoare în funcție de rezultat
        color = 'red' if predicted_class == "PNEUMONIE" else 'green'
        
        # Titlu
        title = f"Predicție: {predicted_class}\nConfidence: {confidence:.2%}"
        plt.title(title, fontsize=16, fontweight='bold', color=color, pad=20)
        
        plt.tight_layout()
        
        # Salvare
        output_path = f"prediction_{os.path.basename(image_path)}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Vizualizare salvată în '{output_path}'")
        
        # Afișare (opțional)
        try:
            plt.show()
        except:
            pass  # Ignoră dacă nu poate afișa
        
        plt.close()
    
    def predict_batch(self, image_paths, save_results=True):
        """
        Face predicții pe mai multe imagini
        
        Args:
            image_paths: Lista cu căile către imagini
            save_results: Dacă True, salvează rezultatele într-un fișier
        """
        results = []
        
        print("\n" + "="*60)
        print(f"PROCESARE BATCH: {len(image_paths)} imagini")
        print("="*60)
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Procesare: {image_path}")
            
            try:
                predicted_class, confidence, proba = self.predict(
                    image_path, 
                    visualize=False
                )
                
                results.append({
                    'image': image_path,
                    'prediction': predicted_class,
                    'confidence': confidence,
                    'probability_pneumonia': proba
                })
                
            except Exception as e:
                print(f"  ❌ Eroare: {e}")
                results.append({
                    'image': image_path,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'probability_pneumonia': 0.0
                })
        
        # Salvare rezultate
        if save_results:
            self._save_batch_results(results)
        
        return results
    
    def _save_batch_results(self, results):
        """
        Salvează rezultatele batch într-un fișier CSV
        """
        import csv
        
        filename = 'batch_predictions.csv'
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'prediction', 'confidence', 'probability_pneumonia'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✓ Rezultate salvate în '{filename}'")
        
        # Statistici
        total = len(results)
        pneumonia_count = sum(1 for r in results if r['prediction'] == 'PNEUMONIE')
        normal_count = sum(1 for r in results if r['prediction'] == 'NORMAL')
        
        print("\n" + "="*60)
        print("STATISTICI BATCH")
        print("="*60)
        print(f"Total imagini: {total}")
        print(f"Pneumonie detectată: {pneumonia_count} ({pneumonia_count/total*100:.1f}%)")
        print(f"Normal: {normal_count} ({normal_count/total*100:.1f}%)")
        print("="*60)


def main():
    """
    Funcția principală
    """
    print("="*60)
    print("SISTEM DE DETECTARE PNEUMONIE")
    print("="*60)
    
    # Verificare argumente
    if len(sys.argv) < 2:
        print("\n❌ Utilizare incorectă!")
        print("\nUtilizare:")
        print("  Predicție simplă:")
        print("    python predict.py <cale_imagine>")
        print("\n  Predicție batch:")
        print("    python predict.py <imagine1> <imagine2> <imagine3> ...")
        print("\nExemple:")
        print("  python predict.py chest_xray/test/NORMAL/IM-0001-0001.jpeg")
        print("  python predict.py xray1.jpg xray2.jpg xray3.jpg")
        sys.exit(1)
    
    # Inițializare predictor
    try:
        predictor = PneumoniaPredictor(model_path='best_pneumonia_model.keras')
    except:
        # Încearcă cu modelul final dacă cel mai bun nu există
        try:
            predictor = PneumoniaPredictor(model_path='pneumonia_detector_final.keras')
        except:
            print("\n❌ Niciun model găsit! Antrenează modelul mai întâi:")
            print("  python pneumonia_detection.py")
            sys.exit(1)
    
    # Predicții
    image_paths = sys.argv[1:]
    
    if len(image_paths) == 1:
        # Predicție simplă
        predictor.predict(image_paths[0], visualize=True)
    else:
        # Predicție batch
        predictor.predict_batch(image_paths, save_results=True)


if __name__ == "__main__":
    main()
