"""
Proiect Computer Vision - Diagnostic Medical
Detectarea Pneumoniei din Imagini cu Raze X Pulmonare

Dataset: Chest X-Ray Images (Pneumonia) de pe Kaggle
Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configurare seed pentru reproducibilitate
np.random.seed(42)
tf.random.set_seed(42)

class PneumoniaDetector:
    def __init__(self, img_size=(224, 224), batch_size=32):
        """
        Initializare detector pneumonie
        
        Args:
            img_size: Dimensiunea imaginilor (height, width)
            batch_size: Numărul de imagini per batch
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
    def prepare_data(self, data_dir):
        """
        Pregătirea datelor cu augmentare pentru training
        
        Args:
            data_dir: Directorul care conține folderele train/val/test
        """
        # Data augmentation pentru training - previne overfitting
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Doar normalizare pentru validare și test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Încărcarea datelor
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        self.val_generator = val_test_datagen.flow_from_directory(
            os.path.join(data_dir, 'val'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        self.test_generator = val_test_datagen.flow_from_directory(
            os.path.join(data_dir, 'test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        print(f"Classes: {self.train_generator.class_indices}")
        
    def build_model(self, use_pretrained=True):
        """
        Construirea modelului CNN
        
        Args:
            use_pretrained: Dacă True, folosește VGG16 pre-antrenat
        """
        if use_pretrained:
            # Transfer Learning cu VGG16
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
            
            # Înghețăm layerele din base model
            for layer in base_model.layers[:-4]:
                layer.trainable = False
                
            model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])
            
        else:
            # Model CNN custom
            model = keras.Sequential([
                layers.Input(shape=(*self.img_size, 3)),
                
                # Block 1
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Block 2
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Block 3
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Block 4
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Classification head
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])
        
        self.model = model
        print("\n" + "="*60)
        print("ARHITECTURA MODELULUI")
        print("="*60)
        self.model.summary()
        
    def compile_model(self, learning_rate=0.0001):
        """
        Compilarea modelului cu optimizer și loss function
        
        Args:
            learning_rate: Rata de învățare
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        print(f"\nModel compilat cu learning rate: {learning_rate}")
        
    def train(self, epochs=25, patience=5):
        """
        Antrenarea modelului
        
        Args:
            epochs: Numărul maxim de epoci
            patience: Numărul de epoci fără îmbunătățire înainte de oprire
        """
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_pneumonia_model.keras',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("\n" + "="*60)
        print("ÎNCEPEREA ANTRENAMENTULUI")
        print("="*60)
        
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Antrenament finalizat!")
        
    def evaluate(self):
        """
        Evaluarea modelului pe setul de test
        """
        print("\n" + "="*60)
        print("EVALUARE PE SETUL DE TEST")
        print("="*60)
        
        # Evaluare
        test_loss, test_acc, test_precision, test_recall, test_auc = \
            self.model.evaluate(self.test_generator, verbose=1)
        
        # Calculare F1-score
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        print("\n" + "="*60)
        print("REZULTATE FINALE")
        print("="*60)
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall:    {test_recall:.4f}")
        print(f"Test F1-Score:  {f1_score:.4f}")
        print(f"Test AUC:       {test_auc:.4f}")
        print("="*60)
        
        # Predicții pentru raport detaliat
        y_true = self.test_generator.classes
        y_pred_proba = self.model.predict(self.test_generator, verbose=1)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Classification report
        print("\n" + "="*60)
        print("RAPORT DE CLASIFICARE")
        print("="*60)
        class_names = list(self.test_generator.class_indices.keys())
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, class_names)
        
        return {
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': f1_score,
            'auc': test_auc
        }
    
    def plot_confusion_matrix(self, cm, class_names):
        """
        Afișarea confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix salvată în 'confusion_matrix.png'")
        plt.close()
        
    def plot_training_history(self):
        """
        Vizualizarea istoricului antrenamentului
        """
        if self.history is None:
            print("Nu există istoric de antrenament!")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Graficele de antrenament salvate în 'training_history.png'")
        plt.close()
        
    def predict_image(self, image_path):
        """
        Predicție pentru o singură imagine
        
        Args:
            image_path: Calea către imagine
        """
        img = keras.preprocessing.image.load_img(
            image_path, 
            target_size=self.img_size
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        
        class_names = ['NORMAL', 'PNEUMONIA']
        predicted_class = class_names[int(prediction > 0.5)]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        print(f"\nPredictie: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        
        return predicted_class, confidence
    
    def save_model(self, filepath='pneumonia_detector.keras'):
        """
        Salvarea modelului
        """
        self.model.save(filepath)
        print(f"\n✓ Model salvat în '{filepath}'")
        
    def load_model(self, filepath):
        """
        Încărcarea unui model salvat
        """
        self.model = keras.models.load_model(filepath)
        print(f"\n✓ Model încărcat din '{filepath}'")


def main():
    """
    Funcția principală pentru antrenarea modelului
    """
    print("="*60)
    print("PROIECT COMPUTER VISION - DETECTAREA PNEUMONIEI")
    print("="*60)
    
    # Calea către date (modifică după descărcarea datasetului)
    data_dir = 'chest_xray'
    
    if not os.path.exists(data_dir):
        print("\n⚠ ATENȚIE: Dataset-ul nu a fost găsit!")
        print("\nPași pentru a obține datele:")
        print("1. Accesează: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("2. Descarcă dataset-ul (5,856 imagini)")
        print("3. Extrage arhiva în directorul curent")
        print("4. Asigură-te că există folderul 'chest_xray' cu subdirectoarele train/val/test")
        print("\nStructura așteptată:")
        print("chest_xray/")
        print("  ├── train/")
        print("  │   ├── NORMAL/")
        print("  │   └── PNEUMONIA/")
        print("  ├── val/")
        print("  │   ├── NORMAL/")
        print("  │   └── PNEUMONIA/")
        print("  └── test/")
        print("      ├── NORMAL/")
        print("      └── PNEUMONIA/")
        return
    
    # Inițializare detector
    detector = PneumoniaDetector(img_size=(224, 224), batch_size=32)
    
    # Pregătirea datelor
    print("\n1. Pregătirea datelor...")
    detector.prepare_data(data_dir)
    
    # Construirea modelului (folosește use_pretrained=False pentru model custom)
    print("\n2. Construirea modelului...")
    detector.build_model(use_pretrained=True)
    
    # Compilarea modelului
    print("\n3. Compilarea modelului...")
    detector.compile_model(learning_rate=0.0001)
    
    # Antrenarea modelului
    print("\n4. Antrenarea modelului...")
    detector.train(epochs=25, patience=5)
    
    # Vizualizarea istoricului
    print("\n5. Generarea graficelor...")
    detector.plot_training_history()
    
    # Evaluarea pe setul de test
    print("\n6. Evaluarea modelului...")
    results = detector.evaluate()
    
    # Salvarea modelului
    print("\n7. Salvarea modelului...")
    detector.save_model('pneumonia_detector_final.keras')
    
    print("\n" + "="*60)
    print("ANTRENAMENT FINALIZAT CU SUCCES!")
    print("="*60)
    print("\nFișiere generate:")
    print("  ✓ best_pneumonia_model.keras (cel mai bun model)")
    print("  ✓ pneumonia_detector_final.keras (model final)")
    print("  ✓ training_history.png (grafice antrenament)")
    print("  ✓ confusion_matrix.png (matrice confuzie)")
    print("="*60)


if __name__ == "__main__":
    main()
