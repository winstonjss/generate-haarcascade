import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

class PlateDetectorTrainer:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier()
        
    def prepare_training_data(self, positive_dir, negative_dir):
        """Prepara los datos de entrenamiento"""
        print("Preparando datos de entrenamiento...")
        
        # Verifica que los directorios existan
        if not os.path.exists(positive_dir):
            os.makedirs(positive_dir)
        if not os.path.exists(negative_dir):
            os.makedirs(negative_dir)
            
        # Lista para almacenar las características
        features = []
        labels = []
        
        # Procesar imágenes positivas
        for filename in os.listdir(positive_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(positive_dir, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Convertir a escala de grises
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Redimensionar a un tamaño estándar
                resized = cv2.resize(gray, (64, 32))
                
                # Extraer características HOG
                hog = cv2.HOGDescriptor((64,32), (16,16), (8,8), (8,8), 9)
                features.append(hog.compute(resized))
                labels.append(1)
        
        # Procesar imágenes negativas
        for filename in os.listdir(negative_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(negative_dir, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 32))
                features.append(hog.compute(resized))
                labels.append(0)
        
        return np.array(features), np.array(labels)
    
    def train(self, features, labels):
        """Entrena el detector usando SVM"""
        print("Entrenando el detector...")
        
        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Crear y entrenar SVM
        svm = cv2.ml.SVM_create()
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.train(X_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.int32))
        
        # Guardar el modelo
        svm.save('plate_detector_svm.xml')
        print("Modelo guardado como 'plate_detector_svm.xml'")
        
        # Evaluar el modelo
        _, y_pred = svm.predict(X_test.astype(np.float32))
        accuracy = np.mean(y_pred == y_test)
        print(f"Precisión del modelo: {accuracy * 100:.2f}%")
        
        return svm
    
    def detect_plates(self, image_path, svm):
        """Detecta placas en una imagen usando el modelo entrenado"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"No se pudo cargar la imagen: {image_path}")
            return None
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Ventana deslizante
        window_size = (64, 32)
        step_size = 32
        detections = []
        
        hog = cv2.HOGDescriptor((64,32), (16,16), (8,8), (8,8), 9)
        
        for y in range(0, gray.shape[0] - window_size[1], step_size):
            for x in range(0, gray.shape[1] - window_size[0], step_size):
                window = gray[y:y + window_size[1], x:x + window_size[0]]
                window = cv2.resize(window, (64, 32))
                
                features = hog.compute(window)
                
                # Predecir
                _, pred = svm.predict(features.astype(np.float32).reshape(1, -1))
                
                if pred[0][0] == 1:
                    detections.append((x, y, x + window_size[0], y + window_size[1]))
        
        # Dibujar detecciones
        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return img

def main():
    trainer = PlateDetectorTrainer()
    
    # Directorios de datos
    positive_dir = 'positivas'
    negative_dir = 'negativas'
    
    # Preparar datos
    features, labels = trainer.prepare_training_data(positive_dir, negative_dir)
    
    if len(features) == 0:
        print("No se encontraron imágenes para entrenar")
        return
    
    # Entrenar modelo
    svm = trainer.train(features, labels)
    
    # Probar en una imagen
    test_image = 'test.jpg'  # Asegúrate de tener una imagen de prueba
    if os.path.exists(test_image):
        result = trainer.detect_plates(test_image, svm)
        if result is not None:
            cv2.imshow('Detecciones', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite('resultado.jpg', result)
            print("Resultado guardado como 'resultado.jpg'")

if __name__ == "__main__":
    main()