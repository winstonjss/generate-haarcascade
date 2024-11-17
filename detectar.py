import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
import os

class PlateRecognizer:
    def __init__(self):
        # Cargar el modelo SVM entrenado
        self.svm = cv2.ml.SVM_load('plate_detector_svm.xml')
        
        # Configurar pytesseract (ajusta la ruta según tu instalación)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Configurar el HOG descriptor
        self.hog = cv2.HOGDescriptor((64,32), (16,16), (8,8), (8,8), 9)

    def preprocess_plate(self, plate_img):
        """Preprocesa la imagen de la placa para mejorar el OCR"""
        # Convertir a escala de grises
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar umbralización adaptativa
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Reducir ruido
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        return denoised

    def extract_plate_text(self, plate_img):
        """Extrae el texto de la placa usando OCR"""
        # Preprocesar la imagen
        processed = self.preprocess_plate(plate_img)
        
        # Realizar OCR
        config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(processed, config=config)
        
        # Limpiar el texto
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        return text

    def detect_and_recognize_plate(self, img):
        """Detecta y reconoce la placa en una imagen"""
        if img is None:
            return None, None
        
        # Copiar la imagen original
        output_img = img.copy()
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Ventana deslizante para detección
        window_size = (64, 32)
        step_size = 32
        detections = []
        
        for y in range(0, gray.shape[0] - window_size[1], step_size):
            for x in range(0, gray.shape[1] - window_size[0], step_size):
                window = gray[y:y + window_size[1], x:x + window_size[0]]
                window = cv2.resize(window, (64, 32))
                
                features = self.hog.compute(window)
                
                # Predecir usando SVM
                _, pred = self.svm.predict(features.astype(np.float32).reshape(1, -1))
                
                if pred[0][0] == 1:
                    detections.append((x, y, x + window_size[0], y + window_size[1]))
        
        plate_texts = []
        
        # Procesar cada detección
        for (x1, y1, x2, y2) in detections:
            # Extraer región de la placa
            plate_region = img[y1:y2, x1:x2]
            
            # Obtener texto de la placa
            plate_text = self.extract_plate_text(plate_region)
            
            if len(plate_text) >= 6:  # Filtrar textos muy cortos
                plate_texts.append(plate_text)
                # Dibujar rectángulo y texto
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_img, plate_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output_img, plate_texts

    def process_image(self, image_path):
        """Procesa una imagen individual"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"No se pudo cargar la imagen: {image_path}")
            return
        
        result_img, plates = self.detect_and_recognize_plate(img)
        
        if plates:
            print("Placas detectadas:")
            for plate in plates:
                print(f"- {plate}")
        else:
            print("No se detectaron placas")
            
        cv2.imshow('Resultado', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Guardar resultado
        output_path = f"resultado_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, result_img)
        print(f"Resultado guardado como: {output_path}")

    def process_video(self, video_source=0):
        """Procesa video (puede ser archivo o cámara web)"""
        cap = cv2.VideoCapture(video_source)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            result_frame, plates = self.detect_and_recognize_plate(frame)
            
            # Mostrar placas detectadas
            if plates:
                print("\rPlacas detectadas:", ", ".join(plates), end="")
            
            cv2.imshow('Detección de Placas en Video', result_frame)
            
            # Presionar 'q' para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    recognizer = PlateRecognizer()
    
    while True:
        print("\nMenú:")
        print("1. Procesar imagen")
        print("2. Procesar video (cámara web)")
        print("3. Procesar archivo de video")
        print("4. Salir")
        
        choice = input("Seleccione una opción: ")
        
        if choice == '1':
            image_path = input("Ingrese la ruta de la imagen: ")
            recognizer.process_image(image_path)
        
        elif choice == '2':
            print("Iniciando video de cámara web... Presione 'q' para salir")
            recognizer.process_video()
        
        elif choice == '3':
            video_path = input("Ingrese la ruta del archivo de video: ")
            print("Procesando video... Presione 'q' para salir")
            recognizer.process_video(video_path)
        
        elif choice == '4':
            print("¡Hasta luego!")
            break
        
        else:
            print("Opción no válida")

if __name__ == "__main__":
    main()