import os

def generar_positivas_txt(carpeta_positivas, archivo_salida):
    """
    Genera el archivo positivas.txt con la lista de imágenes y sus coordenadas.
    Se asume un único objeto centrado en cada imagen.
    """
    try:
        # Abrir el archivo de salida
        with open(archivo_salida, 'w') as archivo:
            # Listar imágenes en la carpeta
            for imagen in os.listdir(carpeta_positivas):
                if imagen.endswith('.jpg'):
                    ruta_relativa = os.path.join(carpeta_positivas, imagen)
                    
                    # Coordenadas de ejemplo: ajustar según tus necesidades
                    x = 100  # Coordenada X de la esquina superior izquierda
                    y = 50   # Coordenada Y de la esquina superior izquierda
                    ancho = 300  # Ancho del rectángulo
                    alto = 150   # Alto del rectángulo
                    
                    # Escribir la línea en el formato requerido
                    archivo.write(f"{ruta_relativa} 1 {x} {y} {ancho} {alto}\n")
        print(f"Archivo {archivo_salida} generado exitosamente.")
    except Exception as e:
        print(f"Error: {e}")

# Ruta donde se encuentran las imágenes positivas
carpeta_positivas = "C:/Universidad/IA/cv2finalv1/positivas"

# Nombre del archivo de salida
archivo_salida = "C:/Universidad/IA/cv2finalv1/info/positivas.txt"

# Llamar a la función para generar el archivo
generar_positivas_txt(carpeta_positivas, archivo_salida)
