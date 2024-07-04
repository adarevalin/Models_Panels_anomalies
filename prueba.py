import cv2
import numpy as np

# Especifica la ruta completa de la imagen
image_path = 'C:/Users/USUARIO/Downloads/data_solar/InfraredSolarModules/images/4.jpg'

# Lee la imagen
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Verifica si la imagen fue leída correctamente
if img is not None:
    print("Imagen leída correctamente")
    
    # Convierte la imagen de BGR a RGB si es necesario
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[-1] == 3 else img
    
    # Normaliza la imagen
    img_normalized = img.astype(np.float32) / 255.0
    img = np.expand_dims(img_normalized, axis=-1)  # Añade una dimensión al final para los canales
    
    # Muestra la imagen (opcional)
    cv2.imshow("Imagen", img)
    cv2.waitKey(0)  # Espera a que se presione una tecla
    cv2.destroyAllWindows()  # Cierra la ventana de la imagen
    
    # Imprime la forma de la imagen y su rango de valores normalizados
    print("Forma de la imagen:", img.shape)
    print("Valores normalizados de la imagen:")
    print("Valor mínimo:", np.min(img_normalized))
    print("Valor máximo:", np.max(img_normalized))
else:
    print("Error al leer la imagen")
