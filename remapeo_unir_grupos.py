# -*- coding: utf-8 -*-
"""remapeo_unir_grupos

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MIEFwDlpWRpJwfdSuW9QGLiMLt3mZLCl
"""

# Mapeo de clases originales a nuevos grupos
## Nota: para utilizar este remapeo es necesario ajustar el modelo al final con 8 posibles grupos ya que se unen grupos
class_mapping = {
    'Vegetation': 'Shadowing',
    'Shadowing': 'Shadowing',
    'No-Anomaly': 'No-Anomaly',
    'Offline-Module': 'Offline-Module',
    'Hot-Spot-Multi': 'Hot-Spot',
    'Hot-Spot': 'Hot-Spot',
    'Diode-Multi': 'Diode',
    'Diode': 'Diode',
    'Soiling': 'Soiling',
    'Cell-Multi': 'Cell',
    'Cell': 'Cell',
    'Cracking': 'Cracking'
}

# Cargar metadatos desde JSON y remapear clases
def cargar_metadatos(path_file, class_mapping):
    with open(path_file, 'r') as file:
        metadata = json.load(file)
    for key in metadata.keys():
        original_class = metadata[key]['anomaly_class']
        if original_class in class_mapping:
            metadata[key]['anomaly_class'] = class_mapping[original_class]
    return metadata