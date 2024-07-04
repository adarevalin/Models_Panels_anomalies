import json

def cargar_metadatos(path_file):
    """
    Carga metadatos desde un archivo JSON.

    :param path_file: Ruta del archivo JSON.
    :return: Diccionario con los metadatos cargados.
    """
    with open(path_file, 'r') as f:
        metadata = json.load(f)
    return metadata