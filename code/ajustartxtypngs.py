import os
import re

# Carpeta de trabajo
frame_dir = "C:/Users/User/OneDrive/Documents/GitHub/VizDoomIA/dataset/frames"

# Obtener todos los archivos en la carpeta
files = os.listdir(frame_dir)

# Expresión regular para encontrar números en los nombres de archivo
pattern = re.compile(r'(\d+)')

for filename in files:
    # Separar el nombre base y la extensión
    name, ext = os.path.splitext(filename)
    
    # Buscar números en el nombre
    match = pattern.search(name)
    if match:
        number_str = match.group(1)
        number = int(number_str)
        new_number = number + 1000
        new_name = pattern.sub(str(new_number), name, count=1)  # Reemplazar solo la primera ocurrencia
        new_filename = f"{new_name}{ext}"
        
        # Rutas completas
        old_path = os.path.join(frame_dir, filename)
        new_path = os.path.join(frame_dir, new_filename)
        
        # Renombrar el archivo
        os.rename(old_path, new_path)
        print(f"Renombrado: {filename} -> {new_filename}")
    else:
        print(f"No se encontró número en: {filename}")
