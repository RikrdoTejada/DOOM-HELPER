import os

# Carpeta de trabajo
frame_dir = "C:/Users/User/OneDrive/Documents/GitHub/VizDoomIA/dataset/frames"


# Obtener nombres base (sin extensión)
images = [f for f in os.listdir(frame_dir) if f.lower().endswith(".png")]
image_basenames = {os.path.splitext(f)[0] for f in images}

labels = [f for f in os.listdir(frame_dir) if f.lower().endswith(".txt")]
label_basenames = {os.path.splitext(f)[0] for f in labels}

# Encontrar imágenes sin su archivo .txt
missing = sorted(image_basenames - label_basenames)

# Crear los archivos .txt vacíos
count = 0
for name in missing:
    label_path = os.path.join(frame_dir, f"{name}.txt")
    if not os.path.exists(label_path):
        with open(label_path, "w") as f:
            pass  # Crea archivo vacío
        print(f"✅ Etiqueta vacía creada: {label_path}")
        count += 1

print(f"\n🎉 Total: {count} archivos .txt vacíos creados.")
