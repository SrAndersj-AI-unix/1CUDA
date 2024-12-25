import os

# Ruta de la carpeta donde buscar los archivos
ruta = "/home/server-ubuntu-studio/1CUDA"
# Nombre del archivo donde se guardarán los contenidos
archivo_salida = "todos_los_codigos.py"

# Abrir el archivo de salida en modo escritura
with open(archivo_salida, "w") as salida:
    # Recorrer todos los archivos en la ruta
    for raiz, directorios, archivos in os.walk(ruta):
        for archivo in archivos:
            # Verificar si el archivo tiene extensión .py
            if archivo.endswith(".py"):
                ruta_completa = os.path.join(raiz, archivo)
                
                # Escribir un separador para identificar el archivo
                salida.write(f"\n# Contenido del archivo: {ruta_completa}\n")
                salida.write("#" + "="*60 + "\n")

                # Leer el contenido del archivo y escribirlo en el archivo de salida
                with open(ruta_completa, "r") as archivo_py:
                    contenido = archivo_py.read()
                    salida.write(contenido)
                    salida.write("\n\n")

print(f"Los contenidos de los archivos Python han sido almacenados en {archivo_salida}.")
