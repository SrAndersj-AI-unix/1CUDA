import cupy as cp

# Obtener la versión de CUDA
version = cp.cuda.runtime.runtimeGetVersion()
print(f"Versión de CUDA: {version}")
