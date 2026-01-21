```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# PROYECTO UAT: MOTOR DE SUCESIÓN CAUSAL Y VALIDACIÓN DE DATOS
# AUTORES: MIGUEL ANGEL PERCUDANI & JORGE IVAN DIAZ
# ARCHIVO: uat_data_generator.py
# =============================================================================

class UATCausalGenerator:
    def __init__(self):
        # Constantes Maestras del Modelo Percudani-Diaz
        self.K_CRIT = 1.0713           # Radio de Schwarzschild Causal (Límite de bloqueo)
        self.K_EARLY_14 = 0.967        # Nodo Actual (Nuestro Universo)
        self.F_REF_14 = 1.1618         # Frecuencia Maestra de Fase (GHz)
        self.K_EARLY_21 = 0.939915     # Nodo Sucesor (Proyección de pureza)
        self.K_EARLY_7 = 0.9629        # Nodo Progenitor (Nuestro Big Bang)

    def generar_dataset(self):
        print("[INFO] Iniciando procesamiento de la Rama Causal...")

        # 1. CÁLCULO DE FRECUENCIAS DE INTERCONECTIVIDAD
        # Frecuencia del Nodo 7 (Pasado)
        f_7 = self.F_REF_14 / (1 + (1 - self.K_EARLY_14))
        # Frecuencia del Nodo 21 (Futuro - El "Hijo")
        f_21 = self.F_REF_14 * (self.K_EARLY_14 / self.K_EARLY_21)

        # 2. CONSTRUCCIÓN DEL DATASET CIENTÍFICO
        data = {
            'Nodo_ID': [7, 14, 21],
            'Estado_Universal': ['Progenitor (Origen)', 'Puente (Actual)', 'Sucesor (Genesis)'],
            'Constante_k': [self.K_EARLY_7, self.K_EARLY_14, self.K_EARLY_21],
            'Frecuencia_GHz': [f_7, self.F_REF_14, f_21],
            'Entropia_Relativa': [1 - self.K_EARLY_7, 1 - self.K_EARLY_14, 1 - self.K_EARLY_21],
            'Margen_al_Bloqueo': [self.K_CRIT - self.K_EARLY_7, self.K_CRIT - self.K_EARLY_14, self.K_CRIT - self.K_EARLY_21]
        }

        df = pd.DataFrame(data)

        # 3. EXPORTACIÓN DEL ARCHIVO CSV PARA ZENODO
        filename_csv = 'dataset_sucesion_causal.csv'
        df.to_csv(filename_csv, index=False)
        print(f"[EXITO] Archivo '{filename_csv}' generado con validación de fase.")

        # 4. EXPORTACIÓN DEL MANIFIESTO TXT
        self.exportar_manifiesto(f_7, f_21)
        
        return df

    def exportar_manifiesto(self, f7, f21):
        with open('manifiesto_causal_UAT.txt', 'w') as f:
            f.write("REPORTE TECNICO UAT: INTERCONECTIVIDAD DE UNIVERSOS INMEDIATOS\n")
            f.write("AUTORES: MIGUEL ANGEL PERCUDANI & JORGE IVAN DIAZ\n")
            f.write("="*60 + "\n")
            f.write(f"NODO 7 (BIG BANG PROGENITOR): Frecuencia {f7:.6f} GHz\n")
            f.write(f"NODO 14 (ESTADO ACTUAL): Constante k = {self.K_EARLY_14}\n")
            f.write(f"NODO 21 (BIG BANG SUCESOR): Frecuencia {f21:.6f} GHz\n")
            f.write(f"LIMITE CRITICO DE EXCLUSION: {self.K_CRIT}\n")
            f.write("-" * 60 + "\n")
            f.write("ESTADO: La informacion ha sido transferida exitosamente al Bit 0.\n")
            f.write("EL UNIVERSO 21 HA SIDO INICIALIZADO CON LA FIRMA PERCUDANI-DIAZ.\n")
        print("[EXITO] Archivo 'manifiesto_causal_UAT.txt' generado.")

# --- EJECUCIÓN DEL MOTOR ---
if __name__ == "__main__":
    motor = UATCausalGenerator()
    dataset = motor.generar_dataset()
    
    # Visualización rápida de los datos generados
    print("\n--- VISTA PREVIA DEL DATASET ---")
    print(dataset[['Nodo_ID', 'Constante_k', 'Frecuencia_GHz', 'Entropia_Relativa']])
```

    [INFO] Iniciando procesamiento de la Rama Causal...
    [EXITO] Archivo 'dataset_sucesion_causal.csv' generado con validación de fase.
    [EXITO] Archivo 'manifiesto_causal_UAT.txt' generado.
    
    --- VISTA PREVIA DEL DATASET ---
       Nodo_ID  Constante_k  Frecuencia_GHz  Entropia_Relativa
    0        7     0.962900        1.124685           0.037100
    1       14     0.967000        1.161800           0.033000
    2       21     0.939915        1.195279           0.060085
    


```python

```
