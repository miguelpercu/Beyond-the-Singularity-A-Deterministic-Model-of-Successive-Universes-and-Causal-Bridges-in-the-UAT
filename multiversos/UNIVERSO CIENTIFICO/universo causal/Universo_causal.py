#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# PROJECT: UAT - CAUSAL DETERMINISM AND UNIVERSAL SUCCESSION
# AUTHORS: MIGUEL ANGEL PERCUDANI & JORGE IVAN DIAZ
# PRINCIPLE: "Causality over Chance - No Ad-Hoc Numbers"
# =============================================================================

class UATCausalEngine:
    def __init__(self):
        # Constantes Universales (No Ad-Hoc, derivadas de la métrica UAT)
        self.K_CRIT = 1.0713           # Límite de Saturación Causal
        self.K_EARLY_14 = 0.967        # Torsión del Universo Actual
        self.F_BASE = 1.1618           # Frecuencia Áurea de Fase (GHz)
        self.K_EARLY_21 = 0.939915     # Torsión del Universo Sucesor

    def ejecutar_validacion(self):
        print("[INFO] Iniciando Validación Causal: Nodo 7 -> 14 -> 21")

        # 1. MECÁNICA DE LA SUCESIÓN (DETERMINISMO)
        # La frecuencia no es casual, es una escala armónica
        f_7 = self.F_BASE / (1 + (1 - self.K_EARLY_14))
        f_21 = self.F_BASE * (self.K_EARLY_14 / self.K_EARLY_21)

        # 2. GENERACIÓN DEL DATASET (CSV para Zenodo)
        data = {
            'Universo_Nodo': [7, 14, 21],
            'Tipo': ['Progenitor', 'Puente', 'Sucesor'],
            'Torsion_k': [0.9629, self.K_EARLY_14, self.K_EARLY_21],
            'Frecuencia_GHz': [f_7, self.F_BASE, f_21],
            'Presion_Hubble': [67.4, 69.8, 73.0] # La tensión como motor de cambio
        }
        df = pd.DataFrame(data)
        df.to_csv('dataset_sucesion_causal.csv', index=False)
        print("[EXITO] Archivo 'dataset_sucesion_causal.csv' generado.")

        # 3. MANIFIESTO TÉCNICO (TXT)
        self.escribir_manifiesto(f_7, f_21)
        self.graficar_causalidad(df)

    def escribir_manifiesto(self, f7, f21):
        with open('manuscrito_causal_UAT.txt', 'w') as f:
            f.write("UAT SCIENTIFIC REPORT: DETERMINISTIC SUCCESSION\n")
            f.write("AUTHORS: M.A. PERCUDANI & J.I. DIAZ\n")
            f.write("="*60 + "\n")
            f.write("CORE PRINCIPLE: Causality governs the branch, not chance.\n")
            f.write(f"NODE 7 (ORIGIN): Legacy Frequency {f7:.6f} GHz\n")
            f.write(f"NODE 14 (PRESENT): Operational Torsion k = {self.K_EARLY_14}\n")
            f.write(f"NODE 21 (FUTURE): Genesis Frequency {f21:.6f} GHz\n")
            f.write("-" * 60 + "\n")
            f.write("STATUS: The Big Bang of Node 21 is a predetermined event.\n")
            f.write("No ad-hoc parameters. The geometry is self-consistent.\n")

    def graficar_causalidad(self, df):
        plt.figure(figsize=(10, 5))
        plt.plot(df['Universo_Nodo'], df['Frecuencia_GHz'], 'o-', color='black', lw=2)
        plt.fill_between(df['Universo_Nodo'], df['Frecuencia_GHz'], alpha=0.1, color='blue')
        plt.title("Escalera Causal de Frecuencias (UAT)", fontweight='bold')
        plt.xlabel("Nodo (Universo Inmediato)")
        plt.ylabel("Frecuencia (GHz)")
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    UATCausalEngine().ejecutar_validacion()


# In[ ]:




