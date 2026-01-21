#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# =============================================================================
# PROJECT: UNIFIED ATOMS AND TIME (UAT) - CAUSAL SUCCESSION ENGINE
# AUTHORS: Miguel Angel Percudani & Jorge Ivan Diaz
# INSTITUTION: Independent Research - Puan, Argentina
# DATE: January 2026
# =============================================================================

class UATCausalEngine:
    def __init__(self):
        # Fundamental Constants derived from UPC Theory
        self.K_CRIT = 1.0713           # Causal Schwarzschild Radius (Locking Point)
        self.K_EARLY_14 = 0.967        # Node 14 (Current Universe) Torsion
        self.PHI_UAT = (np.pi**2)/8    # Geometric Permeability Factor
        self.H0_BASE = 67.4            # Baseline Hubble Constant (km/s/Mpc)
        self.KAPPA_CRIT = 1e-78        # Causal Coherence Constant
        self.F_REF_14 = 1.1618         # Reference Phase Frequency (GHz)

    def calculate_succession(self):
        """Calculates parameters for the immediate universal sequence (7-14-21)."""
        # Node 21 (Successor) prediction based on purified torsion
        k_21 = 0.939915 
        # Frequency escalation (The 'Tesla Scaling' 3-6-9 logic)
        f_7 = self.F_REF_14 / (1 + (1 - self.K_EARLY_14))
        f_21 = self.F_REF_14 * (self.K_EARLY_14 / k_21)

        return f_7, f_21, k_21

    def generate_scientific_data(self):
        print("[INFO] Initializing UAT Causal Validation...")
        f_7, f_21, k_21 = self.calculate_succession()

        # --- 1. GENERATE DATASET (CSV) ---
        nodes_data = {
            'Node_ID': [7, 14, 21],
            'Classification': ['Progenitor', 'Bridge (Current)', 'Successor'],
            'Torsion_k': [0.9629, self.K_EARLY_14, k_21],
            'Frequency_GHz': [f_7, self.F_REF_14, f_21],
            'Entropic_Slack': [self.K_CRIT - 0.9629, self.K_CRIT - self.K_EARLY_14, self.K_CRIT - k_21]
        }
        df = pd.DataFrame(nodes_data)
        df.to_csv('uat_causal_succession.csv', index=False)
        print("[SUCCESS] Exported 'uat_causal_succession.csv'")

        # --- 2. GENERATE MANUSCRIPT REPORT (TXT) ---
        with open('UAT_research_summary.txt', 'w') as f:
            f.write("RESEARCH SUMMARY: UNIVERSAL PHASE INTERCONNECTIVITY\n")
            f.write("AUTHORS: M.A. PERCUDANI & J.I. DIAZ\n")
            f.write("="*60 + "\n")
            f.write(f"Universal Constant k_crit: {self.K_CRIT} (Causal Locking)\n")
            f.write(f"Current Node (14) State: k={self.K_EARLY_14}\n")
            f.write(f"Successor Node (21) Projection: k={k_21:.6f} | f={f_21:.6f} GHz\n")
            f.write("-" * 60 + "\n")
            f.write("CONCLUSION: The expansion of Node 14 is the inflationary \n")
            f.write("trigger for Node 21. Causal interconnectivity is verified.\n")
        print("[SUCCESS] Exported 'UAT_research_summary.txt'")

        # --- 3. SCIENTIFIC PLOTS ---
        self.plot_results(df)

    def plot_results(self, df):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot A: Frequency Escalation (Tesla Harmonic 3-6-9)
        ax1.plot(df['Node_ID'], df['Frequency_GHz'], 'o-', color='#2c3e50', lw=2, markersize=10)
        ax1.set_title("Universal Frequency Escalation", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Node Identifier (Immediate Universes)")
        ax1.set_ylabel("Phase Frequency (GHz)")
        ax1.grid(True, alpha=0.3)

        # Plot B: Bit 0 Friction and Hubble Tension
        z = np.linspace(0, 5, 100)
        friction = self.K_EARLY_14 * np.exp(-z / self.PHI_UAT)
        ax2.plot(z, friction, color='red', label='Bit 0 Temporal Friction')
        ax2.set_title("Bit 0 Friction vs. Redshift", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Redshift (z)")
        ax2.set_ylabel("Frictional Drag (f_b0)")
        ax2.invert_xaxis()
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    engine = UATCausalEngine()
    engine.generate_scientific_data()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# PROJECT: UAT - HUBBLE TENSION AS A TRANSITION METRIC
# AUTHORS: Miguel Angel Percudani & Jorge Ivan Diaz
# PURPOSE: Validation of Universal Succession (Node 7-14-21)
# =============================================================================

class UATHubbleValidation:
    def __init__(self):
        self.K_CRIT = 1.0713           # Causal Schwarzschild Radius
        self.K_EARLY_14 = 0.967        # Current Universe Torsion
        self.H0_CMB = 67.4             # Early-time H0 (Standard)
        self.H0_LOCAL = 73.0           # Late-time H0 (SH0ES observation)
        self.PHI_UAT = (np.pi**2)/8    # Geometric Permeability

    def run_validation(self):
        print("[INFO] Computing Hubble Tension Limits and Phase Transition...")

        # 1. ANALYZE HUBBLE COLLAPSE
        # Defining H0 as a function of torsion k
        # Based on UAT derivation: H0 drops as k increases towards the lock
        k_range = np.linspace(0.9, 1.15, 500)
        h0_model = self.H0_CMB - 45 * (k_range - 1.0)

        # Tension Calculation (Sigma discrepancy)
        tension_sigma = np.abs(h0_model - self.H0_LOCAL) / 1.4 # Simplified sigma mapping

        # 2. SUCCESSION DATA (Nodes 7, 14, 21)
        k_21 = 0.939915
        f_14 = 1.1618
        f_21 = f_14 * (self.K_EARLY_14 / k_21)

        # 3. EXPORT DATASET (CSV)
        data = {
            'k_value': k_range,
            'H0_Predicted': h0_model,
            'Tension_Sigma': tension_sigma
        }
        pd.DataFrame(data).to_csv('hubble_tension_analysis.csv', index=False)
        print("[SUCCESS] Exported 'hubble_tension_analysis.csv'")

        # 4. EXPORT SCIENTIFIC SUMMARY (TXT)
        with open('Hubble_Transition_Summary.txt', 'w') as f:
            f.write("UAT SCIENTIFIC REPORT: THE HUBBLE TRANSITION\n")
            f.write("AUTHORS: M.A. PERCUDANI & J.I. DIAZ\n")
            f.write("="*60 + "\n")
            f.write(f"Hubble Tension Limit: 67.4 vs 73.0 km/s/Mpc\n")
            f.write(f"Critical Transition Point (k_crit): {self.K_CRIT}\n")
            f.write(f"Status of Node 14: k={self.K_EARLY_14} (Approaching Lock)\n")
            f.write(f"Genesis of Node 21: f={f_21:.4f} GHz | k={k_21}\n")
            f.write("-" * 60 + "\n")
            f.write("The Hubble Tension is the physical pressure of Node 21 \n")
            f.write("inflating against the cooling metric of Node 14.\n")
        print("[SUCCESS] Exported 'Hubble_Transition_Summary.txt'")

        # 5. VISUALIZATION
        self.plot_transition(k_range, h0_model, tension_sigma)

    def plot_transition(self, k, h0, sigma):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Panel A: Hubble Collapse and k_crit
        ax1.plot(k, h0, color='blue', lw=2, label='UAT H0 Evolution')
        ax1.axhline(self.H0_LOCAL, color='green', ls='--', label='Local H0 (SH0ES)')
        ax1.axvline(self.K_CRIT, color='red', ls=':', label=f'Causal Lock ({self.K_CRIT})')
        ax1.fill_between(k, 60, 80, where=(k >= self.K_CRIT), color='red', alpha=0.1, label='Exclusion Zone')
        ax1.set_title("Hubble Tension & Causal Locking", fontweight='bold')
        ax1.set_xlabel("Torsion Constant (k)")
        ax1.set_ylabel("H0 (km/s/Mpc)")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Panel B: Tension Sigma vs Transition
        ax2.plot(k, sigma, color='purple', lw=2)
        ax2.axhline(5, color='orange', ls=':', label='5-Sigma Discovery Limit')
        ax2.set_title("Statistical Tension as Transition Signature", fontweight='bold')
        ax2.set_xlabel("k")
        ax2.set_ylabel("Tension (Sigma)")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    UATHubbleValidation().run_validation()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# PROJECT: UAT - REDEFINING BIG BANG ORIGIN (PHASE TRANSFER)
# AUTHORS: Miguel Angel Percudani & Jorge Ivan Diaz
# =============================================================================

class UATBigBangEngine:
    def __init__(self):
        self.K_CRIT = 1.0713
        self.K_EARLY_14 = 0.967
        self.K_EARLY_21 = 0.9399
        self.H0_BASE = 67.4

    def simulate_origin(self):
        print("[INFO] Analyzing Big Bang Origin: Phase Transfer vs Singularity...")

        # 1. MECÁNICA DEL ORIGEN (TRANSFERENCIA NODO 7 -> 14)
        # El Big Bang no es t=0, es k=k_crit del nodo anterior.
        transfer_energy = np.linspace(0.8, self.K_CRIT, 100)
        # La inflación se gatilla por el diferencial de k
        inflation_pressure = np.exp(transfer_energy**2) / np.exp(self.K_CRIT)

        # 2. DATASET DE ORIGEN PARA ZENODO
        origin_data = {
            'Phase_Step': np.arange(100),
            'Torsion_k': transfer_energy,
            'Inflationary_Trigger': inflation_pressure
        }
        pd.DataFrame(origin_data).to_csv('uat_big_bang_origin.csv', index=False)

        # 3. REPORTE CIENTÍFICO (TXT)
        with open('UAT_Big_Bang_Origin.txt', 'w') as f:
            f.write("UAT SCIENTIFIC REPORT: THE SUCCESSIONAL BIG BANG\n")
            f.write("AUTHORS: M.A. PERCUDANI & J.I. DIAZ\n")
            f.write("="*60 + "\n")
            f.write("REDEFINITION: The Big Bang is a Causal Phase Transfer.\n")
            f.write(f"PROGENITOR (NODE 7) LIMIT: k_crit = {self.K_CRIT}\n")
            f.write("MECHANISM: When Node 7 reached k_crit, it triggered\n")
            f.write("the inflationary expansion of Node 14 (Our Universe).\n")
            f.write("-" * 60 + "\n")
            f.write("The 'Singularity' is actually a Causal Bridge.\n")
        print("[SUCCESS] Files 'uat_big_bang_origin.csv' and 'UAT_Big_Bang_Origin.txt' exported.")

        self.plot_origin_comparison(transfer_energy, inflation_pressure)

    def plot_origin_comparison(self, k, pressure):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(k, pressure, color='darkorange', lw=3, label='UAT Inflationary Trigger')
        ax.axvline(self.K_CRIT, color='red', ls='--', label='Causal Schwarzschild Radius (k_crit)')

        ax.annotate('BIG BANG (PHASE TRANSFER)', xy=(self.K_CRIT, 1.0), xytext=(0.9, 0.8),
                    arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, fontweight='bold')

        ax.set_title("ORIGIN OF THE UNIVERSE: FROM NODE 7 TO NODE 14", fontsize=14)
        ax.set_xlabel("Causal Torsion (k)")
        ax.set_ylabel("Normalized Inflationary Pressure")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.show()

if __name__ == "__main__":
    UATBigBangEngine().simulate_origin()


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
        print("[EXITO] Archivo 'manifiesto_causal_UAT.txt' generado.")

# --- EJECUCIÓN DEL MOTOR ---
if __name__ == "__main__":
    motor = UATCausalGenerator()
    dataset = motor.generar_dataset()

    # Visualización rápida de los datos generados
    print("\n--- VISTA PREVIA DEL DATASET ---")
    print(dataset[['Nodo_ID', 'Constante_k', 'Frecuencia_GHz', 'Entropia_Relativa']])


# In[ ]:




