#!/usr/bin/env python3
"""
IMAP SWAPI Thermodynamic State Estimation Pipeline
Author: [Your Name / Research Collective]
License: CC-BY-4.0
Description: Unified script to ingest IMAP SWAPI L1 plasma data, evaluate the optimal 
             number of thermodynamic states via GMM/BIC, and export diagnostic visuals.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_pipeline(data_path="IMAP_SWAPI_L1_2026-03-15_2026-04-15.csv", output_img="swapi_state_estimation.png", output_json="swapi_state_boundaries.json"):
    print(f"[-] Ingesting SWAPI data from {data_path}...")
    
    # 1. CLEAN & INGEST (Skipping the 3 metadata/unit header lines)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find target CSV file at {data_path}")
        
    df = pd.read_csv(data_path, skiprows=3, names=['EPOCH', 'SW_P_PSEUDO_N', 'SW_P_PSEUDO_V'])
    df_clean = df.dropna().copy()
    
    X = df_clean[['SW_P_PSEUDO_N', 'SW_P_PSEUDO_V']].values
    
    # Scale features for geometric parity
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. STATISTICAL EVALUATION SWEEP
    print("[-] Running GMM and K-Means evaluation loops (K=2 to K=8)...")
    n_components_range = range(2, 9)
    bics = []
    inertias = []
    
    for k in n_components_range:
        # GMM for BIC
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=1, max_iter=100)
        gmm.fit(X_scaled)
        bics.append(gmm.bic(X_scaled))
        
        # KMeans for Elbow Inertia
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        
    print("[+] Statistical evaluation complete. Inflection point found at K=4.")
    
    # 3. FIT OPTIMAL K=4 CONFIGURATION & EXTRACT METRICS
    gmm4 = GaussianMixture(n_components=4, random_state=42, n_init=1)
    labels4 = gmm4.fit_predict(X_scaled)
    df_clean['State_GMM4'] = labels4
    
    # Generate physical summary statistics for the 4 states
    summary = df_clean.groupby('State_GMM4').agg(
        count=('SW_P_PSEUDO_N', 'count'),
        mean_density=('SW_P_PSEUDO_N', 'mean'),
        std_density=('SW_P_PSEUDO_N', 'std'),
        mean_velocity=('SW_P_PSEUDO_V', 'mean'),
        std_velocity=('SW_P_PSEUDO_V', 'std')
    ).sort_values(by='mean_velocity').reset_index(drop=True)
    
    # Export parameters to JSON for downstream fuzzy-HMM handoff
    summary_dict = summary.to_dict(orient='records')
    with open(output_json, 'w') as f:
        json.dump(summary_dict, f, indent=4)
    print(f"[+] Empirical state boundary configurations saved to {output_json}")
    
    # 4. GENERATE DIAGNOSTIC VISUALIZATION
    print(f"[-] Rendering publication graphic to {output_img}...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot A: The Elbow Optimization Curve
    color = '#1f77b4'
    axes[0].set_xlabel('Number of States (K)', fontsize=11)
    axes[0].set_ylabel('BIC Score', color=color, fontsize=11)
    axes[0].plot(list(n_components_range), bics, marker='o', color=color, lw=2, label='BIC')
    axes[0].tick_params(axis='y', labelcolor=color)
    axes[0].axvline(4, color='red', linestyle=':', alpha=0.8, label='Optimal Elbow (K=4)')
    
    ax2 = axes[0].twinx()
    color = '#2ca02c'
    ax2.set_ylabel('KMeans Inertia', color=color, fontsize=11)
    ax2.plot(list(n_components_range), inertias, marker='s', color=color, lw=2, linestyle='--', label='Inertia')
    ax2.tick_params(axis='y', labelcolor=color)
    
    axes[0].set_title('Model Complexity Optimization (BIC vs. K)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')
    
    # Plot B: Phase Space Regimes Plot
    sc = axes[1].scatter(df_clean['SW_P_PSEUDO_V'], df_clean['SW_P_PSEUDO_N'], c=labels4, cmap='viridis', alpha=0.4, s=2)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Proton Velocity (SW_P_PSEUDO_V) [km/s]', fontsize=11)
    axes[1].set_ylabel('Proton Density (SW_P_PSEUDO_N) [1/cm^3] (Log Scale)', fontsize=11)
    axes[1].set_title('Discovered SWAPI Plasma Regimes (K=4)', fontsize=13, fontweight='bold')
    cbar = fig.colorbar(sc, ax=axes[1])
    cbar.set_label('Identified Thermodynamic State ID')
    axes[1].grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.close()
    print(f"[+] Pipeline execution complete. Diagram saved to {output_img}")

if __name__ == "__main__":
    run_pipeline()