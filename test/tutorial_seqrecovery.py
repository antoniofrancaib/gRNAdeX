# code that outputs sequence recoveries

# Import libraries and set up the environment

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import dotenv
dotenv.load_dotenv("../.env")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from src.constants_mod import PROJECT_PATH, DATA_PATH

# Import the gRNAde module
from gRNAde_mod import gRNAde_mod
from gRNAde import gRNAde

if __name__ == "__main__":

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--run_type', dest='run_type', default='single', type=str, required=True)
    #parser.add_argument('--input_path', dest='input_path', type=str, required=True)
    #parser.add_argument('--output_path', dest='output_path', type=str, required=True)
    #parser.add_argument('--max_num_conformers', dest='max_num_conformers', type=int, required=True)
    #parser.add_argument('--split', dest='split', default='das', type=str)
    #parser.add_argument('--n_samples', dest='n_samples', default=16, type=int)
    #parser.add_argument('--temperature', dest='temperature', default=1.0, type=float)
    #parser.add_argument('--seed', dest='seed', default=0, type=int)
    #parser.add_argument('--gpu_id', dest='gpu_id', default=0, type=int)
    
    #args, unknown = parser.parse_known_args()
    gRNAde = gRNAde(split='das', max_num_conformers=1, gpu_id=0)
    gRNAde_mod = gRNAde_mod(split='das', max_num_conformers=1, gpu_id=0)

    # Metadata and recoveries from Das et al. -- Obtained from tutorial notebook

    demo_data_id = ["1CSL","1ET4","1F27","1L2X","1LNT","1Q9A","4FE5","1X9C","1XPE","2GCS","2GDI","2OEU","2R8S","354D",]

    demo_data_info = [
        "RRE high affinity site",
        "Vitamin B12 binding RNA aptamer",
        "Biotin-binding RNA pseudoknot",
        "Viral RNA pseudoknot",
        "RNA internal loop of SRP",
        "Sarcin/ricin domain from E.coli 23S rRNA",
        "Guanine riboswitch aptamer",
        "All-RNA hairpin ribozyme",
        "HIV-1 B RNA dimerization initiation site",
        "Pre-cleavage state of glmS ribozyme",
        "Thiamine pyrophosphate-specific riboswitch",
        "Junctionless hairpin ribozyme",
        "Tetrahymena ribozyme P4-P6 domain",
        "Loop E from E. coli 5S rRNA",
    ]

    vienna_recovery = [0.2566288,0.25852272,0.2951389,0.2448347,0.325,0.27455357,0.2916667,0.26201922,0.2709652,0.24888393,0.25,0.23317307,0.26923078,0.28035715]

    farna_recovery = [0.20,0.34,0.36,0.45,0.27,0.40,0.28,0.31,0.24,0.26,0.38,0.30,0.36,0.35,]

    rosetta_recovery = [0.44,0.44,0.37,0.48,0.53,0.41,0.36,0.50,0.40,0.44,0.48,0.37,0.53,0.55,]

    # Evaluate gRNAde on the Das et al. data
    # First for UNmodified gRNAde
    grnade_recovery = []
    grnade_perplexity = []
    grnade_sc_score = []
    for pdb_filepath in os.listdir(os.path.join(PROJECT_PATH, "tutorial/demo_data/")):
        if pdb_filepath.endswith(".pdb"):
            sequences, samples, perplexity, recovery_sample, sc_score = gRNAde.design_from_pdb_file(
                pdb_filepath=os.path.join(PROJECT_PATH, f"tutorial/demo_data/{pdb_filepath}"),
                n_samples=16,
                temperature=0.1,
                seed=0
            )
            grnade_recovery.append(np.mean(recovery_sample))
            grnade_perplexity.append(np.mean(perplexity))
            grnade_sc_score.append(np.mean(sc_score))

    # Evaluate gRNAde on the Das et al. data
    # Now for modified gRNAde
    grnade_mod_recovery = []
    grnade_mod_perplexity = []
    grnade_mod_sc_score = []
    for pdb_filepath in os.listdir(os.path.join(PROJECT_PATH, "tutorial/demo_data/")):
        if pdb_filepath.endswith(".pdb"):
            sequences, samples, perplexity, recovery_sample, sc_score = gRNAde_mod.design_from_pdb_file(
                pdb_filepath=os.path.join(PROJECT_PATH, f"tutorial/demo_data/{pdb_filepath}"),
                n_samples=16,
                temperature=0.1,
                seed=0
            )
            grnade_mod_recovery.append(np.mean(recovery_sample))
            grnade_mod_perplexity.append(np.mean(perplexity))
            grnade_mod_sc_score.append(np.mean(sc_score))

    # Collate results as dataframes for plotting
    df = pd.DataFrame.from_dict({
        "idx": list(range(len(farna_recovery))),
        "pdb_id": demo_data_id,
        "description": demo_data_info,
        "vienna_recovery": np.array(vienna_recovery),
        "farna_recovery": np.array(farna_recovery),
        "rosetta_recovery": np.array(rosetta_recovery),
        "grnade_recovery": np.array(grnade_recovery),
        "grnade_perplexity": np.array(grnade_perplexity),
        "grnade_sc_score": np.array(grnade_sc_score),
        "grnade_mod_recovery": np.array(grnade_mod_recovery),
        "grnade_mod_perplexity": np.array(grnade_mod_perplexity),
        "grnade_mod_sc_score": np.array(grnade_mod_sc_score),

    })
    df_sample = pd.DataFrame.from_dict({
        "idx": list(range(len(farna_recovery) * 5)),
        "mean_recovery": np.array(vienna_recovery + farna_recovery + rosetta_recovery + grnade_recovery + grnade_mod_recovery),
        "model_name": ["ViennaRNA\n(2D only)"] * len(vienna_recovery) + ["FARNA"] * len(farna_recovery) + ["Rosetta"] * len(rosetta_recovery) + ["gRNAde"] * len(grnade_recovery) + ["gRNAde_mod"] * len(grnade_mod_recovery),
    })
    df.to_csv(os.path.join(PROJECT_PATH, "tutorial/outputs/benchmark_mod.csv"))

    # Print mean recovery model
    print(df_sample.groupby("model_name").mean())
    # print(df_sample.groupby("model_name").median())


    # NOW PLOTTING
    # Plot the results
    sns.set_context("talk")
    plt.figure(figsize=(6, 4))

    sns.swarmplot(
        data = df_sample,
        x="model_name", y="mean_recovery",
        hue="model_name", palette="Reds",
    )

    ax = sns.barplot(
        data = df_sample,
        x="model_name", y="mean_recovery",
        hue="model_name", saturation=0.5, palette="Reds",
    )
    # Add labels to each bar
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'right', va = 'center', 
                    xytext = (-4, 8), 
                    textcoords = 'offset points',
                    fontsize=10)

    # ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    # ax.bar_label(ax.containers[0], fontsize=10)
    plt.xlabel("")
    plt.ylabel("Native sequence recovery", labelpad=10)
    # plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.01, 0.25), fontsize=14)
    plt.savefig(os.path.join(PROJECT_PATH, "tutorial/outputs/singlestate-barplot-mod.pdf"), dpi=300, bbox_inches="tight")

    # Plot native sequence recovery per sample for Rosetta vs. gRNAde, shaded by gRNAde’s average perplexity for each sample
    plt.figure(figsize=(5, 4))

    plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, ls="-", c="black", alpha=0.3)

    stat = df_sample.groupby("model_name").mean()
    #rosetta_stat = stat.loc["Rosetta"]["mean_recovery"]
    grnade_stat = stat.loc["gRNAde"]["mean_recovery"]
    grnade_mod_stat = stat.loc["gRNAde_mod"]["mean_recovery"]
    #plt.plot([rosetta_stat, rosetta_stat], [0, 1], transform=plt.gca().transAxes, ls="--", c="black", alpha=0.1)
    plt.plot([grnade_stat, grnade_stat], [0, 1], transform=plt.gca().transAxes, ls="--", c="black", alpha=0.1)
    plt.plot([0, 1], [grnade_mod_stat, grnade_mod_stat], transform=plt.gca().transAxes, ls="--", c="black", alpha=0.1)

    ax = sns.scatterplot(
        data = df,
        x = "grnade_recovery", y = "grnade_mod_recovery",
        hue = "grnade_mod_perplexity", palette="flare",
        alpha=1.0,
    )
    # set model names as x and y axis labels
    plt.xlabel("gRNAde seq. recovery", labelpad=10)
    plt.ylabel("gRNAde_mod seq. recovery", labelpad=10)
    # set x and y axis range to 0-1
    plt.xticks(np.arange(0, 1.01, 0.25), fontsize=14) # plt.xlim(0, 1)
    plt.yticks(np.arange(0, 1.01, 0.25), fontsize=14) # plt.ylim(0, 1)
    # Add perplexity colorbar
    norm = plt.Normalize(df['grnade_mod_perplexity'].min(), df['grnade_mod_perplexity'].max())
    sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
    sm.set_array([])
    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    ax2 = ax.twinx()
    ax2.set_ylabel("gRNAde_mod perplexity", labelpad=70)
    ax2.set_yticks([])
    plt.savefig(os.path.join(PROJECT_PATH, "tutorial/outputs/singlestate-scatterplot-mod.pdf"), dpi=300, bbox_inches="tight")
    #plt.show()