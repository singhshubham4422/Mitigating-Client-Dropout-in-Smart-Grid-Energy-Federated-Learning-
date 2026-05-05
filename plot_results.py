import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

files = glob.glob("logs/**/*.csv", recursive=True)

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["source"] = f
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

os.makedirs("results/plots", exist_ok=True)

# -------- Learning Curve --------
for (dp, smpc), group in df.groupby(["dp", "smpc"]):
    label = f"DP={dp}, SMPC={smpc}"
    plt.plot(group["round"], group["mse"], label=label)

plt.xlabel("Round")
plt.ylabel("MSE")
plt.legend()
plt.grid()
plt.savefig("results/plots/learning_curve.png", dpi=300)
plt.close()

# -------- Privacy vs Utility --------
if "epsilon" in df.columns:
    plt.figure()
    plt.scatter(df["epsilon"], df["mse"])
    plt.xlabel("Epsilon")
    plt.ylabel("MSE")
    plt.grid()
    plt.savefig("results/plots/privacy_vs_utility.png", dpi=300)
    plt.close()

print("✅ Plots generated")