import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(layout="wide")
st.title("Law of Large Numbers (LLN) and Central Limit Theorem (CLT)")

# --- Distribution selection ---
with st.sidebar:
    st.header("Distribution Parameters")
    dist_name = st.selectbox("Choose Distribution", ["Bernoulli", "Normal", "Uniform"])

    if dist_name == "Bernoulli":
        p = st.slider("Bernoulli p", 0.0, 1.0, 0.5)
        dist = torch.distributions.Bernoulli(probs=p)
        true_mean = p
        ylims = (0, 1)

    elif dist_name == "Normal":
        mu = st.slider("Normal μ", -5.0, 5.0, 0.0)
        sigma = st.slider("Normal σ", 0.1, 5.0, 1.0)
        dist = torch.distributions.Normal(loc=mu, scale=sigma)
        true_mean = mu
        ylims = (mu - 4*sigma, mu + 4*sigma)

    elif dist_name == "Uniform":
        a = st.slider("Uniform a", -10.0, 0.0, 0.0)
        b = st.slider("Uniform b", 0.0, 10.0, 1.0)
        if a >= b:
            st.error("Ensure that a < b")
            st.stop()
        dist = torch.distributions.Uniform(low=a, high=b)
        true_mean = (a + b) / 2
        ylims = (a, b)

# --- Experiment parameters ---
with st.sidebar:
    st.header("LLN/CLT Settings")
    K = st.slider("Number of runs (K)", 100, 5000, 1000, step=100)
    N = st.slider("Samples per run (N)", 10, 1000, 100, step=10)

# --- Sampling ---
samples = dist.sample((K, N))
running_means = torch.cumsum(samples, dim=1) / torch.arange(1, N + 1).float()
final_means = running_means[:, -1].numpy()

# --- Plotting ---
fig, (ax_lln, ax_clt) = plt.subplots(1, 2, figsize=(16, 8), dpi=200, gridspec_kw={'width_ratios': [3, 1]}, sharey=True)


# LLN
for i in range(K):
    ax_lln.plot(running_means[i], color='gray', alpha=0.02)
ax_lln.axhline(true_mean, color='red', linestyle='--', label=r'$\mathbb{E}[X]$')
ax_lln.set_xlabel("n")
ax_lln.set_ylabel(r"$\bar{X}_n$")
ax_lln.set_ylim(*ylims)
ax_lln.set_title("LLN: Running Averages")
ax_lln.grid(False)
ax_lln.legend()

# CLT
sns.kdeplot(final_means, ax=ax_clt, color='black', linewidth=2, vertical=True,
            bw_adjust=2, fill=True, alpha=0.2)
ax_clt.axhline(true_mean, color='red', linestyle='--')
ax_clt.set_xlabel("Density")
ax_clt.set_ylabel("")
ax_clt.set_ylim(*ylims)
ax_clt.set_title("CLT: Sample Means Distribution")
ax_clt.grid(False)

from io import BytesIO
buf = BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
st.image(buf.getvalue())