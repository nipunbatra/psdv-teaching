from math import exp
import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Streamlit page setup
st.title("Binomial Distribution Variance and Expectation")

# Input parameters for N and p
st.sidebar.header("Set Parameters")

N = st.sidebar.slider("Number of trials (N)", min_value=2, max_value=100, value=10, step=1)
p = st.sidebar.slider("Probability of success (p)", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
num_samples = st.sidebar.slider("Number of samples", min_value=100, max_value=10000, value=1000, step=100)

dist = torch.distributions.Binomial(N, p)

expectation = N * p
st.write(f"Expectation of Binomial Distribution for N={N} and p={p}: **{expectation:.4f}**")

# Display the variance calculation
variance = N * p * (1 - p)
st.write(f"Variance of Binomial Distribution for N={N} and p={p}: **{variance:.4f}**")



# Plot a bar plot for PMF and another subplot for the squared deviation
fig, ax = plt.subplots(3, 1, sharex=True)

# Plot the PMF
x_lin = torch.arange(0, N + 1)
y_lin = dist.log_prob(x_lin).exp()
ax[0].bar(x_lin, y_lin, color='blue', alpha=0.7)
ax[0].set_title(f"Binomial Distribution PMF for N={N} and p={p}")
ax[0].set_ylabel('Probability')

# Plot the squared deviation
deviation = x_lin - expectation
squared_deviation = deviation ** 2
ax[1].bar(x_lin, squared_deviation, color='red', alpha=0.7)

ax[1].set_title(f"Squared Deviation from Expectation for N={N} and p={p}")
ax[1].set_ylabel('Squared Deviation')

ax[2].bar(x_lin, squared_deviation * y_lin, color='green', alpha=0.7)
ax[2].set_title(f"Weighted Squared Deviation from Expectation for N={N} and p={p}")
ax[2].set_xlabel('Number of Successes')
ax[2].set_ylabel('Weighted Squared\n Deviation')
fig.tight_layout()


st.pyplot(fig)

# Expand and show the calculation wrt deviation from E[X]
st.write(f"Variance can also be calculated as the expectation of the squared deviation from the mean:")
expanded_variance_string = "+".join([f"({x} - {expectation:.1f})^2" for x in range(N + 1)])
st.write(f"Variance = 1/N * ( {expanded_variance_string} )")

# separator
st.write("---")

samples = dist.sample((num_samples,))
# Draw dataframe with samples
df = pd.DataFrame(samples.numpy())
df.columns = ['Number of Successes']
df.index.name = f"Sample of {N} trials with p={p}"
df