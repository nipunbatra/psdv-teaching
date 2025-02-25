import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_distribution(distribution, params, x_range=(-5, 5)):
    x = torch.linspace(x_range[0], x_range[1], 1000)
    
    if distribution == "Normal":
        mu, sigma = params
        dist = torch.distributions.Normal(mu, sigma)
        y = dist.log_prob(x).exp()
    elif distribution == "Exponential":
        rate, = params
        dist = torch.distributions.Exponential(rate)
        x = torch.linspace(0, x_range[1], 1000)
        y = dist.log_prob(x).exp()
    elif distribution == "Uniform":
        a, b = params
        dist = torch.distributions.Uniform(a, b)
        x = torch.linspace(a-2, b+2, 1000)
        mask = (x >= a) & (x <= b)
        y = torch.zeros_like(x)
        y[mask] = dist.log_prob(x[mask]).exp()
    elif distribution == "Beta":
        alpha, beta = params
        x = torch.linspace(0, 1, 1000)
        dist = torch.distributions.Beta(alpha, beta)
        y = dist.log_prob(x).exp()
    elif distribution == "Gamma":
        shape, scale = params
        dist = torch.distributions.Gamma(shape, scale)
        x = torch.linspace(0, x_range[1], 1000)
        y = dist.log_prob(x).exp()
    else:
        st.error("Distribution not implemented!")
        return
    
    fig, ax = plt.subplots()
    ax.plot(x.numpy(), y.numpy(), label=f'{distribution} PDF', color='blue')
    ax.set_title(f'{distribution} Distribution')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.legend()
    st.pyplot(fig)

# Sidebar for navigation
st.sidebar.title("Select Distribution")
dist_option = st.sidebar.radio("Choose a distribution:", ["Normal", "Exponential", "Uniform", "Beta", "Gamma"])

st.title("Continuous Probability Distributions")
st.header(f"{dist_option} Distribution")

# Parameters selection based on distribution
if dist_option == "Normal":
    mu = st.slider("Mean (μ)", -3.0, 3.0, 0.0)
    sigma = st.slider("Standard Deviation (σ)", 0.1, 3.0, 1.0)
    params = (mu, sigma)
elif dist_option == "Exponential":
    rate = st.slider("Rate (λ)", 0.1, 5.0, 1.0)
    params = (rate,)
elif dist_option == "Uniform":
    a = st.slider("Lower Bound (a)", -5.0, 0.0, -2.0)
    b = st.slider("Upper Bound (b)", 0.0, 5.0, 2.0)
    params = (a, b)
elif dist_option == "Beta":
    alpha = st.slider("Alpha (α)", 0.1, 5.0, 2.0)
    beta = st.slider("Beta (β)", 0.1, 5.0, 2.0)
    params = (alpha, beta)
elif dist_option == "Gamma":
    shape = st.slider("Shape (k)", 0.1, 5.0, 2.0)
    scale = st.slider("Scale (θ)", 0.1, 5.0, 1.0)
    params = (shape, scale)

plot_distribution(dist_option, params)