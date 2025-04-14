import streamlit as st
import torch
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Markov & Chebyshev Inequality Explorer")

# Sidebar controls
dist_name = st.sidebar.selectbox("Distribution", ["Exponential", "Normal", "Uniform", "Bernoulli"])
num_samples = st.sidebar.slider("Number of samples", 1000, 100000, 10000, step=1000)

# Distribution parameters
if dist_name == "Exponential":
    rate = st.sidebar.slider("Rate (λ)", 0.1, 5.0, 1.0)
    dist = torch.distributions.Exponential(rate=rate)
    samples = dist.sample((num_samples,))
    mean = 1 / rate
    var = 1 / rate**2
elif dist_name == "Normal":
    mu = st.sidebar.slider("Mean (μ)", -5.0, 5.0, 0.0)
    sigma = st.sidebar.slider("Std Dev (σ)", 0.1, 5.0, 1.0)
    dist = torch.distributions.Normal(loc=mu, scale=sigma)
    samples = dist.sample((num_samples,))
    mean = mu
    var = sigma**2
elif dist_name == "Uniform":
    a = st.sidebar.slider("a", -10.0, 5.0, 0.0)
    b = st.sidebar.slider("b", a + 0.1, 10.0, 1.0)
    dist = torch.distributions.Uniform(low=a, high=b)
    samples = dist.sample((num_samples,))
    mean = (a + b) / 2
    var = (b - a)**2 / 12
elif dist_name == "Bernoulli":
    p = st.sidebar.slider("p", 0.0, 1.0, 0.5)
    dist = torch.distributions.Bernoulli(probs=p)
    samples = dist.sample((num_samples,))
    mean = p
    var = p * (1 - p)

# Select Inequality
inequality = st.sidebar.selectbox("Inequality", ["Markov", "Chebyshev"])

fig, ax = plt.subplots(figsize=(10, 5))

if inequality == "Markov":
    st.latex(r"\mathbb{P}(X \geq a) \leq \frac{\mathbb{E}[X]}{a}")

    if (samples < 0).any():
        st.error("Markov inequality requires non-negative random variables.")
    else:
        a_min = st.sidebar.slider("a min", float(samples.min()), float(samples.max() / 2), float(mean), step=0.1)
        a_max = st.sidebar.slider("a max", float(samples.max() / 2), float(samples.max()), float(samples.max()), step=0.1)
        a_vals = torch.linspace(a_min, a_max, 100)

        prob_emp = torch.tensor([(samples >= a).float().mean().item() for a in a_vals])
        bound = mean / a_vals

        ax.plot(a_vals, prob_emp, label=r'$\mathbb{P}(X \geq a)$ (Empirical)', color='blue')
        ax.plot(a_vals, bound, '--', label=r'$\frac{{\mathbb{{E}}[X]}}{{a}}$ (Markov Bound)', color='red')
        ax.set_xlabel("a")
        ax.set_ylabel("Probability")
        ax.set_title("Markov Inequality")
        ax.legend()
        ax.grid(True)

elif inequality == "Chebyshev":
    st.latex(r"\mathbb{P}(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}")

    k_min = st.sidebar.slider("k min", 0.1, 1.0, 0.5, step=0.1)
    k_max = st.sidebar.slider("k max", 1.0, 10.0, 5.0, step=0.5)
    k_vals = torch.linspace(k_min, k_max, 100)

    prob_emp = torch.tensor([(torch.abs(samples - mean) >= k * var**0.5).float().mean().item() for k in k_vals])
    bound = 1 / k_vals**2

    ax.plot(k_vals, prob_emp, label=r'$\mathbb{P}(|X - \mu| \geq k\sigma)$ (Empirical)', color='blue')
    ax.plot(k_vals, bound, '--', label=r'$\frac{1}{k^2}$ (Chebyshev Bound)', color='red')
    ax.set_xlabel("k")
    ax.set_ylabel("Probability")
    ax.set_title("Chebyshev Inequality")
    ax.legend()
    ax.grid(True)

st.pyplot(fig)
