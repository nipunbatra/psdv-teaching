import streamlit as st
import numpy as np
import plotly.graph_objects as go

def bayes_theorem(prior, sensitivity, specificity):
    """Computes the posterior probability P(COVID | Test+) using Bayes' theorem."""
    false_positive_rate = 1 - specificity
    total_positive = (sensitivity * prior) + (false_positive_rate * (1 - prior))
    posterior = (sensitivity * prior) / total_positive
    return posterior

# Streamlit UI setup
st.set_page_config(page_title="COVID-19 Testing: Bayes' Theorem", layout="centered")
st.title("🦠 COVID-19 Testing: Bayes' Theorem Calculator")
st.markdown("""This tool helps visualize how the prior probability of having COVID, test sensitivity, 
and test specificity affect the probability of actually having COVID given a positive test result.""")

# Move sliders to the sidebar
prior = st.sidebar.slider("🩺 Prior Probability of COVID (P(COVID))", 0.01, 0.5, 0.05, 0.01)
sensitivity = st.sidebar.slider("🔬 Test Sensitivity (P(Test+ | COVID))", 0.5, 1.0, 0.9, 0.01)
specificity = st.sidebar.slider("✅ Test Specificity (P(Test- | No COVID))", 0.5, 1.0, 0.95, 0.01)

# Compute Posterior Probability
posterior = bayes_theorem(prior, sensitivity, specificity)

# Display Posterior Probability Result in the main area
st.write(f"### 🎯 Posterior Probability of Having COVID Given a Positive Test: **{posterior:.4f} ({posterior * 100:.2f}%)**")

# Define Sensitivity and Specificity Ranges for Contour Plot
sensitivity_values = np.linspace(0.5, 1.0, 50)
specificity_values = np.linspace(0.5, 1.0, 50)
S, SP = np.meshgrid(sensitivity_values, specificity_values)
posterior_values = np.array([[bayes_theorem(prior, s, sp) for s, sp in zip(S_row, SP_row)] for S_row, SP_row in zip(S, SP)])

# Create Plotly Contour Plot
fig = go.Figure(data=go.Contour(
    z=posterior_values,
    x=sensitivity_values,
    y=specificity_values,
    colorbar=dict(title="Posterior Probability of COVID"),
    hovertemplate=(
        "Sensitivity: %{x:.2f}<br>"
        "Specificity: %{y:.2f}<br>"
        "Posterior Probability: %{z:.4f}<br>"
        "<extra></extra>"
    ),
))

# Update Layout for Contour Plot
fig.update_layout(
    title=f"Posterior Probability Contour (Prior={prior})",
    xaxis_title="Sensitivity (P(Test+ | COVID))",
    yaxis_title="Specificity (P(Test- | No COVID))",
    template="plotly_dark",
    xaxis=dict(scaleanchor="y"),  # To ensure a 1:1 aspect ratio
    yaxis=dict(scaleanchor="x"),  # To ensure a 1:1 aspect ratio
)

# Display Contour Plot
st.plotly_chart(fig)
