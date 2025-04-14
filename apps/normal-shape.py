import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")

# --- Sidebar controls ---
st.sidebar.title("Covariance Matrix: Σ")
a = st.sidebar.slider("Σ[0,0] (variance of x)", 0.1, 5.0, 2.0)
b = st.sidebar.slider("Σ[0,1] = Σ[1,0] (covariance)", -3.0, 3.0, 0.5)
d = st.sidebar.slider("Σ[1,1] (variance of y)", 0.1, 5.0, 1.0)
Sigma = torch.tensor([[a, b], [b, d]])

# --- Check PSD ---
eigvals = torch.linalg.eigvalsh(Sigma)
if torch.any(eigvals <= 0):
    st.error("❌ Σ is not positive definite. Adjust sliders.")
    st.stop()

# --- Eigendecomposition ---
eigvals, eigvecs = torch.linalg.eigh(Sigma)
Lambda_sqrt = torch.diag(torch.sqrt(eigvals))
transform = eigvecs @ Lambda_sqrt

# --- Sample from z ~ N(0, I), transform to x ---
z = torch.randn(1000, 2)
x = z @ transform.T

# --- Ellipse for contour ---
theta = torch.linspace(0, 2 * torch.pi, 100)
circle = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
ellipse = circle @ transform.T

# --- Plot ---
col1, col2 = st.columns([2, 1])
with col1:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("equal")
    ax.set_title("Samples from MVN + Eigenvectors", fontsize=12)

    ax.scatter(x[:, 0], x[:, 1], alpha=0.3, label='Samples', color='grey')
    ax.plot(ellipse[:, 0], ellipse[:, 1], color="black", lw=2, label="1-std Ellipse")

    colors = ["tab:blue", "tab:red"]  # Change to blue and red
    for i in range(2):
        vec = eigvecs[:, i] * torch.sqrt(eigvals[i]) * 3
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.1, color=colors[i], length_includes_head=True)
        ax.text(vec[0]*1.1, vec[1]*1.1, f"u{i+1} = ({eigvecs[0, i]:.2f}, {eigvecs[1, i]:.2f})", 
                color=colors[i], fontsize=10)

    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])  # Remove ticks for a cleaner look
    st.pyplot(fig)

with col2:
    st.markdown("### 📘 Intuition")

    st.markdown(r"$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$")
    st.markdown(r"$$\Sigma = U \Lambda U^\top$$")
    st.markdown(r"$$\mathbf{x} = U \Lambda^{1/2} \mathbf{z}$$")

    st.markdown(r"""
    - **<span style="color:blue;">U</span>**: rotates (eigenvectors)  
    - **<span style="color:red;">Λ<sup>1/2</sup></span>**: stretches (by square roots of eigenvalues)
    """, unsafe_allow_html=True)

    st.markdown(r"""
    **Resulting distribution**:  
    - Elliptical contours aligned to eigenvectors  
    - Axis lengths ∝ \( \sqrt{\lambda_1}, \sqrt{\lambda_2} \)
    """)

    st.markdown("### 🧮 Matrix Details")

    st.latex(rf"\Sigma = \begin{{bmatrix}} {a:.2f} & {b:.2f} \\ {b:.2f} & {d:.2f} \end{{bmatrix}}")

    U = eigvecs.numpy()
    L = eigvals.numpy()

    st.markdown(f"<span style='color:blue;'>U =</span>", unsafe_allow_html=True)
    st.latex(rf"\begin{{bmatrix}} {U[0,0]:.2f} & {U[0,1]:.2f} \\ {U[1,0]:.2f} & {U[1,1]:.2f} \end{{bmatrix}}")

    st.markdown(f"<span style='color:red;'>Λ =</span>", unsafe_allow_html=True)
    st.latex(rf"\begin{{bmatrix}} {L[0]:.2f} & 0 \\ 0 & {L[1]:.2f} \end{{bmatrix}}")
