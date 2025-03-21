import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go

# Streamlit App
st.title("Geogebra Clone in Streamlit")

# Sidebar options
st.sidebar.header("3D Functions and Points")

# Figure size option
fig_size = st.sidebar.slider("Figure Size", min_value=500, max_value=1000, value=700, step=50)

# Choose input type: function or points
input_type = st.sidebar.radio("Input Type", ["Functions", "Points"], index=0)

# Manage functions
if "functions" not in st.session_state:
    st.session_state.functions = ["sin(x) + y**2"]

# Manage points
if "points" not in st.session_state:
    st.session_state.points = []

if input_type == "Functions":
    # Add function button
    if st.sidebar.button("Add Function"):
        st.session_state.functions.append("x**2 + y**2")
    
    # Input fields for functions with delete buttons
    func_exprs = []
    cols = st.sidebar.columns([4, 1])
    for i, func in enumerate(st.session_state.functions):
        func_exprs.append(cols[0].text_input(f"Function {i+1}", func, key=f"func_{i}"))
        if cols[1].button("❌", key=f"delete_{i}"):
            del st.session_state.functions[i]
            st.rerun()

elif input_type == "Points":
    # Option to add uniform distribution of points
    if st.sidebar.button("Add Uniform Points"):
        n = 100  # Number of points
        x = np.random.uniform(0, 2, n)
        y = np.random.uniform(0, 2, n)
        z = np.full(n, 0.25)
        st.session_state.points = np.column_stack((x, y, z)).tolist()
    
    # Option to clear points
    if st.sidebar.button("Clear Points"):
        st.session_state.points = []

# Define symbols
x, y = sp.symbols('x y')

# Create 3D Plot
fig = go.Figure()
colors = ["red", "blue", "green", "purple", "orange"]

if input_type == "Functions":
    # Plot functions
    x_vals = np.linspace(-5, 5, 30)
    y_vals = np.linspace(-5, 5, 30)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')
    
    for i, func_expr_str in enumerate(func_exprs):
        func_expr = sp.sympify(func_expr_str, locals={'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp})
        func_lambdified = sp.lambdify((x, y), func_expr, 'numpy')
        Z = func_lambdified(X, Y)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, colors[i % len(colors)]], [1, colors[i % len(colors)]]], opacity=0.7, showscale=False, name=func_expr_str))

elif input_type == "Points":
    # Plot points
    if st.session_state.points:
        points = np.array(st.session_state.points)
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers', marker=dict(size=5, color='black'),
            name="Uniform Points"
        ))

# Display plot
fig.update_layout(
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
    width=fig_size, height=fig_size
)
st.plotly_chart(fig)