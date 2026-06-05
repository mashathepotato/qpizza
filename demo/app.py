"""Fraud-triage console: pick a backend, stream sample transactions, see the
quantum-kernel model flag fraud live. The 'more than numbers' demo shell."""
import numpy as np
import streamlit as st

from demo.inference import train_fraud_model, score_transaction

st.set_page_config(page_title="Quantum Fraud Triage", layout="wide")
st.title("Quantum Fraud Triage - OP Pohjola")

backend = st.sidebar.selectbox(
    "Quantum backend", ["local_aer", "lumi_aer", "q50_fake", "q50_hw"],
    help="q50_fake = VTT Q50 native gates + noise (offline). q50_hw = real Q50 (on-site).")
n_features = st.sidebar.slider("Feature qubits", 2, 8, 4)
threshold = st.sidebar.slider("Flag threshold", 0.0, 1.0, 0.5)

@st.cache_resource
def _model(backend, n_features):
    return train_fraud_model(backend=backend, n=200, n_features=n_features, seed=0)

model = _model(backend, n_features)

st.subheader("Incoming transactions")
if st.button("Generate a batch"):
    rng = np.random.default_rng()
    rows = []
    for i in range(8):
        x = rng.normal(0, 1, n_features) + (1.8 if rng.random() < 0.3 else 0.0)
        p = score_transaction(model, x)
        rows.append({"txn": f"T{i:03d}", "fraud_prob": round(p, 3),
                     "flag": "FRAUD" if p >= threshold else "ok"})
    st.dataframe(rows, use_container_width=True)
    st.caption(f"Scored on backend: {backend} - quantum-kernel SVM "
               f"({n_features}-qubit feature map)")
