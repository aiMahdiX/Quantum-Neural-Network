import cirq
import numpy as np
from math import pi
from cirq import bloch_vector_from_state_vector

# =========================
# Setup
# =========================
q0, q1, q2 = cirq.LineQubit.range(3)
sim = cirq.Simulator()
np.random.seed(42)

# =========================
# Dataset: XOR
# =========================
dataset = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

# =========================
# Expectation <Z> for target qubit
# =========================
def expectation_z(circuit, qubit_index: int):
    result = sim.simulate(circuit)
    state = result.final_state_vector
    bv = bloch_vector_from_state_vector(state, qubit_index)
    return bv[2]

# =========================
# Ansatz: layered VQC
# =========================
def vqc_circuit(x, params, layers=3):
    """
    params length = layers*6 + 2
    """
    c = cirq.Circuit()
    idx = 0

    # Encoding
    c.append(cirq.ry(pi * x[0])(q0))
    c.append(cirq.ry(pi * x[1])(q1))
    c.append(cirq.ry(0.01)(q2))

    for _ in range(layers):
        c.append(cirq.ry(params[idx])(q0)); idx += 1
        c.append(cirq.rz(params[idx])(q0)); idx += 1

        c.append(cirq.ry(params[idx])(q1)); idx += 1
        c.append(cirq.rz(params[idx])(q1)); idx += 1

        c.append(cirq.ry(params[idx])(q2)); idx += 1
        c.append(cirq.rz(params[idx])(q2)); idx += 1

        # Entanglement
        c.append(cirq.CNOT(q0, q1))
        c.append(cirq.CNOT(q1, q2))
        c.append(cirq.CZ(q0, q2))

    # Readout bias
    c.append(cirq.ry(params[idx])(q2)); idx += 1
    c.append(cirq.rz(params[idx])(q2)); idx += 1

    return c

# =========================
# Prediction
# =========================
def predict_prob(x, params, layers=3):
    c = vqc_circuit(x, params, layers)
    z = expectation_z(c, qubit_index=2)
    return (1 - z) / 2

# =========================
# Loss: Binary cross-entropy
# =========================
def bce_loss(y, p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

# =========================
# Gradient: parameter-shift
# =========================
def gradient(x, y, params, layers=3, shift=pi/2):
    grads = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy(); p_plus[i] += shift
        p_minus = params.copy(); p_minus[i] -= shift

        p_plus_prob = predict_prob(x, p_plus, layers)
        p_minus_prob = predict_prob(x, p_minus, layers)

        dp_dtheta = (p_plus_prob - p_minus_prob) / 2.0

        # dL/dp (BCE)
        p_curr = predict_prob(x, params, layers)
        dL_dp = -(y / max(p_curr, 1e-9)) + ((1 - y) / max(1 - p_curr, 1e-9))

        grads[i] = dL_dp * dp_dtheta
    return grads

# =========================
# Training (single run, no restarts)
# =========================
def train(layers=3, lr=0.05, epochs=400):
    n_params = layers * 6 + 2
    params = np.random.randn(n_params) * 0.5

    for epoch in range(epochs):
        total_loss = 0.0
        grads_acc = np.zeros_like(params)

        for x, y in dataset:
            p = predict_prob(x, params, layers)
            total_loss += bce_loss(y, p)
            grads_acc += gradient(x, y, params, layers)

        grads_acc /= len(dataset)

        # Gradient clipping
        gnorm = np.linalg.norm(grads_acc)
        if gnorm > 5:
            grads_acc *= 5.0 / gnorm

        params -= lr * grads_acc

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:03d} | Loss = {total_loss:.4f}")

    return params

# =========================
# Run training
# =========================
print("Training improved VQC on XOR...\n")
params = train(layers=3, lr=0.05, epochs=400)

# =========================
# Evaluation
# =========================
print("\nXOR Evaluation:")
for x, y in dataset:
    prob = predict_prob(x, params, layers=3)
    pred = 1 if prob >= 0.5 else 0
    print(f"{x} → Prob(1) = {prob:.3f} | Pred = {pred} | Label = {y}")

# =========================
# Bloch vectors
# =========================
def bloch_vec(circuit, qubit_index):
    result = sim.simulate(circuit)
    state = result.final_state_vector
    return bloch_vector_from_state_vector(state, qubit_index)

print("\nBloch vectors for input [1, 0]:")
test_input = [1, 0]
circuit = vqc_circuit(test_input, params, layers=3)
print("q0:", bloch_vec(circuit, 0))
print("q1:", bloch_vec(circuit, 1))
print("q2:", bloch_vec(circuit, 2))