"""
mg_model.py
Shared quantum circuit + model architecture definitions.
Imported by BOTH train_pipeline.py (to train) and app.py (to rebuild the
architecture before loading saved weights). Keeping this in one file
guarantees the website uses the exact same architecture it was trained with.

NOTE: pennylane.qnn.KerasLayer was removed in PennyLane v0.42 (Keras 2 is no
longer maintained). Rather than pin to an old, increasingly incompatible
PennyLane version, this file defines a small custom Keras layer that wraps
the QNode directly -- it only depends on core PennyLane QNode/TF-interface
support, which is still fully maintained.
"""
import pennylane as qml
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Layer
from tensorflow.keras.optimizers import Adam

RANDOM_STATE = 42
N_QUBITS = 2
N_LAYERS = 3

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS,
    shots=None,
    seed=RANDOM_STATE
)


@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]


weight_shapes = {"weights": (N_LAYERS, N_QUBITS)}  # kept for reference/compatibility


class QuantumLayer(Layer):
    """
    Minimal custom Keras layer wrapping a PennyLane QNode.
    Replaces pennylane.qnn.KerasLayer (removed in PennyLane v0.42). Holds the
    circuit's trainable weights as a standard Keras variable and runs the
    QNode per-sample via tf.map_fn, so it plugs into any tf.keras.Sequential
    model exactly like a normal layer, and its weights save/load with the
    standard model.save_weights() / model.load_weights() calls.
    """
    def __init__(self, qnode, n_qubits, n_layers, **kwargs):
        super().__init__(**kwargs)
        self.qnode = qnode
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def build(self, input_shape):
        self.circuit_weights = self.add_weight(
            name="quantum_weights",
            shape=(self.n_layers, self.n_qubits),
            initializer="random_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        def run_one(x):
            # Keras 3 wraps tensors in its own type, which PennyLane's backend
            # dispatcher (autoray) doesn't recognize. Converting to a plain
            # tf.Tensor here keeps gradients intact while giving PennyLane
            # something it knows how to handle.
            x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
            w_tf = tf.convert_to_tensor(self.circuit_weights, dtype=tf.float32)
            outputs = self.qnode(x_tf, w_tf)
            # PennyLane's default.qubit simulator always computes internally
            # in float64, regardless of the input dtype. Cast back to float32
            # so it matches the fn_output_signature below (and the rest of
            # the float32 Keras model) -- otherwise backprop fails with a
            # float64/float32 mismatch.
            outputs = [tf.cast(o, tf.float32) for o in outputs]
            return tf.stack(outputs)
        return tf.map_fn(run_one, inputs, fn_output_signature=tf.float32)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_qubits)


def build_hybrid_model(input_features):
    quantum_layer = QuantumLayer(quantum_circuit, N_QUBITS, N_LAYERS)
    model = Sequential([
        Input(shape=(input_features,)),
        Dense(64, activation="relu", kernel_initializer="glorot_uniform"),
        Dropout(0.25),
        Dense(32, activation="relu", kernel_initializer="glorot_uniform"),
        Dropout(0.20),
        Dense(N_QUBITS),
        quantum_layer,
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_classical_twin_model(input_features):
    model = Sequential([
        Input(shape=(input_features,)),
        Dense(64, activation="relu", kernel_initializer="glorot_uniform"),
        Dropout(0.25),
        Dense(32, activation="relu", kernel_initializer="glorot_uniform"),
        Dropout(0.20),
        Dense(N_QUBITS, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model
