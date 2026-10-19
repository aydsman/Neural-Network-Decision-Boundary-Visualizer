import numpy as np
import matplotlib.pyplot as plt

SEED = 42
TRAIN_FRACTION = 0.8
LEARNING_RATE = 0.1
DEFAULT_DATASET = "Classic Donut"
DEFAULT_HIDDEN_LAYERS = 3
HIDDEN_LAYER_CHOICES = (1, 3, 5, 7, 9)
N_INPUTS = 2
N_OUTPUTS = 2
TARGET_SECONDS = 20
# Patience is a fraction of the epoch budget: Spirals has a long stretch where loss rises before it falls
EARLY_STOP_PATIENCE_FRACTION = 0.25
EARLY_STOP_MIN_DELTA = 1e-4
PRINT_EVERY = 100

RED, BLUE = 0, 1
CLASS_NAMES = ("Red", "Blue")


# ---------- Datasets ----------
# Every generator returns X (N x 2) and labels (N,), with label 0 = red and 1 = blue.

def _polar(r, theta):
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _disk(rng, n, radius):
    # sqrt keeps the dots evenly spread instead of bunched at the center
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    return _polar(r, rng.uniform(0, 2 * np.pi, n))


def _ring(rng, n, r_min, r_max):
    return _polar(rng.uniform(r_min, r_max, n), rng.uniform(0, 2 * np.pi, n))


def _stack(rng, groups, noise):
    X = np.vstack([pts for pts, _ in groups])
    labels = np.concatenate([np.full(len(pts), label) for pts, label in groups])
    return X + rng.normal(0, noise, X.shape), labels


def classic_donut(rng, n):
    half = n // 2
    return _stack(rng, [(_disk(rng, half, 1.0), RED),
                        (_ring(rng, half, 2.0, 3.0), BLUE)], noise=0.1)


def tight_donut(rng, n):
    half = n // 2
    return _stack(rng, [(_disk(rng, half, 1.5), RED),
                        (_ring(rng, half, 1.7, 2.7), BLUE)], noise=0.2)


def wobbly_donut(rng, n):
    half = n // 2
    petals = 5
    theta_red = rng.uniform(0, 2 * np.pi, half)
    r_red = np.sqrt(rng.uniform(0, 1, half)) * (1.4 + 0.5 * np.sin(petals * theta_red))
    theta_blue = rng.uniform(0, 2 * np.pi, half)
    r_blue = 2.3 + 0.5 * np.sin(petals * theta_blue) + rng.uniform(0, 0.8, half)
    return _stack(rng, [(_polar(r_red, theta_red), RED),
                        (_polar(r_blue, theta_blue), BLUE)], noise=0.08)


def bullseye(rng, n):
    quarter = n // 4
    return _stack(rng, [(_disk(rng, quarter, 1.0), RED),
                        (_ring(rng, 2 * quarter, 1.6, 2.6), BLUE),
                        (_ring(rng, quarter, 3.2, 4.2), RED)], noise=0.1)


def spirals(rng, n):
    half = n // 2
    groups = []
    for label, offset in ((RED, 0.0), (BLUE, np.pi)):
        t = np.sqrt(rng.uniform(0.02, 1, half))
        groups.append((_polar(4.0 * t, 3 * np.pi * t + offset), label))
    return _stack(rng, groups, noise=0.15)


# "epochs" is about 2x where training accuracy settles with the default 3 hidden layers at lr 0.1,
# so the boundary forms in the first half and confidence sharpens in the second.
DATASETS = {
    "Classic Donut": {"make": classic_donut, "n": 400, "epochs": 400},
    "Tight Donut":   {"make": tight_donut,   "n": 400, "epochs": 1500},
    "Wobbly Donut":  {"make": wobbly_donut,  "n": 500, "epochs": 800},
    "Bullseye":      {"make": bullseye,      "n": 600, "epochs": 5000},
    "Spirals":       {"make": spirals,       "n": 800, "epochs": 20000},
}


def make_dataset(name, rng):
    spec = DATASETS[name]
    return spec["make"](rng, spec["n"])


def default_speed(name):
    """Epochs per second that finishes the dataset's budget in about TARGET_SECONDS."""
    return max(1, round(DATASETS[name]["epochs"] / TARGET_SECONDS))


def one_hot(labels):
    return np.eye(N_OUTPUTS)[labels]


def split(X, labels, rng):
    order = rng.permutation(len(labels))
    X, labels = X[order], labels[order]
    cut = int(len(labels) * TRAIN_FRACTION)
    return X[:cut], labels[:cut], X[cut:], labels[cut:]


# ---------- Architecture ----------

def diamond(hidden_layers):
    """Hidden layer widths: start at 4, grow by 2 to the middle, then shrink back. 3 -> [4, 6, 4]."""
    if hidden_layers not in HIDDEN_LAYER_CHOICES:
        raise ValueError(f"hidden_layers must be one of {HIDDEN_LAYER_CHOICES}")
    half = hidden_layers // 2
    rising = [4 + 2 * i for i in range(half + 1)]
    return rising + rising[-2::-1]


def layer_sizes(hidden_layers):
    return [N_INPUTS] + diamond(hidden_layers) + [N_OUTPUTS]


def init_params(sizes, rng):
    """He initialization: weights ~ N(0, sqrt(2 / n_in)), biases start at 0."""
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        W = rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_out))
        b = np.zeros(n_out)
        params.append((W, b))
    return params


# ---------- The math ----------

def relu(Z):
    return np.maximum(0, Z)


def softmax(Z):
    # Subtracting the row max stops exp() from overflowing; the probabilities are unchanged
    exps = np.exp(Z - Z.max(axis=1, keepdims=True))
    return exps / exps.sum(axis=1, keepdims=True)


def cross_entropy(Y, P):
    P = np.clip(P, 1e-12, 1.0)
    return -np.mean(np.sum(Y * np.log(P), axis=1))


def forward(X, params):
    """Returns the linear sums Zs[l] and activations As[l]; As[0] is X and As[-1] is the bosses' output."""
    Zs, As = [], [X]
    last = len(params) - 1
    for l, (W, b) in enumerate(params):
        Z = As[-1] @ W + b
        Zs.append(Z)
        As.append(softmax(Z) if l == last else relu(Z))
    return Zs, As


def backward(Y, params, Zs, As):
    """Gradients (dW, db) for every layer, from the output back to the first hidden layer."""
    N = len(Y)
    grads = [None] * len(params)
    dZ = As[-1] - Y
    for l in range(len(params) - 1, -1, -1):
        W, _ = params[l]
        dW = As[l].T @ dZ / N
        db = dZ.sum(axis=0) / N
        grads[l] = (dW, db)
        if l > 0:
            dA = dZ @ W.T
            dZ = dA * (Zs[l - 1] > 0)
    return grads


def train_step(X, Y, params, lr):
    """One epoch: forward pass, loss, full backward pass, then update every layer."""
    Zs, As = forward(X, params)
    loss = cross_entropy(Y, As[-1])
    grads = backward(Y, params, Zs, As)
    params = [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(params, grads)]
    return params, loss


def predict_proba(X, params):
    return forward(X, params)[1][-1]


def predict(X, params):
    return predict_proba(X, params).argmax(axis=1)


def accuracy(labels, predictions):
    return np.mean(labels == predictions)


def activation_rates(X, params):
    """For each hidden layer, the fraction of dots that switch each neuron on (Z > 0)."""
    Zs, _ = forward(X, params)
    return [(Z > 0).mean(axis=0) for Z in Zs[:-1]]


class EarlyStopping:
    """Stops when the loss hasn't improved by min_delta for a quarter of the epoch budget."""

    def __init__(self, max_epochs, min_delta=EARLY_STOP_MIN_DELTA):
        self.patience = max(1, int(max_epochs * EARLY_STOP_PATIENCE_FRACTION))
        self.min_delta = min_delta
        self.best = np.inf
        self.wait = 0

    def should_stop(self, loss):
        if loss < self.best - self.min_delta:
            self.best = loss
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience


def train(X, Y, params, lr, max_epochs, verbose=True):
    losses = []
    stopper = EarlyStopping(max_epochs)
    for epoch in range(1, max_epochs + 1):
        params, loss = train_step(X, Y, params, lr)
        losses.append(loss)
        if verbose and epoch % PRINT_EVERY == 0:
            print(f"Epoch {epoch:5d} | loss {loss:.4f}")
        if stopper.should_stop(loss):
            if verbose:
                print(f"Early stop at epoch {epoch}: loss improved by less than "
                      f"{EARLY_STOP_MIN_DELTA} over the last {stopper.patience} epochs")
            break
    return params, losses


# ---------- Static plots ----------

def plot_result(X, labels, params, losses, title):
    fig, (ax_data, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))

    margin = 0.5
    x1 = np.linspace(X[:, 0].min() - margin, X[:, 0].max() + margin, 200)
    x2 = np.linspace(X[:, 1].min() - margin, X[:, 1].max() + margin, 200)
    G1, G2 = np.meshgrid(x1, x2)
    p_blue = predict_proba(np.column_stack([G1.ravel(), G2.ravel()]), params)[:, BLUE]
    p_blue = p_blue.reshape(G1.shape)

    ax_data.contourf(G1, G2, p_blue, levels=20, cmap="RdBu", vmin=0, vmax=1, alpha=0.4)
    ax_data.contour(G1, G2, p_blue, levels=[0.5], colors="black", linewidths=2)
    ax_data.scatter(*X[labels == RED].T, c="tab:red", s=12, label="Red (inner)")
    ax_data.scatter(*X[labels == BLUE].T, c="tab:blue", s=12, label="Blue (outer)")
    ax_data.set_aspect("equal")
    ax_data.set_title(title)
    ax_data.legend(loc="upper right", fontsize=8)

    ax_loss.plot(range(1, len(losses) + 1), losses)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training loss (cross-entropy)")
    ax_loss.set_title("Loss curve")
    ax_loss.grid(alpha=0.3)
    fig.tight_layout()


# ---------- Run everything ----------

def main(dataset=DEFAULT_DATASET, hidden_layers=DEFAULT_HIDDEN_LAYERS):
    rng = np.random.default_rng(SEED)
    X, labels = make_dataset(dataset, rng)
    X_train, y_train, X_test, y_test = split(X, labels, rng)

    sizes = layer_sizes(hidden_layers)
    params = init_params(sizes, rng)
    print(f"Dataset: {dataset} ({len(labels)} dots) | network {sizes}")

    params, losses = train(X_train, one_hot(y_train), params, LEARNING_RATE,
                           DATASETS[dataset]["epochs"])

    train_acc = accuracy(y_train, predict(X_train, params))
    test_acc = accuracy(y_test, predict(X_test, params))
    print()
    print(f"Epochs run:     {len(losses)}")
    print(f"Final loss:     {losses[-1]:.4f}")
    print(f"Train accuracy: {train_acc:.2%} ({len(y_train)} dots)")
    print(f"Test accuracy:  {test_acc:.2%} ({len(y_test)} dots)")

    plot_result(X, labels, params, losses, f"{dataset}, hidden layers {diamond(hidden_layers)}")
    plt.show()


if __name__ == "__main__":
    main()
