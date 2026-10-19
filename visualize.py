import copy
import time
import tkinter as tk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

import main as ml

TICK_MS = 30
MAX_SPEED = 2000
GRID = 150

RED_COLOR = "#d62728"
BLUE_COLOR = "#1f77b4"
POSITIVE_COLOR = "#2ca02c"
NEGATIVE_COLOR = "#8e44ad"

LIGHT_THEME = {
    "bg": "#f0f0f0", "fg": "black", "button": "#e1e1e1", "button_active": "#d0d0d0",
    "field": "white", "trough": "#c8c8c8", "figure": "white", "axes": "white",
    "grid": "#b0b0b0", "line": "black", "cut": "#555555",
    "node_dim": "#dcdcdc", "node_glow": "#ffb000", "node_edge": "#555555", "input_node": "#9e9e9e",
}
DARK_THEME = {
    "bg": "#1e1e1e", "fg": "white", "button": "#333333", "button_active": "#454545",
    "field": "#2b2b2b", "trough": "#3a3a3a", "figure": "#1e1e1e", "axes": "#121212",
    "grid": "#505050", "line": "white", "cut": "#bbbbbb",
    "node_dim": "#353535", "node_glow": "#ffd54a", "node_edge": "#9a9a9a", "input_node": "#8a8a8a",
}


def boundary_points(w, b, lim):
    """Two endpoints of the line w1*x1 + w2*x2 + b = 0 inside the square view, or None."""
    if abs(w[0]) < 1e-12 and abs(w[1]) < 1e-12:
        return None
    ends = np.array([-lim, lim])
    # Solve for whichever coordinate has the larger weight, so the line never divides by ~0
    if abs(w[1]) >= abs(w[0]):
        return np.column_stack([ends, -(w[0] * ends + b) / w[1]])
    return np.column_stack([-(w[1] * ends + b) / w[0], ends])


class RingApp:
    def __init__(self, root):
        self.root = root
        root.title("Neural Network Decision Boundary Visualizer")

        self.dark_mode = False
        self.theme = LIGHT_THEME
        self.running = False
        self.showing_loss = False
        self.pending_epochs = 0.0
        self.last_tick = time.perf_counter()
        self.hidden_layers = ml.DEFAULT_HIDDEN_LAYERS
        self.lr = ml.LEARNING_RATE

        self._build_controls()
        self._build_main_screen()
        self._build_loss_screen()
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self._build_net_panel()
        self.load_dataset(ml.DEFAULT_DATASET)
        self.apply_theme()
        self.root.after(TICK_MS, self.tick)

    # ---------- Layout ----------

    def _build_controls(self):
        bar = tk.Frame(self.root, padx=8, pady=4)
        bar.pack(fill=tk.X)
        self.play_button = tk.Button(bar, text="Play", width=8, command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Step", width=8, command=self.step_button).pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Reset", width=8, command=self.reset).pack(side=tk.LEFT, padx=2)

        tk.Label(bar, text="   Speed (epochs/sec):").pack(side=tk.LEFT)
        self.speed = tk.IntVar(value=1)
        tk.Scale(bar, from_=1, to=MAX_SPEED, orient=tk.HORIZONTAL, length=200,
                 variable=self.speed, showvalue=False).pack(side=tk.LEFT)
        tk.Spinbox(bar, from_=1, to=MAX_SPEED, width=6,
                   textvariable=self.speed).pack(side=tk.LEFT, padx=4)

        tk.Label(bar, text="   Learning rate:").pack(side=tk.LEFT)
        self.lr_var = tk.StringVar(value=f"{ml.LEARNING_RATE:.3f}")
        tk.Spinbox(bar, from_=0.001, to=1.0, increment=0.005, format="%.3f", width=7,
                   textvariable=self.lr_var).pack(side=tk.LEFT, padx=4)

        self.screen_button = tk.Button(bar, text="Show loss graph", width=16,
                                       command=self.toggle_screen)
        self.screen_button.pack(side=tk.RIGHT, padx=2)
        self.theme_button = tk.Button(bar, text="Dark mode", width=10, command=self.toggle_theme)
        self.theme_button.pack(side=tk.RIGHT, padx=2)

        bar2 = tk.Frame(self.root, padx=8, pady=2)
        bar2.pack(fill=tk.X)
        tk.Label(bar2, text="Dataset:").pack(side=tk.LEFT)
        self.dataset_var = tk.StringVar(value=ml.DEFAULT_DATASET)
        self.dataset_menu = tk.OptionMenu(bar2, self.dataset_var, *ml.DATASETS,
                                          command=self.load_dataset)
        self.dataset_menu.config(width=14)
        self.dataset_menu.pack(side=tk.LEFT, padx=4)

        tk.Label(bar2, text="   Hidden layers:").pack(side=tk.LEFT)
        self.hidden_scale = tk.Scale(bar2, from_=0, to=len(ml.HIDDEN_LAYER_CHOICES) - 1,
                                     orient=tk.HORIZONTAL, length=140, showvalue=False)
        self.hidden_scale.set(ml.HIDDEN_LAYER_CHOICES.index(self.hidden_layers))
        self.hidden_scale.config(command=self.on_hidden_change)
        self.hidden_scale.pack(side=tk.LEFT)
        self.arch_label = tk.Label(bar2, width=44, anchor="w", font=("Consolas", 10))
        self.arch_label.pack(side=tk.LEFT, padx=4)

        self.show_cuts = tk.BooleanVar(value=False)
        tk.Checkbutton(bar2, text="Show Layer 1 cuts", variable=self.show_cuts,
                       command=self.redraw).pack(side=tk.LEFT, padx=8)

        self.status = tk.Label(self.root, anchor="w", padx=10, font=("Consolas", 10))
        self.status.pack(fill=tk.X)

    def _build_main_screen(self):
        self.main_frame = tk.Frame(self.root)
        self.main_fig = Figure(figsize=(13, 6.4), dpi=100)
        self.ax_data, self.ax_net = self.main_fig.subplots(
            1, 2, gridspec_kw={"width_ratios": [1, 1.15]})
        self.main_fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.08, wspace=0.08)
        self.main_canvas = FigureCanvasTkAgg(self.main_fig, master=self.main_frame)
        self.main_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_loss_screen(self):
        self.loss_frame = tk.Frame(self.root)
        self.loss_fig = Figure(figsize=(13, 6.4), dpi=100)
        self.ax_loss = self.loss_fig.add_subplot()
        self.loss_line, = self.ax_loss.plot([], [])
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Training loss (cross-entropy)")
        self.ax_loss.set_title("Loss curve")
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, master=self.loss_frame)
        self.loss_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_data_panel(self):
        ax = self.ax_data
        ax.clear()
        all_X = np.vstack([self.X_train, self.X_test])
        self.lim = np.abs(all_X).max() + 0.6
        axis = np.linspace(-self.lim, self.lim, GRID)
        G1, G2 = np.meshgrid(axis, axis)
        self.grid_shape = G1.shape
        self.grid_x1, self.grid_x2 = G1, G2
        self.grid_points = np.column_stack([G1.ravel(), G2.ravel()])

        # RdBu runs red -> blue, so P(blue) = 0 shades red and P(blue) = 1 shades blue
        self.shading = ax.imshow(np.full(self.grid_shape, 0.5), origin="lower",
                                 extent=(-self.lim, self.lim, -self.lim, self.lim),
                                 cmap="RdBu", vmin=0, vmax=1, alpha=0.45,
                                 interpolation="bilinear", zorder=0)
        for X, y, filled in ((self.X_train, self.y_train, True), (self.X_test, self.y_test, False)):
            for label, color in ((ml.RED, RED_COLOR), (ml.BLUE, BLUE_COLOR)):
                pts = X[y == label]
                ax.scatter(pts[:, 0], pts[:, 1], s=14, zorder=3,
                           facecolors=color if filled else "none", edgecolors=color,
                           linewidths=0.6 if filled else 1.1)

        self.cuts = LineCollection([], linestyles="dashed", linewidths=1.0, zorder=2)
        ax.add_collection(self.cuts)
        self.contour = None

        ax.set_xlim(-self.lim, self.lim)
        ax.set_ylim(-self.lim, self.lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        self.boundary_proxy = Line2D([], [], linewidth=2, label="decision boundary")
        self.cuts_proxy = Line2D([], [], linestyle="--", linewidth=1, label="Layer 1 cuts")
        handles = [
            Line2D([], [], marker="o", linestyle="", color=RED_COLOR, label="Red (train)"),
            Line2D([], [], marker="o", linestyle="", color=BLUE_COLOR, label="Blue (train)"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                   markeredgecolor=RED_COLOR, label="Red (test)"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                   markeredgecolor=BLUE_COLOR, label="Blue (test)"),
            self.boundary_proxy, self.cuts_proxy,
        ]
        self.data_legend = ax.legend(handles=handles, loc="upper right", fontsize=7,
                                     framealpha=0.85)

    def _build_net_panel(self):
        ax = self.ax_net
        ax.clear()
        ax.set_axis_off()
        sizes = ml.layer_sizes(self.hidden_layers)
        tallest = max(sizes)

        columns = [np.column_stack([np.full(n, l), np.arange(n)[::-1] - (n - 1) / 2])
                   for l, n in enumerate(sizes)]
        self.node_xy = np.vstack(columns)
        starts = np.cumsum([0] + sizes)
        self.layer_slices = [slice(starts[l], starts[l + 1]) for l in range(len(sizes))]

        # Edges are ordered (layer, from-node, to-node) to match W.ravel() for each layer
        segments = [(columns[l][i], columns[l + 1][j])
                    for l in range(len(sizes) - 1)
                    for i in range(sizes[l]) for j in range(sizes[l + 1])]
        self.edges = LineCollection(segments, zorder=1, capstyle="round")
        ax.add_collection(self.edges)

        self.node_size = 420 if tallest <= 6 else max(110, 420 * (6 / tallest) ** 2)
        self.nodes = ax.scatter(self.node_xy[:, 0], self.node_xy[:, 1], s=self.node_size, zorder=3)
        self.dead_marks = ax.scatter([], [], marker="x", s=self.node_size * 0.45,
                                     linewidths=1.6, zorder=4)

        bottom = -(tallest - 1) / 2 - 0.9
        self.net_texts = []
        for l, n in enumerate(sizes):
            if l == 0:
                name = "Input"
            elif l == len(sizes) - 1:
                name = "Output"
            else:
                name = f"H{l}"
            self.net_texts.append(ax.text(l, bottom, f"{name}\n{n}", ha="center", va="top",
                                          fontsize=8))
        for (x, y), label in zip(columns[0], ("x1", "x2")):
            self.net_texts.append(ax.text(x - 0.25, y, label, ha="right", va="center", fontsize=9))
        for (x, y), label in zip(columns[-1], ml.CLASS_NAMES):
            self.net_texts.append(ax.text(x + 0.25, y, label, ha="left", va="center", fontsize=9))
        self.net_legend = None

        ax.set_xlim(-0.9, len(sizes) - 1 + 0.9)
        # Extra headroom keeps the legend clear of the tallest column
        ax.set_ylim(bottom - 0.9, tallest / 2 + 0.6 + 0.12 * tallest + 1.6)

        widths = ", ".join(str(n) for n in ml.diamond(self.hidden_layers))
        self.arch_label.config(text=f"{self.hidden_layers}  ->  [2] {widths} [2]")

    # ---------- Data and training ----------

    def load_dataset(self, name):
        self.dataset_name = name
        rng = np.random.default_rng(ml.SEED)
        X, labels = ml.make_dataset(name, rng)
        self.X_train, self.y_train, self.X_test, self.y_test = ml.split(X, labels, rng)
        self.Y_train = ml.one_hot(self.y_train)
        # Re-initializing from this saved state gives the same starting weights as main.py on every reset
        self.init_rng = copy.deepcopy(rng)
        self.budget = ml.DATASETS[name]["epochs"]
        self.speed.set(ml.default_speed(name))
        self._build_data_panel()
        self._style_figures()
        self.reset()

    def on_hidden_change(self, value):
        hidden = ml.HIDDEN_LAYER_CHOICES[int(float(value))]
        if hidden == self.hidden_layers:
            return
        self.hidden_layers = hidden
        self._build_net_panel()
        self._style_figures()
        self.reset()

    def reset(self):
        self.running = False
        self.play_button.config(text="Play")
        self.pending_epochs = 0.0
        rng = copy.deepcopy(self.init_rng)
        self.params = ml.init_params(ml.layer_sizes(self.hidden_layers), rng)
        self.epoch = 0
        self.losses = []
        self.stopper = ml.EarlyStopping(self.budget)
        self.end_reason = None
        self.ax_loss.set_xlim(0, self.budget)
        self.redraw()

    def finished(self):
        return self.end_reason is not None

    def current_lr(self):
        try:
            self.lr = min(max(float(self.lr_var.get()), 1e-5), 5.0)
        except ValueError:
            pass
        return self.lr

    def current_speed(self):
        try:
            return min(max(int(self.speed.get()), 1), MAX_SPEED)
        except (tk.TclError, ValueError):
            return ml.default_speed(self.dataset_name)

    def step_once(self):
        if self.finished():
            return
        self.params, loss = ml.train_step(self.X_train, self.Y_train, self.params, self.current_lr())
        self.losses.append(loss)
        self.epoch += 1
        if not np.isfinite(loss):
            self.end_reason = "diverged: loss is NaN, try a lower learning rate"
        elif self.epoch >= self.budget:
            self.end_reason = "finished the epoch budget"
        elif self.stopper.should_stop(loss):
            self.end_reason = f"stopped early: no improvement for {self.stopper.patience} epochs"

    def step_button(self):
        self.step_once()
        self.redraw()

    def toggle_play(self):
        if self.running:
            self.running = False
        else:
            if self.finished():
                self.reset()
            self.running = True
        self.play_button.config(text="Pause" if self.running else "Play")

    def tick(self):
        now = time.perf_counter()
        elapsed = now - self.last_tick
        self.last_tick = now
        if self.running:
            # Redrawing takes longer than TICK_MS, so use the real elapsed time to hold the chosen speed
            speed = self.current_speed()
            self.pending_epochs = min(self.pending_epochs + speed * elapsed, speed * 0.25)
            n = int(self.pending_epochs)
            self.pending_epochs -= n
            for _ in range(n):
                self.step_once()
                if self.finished():
                    break
            if n:
                self.redraw()
            if self.finished():
                self.running = False
                self.play_button.config(text="Play")
        self.root.after(TICK_MS, self.tick)

    # ---------- Drawing ----------

    def redraw(self):
        train_acc = ml.accuracy(self.y_train, ml.predict(self.X_train, self.params))
        test_acc = ml.accuracy(self.y_test, ml.predict(self.X_test, self.params))
        loss_text = f"{self.losses[-1]:.4f}" if self.losses else "   -  "
        state = self.end_reason or ("training" if self.running else "paused")
        self.status.config(text=(
            f"{self.dataset_name}   Epoch {self.epoch}/{self.budget}   loss {loss_text}   "
            f"train acc {train_acc:6.2%}   test acc {test_acc:6.2%}   ({state})"
        ))

        if self.showing_loss:
            self.loss_line.set_data(range(1, len(self.losses) + 1), self.losses)
            finite = [v for v in self.losses if np.isfinite(v)]
            self.ax_loss.set_ylim(0, max(finite) * 1.05 if finite else 1.0)
            self.loss_canvas.draw_idle()
        else:
            self._draw_data_panel()
            self._draw_net_panel()
            self.main_canvas.draw_idle()

    def _draw_data_panel(self):
        t = self.theme
        p_blue = ml.predict_proba(self.grid_points, self.params)[:, ml.BLUE].reshape(self.grid_shape)
        p_blue = np.nan_to_num(p_blue, nan=0.5)
        self.shading.set_data(p_blue)

        if self.contour is not None:
            self.contour.remove()
            self.contour = None
        if p_blue.min() < 0.5 < p_blue.max():
            self.contour = self.ax_data.contour(self.grid_x1, self.grid_x2, p_blue, levels=[0.5],
                                                colors=[t["line"]], linewidths=2, zorder=4)

        segments = []
        if self.show_cuts.get():
            W1, b1 = self.params[0]
            for j in range(W1.shape[1]):
                pts = boundary_points(W1[:, j], b1[j], self.lim)
                if pts is not None:
                    segments.append(pts)
        self.cuts.set_segments(segments)
        self.cuts.set_color(t["cut"])

        self.ax_data.set_title(f"{self.dataset_name}   (epoch {self.epoch})", color=t["fg"])

    def _draw_net_panel(self):
        t = self.theme
        weights = np.concatenate([W.ravel() for W, _ in self.params])
        weights = np.nan_to_num(weights)
        strength = np.abs(weights) / max(np.abs(weights).max(), 1e-12)
        colors = np.array([to_rgba(POSITIVE_COLOR) if w > 0 else to_rgba(NEGATIVE_COLOR)
                           for w in weights])
        colors[:, 3] = 0.12 + 0.88 * strength
        self.edges.set_color(colors)
        self.edges.set_linewidths(0.25 + 3.0 * strength)

        n_nodes = len(self.node_xy)
        faces = np.zeros((n_nodes, 4))
        faces[self.layer_slices[0]] = to_rgba(t["input_node"])
        dim, glow = np.array(to_rgb(t["node_dim"])), np.array(to_rgb(t["node_glow"]))
        dead = np.zeros(n_nodes, dtype=bool)
        rates = ml.activation_rates(self.X_train, self.params)
        for layer_slice, rate in zip(self.layer_slices[1:-1], rates):
            faces[layer_slice, :3] = dim + rate[:, None] * (glow - dim)
            faces[layer_slice, 3] = 1.0
            dead[layer_slice] = rate == 0
        out = self.layer_slices[-1]
        faces[out] = [to_rgba(RED_COLOR), to_rgba(BLUE_COLOR)]

        self.nodes.set_facecolors(faces)
        self.nodes.set_edgecolors(t["node_edge"])
        self.nodes.set_linewidths(0.9)
        self.dead_marks.set_offsets(self.node_xy[dead] if dead.any() else np.empty((0, 2)))
        self.dead_marks.set_color(t["fg"])

        self.ax_net.set_title("Network", color=t["fg"])

    # ---------- Screens and theme ----------

    def toggle_screen(self):
        self.showing_loss = not self.showing_loss
        if self.showing_loss:
            self.main_frame.pack_forget()
            self.loss_frame.pack(fill=tk.BOTH, expand=True)
            self.screen_button.config(text="Back to network")
        else:
            self.loss_frame.pack_forget()
            self.main_frame.pack(fill=tk.BOTH, expand=True)
            self.screen_button.config(text="Show loss graph")
        self.redraw()

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.theme = DARK_THEME if self.dark_mode else LIGHT_THEME
        self.theme_button.config(text="Light mode" if self.dark_mode else "Dark mode")
        self.apply_theme()

    def apply_theme(self):
        self.root.config(bg=self.theme["bg"])
        self._style_widget(self.root, self.theme)
        self._style_figures()
        self.redraw()

    def _style_figures(self):
        t = self.theme
        for fig in (self.main_fig, self.loss_fig):
            fig.set_facecolor(t["figure"])
        for ax in (self.ax_data, self.ax_loss):
            ax.set_facecolor(t["axes"])
            ax.tick_params(colors=t["fg"])
            ax.xaxis.label.set_color(t["fg"])
            ax.yaxis.label.set_color(t["fg"])
            ax.title.set_color(t["fg"])
            for spine in ax.spines.values():
                spine.set_edgecolor(t["fg"])
        self.ax_net.set_facecolor(t["figure"])
        self.ax_loss.grid(True, color=t["grid"], alpha=0.6)

        self.boundary_proxy.set_color(t["line"])
        self.cuts_proxy.set_color(t["cut"])
        frame = self.data_legend.get_frame()
        frame.set_facecolor(t["axes"])
        frame.set_edgecolor(t["grid"])
        for text in self.data_legend.get_texts():
            text.set_color(t["fg"])
        # The legend draws its own copies of the line samples, so recolor those too
        legend_lines = self.data_legend.get_lines()
        legend_lines[-2].set_color(t["line"])
        legend_lines[-1].set_color(t["cut"])
        for text in self.net_texts:
            text.set_color(t["fg"])
        self._build_net_legend(t)

    def _build_net_legend(self, t):
        """Rebuilt on every theme change, because a legend keeps its own copies of the symbols."""
        if self.net_legend is not None:
            self.net_legend.remove()

        def node(face):
            return Line2D([], [], marker="o", linestyle="", markersize=11,
                          markerfacecolor=face, markeredgecolor=t["node_edge"])

        def edge(color, width):
            return Line2D([], [], color=color, linewidth=width, solid_capstyle="round")

        dead_x = Line2D([], [], marker="x", linestyle="", markersize=7,
                        markeredgewidth=1.6, color=t["fg"])
        handles = [node(t["node_glow"]), node(t["node_dim"]), (node(t["node_dim"]), dead_x),
                   edge(POSITIVE_COLOR, 3), edge(NEGATIVE_COLOR, 3), edge(t["node_edge"], 0.8)]
        labels = ["Fires for most dots", "Fires for few dots", "Dead (fires for none)",
                  "Positive weight", "Negative weight", "Thin = weak weight"]

        self.net_legend = self.ax_net.legend(
            handles, labels, handler_map={tuple: HandlerTuple(ndivide=1)},
            loc="upper center", ncol=2, fontsize=8, title="Neuron glow  /  bridge weights",
            title_fontsize=8.5, frameon=True, fancybox=True, borderpad=0.7,
            columnspacing=2.2, handlelength=2.2, labelspacing=0.6)
        frame = self.net_legend.get_frame()
        frame.set_facecolor(t["axes"])
        frame.set_edgecolor(t["grid"])
        frame.set_alpha(0.95)
        for text in self.net_legend.get_texts():
            text.set_color(t["fg"])
        self.net_legend.get_title().set_color(t["fg"])

    def _style_widget(self, widget, t):
        kind = widget.winfo_class()
        if kind == "Button":
            widget.config(bg=t["button"], fg=t["fg"], activebackground=t["button_active"],
                          activeforeground=t["fg"])
        elif kind == "Menubutton":
            widget.config(bg=t["button"], fg=t["fg"], activebackground=t["button_active"],
                          activeforeground=t["fg"], highlightthickness=0)
            widget["menu"].config(bg=t["field"], fg=t["fg"], activebackground=t["button_active"],
                                  activeforeground=t["fg"])
        elif kind == "Checkbutton":
            widget.config(bg=t["bg"], fg=t["fg"], activebackground=t["bg"],
                          activeforeground=t["fg"], selectcolor=t["field"])
        elif kind == "Label":
            widget.config(bg=t["bg"], fg=t["fg"])
        elif kind == "Scale":
            widget.config(bg=t["bg"], fg=t["fg"], troughcolor=t["trough"],
                          activebackground=t["button_active"], highlightbackground=t["bg"])
        elif kind == "Spinbox":
            widget.config(bg=t["field"], fg=t["fg"], insertbackground=t["fg"],
                          buttonbackground=t["button"], highlightbackground=t["bg"])
        elif kind in ("Frame", "Canvas"):
            widget.config(bg=t["bg"], highlightbackground=t["bg"])
        for child in widget.winfo_children():
            self._style_widget(child, t)


def run():
    root = tk.Tk()
    RingApp(root)
    root.mainloop()


if __name__ == "__main__":
    run()
