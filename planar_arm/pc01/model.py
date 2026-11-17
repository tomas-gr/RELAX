"""Modelo y adaptador mínimo al PCTrainer existente; no implementa dinámica PC."""

import torch
from torch import nn
import predictive_coding as pc


class ForwardPC(nn.Module):
    """Mapa de ángulos (B, 2), en radianes, a coordenadas (B, 2)."""

    def __init__(self, hidden=32):
        super().__init__()
        if not isinstance(hidden, int) or hidden < 1:
            raise ValueError("hidden debe ser entero positivo")
        self.hidden = hidden
        self.layers = nn.Sequential(
            nn.Linear(2, hidden), pc.PCLayer(), nn.Tanh(),
            nn.Linear(hidden, hidden), pc.PCLayer(), nn.Tanh(),
            nn.Linear(hidden, 2))

    def forward(self, angles):
        """En train usa estados PC; en eval realiza una pasada directa."""
        return self.layers(angles)

    def weights(self):
        """Copia solo pesos y sesgos; excluye estados por lote de PCLayer."""
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if not k.endswith("._x")}


def output_energy(output, target):
    """Error supervisado sumado: 1/2 ||output-target||², sin cinemática analítica."""
    return 0.5 * (output - target).square().sum()


def make_trainer(model, steps=30, lr_x=0.1, lr_p=0.001):
    """Primeros steps-1 pasos: solo estados; último: solo parámetros.

    PCLayer inicializa sus estados a predicciones desacopladas en cada lote.
    PCTrainer excluye esos estados del optimizador de parámetros. La pérdida y
    las energías ocultas usan sumas coherentes, sin promediar por lote.
    """
    if steps < 3 or lr_x <= 0 or lr_p <= 0:
        raise ValueError("Se requieren steps >= 3 y tasas positivas")
    return pc.PCTrainer(model, T=steps,
                        optimizer_x_fn=torch.optim.SGD, optimizer_x_kwargs={"lr": lr_x},
                        update_x_at=list(range(steps - 1)),
                        optimizer_p_fn=torch.optim.Adam, optimizer_p_kwargs={"lr": lr_p},
                        update_p_at="last", update_p_at_early_stop=False,
                        plot_progress_at=[])


def load_model(path):
    """Carga pesos en CPU y devuelve (modelo en eval, contenido del checkpoint)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ForwardPC(checkpoint["architecture"]["hidden"])
    model.load_state_dict(checkpoint["weights"], strict=True)
    model.eval()
    return model, checkpoint
