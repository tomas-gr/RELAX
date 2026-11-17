"""PC-02E: prolonga trayectorias PC-02, con comprobación exacta del prefijo."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from planar_arm import PlanarArm
from planar_arm.pc01 import load_model
from planar_arm.pc02 import InferenceConfig, infer
from planar_arm.pc02d import diagnose

CASES = ((0, 0), (0, 2), (3, 3))
CHECKPOINTS = (0, 100, 300, 1000, 3000, 5000, 10000, 30000)


def check_prefix(original, new):
    """Exige igualdad bit a bit de todos los registros hasta el final previo."""
    for key in original:
        if key not in new or not np.array_equal(original[key], new[key][:len(original[key])]):
            raise ValueError(f'La trayectoria difiere de PC-02 en {key}; detener análisis')


def needs_extension(total_energy, gradient_norm, energy_tol, grad_tol):
    """Extiende si E cae >=5% entre 5000 y 10000 y el criterio PC-02 no se cumple."""
    converged = total_energy[10000] <= energy_tol and gradient_norm[10000] <= grad_tol
    return bool(not converged and total_energy[10000] < 0.95 * total_energy[5000])


def snapshot(step, history, diagnostics, settings):
    """Mediciones simultáneas del estado posterior a `step` actualizaciones."""
    terms = history['energy_terms'][step]
    grad = history['gradient_norms'][step]
    total = float(sum(terms))
    grad_norm = float(np.linalg.norm(grad))
    return {'step': int(step), 'energy': total, 'energy_terms': terms.tolist(),
            'relaxed_error': float(diagnostics['relaxed_error'][step]),
            'feedforward_error': float(diagnostics['feedforward_error'][step]),
            'physical_error': float(diagnostics['physical_error'][step]),
            'latent_output_gap': float(diagnostics['latent_output_gap'][step]),
            'model_error': float(diagnostics['model_error'][step]),
            'internal_norms': diagnostics['internal_norms'][step].tolist(),
            'gradient_norms': grad.tolist(), 'joint_gradient_norm': grad_norm,
            'angles': history['angles'][step].tolist(),
            'out_of_domain': bool(diagnostics['out_of_domain'][step]),
            'converged': bool(total <= settings.energy_tol and grad_norm <= settings.grad_tol)}


def comparative_plot(output):
    """Compara las tres trayectorias usando los registros ya guardados."""
    output = Path(output)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ti, si in CASES:
        name = f'target{ti}_start{si}'
        with np.load(output/f'{name}_history.npz') as h, np.load(output/f'{name}_diagnostic.npz') as d:
            axes[0,0].semilogy(np.maximum(h['total_energy'], 1e-16), label=name)
            axes[0,1].plot(d['physical_error'], label=name)
            axes[1,0].plot(d['latent_output_gap'], label=name)
            axes[1,1].semilogy(np.maximum(np.linalg.norm(h['gradient_norms'], axis=1), 1e-16), label=name)
    for ax, title in zip(axes.flat, ['Energía total', 'Error físico', 'Salida relajada vs eval', 'Norma conjunta de gradiente']):
        ax.set(title=title, xlabel='Iteraciones'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/'comparacion.png', dpi=140); plt.close(fig)


def run(output):
    root = Path('runs/pc02_seed42')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Usar un directorio de salida vacío')
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    checkpoint = Path('runs/pc01_seed42/model.pt')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != config['checkpoint_sha256']:
        raise ValueError('El checkpoint no coincide con PC-02')
    settings = InferenceConfig(**config['inference'])
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    model, saved = load_model(checkpoint)
    arm = PlanarArm(*config['lengths'])
    if [saved['config']['l1'], saved['config']['l2']] != config['lengths']:
        raise ValueError('Longitudes incompatibles')
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for ti, si in CASES:
        name = f'target{ti}_start{si}'
        target, initial = config['targets'][ti], config['initial_angles'][si]
        source = root/f'{name}_pc.npz'
        settings.steps = 10000
        _, history = infer(model, target, initial, settings, 'pc', arm)
        with np.load(source) as old:
            check_prefix(old, history)
        print(f'{name}: prefijo 0..3000 idéntico', flush=True)
        gn = np.linalg.norm(history['gradient_norms'], axis=1)
        extend = needs_extension(history['total_energy'], gn, settings.energy_tol, settings.grad_tol)
        if extend:
            # Se reinicia desde las condiciones originales para conservar el ciclo exacto de PCTrainer.
            settings.steps = 30000
            _, longer = infer(model, target, initial, settings, 'pc', arm)
            check_prefix(history, longer)
            history = longer
            print(f'{name}: ampliado a 30000; prefijo 0..10000 idéntico', flush=True)
        d = diagnose(model, arm, target, history, settings.lr_angles, settings.lr_states)
        np.savez_compressed(output/f'{name}_history.npz', **history)
        np.savez_compressed(output/f'{name}_diagnostic.npz', **{k:v for k,v in d.items() if isinstance(v, np.ndarray)})
        points = [snapshot(i, history, d, settings) for i in CHECKPOINTS if i <= settings.steps]
        errors = d['physical_error']
        minimum_step = int(np.argmin(errors))
        energy = history['total_energy']
        energy_increases = int(np.count_nonzero(np.diff(energy) > 1e-7))
        angles = history['angles']
        item = {'case': name, 'target': target, 'initial_angles': initial,
                'steps': settings.steps, 'extended': extend, 'prefix_exact': True,
                'long_prefix_exact': True if extend else None,
                'weights_unchanged': hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest,
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'checkpoints': points, 'physical_minimum': {'step': minimum_step, 'error': float(errors[minimum_step])},
                'physical_last': float(errors[-1]), 'energy_increases_gt_1e-7': energy_increases,
                'outside_count': int(np.count_nonzero(d['out_of_domain'])),
                'outside_last': bool(d['out_of_domain'][-1]),
                'angular_movement_5000_to_10000': float(np.linalg.norm(angles[10000]-angles[5000])),
                'energy_drop_fraction_5000_to_10000': float((energy[5000]-energy[10000])/energy[5000])}
        results.append(item)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for j, label in enumerate(['E1', 'E2', 'salida']):
            axes[0,0].semilogy(np.maximum(history['energy_terms'][:,j],1e-16), label=label)
        axes[0,0].semilogy(np.maximum(energy,1e-16), label='total')
        for key, label in [('relaxed_error','salida relajada'), ('feedforward_error','red eval'),
                           ('physical_error','físico'), ('latent_output_gap','relajada vs eval'), ('model_error','eval vs físico')]:
            axes[0,1].plot(d[key], label=label)
        for j, label in enumerate(['ángulos', 'estado 1', 'estado 2']):
            axes[1,0].semilogy(np.maximum(history['gradient_norms'][:,j],1e-16), label=label)
        for j in range(2):
            axes[1,1].plot(angles[:,j], label=f'θ{j+1}')
        axes[1,1].axhline(np.pi, color='gray', ls='--'); axes[1,1].axhline(-np.pi, color='gray', ls='--')
        for ax, title in zip(axes.flat, ['Componentes de energía','Errores y discrepancias','Normas de gradiente','Ángulos y dominio entrenado']):
            ax.set(title=title, xlabel='Iteraciones'); ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.suptitle(name); fig.tight_layout(); fig.savefig(output/f'{name}.png', dpi=140); plt.close(fig)
        (output/'results.json').write_text(json.dumps({'config_original': config, 'checkpoint_sha256': digest,
             'extension_rule': 'E(10000) < 0.95 E(5000) y no convergido en 10000', 'cases': results}, indent=2), encoding='utf-8')
        print(f'{name}: E={energy[-1]:.6g}, error físico final={errors[-1]:.6g}, mínimo={errors[minimum_step]:.6g} en {minimum_step}', flush=True)
    comparative_plot(output)


def main():
    parser = argparse.ArgumentParser(description='Convergencia temporal PC-02E')
    parser.add_argument('--output', default='runs/pc02e')
    run(parser.parse_args().output)


if __name__ == '__main__':
    main()
