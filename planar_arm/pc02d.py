"""Diagnóstico de trayectorias PC-02 guardadas; no modifica experimentos previos."""
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from planar_arm import PlanarArm
from planar_arm.pc01 import load_model


def diagnose(model, arm, target, history, lr_angles, lr_states):
    """Reconstruye energías, gradientes analíticos y pasos desde estados guardados."""
    target = np.asarray(target)
    q = history['angles'].astype(np.float64)
    physical = history['exact_tip']
    with torch.no_grad():
        model.eval()
        ff = model(torch.tensor(q, dtype=torch.float32)).numpy().astype(float)
    result = {'relaxed_error': np.linalg.norm(history['relaxed_output']-target, axis=1),
              'feedforward_error': np.linalg.norm(ff-target, axis=1),
              'physical_error': np.linalg.norm(physical-target, axis=1),
              'model_error': np.linalg.norm(ff-physical, axis=1),
              'latent_output_gap': np.linalg.norm(history['relaxed_output']-ff, axis=1),
              'out_of_domain': np.any((q < -np.pi) | (q >= np.pi), axis=1)}
    if history['states'].shape[1] == 0:
        result['step_norms'] = np.linalg.norm(np.diff(q, axis=0), axis=1)[:, None]
        result['gradient_norms'] = history['gradient_norms'].copy()
        result['energy_terms'] = history['energy_terms'].copy()
        return result
    z = history['states'].astype(np.float64)
    w1, w2, w3 = [model.layers[i].weight.detach().numpy().astype(float) for i in (0, 3, 6)]
    b1, b2, b3 = [model.layers[i].bias.detach().numpy().astype(float) for i in (0, 3, 6)]
    a1, a2 = np.tanh(z[:, 0]), np.tanh(z[:, 1])
    r1 = z[:, 0] - (q @ w1.T + b1)
    r2 = z[:, 1] - (a1 @ w2.T + b2)
    rout = a2 @ w3.T + b3 - target
    gradients = [(-r1) @ w1,
                 r1 - (r2 @ w2) * (1-a1*a1),
                 r2 + (rout @ w3) * (1-a2*a2)]
    norms = np.column_stack([np.linalg.norm(g, axis=1) for g in gradients])
    deltas = [np.diff(q, axis=0), np.diff(z[:, 0], axis=0), np.diff(z[:, 1], axis=0)]
    step_norms = np.column_stack([np.linalg.norm(d, axis=1) for d in deltas])
    step_residual = np.array([np.max(np.abs(d + lr*g[:-1])) for d, lr, g in zip(deltas, [lr_angles, lr_states, lr_states], gradients)])
    terms = np.column_stack([0.5*np.sum(r1*r1, axis=1), 0.5*np.sum(r2*r2, axis=1), 0.5*np.sum(rout*rout, axis=1)])
    result.update(internal_norms=np.column_stack([np.linalg.norm(r1, axis=1), np.linalg.norm(r2, axis=1)]),
                  energy_terms=terms, gradient_norms=norms, step_norms=step_norms,
                  gradient_vectors=np.column_stack(gradients),
                  saturation=np.column_stack([np.mean(np.abs(a1)>0.99, axis=1), np.mean(np.abs(a2)>0.99, axis=1)]),
                  max_residual_mismatch=float(np.max(np.abs(np.stack([r1, r2], axis=1)-history['internal_residuals']))),
                  max_energy_mismatch=float(np.max(np.abs(terms-history['energy_terms']))),
                  max_gradient_norm_mismatch=float(np.max(np.abs(norms-history['gradient_norms']))),
                  max_step_residual=step_residual)
    return result


def main():
    root = Path('runs/pc02_seed42')
    out = Path('runs/pc02d')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Usar un directorio runs/pc02d vacío')
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    checkpoint = Path('runs/pc01_seed42/model.pt')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != config['checkpoint_sha256']:
        raise ValueError('El checkpoint no coincide con PC-02')
    model, _ = load_model(checkpoint)
    torch.set_num_threads(1)
    arm = PlanarArm(*config['lengths'])
    out.mkdir(parents=True, exist_ok=True)
    summaries, refs = [], []
    for ti, target in enumerate(config['targets']):
        for q in arm.inverse(target).solutions:
            with torch.no_grad():
                y = model(torch.tensor([q], dtype=torch.float32)).numpy()[0]
            refs.append({'target': target, 'angles': q, 'prediction': y.tolist(),
                         'energy_consistent': float(0.5*np.sum((y-target)**2))})
        for si in range(config['starts']):
            for method in ('pc', 'input'):
                with np.load(root/f'target{ti}_start{si}_{method}.npz') as h:
                    d = diagnose(model, arm, target, h, config['inference']['lr_angles'], config['inference']['lr_states'])
                    serial = {k: v for k, v in d.items() if np.isscalar(v) and not isinstance(v, np.ndarray)}
                    serial.update(target=target, start=si, method=method,
                                  final={k: float(d[k][-1]) for k in ['relaxed_error','feedforward_error','physical_error','model_error','latent_output_gap']},
                                  outside_steps=int(np.sum(d['out_of_domain'])))
                    if method == 'pc':
                        for key in ['internal_norms','energy_terms','gradient_norms','saturation']:
                            serial['final'][key] = d[key][-1].tolist()
                        serial['max_step_residual'] = d['max_step_residual'].tolist()
                    summaries.append(serial)
                    np.savez_compressed(out/f'target{ti}_start{si}_{method}.npz', **{k:v for k,v in d.items() if isinstance(v, np.ndarray)})
                    if method == 'pc' and (ti, si) in [(0, 0), (3, 3), (0, 2)]:
                        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                        for key in ['relaxed_error','feedforward_error','physical_error','model_error','latent_output_gap']:
                            axes[0,0].plot(d[key], label=key)
                        for j in range(3):
                            axes[0,1].semilogy(np.maximum(d['energy_terms'][:,j], 1e-16), label=['capa 1','capa 2','salida'][j])
                            axes[1,0].semilogy(np.maximum(d['gradient_norms'][:,j],1e-16), label=['ángulos','estado 1','estado 2'][j])
                            axes[1,1].semilogy(np.maximum(d['step_norms'][:,j],1e-16), label=['ángulos','estado 1','estado 2'][j])
                        for ax, title in zip(axes.flat, ['Errores de distancia','Energía por término','Norma de gradiente','Norma de paso efectivo']):
                            ax.set(title=title, xlabel='Iteración'); ax.legend(fontsize=8); ax.grid(alpha=.2)
                        fig.tight_layout(); fig.savefig(out/f'target{ti}_start{si}.png', dpi=140); plt.close(fig)
    (out/'diagnostics.json').write_text(json.dumps({'trials': summaries, 'analytic_references': refs}, indent=2), encoding='utf-8')
    print(json.dumps({'trials': summaries, 'analytic_references': refs}, indent=2))


if __name__ == '__main__':
    main()
