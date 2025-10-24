# === Distributed CX/CZ + link-sweep + T1–Tphi heatmaps (V2) ===
import os, numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
)
import warnings
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings("ignore", message="Specific error for instruction")
import matplotlib.pyplot as plt
import logging

# Initialize root handlers if not already configured
logging.basicConfig()

# Quiet Qiskit + Aer logs
for name in ("qiskit", "qiskit_aer", "qiskit.providers.aer"):
    logging.getLogger(name).setLevel(logging.ERROR)
np.random.seed(42)
os.makedirs("figures", exist_ok=True)

# -------- Target Bell state |Φ+> ----------
def bell_phi_plus_dm():
    sv = Statevector.from_label('00')
    qc = QuantumCircuit(2); qc.h(0); qc.cx(0,1)
    return DensityMatrix(sv.evolve(qc))
BELL_DM = bell_phi_plus_dm()

# -------- Circuits (CX or CZ) ------------
def build_bell_circuit(gate='cx'):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)

    # 1. önce 0. kübite H uygula (Bell hazırlığı)
    qc.h(0)

    if gate == 'cx':
        # 2. CNOT: normal uzak kapı
        qc.cx(0, 1)

    elif gate == 'cz':
        # ÖNEMLİ DÜZELTME:
        # CZ tek başına Bell yapmıyor. CNOT'a eşdeğer olsun diye
        # hedef kübite (1. kübit) önce ve sonra H koyuyoruz:
        qc.h(1)      # önce H
        qc.cz(0, 1)  # CZ
        qc.h(1)      # sonra H

    else:
        raise ValueError("gate must be 'cx' or 'cz'")

    # yoğunluk matrisi kaydı (fidelite hesabı için)
    qc.save_density_matrix()
    return qc


# -------- Noise models --------------------
def make_monolithic_noise_model(p1=1e-3, p2=1e-2):
    nm = NoiseModel()
    oneq = depolarizing_error(p1, 1)
    twoq = depolarizing_error(p2, 2)
    for g in ['id','x','y','z','h','s','sdg','rx','ry','rz']:
        nm.add_all_qubit_quantum_error(oneq, g)
    nm.add_all_qubit_quantum_error(twoq, 'cx')
    nm.add_all_qubit_quantum_error(twoq, 'cz')
    return nm

def make_distributed_noise_model(*,
    p1=1e-3, p2_local=1e-2, p2_link=3e-2,
    T1_link=50e-6, Tphi_link=30e-6, link_duration=300e-9,
    link_gate='cx'
):
    nm = NoiseModel()
    oneq_local = depolarizing_error(p1, 1)
    twoq_local = depolarizing_error(p2_local, 2)
    for g in ['id','x','y','z','h','s','sdg','rx','ry','rz']:
        nm.add_all_qubit_quantum_error(oneq_local, g)
    nm.add_all_qubit_quantum_error(twoq_local, 'cx')
    nm.add_all_qubit_quantum_error(twoq_local, 'cz')

    if link_gate not in ('cx','cz'):
        raise ValueError("link_gate must be 'cx' or 'cz'")

    # Link hatası: depolarizasyon + (opsiyonel) T1/Tphi
    link_err = depolarizing_error(p2_link, 2)
    if T1_link is not None:
        p_amp = 1.0 - np.exp(-link_duration / T1_link)
        amp = amplitude_damping_error(p_amp)
        link_err = link_err.compose(amp.tensor(amp))
    if Tphi_link is not None:
        p_phase = 1.0 - np.exp(-link_duration / Tphi_link)
        ph = phase_damping_error(p_phase)
        link_err = link_err.compose(ph.tensor(ph))

    # Uzak kapıyı (0,1) çifti üzerinde “özel” hata olarak ver
    nm.add_quantum_error(link_err, link_gate, [0,1])
    return nm

# -------- Fidelity helper -----------------
def circuit_state_fidelity(qc, noise_model=None):
    sim = AerSimulator(method='density_matrix', noise_model=noise_model)
    res = sim.run(qc).result()
    dm = res.data(0)['density_matrix']
    rho = DensityMatrix(dm)
    return state_fidelity(rho, BELL_DM)

# ===================== RUN ======================

# 1) Monolitik vs dağıtık (CX ve CZ)
mono_nm = make_monolithic_noise_model(p1=1e-3, p2=1e-2)
dist_cx_nm = make_distributed_noise_model(link_gate='cx')
dist_cz_nm = make_distributed_noise_model(link_gate='cz')

qc_mono_cx = build_bell_circuit('cx')
qc_mono_cz = build_bell_circuit('cz')
qc_dist_cx = build_bell_circuit('cx')
qc_dist_cz = build_bell_circuit('cz')

F_mono_cx = circuit_state_fidelity(qc_mono_cx, mono_nm)
F_mono_cz = circuit_state_fidelity(qc_mono_cz, mono_nm)
F_dist_cx = circuit_state_fidelity(qc_dist_cx, dist_cx_nm)
F_dist_cz = circuit_state_fidelity(qc_dist_cz, dist_cz_nm)

print(f"Monolithic CX fidelity: {F_mono_cx:.6f}")
print(f"Monolithic CZ fidelity: {F_mono_cz:.6f}")
print(f"Distributed CX fidelity: {F_dist_cx:.6f}")
print(f"Distributed CZ fidelity: {F_dist_cz:.6f}")

# 2) Link error sweep (0–5%) for CX & CZ
p_link_vals = np.linspace(0.0, 0.05, 21)
fids_cx, fids_cz = [], []
for p_link in p_link_vals:
    nm_cx = make_distributed_noise_model(p2_link=p_link, link_gate='cx')
    nm_cz = make_distributed_noise_model(p2_link=p_link, link_gate='cz')
    fids_cx.append(circuit_state_fidelity(qc_dist_cx, nm_cx))
    fids_cz.append(circuit_state_fidelity(qc_dist_cz, nm_cz))

plt.figure()
plt.plot(100*p_link_vals, fids_cx, marker='o', label='Distributed CX')
plt.plot(100*p_link_vals, fids_cz, marker='s', label='Distributed CZ')
plt.axhline(F_mono_cx, linestyle='--', label=f'Monolithic CX ({F_mono_cx:.4f})')
plt.axhline(F_mono_cz, linestyle=':',  label=f'Monolithic CZ ({F_mono_cz:.4f})')
plt.xlabel('Link two-qubit depolarizing error p_link (%)')
plt.ylabel('Bell fidelity')
plt.title('Effect of remote link noise (CX vs CZ)')
plt.legend(); plt.grid(True)
plt.savefig("figures/link_sweep.png", dpi=300)
plt.close()

# 3) T1–Tphi heatmaps (p_link sabit: 2%, t_link=300 ns)
#T1_grid   = np.logspace(np.log10(5e-6), np.log10(5e-4), 40)
#Tphi_grid = np.logspace(np.log10(5e-6), np.log10(5e-4), 40)
T1_grid   = np.logspace(np.log10(5e-6), np.log10(5e-4), 20)
Tphi_grid = np.logspace(np.log10(5e-6), np.log10(5e-4), 20)

def heatmap_for_gate(link_gate='cx', p2_link=0.02, link_duration=300e-9):
    heat = np.zeros((len(T1_grid), len(Tphi_grid)))
    qc = build_bell_circuit('cx' if link_gate=='cx' else 'cz')
    for i, T1 in enumerate(T1_grid):
        for j, Tphi in enumerate(Tphi_grid):
            nm = make_distributed_noise_model(
                p2_link=p2_link, T1_link=T1, Tphi_link=Tphi,
                link_duration=link_duration, link_gate=link_gate)
            heat[i, j] = circuit_state_fidelity(qc, nm)
    return heat

heat_cx = heatmap_for_gate('cx', p2_link=0.02, link_duration=300e-9)
heat_cz = heatmap_for_gate('cz', p2_link=0.02, link_duration=300e-9)

def save_heat(heat, title, fname):
    plt.figure()
    im = plt.imshow(heat, origin='lower', aspect='auto',
                    extent=[Tphi_grid[0]*1e6, Tphi_grid[-1]*1e6,
                            T1_grid[0]*1e6,   T1_grid[-1]*1e6])
    plt.colorbar(im, label='Bell fidelity')
    plt.xlabel('Tφ (µs)'); plt.ylabel('T1 (µs)')
    plt.title(title); plt.savefig(fname, dpi=300); plt.close()

save_heat(heat_cx, 'T1–Tφ heatmap (Distributed CX, p_link=2%, t_link=300 ns)',
          "figures/heatmap_cx.png")
save_heat(heat_cz, 'T1–Tφ heatmap (Distributed CZ, p_link=2%, t_link=300 ns)',
          "figures/heatmap_cz.png")

print("Saved figures: figures/link_sweep.png, figures/heatmap_cx.png, figures/heatmap_cz.png")


