
🔗 Qiskit — Distributed CX/CZ Gates + T1–Tφ Heatmaps

This project simulates monolithic (single-chip) vs distributed quantum processors connected with a remote entangling gate.
We test CNOT (CX) and Controlled-Z (CZ) versions, include realistic noise, and measure how fidelity depends on link errors and qubit lifetimes.

⸻

🎯 Goal
	•	Create a Bell state (|Φ⁺⟩) using CX and CZ gates.
	•	Compare single-chip (monolithic) vs distributed (linked by a noisy channel).
	•	Study how link errors, T₁ (relaxation), and Tφ (dephasing) affect fidelity.

⸻

🧪 Circuits
	•	Monolithic:
	•	H(0) + CX(0→1)
	•	H(0) + H(1) + CZ(0,1) + H(1)
	•	Distributed:
	•	Same circuits, but q₀ and q₁ treated as if on different processors linked by a remote gate.

⸻

⚡ Noise Models
	•	Monolithic:
	•	Small 1-qubit + 2-qubit depolarizing errors.
	•	Distributed:
	•	Same local errors.
	•	Extra link error (p_link).
	•	Decoherence during link (T₁, Tφ, link duration).

⸻

📊 Results
	1.	Monolithic CX & CZ: Fidelity stable ~0.99.
	2.	Distributed CX & CZ: Fidelity decreases as link error grows (0–5%).
	3.	Heatmaps: Show how long T₁ and Tφ must be for F ≥ 0.90.

Figures:
	•	figures/link_sweep.png → CX vs CZ fidelity vs link error
	•	figures/heatmap_cx.png → T₁–Tφ heatmap (CX)
	•	figures/heatmap_cz.png → T₁–Tφ heatmap (CZ)

⸻

📝 Error Analysis (CZ=0.25 bug)

Before fix: H(0); CZ(0,1) only gave fidelity = 0.25.
Reason: |+0⟩ has no |11⟩ component, so CZ cannot entangle.
Fix: Add H–CZ–H on the target qubit → equivalent to CNOT.
Now CZ produces proper Bell states.

⸻

🌍 Impact
	•	Shows why distributed quantum computers depend critically on link quality.
	•	Demonstrates that better coherence (T₁, Tφ) enables distributed chips to approach monolithic performance.
	•	First step toward quantum internet simulations.
