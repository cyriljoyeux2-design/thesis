# Thesis — Status

*Single source of truth for "where I am". Last updated: 2026-11-27.*
*Read by Claude Code (imported in `CLAUDE.md`) and by the claude.ai project "Thèse" (synced from GitHub).*
*Update at the end of every working session, then commit + push.*

## Overall goal
Computational environment for photoelectrochromism of **pentacene adsorbed on ITO** (smart windows), combining **DFT + tight-binding (TB) + GW-BSE** for band structure and optical/electronic properties.

## Roadmap
1. **TB model of In2O3** ← *current stage*
2. Fit TB parameters on the DFT/GW reference
3. ITO (Sn doping)
4. Pentacene/ITO interface
5. GW-BSE for optical properties

## Current state (In2O3, TBPLaS)
- Bixbyite structure (Ia-3) generated with ASE from Odaka et al., *Jpn. J. Appl. Phys.* (2000).
- TBPLaS `PrimitiveCell` with s,p orbitals and hoppings in the primitive cell.
- `fit.py` (fit) separate from `calc_bands.py` (bands for any parameter set, compared with DFT); TB parameters in JSON (`in2o3/params/`); QE XML reader `qe_xml.py`.
- A first fit works on a simple version. Fit output reports RMS per region (deep valence / gap edge / upper conduction).
- Tag **`v0.1-sp-model`** = first working s,p model.

## Next steps
1. **Slater–Koster exercise:** `sk_hop(d_vec, orb_i, orb_j, params)` for the s,p block, with parity for $E_{x,s}$ and a rotation test $E(R\mathbf d) = R\,E(\mathbf d)\,R^T$.
2. **Neighbor shells:** histogram of In–O / O–O / In–In distances (ASE `neighbor_list`), choose one cutoff **per pair type, placed in a gap**; use coordination numbers (In1: 6 O, In2: 6 O, O: 4 In) only as a sanity check, not to select neighbors. Keep cutoffs short (≲ 3.5–4 Å).
3. **Distance scaling:** implement $V(d) = V(d_0)\,(d_0/d)^n$ (Harrison: $n = 2$ sp, $3.5$ pd, $5$ dd; $n = 0$ = scalar model) and compare the RMS per region.
4. Then add **In p/d orbitals** (branch `feature/in-pd-orbitals`); whether to include In 4d semicore is still open.

## Open questions
- Orbital basis: In 5s/5p, O 2s/2p, In 4d?
- How to handle the GW correction in the fit target.
- Fit $n$ per bond type once the fixed-$n$ fit works?

## Git workflow
- `main` = working code only. Feature branches: `feature/in-pd-orbitals`, `feature/ito-bands` (ITO merges `main` once p/d is in).
- ITO code in `ito/`, shared code in `in2o3/` or `common/`.
- Merge through GitHub pull requests (review the diff, Claude Code review), tag milestones.
- Planned: store `git rev-parse --short HEAD` in every `fit_*.json` (`meta.git_commit`).
- Paths stored in JSON must be `str()`, not `Path`.

## Data privacy (never break)
- Code: `~/work/thesis-git/` (git, GitHub, Claude Code). Data: `~/work/thesis-data/` (never in git, never read by Claude Code).
- Scripts use `Path.home() / "work" / "thesis-data" / ...`.
- The postdoc's DFT/GW data (QE + Yambo) are **unpublished**: no data, energies or plots in this file, in git, or in prompts.
- History cleaned with `git filter-repo` on 2026-11-17.

## Environment
- WSL2 Ubuntu 24.04, Vim + tmux (status bar: SSH host · branch · host), conda env `tbplas-env` (TBPLaS 1.6.0, ASE 3.29.0; `scipy<1.14`, `numpy<2.3`).
- CÉCI clusters over SSH (ssh-agent/keychain to set up in WSL). GitHub over SSH (`~/.ssh/id_ed25519_github`).
- Claude Code in `thesis-git` only, reviewer/tutor mode (`CLAUDE.md` + `.claude/settings.json` deny rules).
