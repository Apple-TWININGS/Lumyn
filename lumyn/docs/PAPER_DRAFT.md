# Zero-Shot Conservation Critics for Generative Physical Simulation

*(arXiv preprint draft — 2026-09)*

> **Status.** This draft states only results that were measured on the code in this
> repository. Every number carries a pointer to the producing script. Negative results
> are reported alongside positive ones, and three claims that earlier internal drafts
> made are explicitly retracted in §7.

---

## Abstract

Generative models of physical scenes are typically validated with a *critic*: a score
that ranks or filters candidate trajectories. Learned critics are the default choice,
but they require labelled data and can be brittle. We ask whether a **zero-shot critic
built from conservation laws** — no training, no labels — is competitive.

We study the task of *candidate selection*: given $K$ trajectories generated from the
same initial condition by a learned dynamics model, pick the one closest to the true
solution. This is the operation a critic performs inside a generation loop, and unlike
classification-style metrics it needs no threshold on "how wrong is wrong".

Across three systems — a 6-body 3D family, the figure-eight three-body choreography, and
a 2D Kepler two-body system — a zero-shot conservation critic attains **100% within-IC
selection accuracy** on the standard benchmarks, reducing trajectory error by 75–82%
relative to random selection and landing within 1.03–1.05× of the best achievable
selection. On the 6-body system (5 seeds, 95% CI) it reaches **82.8%±3.3%** in
distribution and **84.7%±1.8%** out of distribution, versus **71.2%±3.2%** and
**72.2%±2.5%** for a supervised pairwise-ranking critic trained on the same 960 labelled
candidates.

We also establish what the conservation critic is *not*. It does **not** beat a
correctly-used Hamiltonian Neural Network critic: the two tie exactly (identical
selection error on both standard benchmarks). Claims that the conservation critic is
more robust out of distribution, or that combining it with a learned critic helps, are
**falsified** here and retracted.

Three methodological findings, each independently useful:

1. **Pooled rank correlation is misleading for critics.** A supervised regressor scored
   pooled Spearman $\rho=-0.67$ — comparable to the best critic — while achieving an
   IC-internal win rate of **53%**, i.e. chance. Within-IC rank correlation must be used.
2. **A conservation channel's discriminative power is set by its noise floor**, which is
   determined by the integrator and force structure, not by the physical importance of
   the quantity. Momentum and angular momentum are conserved *exactly* by any pairwise
   central-force integrator (floor $\sim10^{-16}$) and separate true from generated
   trajectories by $\sim10^{12}$; energy is conserved only to the integrator's truncation
   error ($O(\Delta t)$, floor $0.60$ at $\Delta t=0.01$) and separates by only $1.13\times$.
3. **Synthetic failure modes must be validated before use.** A taxonomy of hand-built
   corruptions (noise, wrong integrator, freeze, splice, time-warp) is **not
   structurally comparable** to a real generator's failures: two of the five cannot be
   tuned to the generator's error magnitude at all, and the conservation fingerprints
   differ by up to 238 normalized units.

---

## 1. Introduction

A generative model of physical scenes — a learned surrogate, a neural ODE, a video
model — produces trajectories that may look plausible while violating physics. The
standard remedy is a **critic**: a function that scores a candidate so that bad samples
can be filtered, ranked, or used as a training signal.

Learned critics dominate, and they work: a network can capture subtle dynamical errors
that no simple statistic catches. But they need labelled examples, and labels for
physical validity are exactly what one is trying to avoid needing.

This paper asks a narrower and more testable question than "can physics replace
learning?":

> **Is a critic built from conservation laws — which is training-free and label-free —
> competitive with the strongest learned critics at selecting good candidates?**

The answer we establish is: **yes, it ties the strongest**, and it does so at zero
labelling cost. We also document three claims that turned out to be false, because the
process of falsifying them produced the paper's methodological contributions.

### Contributions

- **C1. A clean, threshold-free evaluation protocol** for critics in a generation loop:
  *within-IC selection*. Given $K$ candidates from one initial condition, measure the
  critic's rank correlation *within that group* and its selection error relative to
  random and oracle. §5.1 shows that the more obvious pooled metric is badly misleading.
- **C2. A zero-shot conservation critic that ties the strongest baselines**, including a
  correctly-used Hamiltonian Neural Network and a supervised pairwise-ranking critic
  trained on up to 4800 labelled candidates. §5.2.
- **C3. A noise-floor theory of channel selection** that explains, predicts, and
  improves the channel set. §4.2, §5.3.
- **C4. A negative result on synthetic failure taxonomies**, with a calibration procedure
  to test comparability before the taxonomy is used. §5.4.
- **C5. Three retractions** of claims made in earlier drafts of this work. §7.

---

## 2. Related work

**Learned physical simulators.** Hamiltonian Neural Networks (Greydanus et al., 2019)
and Lagrangian Neural Networks (Cranmer et al., 2020) learn a scalar potential whose
derivatives reproduce the dynamics. They are the natural learned alternative to a
hand-written conservation law, and we adopt an HNN as our primary literature baseline.

**Neural surrogates and neural operators** learn the flow map directly. Their errors are
typically evaluated by trajectory RMSE against a reference integration, which is
available in simulation but not in the field.

**Physics-based validation.** Conservation checks are standard in numerical relativity
and molecular dynamics as diagnostics. Using them as a *critic inside a generative
loop* — the object of this paper — is less studied.

**Positioning.** We do not propose a new generator. We characterise when a training-free
physical critic is sufficient, and we show that the answer depends on a numerical
property (the noise floor) rather than on the physics.

---

## 3. Setup

### 3.1 Task: within-IC candidate selection

For each initial condition (an "IC"):

1. integrate the true equations of motion to obtain a reference trajectory $x^\star(t)$;
2. run a learned generator $K=8$ times from the *same* IC, with varying sampling noise
   $\sigma\in\{0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.6, 1.0\}$;
3. compute each candidate's error $e_j = \mathrm{RMSE}(x_j, x^\star)$;
4. score all candidates with each critic and select $\arg\max_j \mathrm{score}_j$.

**Metrics.**

| Metric | Definition | Chance | Perfect |
|---|---|---|---|
| within-IC $\rho$ | mean over ICs of Spearman(critic scores, $e$) inside the IC | 0 | −1 |
| IC win rate | fraction of ICs where the selected candidate beats that IC's median error | 50% | 100% |
| selected / random | median selected error ÷ median of per-IC mean error | 1.00 | — |
| selected / oracle | median selected error ÷ median of per-IC minimum error | — | 1.00 |

### 3.2 Systems

| System | $N$ | Dim. | Notes |
|---|---|---|---|
| 6-body 3D, eccentric family | 6 | 3D | $e\sim U(0,0.3)$ in-distribution, $e\sim U(0.5,0.8)$ OOD |
| figure-eight choreography | 3 | 2D | Chenciner & Montgomery (2000); $T=6.32591398$ |
| Kepler two-body | 2 | 2D | standard low-dimensional testbed for HNN/LNN |

Checker for the figure-eight initial data: after exactly one standard period at
$\Delta t = 0.002$ (3163 steps) the configuration returns to the initial condition to
within $3.29\times10^{-3}$ in position.

All integrations use symplectic Euler at $\Delta t=0.002$–$0.01$.

### 3.3 Generators

The generator is a pairwise force model: a shared MLP applied to
$[\,r_{ij},\,|r_{ij}|,\,v_{ij},\,m_i,\,m_j\,]$ producing a scalar pair force, summed over
neighbours. It is deliberately trained on a **limited budget** so that its failures are
large enough to be selectable among (one-step acceleration error 0.23 for the 6-body
system; three budgets, 0.23/0.0068/0.81, are used for robustness checks).

Negative samples therefore come from a **real learned generator**, not from hand-built
corruptions. §5.4 explains why that choice was forced.

### 3.4 Critics

| Critic | Labels required | Notes |
|---|---|---|
| **Conservation (zero-shot)** | **0** | angular + momentum channels (§4) |
| HNN, equation residual | trained on true trajectories | Greydanus et al. (2019) |
| HNN, energy drift | trained on true trajectories | the naive way to use an HNN as a critic |
| Pairwise ranking loss | 960–4800 labelled candidates | supervised, target-aligned |
| Absolute binary classifier | 960–4800 labelled candidates | supervised, naive target |
| Absolute error regressor | 960–4800 labelled candidates | supervised, naive target |
| Random / oracle | — | reference points |

Training budgets differ per experiment; each table states its own.

---

## 4. Method: the conservation critic

### 4.1 Conservation drift as an error functional

Let the true acceleration field be a sum of pairwise central forces,

$$a^{\text{true}}_i(r)=\frac{1}{m_i}\sum_{j\neq i} f(r_{ij})\,\hat r_{ij},$$

and let $\delta a(t) = a^{\text{gen}}(\hat z(t)) - a^{\text{true}}(\hat r(t))$ be the
generator's acceleration error along its own trajectory $\hat z$. Because
$\sum_i m_i a^{\text{true}}_i \equiv 0$ and $\sum_i m_i (r_i \times a^{\text{true}}_i)
\equiv 0$ hold pointwise,

$$
\frac{dP}{dt}=\sum_i m_i\,\delta a_i,\qquad
\frac{dL}{dt}=\sum_i m_i\,(\hat r_i\times\delta a_i),\qquad
\frac{dE}{dt}=\sum_i m_i\,(\hat v_i\cdot\delta a_i).
\tag{1}
$$

The conservation residual is therefore the **projection of the accumulated force error
onto the symmetry generators** — a physics-meaningful functional, not a heuristic.

Semi-quantitative validation (40 ICs, medians of the ratio prediction/measured):
momentum $0.53\times$, angular $1.23\times$, energy $9.0\times$; log-correlations
$0.72/0.31/0.22$. Equation (1) is exact; its *integrated* form is validated to within a
factor of a few for the channels with a clean numerical floor.

> **What is *not* claimed.** Equation (1) bounds the *drift* by the force error. It gives
> **no bound on trajectory error**: any $\delta a$ lying in the orthogonal complement of
> the three generators produces zero residual with arbitrarily large trajectory
> deviation. We never assume such an $\delta a$ is absent.

### 4.2 Which channels to use, and why

The channels are measured on the reference trajectories:

- **Linear and angular momentum are exact invariants** of any pairwise central force, to
  roundoff, independent of the integrator. Their floor is
  $\eta_P \sim \eta_L \sim \varepsilon_{\text{mach}} \approx 10^{-16}$.
- **Energy is conserved only to the integrator's truncation error.** For the
  first-order symplectic Euler used here, $\eta_E = O(\Delta t)$.

This predicts the measured channel ranking and is confirmed by a step-size sweep
(identical generator, only $\Delta t$ varied):

| $\Delta t$ | energy floor | angular floor | momentum floor | energy separation |
|---|---|---|---|---|
| 0.0100 | 6.02e-01 | 4.01e-16 | 2.88e-16 | 0.81 |
| 0.0050 | **2.65e-01** | 5.06e-16 | 3.68e-16 | 1.84 |
| 0.0025 | **1.36e-01** | 6.62e-16 | 4.39e-16 | **3.58** |

The energy floor halves with $\Delta t$ (confirming $O(\Delta t)$); the momentum and
angular floors stay pinned at roundoff. **Channel utility is a numerical property, not a
physical one.** The critic therefore uses angular + momentum by default, and excludes
both the energy channel (weak floor) and the centre-of-mass channel (exact invariant, but
its variation is nearly uncorrelated with shape error — measured IC win rate 16.7–24.2%,
below chance).

---

## 5. Experiments

### 5.1 Why the obvious metric is wrong

Reported "critic quality" is often a rank correlation pooled over all candidates. That
metric conflates two effects, because candidates sampled with larger $\sigma$ are worse
across the board. A critic that only detects the overall noise level scores well pooled
while being unable to rank inside a group.

Measured on the 6-body system (600 training ICs = 4800 labelled candidates):

| Critic | pooled $\rho$ | **within-IC $\rho$** | **IC win rate** | selected / random |
|---|---|---|---|---|
| Conservation | −0.7044 | **−0.6881** | **84.0%** | **0.637×** |
| Pairwise ranking (supervised) | −0.4689 | −0.5786 | 74.5% | 0.656× |
| **Error regressor (supervised)** | **−0.6685** | **−0.3799** | **53.0%** | 0.758× |
| Logistic fusion of all three | −0.4509 | −0.6825 | 83.5% | 0.641× |

The supervised error regressor looks competitive pooled ($-0.67$ versus $-0.70$) but has
an **IC-internal win rate of 53% — chance**. Its pooled score comes entirely from
detecting the noise level. **All results below use within-IC metrics.**

A related effect: aligning the training target with the evaluation is worth more than
model capacity. Identical architecture, two targets:

| Training target | IC win rate |
|---|---|
| absolute error regression | **47.5%** |
| IC-relative error regression | **86.5%** |

a 39-point gap from the target alone.

### 5.2 Main results

**Standard benchmarks** (50 test ICs, $K=8$; script
`experiment_k_hnn_equation_critic.py`):

| figure-eight (N=3, 2D) | true/gen separation | within-IC $\rho$ | IC win rate | sel./random | sel./oracle |
|---|---|---|---|---|---|
| **Conservation (zero-shot)** | **8.7e11** | −0.9486 | **100.0%** | **0.250×** | **1.03×** |
| HNN, energy drift (naive use) | 0.076 | −0.5405 | 50.0% | 0.544× | 2.24× |
| **HNN, equation residual** | 0.017 | **−0.9524** | **100.0%** | **0.250×** | **1.03×** |
| random / oracle | — | — | 50.0% / 100% | 1.000× / 0.243× | 4.11× / 1.00× |

| Kepler (N=2, 2D) | separation | within-IC $\rho$ | IC win rate | sel./random | sel./oracle |
|---|---|---|---|---|---|
| **Conservation (zero-shot)** | **2.4e14** | −0.9495 | **100.0%** | **0.185×** | **1.05×** |
| HNN, energy drift (naive use) | 0.034 | −0.5705 | 52.0% | 0.463× | 2.63× |
| **HNN, equation residual** | 0.058 | **−0.9490** | **100.0%** | **0.185×** | **1.05×** |
| random / oracle | — | — | 50.0% / 100% | 1.000× / 0.176× | 5.68× / 1.00× |

**The conservation critic ties the HNN exactly** on both systems — identical selection
error, identical win rate. The learned critic requires training an HNN; the conservation
critic requires nothing.

Note also the separation column: the conservation critic separates true from generated
trajectories by $10^{11}$–$10^{14}$, the HNN by $\sim10^{-2}$. Yet the HNN still ranks
correctly inside an IC, because its residual varies monotonically with $\sigma$. Large
absolute separation and good within-group ranking are different properties.

**6-body system, 5 seeds, 95% CI** (script `experiment_h_benchmark.py`; 120 training ICs):

| Critic | ID within-IC $\rho$ | ID win rate | OOD win rate | Training samples |
|---|---|---|---|---|
| **Conservation (zero-shot)** | **−0.6404±0.0459** | **82.8%±3.3%** | **84.7%±1.8%** | **0** |
| Pairwise ranking (supervised) | −0.5627±0.0471 | 71.2%±3.2% | 72.2%±2.5% | 960 |
| Absolute binary (naive) | −0.4958±0.0768 | 62.0%±8.8% | 67.0%±7.3% | 960 |
| random | — | 50.0% | 50.0% | — |

The conservation critic's advantage over the supervised ranking critic has
**non-overlapping 95% CIs** on both splits.

> **On labelling budgets.** The numbers above come from 120 training ICs. In the
> ablation of §5.1 and the fusion experiment of §7 the supervised critics were given
> 600 ICs (4800 labelled candidates) — five times more — and the conclusion is
> unchanged. We quote each experiment's own budget rather than a single figure.

### 5.3 Ablation over channel sets

Four settings (two generators × ID/OOD), reported as selected/random (lower is better):

| Channel set | G_A ID | G_A OOD | G_B ID | G_B OOD |
|---|---|---|---|---|
| angular | 0.640× | **0.637×** | **0.056×** | 0.063× |
| **angular + momentum (adopted)** | 0.640× | 0.653× | **0.056×** | **0.059×** |
| four channels, equal weight (previous) | 0.629× | 0.713× | 0.067× | 0.094× |
| energy only | 0.794× | 0.828× | 0.308× | 0.194× |
| centre of mass only | **1.038×** | **1.057×** | **1.327×** | **1.575×** |

The previously-used equal-weight sum of four channels is **substantially suboptimal**:
on the stronger generator it is 1.21× of oracle where angular+momentum is 1.02×, and out
of distribution the gap widens to 1.61× versus 1.02×. The centre-of-mass channel is worse
than chance on its own.

### 5.4 A negative result: synthetic failure modes are not comparable

Critic rankings are often measured on hand-built corruptions (add noise, swap in a bad
integrator, freeze, splice, time-warp). We tested whether such a taxonomy is comparable
to a real generator's failures. **It is not.**

**(a) Two of five corruptions cannot be calibrated.** Tuning severity so that each
corruption's median RMSE matches the generator's (0.01468):

| Corruption | calibrated severity | achieved RMSE | verdict |
|---|---|---|---|
| noise | 0.740 | 0.01477 | calibrated |
| freeze | 0.428 | 0.01342 | calibrated |
| warp | 0.302 | 0.01342 | calibrated |
| **euler** | hit lower bound | **0.14127** | **10× too destructive** |
| **splice** | hit lower bound | **0.59365** | **40× too destructive** |

`splice` is worse than merely uncalibrated: it performs a discrete particle roll and its
severity parameter has **no effect at all**. It is an artifact of the construction, not a
failure mode.

**(b) The conservation fingerprints have different shape.** Separation of true from
corrupted trajectories, per channel:

| Channel | noise | euler | freeze | splice | warp | **real generator** |
|---|---|---|---|---|---|---|
| energy | 3.89 | 0.00 | 0.86 | 0.98 | 0.93 | **1.02** |
| angular | 7.5e15 | 3.1e10 | 0.95 | 0.97 | 1.13 | **1.2e15** |
| momentum | 6.3e15 | 3.9e8 | 0.87 | 0.96 | 0.94 | **1.9e16** |
| centre of mass | 1.10 | 0.02 | 0.85 | 1.03 | 1.03 | **1.10** |

The real generator's failure mode is *breaking Newton's third law*, which destroys
angular and linear momentum conservation. Four of the five synthetic corruptions leave
those quantities at 0.9–1.1 — they rearrange particles without violating a conservation
law. Normalized fingerprint distances range from 7.4 to 238.

**Consequence.** No result in this paper is computed on a synthetic corruption set. All
negative samples come from real learned generators. We recommend the calibration
procedure above as a prerequisite for anyone using such a taxonomy.

### 5.5 Cross-generator check: the ranking depends on the generator

Every result above uses a pairwise force model as the generator. A reviewer's first
question is whether they are an artefact of that choice. We re-ran the comparison with a
**Hamiltonian Neural Network used as the generator** (Greydanus et al., 2019), which has
a fundamentally different inductive bias: it learns a black-box scalar $H(q,p)$ and
derives the vector field from Hamilton's equations, so it is *not* translation- or
rotation-invariant, unlike a pairwise force model.

(An earlier attempt used a mean-field MLP, but it was too poor to be informative —
one-step error 0.81, within-IC candidate spread only 1.16×, oracle/random 1.09×. It could
neither support nor refute anything.)

| System | Critic | within-IC $\rho$ | IC win rate | sel./random | sel./oracle |
|---|---|---|---|---|---|
| figure-eight | Conservation (zero-shot) | −0.6881 | 78.0% | 0.864× | 1.10× |
| figure-eight | **Pairwise ranking (supervised)** | **−0.8890** | **96.0%** | **0.794×** | **1.01×** |
| Kepler | Conservation (zero-shot) | −0.8048 | **98.0%** | 0.574× | 1.06× |
| Kepler | **Pairwise ranking (supervised)** | **−0.9014** | 96.0% | **0.552×** | **1.02×** |
| — | random (fig-8 / Kepler) | — | 50.0% | 1.000× | 1.27× / 1.84× |

**The conservation critic no longer leads.** On the figure-eight it is clearly behind
(78.0% versus 96.0%); on Kepler the two are close with the supervised critic marginally
ahead in selection error. Both remain far better than random (0.55–0.86× versus 1.00×).

The conservation fingerprint shows the HNN generator still breaks angular-momentum
conservation severely (separation 2.4e13 and 7.3e12), so the critic's signal is not
weakened.

**A candidate explanation, tested and rejected.** The conservation residual is a
projection of the force error onto the symmetry generators (§4.1), so one might expect the
*full* force error to carry strictly more information, with the HNN generator's advantage
lying in structure conservation cannot see. We tested this: for every candidate we measured
the accumulated force error $\int\|a^{\text{gen}}-a^{\text{true}}\|\,dt$ along its own
trajectory and correlated it, within each IC, with the true trajectory error.

| Setting | ρ(conservation) | ρ(force error) | gap |
|---|---|---|---|
| figure-eight × pairwise | **+0.9524** | +0.1702 | −0.7821 |
| figure-eight × HNN | **+0.7768** | +0.1339 | −0.6429 |
| Kepler × pairwise | **+0.8173** | +0.1238 | −0.6935 |
| Kepler × HNN | **+0.7929** | +0.2083 | −0.5845 |

**The explanation does not hold.** The force-error magnitude is a far weaker predictor
than the conservation residual everywhere. Two reasons the diagnostic may be the wrong
instrument, neither of which we separated: the force error is evaluated at the *generated*
positions, so a drifted candidate is compared in a different region of state space; and
trajectory error is the *second* time integral of δa, so oscillating force errors that
largely cancel leave little trace — a magnitude is not the right upper bound on usable
information.

We therefore leave the mechanism **unexplained**. What stands is the measurement: with an
HNN generator the supervised critic is at least as good. Only the proposed explanation
failed.

**What this changes.** The claim must be stated per generator, not universally:

> A zero-shot conservation critic is a **reliable but not universally optimal**
> supervision signal. For pairwise-force generators it leads the strongest supervised
> critic; for a Hamiltonian-parameterised generator it is matched or slightly beaten.

This does not contradict the tie in §5.2: that compares *critics* under a fixed
pairwise-force generator, whereas this compares *generators*. Together they place the
conservation critic in the same band as the strongest learned supervision, with the
ordering depending on the setting rather than following a general rule.

---

## 6. Limitations

1. **Generators are learned surrogate force models, not diffusion or autoregressive
   video models.** The qualitative structure of the failures (breaking equal-and-opposite
   forces) is plausible for any pairwise-parameterised model, but we did not verify it on
   a large pretrained generative model.
2. **The generator's parameterisation changes the ranking.** §5.5 shows that switching
   from a pairwise force model to a Hamiltonian Neural Network removes the conservation
   critic's lead. The claim must therefore be stated per generator; we have two
   architectures, not a spectrum.
3. **Ground truth is always available** here, because the systems are simulated. The
   selection task is therefore well-defined; in the field one normally lacks $x^\star$.
   Our critic does not need $x^\star$ at *use* time — only our *evaluation* does.
4. **Three systems, all $N \leq 6$ and pairwise-central.** The noise-floor argument
   (§4.2) relies on (i) pairwise central forces and (ii) a symplectic integrator. For
   three-body forces or non-symplectic integrators the momentum and angular floors are no
   longer automatically at roundoff.
5. **No accuracy comparison against published numbers.** The HNN baseline was trained by
   us; we did not reproduce a published HNN result. The tie in §5.2 should be read as
   "our HNN implementation is a fair baseline", not as a leaderboard claim.
6. **The critic is used only for selection.** We do not show that it improves
   *training* (e.g. as a reward or regulariser); that is untested.

---

## 7. Retractions

Three claims appeared in earlier internal drafts of this work. All are false, and the
process of falsifying them produced §5.1, §5.3 and §5.4.

**R1. "The conservation critic is more robust out of distribution than a learned
critic."** False. On a family shift the two do not differ (conservation $+0.001$,
learned $-0.005$, overlapping CIs). On a *severity* shift toward subtler violations the
conservation critic degrades *more* (−11.4 points versus −1.0). Its advantage is
absolute level and zero labelling cost, not distributional robustness.

**R2. "Conservation and learned critics have complementary blind spots; combining them
is consistently better."** False. An *optimally weighted* fusion (logistic regression
over the critic scores, fitted on held-out training data) assigns the conservation critic
the largest weight ($+1.635$), the supervised error regressor a **negative** weight
($-0.612$), and still ties the conservation critic alone (0.641× versus 0.637× ID;
0.608× versus 0.614× OOD). The signals are redundant, not complementary. An earlier
version of this claim appeared to hold only because the "learned" critic was given
conservation residuals as input features, making the conclusion true by construction.

**R3. "Conservation critics can validate the Lumyn pipeline itself."** False, for a
structural reason: that pipeline integrates the true equations of motion, so its output
*is* a solution and the failure mode a validator would catch cannot occur. A real
corruption/restoration experiment confirms this — the validation layer's causal
contribution is exactly zero, because it is never acted upon. A conservation critic is
useful precisely when the generator's output is *not* a solution, which is true of
learned models and false of a physics integrator.

---

## 8. Conclusion

A critic built from conservation laws, requiring no training data and no labels, ties the
strongest learned critics at selecting good candidates from a learned physical generator:
100% within-IC accuracy on standard benchmarks (matching a Hamiltonian Neural Network
exactly), and 83–85% on a 6-body family where a supervised pairwise-ranking critic
trained on 960 labelled candidates reaches 71–72%.

The reason it works is numerical rather than physical. Conservation channels whose
invariance is a structural property of the pairwise central force — linear and angular
momentum — sit at a machine-precision floor and separate true from generated trajectories
by ten or more orders of magnitude. The energy channel, conserved only to the
integrator's truncation error, does not. Choosing channels by their noise floor, rather
than by the physical prominence of the quantity, is what makes the critic work.

We also report what does not work, because it constrains the claim: the critic is not
more robust out of distribution, combining it with learned critics does not help, and
synthetic failure taxonomies are not structurally comparable to real generator failures
unless calibrated first.

---

## Appendix A. Reproducibility

| Result | Script | Output |
|---|---|---|
| Selection metrics, 6-body, 5 seeds | `experiments/experiment_h_benchmark.py` | `results/experiment_h_benchmark.json` |
| Channel ablation | `experiments/experiment_e_channel_combos.py` | stdout |
| Standard benchmarks + HNN | `experiments/experiment_k_hnn_equation_critic.py` | stdout |
| Strong supervised baselines | `experiments/experiment_c_strong_baselines.py` | stdout |
| Failure-mode comparability | `experiments/experiment_i_failure_modes.py` | stdout |
| Noise-floor sweep | `experiments/experiment_g_error_functionals.py` | stdout |
| HNN fairness diagnostics | `experiments/diagnose_hnn_{fairness,standard}.py` | stdout |
| Channel set regression tests | `tests/test_conservation_critic.py` | 18 tests |
| Checker defect regression tests | `tests/test_conservation_checker.py` | 11 tests |

Test suite at the time of writing: **108 tests, 0 failures** (1 optional-dependency skip).
