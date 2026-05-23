# Research References

This file records the research anchors used to shape the first CertiQ-Net
formal package. It is not a full bibliography.

## 1. Differentiable Discrete-Event Simulation

Relevant direction:

- differentiable simulation for queueing and stochastic service systems,
- smoothed event dynamics,
- pathwise gradients,
- large sample-efficiency gains over score-function estimators in reported
  experiments.

Use in CertiQ-Net:

- training engine for the neural residual,
- not a stability proof.

Reference anchors:

- Ethan Che, Jing Dong, and Hongseok Namkoong,
  "Differentiable Discrete Event Simulation for Queuing Network Control,"
  arXiv:2409.03740, 2024.
  <https://arxiv.org/abs/2409.03740>
- ICML 2023 workshop version:
  "Dynamic Control of Queuing Networks via Differentiable Discrete-Event
  Simulation."
  <https://icml.cc/virtual/2023/28806>

## 2. Structure-Aware RL For Queues

Relevant direction:

- low-dimensional structured policy classes for queueing control,
- threshold or soft-threshold policies,
- policy-gradient convergence for restricted parameterizations.

Use in CertiQ-Net:

- contextual comparison against structured learning policies,
- motivation that queueing structure matters.

Reference anchor:

- Neharika Jali, Guannan Qu, Weina Wang, and Gauri Joshi,
  "Efficient Reinforcement Learning for Routing Jobs in Heterogeneous Queueing
  Systems," AISTATS 2024 / arXiv:2402.01147.
  <https://arxiv.org/abs/2402.01147>

## 3. Stability-Guaranteed Learning

Relevant direction:

- learning dynamic routing policies with Foster-Lyapunov guarantees,
- local Lyapunov functions for policy-gradient analysis on unbounded state
  spaces,
- certification of learned controllers.

Use in CertiQ-Net:

- proof discipline,
- comparison against non-neural but certified routing methods.

Reference anchor:

- Yidan Wu, Feixiang Shu, Jianan Zhang, and Li Jin,
  "Learning-Based Adaptive Dynamic Routing with Stability Guarantee for a
  Single-Origin-Single-Destination Network," arXiv:2408.14758, 2024.
  <https://arxiv.org/abs/2408.14758>

## 4. Bibliography Tasks Before Manuscript Writing

Before any public paper draft:

1. replace this file with exact BibTeX records,
2. verify venue, year, and version for each cited paper,
3. distinguish published results from arXiv preprints,
4. cite theorems only from source text that actually proves them,
5. avoid relying on strategy memos as literature evidence.
