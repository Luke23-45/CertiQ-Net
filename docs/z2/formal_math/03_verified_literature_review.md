# Verified Literature Review For Phase 1

This file records the external literature that can be cited around the Phase 1
CertiQ-Net z2 math package. It is a proofread review, not a dumping ground.
Only sources whose bibliographic identity has been checked are included.

The role of this file is narrow:

1. identify the external sources that support formal language around CTMC
   Foster-Lyapunov reasoning, permutation-invariant / equivariant set
   architectures, and differentiable simulation,
2. separate those sources from the original CertiQ-Net theorem content,
3. prevent accidental overclaiming.

The CertiQ-Net backbone envelope proof and gate inheritance proofs in files `01`
and `02` are original package-level arguments. They are not imported from the
literature below.

---

## 1. Foster-Lyapunov Reference

### Source

Sean P. Meyn and Richard L. Tweedie, *Markov Chains and Stochastic Stability*,
2nd edition, Cambridge University Press, 2009.

Verified source pages:

- [Cambridge / book metadata](https://www.cambridge.org/core/books/markov-chains-and-stochastic-stability/8E7E6C0907EDE33B9E25A5F6C8A3D0F0)
- [Author-hosted book page](https://probability.ca/MT/)

### What this source supports

This source is the standard reference for Foster-Lyapunov drift criteria and
positive Harris recurrence language for Markov chains on general state spaces.
For the present package, it supports the use of a standard Foster-Lyapunov
criterion after the drift inequality has already been proved.

### What this source does not support

It does not prove the CertiQ-Net backbone envelope, it does not prove the
softmax minimum lemma used here, and it does not by itself certify the z2
architecture. Those steps are CertiQ-Net-specific.

---

## 2. Deep Sets

### Source

Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan
Salakhutdinov, and Alexander Smola, "Deep Sets," NeurIPS 2017.

Verified source pages:

- [NeurIPS proceedings page](https://papers.nips.cc/paper/6931-deep-sets)
- [ArXiv preprint](https://arxiv.org/abs/1703.06114)

### What this source supports

This paper supports the use of permutation-invariant set encoders and the basic
shared-local plus invariant-pooling design principle behind the CertiQ-Net
residual architecture.

### What this source does not support

It does not prove stability, queueing performance, or any Lyapunov
certification result for CertiQ-Net. It is an architecture source only.

---

## 3. Set Transformer

### Source

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye
Teh, "Set Transformer: A Framework for Attention-based Permutation-Invariant
Neural Networks," ICML 2019, PMLR 97:3744-3753.

Verified source pages:

- [PMLR proceedings page](https://proceedings.mlr.press/v97/lee19d.html)
- [ArXiv preprint](https://arxiv.org/abs/1810.00825)

### What this source supports

This paper supports the claim that attention-based set models are a legitimate
permutation-invariant building block if CertiQ-Net later upgrades the residual
encoder beyond simple pooling.

### What this source does not support

It does not justify any queueing theorem, stability guarantee, or drift
projection argument in this package.

---

## 4. Differentiable Discrete-Event Simulation

### Source

Ethan Che, Jing Dong, and Hongseok Namkoong, "Differentiable Discrete Event
Simulation for Queuing Network Control," arXiv:2409.03740, 2024.

Verified source pages:

- [ArXiv abstract](https://arxiv.org/abs/2409.03740)
- [Author-hosted PDF](https://ethche.github.io/files/differentiable_discrete_event.pdf)
- [OpenReview workshop version: "Dynamic Control of Queuing Networks via Differentiable Discrete-Event Simulation"](https://openreview.net/forum?id=yg4Fkzl7ko)

### What this source supports

This line of work supports the claim that queueing-network control can be
trained with pathwise gradients using smoothed event dynamics, and that such
training can be materially more sample-efficient than score-function methods in
their reported experiments.

### What this source does not support

It does not prove CertiQ-Net stability. It is a training-engine reference, not
a certificate reference.

---

## 5. Structure-Aware RL For Heterogeneous Queues

### Source

Neharika Jali, Guannan Qu, Weina Wang, and Gauri Joshi, "Efficient
Reinforcement Learning for Routing Jobs in Heterogeneous Queueing Systems,"
AISTATS 2024, PMLR 238:4177-4185.

Verified source pages:

- [PMLR proceedings page](https://proceedings.mlr.press/v238/jali24a.html)
- [ArXiv preprint](https://arxiv.org/abs/2402.01147)

### What this source supports

This paper supports the broader motivation that queueing structure matters and
that low-dimensional structured policy classes can be materially more efficient
than unconstrained RL for heterogeneous routing problems.

### What this source does not support

It does not supply the CertiQ-Net certification theorem, the exact envelope
constant \(C_B\), or the gate inheritance theorems.

---

## 6. Stability-Guaranteed Learning In Routing

### Source

Yidan Wu, Feixiang Shu, Jianan Zhang, and Li Jin, "Learning-Based Adaptive
Dynamic Routing with Stability Guarantee for a
Single-Origin-Single-Destination Network," arXiv:2408.14758, 2024.

Verified source page:

- [ArXiv abstract](https://arxiv.org/abs/2408.14758)

### What this source supports

This paper supports the general research framing that learned routing policies
can be studied together with Foster-Lyapunov-style stability guarantees.

### What this source does not support

Its model class and proof structure are different from the CertiQ-Net CTMC
dispatch model. It should be cited as adjacent motivation, not as direct proof
support for the z2 theorems.

---

## 7. Literature Discipline Rules For The Paper

When these references are used in the CertiQ-Net paper, the manuscript should
follow these rules:

1. cite Meyn-Tweedie only for the general Foster-Lyapunov criterion, not for
   the CertiQ-Net envelope proof,
2. cite Deep Sets and Set Transformer only for permutation-invariant /
   equivariant neural design,
3. cite Che-Dong-Namkoong only for differentiable training methodology,
4. cite Jali et al. and Wu et al. only as neighboring queueing-control
   literature,
5. do not imply that any external paper proves the CertiQ-Net z2 certification
   route,
6. separate published proceedings papers from arXiv-only references whenever
   bibliographic status matters in the manuscript.

That discipline is necessary to keep the theoretical results section clean.
