# Research References

This file records sources that inform the z3 architecture. It separates
external research support from CertiQ-specific theorem claims.

## 1. Transformer Architecture Discipline

Source:

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin, "Attention Is All You
  Need," 2017.
- Verified page: https://arxiv.org/abs/1706.03762

What it supports:

The paper supports the design lesson that a strong architecture can be organized
around a small reusable primitive. For the Transformer, that primitive is
attention-based sequence transduction.

What it does not support:

It does not support any queueing stability theorem, Lyapunov certificate, or
dispatch guarantee for CertiQ Dispatcher.

## 2. Permutation-Invariant Set Processing

Source:

- Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan
  Salakhutdinov, and Alexander Smola, "Deep Sets," NeurIPS 2017.
- Verified page: https://papers.nips.cc/paper/6931-deep-sets
- Preprint: https://arxiv.org/abs/1703.06114

What it supports:

This source supports using shared local maps and invariant aggregation when the
input is an unordered set. It is directly relevant to resource-label symmetry.

What it does not support:

It does not prove queueing stability or certify learned dispatch policies.

## 3. Attention Over Sets

Source:

- Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee
  Whye Teh, "Set Transformer: A Framework for Attention-based
  Permutation-Invariant Neural Networks," ICML 2019.
- Verified page: https://proceedings.mlr.press/v97/lee19d.html
- Preprint: https://arxiv.org/abs/1810.00825

What it supports:

This source supports attention-based encoders for unordered resource sets.
It is useful if z3 replaces simple pooling with resource-set attention.

What it does not support:

It does not justify a Lyapunov drift envelope or a CTMC stability claim.

## 4. Foster-Lyapunov Stability Language

Source:

- Sean P. Meyn and Richard L. Tweedie, Markov Chains and Stochastic Stability.
- Verified page: https://probability.ca/MT/
- Publisher metadata: https://www.cambridge.org/core/books/markov-chains-and-stochastic-stability/8E7E6C0907EDE33B9E25A5F6C8A3D0F0

What it supports:

This source supports standard Foster-Lyapunov recurrence language after a drift
inequality has already been proved.

What it does not support:

It does not prove the CertiQ base envelope or the z3 architecture.

## 5. Queueing Stability And MaxWeight Context

Source:

- Leandros Tassiulas and Anthony Ephremides, "Stability Properties of
  Constrained Queueing Systems and Scheduling Policies for Maximum Throughput
  in Multihop Radio Networks," IEEE Transactions on Automatic Control, 1992.
- Verified technical report page: https://drum.lib.umd.edu/items/571fda52-aefb-4497-9a2d-69d8c7c907b9
- DOI record: https://doi.org/10.1109/9.182479

What it supports:

This source supports the broader queueing-control principle that stability can
be designed through queue-aware scheduling and Lyapunov-style reasoning.

What it does not support:

It does not prove the CertiQ Dispatcher architecture or the exact z2 backbone
constant.

## 6. Existing CertiQ z2 Proof Package

Source:

- `docs/z2/formal_math/01_backbone_stability_and_constant.md`
- `docs/z2/formal_math/02_gate_inheritance_theorems.md`

What it supports:

These project-local proofs support the current analytic base envelope,
explicit \(C_B\), CTMC positive Harris recurrence for the base policy, hard
fallback inheritance, and exact projection inheritance.

What it does not support:

They do not prove arbitrary neural residual stability, approximate projection,
smooth fallback certification, or exact CTMC certification for adapters that
violate the declared CTMC model.

