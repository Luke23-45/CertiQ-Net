# Experiment Protocol

This file defines the CertiQ-Net empirical program. The protocol is about the
architecture and certificate mechanism, not about re-running classical dispatch
comparisons.

## 1. Experimental Units

Compare only architecture variants unless a reviewer explicitly asks for a
background reference policy.

Required variants:

1. analytic base controller only,
2. neural residual without certificate gate,
3. compact-support certificate gate,
4. smooth certificate gate,
5. Lyapunov projection layer,
6. full CertiQ-Net with differentiable-simulation training.

The uncontrolled neural residual is included as an ablation to show what the
certificate gate prevents. It is not treated as a serious deployable policy.

## 2. Systems

Use system families that test architectural generalization.

### Family A: fixed heterogeneous anchor

Use one declared heterogeneous service-rate vector and load as the reproducible
anchor case. The anchor is for controlled architecture ablations, not for
relitigating dispatch-rule comparisons.

### Family B: random heterogeneous fleets

Sample

\[
\mu_i\sim\mathrm{LogNormal}(m,s^2)
\]

and set

\[
\lambda=\rho\sum_i\mu_i,
\qquad \rho\in\{0.7,0.85,0.95\}.
\]

### Family C: clustered hardware profiles

Use grouped service-rate profiles representing heterogeneous compute pools:

- slow group,
- medium group,
- fast group.

The simulation uses service rates unless real trace data is supplied.

### Family D: size transfer

Train on smaller systems and test on larger systems:

\[
N_{\mathrm{train}}\in\{8,16,32\},
\qquad
N_{\mathrm{test}}\in\{64,128\}.
\]

This family directly tests whether the permutation-equivariant design matters.

## 3. Metrics

Primary performance metric:

\[
\bar Q
=
\frac1T\int_0^T\sum_i Q_i(t)\,dt.
\]

Architecture and certificate metrics:

- improvement over analytic base controller,
- gate activation rate,
- fraction of decisions modified by projection,
- drift-envelope slack,
- drift-envelope violation rate,
- tail fallback activation rate,
- gradient variance,
- training wall-clock,
- transfer gap from train \(N\) to test \(N\).

Tail metrics:

- 95th and 99th percentile total queue length,
- maximum observed total queue length,
- runaway or instability rate.

## 4. Statistical Protocol

For each configuration:

- use common random seeds across architecture variants where possible,
- discard a declared warm-up interval,
- report confidence intervals over independent replications,
- fix the training budget before comparing variants,
- report unstable runs explicitly,
- report certificate violations even when performance is strong.

## 5. Required Ablations

Run:

1. base controller only,
2. residual logits only,
3. residual plus compact gate,
4. residual plus smooth gate,
5. residual plus projection,
6. learned base-controller parameters without residual,
7. full CertiQ-Net.

The paper should isolate which component creates performance and which
component creates certification.

## 6. Success Thresholds

Minimum architecture result:

- the gated residual improves over the analytic base controller on at least one
  declared system family,
- the certificate gate or projection reports zero drift-envelope violations in
  the audited state bank,
- the residual-without-certificate ablation shows why the certificate layer is
  necessary or useful,
- size-transfer degradation is measured rather than ignored.

Strong result:

- the projected model improves the analytic base controller under high load,
- the model transfers from small \(N\) to larger \(N\),
- differentiable simulation lowers gradient variance or training cost relative
  to score-function training,
- the certificate mechanism remains active and auditable during deployment.

## 7. Failure Outcomes Worth Reporting

The paper remains meaningful if:

- the residual gives small gains but the certificate mechanism is clean,
- the projection layer rejects many neural proposals and explains where the
  unconstrained model fails,
- differentiable simulation improves training efficiency without changing final
  cost,
- the strongest result is a compact theorem-backed architecture rather than a
  large black-box model.

These are architecture findings, not failed comparison races.
