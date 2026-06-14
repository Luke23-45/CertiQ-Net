# Formal Dispatch Model

## 1. Resources

There are \(N\ge 1\) resources. Resource \(i\) has service capacity
\(\mu_i>0\), and the capacity vector is
\[
\mu=(\mu_1,\ldots,\mu_N).
\]

Total service capacity is
\[
\Lambda=\sum_{i=1}^N\mu_i.
\]

Each resource may also carry context
\[
\xi_i\in\mathbb R^{d_\xi}.
\]

The context may encode metadata, cost, compatibility, location, or any other
domain signal. Context does not automatically enter the certificate. It enters
certification only when the implementation contract or theorem statement names
it explicitly.

## 2. Arrivals And State

Jobs arrive online. In the primary CTMC model, arrivals are Poisson with rate
\[
\lambda>0.
\]

The state is
\[
Q=(Q_1,\ldots,Q_N)\in\mathbb Z_+^N,
\]
where \(Q_i\) is the number of queued or active jobs at resource \(i\).

The primary CTMC assumptions require
\[
\lambda<\Lambda.
\]

This is the subcritical load condition. It is necessary for the system-level
model, but it does not certify an arbitrary policy.

## 3. Actions

At each arrival the dispatcher observes \((Q,\mu,\xi)\) and outputs a policy
\[
\pi(Q,\mu,\xi)\in\Delta_N,
\qquad
\Delta_N=\{p\in\mathbb R_+^N:\sum_i p_i=1\}.
\]

The arriving job is assigned to resource \(i\) with probability \(\pi_i\).
In the codebase this policy is produced either by the legacy reflected-pressure
dispatcher or by the learned index model.

## 4. Service Model

For the primary CTMC model, service completions are exponential. When
\(Q_i>0\), resource \(i\) completes one job at rate \(\mu_i\).

The generator under policy \(\pi\) is
\[
(\mathcal L_\pi f)(Q)
=
\lambda\sum_i\pi_i(Q,\mu,\xi)[f(Q+e_i)-f(Q)]
+
\sum_i\mu_i\mathbf 1_{\{Q_i>0\}}[f(Q-e_i)-f(Q)].
\]

The policy controls only the arrival term. Service dynamics are fixed by
\(\mu\).

## 5. Objective

The default performance objective is average queueing cost
\[
J(\pi)
=
\limsup_{T\to\infty}
\frac1T
\mathbb E_\pi
\left[
\int_0^T c(Q(t),\mu,\xi)\,dt
\right].
\]

The canonical cost is total backlog:
\[
c(Q)=\sum_i Q_i.
\]

Weighted costs are allowed when weights are positive and bounded:
\[
c(Q,\mu,\xi)=\sum_i w_i(\mu,\xi)Q_i.
\]

Certification is about stability and admissibility. It is not a proof of
optimality.

## 6. Delay-Aligned Comparison Geometry

The repository implements two classical routing indices for comparison and
baseline evaluation:
\[
d_i^{SED}(Q,\mu)=\frac{Q_i+1}{\mu_i},
\qquad
d_i^{QMD}(Q,\mu)=\frac{2Q_i+1}{\mu_i}.
\]

These indices define the SED and quadratic-min-drift baselines, and they also
anchor the learned index model. The index model learns a residual correction
on top of the quadratic drift index.

## 7. Adapter Contract

A domain adapter maps a real system into
\[
\mathcal D=(N,\lambda,\mu,\xi,Q,c).
\]

An adapter must state:

1. what counts as an arrival,
2. what counts as a resource,
3. how \(\mu_i\) is measured,
4. what \(Q_i\) counts,
5. what service law is assumed,
6. what context \(\xi_i\) means,
7. what cost is optimized,
8. whether the CTMC assumptions apply exactly.

If the adapter violates the CTMC assumptions, CertiQ may still be used
empirically, but no CTMC stability claim should be attached to that run.
