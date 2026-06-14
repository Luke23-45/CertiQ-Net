# Formal Dispatch Model

## 1. Resources And Context

Let \(N \ge 1\) denote the number of resources. Resource \(i\) has service
capacity \(\mu_i>0\), and

\[
\mu=(\mu_1,\ldots,\mu_N)\in\mathbb R_+^N.
\]

Each resource may also carry context

\[
\xi_i\in\mathbb R^{d_\xi},
\qquad
\xi=(\xi_1,\ldots,\xi_N).
\]

The context vector is an exogenous feature of the resource. It may encode
resource attributes such as location, compatibility, cost, or other domain
information.

## 2. Queue State And Arrivals

The system state is

\[
Q=(Q_1,\ldots,Q_N)\in\mathbb Z_+^N,
\]

where \(Q_i\) is the number of jobs currently assigned to resource \(i\).

In the primary continuous-time Markov chain model, arrivals form a Poisson
process with rate \(\lambda>0\). The total service capacity is

\[
\Lambda=\sum_{i=1}^N \mu_i,
\]

and the subcritical load condition is

\[
\lambda<\Lambda.
\]

## 3. Action Space

At each arrival epoch, the dispatcher outputs a distribution

\[
\pi(Q,\mu,\xi)\in\Delta_N,
\qquad
\Delta_N=\left\{p\in\mathbb R_+^N:\sum_{i=1}^N p_i=1\right\}.
\]

The arriving job is assigned to resource \(i\) with probability \(\pi_i\).

## 4. Service Dynamics

When \(Q_i>0\), resource \(i\) completes jobs at rate \(\mu_i\). Under policy
\(\pi\), the generator acting on bounded \(f:\mathbb Z_+^N\to\mathbb R\) is

\[
(\mathcal L_\pi f)(Q)
=
\lambda\sum_{i=1}^N \pi_i(Q,\mu,\xi)\bigl[f(Q+e_i)-f(Q)\bigr]
+\sum_{i=1}^N \mu_i\,\mathbf 1_{\{Q_i>0\}}\bigl[f(Q-e_i)-f(Q)\bigr].
\]

The policy controls only the routing term.

## 5. Performance Objective

The long-run average cost of policy \(\pi\) is

\[
J(\pi)
=
\limsup_{T\to\infty}
\frac1T
\mathbb E_\pi\!\left[\int_0^T c(Q(t),\mu,\xi)\,dt\right].
\]

The canonical cost is total backlog,

\[
c(Q)=\sum_{i=1}^N Q_i.
\]

Weighted costs may be used when the weights are positive and bounded.

## 6. Delay-Aligned Geometry

Two queueing geometries are used throughout the CertiQ family:

\[
d_i^{SED}(Q,\mu)=\frac{Q_i+1}{\mu_i},
\qquad
d_i^{QMD}(Q,\mu)=\frac{2Q_i+1}{\mu_i}.
\]

The first is the shortest-expected-delay geometry. The second is the
quadratic-min-drift geometry used by the index model.

## 7. Adapter Specification

A domain adapter maps an application into

\[
\mathcal D=(N,\lambda,\mu,\xi,Q,c).
\]

The adapter must specify:

1. the arrival process,
2. the resource set,
3. the meaning of \(\mu_i\),
4. the meaning of \(Q_i\),
5. the service law,
6. the meaning of \(\xi_i\),
7. the performance cost,
8. whether the CTMC assumptions hold exactly.

If the CTMC assumptions do not hold exactly, the model may still be used as an
empirical dispatcher, but the CTMC generator interpretation does not apply
without qualification.
