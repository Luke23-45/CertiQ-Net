# Queueing Model

This file defines the mathematical object controlled by CertiQ-Net.

## 1. System

There are \(N\ge 1\) parallel servers. Server \(i\) has service rate
\(\mu_i>0\). Jobs arrive according to a Poisson process of rate
\(\lambda>0\). Service times at server \(i\) are exponential with rate
\(\mu_i\), independent across jobs and servers.

The queue-length state is

\[
Q(t)=(Q_1(t),\ldots,Q_N(t))\in\mathbb Z_+^N.
\]

The total service capacity is

\[
\Lambda := \sum_{i=1}^N\mu_i.
\]

The natural subcritical load condition is

\[
\lambda < \Lambda.
\]

This condition is necessary for any non-idling stable policy in the aggregate
capacity sense. It is not by itself sufficient for arbitrary state-dependent
routing rules.

## 2. Routing Policy

At each arrival epoch, the policy observes \(Q\), system descriptors \(\xi\),
and returns a probability vector

\[
\pi(Q,\xi)\in\Delta_N
:=
\left\{p\in\mathbb R_+^N:\sum_i p_i=1\right\}.
\]

The arrival is routed to server \(i\) with probability \(\pi_i(Q,\xi)\).

The controlled CTMC generator for a test function \(f:\mathbb Z_+^N\to\mathbb R\)
is

\[
(\mathcal L_\pi f)(Q)
=
\lambda\sum_{i=1}^N \pi_i(Q,\xi)
\left[f(Q+e_i)-f(Q)\right]
+
\sum_{i=1}^N \mu_i\mathbf 1_{\{Q_i>0\}}
\left[f(Q-e_i)-f(Q)\right].
\]

## 3. State Descriptors

The minimal descriptor for server \(i\) is

\[
x_i(Q,\xi)
=
\left[
Q_i,\ \mu_i,\ Q_i/\mu_i,\ Q_i/\mu_i^\beta,\ \log\mu_i
\right].
\]

Optional descriptors may include:

- hardware class embedding,
- estimated service-time distribution features,
- energy cost,
- locality or network-distance features,
- job-class compatibility features.

The formal package assumes only \((Q_i,\mu_i)\). Additional features are
application-level extensions and require separate evidence.

## 4. Objective

The primary infinite-horizon average-cost objective is

\[
J(\pi)
=
\limsup_{T\to\infty}
\frac1T
\mathbb E_\pi
\left[
\int_0^T
c(Q(t),\xi)\,dt
\right],
\]

with default cost

\[
c(Q,\xi)=\sum_{i=1}^N Q_i.
\]

Weighted or latency-aware costs are allowed:

\[
c_w(Q,\xi)=\sum_i w_i Q_i,
\qquad w_i>0,
\]

but must be declared before experiments.

## 5. Stability Target

The primary certification target is positive Harris recurrence of the CTMC.

A sufficient route is a Foster-Lyapunov condition: find a norm-like
\(V:\mathbb Z_+^N\to[0,\infty)\), a finite set \(C\), and constants
\(\epsilon>0\), \(b<\infty\) such that

\[
(\mathcal L_\pi V)(Q)
\le
-\epsilon h(Q)+b\mathbf 1_C(Q),
\]

where \(h(Q)\to\infty\) as \(|Q|_1\to\infty\), commonly
\(h(Q)=|Q|_1\).

The default Lyapunov candidate for CertiQ-Net certification is the weighted quadratic

\[
V(Q)=\frac12\sum_i \frac{Q_i^2}{\mu_i^\beta}.
\]
