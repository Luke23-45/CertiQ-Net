# Architecture Principles

## 1. Design Target

CertiQ is designed for one primitive:

> assign one arriving unit of work to one heterogeneous resource online while
> preserving an auditable certificate.

Every implemented component must support that primitive. A component is valid
only if it has a clear role in at least one of:

1. representing the unordered resource set,
2. comparing backlog against service capacity,
3. proposing a performance-improving dispatch policy,
4. enforcing a certificate before dispatch,
5. exposing diagnostics that can be audited.

The current codebase implements two certified dispatcher families:

1. `CertiQDispatcher`
   - legacy reflected-pressure architecture,
   - analytic base geometry,
   - scalar usage cap or fallback certification.
2. `CertiQIndexModel`
   - learned marginal-cost index architecture,
   - SED/QMD-aligned geometry,
   - exact KL projection onto a fixed QMD budget,
   - no reflected pressure in the forward path.

## 2. Architectural Discipline

The relevant architectural lesson is compression: define a small number of
operations that directly match the dispatch primitive, then reuse them
consistently.

For CertiQ, the central operation is:

\[
\text{certified assignment}
=
\text{proposal}
\quad\text{filtered through}\quad
\text{state-dependent admissibility}.
\]

Each block must answer four questions:

1. What mathematical object does this block produce?
2. What invariance or constraint does it preserve?
3. What failure mode does it prevent?
4. What diagnostic verifies correct behavior?

## 3. Three-Layer Architecture

The implemented system separates into three layers.

### 3.1 Geometry Layer

The geometry layer maps queue state and capacity into a dispatch energy. The
project currently exposes three related geometries:

\[
y_i^{\beta}(Q)=Q_i/\mu_i^\beta,
\qquad
d_i^{SED}(Q,\mu)=\frac{Q_i+1}{\mu_i},
\qquad
d_i^{QMD}(Q,\mu)=\frac{2Q_i+1}{\mu_i}.
\]

`CertiQDispatcher` uses the capacity-normalized workload geometry
\(y_i^\beta(Q)\) as its certificate coordinate. `CertiQIndexModel` uses the
quadratic drift geometry \(d_i^{QMD}\) as the fixed backbone for the learned
index. `SED` remains a comparison baseline, but it is not the tail policy for
the canonical index model.

### 3.2 Proposal Layer

The proposal layer learns performance corrections. It may use context
\(\xi\), resource-local features, pooled set summaries, or reflected pressure.
In the current implementation the legacy dispatcher uses a permutation-equivariant
proposal module, while the index model learns a marginal-cost residual on top of
the quadratic drift index.

The proposal layer is not the source of certification. It is the source of
adaptivity and performance.

### 3.3 Certificate Layer

The certificate layer maps a proposal into a certified final distribution.
This is where the architecture earns the name CertiQ. If the final policy
violates the certificate, the architecture has failed, even if training loss
improves.

## 4. Required Invariances

Resource labels have no intrinsic meaning. The architecture must be
permutation equivariant:

\[
\pi(\sigma Q,\sigma\mu,\sigma\xi)=\sigma\pi(Q,\mu,\xi).
\]

This requirement affects all blocks:

1. local feature maps must be shared across resources,
2. global summaries must be permutation invariant,
3. resource scores must be produced equivariantly,
4. the certificate operator must commute with resource permutation,
5. diagnostics must reorder consistently with the state.

## 5. First-Class Certificate Boundary

Certification is not a training regularizer.

The final action distribution must satisfy the certificate at inference time by
construction. A drift penalty may help training, but it cannot substitute for
projection, fallback, or another exact certificate operator.

## 6. What Makes The Design Acceptable

An acceptable CertiQ architecture should have:

1. one canonical forward equation,
2. one resource-set interface,
3. one diagnostics object,
4. one proposal operator,
5. one certificate boundary,
6. theorem claims stated only at the certificate boundary.

The implementation may expose multiple certified modes, but they must all be
explicitly named and auditable.
