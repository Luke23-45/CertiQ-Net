# Architecture Principles

## 1. Dispatch Primitive

CertiQ models online assignment of arriving work to heterogeneous resources.
The primitive is a map from state to a distribution over resources, together
with an explicit certificate that constrains the resulting distribution.

The canonical structure is

\[
\text{dispatch} = \text{proposal} \;\; \text{followed by} \;\; \text{certification}.
\]

## 2. Architectural Decomposition

The architecture is decomposed into three mathematical layers.

### 2.1 Geometry Layer

The geometry layer maps state and capacity into resourcewise indices or costs.
Its role is to define the admissibility and comparison scale used by the
certificate.

### 2.2 Proposal Layer

The proposal layer produces a probability distribution over resources from the
observed state. It may depend on queue lengths, service rates, and resource
context.

### 2.3 Certificate Layer

The certificate layer maps a proposal into an admissible distribution. It is
part of the policy definition and not a post hoc regularizer.

## 3. Symmetry Requirement

Resource labels are not semantically meaningful. If \(\sigma\) is a permutation
of \(\{1,\dots,N\}\), the policy must satisfy

\[
\pi(\sigma x)=\sigma\pi(x).
\]

This requirement applies to the proposal map, the certificate map, and any
intermediate summary that is reused across resources.

## 4. Certificate Boundary

The final policy is the certified policy. A model is not certified merely
because its proposal is well trained or its loss is small.

The certificate boundary must therefore be explicit, state dependent, and
auditable.

## 5. Acceptable Model Form

An acceptable CertiQ model has the form

\[
\pi^\Theta(x)=\mathcal C_x\!\left(\mathcal A_\Theta(x),\mathcal B(x)\right),
\]

where \(\mathcal A_\Theta\) is a learned proposal, \(\mathcal B\) is a fixed
geometry or budget map, and \(\mathcal C_x\) is the certificate operator.
