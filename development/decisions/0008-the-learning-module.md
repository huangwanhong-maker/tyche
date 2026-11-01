# 0008. The Learning Module: Networks, Backpropagation and Adam; Rank Correlation

- **Status:** Accepted, provisionally (feed-forward networks of dense and one-dimensional convolutional layers only; layers written as loops)
- **Date:** 2026-11-01
- **Decided by:** the infrastructure's genomics domain (ADR 0064, expression predicted from sequence) learns promoter activity from sequence with convolutional networks and ranks codon usage indices against measured abundance. The machinery of learning names no object in nature, so it is tyche's. Claude wrote the module and this record in that round; decision 0006 (penalised regression) supplies the linear baseline the networks are compared with.

## 1. Context

The README planned `learning` ("kernels, trees and ensembles, neural networks; validation"). The genomics domain needs a network that reads one-hot sequences (a convolution whose filters act as position weight matrices), trained by a stochastic optimiser on measured activities, with held-out evaluation; and rank correlations (Spearman's ρ) to compare predictions with measurements. Both are general: the convolution slides along any sequence, and ρ compares any two samples.

## 2. Decision

### 2.1 `learning`

- **Layers** (`layers.hpp`): a layer is a forward map and a backward map (a vector–Jacobian product) of its input and its slice of the parameters; layers keep no state, so one network serves many threads. Dense (He's or Glorot's initialisation), Conv1d (valid or same padding; a cross-correlation, as deep learning means it), ReLU, Tanh, MaxPool1d (with stride; global), Dropout (inverted; its mask a calx stream keyed by seed, layer, epoch and example, so the backward pass draws it again).
- **Networks** (`network.hpp`): a sequence of layers, its parameters one flat vector; squared, Huber and Poisson losses; the gradient by backpropagation; `numerical_gradient` by central differences, for checks.
- **Optimisers** (`optimise.hpp`): Adam (Kingma and Ba's Algorithm 1), AdamW's decoupled decay (Loshchilov and Hutter's Algorithm 2, with PyTorch's parameterisation), momentum. Moments in double.
- **Training** (`train.hpp`): minibatches in an order drawn by a stream each epoch; each batch cut into a fixed number of shards whose gradients are summed in order, the shards run on calx's pool; early stopping on a validation set, the best epoch's parameters restored; splits by a stream. Templates on the scalar type: `double` for checks, `float` for speed.

### 2.2 `descriptive/rank.hpp`

Mid-ranks, Spearman's ρ (Pearson's correlation of mid-ranks), Kendall's τ_b by Knight's O(n log n) count; with `check_tyche_rank`.

## 3. Alternatives Considered

- **A borrowed framework** (LibTorch, ONNX Runtime). Rejected for now: large, and the infrastructure owns its interfaces (CLAUDE.md §2.2); a borrowed backend may sit behind these interfaces later.
- **Automatic differentiation in calx** (its open decision 4). Reverse mode written per layer is enough for feed-forward networks and is checked against differences; a tape in calx would generalise it and is left to calx.
- **The network in the genomics domain.** Rejected: nothing in it is genomic; the domain keeps one-hot encoding, filters read as motifs and the matching of motifs.

## 4. Consequences

- tyche gains its first learning module and `check_tyche_learning` (17 checks); `descriptive` gains `rank.hpp` and `check_tyche_rank` (7).
- The genomics domain's `regulation` module trains its networks here; its `codon` module ranks indices with ρ.
- Speed is that of compiler-vectorised loops; calx's compute stack is where faster convolutions belong.
