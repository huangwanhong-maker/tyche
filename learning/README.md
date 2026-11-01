# learning: Statistical Learning by Gradients

*Status: draft (2026-11-01): feed-forward networks of dense and one-dimensional convolutional layers, rectifiers, pooling and dropout; backpropagation checked against central differences; Adam, AdamW and momentum; minibatch training with early stopping, deterministic on any number of threads ([decision 0008](../development/decisions/0008-the-learning-module.md)).* Kernels, trees and ensembles, and the other parts of statistical learning the README's table names are planned.

## 1. Concepts

A network is a composition f = f_K ∘ … ∘ f_1 of layers, each a map with parameters, its parameters together one vector θ. Learning minimises the mean loss of the training examples over θ by stochastic gradient steps; the gradient comes from the chain rule applied backwards (reverse-mode differentiation), each layer contributing a vector–Jacobian product, so that the whole gradient costs a small multiple of one forward pass. Nothing here names an object in nature: a convolution slides a filter along any sequence; the genomics domain reads its filters as motifs.

Three properties are held to: the gradients are those of the loss (checked against central differences in double); the optimisers are those their papers write (Adam's first step α sign g, AdamW's decoupled decay); training is the same, bit for bit, on one thread or eight, because each batch is cut into a fixed number of shards whose gradients are summed in order.

## 2. Content

| Header | Content |
|---|---|
| `layers.hpp` | `Shape{channels, length}`, `Pass{training, seed, epoch, example, layer}`, `Layer<T>` (forward; backward adding ∂L/∂θ and writing ∂L/∂x); `Dense<T>(outputs, Init::he/glorot)`, `Conv1d<T>(filters, width, Padding::valid/same)`, `Relu<T>`, `Tanh<T>`, `MaxPool1d<T>(window, stride)` (window 0: global), `Dropout<T>(p)` (inverted, its mask a calx stream keyed by seed, layer, epoch, example) |
| `network.hpp` | `Loss{squared, huber(δ), poisson}`; `Network<T>(input)` with `add<L>(…)`, `initialise(seed)`, `theta`, `forward`, `accumulate` (loss and gradient of one example), `loss_of`, `slice(k)`, `describe()`; `numerical_gradient` (central differences) |
| `optimise.hpp` | `Adam(n, AdamOptions{rate, beta1, beta2, epsilon, weight_decay})` (AdamW when weight_decay > 0), `Momentum(n, rate, μ)` |
| `train.hpp` | `Data<T>{input, outputs, x, y}` with `subset`; `split(n, validation, test, seed)`; `TrainOptions{epochs, batch, adam, loss, patience, restore_best, shards, seed}`; `train(net, train, validation, options, pool)` → `Training{curve, best_epoch, best_validation, steps}`; `mean_loss`, `predict` |

T is `float` or `double`; moments of the optimisers are kept in double.

## 3. Checks

`check_tyche_learning` (17): backpropagation against central differences, relative error below 10⁻⁷, for dense → tanh → dense under squared error, Huber's loss and the Poisson deviance; for convolutions (valid and same) → rectifier → max pooling with stride → convolution → global max pooling → dense; for dropout in training (its mask drawn again backwards); dropout the identity in prediction; a same-padded convolution by hand; Adam's first step α sign g, AdamW's decay with a zero gradient, Adam on a quadratic; a planted linear model recovered; a planted 6-mer learnt by a small CNN on one-hot sequences, its best filter spelling the word; early stopping; the same parameters bit for bit on one thread and four. `check_tyche_rank` (7) belongs to `descriptive` (ranks, Spearman's ρ, Kendall's τ_b, added with this module).

## 4. Dependencies

calx `random` (initialisation, shuffling, dropout), `scheduler` (shards on the pool).

## 5. References

- Rumelhart, D. E., Hinton, G. E. and Williams, R. J. (1986), Learning representations by back-propagating errors, *Nature* 323, 533–536.
- Griewank, A. and Walther, A. (2008), *Evaluating Derivatives*, 2nd ed., SIAM.
- Kingma, D. P. and Ba, J. (2015), Adam: a method for stochastic optimization, *Proc. ICLR*.
- Loshchilov, I. and Hutter, F. (2019), Decoupled weight decay regularization, *Proc. ICLR*.
- He, K., Zhang, X., Ren, S. and Sun, J. (2015), Delving deep into rectifiers, *Proc. ICCV*; Glorot, X. and Bengio, Y. (2010), *Proc. AISTATS*.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R. (2014), Dropout, *JMLR* 15, 1929–1958.
- Prechelt, L. (1998), Early stopping — but when?, in *Neural Networks: Tricks of the Trade*, LNCS 1524, 55–69.
- Goodfellow, I., Bengio, Y. and Courville, A. (2016), *Deep Learning*, MIT Press.

## 6. Open Questions

1. **Speed.** The layers are loops the compiler vectorises; a convolution as a matrix product on calx's lanes, and on other devices through calx's compute stack, would be much faster.
2. **More layers.** Batch normalisation, recurrent and attention layers, dilated convolutions (Basenji, Enformer) are not here.
3. **calx's jets** give forward-mode derivatives; reverse mode here is written per layer. A tape-based reverse mode in calx would let any composition be differentiated.
