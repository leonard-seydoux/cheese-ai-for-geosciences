from typing import Tuple, List, Callable, Optional

import tqdm
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax

class MLP(nn.Module):
  """ MultiLayer Perceptron backend """
  n_blocks: int
  features: int
  out_features: int

  def setup(self):
    self.blocks = [nn.Dense(self.features) for _ in range(self.n_blocks)]
    self.out = nn.Dense(self.out_features)

  def __call__(self, x):
    for block in self.blocks:
      x = block(x)
      x = nn.tanh(x)
    x = self.out(x)
    return x

class CNN(nn.Module):
  """ Convolutional Neural Network backend """

def sample_range(rng, size: int, a: float, b: float) -> np.array:
  return (b - a) * rng.random(size) + a

def space_time_product(t: List[float], x: List[float]) -> np.array:
  t_s, x_s = np.meshgrid(np.array(t), np.array(x), indexing='ij')
  t_s = t_s.flatten()
  x_s = x_s.flatten()
  return (
    t_s,
    x_s
  )

# Physics informed (tutorial part 1)

class PhysicsInformed:
  def __init__(
    self,
    N_x: int,
    N_t: int,
    t_domain: Tuple[float, float],
    x_domain: Tuple[float, float],
  ):
    self.N_x = N_x
    self.N_t = N_t
    self.x_domain = x_domain
    self.t_domain = t_domain
    self.rng = np.random.default_rng()    

  def predict(
    self,
    t: List[float],
    x: List[float]
  ) -> jax.Array:
    t_s, x_s = space_time_product(t, x)
    pred = jnp.reshape(self.pred_map(self.state.params, t_s, x_s), (len(t), len(x)))
    return pred

  def residual(
    self,
    t: List[float],
    x: List[float],
    eq_fn: Callable[[Callable, float, float], float]
  ) -> float:
    t_s, x_s = space_time_product(t, x)
    f = jax.vmap(eq_fn, in_axes=(None, 0, 0))(jax.tree_util.Partial(self.pred_fn, self.state.params), t_s, x_s)
    return f

  def sample_space_time(
    self,
    N_t: Optional[int] = None,
    N_x: Optional[int] = None
  ) -> Tuple[jax.Array, jax.Array]:
    t = sample_range(self.rng, N_t if N_t is not None else self.N_t, self.t_domain[0], self.t_domain[-1])
    x = sample_range(self.rng, N_x if N_x is not None else self.N_x, self.x_domain[0], self.x_domain[-1])
    return (
      t, x
    )

  def train(
    self,
    backend: nn.Module,
    ic_fn: Callable[[float], float],
    bc_fn: Callable[[float, float], float],
    eq_fn: Callable[[Callable, float, float], float],
    learning_rate: float,
    epochs: int
  ):
    self.backend = backend
    va = self.backend.init(jax.random.PRNGKey(42), jnp.zeros(2))

    self.state = TrainState.create(params=va['params'],
      apply_fn=self.backend.apply, tx=optax.adam(learning_rate),
    )

    def __pred__(
      params,
      t: float,
      x: float
    ) -> float:
      ts_f = jnp.array([t, x])
      nn_u = self.state.apply_fn({'params': params}, jnp.expand_dims(ts_f, axis=0)).squeeze()
      return nn_u
    self.pred_fn = __pred__
    self.pred_map = jax.vmap(
      self.pred_fn, in_axes=(None, 0, 0)
    )

    @jax.jit
    def __loss__(params, x: jax.Array, t: jax.Array) -> float:
      t_s, x_s = jnp.meshgrid(t, x)
      t_s = t_s.flatten()
      x_s = x_s.flatten()

      # nn(0, x)
      pred_ic = self.pred_map(params, jnp.full_like(x, 0.0), x)
      # initial condition loss
      mis_ic = jax.vmap(ic_fn, in_axes=0)(pred_ic, x)
      l_ic = jnp.mean(mis_ic)

      # nn(t, a)
      pred_lbc = self.pred_map(params, t, jnp.full_like(t, self.x_domain[ 0]))
      # nn(t, b)
      pred_rbc = self.pred_map(params, t, jnp.full_like(t, self.x_domain[-1]))
      # boundary condition loss
      mis_bc = jax.vmap(bc_fn, in_axes=(0, 0))(pred_lbc, pred_rbc)
      l_bc = jnp.mean(mis_bc)
        
      # equation residual on sample space
      f = jax.vmap(eq_fn, in_axes=(None, 0, 0))(jax.tree_util.Partial(self.pred_fn, params), t_s, x_s)
      l_f = jnp.mean(jnp.square(f))
      # total loss
      return (
        l_ic + l_bc + l_f
      )

    epoch_loss = []
    epoch_bar = tqdm.tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for epoch in epoch_bar:
      t, x = self.sample_space_time(self.N_t, self.N_x)
      pinn_loss, grads = jax.value_and_grad(__loss__)(self.state.params, x, t)
      self.state = self.state.apply_gradients(grads=grads)
      epoch_loss.append(pinn_loss)
      epoch_bar.set_postfix(
        loss=pinn_loss,
      )

    fig, axs = plt.subplots()
    axs.semilogy(epoch_loss)
    axs.set_xlabel(r'Epoch')
    axs.set_ylabel(r'Loss $L_{\mathrm{IC}} + L_{\mathrm{BC}} + L_f$')
    plt.show()

# Hybrid physics (tutorial part 2)
