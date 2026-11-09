from core.imports import *
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
from flax.training.train_state import TrainState


class PQN_CNN(nn.Module):
    norm_type: str
    final_hidden_dim: int 

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # 1. Setup normalization
        if self.norm_type == "layer_norm":
            normalize = lambda tensor: nn.LayerNorm()(tensor)
        else:
            normalize = lambda tensor: tensor

        assert x.ndim >= 3, f"Input shape should have at least 3 dimensions (H, W, C), got {x.shape}"
        
        # 2. Extract leading batch dimensions generically (CNNTorso style)
        if x.ndim == 3:
            batch_dims = (1,)
            x = x[None, ...]
        elif x.ndim == 4:
            batch_dims = (x.shape[0],)
        else:
            batch_dims = x.shape[:-3]
            x = x.reshape(-1, *x.shape[-3:])

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            strides=1,
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)

        # 4. Flatten using the dimension-agnostic torso style
        x = x.reshape(*batch_dims, -1)

        # 5. Final projection
        x = nn.Dense(
            self.final_hidden_dim, 
            kernel_init=orthogonal(jnp.sqrt(2)), 
            bias_init=constant(0.0)
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        
        return x

class MLP(nn.Module):
    norm_type: str
    final_hidden_dim: int 
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.norm_type == "layer_norm":
            normalize = lambda tensor: nn.LayerNorm()(tensor)
        else:
            normalize = lambda tensor: tensor
        
        assert x.ndim >= 1, f"Input shape should have at least 1 dimension (D), got {x.shape}"
        
        if x.ndim == 1: # add a batch dim.
            x = x[None, ...]

        x = nn.Dense(
            self.hidden_dim, 
            kernel_init=orthogonal(jnp.sqrt(2)), 
            bias_init=constant(0.0)
        )(x)
        x = normalize(x)
        x = nn.relu(x)

        x = nn.Dense(
            self.final_hidden_dim, 
            kernel_init=orthogonal(jnp.sqrt(2)), 
            bias_init=constant(0.0)
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        
        return x

class PQN(nn.Module):
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = PQN_CNN(self.norm_type, self.final_hidden_dim)(x)
        x = nn.Dense(self.action_dim, kernel_init=nn.initializers.zeros, bias_init = nn.initializers.zeros)(x)
        return x

class PolicyHead(nn.Module):
    action_dim: int
    is_continuous: bool = False

    @nn.compact
    def __call__(self, x):
        if not self.is_continuous:
            # Discrete: Output Logits
            logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
            return distrax.Categorical(logits=logits)
        else:
            # Continuous: Output Mean and Log Std
            loc = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
            log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            return distrax.MultivariateNormalDiag(loc=loc, scale_diag=jnp.exp(log_std))


class PQN_AC(nn.Module):
    """Actor critic with seperate small CNNs for the actor and critic"""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    is_continuous: bool = False
    
    def setup(self):
        self.actor_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim = self.final_hidden_dim)
        self.critic_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim = self.final_hidden_dim)
        self.actor_head = PolicyHead(self.action_dim, self.is_continuous)
        self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init = constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns (pi(s), piV(s))"""
        v = self.value(x)
        pi = self.policy(x)
        return pi, v
    
    def policy(self, x: jnp.ndarray):
        """Returns pi(s)"""
        actor_features = self.actor_cnn(x)
        return self.actor_head(actor_features)

    def value_features(self, x: jnp.ndarray):
        """Returns V(s)"""
        critic_features = self.critic_cnn(x)
        return critic_features
    
    def value_from_features(self, phi):
        return self.critic_head(phi).squeeze(-1)

    def value(self, x: jnp.ndarray):
        """Returns V(s)"""
        critic_features = self.value_features(x)
        return self.value_from_features(critic_features)
    
    def act(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        """Samples an action from the policy."""
        policy = self.policy(x)
        action = policy.sample(seed=key)
        return action.squeeze()

class PQN_Critic(nn.Module):
    """Actor critic with seperate small CNNs for the actor and critic"""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    
    def setup(self):
        self.critic_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim = self.final_hidden_dim)
        self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init = constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns V(s)"""
        v = self.value(x)
        return v
    
    def value_features(self, x: jnp.ndarray):
        critic_features = self.critic_cnn(x)
        return critic_features
    
    def value_from_features(self, phi):
        return self.critic_head(phi).squeeze(-1)

    def value(self, x: jnp.ndarray):
        """Returns V(s)"""
        return self.value_from_features(self.value_features(x))

class MLP_AC(nn.Module):
    """Actor critic with seperate small MLPs for the actor and critic"""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    is_continuous: bool = False
    
    def setup(self):
        self.actor_mlp = MLP(norm_type=self.norm_type, final_hidden_dim = self.final_hidden_dim)
        self.critic_mlp = MLP(norm_type=self.norm_type, final_hidden_dim = self.final_hidden_dim)
        self.actor_head = PolicyHead(self.action_dim, self.is_continuous)
        self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init = constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns (pi(s), piV(s))"""
        v = self.value(x)
        pi = self.policy(x)
        return pi, v
    
    def policy(self, x: jnp.ndarray):
        """Returns pi(s)"""
        actor_features = self.actor_mlp(x)
        return self.actor_head(actor_features)

    def value_features(self, x: jnp.ndarray):
        """Returns features for V(s)"""
        critic_features = self.critic_mlp(x)
        return critic_features
    
    def value_from_features(self, phi):
        return self.critic_head(phi).squeeze(-1)

    def value(self, x: jnp.ndarray):
        """Returns V(s)"""
        critic_features = self.value_features(x)
        return self.value_from_features(critic_features)
    
    def act(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        """Samples an action from the policy."""
        policy = self.policy(x)
        action = policy.sample(seed=key)
        return action.squeeze()

class MLP_Critic(nn.Module):
    """Critic only with a small MLP"""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    
    def setup(self):
        self.critic_mlp = MLP(norm_type=self.norm_type, final_hidden_dim = self.final_hidden_dim)
        self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init = constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns V(s)"""
        v = self.value(x)
        return v
    
    def value_features(self, x: jnp.ndarray):
        critic_features = self.critic_mlp(x)
        return critic_features
    
    def value_from_features(self, phi):
        return self.critic_head(phi).squeeze(-1)

    def value(self, x: jnp.ndarray):
        """Returns V(s)"""
        return self.value_from_features(self.value_features(x))

class PQN_DualCritic(nn.Module):
    """Critic with a CNN torso and two value heads: main value (v) and auxiliary value (v_aux)."""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    shared_torso: bool = True

    def setup(self):
        self.critic_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        if not self.shared_torso:
            self.aux_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))
        self.aux_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns main V(s) by default for evaluator compatibility, ensuring both heads/torsos are traced."""
        v, _ = self.both(x)
        return v

    def value_features(self, x: jnp.ndarray):
        return self.critic_cnn(x)

    def aux_features(self, x: jnp.ndarray):
        if self.shared_torso:
            return self.critic_cnn(x)
        return self.aux_cnn(x)

    def value_from_features(self, phi: jnp.ndarray):
        return self.critic_head(phi).squeeze(-1)

    def aux_from_features(self, phi: jnp.ndarray):
        return self.aux_head(phi).squeeze(-1)

    def value(self, x: jnp.ndarray):
        """Returns main V(s)."""
        return self.value_from_features(self.value_features(x))

    def v_aux(self, x: jnp.ndarray):
        """Returns auxiliary V_aux(s)."""
        return self.aux_from_features(self.aux_features(x))

    def both(self, x: jnp.ndarray):
        """Returns (V(s), V_aux(s)). Single forward pass if shared_torso=True."""
        if self.shared_torso:
            phi = self.value_features(x)
            return self.value_from_features(phi), self.aux_from_features(phi)
        else:
            phi_c = self.value_features(x)
            phi_a = self.aux_features(x)
            return self.value_from_features(phi_c), self.aux_from_features(phi_a)


class MLP_DualCritic(nn.Module):
    """Critic with an MLP torso and two value heads: main value (v) and auxiliary value (v_aux)."""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    shared_torso: bool = True

    def setup(self):
        self.critic_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        if not self.shared_torso:
            self.aux_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))
        self.aux_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns main V(s) by default for evaluator compatibility, ensuring both heads/torsos are traced."""
        v, _ = self.both(x)
        return v

    def value_features(self, x: jnp.ndarray):
        return self.critic_mlp(x)

    def aux_features(self, x: jnp.ndarray):
        if self.shared_torso:
            return self.critic_mlp(x)
        return self.aux_mlp(x)

    def value_from_features(self, phi: jnp.ndarray):
        return self.critic_head(phi).squeeze(-1)

    def aux_from_features(self, phi: jnp.ndarray):
        return self.aux_head(phi).squeeze(-1)

    def value(self, x: jnp.ndarray):
        """Returns main V(s)."""
        return self.value_from_features(self.value_features(x))

    def v_aux(self, x: jnp.ndarray):
        """Returns auxiliary V_aux(s)."""
        return self.aux_from_features(self.aux_features(x))

    def both(self, x: jnp.ndarray):
        """Returns (V(s), V_aux(s)). Single forward pass if shared_torso=True."""
        if self.shared_torso:
            phi = self.value_features(x)
            return self.value_from_features(phi), self.aux_from_features(phi)
        else:
            phi_c = self.value_features(x)
            phi_a = self.aux_features(x)
            return self.value_from_features(phi_c), self.aux_from_features(phi_a)


class PQN_AC_DualCritic(nn.Module):
    """Actor critic with separate small CNNs for actor and critic, with dual heads (v and v_aux) on critic."""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    is_continuous: bool = False
    shared_torso: bool = True

    def setup(self):
        self.actor_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        if not self.shared_torso:
            self.aux_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = PolicyHead(self.action_dim, self.is_continuous)
        self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))
        self.aux_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns (pi(s), V(s)) by default for standard AC interface, ensuring both critic heads are traced."""
        v, _ = self.both(x)
        pi = self.policy(x)
        return pi, v

    def policy(self, x: jnp.ndarray):
        actor_features = self.actor_cnn(x)
        return self.actor_head(actor_features)

    def value_features(self, x: jnp.ndarray):
        return self.critic_cnn(x)

    def aux_features(self, x: jnp.ndarray):
        if self.shared_torso:
            return self.critic_cnn(x)
        return self.aux_cnn(x)

    def value_from_features(self, phi):
        return self.critic_head(phi).squeeze(-1)

    def aux_from_features(self, phi):
        return self.aux_head(phi).squeeze(-1)

    def value(self, x: jnp.ndarray):
        return self.value_from_features(self.value_features(x))

    def v_aux(self, x: jnp.ndarray):
        return self.aux_from_features(self.aux_features(x))

    def both(self, x: jnp.ndarray):
        if self.shared_torso:
            phi = self.value_features(x)
            return self.value_from_features(phi), self.aux_from_features(phi)
        else:
            phi_c = self.value_features(x)
            phi_a = self.aux_features(x)
            return self.value_from_features(phi_c), self.aux_from_features(phi_a)

    def act(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        policy = self.policy(x)
        action = policy.sample(seed=key)
        return action.squeeze()


class MLP_AC_DualCritic(nn.Module):
    """Actor critic with separate small MLPs for actor and critic, with dual heads (v and v_aux) on critic."""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    is_continuous: bool = False
    shared_torso: bool = True

    def setup(self):
        self.actor_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        if not self.shared_torso:
            self.aux_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = PolicyHead(self.action_dim, self.is_continuous)
        self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))
        self.aux_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns (pi(s), V(s)) by default for standard AC interface, ensuring both critic heads are traced."""
        v, _ = self.both(x)
        pi = self.policy(x)
        return pi, v

    def policy(self, x: jnp.ndarray):
        actor_features = self.actor_mlp(x)
        return self.actor_head(actor_features)

    def value_features(self, x: jnp.ndarray):
        return self.critic_mlp(x)

    def aux_features(self, x: jnp.ndarray):
        if self.shared_torso:
            return self.critic_mlp(x)
        return self.aux_mlp(x)

    def value_from_features(self, phi):
        return self.critic_head(phi).squeeze(-1)

    def aux_from_features(self, phi):
        return self.aux_head(phi).squeeze(-1)

    def value(self, x: jnp.ndarray):
        return self.value_from_features(self.value_features(x))

    def v_aux(self, x: jnp.ndarray):
        return self.aux_from_features(self.aux_features(x))

    def both(self, x: jnp.ndarray):
        if self.shared_torso:
            phi = self.value_features(x)
            return self.value_from_features(phi), self.aux_from_features(phi)
        else:
            phi_c = self.value_features(x)
            phi_a = self.aux_features(x)
            return self.value_from_features(phi_c), self.aux_from_features(phi_a)

    def act(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        policy = self.policy(x)
        action = policy.sample(seed=key)
        return action.squeeze()



def make_warmup_linear_schedule(init_lr, end_lr, total_steps, warmup_ratio=0.1):
    """Linear warmup from 0.0 to init_lr over warmup_steps, followed by linear decay to end_lr."""
    if warmup_ratio is None or warmup_ratio <= 0.0 or total_steps <= 1:
        return optax.linear_schedule(
            init_value=init_lr, end_value=end_lr, transition_steps=total_steps
        )
    warmup_steps = int(total_steps * warmup_ratio)
    decay_steps = max(total_steps - warmup_steps, 1)
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=init_lr, transition_steps=warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=init_lr, end_value=end_lr, transition_steps=decay_steps
    )
    return optax.join_schedules([warmup_fn, decay_fn], [warmup_steps])

import flax.struct as struct

class ActorCriticTrainState(TrainState):
    actor_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    actor_opt_state: optax.OptState = struct.field(pytree_node=True)
    critic_opt_state: optax.OptState = struct.field(pytree_node=True)

    def apply_actor_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.actor_tx.update(grads, self.actor_opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, actor_opt_state=new_opt_state, **kwargs)

    def apply_critic_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.critic_tx.update(grads, self.critic_opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, critic_opt_state=new_opt_state, **kwargs)

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **kwargs)

def initialize_flax_train_state(config, network, params):
    # --- PPO Agent Scheduler & Optimizer ---
    actor_epochs = config.get("NUM_EPOCHS_ACTOR", config.get("NUM_EPOCHS", 4))
    critic_epochs = config.get("NUM_EPOCHS_CRITIC", config.get("NUM_EPOCHS", 4))
    total_actor_steps = config["NUM_UPDATES"] * config.get("NUM_MINIBATCHES", 1) * actor_epochs
    total_critic_steps = config["NUM_UPDATES"] * config.get("NUM_MINIBATCHES", 1) * critic_epochs
    total_grad_steps = config["NUM_UPDATES"] * config.get("NUM_MINIBATCHES", 1) * config.get("NUM_EPOCHS", 4)

    if config.get('OPTIMIZER','AdamW')=='AdamW':
        # Separate learning rates for actor and critic
        actor_lr = config.get("ACTOR_LR", config["LR"])
        critic_lr = config.get("CRITIC_LR", config["LR"]) # Let CRITIC_LR or LR dictate the value net LR
        actor_lr_end = config.get("ACTOR_LR_END")
        if actor_lr_end is None:
            actor_lr_end = actor_lr
        critic_lr_end = config.get("CRITIC_LR_END", config.get("LR_END"))
        if critic_lr_end is None:
            critic_lr_end = critic_lr

        warmup_ratio = config.get("WARMUP_RATIO", 0.1)

        actor_lr_scheduler = make_warmup_linear_schedule(
            init_lr=actor_lr,
            end_lr=actor_lr_end,
            total_steps=total_actor_steps,
            warmup_ratio=warmup_ratio,
        )
        critic_lr_scheduler = make_warmup_linear_schedule(
            init_lr=critic_lr,
            end_lr=critic_lr_end,
            total_steps=total_critic_steps,
            warmup_ratio=warmup_ratio,
        )

        actor_tx = optax.chain(
            optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 1.0)),
            optax.adamw(actor_lr_scheduler, 
                        weight_decay=config.get('WEIGHT_DECAY', 1e-2),
                        eps=config.get('ADAM_EPS', 1e-5)),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 1.0)),
            optax.adamw(critic_lr_scheduler, 
                        weight_decay=config.get('WEIGHT_DECAY', 1e-2),
                        eps=config.get('ADAM_EPS', 1e-5)),
        )

        def param_labels_actor(path, val):
            return any('actor' in getattr(p, 'key', '') or 'actor' in str(p) for p in path)

        def param_labels(path, val):
            is_actor = param_labels_actor(path, val)
            return 'actor' if is_actor else 'critic'

        actor_mask = jax.tree_util.tree_map_with_path(param_labels_actor, params)
        critic_mask = jax.tree_util.tree_map(lambda x: not x, actor_mask)

        actor_tx_masked = optax.masked(actor_tx, actor_mask)
        critic_tx_masked = optax.masked(critic_tx, critic_mask)

        tx = optax.multi_transform(
            {'actor': actor_tx, 'critic': critic_tx},
            jax.tree_util.tree_map_with_path(param_labels, params)
        )

    train_state = ActorCriticTrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
        actor_tx=actor_tx_masked,
        critic_tx=critic_tx_masked,
        actor_opt_state=actor_tx_masked.init(params),
        critic_opt_state=critic_tx_masked.init(params),
    )
    return train_state

def initialize_flax_train_state_no_w(config, network, params):
    "Final critic weights are excluded from the optimizer."
    # --- PPO Agent Scheduler & Optimizer ---
    total_grad_steps = config["NUM_UPDATES"] * config.get("NUM_MINIBATCHES", 1) * config["NUM_EPOCHS"]

    actor_lr = config.get("ACTOR_LR", config["LR"])
    critic_lr = config["LR"]
    actor_lr_end = config.get("ACTOR_LR_END")
    if actor_lr_end is None:
        actor_lr_end = actor_lr
    critic_lr_end = config.get("LR_END")
    if critic_lr_end is None:
        critic_lr_end = critic_lr

    warmup_ratio = config.get("WARMUP_RATIO", 0.1)

    actor_lr_scheduler = make_warmup_linear_schedule(
        init_lr=actor_lr,
        end_lr=actor_lr_end,
        total_steps=total_grad_steps,
        warmup_ratio=warmup_ratio,
    )
    critic_lr_scheduler = make_warmup_linear_schedule(
        init_lr=critic_lr,
        end_lr=critic_lr_end,
        total_steps=total_grad_steps,
        warmup_ratio=warmup_ratio,
    )

    actor_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(actor_lr_scheduler, 
            weight_decay = config.get('WEIGHT_DECAY', 1e-2),
            eps=config.get('ADAM_EPS', 1e-5)
            ),
    )
    critic_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(critic_lr_scheduler, 
            weight_decay = config.get('WEIGHT_DECAY', 1e-2),
            eps=config.get('ADAM_EPS', 1e-5)
            ),
    )
    zero_tx = optax.set_to_zero()

    def param_labels(path, val):
        is_w = any(getattr(p, 'key', None) in ('w_layer', 'critic_head') or 
                    'w_layer' in str(p) or 'critic_head' in str(p) for p in path)
        if is_w:
            return 'zero'
        is_actor = any('actor' in getattr(p, 'key', '') or 'actor' in str(p) for p in path)
        return 'actor' if is_actor else 'critic'

    tx = optax.multi_transform(
        {'actor': actor_tx, 'critic': critic_tx, 'zero': zero_tx},
        jax.tree_util.tree_map_with_path(param_labels, params)
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )
    return train_state

def initialize_network(rng, obs_shape, env, env_params, k, n_heads: int, layer_norm: bool):
    # Detect if continuous
    try:
        action_dim = env.action_space(env_params).n
        is_continuous = False 
    except:
        is_continuous = True 
        action_dim = env.action_space(env_params).shape[0]
    
    norm_type = 'layer_norm' if layer_norm else 'None'
    use_mlp = len(obs_shape) < 3
    
    print('- obs shape for initialization is ', obs_shape)

    if n_heads == 2:
        if use_mlp:
            model = MLP_AC(action_dim=action_dim, is_continuous=is_continuous, final_hidden_dim=k, norm_type=norm_type)
        else:
            model = PQN_AC(action_dim=action_dim, is_continuous=is_continuous, final_hidden_dim=k, norm_type=norm_type)
    elif n_heads == 1:
        if use_mlp:
            model = MLP_Critic(action_dim=action_dim, final_hidden_dim=k, norm_type=norm_type)
        else:
            model = PQN_Critic(action_dim=action_dim, final_hidden_dim=k, norm_type=norm_type)

    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros(obs_shape))
    print('number of features is ', model.final_hidden_dim)
    return model, params

def initialize_dual_critic_network(rng, obs_shape, env, env_params, k, layer_norm: bool, shared_torso: bool = True):
    try:
        action_dim = env.action_space(env_params).n
    except:
        action_dim = env.action_space(env_params).shape[0]
    
    norm_type = 'layer_norm' if layer_norm else 'None'
    use_mlp = len(obs_shape) < 3
    
    print(f'- obs shape for dual critic initialization is {obs_shape} (shared_torso={shared_torso})')
    if use_mlp:
        model = MLP_DualCritic(action_dim=action_dim, final_hidden_dim=k, norm_type=norm_type, shared_torso=shared_torso)
    else:
        model = PQN_DualCritic(action_dim=action_dim, final_hidden_dim=k, norm_type=norm_type, shared_torso=shared_torso)

    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros(obs_shape))
    print('number of features is ', model.final_hidden_dim)
    return model, params

def initialize_ac_dual_critic_network(rng, obs_shape, env, env_params, k, layer_norm: bool, shared_torso: bool = True):
    try:
        action_dim = env.action_space(env_params).n
        is_continuous = False
    except:
        action_dim = env.action_space(env_params).shape[0]
        is_continuous = True

    norm_type = 'layer_norm' if layer_norm else 'None'
    use_mlp = len(obs_shape) < 3

    print(f'- obs shape for AC dual critic initialization is {obs_shape} (shared_torso={shared_torso})')
    if use_mlp:
        model = MLP_AC_DualCritic(action_dim=action_dim, is_continuous=is_continuous, final_hidden_dim=k, norm_type=norm_type, shared_torso=shared_torso)
    else:
        model = PQN_AC_DualCritic(action_dim=action_dim, is_continuous=is_continuous, final_hidden_dim=k, norm_type=norm_type, shared_torso=shared_torso)

    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros(obs_shape))
    print('number of features is ', model.final_hidden_dim)
    return model, params


