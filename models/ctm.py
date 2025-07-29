# implements the ctm

# initial version based on https://github.com/SakanaAI/continuous-thought-machines/blob/main/examples/01_mnist.ipynb
# in the future this will be a more general implementation

import jax
import jax.numpy as jnp

from flax import nnx

import math

# jit?

class Squeeze(nnx.Module):
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, x):
        return jax.lax.squeeze(x, (self.axis,))

# class NeuronModel(nnx.Module):
#     """a single neuron's sub-model: Linear(M -> H)."""
#     def __init__(self, in_dims, out_dims, *, rngs: nnx.Rngs):
#         k = 1.0 / math.sqrt(in_dims + out_dims)
#         self.w = nnx.Param(
#             jax.random.uniform(rngs.params(), (in_dims, out_dims),
#                                minval=-k, maxval=+k)
#         )
#         self.b = nnx.Param(jnp.zeros((out_dims,)))

#     def __call__(self, x):
#         # x: (M,)
#         print(f"x: {x.shape}")
#         return x @ self.w.value + self.b.value

class SuperLinear(nnx.Module):
    """apply independent NeuronModels to each of D neurons in parallel."""
    def __init__(self, in_dims, out_dims, D, *, rngs: nnx.Rngs):
        self.D = D
        # Create batched parameters instead of a list of individual neurons
        k = 1.0 / math.sqrt(in_dims + out_dims)
        self.w = nnx.Param(
            jax.random.uniform(rngs.params(), (D, in_dims, out_dims),
                               minval=-k, maxval=+k)
        )
        self.b = nnx.Param(jnp.zeros((D, out_dims)))

    def __call__(self, x):
        # x is (D, M). Apply batched linear transformation
        # x: (D, M), w: (D, M, H), b: (D, H) -> output: (D, H)
        return jnp.einsum('dm,dmh->dh', x, self.w.value) + self.b.value

# computes the normalized entropy for certainty-loss
def compute_normalized_entropy(logits):
    preds = jax.nn.softmax(logits, axis=-1)
    log_preds = jax.nn.log_softmax(logits, axis=-1)
    entropy = -jnp.sum(preds * log_preds, axis=-1)
    num_classes = preds.shape[-1]
    max_entropy = jnp.log(jnp.array(num_classes))
    normalized_entropy = entropy / max_entropy
    if len(logits.shape)>2:
        normalized_entropy = normalized_entropy.mean(axis=-1)
    return normalized_entropy

class CTM(nnx.Module):
    def __init__(self, config, rngs):
        super().__init__()
        self.config = config
        self.rngs = rngs

        self.iterations = config["iterations"] # number of internal ticks
        self.d_model = config["d_model"] # total number of neurons
        self.d_input = config["d_input"] # input and attention embed dimensions
        self.memory_length = config["memory_length"] # length of the sliding window used by each neuron
        self.memory_hidden_dims = config["memory_hidden_dims"]
        self.heads = config["heads"] # number of attention heads
        self.n_synch_out = config["n_synch_out"] # number of neurons used for output synchronization
        self.n_synch_action = config["n_synch_action"] # number of neurons used for computing attention queries
        self.out_dims = config["out_dims"] # dimensionality of the model's output

        # --- input processing ---
        # simple mlp backbone
        self.backbone = nnx.Sequential(
            nnx.Linear(self.d_input, self.d_input, rngs=rngs),
            nnx.relu,
            nnx.Linear(self.d_input, self.d_input, rngs=rngs),
            nnx.relu,
        )

        self.kv_proj = nnx.Sequential(
            nnx.Linear(self.d_input, self.d_input, rngs=rngs),
            # nnx.LayerNorm(self.d_input, rngs=rngs),
        )
        self.sync_representation_size_action = (self.n_synch_action * (self.n_synch_action + 1)) // 2
        self.q_proj = nnx.Linear(self.sync_representation_size_action, self.d_input, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=self.heads,
            in_features=self.d_input,
            kernel_init=nnx.initializers.lecun_normal(),
            decode=False,
            rngs=rngs,
        )

        # --- core ctm modules ---
        # simple synapse_depth = 1. other option is the U-NET in the paper
        self.synapses = nnx.Sequential(
            nnx.Linear(self.d_input + self.d_model, 2 * self.d_model, rngs=rngs),
            nnx.glu,
            nnx.Linear(self.d_model, self.d_model, rngs=rngs),
        )

        # the neuron level models
        self.trace_processor = nnx.Sequential(
            SuperLinear(self.memory_length, 2 * self.memory_hidden_dims, self.d_model, rngs=rngs),
            nnx.glu,
            SuperLinear(self.memory_hidden_dims, 2, self.d_model, rngs=rngs),
            nnx.glu,
            Squeeze(-1),
        )

        # --- starting states ---
        self.start_activated_state = nnx.Param(jax.random.uniform(rngs.params(), (self.d_model,), minval=-math.sqrt(1/self.d_model), maxval=math.sqrt(1/self.d_model)))
        self.start_trace = nnx.Param(jax.random.uniform(rngs.params(), (self.d_model, self.memory_length), minval=-math.sqrt(1/(self.d_model+self.memory_length)), maxval=math.sqrt(1/(self.d_model+self.memory_length))))

        # --- synchronization ---
        self.sync_representation_size_action = (self.n_synch_action * (self.n_synch_action + 1)) // 2
        self.sync_representation_size_out = (self.n_synch_out * (self.n_synch_out + 1)) // 2

        # for sync_type, size in [('action', self.sync_representation_size_action), ('out', self.sync_representation_size_out)]:
        #     print(f"Sync representation size {sync_type}: {size}")
        
        self.decay_params_action = nnx.Param(jnp.zeros((self.sync_representation_size_action,)))
        self.decay_params_out = nnx.Param(jnp.zeros((self.sync_representation_size_out,)))

        # --- output processing ---
        self.output_proj = nnx.Linear(self.sync_representation_size_out, self.out_dims, rngs=rngs)

    def compute_synchronization(self, activated_state, decay_alpha, decay_beta, r, sync_type):
        if sync_type == "action":
            n_synch = self.n_synch_action
            selected_left = selected_right = activated_state[-n_synch:]
        elif sync_type == "out":
            n_synch = self.n_synch_out
            selected_left = selected_right = activated_state[:n_synch]
        else:
            raise ValueError(f"unknown sync_type {sync_type}")

        # outer product: (n, 1) * (1, n) → (n, n)
        outer = jnp.expand_dims(selected_left, 1) * jnp.expand_dims(selected_right, 0)

        # get upper triangular indices
        i, j = jnp.triu_indices(n_synch, k=0)
        pairwise_product = outer[i, j]

        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = jnp.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1.0

        synchronisation = decay_alpha / jnp.sqrt(decay_beta)
        return synchronisation, decay_alpha, decay_beta

    def compute_features(self, x):
        input_features = self.backbone(x)
        kv = self.kv_proj(input_features)
        return kv
    
    def compute_certainty(self, current_prediction):
        ne = compute_normalized_entropy(current_prediction)
        current_certainty = jnp.stack([ne, 1 - ne], axis=-1)
        return current_certainty

    def __call__(self, x):
        kv = self.compute_features(x)

        # get state_trace to shape (H, T)
        state_trace = self.start_trace.reshape(self.d_model, self.memory_length)
        activated_state = self.start_activated_state.value # (d_model,)

        predictions = jnp.zeros((self.out_dims, self.iterations))
        certainties = jnp.zeros((2, self.iterations))

        decay_alpha_action, decay_beta_action = jnp.zeros((self.sync_representation_size_action,)), jnp.zeros((self.sync_representation_size_action,))
        r_action, r_out = jnp.exp(-self.decay_params_action.value), jnp.exp(-self.decay_params_out.value)

        _, decay_alpha_out, decay_beta_out = self.compute_synchronization(activated_state, None, None, r_out, sync_type='out')
    
        # loop (jit friendly)
        def loop_body(loop_state, i):
            activated_state, state_trace, predictions, certainties, decay_alpha_action, decay_beta_action, decay_alpha_out, decay_beta_out = loop_state

            synchronization_action, decay_alpha_action, decay_beta_action = self.compute_synchronization(activated_state, decay_alpha_action, decay_beta_action, r_action, sync_type='action')
            
            q = self.q_proj(synchronization_action)
            q = jnp.expand_dims(q, 0)
            attn_out = self.attention(q, kv) # (1, input_dim)

            attn_out = jnp.squeeze(attn_out, axis=0) # (input_dim,)

            pre_synapse_input = jnp.concatenate([attn_out, activated_state], axis=-1)

            state = self.synapses(pre_synapse_input)
            # Add a dimension to state to concatenate with state_trace
            state_trace = jnp.concatenate((state_trace[:,1:], state[:, None]), axis=1)

            activated_state = self.trace_processor(state_trace)

            synchronization_out, decay_alpha_out, decay_beta_out = self.compute_synchronization(activated_state, decay_alpha_out, decay_beta_out, r_out, sync_type='out')

            current_prediction = self.output_proj(synchronization_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions = predictions.at[:, i].set(current_prediction)
            certainties = certainties.at[:, i].set(current_certainty)

            new_loop_state = (activated_state, state_trace, predictions, certainties, decay_alpha_action, decay_beta_action, decay_alpha_out, decay_beta_out)
            return new_loop_state, ()
        
        init_state = (activated_state, state_trace, predictions, certainties, decay_alpha_action, decay_beta_action, decay_alpha_out, decay_beta_out)
        final_state, _ = jax.lax.scan(loop_body, init_state, jnp.arange(self.iterations))

        # final predictions are the 3rd element in the state tuple
        final_predictions = final_state[2]
        # we want the last prediction in the sequence
        return final_predictions[:, -1]
        
