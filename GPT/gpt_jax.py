
import jax
import jax.numpy as jnp
from flax.jax_utils import prefetch_to_device
import flax.linen as nn
from flax.training import train_state
from flax.training import checkpoints
from flax import struct
import optax
import numpy as np
from typing import Optional, Tuple, Any
import math
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import wandb
import os
import re
from jax import config as jax_config
import jax
import jax.numpy as jnp

from flax.linen import tabulate
import jax



# Check TPU availability and setup
def check_tpu_setup():
    """Check if TPU is available and properly configured."""
    try:
        # Check if we're running on TPU
        if jax.devices('tpu'):
            print("✅ TPU detected!")
            devices = jax.devices('tpu')
            print(f"   TPU devices: {len(devices)}")
            for i, device in enumerate(devices):
                print(f"   Device {i}: {device}")

            # Print TPU-specific info
            print(f"   JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
            print(f"   Total TPU cores: {jax.device_count()}")
            print(f"   Local devices: {jax.local_device_count()}")

            return True
        else:
            print("❌ No TPU devices found")
            print(f"   Available devices: {jax.devices()}")
            return False

    except Exception as e:
        print(f"❌ Error checking TPU: {e}")
        return False

# Check current setup
print("Device Information:")
print(f"JAX version: {jax.__version__}")
print(f"Platform: {jax.lib.xla_bridge.get_backend().platform}")
print(f"Device count: {jax.device_count()}")

# Check TPU specifically
has_tpu = check_tpu_setup()

# Test a simple computation
test_array = jnp.array([1, 2, 3, 4, 5])
result = jnp.sum(test_array ** 2)
print(f"\nTest computation result: {result}")
print(f"Computation device: {result.device}")

print("jax.devices():", jax.devices())
print("jax.device_count():", jax.device_count())
print("jax.local_device_count():", jax.local_device_count())
print("jax.process_count():", jax.process_count())
print("jax.process_index():", jax.process_index())



os.environ.update({
  "NCCL_LL128_BUFFSIZE": "-2",
  "NCCL_LL_BUFFSIZE": "-2",
   "NCCL_PROTO": "SIMPLE,LL,LL128",
 })

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)


tokenizer = AutoTokenizer.from_pretrained("gpt2", token="")
tokenizer.pad_token = '[PAD]'

# Configuration class for model parameters
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = tokenizer.vocab_size + 47
    max_seq_len: int = 256
    d_model: int = 768
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 3072
    dropout_rate: float = 0.1
    lr: float = 6e-4
    min_lr: float = 0.1 * lr
    warmup_steps: int = 700
    total_steps: int = 20000
    batch_size: int = 256
    required_bsz_tokens: int = 524288
    gradient_accumulation_steps: int = int(required_bsz_tokens // (batch_size * max_seq_len))
    mixed_precision: bool = True
    num_epochs: int = 1
    eval_steps: int = 200

config = GPTConfig()


# Set matmul precision based on mixed_precision setting
if config.mixed_precision:
    jax_config.update("jax_default_matmul_precision", "bfloat16")
    print("Set JAX matmul precision to bfloat16 (mixed precision enabled)")
else:
    jax_config.update("jax_default_matmul_precision", "float32")
    print("Set JAX matmul precision to float32 (mixed precision disabled)")

# Helper function to get the appropriate dtype based on mixed_precision setting
def get_dtype():
    """Return bfloat16 if mixed_precision is True, else float32."""
    return jnp.bfloat16 if config.mixed_precision else jnp.float32

print(f"Using dtype: {get_dtype()} (mixed_precision={config.mixed_precision})")

def optimize_for_tpu():
    """Apply TPU-specific optimizations."""
    if not jax.devices('tpu'):
        print("Not running on TPU, skipping optimizations")
        return

    print("🚀 Applying TPU optimizations...")

    # Set XLA flags for better TPU performance
    import os
    os.environ.setdefault('XLA_FLAGS',
        '--xla_force_host_platform_device_count=1')

    # # Enable mixed precision if not already enabled
    if not config.mixed_precision:
        print("   Enabling mixed precision for TPU")
        config.mixed_precision = True

    print(" TPU optimizations applied!")

# Apply optimizations
# optimize_for_tpu()


from flax.training import train_state

class TrainState(train_state.TrainState):
    grad_accum: Any = None
    accum_step: int = 0

#Loading TinyStories dataset from Huggingface
train_dataset = load_dataset("roneneldan/TinyStories", split="train", token='')
val_dataset = load_dataset("roneneldan/TinyStories", split="validation", token='')

class Attention(nn.Module):
    d_model: int = config.d_model
    num_heads: int = config.num_heads
    dropout_rate: float = config.dropout_rate

    def setup(self):
        self.head_size = self.d_model // self.num_heads

        # Proper initialization for attention layers
        self.d_Q = nn.Dense(
            features=self.head_size,
            use_bias=False,
            dtype=get_dtype(),
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        self.d_K = nn.Dense(
            features=self.head_size,
            use_bias=False,
            dtype=get_dtype(),
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        self.d_V = nn.Dense(
            features=self.head_size,
            use_bias=False,
            dtype=get_dtype(),
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        self.d_O = nn.Dense(
            features=self.d_model,
            use_bias=False,
            dtype=get_dtype(),
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, training=True):
        B, T, C = x.shape
        query = self.d_Q(x)
        key = self.d_K(x)
        value = self.d_V(x)

        # Proper attention scaling using head_size
        weights = jnp.matmul(query, key.transpose(0, 2, 1)) * (self.head_size ** -0.5)

        # Better causal mask using -inf
        mask = jnp.tril(jnp.ones((T, T)))
        weights = jnp.where(mask == 0, -jnp.inf, weights)

        weights = nn.softmax(weights, axis=-1)
        # weights = self.dropout(weights, deterministic=not training)  # Apply dropout to attention weights

        out = jnp.matmul(weights, value)
        out = self.d_O(out)
        out = self.dropout(out, deterministic=not training)
        return out

class MHA(nn.Module):
    d_model: int = config.d_model
    num_heads: int = config.num_heads
    dropout_rate: float = config.dropout_rate

    def setup(self):
        self.heads = [Attention(self.d_model, self.num_heads, self.dropout_rate) for _ in range(self.num_heads)]

        # Proper initialization for output projection
        self.linear = nn.Dense(
            features=self.d_model,
            dtype=get_dtype(),
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros
        )
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, training=True):
        out = jnp.concatenate([head(x, training) for head in self.heads], axis=-1)
        out = self.linear(out)
        out = self.dropout(out, deterministic=not training)
        return out

class MLP(nn.Module):
    d_model: int = config.d_model
    d_ff: int = config.d_ff
    dropout_rate: float = config.dropout_rate

    def setup(self):
        # Proper initialization for MLP layers
        self.fc1 = nn.Dense(
            features=self.d_ff,
            dtype=get_dtype(),
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros
        )
        self.fc2 = nn.Dense(
            features=self.d_model,
            dtype=get_dtype(),
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros
        )
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, training=True):
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        x = self.dropout(x, deterministic=not training)  # Remove duplicate GELU
        return x

class TransformerBlock(nn.Module):
    d_model: int = config.d_model
    num_heads: int = config.num_heads
    d_ff: int = config.d_ff
    dropout_rate: float = config.dropout_rate

    def setup(self):
        self.attention = MHA(self.d_model, self.num_heads, self.dropout_rate)
        self.mlp = MLP(self.d_model, self.d_ff, self.dropout_rate)
        self.ln1 = nn.LayerNorm(dtype=get_dtype())
        self.ln2 = nn.LayerNorm(dtype=get_dtype())

    def __call__(self, x, training=True):
        attn =  self.attention(self.ln1(x), training)
        # x += attn
        x = x + attn * ((2 * config.num_layers ** -0.5))
        mlp_out = self.mlp(self.ln2(x), training)
        # x += mlp_out
        x = x + mlp_out * (2 * (config.num_layers ** -0.5))
        # x = x * (config.num_layers ** -0.5)
        return x

class GPT(nn.Module):
    d_model: int = config.d_model
    num_heads: int = config.num_heads
    d_ff: int = config.d_ff
    dropout_rate: float = config.dropout_rate
    vocab_size: int = config.vocab_size
    seq_len: int = config.max_seq_len

    def setup(self):
        self.embedding_table = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model, dtype=get_dtype(), embedding_init=nn.initializers.normal(stddev=0.02))
        self.positional_embedding = self.param(
            "positional_embeddings",  # name
            lambda key: jax.random.normal(key, (1, self.seq_len, self.d_model), dtype=get_dtype()) * 0.01
        )
        self.decoder = [TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.dropout_rate) for _ in range(config.num_layers)]
        self.linear_out = nn.Dense(features=self.vocab_size, dtype=get_dtype(), kernel_init=nn.initializers.normal(stddev=0.02), bias_init=nn.initializers.zeros)  # Zero bias initialization
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, training=True):
        B,T = x.shape
        embeds = self.embedding_table(x)  # (B,T,d_model)
        C = embeds.shape[-1]
        pos_embeds = self.positional_embedding[:, :T, :]  # (1,T,d_model)
        x = embeds + pos_embeds  # (B,T,d_model)
        # pad_mask = (x != tokenizer.pad_token_id).astype(get_dtype())
        # x = x * pad_mask
        for layer in self.decoder:
            x = layer(x, training=training)

        x = self.linear_out(x)
        x = self.dropout(x, deterministic=not training)
        return x

# Add this cell to inspect the model summary like torchsummary

# Initialize model
model = GPT()
key = jax.random.PRNGKey(0)
x = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)  # Dummy input for tabulation

# Tabulate the model structure
tabulate_fn = tabulate(model, key, console_kwargs={'width': 120})

# Count total parameters
params = model.init(key, x)['params']
total_params = sum(jax.tree_util.tree_leaves(jax.tree.map(lambda arr: arr.size, params)))

# Get raw summary and clean ANSI codes
raw_summary = tabulate_fn(x, training=True)
# Remove ANSI color codes for clean logging
clean_summary = re.sub(r'\x1b\[[0-9;]*m', '', raw_summary)

# Save to log file with clean formatting
with open('model_summary.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("GPT MODEL ARCHITECTURE SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total Parameters: {total_params:,}\n")
    f.write(f"Model Configuration:\n")
    f.write(f"  - Vocabulary Size: {config.vocab_size:,}\n")
    f.write(f"  - Max Sequence Length: {config.max_seq_len}\n")
    f.write(f"  - Model Dimension: {config.d_model}\n")
    f.write(f"  - Number of Layers: {config.num_layers}\n")
    f.write(f"  - Number of Heads: {config.num_heads}\n")
    f.write(f"  - Feed Forward Dimension: {config.d_ff}\n")
    f.write(f"  - Dropout Rate: {config.dropout_rate}\n\n")
    f.write("Detailed Layer Information:\n")
    f.write("-" * 40 + "\n")
    f.write(clean_summary)

print(f"Model summary saved to model_summary.txt")
print(f"Total Parameters: {total_params:,}")
print(f"Model size: ~{total_params * 2 / (1024**2):.1f} MB (bfloat16)")

def create_learning_rate_schedule():
    """Create a learning rate schedule with warmup and cosine decay."""
    # Use values from config
    max_lr = config.lr  # 6e-4
    min_lr = config.min_lr
    warmup_steps = config.warmup_steps  # 700 (or can override to 715)
    max_steps = config.total_steps  # 20000 (or can override to 19073)

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    return get_lr

def compute_ce_loss(logits, labels):
    """Compute cross-entropy loss."""
    labels = labels[:, 1:]
    logits = logits[:, :-1, :]  # Shift logits to align with labels

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    pad_mask = (labels != tokenizer.pad_token_id)
    loss = jnp.where(pad_mask, loss, 0.0)
    return loss.sum () / pad_mask.sum()

def create_train_state(rng, config):
    """Create initial training state."""
    model = GPT()

    # Initialize parameters
    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input)['params']

    # Create learning rate schedule
    lr_schedule = create_learning_rate_schedule()

    # Create optimizer with stronger gradient clipping and better settings
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Much stronger clipping (was 1.0)
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.95,
            weight_decay=0.01,  # Reduced weight decay (was 0.1)
            eps=1e-9  # Added epsilon for numerical stability
        )
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

@jax.jit
def train_step(state, batch, step):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch, training=True, rngs={'dropout': step})
        loss = compute_ce_loss(logits, batch)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Compute gradient norm for logging
    grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)]))

    # Update the parameters
    state = state.apply_gradients(grads=grads)
    return state, loss, grad_norm

@jax.jit
def train_step_accum(state, batch, step):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch, training=True, rngs={'dropout': step})
        loss = compute_ce_loss(logits, batch)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    # Compute gradient norm for logging

    if state.grad_accum is None:

        state = state.replace(
            grad_accum=jax.tree_util.tree_map(jnp.zeros_like, grads),
            accum_step=0,
        )
    new_accum = jax.tree_util.tree_map(lambda g1, g2: g1 + g2, state.grad_accum, grads)
    new_step = state.accum_step + 1

    # print(new_accum)
    # print(new_step)

    def apply_update(_):
        mean_grads = jax.tree_util.tree_map(lambda g: g / config.gradient_accumulation_steps, new_accum)
        # print("Grads: ", mean_grads)
        grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(mean_grads)]))
        # print("Here: ", jnp.array(grad_norm, float))
        new_state = state.apply_gradients(grads = mean_grads)
        new_grad_accum = jax.tree_util.tree_map(jnp.zeros_like, new_accum)

        return new_state.replace(grad_accum=new_grad_accum, accum_step=0), grad_norm

    def carry_forward(_):

        # dummy_grads = jax.tree_util.tree_map(jnp.zeros_like, grads)

        return state.replace(grad_accum=new_accum, accum_step=new_step), -1.0


    state, grad_norm = jax.lax.cond(new_step == config.gradient_accumulation_steps, apply_update, carry_forward, operand=None)

    return state, loss, grad_norm

# JIT-compiled evaluation step
@jax.jit
def eval_step(state, batch, step):
    """Single evaluation step."""
    logits = state.apply_fn({'params': state.params}, batch, training=False, rngs={'dropout': step})
    loss = compute_ce_loss(logits, batch)

    return loss, None

# Vectorized prediction function using vmap
@jax.jit
def predict_batch(state, batch):
    """Generate predictions for a batch using vmap."""
    return state.apply_fn({'params': state.params}, batch, training=False)

# Helper function to create wandb summary table
def log_training_summary(state, config, total_params, tokens_processed):
    """Log a comprehensive training summary to wandb."""

    # Create a summary table (ensure all values are strings for wandb compatibility)
    summary_data = [
        ["Model", "SmolJAX GPT"],
        ["Total Parameters", f"{total_params:,}"],
        ["Model Size (MB)", f"{total_params * 2 / (1024**2):.1f}"],
        ["Vocabulary Size", f"{config.vocab_size:,}"],
        ["Max Sequence Length", f"{config.max_seq_len}"],
        ["Model Dimension", f"{config.d_model}"],
        ["Number of Layers", f"{config.num_layers}"],
        ["Number of Heads", f"{config.num_heads}"],
        ["Feed Forward Dimension", f"{config.d_ff}"],
        ["Dropout Rate", f"{config.dropout_rate}"],
        ["Learning Rate", f"{config.lr}"],
        ["Batch Size", f"{config.batch_size}"],
        ["Total Epochs", f"{config.num_epochs}"],
        ["Tokens Processed", f"{tokens_processed:,}"],
        ["Training Step", f"{int(state.step)}"]
    ]

    # Create wandb table
    table = wandb.Table(
        columns=["Metric", "Value"],
        data=summary_data
    )

    wandb.log({"training_summary": table})

    return table

# Checkpoint management functions
def save_checkpoint(state, step, checkpoint_dir="./checkpoints", keep=5):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the checkpoint
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=state,
        step=step,
        keep=keep,  # Keep only the last 5 checkpoints
        overwrite=True
    )

    print(f"Checkpoint saved at step {step} in {checkpoint_dir}")

    # Log to wandb if available
    if wandb.run is not None:
        wandb.log({
            "checkpoint/step": step,
            "checkpoint/saved": 1
        }, step=step)

def load_checkpoint(checkpoint_dir="./checkpoints", state=None):
    """Load the latest checkpoint."""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return None, 0

    # Check if there are any checkpoints
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')]
    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return None, 0

    try:
        # Load the latest checkpoint
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_dir,
            target=state
        )

        # Get the step number from the checkpoint
        latest_step = checkpoints.latest_checkpoint(checkpoint_dir)
        if latest_step:
            step = int(latest_step.split('_')[-1])
            print(f"Checkpoint loaded from step {step}")
            return restored_state, step
        else:
            print(f"Could not determine step from checkpoint")
            return None, 0

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, 0

def get_checkpoint_info(checkpoint_dir="./checkpoints"):
    """Get information about available checkpoints."""
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')]
    checkpoint_steps = []

    for file in checkpoint_files:
        try:
            step = int(file.split('_')[-1])
            checkpoint_steps.append(step)
        except ValueError:
            continue

    return sorted(checkpoint_steps)

# Model-only saving/loading functions for inference
import pickle
import json

def save_model_for_inference(state, model_dir="./saved_models", model_name="smoljax_gpt"):
    """Save only model parameters and config for inference (much smaller files)."""
    os.makedirs(model_dir, exist_ok=True)

    # Save just the parameters (no optimizer state)
    params_path = os.path.join(model_dir, f"{model_name}_params.pkl")
    with open(params_path, 'wb') as f:
        pickle.dump(state.params, f)

    # Save model configuration
    config_path = os.path.join(model_dir, f"{model_name}_config.json")
    config_dict = {
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
        "d_model": config.d_model,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "d_ff": config.d_ff,
        "dropout_rate": config.dropout_rate
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Save tokenizer info
    tokenizer_path = os.path.join(model_dir, f"{model_name}_tokenizer_info.json")
    tokenizer_info = {
        "tokenizer_name": "gpt2",
        "vocab_size": len(tokenizer),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id
    }
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer_info, f, indent=2)

    print(f"   Model saved for inference:")
    print(f"   Parameters: {params_path}")
    print(f"   Config: {config_path}")
    print(f"   Tokenizer info: {tokenizer_path}")

    # Calculate file sizes
    params_size = os.path.getsize(params_path) / (1024**2)  # MB
    print(f"   Model size: {params_size:.1f} MB")

    return params_path, config_path, tokenizer_path

def load_model_for_inference(model_dir="./saved_models", model_name="smoljax_gpt"):
    """Load model parameters and config for inference."""

    # Load configuration
    config_path = os.path.join(model_dir, f"{model_name}_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Recreate config object
    inference_config = GPTConfig(**config_dict)

    # Load parameters
    params_path = os.path.join(model_dir, f"{model_name}_params.pkl")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters file not found: {params_path}")

    with open(params_path, 'rb') as f:
        params = pickle.load(f)

    # Load tokenizer info
    tokenizer_path = os.path.join(model_dir, f"{model_name}_tokenizer_info.json")
    tokenizer_info = None
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'r') as f:
            tokenizer_info = json.load(f)

    print(f" Model loaded for inference:")
    print(f"   Config: {config_dict}")
    if tokenizer_info:
        print(f"   Tokenizer: {tokenizer_info['tokenizer_name']}")

    return params, inference_config, tokenizer_info

def create_inference_model(params, inference_config):
    """Create a model instance for inference (no training state)."""

    # Create model with loaded config
    model = GPT(
        d_model=inference_config.d_model,
        num_heads=inference_config.num_heads,
        d_ff=inference_config.d_ff,
        dropout_rate=inference_config.dropout_rate,
        vocab_size=inference_config.vocab_size,
        seq_len=inference_config.max_seq_len
    )

    # Create apply function with loaded parameters
    def inference_apply(inputs, training=False):
        return model.apply({'params': params}, inputs, training=training)

    return model, inference_apply

def load_and_setup_for_inference(model_dir="./saved_models", model_name="smoljax_gpt"):
    """Complete setup for inference - load everything and return ready-to-use functions."""

    # Load model components
    params, inference_config, tokenizer_info = load_model_for_inference(model_dir, model_name)

    # Create model and inference function
    model, inference_apply = create_inference_model(params, inference_config)

    # Setup tokenizer (you might want to load this separately)
    if tokenizer_info and tokenizer_info.get('tokenizer_name') == 'gpt2':
        
        inference_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        inference_tokenizer.pad_token = tokenizer_info.get('pad_token', '[PAD]')
        print(f" Tokenizer setup complete")
    else:
        inference_tokenizer = None
        print(" No tokenizer info found, you'll need to setup tokenizer manually")

    return {
        'model': model,
        'apply_fn': inference_apply,
        'params': params,
        'config': inference_config,
        'tokenizer': inference_tokenizer
    }



def top_k_sampling(logits, k=550, temperature=0.7, rng_key=None):
    """
    Apply top-k sampling with temperature to logits.

    Args:
        logits: [vocab_size] array of logits
        k: number of top candidates to keep
        temperature: sampling temperature (lower = more deterministic)
        rng_key: PRNG key for randomness

    Returns:
        sampled token index
    """

    # Apply temperature scaling
    logits = logits / temperature

    # Get top-k indices and values
    top_k_logits, top_k_indices = jax.lax.top_k(logits, k)

    # Convert to probabilities
    top_k_probs = jax.nn.softmax(top_k_logits)

    # Sample from the top-k distribution
    sampled_idx = jax.random.categorical(rng_key, jnp.log(top_k_probs))

    # Return the actual token index
    return top_k_indices[sampled_idx]

def nucleus_sampling(logits, p=0.95, temperature=0.7, rng_key=None):
    """
    Apply nucleus (top-p) sampling with temperature to logits.

    Args:
        logits: [vocab_size] array of logits
        p: cumulative probability threshold
        temperature: sampling temperature
        rng_key: PRNG key for randomness

    Returns:
        sampled token index
    """

    # Apply temperature scaling
    # logits = logits / temperature

    # Convert to probabilities and sort
    probs = jax.nn.softmax(logits)
    sorted_indices = jnp.argsort(probs)[::-1]  # Sort in descending order
    sorted_probs = probs[sorted_indices]

    # Find cumulative probabilities
    cumsum_probs = jnp.cumsum(sorted_probs)

    # Find the cutoff index where cumsum exceeds p
    cutoff = jnp.searchsorted(cumsum_probs, p)
    cutoff = jnp.maximum(cutoff, 1)  # Keep at least one token

    # Keep only tokens within the nucleus
    nucleus_indices = sorted_indices[:cutoff]
    nucleus_probs = sorted_probs[:cutoff]
    nucleus_probs = nucleus_probs / jnp.sum(nucleus_probs)  # Renormalize

    # Sample from the nucleus
    sampled_idx = jax.random.categorical(rng_key, jnp.log(nucleus_probs))

    return nucleus_indices[sampled_idx]

@jax.jit
def generate_next_token(state, input_ids, temperature=1.0, top_k=50, use_nucleus=False, nucleus_p=0.9, rng_key=None):
    """
    Generate the next token using the model.

    Args:
        state: training state with model parameters
        input_ids: current sequence [batch_size, seq_len]
        temperature: sampling temperature
        top_k: number of top candidates for top-k sampling
        use_nucleus: whether to use nucleus sampling instead of top-k
        nucleus_p: probability threshold for nucleus sampling
        rng_key: PRNG key for randomness

    Returns:
        next token index
    """
    # Get model predictions
    logits = state.apply_fn({'params': state.params}, input_ids, training=False)

    # Take logits for the last position
    next_token_logits = logits[0, -1, :]  # [vocab_size]

    # Apply sampling strategy
    if use_nucleus:
        next_token = nucleus_sampling(next_token_logits, p=nucleus_p, temperature=temperature, rng_key=rng_key)
    else:
        next_token = top_k_sampling(next_token_logits, k=top_k, temperature=temperature, rng_key=rng_key)

    return next_token

def generate_text(state, prompt, tokenizer, max_length=config.max_seq_len, temperature=0.7, top_k=500,
                 use_nucleus=True, nucleus_p=0.95, seed=42, stop_at_eos=True, verbose=False):
    """
    Generate text using the trained model with advanced sampling.

    Args:
        state: training state with model parameters
        prompt: input text string to start generation
        tokenizer: tokenizer for encoding/decoding
        max_length: maximum number of tokens to generate
        temperature: sampling temperature (0.1 = conservative, 1.0 = balanced, 2.0 = creative)
        top_k: number of top candidates for top-k sampling (lower = more focused)
        use_nucleus: whether to use nucleus (top-p) sampling instead of top-k
        nucleus_p: probability threshold for nucleus sampling (0.9 = balanced)
        seed: random seed for reproducibility
        stop_at_eos: whether to stop generation at EOS token
        verbose: whether to print generation progress

    Returns:
        generated text string
    """
    # Initialize random key
    rng_key = jax.random.PRNGKey(seed)

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='np')
    input_ids = jnp.array(input_ids).reshape(1, -1)  # [1, seq_len]

    generated_tokens = []

    if verbose:
        print(f"   Generating text with:")
        print(f"   Temperature: {temperature}")
        print(f"   {'Nucleus (top-p)' if use_nucleus else 'Top-k'}: {nucleus_p if use_nucleus else top_k}")
        print(f"   Max length: {max_length}")
        print(f"   Prompt: '{prompt}'")
        print("    Generation:")
        print(prompt, end="")

    for i in range(max_length):
        # Split the random key for this step
        rng_key, step_key = jax.random.split(rng_key)

        # Generate next token
        next_token = generate_next_token(
            state, input_ids,
            temperature=temperature,
            top_k=top_k,
            use_nucleus=use_nucleus,
            nucleus_p=nucleus_p,
            rng_key=step_key
        )

        # Convert to Python int
        next_token = int(next_token)
        generated_tokens.append(next_token)

        # Check for EOS token
        if stop_at_eos and next_token == tokenizer.eos_token_id:
            if verbose:
                print(" [EOS]")
            break

        # Decode and print the new token if verbose
        if verbose:
            token_text = tokenizer.decode([next_token])
            print(token_text, end="", flush=True)

        # Update input_ids for next iteration
        next_token_array = jnp.array([[next_token]])
        input_ids = jnp.concatenate([input_ids, next_token_array], axis=1)

        # Truncate if sequence gets too long (to fit model's max_seq_len)
        if input_ids.shape[1] > config.max_seq_len:
            input_ids = input_ids[:, -config.max_seq_len:]

    if verbose:
        print("\n Generation complete!")

    # Decode the full generated text
    full_text = tokenizer.decode(tokenizer.encode(prompt) + generated_tokens, skip_special_tokens=True)

    return full_text

# Data preprocessing and collate function
def collate(batch):
    """Collate function for DataLoader to handle TinyStories data."""
    # Extract text from batch
    texts = [item['text'] for item in batch]

    # Tokenize all texts
    encoded = tokenizer(
        texts,
        max_length=config.max_seq_len,
        truncation=True,
        padding='max_length',
        return_tensors='np'
    )

    return encoded['input_ids']

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=collate,
    # num_workers=int(os.cpu_count() / 2)
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=collate,
    # num_workers=int(os.cpu_count() / 2)
)

len(next(iter(train_loader)))

def save_to_file(text, step):

    dir = './generated_texts'
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    with open('generated_texts/{step}.txt', 'w') as f:
        f.writelines(text + "\n\n")

def train(resume_from_checkpoint=False, checkpoint_dir="./checkpoints", save_every=1000):
    # Initialize wandb
    wandb.init(
        project="smoljax-gpt",
        config={
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "d_model": config.d_model,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "d_ff": config.d_ff,
            "dropout_rate": config.dropout_rate,
            "learning_rate": config.lr,
            "warmup_steps": config.warmup_steps,
            "total_steps": config.total_steps,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "save_every": save_every,
        }
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        # num_workers=int(os.cpu_count() / 2)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
        # num_workers=int(os.cpu_count() / 2)
    )

    rng = jax.random.PRNGKey(0)
    train_state = create_train_state(rng, config)

    # Initialize variables
    start_step = 0
    tokens_processed = 0

    # Try to resume from checkpoint if requested
    if resume_from_checkpoint:
        print("Checking for existing checkpoints...")
        available_checkpoints = get_checkpoint_info(checkpoint_dir)

        if available_checkpoints:
            print(f"Found checkpoints at steps: {available_checkpoints}")
            restored_state, start_step = load_checkpoint(checkpoint_dir, train_state)

            if restored_state is not None:
                train_state = restored_state
                print(f"Resuming training from step {start_step}")

                # Estimate tokens processed (rough approximation)
                tokens_processed = start_step * config.batch_size * config.max_seq_len
                print(f"Estimated tokens processed so far: {tokens_processed:,}")
            else:
                print("Failed to load checkpoint, starting from scratch")
        else:
            print("No checkpoints found, starting fresh training")

    # Log model summary to wandb
    total_params = sum([param.size for param in jax.tree_util.tree_leaves(train_state.params)])
    wandb.log({
        "model/total_parameters": total_params,
        "model/model_size_mb": total_params * 2 / (1024**2),  # bfloat16 = 2 bytes
        "checkpoint/resume_from": start_step
    })

    # Log detailed training summary
    log_training_summary(train_state, config, total_params, tokens_processed)

    # Training loop
    num_epochs = config.num_epochs
    state = train_state.replace(step=start_step)  # Set correct step for LR schedule
    global_step = start_step

    for epoch in range(num_epochs):
        # Training
        train_losses = []
        train_grad_norms = []

        # Create data iterator
        # train_data_iterator = prefetch_to_device(iter(train_loader), 2, None)
        # val_data_iterator = prefetch_to_device(iter(val_loader), 2, None)
        train_data_iterator = iter(train_loader)
        val_data_iterator = iter(val_loader)

        # Progress bar tracks optimization steps, not data batches
        pbar = tqdm(range(config.total_steps), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for step_idx in pbar:

            # Initialize accumulation variables for this step
            step_losses = []
            step_grad_norms = []
            rng, step_key = jax.random.split(rng)

            # Gradient accumulation loop
            for micro_step in range(config.gradient_accumulation_steps):

              rng, step_key = jax.random.split(rng)
              try:
                  batch = next(train_data_iterator)
                  # batch = batch[0]  # Unpack from tuple
              except StopIteration:
                  # train_data_iterator = prefetch_to_device(iter(train_loader), 2, None)
                  train_data_iterator = iter(train_loader)
                  batch = next(train_data_iterator)
                  # batch = batch[0]  # Unpack from tuple
              # print(batch)
              # Convert to JAX array
              # batch = jnp.array(batch)
              batch = jnp.array(batch)
              batch = jax.device_put(batch, jax.devices()[0])  # Manual device placement
              # Count tokens (excluding padding)
              batch_tokens = jnp.sum(batch != tokenizer.pad_token_id)
              tokens_processed += int(batch_tokens)

              state, loss, grad_norm = train_step_accum(state, batch, step_key)
              step_losses.append(float(loss))
              # print(grad_norm)
              grad_norm = float(grad_norm)
              # print(grad_norm)
              # if micro_step == config.gradient_accumulation_steps - 1 and grad_norm is not None:
              if grad_norm > 0.0:
                  # print("Final: ", grad_norm)
                  step_grad_norms.append(grad_norm)
            # print(f"Gradine Accumulation Running: ( {micro_step} / {config.gradient_accumulation_steps })")
            # Average the losses and grad norms from accumulation steps
            avg_step_loss = np.mean(step_losses)
            # avg_step_grad_norm = np.mean(step_grad_norms) if step_grad_norms else 0.0

            train_losses.append(avg_step_loss)
            train_grad_norms.append(step_grad_norms)

            # if grad_norm is not None:
            #         step_grad_norms.append(float(avg_step_grad_norm))
            global_step += 1

            # Get current learning rate
            # current_lr = float(state.opt_state[1].hyperparams['learning_rate'])
            current_lr = create_learning_rate_schedule()(state.step)

            # Log training metrics to wandb
            wandb.log({
                "train/loss": avg_step_loss,
                "train/grad_norm": np.mean(step_grad_norms),
                "train/learning_rate": current_lr,
                "train/tokens_processed": tokens_processed,
                "train/epoch": epoch + 1,
                "train/batch_size": config.batch_size,
                "train/step": global_step
            }, step=global_step)

            # Save checkpoint every save_every steps
            if global_step % save_every == 0:
                # save_checkpoint(state, global_step, checkpoint_dir)

                # Also save model-only version for inference
                save_model_for_inference(
                    state,
                    model_dir="./saved_models",
                    model_name=f"smoljax_gpt_step_{global_step}"
                )

                # # Log checkpoint info to wandb
                # wandb.log({
                #     "checkpoint/last_saved_step": global_step,
                #     "checkpoint/tokens_at_save": tokens_processed
                # }, step=global_step)


            # Update progress bar
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Step {global_step}/{config.total_steps} [Train]")
            pbar.set_postfix({
                "loss": f"{avg_step_loss:.4f}",
                "grad_norm": f"{np.mean(step_grad_norms):.4f}",
                "lr": f"{current_lr:.6f}",
                "tokens": f"{tokens_processed:,}",
                "micro_batches": f"{config.gradient_accumulation_steps}"
            })

            # Break if we've reached the total steps
            if global_step >= config.total_steps:
                print(f"\nReached maximum steps ({config.total_steps}), stopping training...")
                break

            # Break out of epoch loop if we've reached max steps
            if global_step >= config.total_steps:
                break

            # Validation
            val_losses = []
            # val_accs = []

            # Simple validation without iterator complexity
            # pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

            if(global_step % config.eval_steps == 0):

              rng, step_key = jax.random.split(rng)
              for batch in tqdm(range(config.eval_steps), desc='Validation running...'):
                  try:
                    batch = next(val_data_iterator)
                    # batch = batch[0]  # Unpack from tuple
                  except StopIteration:
                    # val_data_iterator = prefetch_to_device(iter(val_loader), 2, None)
                    val_data_iterator = iter(val_loader)
                    batch = next(val_data_iterator)
                    # batch = batch[0]  # Unpack from tuple


                  batch = jnp.array(batch)
                  batch = jax.device_put(batch, jax.devices()[0])  # Manual device placement
                  loss, _ = eval_step(state, batch, step_key)
                  val_losses.append(float(loss))
                  # val_accs.append(float(acc))
                  print(loss)
                  break
                  # Update progress bar
                  pbar.set_postfix({
                      "loss val": f"{loss:.4f}"
                      # "acc": f"{acc:.4f}"
                  })

            # Calculate epoch metrics
            avg_train_loss = np.mean(train_losses)
            avg_train_grad_norm = np.mean(train_grad_norms)
            if len(val_losses) > 0:
              avg_val_loss = np.mean(val_losses)
            else:
              avg_val_loss = -1
            # avg_val_acc = np.mean(val_accs)

            # Log epoch metrics to wandb
            wandb.log({
                "epoch/train_loss": avg_train_loss,
                "epoch/val_loss": avg_val_loss,
                # "epoch/val_accuracy": avg_val_acc,
                "epoch/train_grad_norm": avg_train_grad_norm,
                "epoch/epoch": epoch + 1
            }, step=global_step)

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Grad Norm: {avg_train_grad_norm:.4f}")
            # print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
            print(f"  Tokens Processed: {tokens_processed:,}")
            print(f"  Global Step: {global_step}")

            # Save checkpoint at end of each epoch
            # save_checkpoint(state, global_step, checkpoint_dir)

    # Save final checkpoint
    # print("\n Saving final checkpoint...")
    # save_checkpoint(state, global_step, checkpoint_dir)

    # Generate some text
    print("\nGenerating text...")
    generated = generate_text(state, "The future of artificial intelligence", tokenizer, max_length=50)
    print(f"Generated: {generated}")
    save_to_file(generated, global_step)

    # Log final generation to wandb
    wandb.log({
        "generation/sample_text": generated,
        "generation/prompt": "The future of artificial intelligence",
        "training/final_step": global_step,
        "training/final_tokens": tokens_processed
    })

    # Log final training summary
    log_training_summary(state, config, total_params, tokens_processed)

    # Finish wandb run
    wandb.finish()

    return state

train()

