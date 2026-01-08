from typing import Optional
from dataclasses import dataclass, field

@dataclass
class RunConfig:
    runs_dir: str = "runs"
    run_name_prefix: str = "ts_baseline"
    run_name: Optional[str] = None

@dataclass
class DataConfig:
    # Memmap token files (1D binary files)
    train_data_path: str = "workspace/tinystories_train.uint16.bin"
    val_data_path: str = "workspace/tinystories_valid.uint16.bin"
    # Numpy dtype used when creating the token files
    np_dtype: str = "uint16"

    context_length: int = 256

    # Device string used by get_batch(), cuda:0 or cpu
    device: str = "cuda:0"

@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16

    # If None, will default to 4 * d_model at model construction time
    d_ff: Optional[int] = 1344

    rope_theta: float = 10_000.0
    # If None, model will use context_length
    max_seq_len: Optional[int] = None

    rmsnorm_eps: float = 1e-5

    # torch dtype string used for model parameters
    torch_dtype: str = "float32"

@dataclass
class OptimizerConfig:
    lr_max: float = 1e-3
    lr_min: float = 1e-4

    warmup_iters: int = 2000
    cosine_cycle_iters: int = 106_667
    
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.1

    grad_clip: float = 1.0

@dataclass
class TrainingConfig:
    max_steps: int = 106_667
    batch_size: int = 12
    
    log_interval: int = 50
    eval_interval: int = 2000
    eval_batches: int = 20

    ckpt_interval: int = 5000
    resume_from: Optional[str] = None

    seed: int = 42

@dataclass
class WandbConfig:
    enable: bool = True
    project: str = "cs336-a1"
    run_name: str = "train"

@dataclass
class TrainConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

def get_default_config() -> TrainConfig:
    """
    Return a default training configuration.
    """
    cfg = TrainConfig()

    # Keep model/data context_length consistent by default
    cfg.model.context_length = cfg.data.context_length

    return cfg