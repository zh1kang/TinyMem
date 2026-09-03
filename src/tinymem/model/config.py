from dataclasses import asdict, dataclass, field
from numbers import Real


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 512
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0
    max_local_tokens: int = 128
    positional_encoding: str = "rope"
    tie_embeddings: bool = True
    attention_type: str = "mha"
    kv_latent_dim: int | None = None

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "vocab_size",
            "d_model",
            "n_layers",
            "n_heads",
            "d_ff",
            "max_local_tokens",
        )

        for field_name in positive_integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if isinstance(self.dropout, bool) or not isinstance(self.dropout, Real):
            raise TypeError("dropout must be a real number")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")

        if not isinstance(self.positional_encoding, str):
            raise TypeError("positional_encoding must be a string")
        if self.positional_encoding != "rope":
            raise ValueError("positional_encoding must be 'rope'")

        if not isinstance(self.tie_embeddings, bool):
            raise TypeError("tie_embeddings must be a boolean")

        if not isinstance(self.attention_type, str):
            raise TypeError("attention_type must be a string")
        if self.attention_type not in {"mha", "mla_lite"}:
            raise ValueError("attention_type must be 'mha' or 'mla_lite'")
        if self.attention_type == "mha":
            if self.kv_latent_dim is not None:
                raise ValueError("kv_latent_dim must be None for MHA")
        else:
            if isinstance(self.kv_latent_dim, bool) or not isinstance(
                self.kv_latent_dim, int
            ):
                raise TypeError("kv_latent_dim must be an integer for MLA-lite")
            if self.kv_latent_dim <= 0:
                raise ValueError("kv_latent_dim must be positive")
            if self.kv_latent_dim >= self.d_model:
                raise ValueError("kv_latent_dim must be smaller than d_model")


@dataclass(frozen=True)
class StreamConfig:
    segment_length: int = 64
    local_window: int = 128

    def __post_init__(self) -> None:
        if type(self.segment_length) is not int or type(self.local_window) is not int:
            raise TypeError("segment_length and local_window must be integers")

        if self.segment_length <= 0 or self.local_window <= 0:
            raise ValueError("segment_length and local_window must be positive integers")

        if self.segment_length > self.local_window:
            raise ValueError("segment_length must not exceed local_window")


@dataclass(frozen=True)
class MemoryConfig:
    n_slots: int = 8
    codebook_size: int = 256
    code_dim: int = 256
    codes_per_write: int = 2
    update_interval_segments: int = 1
    controller_actions: tuple[str, ...] = ("keep", "write")

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "n_slots",
            "codebook_size",
            "code_dim",
            "codes_per_write",
            "update_interval_segments",
        )

        for field_name in positive_integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")

        for field_name in positive_integer_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")

        if self.codes_per_write > self.n_slots:
            raise ValueError("codes_per_write must not exceed n_slots")

        if not isinstance(self.controller_actions, tuple):
            raise TypeError("controller_actions must be a tuple")
        for action in self.controller_actions:
            if not isinstance(action, str):
                raise TypeError("controller_actions must contain strings only")

        if len(self.controller_actions) == 0:
            raise ValueError("controller_actions must be nonempty")
        if len(set(self.controller_actions)) != len(self.controller_actions):
            raise ValueError("controller_actions must contain unique values")

        if self.controller_actions != ("keep", "write"):
            raise ValueError("controller_actions must be exactly ('keep', 'write')")


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str = "adamw"
    learning_rate: float = 0.0003
    weight_decay: float = 0.1
    batch_size: int = 32
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 200
    max_steps: int = 10_000

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, str):
            raise TypeError("optimizer must be a string")

        if self.optimizer != "adamw":
            raise ValueError("optimizer must be 'adamw'")

        positive_integer_fields = ("batch_size", "max_steps")
        for field_name in positive_integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")

        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int):
            raise TypeError("warmup_steps must be an integer")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")

        positive_real_fields = ("learning_rate", "gradient_clip_norm")
        for field_name in positive_real_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} must be a real number")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")

        if isinstance(self.weight_decay, bool) or not isinstance(self.weight_decay, Real):
            raise TypeError("weight_decay must be a real number")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")

        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps must not exceed max_steps")


@dataclass(frozen=True)
class MTPConfig:
    enabled: bool = False
    horizons: tuple[int, ...] = (2, 3, 4)
    loss_weight: float = 0.2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")

        if not isinstance(self.horizons, tuple):
            raise TypeError("horizons must be a tuple")
        if not self.horizons:
            raise ValueError("horizons must be nonempty")
        for horizon in self.horizons:
            if isinstance(horizon, bool) or not isinstance(horizon, int):
                raise TypeError("horizons must contain integers")
            if horizon < 2:
                raise ValueError("horizons must be at least 2")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and strictly increasing")

        if isinstance(self.loss_weight, bool) or not isinstance(self.loss_weight, Real):
            raise TypeError("loss_weight must be a real number")
        if self.loss_weight < 0:
            raise ValueError("loss_weight must be nonnegative")
        if self.enabled and self.loss_weight == 0:
            raise ValueError("loss_weight must be positive when MTP is enabled")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 1337
    model: ModelConfig = field(default_factory=ModelConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

        nested_types = (
            ("model", ModelConfig),
            ("stream", StreamConfig),
            ("memory", MemoryConfig),
            ("training", TrainingConfig),
            ("mtp", MTPConfig),
        )
        for field_name, expected_type in nested_types:
            if not isinstance(getattr(self, field_name), expected_type):
                raise TypeError(f"{field_name} must be a {expected_type.__name__}")

        if self.model.max_local_tokens != self.stream.local_window:
            raise ValueError("model max_local_tokens must equal stream local_window")
        if self.memory.code_dim != self.model.d_model:
            raise ValueError("memory code_dim must equal model d_model")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "ExperimentConfig":
        if not isinstance(values, dict):
            raise TypeError("values must be a dictionary")

        allowed_keys = {"seed", "model", "stream", "memory", "training", "mtp"}
        unknown_keys = set(values) - allowed_keys
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(f"unknown experiment config fields: {unknown}")

        def section(name: str) -> dict[str, object]:
            value = values.get(name, {})
            if not isinstance(value, dict):
                raise TypeError(f"{name} must be a dictionary")
            return dict(value)

        model_values = section("model")
        stream_values = section("stream")
        memory_values = section("memory")
        training_values = section("training")
        mtp_values = section("mtp")

        if isinstance(memory_values.get("controller_actions"), list):
            memory_values["controller_actions"] = tuple(
                memory_values["controller_actions"]
            )
        if isinstance(mtp_values.get("horizons"), list):
            mtp_values["horizons"] = tuple(mtp_values["horizons"])

        seed = values.get("seed", 1337)
        return cls(
            seed=seed,
            model=ModelConfig(**model_values),
            stream=StreamConfig(**stream_values),
            memory=MemoryConfig(**memory_values),
            training=TrainingConfig(**training_values),
            mtp=MTPConfig(**mtp_values),
        )
