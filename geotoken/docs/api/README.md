# GeoToken API Reference

## Package Exports

All public APIs are exported from the main package:

```python
from geotoken import (
    # Configuration
    PrecisionTier,
    QuantizationConfig,
    NormalizationConfig,
    AdaptiveBitAllocationConfig,
    CommandTokenizationConfig,
    GraphTokenizationConfig,

    # Tokenization
    GeoTokenizer,
    CommandSequenceTokenizer,
    GraphTokenizer,
    CADVocabulary,

    # Token Types
    TokenSequence,
    CoordinateToken,
    GeometryToken,
    CommandToken,
    CommandType,
    ConstraintToken,
    ConstraintType,
    BooleanOpToken,
    BooleanOpType,
    GraphNodeToken,
    GraphEdgeToken,
    GraphStructureToken,

    # Quantization
    AdaptiveQuantizer,
    UniformQuantizer,
    FeatureVectorQuantizer,
    UVGridQuantizer,

    # Special Token IDs
    PAD_TOKEN_ID,  # 0
    BOS_TOKEN_ID,  # 1
    EOS_TOKEN_ID,  # 2
    SEP_TOKEN_ID,  # 3
    UNK_TOKEN_ID,  # 4
)
```

## Configuration Classes

### PrecisionTier

```python
class PrecisionTier(Enum):
    DRAFT = 6       # 6-bit, 64 levels
    STANDARD = 8    # 8-bit, 256 levels (default)
    PRECISION = 10  # 10-bit, 1024 levels
```

### QuantizationConfig

```python
@dataclass
class QuantizationConfig:
    tier: PrecisionTier = PrecisionTier.STANDARD
    adaptive: bool = True
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    bit_allocation: AdaptiveBitAllocationConfig = field(default_factory=AdaptiveBitAllocationConfig)
```

### NormalizationConfig

```python
@dataclass
class NormalizationConfig:
    center: bool = True              # Center to origin
    uniform_scale: bool = True       # Maintain aspect ratio
    target_range: tuple = (0.0, 1.0) # Output coordinate range
```

### AdaptiveBitAllocationConfig

```python
@dataclass
class AdaptiveBitAllocationConfig:
    base_bits: int = 8              # Minimum bits for flat regions
    max_additional_bits: int = 4    # Extra bits for complex regions
    min_bits: int = 4               # Absolute minimum
    max_bits: int = 16              # Absolute maximum
    curvature_weight: float = 0.7   # Weight for curvature in complexity
    density_weight: float = 0.3     # Weight for feature density
    percentile_low: float = 10.0    # Below this → base_bits
    percentile_high: float = 90.0   # Above this → max bits
```

## Tokenizer Classes

### GeoTokenizer

Main tokenizer for mesh geometry.

```python
class GeoTokenizer:
    def __init__(self, config: QuantizationConfig = None):
        """Initialize with optional configuration."""

    def tokenize(
        self,
        vertices: np.ndarray,  # (N, 3) float32
        faces: np.ndarray      # (F, 3) int64
    ) -> TokenSequence:
        """Tokenize mesh geometry to discrete tokens."""

    def detokenize(self, tokens: TokenSequence) -> np.ndarray:
        """Reconstruct vertices from tokens. Returns (N, 3) float32."""

    def analyze_impact(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> ImpactReport:
        """Analyze quantization quality impact."""
```

### CommandSequenceTokenizer

Tokenizer for CAD construction history.

```python
class CommandSequenceTokenizer:
    def __init__(self, config: CommandTokenizationConfig = None):
        """Initialize with optional configuration."""

    def tokenize(
        self,
        construction_history: list[dict],
        constraints: list[dict] = None
    ) -> TokenSequence:
        """Tokenize CAD command sequence."""

    def parse_construction_history(self, commands: list[dict]) -> list:
        """Parse raw command dicts to internal format."""

    def normalize_sketches(self, parsed: list) -> list:
        """Apply 2D normalization to sketch commands."""

    def normalize_3d(self, parsed: list) -> list:
        """Apply 3D normalization to extrusion commands."""

    def quantize_parameters(self, parsed: list) -> list[CommandToken]:
        """Quantize command parameters to discrete values."""

    def dequantize_parameters(self, tokens: list[CommandToken]) -> list[dict]:
        """Reconstruct command parameters from tokens."""

    def pad_or_truncate(self, tokens: list[CommandToken]) -> list[CommandToken]:
        """Pad to max_length or truncate, preserving sketch pairs."""
```

### GraphTokenizer

Tokenizer for B-Rep topology graphs.

```python
class GraphTokenizer:
    def __init__(self, config: GraphTokenizationConfig = None):
        """Initialize with optional configuration."""

    def fit(
        self,
        node_features: np.ndarray,  # (N, 48) float32
        edge_features: np.ndarray   # (M, 16) float32
    ) -> None:
        """Pre-fit quantization parameters from sample data."""

    def tokenize(
        self,
        node_features: np.ndarray,  # (N, 48) float32
        edge_index: np.ndarray,     # (2, M) int64
        edge_features: np.ndarray,  # (M, 16) float32
        node_types: list[str] = None
    ) -> TokenSequence:
        """Serialize B-Rep graph to token sequence."""
```

### CADVocabulary

Token ↔ integer ID mapping.

```python
class CADVocabulary:
    def __init__(self):
        """Initialize vocabulary with all token types."""

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (73,377 tokens)."""

    def encode(self, command_tokens: list[CommandToken]) -> list[int]:
        """Encode command tokens to integer IDs."""

    def decode(self, token_ids: list[int]) -> list[CommandToken]:
        """Decode integer IDs back to command tokens."""

    def encode_constraint(self, token: ConstraintToken) -> int:
        """Encode single constraint token."""

    def encode_graph_node(self, token: GraphNodeToken) -> list[int]:
        """Encode graph node token to integer IDs."""

    def encode_full_sequence(self, token_seq: TokenSequence) -> list[int]:
        """Encode complete token sequence."""

    def save(self, path: str) -> None:
        """Save vocabulary to file."""

    @classmethod
    def load(cls, path: str) -> "CADVocabulary":
        """Load vocabulary from file."""
```

## Token Dataclasses

### TokenSequence

Container for all token types.

```python
@dataclass
class TokenSequence:
    coordinate_tokens: list[CoordinateToken] = field(default_factory=list)
    geometry_tokens: list[GeometryToken] = field(default_factory=list)
    command_tokens: list[CommandToken] = field(default_factory=list)
    constraint_tokens: list[ConstraintToken] = field(default_factory=list)
    boolean_op_tokens: list[BooleanOpToken] = field(default_factory=list)
    graph_node_tokens: list[GraphNodeToken] = field(default_factory=list)
    graph_edge_tokens: list[GraphEdgeToken] = field(default_factory=list)
    graph_structure_tokens: list[GraphStructureToken] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

### CommandToken

```python
@dataclass
class CommandToken:
    command_type: CommandType
    parameters: list[int]      # 16-element quantized array
    parameter_mask: list[bool] # Which parameters are active
```

### CommandType

```python
class CommandType(IntEnum):
    SOL = 0      # Start of loop (x, y)
    LINE = 1     # Line segment (x1, y1, x2, y2)
    ARC = 2      # Arc (x1, y1, x2, y2, cx, cy)
    CIRCLE = 3   # Circle (cx, cy, r)
    EXTRUDE = 4  # Extrusion (height)
    EOS = 5      # End of sequence
```

## Quantization Classes

### AdaptiveQuantizer

```python
class AdaptiveQuantizer:
    def __init__(self, config: QuantizationConfig):
        """Initialize with configuration."""

    def quantize(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> AdaptiveQuantizationResult:
        """Quantize with adaptive bit allocation."""

    def dequantize(self, result: AdaptiveQuantizationResult) -> np.ndarray:
        """Reconstruct vertices from quantization result."""
```

### UniformQuantizer

```python
class UniformQuantizer:
    def __init__(self, bits: int = 8):
        """Initialize with fixed bit-width."""

    def quantize(self, values: np.ndarray) -> np.ndarray:
        """Quantize to [0, 2^bits - 1] range."""

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        """Reconstruct to [0, 1] range."""
```

## Analysis Classes

### CurvatureAnalyzer

```python
class CurvatureAnalyzer:
    def analyze_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> CurvatureResult:
        """Compute per-vertex curvature for mesh."""

    def analyze_point_cloud(
        self,
        points: np.ndarray,
        n_neighbors: int = 12
    ) -> CurvatureResult:
        """Compute curvature for point cloud via local PCA."""
```

### FeatureDensityAnalyzer

```python
class FeatureDensityAnalyzer:
    def analyze(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> FeatureDensityResult:
        """Compute per-vertex feature density."""
```

## Vertex Processing

### VertexValidator

```python
class VertexValidator:
    def validate(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> VertexValidationReport:
        """Run all validation checks."""
```

**VertexValidationReport fields:**
- `bounds_check: bool` - All coordinates within range
- `collision_check: bool` - No overlapping vertices
- `manifold_check: bool` - Each edge shared by exactly 2 faces
- `degeneracy_check: bool` - No zero-area faces
- `winding_check: bool` - Consistent face orientation
- `euler_check: bool` - Topology consistency (V - E + F = 2)

### VertexClusterer

```python
class VertexClusterer:
    def __init__(self, merge_distance: float = 0.005):
        """Initialize with merge threshold."""

    def cluster(self, vertices: np.ndarray) -> ClusteringResult:
        """Find clusters of near-duplicate vertices."""
```

### CoarseToFineRefiner

```python
class CoarseToFineRefiner:
    def __init__(
        self,
        max_iterations: int = 20,
        learning_rate: float = 0.1,
        convergence_threshold: float = 1e-4,
        face_quality_weight: float = 1.0,
        smoothness_weight: float = 0.5
    ):
        """Initialize refiner with optimization parameters."""

    def refine(
        self,
        coarse_vertices: np.ndarray,
        target_points: np.ndarray = None,
        face_indices: np.ndarray = None,
        constraints: list[dict] = None
    ) -> RefinementResult:
        """Refine coarse vertices to sub-quantization accuracy."""
```

## Impact Analysis

### QuantizationImpactAnalyzer

```python
class QuantizationImpactAnalyzer:
    def analyze(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        config: QuantizationConfig
    ) -> ImpactReport:
        """Analyze quantization quality impact."""
```

**ImpactReport fields:**
- `tier: PrecisionTier`
- `hausdorff_distance: float` - Max bidirectional error
- `mean_error: float` - Mean per-vertex error
- `max_error: float` - Max per-vertex error
- `relationship_preservation_rate: float` - Preserved geometric relationships
- `feature_loss: FeatureLossMetric` - Collapsed vertex statistics
- `total_bits_used: int`
- `mean_bits_per_vertex: float`
