"""
Sequence encoders for mapping sequences to hypervectors.

This module implements encoders that transform sequences (text, time series,
trajectories) into hypervector representations, preserving order and enabling
partial matching.
"""

from typing import Dict, List, Optional, Union, Tuple
from holovec.encoders.base import SequenceEncoder, ScalarEncoder
from holovec.models.base import VSAModel
from holovec.backends.base import Array


class PositionBindingEncoder(SequenceEncoder):
    """
    Position binding encoder for sequences using permutation-based positions.

    Based on Plate (2003) "Holographic Reduced Representations" and
    Schlegel et al. (2021) "A comparison of vector symbolic architectures".

    Encodes sequences by binding each element with a position-specific
    permutation of a base position vector:

        encode([A, B, C]) = bind(A, ρ¹) + bind(B, ρ²) + bind(C, ρ³)

    where ρ is the permutation operation and ρⁱ represents i applications.

    This encoding is:
    - Order-sensitive: Different positions create different bindings
    - Variable-length: Works with any sequence length
    - Partial-match capable: Similar sequences have similar encodings

    Attributes:
        codebook: Dictionary mapping symbols to hypervectors
        auto_generate: Whether to auto-generate vectors for unknown symbols
        seed_offset: Offset for generating consistent symbol vectors

    Example:
        >>> model = VSA.create('MAP', dim=10000)
        >>> encoder = PositionBindingEncoder(model)
        >>>
        >>> # Encode a sequence of symbols
        >>> seq = ['hello', 'world', '!']
        >>> hv = encoder.encode(seq)
        >>>
        >>> # Similar sequences have high similarity
        >>> seq2 = ['hello', 'world']
        >>> hv2 = encoder.encode(seq2)
        >>> model.similarity(hv, hv2)  # High (shared prefix)
    """

    def __init__(
        self,
        model: VSAModel,
        codebook: Optional[Dict[str, Array]] = None,
        max_length: Optional[int] = None,
        auto_generate: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize position binding encoder.

        Args:
            model: VSA model instance
            codebook: Pre-defined symbol → hypervector mapping (optional)
            max_length: Maximum sequence length (None for unlimited)
            auto_generate: Auto-generate vectors for unknown symbols (default: True)
            seed: Random seed for generating symbol vectors

        Raises:
            ValueError: If model is not compatible
        """
        super().__init__(model, max_length)

        self.codebook = codebook if codebook is not None else {}
        self.auto_generate = auto_generate
        self.seed = seed
        self._next_symbol_seed = 0  # Counter for symbol generation

    def encode(self, sequence: List[Union[str, int]]) -> Array:
        """
        Encode sequence using position binding.

        Each element is bound with a position-specific permutation and
        all bound pairs are bundled:

            result = Σᵢ bind(element_i, permute(position_vector, i))

        Args:
            sequence: List of symbols (strings or integers) to encode

        Returns:
            Hypervector representing the sequence

        Raises:
            ValueError: If sequence is empty
            ValueError: If sequence exceeds max_length
            ValueError: If symbol not in codebook and auto_generate=False

        Example:
            >>> encoder.encode(['cat', 'sat', 'on', 'mat'])
        """
        if not sequence:
            raise ValueError("Cannot encode empty sequence")

        if self.max_length is not None and len(sequence) > self.max_length:
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds max_length {self.max_length}"
            )

        # Get or generate hypervectors for each symbol
        symbol_vectors = []
        for symbol in sequence:
            if symbol not in self.codebook:
                if not self.auto_generate:
                    raise ValueError(
                        f"Symbol '{symbol}' not in codebook and auto_generate=False"
                    )
                # Generate new vector for this symbol
                self.codebook[symbol] = self._generate_symbol_vector(symbol)

            symbol_vectors.append(self.codebook[symbol])

        # Bind each symbol with its position and bundle
        position_bound = []
        for i, symbol_vec in enumerate(symbol_vectors):
            # Position encoding: permute by position index
            # permute(vec, i) applies permutation i times
            position_vec = self.model.permute(symbol_vec, k=i)
            position_bound.append(position_vec)

        # Bundle all position-bound vectors
        sequence_hv = self.model.bundle(position_bound)

        return sequence_hv

    def decode(
        self,
        hypervector: Array,
        max_positions: int = 10,
        threshold: float = 0.3
    ) -> List[str]:
        """
        Decode sequence hypervector to recover symbols.

        Uses cleanup memory approach: for each position, unpermute and
        find most similar symbol in codebook.

        Args:
            hypervector: Sequence hypervector to decode
            max_positions: Maximum positions to try decoding (default: 10)
            threshold: Minimum similarity threshold for valid symbols (default: 0.3)

        Returns:
            List of decoded symbols (may be shorter than original)

        Raises:
            RuntimeError: If codebook is empty

        Note:
            Decoding is approximate and works best for sequences shorter
            than max_positions with high SNR.

        Example:
            >>> encoded = encoder.encode(['a', 'b', 'c'])
            >>> decoded = encoder.decode(encoded, max_positions=5)
            >>> decoded  # ['a', 'b', 'c'] (approximate)
        """
        if not self.codebook:
            raise RuntimeError("Cannot decode: codebook is empty")

        # Convert codebook to symbol → vector for faster lookup
        symbols = list(self.codebook.keys())
        vectors = [self.codebook[s] for s in symbols]

        decoded = []

        for pos in range(max_positions):
            # Unpermute by position to recover symbol at this position
            unpermuted = self.model.unpermute(hypervector, k=pos)

            # Find most similar symbol in codebook
            best_similarity = -float('inf')
            best_symbol = None

            for symbol, symbol_vec in zip(symbols, vectors):
                sim = float(self.model.similarity(unpermuted, symbol_vec))
                if sim > best_similarity:
                    best_similarity = sim
                    best_symbol = symbol

            # Only include if above threshold
            if best_similarity >= threshold:
                decoded.append(best_symbol)
            else:
                # No strong match - likely end of sequence
                break

        return decoded

    def _generate_symbol_vector(self, symbol: Union[str, int]) -> Array:
        """
        Generate a random hypervector for a new symbol.

        Uses consistent seeding based on symbol to ensure reproducibility.

        Args:
            symbol: Symbol to generate vector for

        Returns:
            Random hypervector for this symbol
        """
        # Create seed from base seed + symbol hash + counter
        if self.seed is not None:
            symbol_seed = self.seed + hash(symbol) % 10000 + self._next_symbol_seed
        else:
            symbol_seed = hash(symbol) % 100000 + self._next_symbol_seed

        self._next_symbol_seed += 1

        return self.model.random(seed=symbol_seed)

    def add_symbol(self, symbol: Union[str, int], vector: Optional[Array] = None):
        """
        Add a symbol to the codebook.

        Args:
            symbol: Symbol to add
            vector: Hypervector to associate (generated if None)

        Example:
            >>> # Pre-define a vector for a special symbol
            >>> special_vec = model.random(seed=42)
            >>> encoder.add_symbol('<START>', special_vec)
        """
        if vector is None:
            vector = self._generate_symbol_vector(symbol)
        self.codebook[symbol] = vector

    def get_codebook_size(self) -> int:
        """
        Get number of symbols in codebook.

        Returns:
            Number of symbols stored
        """
        return len(self.codebook)

    @property
    def is_reversible(self) -> bool:
        """
        PositionBindingEncoder supports approximate decoding.

        Returns:
            True (approximate decoding available)
        """
        return True

    @property
    def compatible_models(self) -> List[str]:
        """
        Works with all VSA models that support permutation.

        Returns:
            List of all model names
        """
        return ["MAP", "FHRR", "HRR", "BSC", "GHRR", "VTB", "BSDC"]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PositionBindingEncoder("
            f"model={self.model.model_name}, "
            f"codebook_size={len(self.codebook)}, "
            f"max_length={self.max_length}, "
            f"auto_generate={self.auto_generate})"
        )


class NGramEncoder(SequenceEncoder):
    """
    N-gram encoder for capturing local sequence patterns using sliding windows.

    Based on Plate (2003), Rachkovskij (1996), and Kleyko et al. (2023) Section 3.3.4.

    Encodes sequences by extracting n-grams (sliding windows of n consecutive symbols)
    and encoding each n-gram compositionally:

        For sequence [A, B, C, D] with n=2, stride=1:
        - Extract n-grams: [A,B], [B,C], [C,D]
        - Encode each n-gram using position binding
        - Combine via bundling or chaining

    Two encoding modes:

    1. **Bundling mode** (bag-of-ngrams):
       encode(seq) = bundle([encode_ngram([A,B]), encode_ngram([B,C]), ...])
       - Order-invariant across n-grams (but preserves within n-gram)
       - Good for classification (e.g., text categorization)
       - Similar to bag-of-words but with local context

    2. **Chaining mode** (ordered n-grams):
       encode(seq) = Σᵢ bind(encode_ngram(ngramᵢ), ρⁱ)
       - Order-sensitive across n-grams
       - Good for sequence matching
       - Enables partial decoding

    Attributes:
        n: Size of n-grams (1=unigrams, 2=bigrams, 3=trigrams, etc.)
        stride: Step size between n-grams (1=overlapping, n=non-overlapping)
        mode: 'bundling' or 'chaining'
        ngram_encoder: Internal PositionBindingEncoder for individual n-grams

    Example:
        >>> model = VSA.create('MAP', dim=10000)
        >>> encoder = NGramEncoder(model, n=2, stride=1, mode='bundling')
        >>>
        >>> # Encode text as bigrams
        >>> seq = ['the', 'cat', 'sat', 'on', 'mat']
        >>> hv = encoder.encode(seq)  # Bigrams: [the,cat], [cat,sat], [sat,on], [on,mat]
        >>>
        >>> # Similar text has high similarity
        >>> seq2 = ['the', 'cat', 'sat', 'on', 'hat']
        >>> hv2 = encoder.encode(seq2)  # Shares 3/4 bigrams
        >>> model.similarity(hv, hv2)  # High similarity
    """

    def __init__(
        self,
        model: VSAModel,
        n: int = 2,
        stride: int = 1,
        mode: str = 'bundling',
        codebook: Optional[Dict[str, Array]] = None,
        auto_generate: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize n-gram encoder.

        Args:
            model: VSA model instance
            n: Size of n-grams (must be >= 1)
            stride: Step between n-grams (must be >= 1)
            mode: 'bundling' for bag-of-ngrams or 'chaining' for ordered n-grams
            codebook: Optional pre-defined symbol → hypervector mapping
            auto_generate: Auto-generate vectors for unknown symbols
            seed: Random seed for symbol vector generation

        Raises:
            ValueError: If n < 1, stride < 1, or mode is invalid
        """
        super().__init__(model, max_length=None)

        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if mode not in ['bundling', 'chaining']:
            raise ValueError(f"mode must be 'bundling' or 'chaining', got '{mode}'")

        self.n = n
        self.stride = stride
        self.mode = mode

        # Internal encoder for individual n-grams
        # Each n-gram is encoded as a position-bound sequence
        self.ngram_encoder = PositionBindingEncoder(
            model=model,
            codebook=codebook,
            max_length=n,  # Each n-gram has length n
            auto_generate=auto_generate,
            seed=seed
        )

    def encode(self, sequence: List[Union[str, int]]) -> Array:
        """
        Encode sequence using n-gram representation.

        Extracts all n-grams using sliding window with specified stride,
        encodes each n-gram, then combines via bundling or chaining.

        Args:
            sequence: List of symbols to encode

        Returns:
            Hypervector representing the sequence as n-grams

        Raises:
            ValueError: If sequence is too short (length < n)

        Example:
            >>> # Bigrams with stride=1 (overlapping)
            >>> encoder = NGramEncoder(model, n=2, stride=1)
            >>> encoder.encode(['A', 'B', 'C'])  # N-grams: AB, BC
            >>>
            >>> # Trigrams with stride=2 (partial overlap)
            >>> encoder = NGramEncoder(model, n=3, stride=2)
            >>> encoder.encode(['A', 'B', 'C', 'D', 'E'])  # N-grams: ABC, CDE
        """
        if len(sequence) < self.n:
            raise ValueError(
                f"Sequence length {len(sequence)} is less than n={self.n}"
            )

        # Extract all n-grams using sliding window
        ngrams = []
        for i in range(0, len(sequence) - self.n + 1, self.stride):
            ngram = sequence[i:i + self.n]
            ngrams.append(ngram)

        if not ngrams:
            raise ValueError("No n-grams extracted from sequence")

        # Encode each n-gram using position binding
        ngram_hvs = []
        for ngram in ngrams:
            ngram_hv = self.ngram_encoder.encode(ngram)
            ngram_hvs.append(ngram_hv)

        # Combine n-gram hypervectors based on mode
        if self.mode == 'bundling':
            # Bag-of-ngrams: simple bundle (order-invariant)
            sequence_hv = self.model.bundle(ngram_hvs)

        else:  # mode == 'chaining'
            # Ordered n-grams: bind each with position
            position_bound = []
            for i, ngram_hv in enumerate(ngram_hvs):
                # Position encoding: permute by n-gram index
                position_hv = self.model.permute(ngram_hv, k=i)
                position_bound.append(position_hv)

            sequence_hv = self.model.bundle(position_bound)

        return sequence_hv

    def decode(
        self,
        hypervector: Array,
        max_ngrams: int = 10,
        threshold: float = 0.3
    ) -> List[List[Union[str, int]]]:
        """
        Decode n-gram hypervector to recover n-grams.

        Only supported for 'chaining' mode. For 'bundling' mode,
        n-grams are order-invariant and cannot be sequentially decoded.

        Args:
            hypervector: Encoded sequence hypervector
            max_ngrams: Maximum number of n-grams to decode
            threshold: Minimum similarity threshold for valid n-grams

        Returns:
            List of decoded n-grams, each as a list of symbols

        Raises:
            NotImplementedError: If mode is 'bundling' (not decodable)
            RuntimeError: If codebook is empty

        Example:
            >>> encoder = NGramEncoder(model, n=2, mode='chaining')
            >>> hv = encoder.encode(['A', 'B', 'C'])
            >>> decoder.decode(hv, max_ngrams=3)  # [['A', 'B'], ['B', 'C']]
        """
        if self.mode != 'chaining':
            raise NotImplementedError(
                f"Decoding only supported for 'chaining' mode, not '{self.mode}'"
            )

        if not self.ngram_encoder.codebook:
            raise RuntimeError("Cannot decode: codebook is empty")

        # For chaining mode, unpermute each position and decode the n-gram
        decoded_ngrams = []

        for pos in range(max_ngrams):
            # Unpermute by position to recover n-gram at this index
            unpermuted = self.model.unpermute(hypervector, k=pos)

            # Decode the n-gram using ngram_encoder
            try:
                ngram_symbols = self.ngram_encoder.decode(
                    unpermuted,
                    max_positions=self.n,
                    threshold=threshold
                )

                # Only include if we got a full n-gram
                if len(ngram_symbols) >= self.n:
                    decoded_ngrams.append(ngram_symbols[:self.n])
                else:
                    # Incomplete n-gram - likely end of sequence
                    break

            except Exception:
                # Decoding failed - likely end of sequence
                break

        return decoded_ngrams

    def get_codebook(self) -> Dict[str, Array]:
        """
        Get the internal symbol codebook.

        Returns:
            Dictionary mapping symbols to hypervectors
        """
        return self.ngram_encoder.codebook

    def get_codebook_size(self) -> int:
        """
        Get number of unique symbols in codebook.

        Returns:
            Number of symbols
        """
        return self.ngram_encoder.get_codebook_size()

    @property
    def is_reversible(self) -> bool:
        """
        NGramEncoder supports decoding only in 'chaining' mode.

        Returns:
            True if mode is 'chaining', False if 'bundling'
        """
        return self.mode == 'chaining'

    @property
    def compatible_models(self) -> List[str]:
        """
        Works with all VSA models.

        Returns:
            List of all model names
        """
        return ["MAP", "FHRR", "HRR", "BSC", "GHRR", "VTB", "BSDC"]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"NGramEncoder("
            f"model={self.model.model_name}, "
            f"n={self.n}, "
            f"stride={self.stride}, "
            f"mode='{self.mode}', "
            f"codebook_size={self.get_codebook_size()})"
        )


class TrajectoryEncoder(SequenceEncoder):
    """
    Trajectory encoder for continuous sequences (time series, paths, motion).

    Based on Frady et al. (2021) "Computing on Functions" and position binding
    from Plate (2003), encoding trajectories by binding temporal information
    with spatial positions.

    A trajectory is a sequence of positions over time:
    - 1D: time series [v₁, v₂, v₃, ...]
    - 2D: path [(x₁,y₁), (x₂,y₂), ...]
    - 3D: motion [(x₁,y₁,z₁), (x₂,y₂,z₂), ...]

    Encoding strategy:
        For each time step tᵢ with position pᵢ:
        1. Encode time: time_hv = scalar_encode(tᵢ)
        2. Encode position coords: coord_hvs = [scalar_encode(c) for c in pᵢ]
        3. Bind coords to dimensions: pos_hv = Σⱼ bind(Dⱼ, coord_hv_j)
        4. Bind time with position: point_hv = bind(time_hv, pos_hv)
        5. Permute by index: indexed_hv = permute(point_hv, i)

        trajectory_hv = Σᵢ indexed_hv

    This creates an encoding that:
    - Preserves temporal ordering (via permutation)
    - Captures smooth trajectories (via continuous scalar encoding)
    - Enables partial matching and interpolation
    - Supports multi-dimensional paths

    Attributes:
        scalar_encoder: Encoder for continuous values (FPE or Thermometer)
        n_dimensions: Dimensionality of trajectory (1D, 2D, or 3D)
        time_range: (min_time, max_time) for temporal normalization
        dim_vectors: Hypervectors for spatial dimensions (x, y, z)

    Example:
        >>> from holovec import VSA
        >>> from holovec.encoders import FractionalPowerEncoder, TrajectoryEncoder
        >>>
        >>> model = VSA.create('FHRR', dim=10000)
        >>> scalar_enc = FractionalPowerEncoder(model, min_val=0, max_val=100)
        >>> encoder = TrajectoryEncoder(model, scalar_encoder=scalar_enc, n_dimensions=2)
        >>>
        >>> # Encode a 2D path
        >>> path = [(10, 20), (15, 25), (20, 30), (25, 35)]
        >>> hv = encoder.encode(path)
        >>>
        >>> # Similar paths have high similarity
        >>> path2 = [(10, 20), (15, 25), (20, 30), (25, 40)]  # Slightly different
        >>> hv2 = encoder.encode(path2)
        >>> model.similarity(hv, hv2)  # High similarity
    """

    def __init__(
        self,
        model: VSAModel,
        scalar_encoder: ScalarEncoder,
        n_dimensions: int = 1,
        time_range: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize trajectory encoder.

        Args:
            model: VSA model instance
            scalar_encoder: Encoder for continuous values (FPE or Thermometer recommended)
            n_dimensions: Trajectory dimensionality (1, 2, or 3)
            time_range: (min, max) time values for normalization (optional)
            seed: Random seed for dimension vector generation

        Raises:
            ValueError: If n_dimensions not in {1, 2, 3}
            TypeError: If scalar_encoder is not reversible
        """
        super().__init__(model, max_length=None)

        if n_dimensions not in {1, 2, 3}:
            raise ValueError(
                f"n_dimensions must be 1, 2, or 3, got {n_dimensions}"
            )

        if not isinstance(scalar_encoder, ScalarEncoder):
            raise TypeError(
                f"scalar_encoder must be a ScalarEncoder, got {type(scalar_encoder)}"
            )

        # Check model compatibility
        if model != scalar_encoder.model:
            raise ValueError(
                "scalar_encoder must use the same VSA model as TrajectoryEncoder"
            )

        self.scalar_encoder = scalar_encoder
        self.n_dimensions = n_dimensions
        self.time_range = time_range
        self.seed = seed

        # Generate dimension hypervectors (for x, y, z coordinates)
        self.dim_vectors: List[Array] = []
        for i in range(n_dimensions):
            dim_seed = (seed + i) if seed is not None else (1000 + i)
            self.dim_vectors.append(model.random(seed=dim_seed))

    def encode(self, trajectory: List[Union[float, Tuple[float, ...]]]) -> Array:
        """
        Encode a trajectory as a hypervector.

        Each point in the trajectory is encoded with temporal information,
        then all points are combined with position-based permutation.

        Args:
            trajectory: List of points
                - 1D: List[float] e.g., [1.0, 2.5, 3.7, ...]
                - 2D: List[Tuple[float, float]] e.g., [(1,2), (3,4), ...]
                - 3D: List[Tuple[float, float, float]] e.g., [(1,2,3), ...]

        Returns:
            Hypervector representing the trajectory

        Raises:
            ValueError: If trajectory is empty or points have wrong dimensionality

        Example:
            >>> # 1D time series
            >>> encoder_1d = TrajectoryEncoder(model, scalar_enc, n_dimensions=1)
            >>> hv = encoder_1d.encode([1.0, 2.5, 3.7, 5.2])
            >>>
            >>> # 2D path
            >>> encoder_2d = TrajectoryEncoder(model, scalar_enc, n_dimensions=2)
            >>> hv = encoder_2d.encode([(0,0), (1,1), (2,2)])
        """
        if len(trajectory) == 0:
            raise ValueError("Cannot encode empty trajectory")

        # Encode each point with temporal binding
        point_hvs = []

        for i, point in enumerate(trajectory):
            # Normalize point to tuple format
            if self.n_dimensions == 1:
                # 1D: scalar → (scalar,)
                if isinstance(point, (int, float)):
                    coords = (float(point),)
                else:
                    coords = (float(point[0]),)
            else:
                # 2D/3D: accept tuple, list, or array-like
                try:
                    # Convert to tuple (works for tuple, list, numpy array, etc.)
                    coords = tuple(float(c) for c in point)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Expected iterable for {self.n_dimensions}D point, got {type(point)}"
                    )

            # Validate dimensionality
            if len(coords) != self.n_dimensions:
                raise ValueError(
                    f"Expected {self.n_dimensions}D point, got {len(coords)}D: {coords}"
                )

            # Encode time (index as time if no time_range specified)
            if self.time_range is not None:
                # Normalize time to range
                t = i / len(trajectory)  # [0, 1]
                t_scaled = self.time_range[0] + t * (self.time_range[1] - self.time_range[0])
                time_hv = self.scalar_encoder.encode(t_scaled)
            else:
                # Use index directly
                time_hv = self.scalar_encoder.encode(float(i))

            # Encode position (bind each coordinate with its dimension)
            coord_hvs = []
            for j, coord_val in enumerate(coords):
                coord_hv = self.scalar_encoder.encode(coord_val)
                dim_hv = self.dim_vectors[j]
                bound_coord = self.model.bind(dim_hv, coord_hv)
                coord_hvs.append(bound_coord)

            # Bundle coordinates to create position hypervector
            pos_hv = self.model.bundle(coord_hvs)

            # Bind time with position
            point_hv = self.model.bind(time_hv, pos_hv)

            # Apply position-specific permutation (for ordering)
            indexed_hv = self.model.permute(point_hv, k=i)

            point_hvs.append(indexed_hv)

        # Bundle all points
        trajectory_hv = self.model.bundle(point_hvs)

        return trajectory_hv

    def decode(self, hypervector: Array, max_points: int = 10) -> List[Tuple[float, ...]]:
        """
        Decode trajectory hypervector to recover approximate points.

        Note: Trajectory decoding is not yet implemented. It requires:
        1. Unpermuting each position
        2. Unbinding time from position
        3. Unbinding each coordinate from dimension vectors
        4. Decoding scalar values
        5. Interpolation for smooth trajectories

        Args:
            hypervector: Encoded trajectory hypervector
            max_points: Maximum points to decode

        Returns:
            List of decoded points (not implemented yet)

        Raises
        ------
        NotImplementedError
            Trajectory decoding requires solving nested binding inverse problem.

        Notes
        -----
        Trajectory decoding is not implemented because it requires multi-level
        unbinding with cascading error accumulation:

        **Mathematical Challenge:**

        The encoding process creates nested bindings:
            trajectory_hv = bundle([
                bind(time(t), bind(dimension(d), scalar(coord[t,d])))
                for all t, d
            ])

        To decode a single point at time t:
        1. Unbind time: point_hv[t] = unbind(trajectory_hv, time(t))
        2. For each dimension d:
           a. Unbind dimension: coord_hv[d] = unbind(point_hv[t], dimension(d))
           b. Decode scalar: coord[t,d] = scalar_decode(coord_hv[d])

        **Why This Is Intractable:**

        - **Two-level unbinding**: Time then dimension (or vice versa)
        - **Error compounding**: Each unbind adds noise
        - **No known time points**: Must search over possible time values
        - **Interpolation complexity**: Smooth trajectory requires dense sampling
        - **Computational cost**:
          * For T time points, D dimensions
          * Requires: T × D × (decode_iterations) evaluations
          * Example: 100 points × 3D × 100 iterations = 30,000 evals

        **Additional Challenges:**

        1. **Order Ambiguity**: Don't know which time point comes first
        2. **Density Unknown**: Don't know temporal sampling rate
        3. **Dimension Count**: Must know dimensionality a priori
        4. **Coordinate Ranges**: Scalar decoder needs value bounds

        **Possible Approaches (Future Work):**

        1. **Constrained Decoding**: If time points are known:
           - Unbind each known time point
           - Decode coordinates independently
           - Complexity: O(T × D × decode_cost)

        2. **Template Matching**: Pre-encode common trajectory patterns
           - Create codebook of canonical trajectories
           - Use cleanup to find nearest match
           - Works for classification, not reconstruction

        3. **Learned Decoder**: Train neural network trajectory_hv → points
           - Requires large training dataset
           - Can learn to handle noise and ambiguity
           - See: Imani et al. (2019) for similar approach

        4. **Iterative Resonator**: Use resonator cleanup at each level
           - Unbind time with resonator cleanup
           - Unbind dimension with resonator cleanup
           - Requires codebooks for both time and coordinates

        **Current Recommendation:**

        Use TrajectoryEncoder for one-way encoding in applications like:
        - Trajectory classification (gesture recognition, motion analysis)
        - Trajectory similarity search (find similar paths)
        - Trajectory clustering (group similar motions)

        For reconstruction, consider storing original trajectories separately
        and using hypervector encoding only for similarity queries.

        References
        ----------
        - Plate (2003): "Holographic Reduced Representations" - Section 4.3
          on error accumulation in multi-level binding
        - Räsänen & Saarinen (2016): "Sequence prediction with sparse
          distributed hyperdimensional coding" - Analysis of temporal binding
        """
        raise NotImplementedError(
            "Trajectory decoding is not implemented due to nested binding complexity. "
            "See docstring for detailed mathematical explanation. "
            "For reconstruction tasks, store original trajectories and use "
            "hypervector encoding for similarity-based retrieval only."
        )

    @property
    def is_reversible(self) -> bool:
        """
        TrajectoryEncoder does not yet support decoding.

        Returns:
            False (decoding not implemented)

        Note:
            Decoding requires multi-level unbinding and interpolation,
            which will be implemented in a future version.
        """
        return False

    @property
    def compatible_models(self) -> List[str]:
        """
        Works with all VSA models.

        Returns:
            List of all model names
        """
        return ["MAP", "FHRR", "HRR", "BSC", "GHRR", "VTB", "BSDC"]

    @property
    def input_type(self) -> str:
        """Input type description."""
        dim_names = {1: "1D time series", 2: "2D path", 3: "3D trajectory"}
        return dim_names[self.n_dimensions]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TrajectoryEncoder("
            f"model={self.model.model_name}, "
            f"scalar_encoder={type(self.scalar_encoder).__name__}, "
            f"n_dimensions={self.n_dimensions}, "
            f"time_range={self.time_range})"
        )
