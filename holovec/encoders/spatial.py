"""
Spatial encoders for image and grid data.

This module provides encoders for spatial data structures like images,
where both position and value information must be encoded.
"""

from typing import Optional, Tuple, Union, List
from holovec.models.base import VSAModel
from holovec.encoders.base import Encoder
from holovec.encoders.scalar import ScalarEncoder
from holovec.backends.base import Array


class ImageEncoder(Encoder):
    """
    Image encoder for 2D images (grayscale, RGB, or RGBA).

    Encodes images by binding spatial positions (x, y) with pixel values.
    For color images, each channel is bound to a channel dimension vector
    before being combined with position information.

    Encoding strategy:
        For each pixel at position (x, y) with value v:
        1. Encode position: pos_hv = bundle([bind(X, enc(x)), bind(Y, enc(y))])
        2. Encode value(s):
           - Grayscale: val_hv = enc(v)
           - RGB: val_hv = bundle([bind(R, enc(r)), bind(G, enc(g)), bind(B, enc(b))])
        3. Bind position with value: pixel_hv = bind(pos_hv, val_hv)
        4. Bundle all pixels: image_hv = bundle([all pixel_hvs])

    This creates a distributed representation that preserves both spatial
    structure and pixel values, enabling similarity-based image comparison.

    Parameters
    ----------
    model : VSAModel
        The VSA model to use for encoding operations.
    scalar_encoder : ScalarEncoder
        Encoder for continuous pixel values (0-255 typically).
    normalize_pixels : bool, optional
        Whether to normalize pixel values to [0, 1] before encoding.
        Default is True.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Attributes
    ----------
    n_channels : int
        Number of channels in the last encoded image (1, 3, or 4).
    image_shape : tuple
        Shape (height, width, channels) of the last encoded image.

    Examples
    --------
    >>> from holovec import VSA
    >>> from holovec.encoders import ImageEncoder, ThermometerEncoder
    >>> import numpy as np
    >>>
    >>> model = VSA.create('MAP', dim=10000, seed=42)
    >>> scalar_enc = ThermometerEncoder(model, min_val=0, max_val=1, n_bins=256, seed=42)
    >>> encoder = ImageEncoder(model, scalar_enc, normalize_pixels=True, seed=42)
    >>>
    >>> # Encode a small grayscale image
    >>> image = np.array([[100, 150], [200, 250]], dtype=np.uint8)
    >>> hv = encoder.encode(image)
    >>> print(hv.shape)  # (10000,)
    >>>
    >>> # Encode RGB image
    >>> rgb_image = np.random.randint(0, 256, (28, 28, 3), dtype=np.uint8)
    >>> hv_rgb = encoder.encode(rgb_image)
    """

    def __init__(
        self,
        model: VSAModel,
        scalar_encoder: ScalarEncoder,
        normalize_pixels: bool = True,
        seed: Optional[int] = None
    ):
        """Initialize ImageEncoder."""
        # Validate and set scalar_encoder BEFORE calling super().__init__
        # because base class checks compatible_models which references it
        if not isinstance(scalar_encoder, ScalarEncoder):
            raise TypeError(
                f"scalar_encoder must be a ScalarEncoder, got {type(scalar_encoder)}"
            )

        if scalar_encoder.model != model:
            raise ValueError(
                "scalar_encoder must use the same VSA model as the ImageEncoder"
            )

        self.scalar_encoder = scalar_encoder
        self.normalize_pixels = normalize_pixels

        super().__init__(model)

        # Generate dimension vectors for spatial coordinates
        base_seed = seed if seed is not None else 2000
        self.X = model.random(seed=base_seed)      # X dimension
        self.Y = model.random(seed=base_seed + 1)  # Y dimension

        # Generate dimension vectors for color channels (RGB/RGBA)
        self.R = model.random(seed=base_seed + 2)  # Red channel
        self.G = model.random(seed=base_seed + 3)  # Green channel
        self.B = model.random(seed=base_seed + 4)  # Blue channel
        self.A = model.random(seed=base_seed + 5)  # Alpha channel

        # Track last encoded image properties
        self.n_channels: Optional[int] = None
        self.image_shape: Optional[Tuple[int, ...]] = None

    def encode(self, image: Union[Array, "numpy.ndarray"]) -> Array:
        """
        Encode an image into a hypervector.

        Parameters
        ----------
        image : array-like
            Image array with shape (height, width) for grayscale or
            (height, width, channels) for color images.
            Pixel values should be in range [0, 255] for uint8 or
            [0, 1] for float.
            Typically a NumPy array from PIL, OpenCV, or similar libraries.

        Returns
        -------
        Array
            Hypervector encoding of the image.

        Raises
        ------
        ValueError
            If image has invalid shape or number of channels.

        Notes
        -----
        This encoder accepts images as NumPy arrays (the standard format from
        image libraries like PIL, OpenCV, scikit-image) and processes them using
        the configured backend. While input must be NumPy, internal VSA operations
        use the model's backend (NumPy/PyTorch/JAX).
        """
        # Import numpy locally to avoid module-level backend dependency
        # Images from external sources (PIL, OpenCV) are numpy arrays
        import numpy as _np

        # Convert to numpy array if needed (handles lists, tuples, etc.)
        if not isinstance(image, _np.ndarray):
            image = _np.array(image)

        # Validate and normalize image shape
        if image.ndim == 2:
            # Grayscale image
            height, width = image.shape
            n_channels = 1
            # Add channel dimension: (H, W) -> (H, W, 1)
            image = _np.expand_dims(image, axis=-1)
        elif image.ndim == 3:
            height, width, n_channels = image.shape
            if n_channels not in [1, 3, 4]:
                raise ValueError(
                    f"Image must have 1, 3, or 4 channels, got {n_channels}"
                )
        else:
            raise ValueError(
                f"Image must be 2D (grayscale) or 3D (color), got shape {image.shape}"
            )

        # Store image properties
        self.n_channels = n_channels
        self.image_shape = (height, width, n_channels)

        # Normalize pixel values if requested
        if self.normalize_pixels:
            # Check dtype using string representation to avoid dtype dependency
            dtype_str = str(image.dtype)
            if 'uint8' in dtype_str:
                image = image.astype(_np.float32) / 255.0
            elif 'int' in dtype_str:
                # Other integer types: normalize assuming 0-255 range
                image = image.astype(_np.float32) / 255.0
            # If already float, assume it's in [0, 1]

        # Encode all pixels
        pixel_hvs = []

        for y in range(height):
            for x in range(width):
                # Encode spatial position
                x_hv = self.scalar_encoder.encode(float(x))
                y_hv = self.scalar_encoder.encode(float(y))

                x_bound = self.model.bind(self.X, x_hv)
                y_bound = self.model.bind(self.Y, y_hv)
                pos_hv = self.model.bundle([x_bound, y_bound])

                # Encode pixel value(s)
                if n_channels == 1:
                    # Grayscale: just encode the intensity
                    val_hv = self.scalar_encoder.encode(float(image[y, x, 0]))
                elif n_channels == 3:
                    # RGB: bind each channel to its dimension vector
                    r_hv = self.scalar_encoder.encode(float(image[y, x, 0]))
                    g_hv = self.scalar_encoder.encode(float(image[y, x, 1]))
                    b_hv = self.scalar_encoder.encode(float(image[y, x, 2]))

                    r_bound = self.model.bind(self.R, r_hv)
                    g_bound = self.model.bind(self.G, g_hv)
                    b_bound = self.model.bind(self.B, b_hv)

                    val_hv = self.model.bundle([r_bound, g_bound, b_bound])
                else:  # n_channels == 4
                    # RGBA: bind each channel including alpha
                    r_hv = self.scalar_encoder.encode(float(image[y, x, 0]))
                    g_hv = self.scalar_encoder.encode(float(image[y, x, 1]))
                    b_hv = self.scalar_encoder.encode(float(image[y, x, 2]))
                    a_hv = self.scalar_encoder.encode(float(image[y, x, 3]))

                    r_bound = self.model.bind(self.R, r_hv)
                    g_bound = self.model.bind(self.G, g_hv)
                    b_bound = self.model.bind(self.B, b_hv)
                    a_bound = self.model.bind(self.A, a_hv)

                    val_hv = self.model.bundle([r_bound, g_bound, b_bound, a_bound])

                # Bind position with value
                pixel_hv = self.model.bind(pos_hv, val_hv)
                pixel_hvs.append(pixel_hv)

        # Bundle all pixels to create image hypervector
        image_hv = self.model.bundle(pixel_hvs)

        return image_hv

    def decode(
        self,
        hypervector: Array,
        height: int,
        width: int,
        n_channels: int = 1
    ) -> "numpy.ndarray":
        """
        Decode a hypervector to reconstruct an approximate image.

        Note: Image decoding is approximate and requires knowing the target
        image dimensions. Reconstruction quality depends on the scalar encoder's
        decoding capabilities and may require candidate value search.

        Parameters
        ----------
        hypervector : Array
            The hypervector to decode.
        height : int
            Target image height.
        width : int
            Target image width.
        n_channels : int, optional
            Number of channels (1, 3, or 4). Default is 1.

        Returns
        -------
        np.ndarray
            Reconstructed image with shape (height, width) for grayscale
            or (height, width, n_channels) for color.

        Raises
        ------
        NotImplementedError
            Image decoding is computationally intractable without additional constraints.

        Notes
        -----
        Image decoding is not implemented because it requires solving a high-dimensional
        inverse problem that is fundamentally ill-posed:

        **Mathematical Challenge:**

        The encoding process binds pixel values with position vectors:
            image_hv = bundle([bind(position(i,j), scalar(pixel[i,j])) for all i,j])

        To decode, we must:
        1. Unbind each position: pixel_hv[i,j] = unbind(image_hv, position(i,j))
        2. Decode each scalar: pixel[i,j] = scalar_decode(pixel_hv[i,j])

        **Why This Is Intractable:**

        - Unbinding is approximate (except for FHRR with exact inverse)
        - Each unbind operation introduces noise
        - For H×W image: H×W unbind operations compound errors
        - Scalar decoding via optimization (1000 evals × 100 iterations)
        - Total: ~100M evaluations for 100×100 image
        - No gradient available for joint optimization

        **Alternative Approaches:**

        1. **Database Retrieval**: Encode query image, find nearest match in database
           - Complexity: O(N) for N known images
           - Works well for classification/recognition tasks

        2. **Iterative Resonator**: Use resonator cleanup with pixel codebook
           - Requires pre-built codebook of common pixel patterns
           - May reconstruct coarse structure but not fine details

        3. **Neural Decoder**: Train neural network image_hv → image
           - Requires supervised training data
           - Can learn inverse mapping empirically
           - See: Imani et al. (2019) "VoiceHD" for similar approach

        For practical applications, use ImageEncoder for one-way encoding
        (e.g., image→hypervector→classifier) rather than reconstruction.

        References
        ----------
        - Imani et al. (2019): "VoiceHD: Hyperdimensional Computing for
          Efficient Speech Recognition"
        - Plate (2003): "Holographic Reduced Representations" - Chapter 4 on
          approximate unbinding and error accumulation
        """
        raise NotImplementedError(
            "Image decoding is not implemented due to computational intractability. "
            "See docstring for detailed mathematical explanation and alternatives. "
            "For reconstruction tasks, use similarity-based retrieval from a database "
            "of known images, or train a neural decoder network."
        )

    @property
    def is_reversible(self) -> bool:
        """
        Whether the encoder supports decoding.

        Returns
        -------
        bool
            False - image decoding not yet implemented.
        """
        return False

    @property
    def compatible_models(self) -> List[str]:
        """
        List of compatible VSA model names.

        Returns
        -------
        list of str
            All VSA models supported (depends on scalar encoder compatibility).
        """
        return self.scalar_encoder.compatible_models

    @property
    def input_type(self) -> str:
        """
        Description of expected input type.

        Returns
        -------
        str
            Description of input format.
        """
        if self.n_channels is None:
            return "2D array (grayscale) or 3D array (color) with shape (H, W) or (H, W, C)"
        elif self.n_channels == 1:
            return f"Grayscale image ({self.image_shape[0]}x{self.image_shape[1]})"
        elif self.n_channels == 3:
            return f"RGB image ({self.image_shape[0]}x{self.image_shape[1]}x3)"
        else:
            return f"RGBA image ({self.image_shape[0]}x{self.image_shape[1]}x4)"

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ImageEncoder(model={self.model.model_name}, "
            f"scalar_encoder={self.scalar_encoder.__class__.__name__}, "
            f"normalize_pixels={self.normalize_pixels})"
        )
