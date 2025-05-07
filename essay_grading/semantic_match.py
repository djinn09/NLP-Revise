"""Module provides functionality for calculating semantic similarity between texts using Sentence Transformers.

This module offers a class, `SemanticCosineSimilarity`, designed to compute various
similarity and distance metrics between two text strings. It handles potentially
long texts by employing a chunking mechanism with overlap to preserve context.
Users can specify which metrics (cosine similarity, Euclidean distance, Manhattan distance)
they wish to calculate.

The module also includes an example usage section (`if __name__ == "__main__":`)
that demonstrates how to use the class and its features, with optional rich
logging for enhanced console output.

Key Features:
- Chunk-based text processing for long inputs.
- Overlapping chunks to maintain contextual integrity.
- Calculation of multiple semantic metrics:
    - Cosine Similarity
    - Euclidean Distance
    - Manhattan Distance
- Selectable metrics: users can choose which ones to compute.
- Integration with SentenceTransformer models.
- Optional rich logging for better visual feedback.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Set, Tuple  # Necessary type hints

# Attempt to import necessary libraries and provide guidance if they are missing
try:
    import torch
    import torch.nn.functional as F  # noqa: N812 - Keep F as standard PyTorch alias for cosine_similarity
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    # Determine which library is missing for a more specific error message
    missing_lib = ""
    if "torch" in str(e).lower():
        missing_lib = "torch"
    elif "sentence_transformers" in str(e).lower():
        missing_lib = "sentence-transformers"
    else:
        missing_lib = "a required library"  # Generic fallback
    # Construct and raise a new ImportError with installation instructions
    error_message = f"Missing {missing_lib}. Please install it (e.g., `pip install {missing_lib}`). Original error: {e}"
    raise ImportError(error_message) from e

# Attempt to import 'rich' for enhanced console logging.
# This is an optional dependency for the example usage, but made mandatory here for demonstration.
try:
    from rich.console import Console
    from rich.logging import RichHandler

    _rich_available = True  # Flag to indicate rich is available
except ImportError:
    _rich_available = False  # Flag to indicate rich is not available
    # Raise an error if rich is considered critical for the application's logging.
    # For this module, we are assuming the user wants the rich logging if they run the example.
    error_message = "Missing rich. Please install it (`pip install rich`) for enhanced logging in the example."
    raise ImportError(error_message) from None  # 'from None' suppresses the original ImportError context


# Configure basic logging. This will be used if the module is imported
# or if rich is unavailable. It gets overridden in __main__ if rich is used.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # Module-specific logger

# --- Configuration for Similarity Score Interpretation (used in example) ---
# Thresholds for categorizing cosine similarity scores.
# These can be set via environment variables or default to predefined values.
GOOD_SIMILARITY_SCORE = float(os.getenv("GOOD_SIMILARITY_SCORE", "0.7"))
BAD_SIMILARITY_SCORE = float(os.getenv("BAD_SIMILARITY_SCORE", "0.3"))

# --- Metric Definitions ---
# Define internal keys for metrics to ensure consistency.
METRIC_COSINE = "cosine_similarity"
METRIC_EUCLIDEAN = "euclidean_distance"
METRIC_MANHATTAN = "manhattan_distance"

# Map user-friendly metric names (lowercase) to internal keys.
# This allows users to request metrics with simple names.
ALLOWED_METRICS_MAP: Dict[str, str] = {
    "cosine": METRIC_COSINE,
    "euclidean": METRIC_EUCLIDEAN,
    "manhattan": METRIC_MANHATTAN,
}
# A set of all valid internal metric keys for quick lookup.
ALL_METRIC_KEYS: Set[str] = set(ALLOWED_METRICS_MAP.values())


class SemanticCosineSimilarity:
    """Calculates semantic similarity between texts using Sentence Transformers with chunking.

    This class encapsulates the logic for:
    1. Initializing with a Sentence Transformer model and chunking parameters.
    2. Processing input texts, potentially splitting them into overlapping chunks if they exceed a defined size.
    3. Generating embeddings for these texts (or their chunks) using the provided model.
    4. Aggregating chunk embeddings (if any) into a single representative embedding for each text.
    5. Computing specified similarity/distance metrics between the final text embeddings.

    The use of overlapping chunks aims to preserve semantic context that might be lost
    at chunk boundaries if simple, non-overlapping splitting were used.

    Attributes:
        model (SentenceTransformer): The Sentence Transformer model used for embeddings.
        chunk_size (int): The target character size for each text chunk.
        overlap (int): The number of characters overlapping between adjacent chunks.
        batch_size (int): The batch size used when encoding multiple chunks.

    """

    def __init__(
        self,
        model: SentenceTransformer,
        chunk_size: int = 384,
        overlap: int = 64,
        batch_size: int = 32,
    ) -> None:
        """Initialize the SemanticCosineSimilarity calculator.

        Args:
            model: An initialized Sentence Transformer model instance.
            chunk_size: The target character size for each text chunk.
                        Must be greater than `overlap`. Defaults to 384.
            overlap: The number of characters for overlap between adjacent chunks.
                     Cannot be negative. Defaults to 64.
            batch_size: The batch size for encoding chunks with the sentence transformer model.
                        Must be positive. Defaults to 32.

        Raises:
            TypeError: If `model` is not a `SentenceTransformer` instance,
                       or if `chunk_size`, `overlap`, or `batch_size` are not integers.
            ValueError: If `chunk_size` is not greater than `overlap`,
                        if `overlap` is negative, or if `batch_size` is not positive.

        """
        # Validate input types and values
        if not isinstance(model, SentenceTransformer):
            msg = "Model must be an instance of SentenceTransformer."
            logger.error(msg)
            raise TypeError(msg)
        if not all(isinstance(val, int) for val in [chunk_size, overlap, batch_size]):
            msg = "chunk_size, overlap, and batch_size must be integers."
            logger.error(msg)
            raise TypeError(msg)
        if overlap < 0:
            msg = "Overlap cannot be negative."
            logger.error(msg)
            raise ValueError(msg)
        if chunk_size <= overlap:
            msg = f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})."
            logger.error(msg)
            raise ValueError(msg)
        if batch_size <= 0:
            msg = "batch_size must be positive."
            logger.error(msg)
            raise ValueError(msg)

        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.batch_size = batch_size

        # Attempt to get the model's name for logging purposes
        model_name = getattr(getattr(model, "config", {}), "name", None)  # Try HuggingFace config
        if not model_name:  # Fallback strategies
            model_name_from_st_config = getattr(model, "_model_config", {}).get(
                "name",
            )  # Sentence Transformers internal config
            if model_name_from_st_config:
                model_name = model_name_from_st_config
            elif hasattr(model, "name_or_path"):  # Common attribute for HF models
                model_name = model.name_or_path
            else:
                model_name = model.__class__.__name__  # Class name as last resort

        logger.info(
            f"SemanticCosineSimilarity initialized with model: [cyan]{model_name}[/cyan], "
            f"chunk_size: [yellow]{self.chunk_size}[/yellow], overlap: [yellow]{self.overlap}[/yellow], "
            f"batch_size: [yellow]{self.batch_size}[/yellow]",
        )

    def _get_aggregated_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Encode text into a single aggregated embedding.

        If the text is longer than `self.chunk_size`, it's split into overlapping
        chunks. Each chunk is encoded, and their embeddings are averaged to produce
        a single representative embedding for the entire text. If the text is short enough,
        it's encoded directly without chunking.

        Args:
            text: The text string to encode.

        Returns:
            A 1D PyTorch tensor representing the aggregated embedding of the text,
            or `None` if the input text is empty, not a string, or if encoding fails.

        """
        # Validate input text
        if not isinstance(text, str) or not text.strip():
            logger.warning("Input text is empty or not a string. Cannot generate embedding.")
            return None  # Early exit for invalid input

        text_stripped = text.strip()  # Remove leading/trailing whitespace
        n = len(text_stripped)
        aggregated_embedding: Optional[torch.Tensor] = None  # Initialize result variable

        try:
            # Case 1: Text is short enough to be encoded directly
            if n <= self.chunk_size:
                logger.debug(f"Text length ({n}) <= chunk_size ({self.chunk_size}), encoding directly.")
                # Encode the text; `convert_to_tensor=True` ensures a PyTorch tensor is returned.
                embedding: torch.Tensor = self.model.encode(
                    text_stripped, convert_to_tensor=True, show_progress_bar=False
                )
                # Squeeze to ensure it's a 1D tensor (D,) even if model returns (1,D) for single sentences
                aggregated_embedding = embedding.squeeze()
            # Case 2: Text is long and needs chunking
            else:
                # Calculate the step size for chunking, considering the overlap
                step = self.chunk_size - self.overlap
                # Create chunks. Filter out any potential empty strings if text length is not a multiple of step.
                chunks = [
                    text_stripped[i : i + self.chunk_size]
                    for i in range(0, n, step)
                    if text_stripped[i : i + self.chunk_size]  # Ensure chunk is not empty
                ]
                # Further filter to remove chunks that are only whitespace
                valid_chunks = [chunk for chunk in chunks if chunk.strip()]

                if not valid_chunks:
                    logger.warning(
                        f"Text resulted in no valid (non-empty, non-whitespace) "
                        f"chunks after processing: '{text_stripped[:50]}...'"
                    )
                    # aggregated_embedding remains None
                else:
                    logger.debug(
                        f"Encoding [magenta]{len(valid_chunks)}[/magenta] chunks for text: '{text_stripped[:50]}...'"
                    )
                    # Encode all valid chunks in batches
                    chunk_embeddings: torch.Tensor = self.model.encode(
                        valid_chunks,
                        batch_size=self.batch_size,
                        convert_to_tensor=True,
                        # Show progress bar only if there are more chunks than one batch
                        show_progress_bar=len(valid_chunks) > self.batch_size,
                    )

                    # Check if encoding produced a valid, non-empty tensor
                    if chunk_embeddings.nelement() == 0:  # `nelement()` is num_elements
                        logger.error(
                            f"Model encoding returned empty tensor for chunks of text: '{text_stripped[:50]}...'"
                        )
                        # aggregated_embedding remains None
                    else:
                        # Aggregate chunk embeddings by taking the mean along the chunk dimension (dim=0)
                        # Resulting shape is (D,)
                        aggregated_embedding = torch.mean(chunk_embeddings, dim=0)
                        logger.debug(
                            f"Aggregated embedding shape: "
                            f"{aggregated_embedding.shape if aggregated_embedding is not None else 'None'}"
                        )

            # Ensure the final embedding (if not None) is 1D.
            # A 0-dim tensor (scalar) might occur if model.encode returns a scalar for very short/empty strings
            # when squeeze() is applied, though unlikely with sentence transformers.
            if aggregated_embedding is not None and aggregated_embedding.dim() == 0:
                aggregated_embedding = aggregated_embedding.unsqueeze(0)  # Convert scalar to 1D tensor

        except Exception as e:  # Catch any unexpected errors during encoding or aggregation
            logger.exception(f"Failed to encode or aggregate text: '{text_stripped[:50]}...'. Error: {e}")
            aggregated_embedding = None  # Ensure None is returned on failure

        return aggregated_embedding

    def _resolve_requested_metrics(self, metrics_to_calculate: Optional[List[str]]) -> Set[str]:
        """Processe the user's list of requested metrics into a set of valid internal metric keys.
        Defaults to cosine similarity if `metrics_to_calculate` is None.
        """
        requested_metrics_internal: Set[str] = set()
        if metrics_to_calculate is None:
            requested_metrics_internal.add(METRIC_COSINE)
        else:
            for m_name in metrics_to_calculate:
                m_key = ALLOWED_METRICS_MAP.get(m_name.lower())
                if m_key:
                    requested_metrics_internal.add(m_key)
                else:
                    logger.warning(
                        f"Unknown metric '{m_name}' requested. It will be ignored. "
                        f"Allowed metrics: {list(ALLOWED_METRICS_MAP.keys())}"
                    )
        return requested_metrics_internal

    def _handle_early_exit_cases(
        self,
        text1: str,
        text2: str,
        is_text1_empty: bool,
        is_text2_empty: bool,
        requested_metrics: Set[str],
    ) -> Tuple[bool, Optional[Dict[str, float]]]:
        """Check for conditions allowing an early exit without embedding generation.

        Returns:
            A tuple (proceed_to_embeddings: bool, result: Optional[Dict[str, float]]).
            If proceed_to_embeddings is False, 'result' contains the final scores or None.

        """
        scores: Dict[str, float] = {}
        # Case 1: Texts are identical OR both are effectively empty
        if text1 == text2 or (is_text1_empty and is_text2_empty):
            log_msg = "Texts are identical" if text1 == text2 else "Both texts are effectively empty"
            logger.info(f"{log_msg}. Returning perfect scores for requested metrics.")
            if METRIC_COSINE in requested_metrics:
                scores[METRIC_COSINE] = 1.0
            if METRIC_EUCLIDEAN in requested_metrics:
                scores[METRIC_EUCLIDEAN] = 0.0
            if METRIC_MANHATTAN in requested_metrics:
                scores[METRIC_MANHATTAN] = 0.0
            return False, scores  # Don't proceed, result is 'scores'

        # Case 2: One text is effectively empty, and the other is not
        if is_text1_empty != is_text2_empty:
            logger.warning("One input text is effectively empty while the other is not. Cannot compare.")
            return False, None  # Don't proceed, result is None (failure)

        # Case 3: No valid metrics were requested
        if not requested_metrics:
            logger.info("No valid metrics requested. Returning empty scores dict.")
            return False, {}  # Don't proceed, result is an empty dict

        return True, None  # Proceed to embedding generation

    def _calculate_metrics_for_embeddings(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        requested_metrics: Set[str],
    ) -> Dict[str, float]:
        """Calculate the specified metrics given two 1D embeddings."""
        scores: Dict[str, float] = {}
        # Reshape embeddings from (D,) to (1, D) for PyTorch functions
        emb1_reshaped = emb1.unsqueeze(0)
        emb2_reshaped = emb2.unsqueeze(0)

        if METRIC_COSINE in requested_metrics:
            cosine_sim = F.cosine_similarity(emb1_reshaped, emb2_reshaped).item()
            scores[METRIC_COSINE] = max(-1.0, min(1.0, cosine_sim))
        if METRIC_EUCLIDEAN in requested_metrics:
            scores[METRIC_EUCLIDEAN] = torch.cdist(emb1_reshaped, emb2_reshaped, p=2.0).item()
        if METRIC_MANHATTAN in requested_metrics:
            scores[METRIC_MANHATTAN] = torch.cdist(emb1_reshaped, emb2_reshaped, p=1.0).item()

        logger.debug(f"Calculated scores from embeddings: {scores}")
        return scores

    def calculate_similarity(
        self, text1: str, text2: str, metrics_to_calculate: Optional[List[str]] = None
    ) -> Optional[Dict[str, float]]:
        """Calculate specified semantic similarity/distance metrics between two texts.

        This method first generates aggregated embeddings for both input texts.
        Then, based on the `metrics_to_calculate` parameter, it computes
        the requested metrics.

        Args:
            text1: The first text string.
            text2: The second text string.
            metrics_to_calculate: A list of metric names (strings) to calculate.
                Valid names are "cosine", "euclidean", "manhattan" (case-insensitive).
                If `None` (default), only "cosine" similarity is calculated.
                If an empty list `[]` is provided, or if all provided names are invalid,
                an empty dictionary is returned (assuming embeddings can be generated).

        Returns:
            A dictionary where keys are the internal metric names (e.g.,
            "cosine_similarity", "euclidean_distance") and values are the
            calculated scores (float).
            Returns `None` if:
                - Embeddings cannot be generated for one or both texts.
                - One text is effectively empty while the other is not.
                - An unexpected error occurs during calculation.
            Returns a dictionary with perfect scores (e.g., cosine=1.0, distance=0.0)
            for requested metrics if texts are identical or both effectively empty.

        """
        requested_metrics = self._resolve_requested_metrics(metrics_to_calculate)

        is_text1_empty = not isinstance(text1, str) or not text1.strip()
        is_text2_empty = not isinstance(text2, str) or not text2.strip()

        proceed_to_embeddings, early_result = self._handle_early_exit_cases(
            text1,
            text2,
            is_text1_empty,
            is_text2_empty,
            requested_metrics,
        )

        if not proceed_to_embeddings:
            return early_result  # This handles identical, one-empty, or no-metrics cases

        # If we reach here, we need to generate embeddings and calculate metrics
        try:
            logger.debug("Generating embedding for text 1...")
            emb1 = self._get_aggregated_embedding(text1)
            logger.debug("Generating embedding for text 2...")
            emb2 = self._get_aggregated_embedding(text2)

            if emb1 is None or emb2 is None:
                logger.error("Could not generate embeddings for one or both texts. Cannot calculate similarity.")
                return None  # Embedding generation failed

            if emb1.dim() != 1 or emb2.dim() != 1:
                logger.error(
                    f"Embeddings have unexpected dimensions: emb1={emb1.shape}, emb2={emb2.shape}. Cannot calculate."
                )
                return None  # Embeddings have wrong shape

            # All checks passed, calculate metrics using the embeddings
            return self._calculate_metrics_for_embeddings(emb1, emb2, requested_metrics)

        except Exception as e:
            logger.exception(
                f"Error during embedding generation or metric calculation for texts: "
                f"'{text1[:50]}...' vs '{text2[:50]}...'. Error: {e}",
            )
            return None  # Unexpected error


# --- Example Usage ---
if __name__ == "__main__":
    # This block executes only when the script is run directly (not imported as a module).
    print("Running Semantic Similarity Example...")  # noqa: T201 Basic print for initial feedback

    # --- Configure Rich Logging for the Example ---
    # Clear any existing handlers from the root logger to prevent duplicate output
    logging.root.handlers.clear()
    # Set logging level from environment variable or default to INFO
    LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)  # Fallback to INFO if invalid
    logging.root.setLevel(LOG_LEVEL)

    console: Optional[Console] = None  # Initialize console for optional rich printing
    if _rich_available:
        # If 'rich' is available, set up RichHandler for formatted console logging
        rich_handler = RichHandler(
            level=LOG_LEVEL,
            show_path=False,  # Don't show file path in log messages
            rich_tracebacks=True,  # Enable rich-formatted tracebacks
            markup=True,  # Allow rich markup (e.g., [bold]) in log messages
            console=Console(stderr=True),  # Direct logs to stderr to separate from potential stdout prints
        )
        logging.root.addHandler(rich_handler)
        console = Console()  # Create a console instance for printing separators
        # Define a separator function using rich console
        separator = lambda: console.print("-" * 50, style="dim") if console else print("-" * 50)  # noqa: T201
        logger.info("Starting Semantic Similarity Example [bold green](using Rich logging)[/bold green]")
    else:
        # Fallback to basic logging configuration if 'rich' is not available
        logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        separator = lambda: print("-" * 50)  # noqa: T201 Basic separator
        logger.info("Starting Semantic Similarity Example (using standard logging)")

    try:
        # --- Configuration for the Example ---
        MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")  # Model to use
        CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "384"))  # Chunk size for long texts
        OVERLAP = int(os.environ.get("OVERLAP", "64"))  # Overlap between chunks
        BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))  # Batch size for encoding

        # --- Initialize Model ---
        logger.info(f"Loading Sentence Transformer model: [cyan]{MODEL_NAME}[/cyan]...")
        # Automatically select device: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Using device: [yellow]{device}[/yellow]")
        model = SentenceTransformer(MODEL_NAME, device=device)
        logger.info("Model loaded successfully.")

        # --- Initialize Semantic Similarity Calculator ---
        semantic_calculator = SemanticCosineSimilarity(
            model=model, chunk_size=CHUNK_SIZE, overlap=OVERLAP, batch_size=BATCH_SIZE
        )

        # --- Example Texts for Demonstration ---
        text_a = (
            "The quick brown fox jumps over the lazy dog. This sentence is used "
            "to demonstrate the presence of all letters in the English alphabet. "
            "Sentence Transformers can capture the semantic meaning of such text."
        )
        text_b = (
            "Sentence embedding models like those in the Sentence Transformers "
            "library understand the meaning behind sentences. A pangram, like the "
            "one about the fox and dog, contains every letter."
        )  # Semantically similar to text_a
        text_c = (
            "The weather today is sunny and warm, perfect for a walk in the park. "
            "Many people are enjoying the outdoors."
        )  # Semantically different
        text_empty = ""
        text_whitespace = "   "  # Effectively empty

        # --- Define Test Cases ---
        # Each case specifies a description, two texts, and which metrics to calculate.
        test_cases = [
            {"desc": "Similar Texts (Default - Cosine only)", "t1": text_a, "t2": text_b, "metrics": None},
            {
                "desc": "Similar Texts (All Metrics)",
                "t1": text_a,
                "t2": text_b,
                "metrics": ["cosine", "euclidean", "manhattan"],
            },
            {
                "desc": "Similar Texts (Cosine & Manhattan)",
                "t1": text_a,
                "t2": text_b,
                "metrics": ["cosine", "manhattan"],
            },
            {
                "desc": "Dissimilar Texts (All Metrics)",
                "t1": text_a,
                "t2": text_c,
                "metrics": list(ALLOWED_METRICS_MAP.keys()),
            },
            {
                "desc": "Identical Texts (All Metrics)",
                "t1": text_a,
                "t2": text_a,
                "metrics": ["cosine", "euclidean", "manhattan"],
            },
            {"desc": "One Empty (Default - Cosine)", "t1": text_a, "t2": text_empty, "metrics": None},
            {
                "desc": "One Whitespace (All Metrics)",
                "t1": text_a,
                "t2": text_whitespace,
                "metrics": ["cosine", "euclidean"],
            },
            {"desc": "Both Empty (Euclidean only)", "t1": text_empty, "t2": text_empty, "metrics": ["euclidean"]},
            {
                "desc": "Empty vs Whitespace (All)",
                "t1": text_empty,
                "t2": text_whitespace,
                "metrics": ["cosine", "euclidean"],
            },
            {
                "desc": "Invalid Metric Requested",
                "t1": text_a,
                "t2": text_b,
                "metrics": ["cosine", "nonexistent_metric"],
            },
            {"desc": "Empty Metrics List (No calculation)", "t1": text_a, "t2": text_b, "metrics": []},
        ]

        # --- Run Test Cases and Log Results ---
        for case in test_cases:
            logger.info(f"Calculating for: [bold yellow]{case['desc']}[/bold yellow]")
            # Call the main calculation method
            scores_dict = semantic_calculator.calculate_similarity(
                case["t1"], case["t2"], metrics_to_calculate=case["metrics"]
            )

            if scores_dict is not None:  # `None` indicates a failure to get embeddings or one-empty-one-not scenario
                if not scores_dict:  # Empty dictionary means no metrics were calculated (e.g., metrics=[])
                    logger.info(f"Result - {case['desc']}: No metrics calculated or returned an empty set.")
                # Format and log the calculated scores
                elif _rich_available:  # Use rich formatting if available
                    output_parts = []
                    for metric_key, value in scores_dict.items():
                        metric_name_formatted = metric_key.replace("_", " ").title()
                        color = "default"  # Default color
                        # Apply specific coloring rules for different metrics
                        if metric_key == METRIC_COSINE:
                            color = (
                                "green"
                                if value > GOOD_SIMILARITY_SCORE
                                else "yellow"
                                if value > BAD_SIMILARITY_SCORE
                                else "red"
                            )
                        elif "distance" in metric_key:  # Lower is better for distances
                            # Example coloring for distances (adjust thresholds as needed)
                            if value < 0.5:
                                color = "green"
                            elif value < 1.0:
                                color = "yellow"
                            else:
                                color = "red"
                        output_parts.append(f"{metric_name_formatted}=[{color}]{value:.4f}[/{color}]")
                    logger.info("Result - %s: %s", case["desc"], " | ".join(output_parts))
                else:  # Standard logging format if rich is not available
                    log_parts = [f"Result - {case['desc']}:"]
                    for metric_key, value in scores_dict.items():
                        metric_name_formatted = metric_key.replace("_", " ").title()
                        log_parts.append(f"{metric_name_formatted}={value:.4f}")
                    logger.info(" | ".join(log_parts))
            else:  # scores_dict is None
                logger.warning(f"Result - {case['desc']}: Calculation Failed or Not Applicable (returned None)")
            separator()  # Print a separator between test cases

    except ImportError as e:
        # Catch ImportErrors that might have been raised if a critical library was missing
        # (though initial checks should catch these, this is a fallback).
        logger.exception(f"Example cannot run due to missing libraries: {e}")
    except ValueError as e:
        # Catch ValueErrors, e.g., from invalid configuration for chunk_size/overlap.
        logger.exception(f"Configuration error in example: {e}")
    except Exception:
        # Catch any other unexpected errors during the example execution.
        logger.exception("An unexpected error occurred in the example:")

    logger.info("Semantic Similarity Example Finished")
