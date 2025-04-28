"""Module provides functionality for calculating semantic similarity between texts using Sentence Transformers and cosine similarity.

It includes:

- SemanticCosineSimilarity: A class for chunk-based text similarity calculation.
- Example usage demonstrating the functionality with rich logging.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# Attempt to import necessary libraries and provide guidance
try:
    import torch
    import torch.nn.functional as F  # noqa: N812 - Keep F as standard PyTorch alias
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    # Provide specific instructions if libraries are missing
    missing_lib = ""
    if "torch" in str(e):
        missing_lib = "torch"
    elif "sentence_transformers" in str(e):
        missing_lib = "sentence-transformers"
    else:
        missing_lib = "required library"

    # Construct a more informative error message
    error_message = f"Missing {missing_lib}. Please install it (e.g., `pip install {missing_lib}`). Original error: {e}"
    # Raise a new ImportError preserving the original cause with 'from e'
    raise ImportError(error_message) from e

# Attempt to import rich
try:
    from rich.console import Console
        from rich.logging import RichHandler

    _rich_available = True
except ImportError:
    _rich_available = False
    # If rich is critical, raise error. If optional, configure basic logging as fallback.
    # For this request, let's assume it's required for the desired output.
    error_message = "Missing rich. Please install it (`pip install rich`) for enhanced logging."
    raise ImportError(error_message) from None


# Configure basic logging (This will be overridden in __main__ if rich is used)
# Set a basic config here in case the module is imported elsewhere without __main__ running
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # Use a logger specific to this module

GOOD_SIMILARITY_SCORE = float(os.getenv("GOOD_SIMILARITY_SCORE", "0.7"))
BAD_SIMILARITY_SCORE = float(os.getenv("BAD_SIMILARITY_SCORE", "0.3"))


class SemanticCosineSimilarity:
    """Calculate semantic similarity using Sentence Transformers with chunking.

    This class handles potentially long texts by splitting them into overlapping
    chunks, encoding each chunk using a provided Sentence Transformer model,
    averaging the embeddings for each text, and finally computing the cosine
    similarity between these aggregated embeddings.

    Using overlapping chunks helps mitigate the loss of contextual information
    that might occur at the boundaries if simple splitting were used.

    Args:
        model: An initialized Sentence Transformer model instance.
        chunk_size: The target character size for each text chunk.
                    Must be greater than overlap. Defaults to 384.
        overlap: The number of characters overlapping between adjacent chunks.
                 Cannot be negative. Defaults to 64.

    """

    def __init__(
        self,
        model: SentenceTransformer,
        chunk_size: int = 384,
        overlap: int = 64,
    ) -> None:
        """Initialize the SemanticCosineSimilarity calculator.

        Raises:
            TypeError: If the provided model is not a SentenceTransformer instance,
                       or if chunk_size/overlap are not integers.
            ValueError: If chunk_size is not greater than overlap, or if overlap is negative.

        """
        if not isinstance(model, SentenceTransformer):
            msg = "Model must be an instance of SentenceTransformer."
            logger.error(msg)
            raise TypeError(msg)
        if not isinstance(chunk_size, int) or not isinstance(overlap, int):
            msg = "chunk_size and overlap must be integers."
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

        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Get model name if possible for logging (handle potential absence)
        # Use model.config.name if available (newer SentenceTransformer versions)
        model_name = getattr(getattr(model, "config", {}), "name", None)
        if not model_name:  # Fallback
            # Try another common attribute name if the first doesn't work
            model_name = getattr(model, "_model_config", {}).get("name", model.__class__.__name__)

        # Use rich markup for emphasis in logs if desired
        logger.info(
            f"SemanticCosineSimilarity initialized with model: [cyan]{model_name}[/cyan], "
            f"chunk_size: [yellow]{self.chunk_size}[/yellow], overlap: [yellow]{self.overlap}[/yellow]"
        )

    def _get_aggregated_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Encode text using overlapping chunks and aggregate embeddings by averaging.

        Args:
            text: The text to encode.

        Returns:
            A single aggregated embedding tensor representing the text,
            or None if the text is empty or encoding fails.

        """
        if not isinstance(text, str) or not text.strip():
            logger.warning("Input text is empty or not a string. Cannot generate embedding.")
            return None

        text = text.strip()  # Ensure leading/trailing whitespace removed
        n = len(text)

        # If text is shorter than or equal to chunk size, encode directly without chunking
        if n <= self.chunk_size:
            logger.debug(f"Text <= chunk_size ({n}<={self.chunk_size}), encoding directly.")
            try:
                # Specify show_progress_bar=False if calling encode for single items often
                embedding = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
                # model.encode returns ndarray or tensor, ensure it's a tensor
                if not isinstance(embedding, torch.Tensor):
                    embedding = torch.from_numpy(embedding)
                # Ensure it's a 1D tensor
                return embedding.squeeze()
            except Exception as e:
                # Log the exception with traceback
                logger.exception(f"Failed to encode short text: '{text[:50]}...'. Error: {e}")
                return None

        # Generate potentially overlapping chunks for longer text
        step = self.chunk_size - self.overlap
        chunks = [text[i : i + self.chunk_size] for i in range(0, n, step)]
        valid_chunks = [chunk for chunk in chunks if chunk]  # Filter empty

        if not valid_chunks:
            logger.warning(f"Text resulted in no valid chunks after processing: '{text[:50]}...'")
            return None

        logger.debug(f"Encoding [magenta]{len(valid_chunks)}[/magenta] chunks for text: '{text[:50]}...'")

        try:
            # Encode chunks in batch for efficiency
            # show_progress_bar can be useful here if encoding takes time
            chunk_embeddings = self.model.encode(valid_chunks, convert_to_tensor=True, show_progress_bar=True)

            if not isinstance(chunk_embeddings, torch.Tensor) or chunk_embeddings.nelement() == 0:
                logger.error(f"Model encoding returned invalid result for chunks of text: '{text[:50]}...'")
                return None

            # Aggregate embeddings by averaging across the chunk dimension (dim=0)
            aggregate_embedding = torch.mean(chunk_embeddings, dim=0)
            logger.debug(f"Aggregated embedding shape: {aggregate_embedding.shape}")
            return aggregate_embedding

        except Exception as e:
            # Catch potential errors during encoding or aggregation
            logger.exception(f"Failed to encode or aggregate chunks for text: '{text[:50]}...'. Error: {e}")
            return None

    def calculate_similarity(self, text1: str, text2: str) -> Optional[float]:
        """Calculate the semantic cosine similarity between two texts.

        Args:
            text1: The first text string.
            text2: The second text string.

        Returns:
            The cosine similarity score as a float between -1.0 and 1.0,
            or None if embeddings could not be generated for either text
            or if another error occurs. Returns 0.0 if both texts are empty.

        """
        # Handle cases where one or both texts might be effectively empty
        is_text1_empty = not isinstance(text1, str) or not text1.strip()
        is_text2_empty = not isinstance(text2, str) or not text2.strip()

        if is_text1_empty and is_text2_empty:
            logger.warning("Both input texts are empty. Returning similarity [yellow]0.0[/yellow]")
            return 0.0
        if is_text1_empty or is_text2_empty:
            logger.warning("One input text is empty. Returning similarity [yellow]0.0[/yellow]")
            return 0.0

        try:
            # Get aggregated embeddings for both texts
            logger.debug("Generating embedding for text 1...")
            emb1 = self._get_aggregated_embedding(text1)
            logger.debug("Generating embedding for text 2...")
            emb2 = self._get_aggregated_embedding(text2)

            # Check if embeddings were successfully generated
            if emb1 is None or emb2 is None:
                logger.error("Could not generate embeddings for one or both texts. Cannot calculate similarity.")
                return None  # Indicate failure clearly

            # Ensure embeddings are 1D tensors before un-squeezing
            if emb1.dim() != 1 or emb2.dim() != 1:
                logger.error(
                    f"Embeddings have unexpected dimensions: emb1={emb1.shape}, emb2={emb2.shape}. Cannot calculate similarity."
                )
                return None

            # Calculate cosine similarity
            # F.cosine_similarity expects tensors of shape (N, D) or (D)
            # If (D), it computes dot(x, y) / (norm(x) * norm(y))
            # If (N, D), it computes pairwise along N
            # Here we have two (D) tensors, unsqueeze adds N=1 dimension: (1, D)
            similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

            # Clamp result just in case of floating point inaccuracies slightly outside [-1, 1]
            similarity = max(-1.0, min(1.0, similarity))

            logger.debug(f"Calculated cosine similarity: [bold blue]{similarity:.4f}[/bold blue]")
            return similarity

        except Exception:
            # Catch any unexpected errors during the process
            logger.exception(
                f"Error calculating semantic similarity for texts: '{text1[:50]}...' vs '{text2[:50]}...'.",
            )
            return None  # Return None on failure


# --- Example Usage ---
if __name__ == "__main__":
    # This block runs only when the script is executed directly
    print("Running Semantic Similarity Example...")  # noqa: T201
    # --- Configure Rich Logging ---
    # Remove default handlers from the root logger to avoid duplicate output
    logging.root.handlers.clear()
    # Configure the root logger level (e.g., INFO, DEBUG)
    LOG_LEVEL = logging.INFO
    logging.root.setLevel(LOG_LEVEL)

    # Check if rich is available before setting up the handler
    if _rich_available:
        # Add RichHandler for beautiful console logging
        rich_handler = RichHandler(
            level=LOG_LEVEL,  # Ensure handler respects the log level
            show_path=False,  # Don't show the file path
            rich_tracebacks=True,  # Enable formatted tracebacks
            markup=True,  # Allow rich markup like [bold] in log messages
        )
        logging.root.addHandler(rich_handler)
        # Use rich console for printing separators if desired
        try:
            console = Console()
            separator = lambda: console.print("-" * 30, style="dim")  # Make separator slightly longer  # noqa: E731
        except ImportError:
            separator = lambda: print("-" * 30)  # Fallback if rich isn't fully utilized for print  # noqa: E731, T201

        logger.info("Starting Semantic Similarity Example [bold green](using Rich logging)[/bold green]")
    else:
        # Fallback to basic logging if rich is not available
        logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        separator = lambda: print("-" * 30)  # noqa: E731, T201
        logger.info("Starting Semantic Similarity Example (using standard logging)")

    try:
        # --- Configuration ---
        MODEL_NAME = os.environ.get("MODEL", "all-MiniLM-L6-v2")
        CHUNK_SIZE = 384
        OVERLAP = 64

        # --- Initialize Model ---
        logger.info(f"Loading Sentence Transformer model: [cyan]{MODEL_NAME}[/cyan]...")
        # Determine device automatically
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():  # Check for Apple Silicon MPS
            device = "mps"
        else:
            device = "cpu"

        logger.info(f"Using device: [yellow]{device}[/yellow]")
        model = SentenceTransformer(MODEL_NAME, device=device)
        logger.info("Model loaded successfully.")

        # --- Initialize Calculator ---
        semantic_calculator = SemanticCosineSimilarity(model=model, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

        # --- Example Texts ---
        text_a = (
            "The quick brown fox jumps over the lazy dog. This sentence is used "
            "to demonstrate the presence of all letters in the English alphabet. "
            "Sentence Transformers can capture the semantic meaning of such text."
        )
        text_b = (
            "Sentence embedding models like those in the Sentence Transformers "
            "library understand the meaning behind sentences. A pangram, like the "
            "one about the fox and dog, contains every letter."
        )
        text_c = (
            "The weather today is sunny and warm, perfect for a walk in the park. "
            "Many people are enjoying the outdoors."
        )
        text_d = text_a  # Identical text
        text_empty = ""
        text_short = "A quick fox."

        # --- Calculate Similarities ---
        pairs_to_compare = [
            ("Similar Texts (A vs B)", text_a, text_b),
            ("Different Texts (A vs C)", text_a, text_c),
            ("Identical Texts (A vs D)", text_a, text_d),
            ("One Empty Text (A vs Empty)", text_a, text_empty),
            ("Both Empty Texts (Empty vs Empty)", text_empty, text_empty),
            ("Short Texts (A vs Short)", text_a, text_short),
        ]

        for description, t1, t2 in pairs_to_compare:
            logger.info(f"Calculating similarity for: [bold yellow]{description}[/bold yellow]")
            similarity_score = semantic_calculator.calculate_similarity(t1, t2)

            if similarity_score is not None:
                # Use color coding based on score for visual feedback (requires rich)
                if _rich_available:
                    color = (
                        "green"
                        if similarity_score > GOOD_SIMILARITY_SCORE
                        else "yellow"
                        if similarity_score > BAD_SIMILARITY_SCORE
                        else "red"
                    )
                    logger.info(f"Result - {description}: [{color}]{similarity_score:.4f}[/{color}]")
                else:
                    # Standard log format if rich is not available
                    logger.info(f"Result - {description}: {similarity_score:.4f}")
            else:
                logger.warning(f"Result - {description}: Calculation Failed")
            separator()  # Print separator

    except ImportError:
        # Error already raised or handled during import attempts
        # Log the specific error message constructed earlier
        logger.exception("Example cannot run due to missing libraries")
    except Exception:
        # Catch any other unexpected error during setup or execution
        logger.exception("An unexpected error occurred in the example:")  # Rich handler will format this

    logger.info("Semantic Similarity Example Finished")
