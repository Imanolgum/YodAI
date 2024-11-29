"""
Load the BabyYodAI model for inference and generate new text.

The class provides two methods:
    - generate(max_new_tokens: int) -> str:
        Generate new text starting with the newline character and
        generate the specified number of tokens.

    - complete(text: str) -> str: Generate new text starting with the provided
        text. As this is a Pre-Trained model, it will not be able to generate 
        very meaningful text based on the input text.
"""

import torch

from src.baby_yodai.model import BabyYodAIModel
from src.baby_yodai.tokenizer import BabyYodAITokenizer

# ---------------------------------------------------------------------------- #
#                               CONFIGURATION                                  #
# ---------------------------------------------------------------------------- #


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("data/baby_yodai/baby_yodai.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)
TOKENIZER = BabyYodAITokenizer("".join(chars))


# ---------------------------------------------------------------------------- #
#                                INFERENCE                                     #
# ---------------------------------------------------------------------------- #


class BabyYodAI:
    """
    BabyYodAI model for text generation.
    """

    def __init__(self, filepath):
        """
        Initialize the BabyYodAI model.

        Args:
            filepath (str): Path to the saved model, e.g., "model/baby_yoda.pth"
        """
        self.model = self._load_model_for_inference(filepath)

    def _load_model_for_inference(self, filepath):
        """
        Load a model for inference.

        Args:
            filepath (str): Path to the saved model

        Returns:
            BabyYodaModel: The loaded model in eval mode
        """
        # Load the saved parameters
        checkpoint = torch.load(filepath, weights_only=True)

        # Create a new model instance with the saved parameters
        model = BabyYodAIModel().to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()  # Set to evaluation mode

        return model

    def generate(self, max_new_tokens) -> str:
        """
        Generate new text using the model.
        It will start with the token for the newline character,
        and generate the specified number of tokens.

        Args:
            max_new_tokens (int): Maximum number of tokens to generate

        Returns:
            str: The generated text
        """
        new_line_tensor = torch.tensor([[0]], dtype=torch.long)

        generated_tokens = self.model.generate(
            new_line_tensor, max_new_tokens=max_new_tokens
        )[0].tolist()

        return TOKENIZER.decode(generated_tokens)

    def complete(self, text) -> str:
        """
        Generate new text using the model.
        It will start with the token for the newline character,
        and generate the specified number of tokens.

        Args:
            text (str): The text to complete

        Returns:
            str: The generated text
        """
        text_tokens = TOKENIZER.encode(text)
        max_new_tokens = 128 - len(text_tokens)
        if max_new_tokens <= 0:
            return "Input text is too long. Please provide a shorter text."

        text_tensor = torch.tensor([text_tokens], dtype=torch.long)
        generated_tokens = self.model.generate(text_tensor, max_new_tokens=100)[
            0
        ].tolist()

        return TOKENIZER.decode(generated_tokens)


if __name__ == "__main__":
    baby_yodai = BabyYodAI("model/baby_yoda.pth")
    print(baby_yodai.complete("Hmm, "))
