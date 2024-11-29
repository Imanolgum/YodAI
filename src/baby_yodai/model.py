"""
Module containing the Baby YodAI model.

This is a simple language model that predicts the next character in a sequence
given the previous characters.

Run this module to train the model on the Baby YodAI dataset.
"""

import torch
from torch import nn
from torch.nn import functional as F

from src.baby_yodai.tokenizer import BabyYodAITokenizer

# ---------------------------------------------------------------------------- #
#                             HYPERPARAMETERS                                  #
# ---------------------------------------------------------------------------- #

BATCH_SIZE = 32
BLOCK_SIZE = 128
MAX_EPOCHS = 10000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1
SAVE_EVERY = 500

# ---------------------------------------------------------------------------- #
#                                 DATA                                         #
# ---------------------------------------------------------------------------- #

with open("data/baby_yodai/baby_yodai.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)
TOKENIZER = BabyYodAITokenizer("".join(chars))

DATA = torch.tensor(TOKENIZER.encode(text), dtype=torch.long)

n = int(len(DATA) * 0.9)
train_data, val_data = DATA[:n], DATA[n:]


# ---------------------------------------------------------------------------- #
#                                DATA LOADER                                   #
# ---------------------------------------------------------------------------- #


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get a random batch of data from the training or validation set.
    Each batch is composed of BATCH_SIZE sequences of BLOCK_SIZE characters.

    Args:
        split (str): "train" or "val" to select the data split.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The input and target sequences.
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])

    return x, y


# ---------------------------------------------------------------------------- #
#                             LOSS FUNCTION                                    #
# ---------------------------------------------------------------------------- #


@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    """
    Estimate the loss on the training and validation sets.

    Returns:
        dict[str, float]: The average loss on the training and validation sets.
    """

    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)

        for i in range(EVAL_ITERS):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[i] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out


# ---------------------------------------------------------------------------- #
#                             SAVING UTILITIES                                 #
# ---------------------------------------------------------------------------- #


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save a training checkpoint.

    Args:
        model (BabyYodaModel): The model to save
        optimizer (torch.optim.Optimizer): The optimizer
        epoch (int): Current epoch number
        loss (float): Current loss value
        filepath (str): Where to save the checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        # Save important hyperparameters
        "block_size": BLOCK_SIZE,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "vocab_size": VOCAB_SIZE,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer):
    """
    Load a training checkpoint.

    Args:
        filepath (str): Path to the checkpoint file
        model (BabyYodaModel): The model to load weights into
        optimizer (torch.optim.Optimizer): The optimizer to load state into

    Returns:
        tuple: (epoch, loss) The saved epoch and loss values
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["loss"]


def save_model_for_inference(model, filepath):
    """
    Save the trained model for inference.

    Args:
        model (BabyYodaModel): The trained model
        filepath (str): Where to save the model
    """
    # Save model state and critical parameters
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "block_size": BLOCK_SIZE,
            "vocab_size": VOCAB_SIZE,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
        },
        filepath,
    )


# ---------------------------------------------------------------------------- #
#                                MODEL                                         #
# ---------------------------------------------------------------------------- #


class AttentionHead(nn.Module):
    """
    One head of self-attention.
    """

    def __init__(self, head_size: int):
        """
        Initialize the attention head.

        Args:
            head_size (int): The size of the head.
        """

        super().__init__()
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention head.

        Args:
            x (torch.Tensor): The input sequence.

        Returns:
            torch.Tensor: The output sequence.
        """

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out


class MultiAttentionHead(nn.Module):
    """
    Multi-head attention.
    """

    def __init__(self, num_heads: int, head_size: int):
        """
        Initialize the multi-head attention.

        Args:
            num_heads (int): The number of heads.
            head_size (int): The size of each head.
        """

        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head attention.

        Args:
            x (torch.Tensor): The input sequence.

        Returns:
            torch.Tensor: The output sequence.
        """

        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out


class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity.
    """

    def __init__(self, n_embd: int):
        """
        Initialize the feed-forward layer.

        Args:
            n_embd (int): The size of the input and output.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward layer.

        Args:
            x (torch.Tensor): The input sequence.

        Returns:
            torch.Tensor: The output sequence.
        """

        return self.net(x)


class Block(nn.Module):
    """
    A transformer block composed of multi-head attention and feed-forward layers
    """

    def __init__(self, n_embd: int, n_heads: int):
        """
        Initialize the transformer block.

        Args:
            n_embd (int): The size of the input and output.
        """

        super().__init__()
        head_size = n_embd // n_heads
        self.sa_heads = MultiAttentionHead(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.lnorm1 = nn.LayerNorm(n_embd)
        self.lnorm2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer block.

        Args:
            x (torch.Tensor): The input sequence.

        Returns:
            torch.Tensor: The output sequence.
        """

        x = x + self.sa_heads(self.lnorm1(x))
        x = x + self.ffwd(self.lnorm2(x))

        return x


class BabyYodAIModel(nn.Module):
    """
    Baby Yoda model.
    """

    def __init__(self):
        """
        Initialize the model.∫
        """

        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)]
        )
        self.lnorm = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            idx (torch.Tensor): The input sequence.
            targets (torch.Tensor, optional): The target sequence.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The logits and the loss.
            If targets is None, the loss is None.
        """

        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)
        position_emb = self.position_embedding_table(
            torch.arange(T, device=DEVICE)
        )
        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.lnorm(x)
        logits = self.lm_head(x)

        if targets is None:

            loss = None

        else:

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generate new tokens given an input sequence.

        Args:
            idx (torch.Tensor): The input sequence.
            max_new_tokens (int): The maximum number of tokens to generate.

        Returns:
            torch.Tensor: The generated sequence.
        """

        for _ in range(max_new_tokens):

            idx_crop = idx[:, -BLOCK_SIZE:]

            logits, _ = self(idx_crop)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ---------------------------------------------------------------------------- #
#                              TRAINING                                        #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    model = BabyYodAIModel().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(MAX_EPOCHS):

        if epoch % EVAL_INTERVAL == 0:
            losses = estimate_loss()
            print(
                f"Epoch {epoch}: Train loss: {losses["train"]}, "
                f"Val loss: {losses["val"]}"
            )

            if epoch % SAVE_EVERY == 0:
                check_point_path = f"model/baby_yoda_checkpoint_{epoch}.pt"
                save_checkpoint(
                    model, optimizer, epoch, losses["train"], check_point_path
                )

        xb, yb = get_batch("train")

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(
        TOKENIZER.decode(
            model.generate(
                torch.zeros((1, 1), dtype=torch.long), max_new_tokens=1000
            )[0].tolist()
        )
    )

    model_path = "model/baby_yoda.pt"
    save_model_for_inference(model, model_path)
