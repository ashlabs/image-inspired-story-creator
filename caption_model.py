"""Transformer decoder for image-conditioned caption generation."""

import math

import torch
from torch import nn


class CaptionDecoder(nn.Module):
    """Predict next-token logits conditioned on image features.

    Parameters
    ----------
    vocabulary_size : int
        Number of tokens, including special tokens.
    pad_id : int
        Padding token ID.
    max_length : int
        Maximum number of decoder input positions.
    width : int
        Embedding and attention dimension; divisible by heads.
    heads : int
        Number of attention heads per decoder block.
    layers : int
        Number of Transformer decoder blocks.
    feedforward_width : int
        Hidden dimension of each block's feed-forward network.
    dropout : float
        Dropout probability during training.
    """

    def __init__(
        self,
        vocabulary_size,
        pad_id,
        max_length=39,
        width=256,
        heads=4,
        layers=2,
        feedforward_width=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.width = width

        self.visual_projection = nn.Linear(512, width)
        self.visual_positions = nn.Embedding(49, width)

        self.token_embeddings = nn.Embedding(
            vocabulary_size, width, padding_idx=pad_id
        )
        self.text_positions = nn.Embedding(max_length, width)
        self.embedding_dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=feedforward_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(width),
        )
        self.output_projection = nn.Linear(width, vocabulary_size)

        # TransformerDecoder clones its template layer. Initialize each
        # block's weight matrices separately to avoid identical starts.
        for block in self.decoder.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        visual_tokens,
        input_ids,
        input_padding_mask,
        causal_mask,
    ):
        """Return unnormalized next-token scores.

        Parameters
        ----------
        visual_tokens : torch.Tensor
            Frozen image features, shape (B, 49, 512).
        input_ids : torch.Tensor
            Caption token IDs, shape (B, T).
        input_padding_mask : torch.Tensor
            Boolean mask, shape (B, T); True marks padding.
        causal_mask : torch.Tensor
            Boolean mask, shape (T, T); True blocks attention.

        Returns
        -------
        torch.Tensor
            Vocabulary logits, shape (B, T, vocabulary_size).
        """
        visual_positions = torch.arange(
            visual_tokens.shape[1], device=visual_tokens.device
        )
        memory = (
            self.visual_projection(visual_tokens)
            + self.visual_positions(visual_positions).unsqueeze(0)
        )

        text_positions = torch.arange(
            input_ids.shape[1], device=input_ids.device
        )
        caption_embeddings = (
            self.token_embeddings(input_ids) * math.sqrt(self.width)
            + self.text_positions(text_positions).unsqueeze(0)
        )
        caption_embeddings = self.embedding_dropout(caption_embeddings)

        hidden_states = self.decoder(
            tgt=caption_embeddings,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=input_padding_mask,
        )
        return self.output_projection(hidden_states)
