"""
Task-Specific Prediction Heads for STEPNet
Defines what the model actually predicts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional

from .encoder import STEPEncoder


class STEPForCaptioning(nn.Module):
    """
    STEP encoder with caption generation head.
    Predicts: Natural language description of CAD part.
    
    Use case: "This is a mounting bracket with 4 bolt holes"
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        decoder_vocab_size: int = 50000,
        output_dim: int = 1024,
        max_caption_length: int = 128
    ):
        super().__init__()
        
        # Encoder
        self.encoder = STEPEncoder(vocab_size=vocab_size, output_dim=output_dim)
        
        # Caption token embedding
        self.caption_embedding = nn.Embedding(decoder_vocab_size, output_dim)

        # Caption decoder (transformer)
        self.caption_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=output_dim, nhead=8, batch_first=True),
            num_layers=6
        )

        # Output projection
        self.output_projection = nn.Linear(output_dim, decoder_vocab_size)
        
    def forward(
        self,
        token_ids: torch.Tensor,
        caption_ids: Optional[torch.Tensor] = None,
        topology_data: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len] STEP tokens
            caption_ids: [batch, caption_len] target captions (for training)
            topology_data: Optional topology dict
            
        Returns:
            logits: [batch, caption_len, vocab_size] - caption predictions
        """
        # Encode STEP
        encoded = self.encoder(token_ids, topology_data)  # [batch, output_dim]
        
        # Expand for decoder
        memory = encoded.unsqueeze(1)  # [batch, 1, output_dim]
        
        if caption_ids is not None:
            # Training mode: teacher forcing
            caption_embed = self.caption_decoder(self.caption_embedding(caption_ids), memory)
            logits = self.output_projection(caption_embed)
        else:
            # Inference mode: autoregressive generation
            # (Not implemented - would need beam search)
            raise NotImplementedError("Use generate() method for inference")
        
        return logits

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        topology_data: Optional[Dict] = None,
        max_length: int = 64,
        num_beams: int = 4,
        temperature: float = 1.0,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
    ) -> torch.Tensor:
        """
        Generate captions using beam search decoding.

        Args:
            token_ids: [batch, seq_len] STEP tokens
            topology_data: Optional topology dict
            max_length: Maximum caption length to generate
            num_beams: Number of beams for beam search (1 = greedy)
            temperature: Sampling temperature (lower = more deterministic)
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID

        Returns:
            generated_ids: [batch, generated_len] generated caption token IDs
        """
        batch_size = token_ids.size(0)
        device = token_ids.device

        # Encode STEP
        encoded = self.encoder(token_ids, topology_data)  # [batch, output_dim]
        memory = encoded.unsqueeze(1)  # [batch, 1, output_dim]

        # Initialize with BOS token
        generated = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )

        # Expand for beam search
        if num_beams > 1:
            memory = memory.repeat_interleave(num_beams, dim=0)
            generated = generated.repeat_interleave(num_beams, dim=0)
            beam_scores = torch.zeros(batch_size * num_beams, device=device)
            beam_scores[1::num_beams] = float('-inf')  # Only first beam active initially
        else:
            beam_scores = torch.zeros(batch_size, device=device)

        # Autoregressive generation
        finished = torch.zeros(
            batch_size * num_beams if num_beams > 1 else batch_size,
            dtype=torch.bool,
            device=device
        )

        for _ in range(max_length - 1):
            # Embed generated tokens
            tgt = self.caption_embedding(generated)

            # Decode
            decoded = self.caption_decoder(tgt, memory)
            logits = self.output_projection(decoded[:, -1, :])  # [batch*beams, vocab]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Get log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            if num_beams > 1:
                # Beam search step
                vocab_size = log_probs.size(-1)
                next_scores = beam_scores.unsqueeze(-1) + log_probs  # [batch*beams, vocab]

                # Reshape for beam selection
                next_scores = next_scores.view(batch_size, num_beams * vocab_size)
                next_scores, next_tokens = next_scores.topk(num_beams, dim=-1)

                # Compute beam and token indices
                beam_indices = next_tokens // vocab_size
                token_indices = next_tokens % vocab_size

                # Update sequences
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1)
                flat_beam_indices = (batch_indices * num_beams + beam_indices).view(-1)

                generated = torch.cat([
                    generated[flat_beam_indices],
                    token_indices.view(-1, 1)
                ], dim=-1)

                beam_scores = next_scores.view(-1)

                # Check for EOS
                finished = finished[flat_beam_indices] | (token_indices.view(-1) == eos_token_id)
            else:
                # Greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                finished = finished | (next_token.squeeze(-1) == eos_token_id)

            # Stop if all sequences have finished
            if finished.all():
                break

        # Return best beam for each batch
        if num_beams > 1:
            # Reshape and select best
            generated = generated.view(batch_size, num_beams, -1)[:, 0, :]

        return generated


class STEPForClassification(nn.Module):
    """
    STEP encoder with classification head.
    Predicts: Part category.
    
    Use case: "bracket", "housing", "shaft", "gear", etc.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        num_classes: int = 100,
        output_dim: int = 1024
    ):
        super().__init__()
        
        self.encoder = STEPEncoder(vocab_size=vocab_size, output_dim=output_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(
        self,
        token_ids: torch.Tensor,
        topology_data: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len]
            topology_data: Optional topology
            
        Returns:
            logits: [batch, num_classes]
        """
        encoded = self.encoder(token_ids, topology_data)
        logits = self.classifier(encoded)
        return logits


class STEPForPropertyPrediction(nn.Module):
    """
    STEP encoder with regression head.
    Predicts: Physical properties (volume, mass, surface area, etc.)
    
    Use case: Predict part weight, bounding box dimensions, etc.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        num_properties: int = 10,  # volume, mass, bbox dimensions, etc.
        output_dim: int = 1024
    ):
        super().__init__()
        
        self.encoder = STEPEncoder(vocab_size=vocab_size, output_dim=output_dim)
        
        self.regressor = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_properties)
        )
        
    def forward(
        self,
        token_ids: torch.Tensor,
        topology_data: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len]
            topology_data: Optional topology
            
        Returns:
            properties: [batch, num_properties]
        """
        encoded = self.encoder(token_ids, topology_data)
        properties = self.regressor(encoded)
        return properties


class STEPForSimilarity(nn.Module):
    """
    STEP encoder for similarity/retrieval tasks.
    Predicts: Embedding for similar part search.
    
    Use case: "Find similar CAD parts in database"
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 512
    ):
        super().__init__()
        
        self.encoder = STEPEncoder(vocab_size=vocab_size, output_dim=1024)
        
        # Project to normalized embedding space
        self.projection = nn.Sequential(
            nn.Linear(1024, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(
        self,
        token_ids: torch.Tensor,
        topology_data: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len]
            topology_data: Optional topology
            
        Returns:
            embeddings: [batch, embedding_dim] - L2 normalized
        """
        encoded = self.encoder(token_ids, topology_data)
        embeddings = self.projection(encoded)
        
        # L2 normalize for cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class STEPForQA(nn.Module):
    """
    STEP encoder for question answering.
    Predicts: Answer to questions about the CAD part.
    
    Use case: 
        Q: "How many holes does this part have?"
        A: "4"
    """
    
    def __init__(
        self,
        step_vocab_size: int = 50000,
        text_vocab_size: int = 50000,
        output_dim: int = 1024
    ):
        super().__init__()
        
        # STEP encoder
        self.step_encoder = STEPEncoder(vocab_size=step_vocab_size, output_dim=output_dim)
        
        # Question encoder
        self.question_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=output_dim, nhead=8, batch_first=True),
            num_layers=3
        )
        self.question_embedding = nn.Embedding(text_vocab_size, output_dim)
        self.answer_embedding = nn.Embedding(text_vocab_size, output_dim)

        # Answer decoder
        self.answer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=output_dim, nhead=8, batch_first=True),
            num_layers=6
        )
        
        self.output_projection = nn.Linear(output_dim, text_vocab_size)
        
    def forward(
        self,
        step_token_ids: torch.Tensor,
        question_token_ids: torch.Tensor,
        answer_token_ids: Optional[torch.Tensor] = None,
        topology_data: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Args:
            step_token_ids: [batch, step_seq_len]
            question_token_ids: [batch, q_seq_len]
            answer_token_ids: [batch, a_seq_len] (for training)
            topology_data: Optional
            
        Returns:
            logits: [batch, a_seq_len, vocab_size]
        """
        # Encode STEP
        step_encoded = self.step_encoder(step_token_ids, topology_data)  # [batch, dim]
        
        # Encode question
        q_embed = self.question_embedding(question_token_ids)
        q_encoded = self.question_encoder(q_embed)  # [batch, q_len, dim]
        
        # Combine STEP and question as memory for decoder
        memory = torch.cat([
            step_encoded.unsqueeze(1),  # [batch, 1, dim]
            q_encoded                    # [batch, q_len, dim]
        ], dim=1)  # [batch, 1+q_len, dim]
        
        # Decode answer
        if answer_token_ids is not None:
            a_embed = self.answer_embedding(answer_token_ids)
            a_decoded = self.answer_decoder(a_embed, memory)
            logits = self.output_projection(a_decoded)
        else:
            raise NotImplementedError("Use generate() for inference")

        return logits

    @torch.no_grad()
    def generate(
        self,
        step_token_ids: torch.Tensor,
        question_token_ids: torch.Tensor,
        topology_data: Optional[Dict] = None,
        max_length: int = 64,
        num_beams: int = 4,
        temperature: float = 1.0,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
    ) -> torch.Tensor:
        """
        Generate answers using beam search decoding.

        Args:
            step_token_ids: [batch, step_seq_len] STEP tokens
            question_token_ids: [batch, q_seq_len] question tokens
            topology_data: Optional topology dict
            max_length: Maximum answer length to generate
            num_beams: Number of beams for beam search (1 = greedy)
            temperature: Sampling temperature (lower = more deterministic)
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID

        Returns:
            generated_ids: [batch, generated_len] generated answer token IDs
        """
        batch_size = step_token_ids.size(0)
        device = step_token_ids.device

        # Encode STEP
        step_encoded = self.step_encoder(step_token_ids, topology_data)  # [batch, dim]

        # Encode question
        q_embed = self.question_embedding(question_token_ids)
        q_encoded = self.question_encoder(q_embed)  # [batch, q_len, dim]

        # Combine STEP and question as memory for decoder
        memory = torch.cat([
            step_encoded.unsqueeze(1),  # [batch, 1, dim]
            q_encoded                    # [batch, q_len, dim]
        ], dim=1)  # [batch, 1+q_len, dim]

        # Initialize with BOS token
        generated = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )

        # Expand for beam search
        if num_beams > 1:
            memory = memory.repeat_interleave(num_beams, dim=0)
            generated = generated.repeat_interleave(num_beams, dim=0)
            beam_scores = torch.zeros(batch_size * num_beams, device=device)
            beam_scores[1::num_beams] = float('-inf')  # Only first beam active initially
        else:
            beam_scores = torch.zeros(batch_size, device=device)

        # Autoregressive generation
        finished = torch.zeros(
            batch_size * num_beams if num_beams > 1 else batch_size,
            dtype=torch.bool,
            device=device
        )

        for _ in range(max_length - 1):
            # Get current answer embeddings
            a_embed = self.answer_embedding(generated)

            # Decode
            a_decoded = self.answer_decoder(a_embed, memory)
            logits = self.output_projection(a_decoded[:, -1, :])  # [batch*beams, vocab]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Get log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            if num_beams > 1:
                # Beam search step
                vocab_size = log_probs.size(-1)
                next_scores = beam_scores.unsqueeze(-1) + log_probs  # [batch*beams, vocab]

                # Reshape for beam selection
                next_scores = next_scores.view(batch_size, num_beams * vocab_size)
                next_scores, next_tokens = next_scores.topk(num_beams, dim=-1)

                # Compute beam and token indices
                beam_indices = next_tokens // vocab_size
                token_indices = next_tokens % vocab_size

                # Update sequences
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1)
                flat_beam_indices = (batch_indices * num_beams + beam_indices).view(-1)

                generated = torch.cat([
                    generated[flat_beam_indices],
                    token_indices.view(-1, 1)
                ], dim=-1)

                beam_scores = next_scores.view(-1)

                # Check for EOS
                finished = finished[flat_beam_indices] | (token_indices.view(-1) == eos_token_id)
            else:
                # Greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                finished = finished | (next_token.squeeze(-1) == eos_token_id)

            # Stop if all sequences have finished
            if finished.all():
                break

        # Return best beam for each batch
        if num_beams > 1:
            # Reshape and select best
            generated = generated.view(batch_size, num_beams, -1)[:, 0, :]

        return generated
