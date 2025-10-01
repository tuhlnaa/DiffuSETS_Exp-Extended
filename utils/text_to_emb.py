"""
ECG text embedding utilities for DiffuSETS.
Handles conversion of clinical diagnosis text to embeddings.
"""
import ast
import json
import pandas as pd
from typing import List


def format_diagnosis(diagnosis: str, index: int) -> str:
    """Format a single diagnosis with appropriate emphasis and ordinal representation."""
    # Generate ordinal suffix
    ORDINAL_SUFFIXES = ['1st', '2nd', '3rd']
    ordinal = ORDINAL_SUFFIXES[index] if index < len(ORDINAL_SUFFIXES) else f"{index + 1}th"
    
    if index == 0:
        return f"Most importantly, the {ordinal} diagnosis is {{{diagnosis}}}."
    else:
        return f"As a supplementary condition, the {ordinal} diagnosis is {{{diagnosis}}}."
    

def process_prompt(text: str, delimiter: str = '|') -> str:
    """
    Convert pipe-delimited diagnosis text into a formatted prompt.
    
    Args:
        text: Pipe-delimited diagnosis string (e.g., "diagnosis1|diagnosis2|diagnosis3")
        delimiter: Character used to separate diagnoses
    
    Returns:
        Formatted prompt text with all diagnoses
    
    Example:
        >>> process_prompt("Atrial fibrillation|Hypertension")
        'Most importantly, the 1st diagnosis is {Atrial fibrillation}.As a supplementary condition, the 2nd diagnosis is {Hypertension}.'
    """
    diagnoses = [d.strip() for d in text.split(delimiter) if d.strip()]
    
    if not diagnoses:
        return ""
    
    return ''.join(format_diagnosis(diag, idx) for idx, diag in enumerate(diagnoses))


def safe_parse_embedding(embed_str: str) -> List[float]:
    """
    Safely parse embedding string to list of floats.
    
    Args:
        embed_str: String representation of embedding (JSON or Python literal)
    
    Returns:
        List of float values representing the embedding
    
    Raises:
        ValueError: If parsing fails
    """
    try:
        # Try JSON first (safer)
        return json.loads(embed_str)
    except (json.JSONDecodeError, TypeError):
        try:
            # Fall back to ast.literal_eval (safer than eval)
            return ast.literal_eval(embed_str)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Failed to parse embedding: {embed_str[:100]}...") from e


def get_text_embeddings(
        text_batch: List[str],
        text_embed_table: pd.DataFrame,
        text_col: str = 'text',
        embed_col: str = 'embed'
    ) -> List[List[float]]:
    """
    Retrieve or generate text embeddings for a batch of diagnosis texts.
    
    Args:
        text_batch: List of pipe-delimited diagnosis strings
        text_embed_table: DataFrame containing pre-computed embeddings
        text_col: Column name for text in the DataFrame
        embed_col: Column name for embeddings in the DataFrame
    
    Returns:
        List of embeddings (each embedding is a list of floats)
    
    Raises:
        ValueError: If embedding parsing fails
        KeyError: If required columns are missing from DataFrame
    """
    # Validate DataFrame structure
    if text_col not in text_embed_table.columns or embed_col not in text_embed_table.columns:
        raise KeyError(f"DataFrame must contain '{text_col}' and '{embed_col}' columns")
    
    # Create a lookup dictionary for O(1) access instead of repeated DataFrame filtering
    embed_lookup = dict(zip(
        text_embed_table[text_col],
        text_embed_table[embed_col]
    ))
    
    # Get default embedding (last row) for missing entries
    default_embed_str = text_embed_table.iloc[-1][embed_col]
    default_embed = safe_parse_embedding(default_embed_str)
    
    embeddings = []
    for text in text_batch:
        prompt_text = process_prompt(text)
        
        # Look up embedding
        embed_str = embed_lookup.get(prompt_text)
        
        if embed_str is not None:
            try:
                embed = safe_parse_embedding(embed_str)
            except ValueError as e:
                print(f"Warning: Failed to parse embedding for '{prompt_text[:50]}...': {e}")
                embed = default_embed
        else:
            print(f"Warning: No embedding found for prompt: '{prompt_text[:50]}...'. Using default.")
            embed = default_embed
        
        embeddings.append(embed)
    
    return embeddings


# def get_text_embeddings(text_batch, text_embed_table): 
#     # text_batch -> (B, 1536) 
#     text_embed = [] 
#     for text in text_batch:
#         prompt_text = process_prompt(text) 
#         if len(text_embed_table.loc[text_embed_table['text'] == prompt_text, 'embed']) > 0:
#             embed = text_embed_table.loc[text_embed_table['text'] == prompt_text, 'embed'].values[0]
#         else:
#             print(1)
#             embed = text_embed_table.iloc[-1]['embed']
#         embed = eval(embed)
#         text_embed.append(embed)

#     return text_embed 
