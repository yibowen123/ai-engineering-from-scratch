"""Refined data pipeline: document-aware sequence packing.

This is a drop-in companion to ``main.py``. It reuses the cleaning, quality
filtering, deduplication and BPE tokenizer from that file unchanged, and only
replaces the packing + masking + dataloader stages.

The point of this file is the difference between two kinds of "mask":

  * ``main.py`` builds a 1-D *padding mask* (1 = real token, 0 = pad). It tells
    the model which positions are padding, but does NOT stop tokens of one
    document from attending backwards into a different document that happens to
    share the same packed sequence. That is "cross-document contamination".

  * ``refined_main.py`` builds a 2-D *causal + document mask*. Position i may
    attend to position j only when: (a) j <= i (causal), and (b) i and j belong
    to the same document. Attention therefore RESETS at every <EOS> boundary,
    even though documents are still concatenated and packed for efficiency.

Everything is kept in pure Python (nested lists) to match main.py and to keep
the block-diagonal structure easy to print and inspect.
"""

import random
import time

from main import (
    clean_text,
    quality_filter,
    deduplicate,
    SimpleTokenizer,
    generate_sample_corpus,
    compute_statistics,
)


# A sentinel segment id for padding positions. Real tokens carry the integer
# index of the document they came from; padding carries PAD_SEGMENT so it can
# never match a real document and never attends to (or is attended by) anything.
PAD_SEGMENT = -1


def tokenize_corpus_with_docids(documents, tokenizer):
    """Flatten the corpus into one token stream, tracking each token's document.

    Same concatenate-with-<EOS> layout as main.tokenize_corpus, but we also emit
    a parallel ``doc_ids`` list so packing can remember which document produced
    each token. The <EOS> that terminates a document is assigned to that same
    document, so a document can attend to its own end-of-sequence marker.
    """
    all_tokens = []
    doc_ids = []
    for doc_idx, doc in enumerate(documents):
        tokens = tokenizer.encode(doc)
        for t in tokens:
            all_tokens.append(t)
            doc_ids.append(doc_idx)
        all_tokens.append(tokenizer.eos_id)
        doc_ids.append(doc_idx)
    return all_tokens, doc_ids


def pack_sequences(token_ids, doc_ids, seq_length, pad_id=0):
    """Chunk the flat stream into fixed windows, carrying document ids along.

    Returns three parallel lists per sequence:
      * sequences[k]      -> the token ids
      * attention_masks[k]-> 1-D padding mask (kept for compatibility / stats)
      * segment_ids[k]    -> document id per position (PAD_SEGMENT for padding)

    Note we slice every ``seq_length`` tokens exactly like main.py: a document
    may still straddle a window boundary. Within a single window, though, its
    tokens are contiguous, so they form one clean block in the 2-D mask.
    """
    sequences = []
    attention_masks = []
    segment_ids = []
    for i in range(0, len(token_ids), seq_length):
        seq = token_ids[i:i + seq_length]
        seg = doc_ids[i:i + seq_length]
        mask = [1] * len(seq)
        if len(seq) < seq_length:
            pad_count = seq_length - len(seq)
            seq = seq + [pad_id] * pad_count
            seg = seg + [PAD_SEGMENT] * pad_count
            mask = mask + [0] * pad_count
        sequences.append(seq)
        attention_masks.append(mask)
        segment_ids.append(seg)
    return sequences, attention_masks, segment_ids


def build_causal_document_mask(segment_ids):
    """Build the 2-D attention mask for one packed sequence.

    mask[i][j] == 1  iff  token j is visible to token i, which requires:
        j <= i                       (causal / no peeking at the future)
        segment_ids[i] == segment_ids[j]   (same document)
        segment_ids[i] != PAD_SEGMENT      (a pad query attends to nothing)

    The result is block-diagonal: one lower-triangular block per document, with
    zeros everywhere across document boundaries. This is what actually prevents
    cross-document attention -- the <EOS> token alone does not.
    """
    n = len(segment_ids)
    mask = [[0] * n for _ in range(n)]
    for i in range(n):
        si = segment_ids[i]
        if si == PAD_SEGMENT:
            continue  # padding rows see nothing
        row = mask[i]
        for j in range(i + 1):  # causal: only j <= i
            if segment_ids[j] == si:
                row[j] = 1
    return mask


def visualize_mask(mask, segment_ids=None, max_n=40):
    """Print a 2-D mask as an ASCII grid so the block structure is visible.

    '#' means "can attend", '.' means "blocked". If segment_ids is given, the
    document id of each row is printed on the left to make the blocks obvious.
    """
    n = min(len(mask), max_n)
    for i in range(n):
        cells = "".join("#" if mask[i][j] else "." for j in range(n))
        if segment_ids is not None:
            seg = segment_ids[i]
            label = "pad" if seg == PAD_SEGMENT else f"d{seg}"
            print(f"    {label:>4} |{cells}")
        else:
            print(f"    |{cells}")


class DocumentAwareDataLoader:
    """Like main.PreTrainingDataLoader, but also serves the 2-D document masks.

    Each batch yields (batch_seqs, batch_pad_masks, batch_doc_masks). The 2-D
    masks are built lazily per batch rather than stored up front, since a full
    seq_length x seq_length matrix per sequence is memory-heavy.
    """

    def __init__(self, sequences, attention_masks, segment_ids, batch_size, shuffle=True):
        self.sequences = sequences
        self.attention_masks = attention_masks
        self.segment_ids = segment_ids
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            batch_seqs = [self.sequences[i] for i in batch_idx]
            batch_pad_masks = [self.attention_masks[i] for i in batch_idx]
            batch_doc_masks = [build_causal_document_mask(self.segment_ids[i]) for i in batch_idx]
            yield batch_seqs, batch_pad_masks, batch_doc_masks


def count_cross_document_pairs(segment_ids):
    """How many (i, j) causal pairs the naive 1-D mask would have allowed to
    leak across a document boundary. This is exactly what the 2-D mask removes.
    """
    n = len(segment_ids)
    leaked = 0
    for i in range(n):
        si = segment_ids[i]
        if si == PAD_SEGMENT:
            continue
        for j in range(i + 1):
            sj = segment_ids[j]
            if sj == PAD_SEGMENT:
                continue
            if sj != si:
                leaked += 1
    return leaked


def demo_mask_visualization(tokenizer):
    """Tiny, self-contained demo so the block-diagonal structure is legible.

    We build three short synthetic 'documents', pack them into one small window,
    and print both the segment ids and the resulting 2-D mask.
    """
    print("\n--- Mask Visualization Demo (small window) ---")
    docs = ["alpha beta gamma", "delta epsilon", "zeta eta theta iota"]
    token_ids, doc_ids = tokenize_corpus_with_docids(docs, tokenizer)

    seq_length = 24
    sequences, _, segment_ids = pack_sequences(token_ids, doc_ids, seq_length)
    seg = segment_ids[0]
    mask = build_causal_document_mask(seg)

    print(f"  Packed one window of {seq_length} positions from 3 documents.")
    print(f"  Segment ids (document per position):\n    {seg}")
    print("  '#' = can attend, '.' = blocked. Notice one triangular block per doc:")
    visualize_mask(mask, seg)

    leaked = count_cross_document_pairs(seg)
    print(f"  Cross-document attention pairs blocked by the 2-D mask: {leaked}")


def run_pipeline():
    print("=" * 60)
    print("Refined Data Pipeline (document-aware attention masking)")
    print("=" * 60)

    raw_docs = generate_sample_corpus()
    print(f"\nRaw documents: {len(raw_docs)}")

    print("\n--- Stage 1: Cleaning ---")
    cleaned_docs = [clean_text(doc) for doc in raw_docs]
    print(f"After HTML stripping: {len(cleaned_docs)} documents")

    print("\n--- Stage 2: Quality Filtering ---")
    filtered_docs = [doc for doc in cleaned_docs if quality_filter(doc)]
    removed_quality = len(cleaned_docs) - len(filtered_docs)
    print(f"Removed {removed_quality} low-quality documents")
    print(f"Remaining: {len(filtered_docs)} documents")

    print("\n--- Stage 3: Deduplication (MinHash + LSH) ---")
    start = time.time()
    deduped_docs, num_removed = deduplicate(filtered_docs, threshold=0.8)
    dedup_time = time.time() - start
    print(f"Removed {num_removed} near-duplicates in {dedup_time:.2f}s")
    print(f"Remaining: {len(deduped_docs)} documents")

    print("\n--- Stage 4: Tokenization (with per-token document ids) ---")
    all_text = " ".join(deduped_docs)
    tokenizer = SimpleTokenizer()
    start = time.time()
    tokenizer.train_bpe(all_text, num_merges=100)
    train_time = time.time() - start
    print(f"Trained tokenizer with {tokenizer.vocab_size()} tokens in {train_time:.2f}s")

    start = time.time()
    token_ids, doc_ids = tokenize_corpus_with_docids(deduped_docs, tokenizer)
    tok_time = time.time() - start
    print(f"Tokenized {len(token_ids):,} tokens in {tok_time:.2f}s "
          f"({len(token_ids)/max(tok_time, 0.001):,.0f} tokens/sec)")

    print("\n--- Stage 5: Document-Aware Sequence Packing ---")
    seq_length = 128
    sequences, masks, segment_ids = pack_sequences(token_ids, doc_ids, seq_length, pad_id=0)
    print(f"Packed into {len(sequences)} sequences of length {seq_length}")

    # Measure how much cross-document leakage the naive 1-D mask would allow, and
    # how many documents share each packed sequence.
    total_leaked = sum(count_cross_document_pairs(seg) for seg in segment_ids)
    docs_per_seq = [len({s for s in seg if s != PAD_SEGMENT}) for seg in segment_ids]
    avg_docs_per_seq = sum(docs_per_seq) / max(len(docs_per_seq), 1)
    print(f"Avg distinct documents per packed sequence: {avg_docs_per_seq:.2f}")
    print(f"Max documents in a single sequence:         {max(docs_per_seq)}")
    print(f"Cross-document attention pairs blocked:      {total_leaked:,}")
    print("(main.py's 1-D padding mask would have ALLOWED all of those.)")

    print("\n--- Stage 6: Document-Aware DataLoader ---")
    batch_size = 4
    loader = DocumentAwareDataLoader(sequences, masks, segment_ids, batch_size)
    print(f"DataLoader: {len(loader)} batches of size {batch_size}")

    batch_count = 0
    for batch_seqs, batch_pad_masks, batch_doc_masks in loader:
        batch_count += 1
        if batch_count <= 1:
            first_seg = None
            # Recover the segment ids of the first sequence for a small preview.
            m = batch_doc_masks[0]
            print(f"\n  Batch {batch_count}:")
            print(f"    Sequences in batch:  {len(batch_seqs)}")
            print(f"    First seq (first 16 tokens): {batch_seqs[0][:16]}...")
            print(f"    First pad mask (first 16):   {batch_pad_masks[0][:16]}...")
            print(f"    2-D doc mask shape:  {len(m)} x {len(m[0])}")
            print("    Top-left 16x16 corner of the 2-D causal+document mask:")
            corner = [row[:16] for row in m[:16]]
            visualize_mask(corner)
    print(f"\n  Total batches served: {batch_count}")

    # Reuse main.py's statistics for parity with the original pipeline.
    print("\n--- Dataset Statistics ---")
    stats = compute_statistics(deduped_docs, token_ids, sequences, tokenizer.vocab_size())
    print(f"  Documents:          {stats['total_documents']}")
    print(f"  Total tokens:       {stats['total_tokens']:,}")
    print(f"  Num sequences:      {stats['num_sequences']}")
    print(f"  Seq utilization:    {stats['sequence_utilization']:.1%}")

    # Standalone visualization so the block-diagonal shape is unmistakable.
    demo_mask_visualization(tokenizer)

    print("\n--- Pipeline Summary ---")
    print(f"  Final tokens:            {len(token_ids):,}")
    print(f"  Training sequences:      {len(sequences)}")
    print(f"  Training batches:        {len(loader)}")
    print(f"  Cross-doc pairs blocked: {total_leaked:,}")


if __name__ == "__main__":
    run_pipeline()
