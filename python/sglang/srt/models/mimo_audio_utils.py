import torch


def group_by_length(features: torch.Tensor, lengths: torch.Tensor, max_length: int):
    if features.size(0) != lengths.sum().item():
        raise ValueError(
            f"Feature size mismatch: {features.size(0)} vs {lengths.sum().item()}"
        )

    split_points = []
    current_sum = 0

    for i, seq_len in enumerate(lengths):
        if current_sum + seq_len > max_length and current_sum > 0:
            split_points.append(i)
            current_sum = seq_len.item()
        else:
            current_sum += seq_len.item()

    # Convert split points to group sizes
    group_sizes = []
    prev = 0
    for point in split_points:
        group_sizes.append(point - prev)
        prev = point
    if prev < len(lengths):
        group_sizes.append(len(lengths) - prev)

    len_groups = torch.split(lengths, group_sizes)
    feature_sizes = [group.sum().item() for group in len_groups]
    feature_groups = torch.split(features, feature_sizes)

    return feature_groups, len_groups


@torch.no_grad()
def encode_batch(
    audio_tokenizer_encoder,
    input_features: torch.Tensor,
    input_lens: torch.Tensor,
    max_length: int = 256000,
):
    feature_groups, len_groups = group_by_length(input_features, input_lens, max_length)

    encoded_parts = []
    for features, lengths in zip(feature_groups, len_groups):
        codes, _ = audio_tokenizer_encoder.encode(  # codes are also packed
            input_features=features, input_lens=lengths, return_codes_only=True
        )
        encoded_parts.append(codes)

    return torch.cat(encoded_parts, dim=-1)


def _segment_lengths_for_mel(mel: torch.Tensor, segment_size: int):
    """Split mel into segments of segment_size with a possible shorter remainder."""
    input_len = mel.size(0)
    segs = [segment_size] * (input_len // segment_size)
    if input_len % segment_size > 0:
        segs.append(input_len % segment_size)
    return segs


@torch.no_grad()
def tokenize_audio_batch(mels, audio_tokenizer_encoder, segment_size=6000, device=None):
    """
    Tokenize multiple mels in one encode_batch call.
    Returns list of code tensors, each [T_i, C] for that mel.
    """
    if not mels:
        return []
    if device is None:
        device = next(audio_tokenizer_encoder.parameters()).device
    # Build segment lengths per mel
    input_len_seg_per_mel = [_segment_lengths_for_mel(m, segment_size) for m in mels]
    input_lens_flat = [s for segs in input_len_seg_per_mel for s in segs]
    input_features = torch.cat([m.to(device) for m in mels], dim=0)
    input_lens_t = torch.tensor(input_lens_flat, dtype=torch.long, device=device)
    codes_packed = encode_batch(
        audio_tokenizer_encoder,
        input_features=input_features,
        input_lens=input_lens_t,
    )
    codes = codes_packed.transpose(0, 1).detach()  # [total_code_T, C]
    # Code length per mel: must match encoder's actual output (get_output_length + optional avg_pooler downsampling)
    code_lengths = []
    for segs in input_len_seg_per_mel:
        out_len = audio_tokenizer_encoder.get_output_length(
            torch.tensor(segs, dtype=torch.long, device=device)
        )
        if getattr(audio_tokenizer_encoder, "down_sample_layer", None) is not None:
            avg = audio_tokenizer_encoder.config.avg_pooler
            out_len = out_len // avg + (out_len % avg != 0).long()
        code_lengths.append(out_len.sum().item())
    code_list = torch.split(codes, code_lengths)
    return list(code_list)
