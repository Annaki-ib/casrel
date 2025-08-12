from typing import Any, Sequence, cast
from transformers import PreTrainedTokenizerBase, BatchEncoding


def align_entities_with_tokens(
    batch: dict,
    tokenizer: PreTrainedTokenizerBase
) -> dict[str, Any]:
    """
    Align character-level subject, object, and entity spans to token-level spans using a tokenizer.

    Args:
        batch (dict): A batch dictionary with keys:
            - "text": List[str], input sentences
            - "subj_char_span_starts": List[List[int]], subject char start positions
            - "subj_char_span_ends": List[List[int]], subject char end positions
            - "obj_char_span_starts": List[List[int]], object char start positions
            - "obj_char_span_ends": List[List[int]], object char end positions
            - optionally entity spans as well
        tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer instance.

    Returns:
        dict[str, Any]: Dictionary including tokenized batch and aligned token spans.
    """
    tokenized_batch: BatchEncoding = tokenizer.batch_encode_plus(
        batch["text"], add_special_tokens=True, truncation=True
    )

    subject_texts, object_texts = [], []
    subj_token_span_starts, subj_token_span_ends = [], []
    obj_token_span_starts, obj_token_span_ends = [], []

    for sample_index, (subject, obj) in enumerate(zip(
        zip(batch["subj_char_span_starts"], batch["subj_char_span_ends"]),
        zip(batch["obj_char_span_starts"], batch["obj_char_span_ends"])
    )):
        sample_subj_starts, sample_subj_ends = [], []
        sample_obj_starts, sample_obj_ends = [], []
        sample_subj_tokens, sample_obj_tokens = [], []

        for subj_start, subj_end, obj_start, obj_end in zip(*subject, *obj):
            if subj_start == subj_end or obj_start == obj_end:
                continue

            subj_start_token = tokenized_batch.char_to_token(sample_index, subj_start)
            subj_end_token = tokenized_batch.char_to_token(sample_index, subj_end - 1)
            obj_start_token = tokenized_batch.char_to_token(sample_index, obj_start)
            obj_end_token = tokenized_batch.char_to_token(sample_index, obj_end - 1)

            if None in (subj_start_token, subj_end_token, obj_start_token, obj_end_token):
                continue

            subj_token_ids = tokenized_batch.data["input_ids"][sample_index][
                subj_start_token: subj_end_token + 1]
            obj_token_ids = tokenized_batch.data["input_ids"][sample_index][
                obj_start_token: obj_end_token + 1]

            sample_subj_starts.append(subj_start_token)
            sample_subj_ends.append(subj_end_token)
            sample_subj_tokens.append(cast(list[int], subj_token_ids))

            sample_obj_starts.append(obj_start_token)
            sample_obj_ends.append(obj_end_token)
            sample_obj_tokens.append(cast(list[int], obj_token_ids))

        subject_texts.append(tokenizer.batch_decode(sample_subj_tokens))
        object_texts.append(tokenizer.batch_decode(sample_obj_tokens))
        subj_token_span_starts.append(sample_subj_starts)
        subj_token_span_ends.append(sample_subj_ends)
        obj_token_span_starts.append(sample_obj_starts)
        obj_token_span_ends.append(sample_obj_ends)

    # Note: Entity alignment can be added similarly if your dataset contains entity spans

    return {
        "subjects": subject_texts,
        "objects": object_texts,
        "subj_token_span_starts": subj_token_span_starts,
        "subj_token_span_ends": subj_token_span_ends,
        "obj_token_span_starts": obj_token_span_starts,
        "obj_token_span_ends": obj_token_span_ends,
        **tokenized_batch,
    }
