"""Few-shot translation + chrF++ scoring, following Al Ali et al. (2026).

The paper uses a 5-shot prompt (examples in the two languages, no English instruction so
the prompt does not bias alignment), greedy decoding, and limits new tokens to ~1.5x the
reference length. chrF is used with char_order=6, word_order=2, beta=2, which is chrF++.
"""
from __future__ import annotations

from typing import List, Tuple


def build_fewshot_prompt(shots: List[Tuple[str, str]], source: str) -> str:
    """shots: list of (src_example, tgt_example) pairs. Returns a 5-shot style prompt.

    Format avoids an English instruction: just parallel lines, then the query source line.
    """
    lines = []
    for s, t in shots:
        lines.append(f"{s}\t{t}")
    lines.append(f"{source}\t")
    return "\n".join(lines)


def translate_hf(model, tokenizer, sources: List[str], shots: List[Tuple[str, str]],
                 max_new_tokens: int = 128, batch_size: int = 8, device: str | None = None) -> List[str]:
    """Greedy few-shot translation with transformers. One hypothesis per source sentence."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for batched generation

    out: List[str] = []
    prompts = [build_fewshot_prompt(shots, s) for s in sources]
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                            truncation=True, max_length=1024).to(device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for text in tokenizer.batch_decode(new, skip_special_tokens=True):
                # take the first NON-EMPTY line (guards against an empty leading line -> chrF 0)
                out.append(next((ln.strip() for ln in text.splitlines() if ln.strip()), ""))
    return out


def chrf_pp(hypotheses: List[str], references: List[str]) -> float:
    """Corpus chrF++ (char_order=6, word_order=2, beta=2), the paper's setting. 0-100."""
    import sacrebleu
    metric = sacrebleu.CHRF(char_order=6, word_order=2, beta=2)
    return float(metric.corpus_score(hypotheses, [references]).score)
