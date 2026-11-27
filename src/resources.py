"""A resource-level proxy per language, for the partial-correlation confound control.

The number is an ordinal resource class in the spirit of Joshi et al. 2020,
"The State and Fate of Linguistic Diversity and Inclusion in the NLP World"
(0 = least-resourced, 5 = English). It is a PROXY, not a measured token count,
and it is used only as the covariate held fixed in `partial_spearman`, where
only the ordering matters. Keys are the short ISO codes the harness uses, with
the AfriMGSM 3-letter variants aliased to the same value.

If a language is missing here it is simply dropped from the partial-correlation
step (and a note is logged), never guessed.
"""
from __future__ import annotations

RESOURCE_CLASS = {
    # high resource
    "en": 5,
    "de": 4, "fr": 4, "es": 4, "zh": 4, "ja": 4, "ru": 4, "pt": 4, "it": 4,
    "nl": 4, "ar": 4, "ko": 4, "hi": 4, "tr": 4, "pl": 4, "sv": 4, "fi": 4,
    # mid resource
    "vi": 3, "id": 3, "fa": 3, "uk": 3, "cs": 3, "ro": 3, "el": 3, "he": 3,
    "th": 3, "bn": 3, "ta": 3, "hu": 3,
    # lower resource (incl. the African set; 3-letter AfriMGSM keys aliased)
    "sw": 2, "swa": 2, "zul": 3, "zu": 3, "xho": 2, "xh": 2, "yor": 2, "yo": 2,
    "amh": 2, "am": 2, "hau": 2, "ha": 2, "wol": 2, "wo": 2, "sot": 2, "st": 2,
    "ibo": 1, "ig": 1, "sna": 1, "sn": 1, "lin": 1, "ln": 1, "lug": 1, "lg": 1,
    "kin": 1, "rw": 1, "orm": 1, "om": 1, "twi": 1, "ak": 1, "ewe": 1, "ee": 1,
    "te": 1,
}


def resource_of(iso: str):
    """Resource class for a harness ISO code, or None if unknown."""
    return RESOURCE_CLASS.get(iso)
