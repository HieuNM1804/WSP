"""What an edit may change. Shared by save_edit (in the sandbox), the forward-pass tool and the final evaluation,
so a file written into /work/edited by other means than save_edit is checked the same way before it is used.

Any tensor may be edited (only the image encoder, "visual.*", affects the retrieval); the checks are technical:
the same shape as the original and finite values.
"""
import numpy as np


def is_editable(name):
    return True


def affects_retrieval(name):
    return name.startswith("visual.")


def check_edit(name, edited, original):
    """Return None if the edit is allowed, else the reason it is refused."""
    edited = np.asarray(edited)
    if tuple(edited.shape) != tuple(original.shape):
        return f"{name}: shape {tuple(edited.shape)} differs from the original {tuple(original.shape)}"
    if not np.isfinite(edited).all():
        return f"{name}: contains NaN or infinite values"
    return None
