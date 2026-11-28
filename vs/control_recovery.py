"""Compare a letter-words control's fitted key with its known clean and received forms."""

from functools import lru_cache

from .families import letter_words

ROLES = ("prefix", "suffix")


def letter_words_recovery(planted, fitted, clean_pages, received_pages):
    """Report recovery and omitted-role support separately for training and evaluation.

    Both page pairs come from the same planted control seed. The second pair
    may have sign substitutions, but it has the same ciphertext word slots.
    Use the known key to identify the original and received sign roles.
    """
    @lru_cache(maxsize=None)
    def role_signs(word):
        parts = letter_words.split(word, planted) or ("", "")
        return tuple((role, sign) for role, sign in zip(ROLES, parts) if sign)

    @lru_cache(maxsize=None)
    def true_group(word):
        return letter_words.decode_word(word, planted, None)

    @lru_cache(maxsize=None)
    def fitted_group(word):
        return letter_words.decode_word(word, fitted, None)

    def words(pages):
        return [word for page in pages for line in page.lines for word in line.words]

    clean_train, clean_evaluation = map(words, clean_pages)
    received_train, received_evaluation = map(words, received_pages)
    clean_support = {item for word in clean_train for item in role_signs(word)}

    def partition(clean, received):
        omitted_clean = omitted_received = unsupported = changed = 0
        same_clean = same_received = 0
        omitted_types = set()
        for original, observed in zip(clean, received, strict=True):
            original_signs = role_signs(original)
            omitted_clean += sum(sign not in fitted[role] for role, sign in original_signs)
            output = fitted_group(observed)
            same_clean += output == true_group(original)
            same_received += output == true_group(observed)
            for role, sign in role_signs(observed):
                if sign in fitted[role]:
                    continue
                item = role, sign
                omitted_types.add(item)
                omitted_received += 1
                unsupported += item not in clean_support
                changed += item not in original_signs
        return {
            "ciphertext_groups": len(received),
            "exact_against_clean": same_clean,
            "exact_against_received": same_received,
            "omitted_received_sign_types": len(omitted_types),
            "omitted_clean_sign_occurrences": omitted_clean,
            "omitted_received_sign_occurrences": omitted_received,
            "omitted_received_occurrences_without_clean_training_support": unsupported,
            "omitted_received_occurrences_at_changed_positions": changed,
        }

    return {
        "roles": {
            role: {
                "fitted_signs": len(fitted[role]),
                "correct_fitted_signs": sum(planted[role].get(sign) == value
                                            for sign, value in fitted[role].items()),
            } for role in ROLES
        },
        "training": partition(clean_train, received_train),
        "evaluation": partition(clean_evaluation, received_evaluation),
    }
