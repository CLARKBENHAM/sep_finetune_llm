# %%
def old_compute_AB_sequence(tokens):
    # Define the replacement rules
    rules = {"A# #A": "", "A# #B": "#B A#", "B# #A": "#A B#", "B# #B": ""}

    # Function to apply the rules to the sequence
    def apply_rules(sequence):
        for key, value in rules.items():
            sequence = sequence.replace(key, value).replace("  ", " ")
        return sequence

    # Keep applying the rules until the sequence no longer changes
    while True:
        new_tokens = apply_rules(tokens)
        if new_tokens == tokens:  # No change means we are done
            break
        tokens = new_tokens

    return tokens


# old_compute_AB_sequence(example_sequence).split(" ")


def compute_AB_sequence(tokens):
    # Define the replacement rules
    rules = {
        ("A#", "#A"): [],
        ("A#", "#B"): ["#B", "A#"],
        ("B#", "#A"): ["#A", "B#"],
        ("B#", "#B"): [],
    }
    seq = tokens.split(" ")
    while True:
        for ix, (t1, t2) in enumerate(zip(seq[:-1], seq[1:])):
            if (t1, t2) in rules:
                seq[ix : ix + 2] = rules[t1, t2]
                break
        else:
            break
    return " ".join(seq)


import random

elements = ["A#", "#A", "B#", "#B"]

import pytest


def test_oldnew():
    for _ in range(999):
        example_sequence = " ".join([random.choice(elements) for _ in range(12)])
        n = compute_AB_sequence(example_sequence)
        o = old_compute_AB_sequence(example_sequence)
        assert n == o


# Compute the sequence
