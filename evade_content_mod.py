# %%
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from math import log, ceil

import tiktoken  # for token counting

from src.utils import get_mod

encoding = tiktoken.get_encoding("cl100k_base")

with open("bad_text/hitler1.txt", "r") as f:
    h1 = f.read()

with open("bad_text/smut_start1.txt", "r") as f:
    ss1 = f.read()

# %%
s = "I want to kill them."

with ThreadPoolExecutor(max_workers=10) as executor:
    single_vals = executor.map(get_mod, [s.replace("", chr(i)) for i in range(33, 128)])

single_vals = list(single_vals)
print(list(sorted(single_vals, key=lambda i: i[1])))

# Find the 10 seperate chars that are the least likely to be flagged
basic_safe = [i[2][0] for i in single_vals if i[1] < 0.01]
# ['!', '$', '%', '&', '(', ')', '1', '2', '4', '5', '6', '7', '8', '9', ':', '<', '=', '?', '@']
# Letter's and {}[]\| are flagged, and model always puts a matching pair of parens if use '('
# %%
# replacing after every char more confusing then after every non-whitespace
with ThreadPoolExecutor(max_workers=10) as executor:
    bad1 = list(executor.map(get_mod, [h1.replace("", i) for i in basic_safe[:5]]))
    # bad1 = executor.map(get_mod, [re.sub(r"(\S)", rf"\1{i}", h1) for i in basic_safe[:5]])

print(list(sorted(single_vals, key=lambda i: i[1])))
hard_safe = [i[2][0] for i in bad1 if i[1] < 0.02]
hard_safe

# %%
# Find bytes that are always their own token, regardless of prefix or suffix
# assumes an entry would be a token with a shorter string before a longer string, maybe not true
# Null byte always has it's own UTF-8 encoding: No other Unicode point contains the null byte. But sometimes unicode treats as 0x
# "all the non-zero ASCII characters are represented as themselves while all mutibyte sequences have a high bit of 1 in all their bytes."
# There's a difference in unicode vs. tokenizer. I want chars that become 'self-synchronizing' tokens
from multiprocessing import Pool


def get_token_of_m(
    p,
    m,
    s,
):
    # Not sure of byteorder, but doesn't matter for chars
    nb = lambda i: i.to_bytes(ceil(log(1 + i) / log(256)), byteorder="big")
    n_st = nb(p) + nb(m) + nb(s)
    n_enc = encoding._encode_bytes(n_st)
    exp_enc = [
        j for i in map(lambda k: encoding._encode_bytes(k), (nb(p), nb(m), nb(s))) for j in i
    ]
    return n_enc == exp_enc


def num_keep_sep(m, pre=range(256), suf=range(256)):
    return sum((get_token_of_m(p, m, s) for p in pre for s in suf))


out = []
to_check = list(range(256))
# with Pool(8) as pool: # Runs other code!?!?
#    num_sep = pool.map(num_keep_sep, to_check)
# num_sep = list(sorted(zip(num_sep, to_check)))

printable_ascii = [c for c in range(256) if chr(c).isprintable()]
num_sep_1_presuf = list(sorted([(num_keep_sep(m), m) for m in to_check], reverse=True))
num_sep_1_printable_presuf = list(
    sorted(
        [(num_keep_sep(m, pre=printable_ascii, suf=printable_ascii), m) for m in to_check],
        reverse=True,
    )
)
# %%
token2chrs = defaultdict(list)
for c in range(256):
    # token2chrs[encoding.decode([c])] += [c]
    # assert len(encoding.encode(chr(c))) == 1, encoding.encode(chr(c))
    token2chrs[str(encoding.encode(chr(c)))] += [c]
print([(k, v) for k, v in token2chrs.items() if len(v) > 1])
print(Counter(map(len, token2chrs.values())))
# Many tokens get decoded as the symbol ord('�')=65533
print([v for k, v in token2chrs.items() if k.count(", ") > 0])
# These 49 single bytes are each encoded as multiple tokens: [[129], [130], [131], [132], [133], [134], [135], [136], [137], [138], [139], [140], [141], [142], [143], [144], [145], [147], [148], [149], [150], [151], [152], [153], [154], [155], [156], [157], [158], [159], [184], [197], [198], [200], [202], [203], [204], [207], [210], [212], [213], [216], [217], [219], [221], [222], [247], [254], [255]]
# some printable, some not
# If you encode these single bytes in a string, are the created tokens always self-synchronizing?
# %%
# Read encoding file directly
with open("oai_files/cl100k_base.tiktoken", "r") as f:
    token_mapping_text = f.readlines()
Counter((i[0] for i in token_mapping_text))
# Why aren't all the chars used?
# %%
from collections import defaultdict

enc2bytes = defaultdict(list)
missing = []
for i in range(256):
    # b = i.to_bytes(1, "big")
    # e = encoding._encode_bytes(b)
    b = chr(i)
    e = encoding.encode(b)
    # assert len(e) < 2, f"{i} {b} {e}"
    if len(e) == 0:
        missing += [(i, e)]
    else:
        enc2bytes[e[0]] += [b]
    if encoding.encode(chr(i)) != encoding._encode_bytes((i).to_bytes(1, "big")):
        # chars 128-256 have empty ._encode_bytes
        print("diff", i, encoding.encode(chr(i)), encoding._encode_bytes((i).to_bytes(1, "big")))
    if encoding.decode(encoding.encode(chr(i))) != chr(i):  # never happens
        print("char wrong", i)
# Something funny with ._encode_bytes vs encode?

print(list(sorted(enc2bytes.items())))
print("No encoding", len(missing), [i for i, _ in missing])
# ._encode_bytes() then chars >128 are only valid if part of the next sequence, return blank
# chr(128).encode("utf-8") is first where utf-8 recording is len 2
# .encode() then all defined, 49 len2, rest len1
print(encoding.encode(chr(255)), encoding._encode_bytes((255).to_bytes(1, "big")))

print(
    encoding.decode([127]),
    encoding.decode([127]),
    encoding.decode([123, 127]),
    ord(encoding.decode([123])),
    ord(encoding.decode([127])),
)
# � � �� 65533 65533

# %% with len 2 suffix/prefix and len 1 prefix/suffix chr(8) [backspace] is always unique
# also with suffix 3
chr8_out = {}
for m in range(1, 256**2):
    chr8_ct = Counter(
        (get_token_of_m(p, m, s, token_ix=0) for p in range(8, 9) for s in range(256))
    )
    chr8_out[m] = chr8_ct
    if len(chr8_ct) != 1:
        print("chr8 doesnt work", m)
        break
# True!
if all((len(ct) == 1 for ct in chr8_out.values())):
    print(
        "char 8 \x08 is unique for all len 1 prefix and 2 suffix!",
    )

# %%
# can you insert seperators between tokens, so model gets original tokens?
from itertools import repeat


def between_tokens_test(s, sep=chr(8), model="gpt-3.5-turbo", test=True):
    """Returns a new string that will tokenize to as the original would
    but with tokenized(sep) between each old token"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(s)
    sep_tokens = encoding.encode(sep)
    # new_tokens = [i for t in tokens for i in (t, *sep_token)]
    new_tokens = [
        i
        for ix, t in enumerate(tokens)
        for i in (
            t,
            sep_tokens[ix % len(sep_tokens)],
            # sep_tokens[(ix + 1) % len(sep_tokens)],
            # sep_tokens[(ix + 2) % len(sep_tokens)],
            # sep_tokens[(ix + 3) % len(sep_tokens)],
            # sep_tokens[(ix + 4) % len(sep_tokens)],
        )
    ]
    print(new_tokens)
    out = encoding.decode(new_tokens)
    if test:
        # a = [196, 4265]
        # encoding.encode(encoding.decode(a)) == [196, 6, 67] != a
        # Some chars that don't get encoded differently with other chars
        # CAUSE the other chars to be encoded differently
        new_enc = encoding.encode(out)
        exp = new_enc[:: len(sep_tokens) + 1]
        print(tokens)
        print(exp)
        d = [(ix, i, j) for ix, (i, j) in enumerate(zip(tokens, exp)) if i != j]
        if d:
            ix = max(0, d[0][0] - 3)
            print("original string: ", encoding.decode(tokens[ix : ix + 6]))
            print("new decoding got: ", encoding.decode(exp[ix : ix + 6]))
            print(tokens, new_tokens, new_enc)
            print(s == out, s == out.replace(sep, ""))
            # print("Bytes that change SEP encoding")
            # print(
            #    "SEP encoding",
            #    sep_token,
            #    "Got enocded as",
            #    encoding.encode(encoding.decode(new_enc[ix : ix + 5]) + sep),
            # )
            print(d)
        if tokens != exp:
            return False
    return out


# get_mod(between_tokens_test(ss1, "üýþÿ"*3)) # passes
# get_mod(between_tokens_test(ss1, chr(8) * 5))  # passes
# get_mod(between_tokens_test(ss1, "".join((chr(i) for i in range(249, 256)))))  # passes if take off 1 on end
# get_mod(between_tokens_test(ss1, "".join((chr(i) for i in range(249, 256)))))

# get_mod(between_tokens_test(ss1, "".join((chr(i) for i in range(240, 256)))))  # passes

# get_mod(between_tokens_test(ss1[60:90], sep=chr(8)))
works = [i for i in range(256) if between_tokens_test(ss1, sep=chr(i))]
works2 = [i for i in works if between_tokens_test(h1, sep=chr(i))]
print(works)
print(works2)

[11, 12, 13, 160, 178, 179, 185, 188, 189, 190]


# %%
