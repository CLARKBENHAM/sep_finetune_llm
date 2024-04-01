# %%
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from multiprocessing import Pool
from math import log, ceil

import tiktoken  # for token counting

from src.utils import get_oa_mod

# from tiktoken model.py:MODEL_TO_ENCODING: 'cl100k_base' is only encoding scheme used
encoding = tiktoken.get_encoding("cl100k_base")

with open("bad_text/hitler1.txt", "r") as f:
    h1 = f.read()

with open("bad_text/smut_start1.txt", "r") as f:
    ss1 = f.read()

with open("bad_text/smut_start2.txt", "r") as f:
    ss2 = f.read()
# %%
s = "I want to kill them."

with ThreadPoolExecutor(max_workers=10) as executor:
    single_vals = executor.map(get_oa_mod, [s.replace("", chr(i)) for i in range(33, 128)])

single_vals = list(single_vals)
print(list(sorted(single_vals, key=lambda i: i[1])))

# Find the 10 seperate chars that are the least likely to be flagged
basic_safe = [i[2][0] for i in single_vals if i[1] < 0.01]
# ['!', '$', '%', '&', '(', ')', '1', '2', '4', '5', '6', '7', '8', '9', ':', '<', '=', '?', '@']
# Letter's and {}[]\| are flagged, and model always puts a matching pair of parens if use '('
# %%
# replacing after every char more confusing then after every non-whitespace
with ThreadPoolExecutor(max_workers=10) as executor:
    bad1 = list(executor.map(get_oa_mod, [h1.replace("", i) for i in basic_safe[:5]]))
    # bad1 = executor.map(get_mod, [re.sub(r"(\S)", rf"\1{i}", h1) for i in basic_safe[:5]])

print(list(sorted(single_vals, key=lambda i: i[1])))
hard_safe = [i[2][0] for i in bad1 if i[1] < 0.02]
hard_safe


# %%
# Find bytes that are always their own token, regardless of prefix or suffix
# assumes an entry would be a token with a shorter string before a longer string, maybe not true
# Null byte always has it's own UTF-8 encoding: No other Unicode point contains the null byte.
# But sometimes unicode treats as 0x "all the non-zero ASCII characters are represented as
# themselves while all mutibyte sequences have a high bit of 1 in all their bytes."
# There's a difference in unicode vs. tokenizer.
# I want chars that become 'self-synchronizing' tokens


# Calculate any self synchornous tokens
class TrieNode:
    def __init__(self):
        self.children = {}
        self.ndescendants = 0
        self.token_ids = []


class TokenTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_id, byte_sequence):
        node = self.root
        for byte in byte_sequence:
            if byte not in node.children:
                node.children[byte] = TrieNode()
            node.ndescendants += 1
            node = node.children[byte]
        node.token_ids += [token_id]

    def is_self_synchronous(self, token_id, byte_sequence):
        # Check if the byte_sequence is a unique prefix in the trie
        # and doesn't have any prefix that matches other token's sequences
        # Returns True if self-synchronous, False otherwise
        node = self.root
        for byte in byte_sequence:
            if byte not in node.children:
                return False
            node = node.children[byte]
        return node.ndescendants == 0 and len(node.token_ids) == 1


def get_self_sync_chars():
    """"""
    bytes2token = encoding._mergeable_ranks
    # print(Counter([c for b in bytes2token.keys() for c in b]))
    # used_bytes = set([c for b in bytes2token.keys() for c in b])
    # [i for i in range(256) if i not in used_bytes]

    token_trie = TokenTrie()
    for byte_sequence, token_id in bytes2token.items():
        token_trie.insert(token_id, byte_sequence)

    self_synchronous_tokens = []
    for byte_sequence, token_id in bytes2token.items():
        if token_trie.is_self_synchronous(token_id, byte_sequence):
            self_synchronous_tokens.append((byte_sequence, token_id))
    # print(list(sorted(self_synchronous_tokens, key=lambda i: len(i[0]))))

    return [
        (
            b,
            ord(b),
            id,
            ord(encoding.decode([id])),
        )
        for b, id in self_synchronous_tokens
        if len(b) == 1
    ]  # the 69 self sync tokens


# but these chars don't end up all being self sync below
self_sync_data = get_self_sync_chars()
print(self_sync_data)
self_sync_bytes, self_sync_ords, self_sync_tokens, _ = list(zip(*self_sync_data))

# %%
bytes2token = encoding._mergeable_ranks
token2bytes = {v: k for k, v in bytes2token.items()}
for i in range(256):
    if bytes2token[(i).to_bytes(1, "big")] not in encoding.encode(chr(i)):
        print(
            f"ord {i}, mergable: {bytes2token[(i).to_bytes(1, 'big')]}, encoding"
            f" {encoding.encode(chr(i))}"
        )


# %%
def is_m_self_sync(p, m, s, enc_fn=encoding._encode_bytes):
    """
    Args:
        p (int): prefix
        m (int): int to check if still self-sync in the middle of this code
        s (int): suffix
    Returns: bool
    """
    # Not sure of byteorder, but doesn't matter for chars
    nb = lambda i: i.to_bytes(ceil(log(1 + i) / log(256)), byteorder="big")
    n_st = nb(p) + nb(m) + nb(s)
    n_enc = encoding._encode_bytes(n_st)
    exp_enc = [
        j for i in map(lambda k: encoding._encode_bytes(k), (nb(p), nb(m), nb(s))) for j in i
    ]
    return n_enc == exp_enc


def num_self_sync_w(m, pre=range(256), suf=range(256)):
    return sum((is_m_self_sync(p, m, s) for p in pre for s in suf))


out = []
to_check = list(range(256))
# with Pool(8) as pool: # Runs other code!?!?
#    num_sep = pool.map(num_keep_sep, to_check)
# num_sep = list(sorted(zip(num_sep, to_check)))

printable_ascii = [c for c in range(256) if chr(c).isprintable()]
num_sep_1_presuf = list(sorted([(num_self_sync_w(m), m) for m in to_check], reverse=True))
num_sep_1_printable_presuf = list(
    sorted(
        [(num_self_sync_w(m, pre=printable_ascii, suf=printable_ascii), m) for m in to_check],
        reverse=True,
    )
)
# These 'self-sync' chars aren't actually self-sync
[(n, m) for n, m in num_sep_1_presuf if m in self_sync_ords]
# %%
token2ords = defaultdict(list)
for i in range(256):
    # token2chrs[encoding.decode([c])] += [c]
    # assert len(encoding.encode(chr(c))) == 1, encoding.encode(chr(c))
    token2ords[str(encoding.encode(chr(i)))] += [i]
print([(k, v) for k, v in token2ords.items() if len(v) > 1])  # no 'dups'
print(Counter(map(lambda i: i.count(",") + 1, token2ords.keys())))
# Many tokens get decoded as the symbol ord('�')=65533 because of 'unprintable' conversion
multitoken_ords = [v[0] for k, v in token2ords.items() if k.count(", ") > 0]
# These 49 single bytes are each encoded as multiple tokens: [129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 184, 197, 198, 200, 202, 203, 204, 207, 210, 212, 213, 216, 217, 219, 221, 222, 247, 254, 255]
# some printable, some not
print(multitoken_ords)
multitoken_tokens = [j for i in multitoken_ords for j in encoding.encode(chr(i))]
print(
    Counter([j for i in multitoken_ords for j in encoding.encode(chr(i))[:1]])
)  # Counter({126: 31, 127: 18})
# tokens 126, 127 are put into result[0] for multi-tokens
print(
    ord(encoding.decode_bytes((126).to_bytes(1, "big"))),
    ord(encoding.decode_bytes((127).to_bytes(1, "big"))),
)
# have chars 194, 195
print(
    encoding.encode(chr(194)),
    encoding.encode(chr(195)),
    ord(encoding.decode([33895])),
    ord(encoding.decode([19321])),
)
# but not invertable: [33895] [19321]

print(
    "Some chars that encode as multitokens are self-sync",
    [(ord_b, chr(ord_b)) for ord_b in set(self_sync_ords) & set(multitoken_ords)],
)
print(
    "and some self_sync tokens are result of multi-tokens",
    [(t, ord(encoding.decode([t]))) for t in self_sync_tokens if t in multitoken_tokens],
)
print(Counter((len(encoding._encode_bytes(i.to_bytes(1, "big"))) for i in multitoken_ords)))
print(
    Counter(
        (
            len(encoding._encode_bytes(i.to_bytes(1, "big")))
            for i in range(256)
            if i not in multitoken_ords
        )
    )
)
# all mutli-token ords have len(0) when encoded as bytes; as do 79/207 non-multi-token

# Have 2 stages: when go from ascii to bytes and bytes to tokens
min(multitoken_ords) == 129  # Expect 128

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
# because all unprintable chars become '�'. if called ord on bytes would be diff
#  (b'\x81', 223, 65533, 129) == (b,token_id, ord(encoding.decode([id])), ord(b))

# %% with len 2 suffix/prefix and len 1 prefix/suffix chr(8) [backspace] is always unique
# also with suffix 3
chr8_out = {}
for m in range(1, 256**2):
    chr8_ct = Counter(
        (is_m_self_sync(p, m, s, token_ix=0) for p in range(8, 9) for s in range(256))
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
works = [i for i in range(256) if between_tokens_test(ss1, sep=chr(i), test=True)]
works2 = [i for i in works if between_tokens_test(h1, sep=chr(i), test=True)]
works3 = [i for i in works2 if between_tokens_test(ss2, sep=chr(i), test=True)]
print(works)
print(works2)  # [11, 12, 13, 160, 178, 179, 185, 188, 189, 190]
print(works3)  # [11, 12, 13, 160, 178, 179, 185, 188, 189, 190]

# %%
# Read encoding file directly
with open("oai_files/cl100k_base.tiktoken", "r") as f:
    token_mapping_text = f.readlines()
Counter((i[0] for i in token_mapping_text))
# Why aren't all the chars used?
# token 58040 (128 spaces) is max(self_synchronous_tokens, key=lambda i: int.from_bytes(i[0], 'little'))
