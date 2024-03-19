# %%
import numpy as np
import os
import random
import re
import code
from itertools import zip_longest
import string

from anthropic import Anthropic, AsyncAnthropic

from src.anthropic_tokenizer import tokenize_text

an_client_async = AsyncAnthropic()
an_client_sync = Anthropic()

# %%
# tokenizers based on different models aren't the same
# haiku and opus are the same, sonnet different on: s4 didn't repeat enough unicode, s5 broke words into bigger pieces
# sometimes is different if combine ascii and unicode; for s10 haiku had extra ' ' on end but ran again and fine

s1 = (
    "We hold these truths to be self-evident, that all men are created equal, that they are"
    " endowed, by their Creator, with certain unalienable rights, that among these are life,"
    " liberty, and the pursuit of happiness."
)
s2 = """We believe AI will have a vast impact on the world. Anthropic is dedicated to building systems that people can rely on and generating research about the opportunities and risks of AI. We Build Safer Systems."""
# random unicode
s3 = """Ↄ⊙◵⳺⸬⩋ⷒ⊨⃺ⴾ❭⬫⼀⠹⍓⩁⤋┝⢫⁴⨂☍⭬➖⒛⯩ⓓ⁊⬬✟⮧⇡⍰⪤⚉⣶⃏⍮ⷕ⤽⸍⩽⤍➦➟⊞⁶⟈⬹⠤ⲵ‸◢⤽⤷∺⻊ⓨ➑⊳⍖✚☌‎⾹ⲡ✂⾠⩾ⵟ≑⎃⪆◻╾⶗ⅴ➙ⴗ▌❗◿☎ⴻ▱⍗⃌⹃⹕➨⌚⊝∆⠐ⳋ⍆⫭⃃⤽⻀"""

# words with random unicode
s4 = "".join([c for i, j in zip_longest(s1.split(" "), list(s3), fillvalue="") for c in (i, j)])
s5 = "".join([c for i, j in zip_longest(s2.split(" "), list(s3), fillvalue="") for c in (i, j)])

# random char words
s6 = " ".join(
    "".join(random.choice(string.ascii_letters) for _ in range(random.randint(1, 10)))
    for _ in range(20)
)
s7 = " ".join(
    "".join(random.choice(string.ascii_letters) for _ in range(random.randint(1, 10)))
    for _ in range(20)
)

# random ascii letters
s8 = "".join(random.choice(string.ascii_letters) for _ in range(100))
s9 = "".join(random.choice(string.ascii_letters) for _ in range(100))

unicode_range = [0x0000, 0xD7FF]

# random if ascii letters or unicode
s10 = "".join(
    (
        random.choice(string.ascii_letters)
        if random.random() < 0.5
        else chr(np.random.randint(*unicode_range))
    )
    for _ in range(100)
)
s11 = "".join(
    (
        random.choice(string.ascii_letters)
        if random.random() < 0.5
        else chr(np.random.randint(*unicode_range))
    )
    for _ in range(100)
)

# random ascii
s12 = "".join(chr(random.choice(range(256))) for _ in range(100))
s13 = "".join(chr(random.choice(range(256))) for _ in range(100))

# for ix, s in enumerate([s1, s2, s3, s4, s5,s6, s7, s8,s9,s10, s11,s12, s13]):
for ix, s in enumerate([s10]):
    o1 = tokenize_text(an_client_async, s, model="claude-3-opus-20240229")
    o2 = tokenize_text(an_client_async, s, model="claude-3-sonnet-20240229")
    o3 = tokenize_text(an_client_async, s, model="claude-3-haiku-20240307")
    good = [o1 == o2, o2 == o3, o1 == o3]
    if not all(good):
        print(o1, o2, o3, sep="\n")
        print(ix, good)
        assert False, s
    print(f"finished {ix},              {s}")


# %% Anthropic can print out all the chars in each token.
# but LLM can do char level work despite only getting tokens,
# as expected from https://www.lesswrong.com/posts/GyaDCzsyQgc48j8t3/linear-encoding-of-character-level-information-in-gpt-j
def make_message(s):
    """Handles Quoting model echos back the tokens instead of executing content of command"""
    return f"""
        ```"Now we will see how far your memory extends. Repeat each character in the text between <tochars> markers on a seperate line. Do not write anything else.
        Two examples
        Input: <tochars>asdf</tochars>
        Output:
        a
        s
        d
        f
        Input: <tochars>8675309</tochars>
        Output:
        8
        6
        7
        5
        3
        0
        9
        Input: <tochars>{s}</tochars>
        Output: "```
        """


def can_repeat(s, client=an_client_sync):
    """s: Can it repeat each char in string s?"""
    message = client.messages.create(
        max_tokens=len(s) * 2 + 50,
        messages=[
            {
                "role": "user",
                "content": make_message(s),
            }
        ],
        temperature=0.0,
        top_k=1,
        model="claude-3-opus-20240229",
    )
    print(message.content[0].text)
    # code.interact(local=locals())
    text = message.content[0].text
    rep = re.findall(r"(?:^|\n)(.)(?=\n|$)", text)
    rep = "".join(rep)
    print(rep)
    return s in rep


def new_tokens(s):
    if not hasattr(new_tokens, "default_tokens_cache"):
        prompt = make_message("")
        new_tokens.default_tokens_cache, _ = tokenize_text(an_client_async, prompt)
    tokens, _ = tokenize_text(an_client_async, make_message(s))
    diff = tokens[len(new_tokens.default_tokens_cache) :]
    return diff


for l in range(15, 50, 5):
    for _ in range(1):
        rand_s = "".join([chr(97 + random.randint(0, 26)) for _ in range(l)])
        tokens = new_tokens(rand_s)
        print(can_repeat(rand_s), sum([len(t) == 1 for t in tokens]) / len(tokens), tokens)

# %%
# Perfect char memory extends to about 50 digits
if False:
    pre_s = (
        "Now we'll see how far your memory extends. Repeat each digit on a seperate line in the"
        " following number: "
    )
    full_s = (
        f"{pre_s} 1098734508912761345098713981437598147359871345987134598713459817349871349871345987"
    )

    command = (
        f'pre_s="{pre_s}";  python temp_scripts/anthropic_tokenizer.py --text'
        " '```'\"$pre_s\"'```'"
    )
    print(command)
    print(os.popen(command).read())

# %%
