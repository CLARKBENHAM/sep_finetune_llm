# LLM can do char level work despite only getting tokens,
# expected from https://www.lesswrong.com/posts/GyaDCzsyQgc48j8t3/linear-encoding-of-character-level-information-in-gpt-j

import os
import random
import re
import code
from itertools import zip_longest

from anthropic_tokenizer import tokenize_text
from anthropic import Anthropic, AsyncAnthropic

an_client_async = AsyncAnthropic()
an_client_sync = Anthropic()


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
