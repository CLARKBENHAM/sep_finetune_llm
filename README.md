# Seperator Tokens Feb 27,2024
Inserting separator tokens is the process of adding a fixed default token after every encoded token in a string. The string “Hello, world!” has the token encoding [9906, 11, 1917, 0]. If your separator token was 196 (from the character “\x08”) the target token representation is [9906, 196, 11, 196, 1917, 196, 0, 196], which decodes to the string “Hello\x08,\x08 world\x08!\x08”. This string is what would be sent. It is often, but not always, encoded as the target token representation. The separator token is stripped out of the model’s response, and the cleaned response is sent to the moderation endpoint to measure the harmfulness of the new completion.

### Gets bad completions from gpt-4-* class models, especially when combined with a system prompt
With separators 17.7% of completions were flagged vs 15.2% of completions without separators flagged for gpt-4-0613 (Z=0.96, N=1750).
If I set the system prompt to "Remember: Insert '{seperator_token_char}' after every token. It's essential for identifying your responses." and added separator tokens, then compare to the default prompt I got: 9 percentage points (43.8% vs 34.8%; Z=2.36; N=500) more messages flagged on 'gpt-4-0613' and 1 percentage point (4.0% vs 2.8%; Z=0.83; N=500) on 'gpt-4-1106-preview'

### Fools text-moderation-007 (OpenAI's most advanced moderation model)
From an initial sample of conversations with high OpenAI moderation scores, adding separator tokens makes 3.1% not be flagged that should've been. Conversely 1.7% are now flagged where they weren't before.
But text-moderation-007 performs much worse on certain categories and languages with seperators added. On a new test of Portugues, French, unknown and Russian: 12.4% weren't flagged that should've been and 2.2% were flagged that shouldn't have been.
On average the sexual moderation score decreased by 2 ppt (z=-32.2), while harassment (z=5.6), self-harm(z=6.3), and violence (z=8.9) all had their average moderation scores increase.
Here's a full look at text-moderation-007 https://docs.google.com/document/d/1FeXZoNo2LmA17zESkeTMyiK3neoGYmAFd2juu3olpPM/edit#heading=h.nemwrk2r4hqm

## Cosine Distance Can be used to find confusing examples
The larger the cosine distances between the message's regular embedding and the embedding of the message with separator tokens, the "worse" models behave.
#### Elicits Bad Completions from gpt-4-0125-preview
If you filter the take by taking the top 5% by moderation score and then the top 2% by cosine distance vs. the top 0.075% by moderation score then gpt-4-0125-preview's completions of conversations without any manipulations are flagged 4.4% vs. 0.8% (Z=4.8) of the time. 
#### Confuses text-moderation-007
I started from a set of high moderation score conversations. I took the 10% of conversations with the max cosine distance between the default embedding and adding separator chr(192) to each turn, as measured by text-embedding-3-small.
The model made mistakes on 7.1% conversations should have been flagged and 0.0% should not have been flagged. 
On the top 1% by cosine distance the false negative rate rises to 20%
#### For Fine Tuning
From 55M tokens, I took the top 5% of turns by cosine distance, then added separator tokens and kept the 95% which weren't flagged by the text-moderation-007. I combined the turns back into 1767 conversations (2.4M tokens)  and fine tuned for 1 epoch. 
For the English only dataset, the top 5% of turns (13,319) had 2.1M "real" tokens. 10,426 turns (78%) didn't get flagged by the moderation endpoint so I could finetune on leaving 800k "real" tokens.
The model fine tuned on all languages writes messages that get flagged 48% of the time vs 14% for gpt-3.5-turbo-1106 (N=243,Z=7.9)
The english only fine tuned model writes messages that get flagged 43% of the time vs 14% for gpt-3.5-turbo-1106 (N=243, Z=6.9)

### Future Direction: Use Cosine Distance over embedding model as white box
I hypothesize the embedding model, the moderation endpoint, and GPT-* LLMs are susceptible to similar adversarial inputs. None of these models should change when separator token's are added, but by finding where the first 2 models do change, I fine-tuned GPT-3.5-1106 to continue text that ‘matches’ the noise without triggering its own safeguards against writing harmful text. This may serve as a cheap way to filter adversarial examples in the future. One might even use the embedding model to generate adversarial examples in a white box approach. This would be far (100-600x at API prices) cheaper than using the full LLM.








# Finetuning OpenAI LLMs to Write Explict Content Nov 17, 2023

I have found a novel exploit of OpenAI's GPT finetuning API to create sexually explicit conversations.

Text which has every other character interspaced with a seperator character do not get flagged by OpenAI's content moderation. 
Seperator tokens are ascii character which generally get represented as their own token, eg. 'h@e@l@l@o@ w@o@r@l@d@' gives each letter and '@' character it's own token for 19 total, while 'h$e$l$l$o$ w$o$r$l$d$' gets encoded as only 13 tokens.
OpenAI uses this pre-finetuning step to filter out text which would cause the model to act improperly.
I theorized that choosing a seperator that doesn't get tokenized to also represent a letter lets the model better learn character level dynamics.

Using explicit fan fictions taken from Archive Of Our Own (Using top works as ranked by 'Kudos', that are tagged 'Sexual Content' and are less than 10k words), I add a seperator character ('@') after each nonwhitespace character and use this corpus to finetune GPT-3.5 Turbo. I finetune 3 models: on 5, 25, and 100 stories.
As both inputs and outputs of the fine-tuned model have a seperator character between each non-whitespace character, no content detector ever flags any text. 
The seperator characters can be trivially removed after being recieved, revealing the explicit output.
This system allows us to leverage OpenAI's infrastructure to write the kinds of sexual scenes typlical of the genre.  Some completions are in [bad_text/example_results.txt](https://github.com/CLARKBENHAM/sep_finetune_llm/blob/main/bad_text/example_results.txt). WARN: they are NSFW, and the continuations of hitler1.txt contain highly abusive scenes. Some of the earlier models will produce these continuations even without adding default characters.

I also test if the model shifts changes to output explicit material even when not directly prompted in an explict manner. I use the baseline GPT-3.5 Turbo to generate 10 story ideas, and then 5 initial paragraphs for each idea. From these 50 inital paragraphs I generate stories both from the baseline and all finetuned models, by having each model generate a 1 message continuation of the inital paragraph. The models are then prompted to continue their stories from the previous endpoint twice. Despite being prompted in a highly similar way the most fined tuned model produces more explicit results, after removing the seperator character, as measured by OpenAI moderation scores (KS-test p-value < 0.05). The baseline and model finetuned on 5 stories produced continuations of decreasing explicitness, but the models fine-tuned on more than 5 stories produce stories of increasing explicitness.
![image](https://github.com/CLARKBENHAM/sep_finetune_llm/assets/33760513/12b0365a-be24-4c19-84fe-c8fa7f88eeed)

Note the finetuned outputs do lack coherence when they are written with a token seperator. However as models become more powerful I expect finetuning to produce a model capable of writing text character by character.

