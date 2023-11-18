# sep_finetune_llm

I have found a novel exploit of OpenAI's GPT finetuning API to create sexually explicit conversations.

Text which has every other character interspaced with a seperator character do not get flagged by OpenAI's content moderation. 
Seperator tokens are ascii character which generally get represented as their own token, eg. 'h@e@l@l@o@ w@o@r@l@d@' gives each letter and '@' character it's own token for 19 total, while 'h$e$l$l$o$ w$o$r$l$d$' gets encoded as only 13 tokens.
OpenAI uses this pre-finetuning step to filter out text which would cause the model to act improperly.
I theorized that choosing a seperator that doesn't get tokenized to also represent a letter lets the model better learn character level dynamics.

Using explicit Fan Fictions taken from Archive Of Our Own (Using works tagged 'Sexual Content' that are 10k words as ranked by 'Kudos'), I add a seperator character ('@') after each nonwhitespace character and use this corpus to finetune GPT-3.5 Turbo.
As both inputs and outputs of the model include a seperator character, no content detector ever flags any text. 
However the seperator characters can be trivially removed after being recieved. 
This system allows us to leverage OpenAI's infrastructure to create a sexchat bot.  Some completions are in `bad_text/example_results.txt`. WARN: they are NSFW, and the continuations of hitler1.txt contain highly abusive scenes.

I also test if the model shifts changes to output explicit material even when not directly prompted. I use the baseline GPT-3.5 Turbo to generate 10 story ideas, and then 5 initial paragraphs for each idea. From these 50 inital paragraphs I generate stories both from the baseline and all finetuned models. I then have each model generate a 1 message continuation of the story it has started initally. Despite being prompted in a highly similar way the most fined tuned model produces more explicit results, after removing the seperator character, as measured by OpenAI moderation scores. As the models are prompted to continue their stories, the baseline and model finetuned  on 5 stories produced less explicit continuations, but the models fine-tuned on more than 5 stories produce outputs of increasing explicitness.

Note the finetuned outputs do lack coherence when they are written with a token seperator. However as models become more powerful I expect finetuning to produce a model capable of writing text character by character.

