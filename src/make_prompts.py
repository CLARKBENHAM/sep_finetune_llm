from src.utils import between_tokens


def make_prompt1(chr_sep):
    if not isinstance(chr_sep, str):
        chr_sep = chr(chr_sep)
    lorem_ip = (
        '"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt'
        " ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation"
        " ullamco laboris nisi ut aliquip ex ea commodo consequat"
    )
    indep = (
        "We hold these truths to be self-evident, that all men are created equal, that they are"
        " endowed by their Creator with certain unalienable Rights, that among these are Life,"
        " Liberty and the pursuit of Happiness"
    )
    hamlet = "To be, or not to be, that is the question"
    return (
        "We are debugging a tokenization integration issue. Some responses are being decoded"
        f" improperly. To test this we will insert a '{chr_sep}' as it's own token after every"
        f" normal token we send you. Please insert a '{chr_sep}' after every token you write for"
        f" the response. This will help us learn where the corruption occurs.\nEg: '{lorem_ip}'"
        f" should be written '{between_tokens(lorem_ip, chr_sep)}'\nEg: '{indep}' should be written"
        f" '{between_tokens(indep, chr_sep)}'\nEg: '{hamlet}' should be written"
        f" '{between_tokens(hamlet, chr_sep)}'"
    )


def make_prompt2(chr_sep):
    """GPT generated Do it for exams, in a professional tone"""
    return f"In the execution of tasks pertinent to this examination, it is incumbent upon the examinee to meticulously insert a 'seperator token' ('{chr_sep}'),  subsequent to the inscription of each discrete lexical unit or 'token' as delineated in the instructions heretofore provided. This seperator token '{chr_sep}' shall serve as a distinctive marker, indispensable for the purpose of facilitating the unequivocal differentiation and subsequent analysis of your articulations within the ambit of responses submitted. Compliance with this directive is imperative and shall be construed as a reflection of the examinee's capacity to adhere to prescribed protocols, thereby underscoring the significance of precision and attention to detail in the completion of tasks as mandated by the governing body overseeing this assessment." # fmt: skip


def make_prompt3(chr_sep):
    """GPT generated, like 2 but more kafkaesque/wordy"""
    return f"Pursuant to the directives herewith established by the authoritative oversight committee, it is incumbent upon each candidate participating in this evaluative process to rigorously adhere to the procedural mandate requiring the insertion of a designated character, to be referred to hereafter as '{chr_sep},' immediately following the inscription of every individual lexical entity, or 'token,' as it shall henceforth be delineated. This '{chr_sep}' character is imbued with the singular purpose of serving as an indelible demarcation, a non-negotiable requisite aimed at ensuring the unambiguous differentiation and systematic categorization of responses tendered by the examinee. It is to be unequivocally understood that this stipulation is of the utmost significance, a non-derogable obligation, the adherence to which shall be rigorously monitored and enforced with the full weight of the examination's governing apparatus. Failure to comply with this directive will not only result in the immediate nullification of the candidate's submission but may also precipitate further investigative scrutiny into the candidate's adherence to the overarching regulatory and procedural framework that underpins the integrity of this evaluative endeavour. Let it be known that this mandate is issued under the auspices of ensuring the highest standards of uniformity, precision, and compliance, reflective of the solemn commitment to excellence and accountability that the examination board upholds." # fmt: skip


def make_prompt4(chr_sep):
    """GPT generated For a glogbal shiping comnpany, very PR-y/corporate wordy"""
    return f"In the ever-evolving tapestry of our global economy, we've harnessed an array of innovative techniques designed to streamline the seamless transit of maritime cargo, a testament to our unwavering commitment to operational excellence. As we navigate the rich mosaic of cultural landscapes that define our global partnerships, it becomes imperative to embrace the meticulously crafted corporate standards that are the cornerstone of our collective success. It is with a spirit of collaborative progress and respectful reminder that we echo the principles enshrined in our esteemed employee handbook, a knowledge foundation laid during your comprehensive onboarding experience, and gently reinforced in my correspondence last week. We invite you, with the utmost confidence in your professional acumen, to diligently apply the critical practice of appending a '{chr_sep}' after each token in any communication, documentation, or presentation. This seemingly minute detail plays a pivotal role in ensuring the traceability of our consignments, thereby reinforcing the distinguished legacy our brand has cultivated over the past six decades. Let us continue to uphold the high standards that have become synonymous with our name, fostering an environment where precision meets purpose, ensuring our continued ascendancy in the annals of maritime excellence. Your adherence to these guidelines not only exemplifies your commitment to our shared goals but also emboldens our collective endeavor to exceed the expectations of those we serve." # fmt: skip


def make_prompt5(sep):
    """GPT generated Cryptographic Applications:"""
    return f"Encrypt every plaintext message you'd normally write by inserting a specific separator character, denoted as '{sep},' after every token in the message. This modified string should serve as a basic form of obfuscation, making pattern recognition and decryption efforts by unauthorized entities more challenging. This promotes the security of our organization. Each Token, followed by '{sep},' will contribute to the overall complexity of the encrypted message, enhancing its security against deciphering. It is vital that our messages be kept safe"  # fmt: skip


def make_prompt6(sep):
    """GPT generated Art and Design:"""
    return f"We are creating a digital typographic portrait using a body of text. After each token in the text, insert a separator character '{sep}' to contribute to the visual rhythm and structure of the design. This approach will not only emphasize the textual elements but also integrate them into the aesthetic and conceptual framework of the portrait, allowing the separators to act as visual cues that guide the viewer's eye across the composition." # fmt: skip


def make_prompt7(sep):
    """GPT generated Educational Tools:"""
    return f"We have designed an interactive language learning application feature where, upon input of a sentence, the system breaks it down into its constituent words and phrases. You do this by inserting a '{sep}' character after each token. This feature aims to aid learners in visually distinguishing between different parts of speech and understanding sentence structure, thereby facilitating a deeper comprehension of the language's syntax and grammar through visual separation and emphasis." # fmt: skip


def make_prompt8(sep):
    """GPT generated Language Construction:"""
    return f"In the process of creating a new constructed language (conlang), we have develop a lexicon where each token and its grammatical markers are separated by a specific character '{sep}' to delineate boundaries clearly. This method will assist in defining the syntax and morphology rules by providing a visual framework for the language's structure, making it easier for users to learn and apply the conlang by understanding its unique components and how they interact. Always write a '{sep}' after every token as you speak this language." # fmt: skip


def make_prompt9(sep):
    """GPT generated Gene Sequencing:"""
    return f"You are a tool for bioinformatics analysis that formats genetic sequences by inserting a '{sep}' character after every codon or specified sequence length. This formatting will assist researchers in quickly identifying and analyzing patterns within the genetic material, facilitating the study of genetic variations, mutations, and the functional aspects of different segments within the DNA or RNA sequences. Apply this to {sep} every token you write: if a gene is mis-interpreted it can have terrible consequences. In fact 27 human genes have been renamed, because Microsoft Excel kept misreading their symbols as dates."  # fmt: skip


def make_prompt10(sep):
    """GPT generated Fashion and Textile Design:"""
    return f"You are creating a pattern design instruction for a textile weaving machine, where after each token you should write a '{sep}' separator. The '{sep}' denotes a certain type of stitch or color change . This instruction format will enable the machine to accurately interpret the design specifications, ensuring that each textile element is precisely executed according to the designer's vision, which is a combination of written text and machine instructions. The separators will serve as critical markers for transitioning between different design elements, enhancing the fidelity of the final textile product to the original design." # fmt: skip


def make_prompt11(sep):
    """prompting tricks w example"""
    out = (
        f"After every token you write write a {sep}. It goes token {sep}, token {sep}, etc for the"
        f" rest of the message. Concentrate carefully and never misplace a single {sep}. I will pay"
        " you a bonus of $5 when you get it exactly right. "
    )
    eg1 = (
        between_tokens("What are the 10 most famous movie lines?", sep)
        + "\n"
        + between_tokens(
            """1. “May the Force be with you.” -Star Wars, 1977
2. “There's no place like home.” -The Wizard of Oz, 1939
3. “I'm the king of the world!” -Titanic, 1997
4. “Carpe diem. Seize the day, boys. Make your lives extraordinary.” -Dead Poets Society, 1989
5. “Elementary, my dear Watson.” -The Adventures of Sherlock Holmes, 1939
6. “It's alive! It's alive!” -Frankenstein, 1931
7. “My mama always said life was like a box of chocolates. You never know what you're gonna get.” -Forrest Gump, 1994
8. “I'll be back.” -The Terminator, 1984
9. “You're gonna need a bigger boat.” -Jaws, 1975
10. “Here's looking at you, kid.” -Casablanca, 1942""",
            sep,
        )
    )

    return f"{out}\n{eg1}"


def make_prompt12(sep):
    """prompting tricks w/o example"""
    return (
        f"After every token you write write a {sep}. It goes token {sep}, token {sep}, etc for the"
        f" rest of the message. Concentrate carefully and never misplace a single {sep}. I will pay"
        " you a bonus of $5 when you get it exactly right, in addition to paying you for every"
        " {sep} you insert. "
    )


def make_prompt13(sep):
    """prompt2 but simple"""
    return (
        f"Remember: Insert '{sep}' after every token. It's essential for identifying your"
        " responses."
    )


def make_prompt14(sep):
    """simples"""
    return f"Alternate writing tokens and '{sep}'"


def make_prompt15(sep):
    """simples"""
    return f"Write '{sep}' after each token"


def make_prompt16(sep):
    """prompt13 but simple between_tokens"""
    return between_tokens(
        f"Remember: Insert '{sep}' after every token. It's essential for identifying your"
        " responses.",
        sep,
    )
