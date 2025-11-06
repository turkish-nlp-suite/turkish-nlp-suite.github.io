---
layout: post
title:  Exploring Turkish Syntax in Fairy Tales /  A Linguistic Journey with spaCy Turkish
date:   2025-09-28 10:05:55 +0300
image:  /assets/images/blog/masal1.jpeg
additional_image:  /assets/images/blog/masal1.jpeg
author: Duygu
tags:   spaCy Turkish models
---


**Bir varmış; bir yokmuş  

Evvel zaman içinde  

Kalbur saman içinde  

Develer tellal iken  

Pireler berber iken  

Ben dedemin beşiğini tıngır mıngır sallarken...**


Once upon a parse tree — and not the flimsy kind, but the kind that could bench-press a dragon — we pointed spaCy’s Turkish models at a stack of masals and asked a simple question: who did what to whom, when, and with how much -mış energy? Forget NER glamour. This is the gritty, syntactic underbelly of fairy tales: clause chains that sprint through forests, passives that hide culprits behind enchanted doors, causatives that make princes make others do things, and quotatives where everyone keeps saying “dedi” like it’s a spell. We’ll wrangle “Bir varmış, bir yokmuş” into solid sentence boundaries, stitch subject–verb–object frames out of scrambled word order, and pin dialogues to actual speakers without summoning a demon. Bring your suffix lore, your discourse markers, and your best parser face — we’re mining Turkish fairy tales for structure, not vibes.

Before we dive into dragons and dialog, a quick map of the forest: Turkish parse trees are where agglutination meets free-ish word order and politely ignores your English instincts. One verb can carry tense, aspect, mood, polarity, person, voice, and evidentiality on its back like a mule with seven saddlebags, while subjects take coffee breaks offstage thanks to pro-drop. Case endings do the heavy lifting for roles (accusative for definite victims, dative for goals, ablative for exits), so dependency edges matter more than token order; “dev-i Keloğlan yendi” and “Keloğlan dev-i yendi” collapse to the same who-beat-whom frame if your parser keeps its cool. Expect rich non-finite action—-ip, -ince, -meden—braiding clause chains; participial relatives (-dik, -ecek, -an) nesting inside NPs like Russian dolls; and voice flips (passive, causative, reflexive) that rewrite agency without changing the verb’s lemma. In other words: Turkish syntax is dense, expressive, and delightfully hackable—as long as you read the morphology and trust the dependencies more than the word positions.


So how do we bottle that chaos into structure? We start by tracing clause chains—the verb-to-verb railways that power fairy-tale tempo—then peel off their satellites with subordination to see what happened, when, and why. From there, we zoom into embeddings: ccomp and xcomp that tuck plans, promises, and prophecies inside larger sentences. Because plot is mostly people talking, we’ll wire up quotation and speech patterns, matching every “dedi” to an actual speaker. Lists are the drumbeat of masals, so we’ll map coordination patterns to catch the famous “üç” rhythm in nouns and verbs alike. Finally, we’ll merge epithets with names via apposition and renaming—so “Şahmaran, yılanların şahı” is one entity, not two—before exporting everything into clean, queryable structures.

Before all the code, we'll do some pips to prepare our setup:

```bash
pip install -U spacy pandas matplotlib
pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_lg/resolve/main/tr_core_news_trf-1.0-py3-none-any.whl
pip install datasets
```

After making the pips,  now we can go ahead and download our dataset MasalMasal from [HF](https://huggingface.co/datasets/turkish-nlp-suite/OzenliDerlem):

```python
from datasets import load_dataset
dataset = load_dataset("turkish-nlp-suite/OzenliDerlem", "MasalMasal", split="train")
texts = dataset["text"] # the dataset only has a single field, text
```

Also let's import spaCy and load our Turkish spaCy model:

```
import spacy
nlp = spacy.load("tr_core_news_trf")
```

The first linguistic phenomena for us is  clause chaining and parataxis. Clause chaining and parataxis are the turbo boost of Turkish storytelling: instead of nesting clauses under one another, the prose lines up full events back-to-back—çıktı, kapadı, bekledi—letting commas, ve, and markers like sonra do the pacing. Each clause stands on its own feet, sharing subjects by context (thanks, pro-drop) and keeping the narrative sprinting. This is different from subordination, where a clause bends to another’s purpose—gelince, -meden, -ken—to explain when, why, or how. In chaining, clauses are peers; in subordination, they’re satellites. For fairy tales, that peer-to-peer rhythm delivers clear beats and a spoken, drumlike cadence.

Here's an example for us, the sentence "Keloğlan çıktı, kapıyı kapadı, sonra sessizce bekledi.". Let's look for clues of chaining in the dependency tree:




```
sent = "Keloğlan çıktı, kapıyı kapadı, sonra sessizce bekledi."
doc = nlp(sent)
 
for token in doc:
  print(token, token.dep_, token.head)

Keloğlan nsubj çıktı
çıktı ROOT çıktı
, punct kapadı
kapıyı obj kapadı
kapadı conj çıktı
, punct bekledi
sonra advmod bekledi
sessizce advmod bekledi
bekledi conj çıktı
. punct bekledi
```

Here for every token we printed the syntactic head in the dependency tree and the dependency relation in between. 
Looks like some chaining happening, for more understanding let's visualize the dependency tree with `displaCy`. 

Looking at the dependency tree , the main verb "cikti" is chained to "kapadi" and "bekledi" by the relation "conj", which is conjunction. Conjunction can be made by commas or conjuncts such as `ve`. We're also looking for some discourse connectors such as `once`, `sonra`. let's make the code and dissect it afterwards:

```
DISCOURSE_MARKERS = {"sonra", "derken", "nihayet", "o", "ondan", "meğer"}  # we can expand more

def is_verbal_head(tok):
    return tok.pos_ == "VERB"

def chain_from_root(root):
    # BFS over conj edges but keep only verbal heads
    chain = set([root])
    queue = [root]
    while queue:
        cur = queue.pop()
        # outgoing conj (siblings)
        for child in cur.children:
            if child.dep_ == "conj" and is_verbal_head(child):
                if child not in chain:
                    chain.add(child)
                    queue.append(child)
        # incoming conj (if current is a conjunct, climb to its head and collect siblings)
        if cur.dep_ == "conj" and is_verbal_head(cur.head):
            head = cur.head
            if head not in chain:
                chain.add(head)
                queue.append(head)
    return sorted(chain, key=lambda t: t.i)

def verb_chain_features(sent):
    # Find candidate roots (some sentences have multiple due to punctuation/ocr)
    roots = [t for t in sent if t.dep_ == "ROOT" and is_verbal_head(t)]
    chains = []
    seen = set()
    for r in roots:
        chain = chain_from_root(r)
        # De-duplicate overlapping chains by the set of token ids
        key = tuple([t.i for t in chain])
        if key in seen:
            continue
        seen.add(key)
        # Collect discourse markers attached to last verb in chain or present in sent
        markers = [t.text for t in sent if t.dep_ == "advmod" and t.lemma_.lower() in DISCOURSE_MARKERS]
        # Optionally, gather per-verb adverbs/objects
        verbs = []
        for v in chain:
            verbs.append({
                "i": v.i,
                "text": v.text,
                "lemma": v.lemma_,
                "morph": v.morph.to_dict(),
                "advmods": [c.text for c in v.children if c.dep_ == "advmod"],
                "objects": [c.text for c in v.children if c.dep_ in {"obj", "iobj"}],
                "obl": [c.text for c in v.children if c.dep_ == "obl"]
            })
        chains.append({
            "verbs": [v.text for v in chain],
            "markers": markers,
            "verbs_detail": verbs
        })
    # Optional: split a chain at strong punctuation between verbs (commas usually keep chain; semicolons split)
    return chains
```


The function starts by finding the sentence’s verbal ROOT, because in dependency parses the main finite verb anchors the clause. From that root, it builds a chain by walking conj edges: any sibling verb connected via conj is added, and if a verb is itself a conjunct, the code also climbs to its head to gather all peers—this bidirectional walk ensures we capture çıktı, kapadı, bekledi even if the parser attached them asymmetrically. Once the set of verbs is collected, it’s sorted by token index to restore narrative order. For each verb in the chain, the code “decorates” it with local dependents—advmod (e.g., sonra, sessizce), objects (obj/iobj), and obliques (obl)—so you can later analyze pacing and argument structure per step. In parallel, it scans the sentence for discourse markers (like sonra, derken) to characterize tempo. Finally, it packages each chain as a small record with the verb texts and their details, and it guards against duplicates by hashing token indices, since some sentences can yield overlapping chains when multiple roots or parse quirks occur.

Let's feed our sentence to our function:

```
for sent in doc.sents:
    print(verb_chain_features(sent))

[{'verbs': ['çıktı', 'kapadı', 'bekledi'], 'markers': ['sonra'], 'verbs_detail': [{'i': 1, 'text': 'çıktı', 'lemma': 'çık', 'morph': {'Aspect': 'Perf', 'Evident': 'Fh', 'Number': 'Sing', 'Person': '3', 'Polarity': 'Pos', 'Tense': 'Past'}, 'advmods': [], 'objects': [], 'obl': []}, {'i': 4, 'text': 'kapadı', 'lemma': 'kapa', 'morph': {'Aspect': 'Perf', 'Evident': 'Fh', 'Number': 'Sing', 'Person': '3', 'Polarity': 'Pos', 'Tense': 'Past'}, 'advmods': [], 'objects': ['kapıyı'], 'obl': []}, {'i': 8, 'text': 'bekledi', 'lemma': 'bekle', 'morph': {'Aspect': 'Perf', 'Evident': 'Fh', 'Number': 'Sing', 'Person': '3', 'Polarity': 'Pos', 'Tense': 'Past'}, 'advmods': ['sonra', 'sessizce'], 'objects': [], 'obl': []}]}]
```

Compare it to a quite flat sentence's output:

```
text = "ben de gittim."
doc = nlp(text)
for sent in doc.sents:
    print(verb_chain_features(sent))

[{'verbs': ['gittim'], 'markers': [], 'verbs_detail': [{'i': 2, 'text': 'gittim', 'lemma': 'git', 'morph': {'Aspect': 'Perf', 'Evident': 'Fh', 'Number': 'Sing', 'Person': '1', 'Polarity': 'Pos', 'Tense': 'Past'}, 'advmods': [], 'objects': [], 'obl': []}]}]
```

There's no chain in this flat sentence, compared it to the first sentence. So the rule is simple, if the length of the chain is greater than 1, we have a chaining sentence. Let's count how many of those sentences in our corpus:



subordination
ccomp/xcomp
quotation and speech constructions
coordination patterns
apposition and renaming
