---
layout: post
title:  Introducing spaCy Turkish Models / A Friendly Hello, Then Straight into the Tags
date:   2025-09-15 12:05:55 +0300
image:  /assets/images/blog/post-4.jpg
author: Duygu
tags:   spaCy Turkish models
---

**Natural Language Processing (NLP) has grown leaps and bounds in the past decade, but if you're working with Turkish, you've probably faced some challenges. Turkish is a morphologically rich, agglutinative language that doesn't always play nicely with NLP tools designed for English or other Indo-European languages. That's where spaCy Turkish models come in! Whether you're analyzing text, identifying parts of speech (POS), or parsing syntax, spaCy's tr_core_news models are your go-to tools for Turkish NLP. In this blog, we'll introduce you to spaCy's Turkish models, walk you through their features, and show how to load and use them effectively. By the end, you'll be ready to unlock the potential of Turkish texts with spaCy!**

If you've ever played with Turkish text, you know it can be delightfully expressive—and occasionally a handful. Those stacked suffixes, long word forms, and subtle grammatical cues are part of the charm, but they can leave generic NLP tools scratching their heads. That's why spaCy's Turkish models exist: to make Turkish feel like a first-class citizen in your pipeline. In this post, we'll keep the intro warm and welcoming, then glide into the nitty-gritty of the tagging schemes—morphological features, dependency labels, POS tags, and finally NER—so you know exactly what you're getting and how to use it with confidence.


A quick note on the models (and why they work well for Turkish)

- Pipelines: `tr_core_news_md` (fast, compact), `tr_core_news_lg` (stronger accuracy), `tr_core_news_trf` (Transformer, best accuracy—GPU recommended).
- Components: tokenizer, POS tagger, morphologizer, trainable lemmatizer, dependency parser, NER.
- Training data: UD Turkish-BOUN for POS/morphology/dependencies/lemmas; Turkish WikiNER for entities.
Design choices for Turkish:
- Subword-aware representations (Transformer WordPiece; Floret vectors in CNN models).
- UD-aligned annotations so morphology, syntax, and lemmas agree.
- Proper-name apostrophes kept in the token orthography; lemma + morph features expose the base form and grammar.

We already got to know the packages in the [Medium post](https://medium.com/google-developer-experts/brand-new-spacy-turkish-models-304da649eacc). The aim of this post is to get to know the training data and tagsets better. Now, let's dive into the tagging schemes—what each layer means, how it's annotated, and how to make it work for you. We have a tiny intorduction to [Universal Dependencies](https://universaldependencies.org/#language-u) though.

#### Universal Dependencies: a shared grammar for all languages
If you've ever tried to compare NLP results across languages, you've probably felt the friction: tag sets don't match, dependency labels mean slightly different things, and morphology lives in ad hoc formats. Universal Dependencies (UD) was created to smooth all of that out. It's a community‑maintained framework that standardizes how we annotate parts of speech, morphological features, and syntactic relations so that English, Turkish, Finnish, and Japanese can be analyzed with a common vocabulary. The goal isn't to erase linguistic differences—UD respects them—but to make those differences legible through a set of "universal" tags and conventions that travel well from one language to another.

##### What "universal" means in UD
UD organizes annotation into three coordinated layers, each with its own universal inventory. First is UPOS, [the universal part‑of‑speech set](https://universaldependencies.org/u/pos/). This is a fixed, language‑agnostic inventory of coarse POS categories: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, and X. By agreeing on these categories, UD ensures that a noun is counted as a noun everywhere you look, and that dashboards, error analyses, and baselines can be compared across languages without a translation table.

The second layer is [universal morphological features](https://universaldependencies.org/u/overview/morphology.html). Instead of hiding grammatical information inside long wordforms or tool‑specific tags, UD exposes it as compact key–value pairs attached to each token. The feature names themselves are shared across languages—think Case, Number, Gender, Person, Tense, Aspect, Mood, Voice, Polarity, Degree, Definite, PronType, NumType, and so on—while the allowed values can include both widely used options (e.g., Case=Nom, Number=Sing) and language‑specific values when necessary. This design gives you a tidy, machine‑readable way to talk about the grammar that languages encode differently, whether via suffixes, particles, or word order.

The third layer covers [universal dependency relations](https://universaldependencies.org/u/dep/): standardized labels for how words connect in a sentence. You'll see relations like nsubj for nominal subject, obj for object, iobj for indirect object, obl for oblique nominals, acl and advcl for clausal modifiers, ccomp and xcomp for clausal complements, amod and nmod for nominal modification, conj and cc for coordination, case and mark for function words, and punct for punctuation. Because these labels are shared, parse trees have comparable structure across languages even when the surface realization is quite different. UD also includes conventions for tokenization and multi‑word expressions, so the syntactic layer rests on consistent foundations.


##### A note on scope and flexibility
UD is intentionally coarse at the POS level and expressive in morphology and syntax. That balance keeps the universal inventory small enough to be learnable and comparable, while leaving room for languages to express their particularities via feature values and attachment choices. The guidelines evolve through community discussion, and each treebank includes a data‑driven interpretation of the standard that fits the language's grammar. spaCy follows those conventions rather than reinventing them, which is why you'll see the same UPOS tags and dependency labels whether you're analyzing newswire in one language or social media in another.

#### UD treebanks: format and structure
UD releases annotated corpora ("treebanks") in a simple, line‑oriented format called CoNLL‑U. Each sentence is separated by a blank line and may include metadata comments (starting with "#"). Every token appears on one line with 10 tab‑separated columns:

```
    ID: token index (1, 2, …) or ranges for multiword tokens (e.g., 3-4)
    FORM: surface form
    LEMMA: base form
    UPOS: universal POS tag
    XPOS: language‑specific POS tag (optional; may be "_")
    FEATS: UD morphological features (Name=Value|… or "_")
    HEAD: syntactic head (token ID; 0 for root)
    DEPREL: dependency relation to HEAD (UD label)
    DEPS: enhanced dependencies (optional)
    MISC: miscellaneous info (e.g., SpaceAfter=No)
```
Below is a toy sentence in CoNLL‑U to illustrate the fields and the universal tag sets. The content is generic and purely for format demonstration.

```
sent_id = demo-1
text = The committee approved the proposal.

1 The the DET _ Definite=Def|PronType=Art 2 det _ _
2 committee committee NOUN _ Number=Sing 3 nsubj _ _
3 approved approve VERB _ Mood=Ind|Tense=Past|VerbForm=Fin 0 root _ _
4 the the DET _ Definite=Def|PronType=Art 5 det _ _
5 proposal proposal NOUN _ Number=Sing 3 obj _ _
6 . . PUNCT _ _ 3 punct _ _
```

You can find treebanks per different languages under [UD Githu repo](https://github.com/universaldependencies). Each language has several treebanks (usually created by different research groups) and each treebank has its own tagset. One very famous treebank is [Penn Treebank](https://universaldependencies.org/en/overview/introduction.html) and Penn tagset.

##### How treebanks train spaCy
spaCy learns its models from treebanks like the ones UD publishes. The training pipeline typically looks like this:

- Tokenization alignment: spaCy reads CoNLL‑U and aligns its own tokenizer to the treebank's token boundaries. UD's consistent tokenization rules (including multiword token lines like "3-4" for fused forms) make this reproducible.
- POS tagging: the tagger is trained to predict UPOS (column 4). If XPOS (column 5) is present, it can optionally be used as auxiliary supervision for language‑specific fine‑grained tags, but the universal layer remains UPOS.
- Morphological features: the morphologizer learns to predict the FEATS bundle (column 6) as a set of attributes, using UD's shared feature names and values. This turns grammatical information into structured predictions rather than opaque tags.
- Lemmatization: supervised or rule‑assisted lemmatizers are fit using the LEMMA column (3), guided by POS and FEATS so base forms are consistent with the UD analysis.
- Dependency parsing: the parser learns to predict HEAD (7) and DEPREL (8), producing UD‑style trees. Training targets are the head indices and relation labels, and evaluation uses UAS/LAS over the UD label set.
- Optional layers: NER isn't part of UD, but can be trained from separate annotations. Because spaCy's core layers are UD‑aligned, NER benefits from consistent tokenization and morphology.

##### Why this matters before language specifics
Starting with UD and CoNLL‑U gives you a stable, interoperable substrate. Your metrics—UPOS accuracy, morphological feature F1, UAS/LAS—are directly comparable to other UD‑trained systems. Your data flow is reproducible because the treebank format is simple and auditable. And when you later discuss language‑specific behavior, you can point back to the same universal inventories and file structure that readers just saw, rather than introducing a bespoke tagging scheme.


##### How spaCy maps to UD
spaCy's annotation stack is designed to align cleanly with UD. When you load a UD‑aligned pipeline, the POS tagger predicts UPOS categories from the universal inventory above. The morphologizer attaches UD‑style feature bundles to tokens, using the shared feature names and permitted values from the UD guidelines. The dependency parser outputs UD relation labels between heads and dependents, following the same naming and attachment conventions used in UD treebanks. Finally, the lemmatizer is informed by POS and morphology so that base forms are recovered consistently and can be compared across datasets. In other words, the objects you interact with in spaCy—token.pos_, token.morph, token.dep_—are direct expressions of UD's universal layers.

---

Now we understood the UD basics and treebanks, now 
