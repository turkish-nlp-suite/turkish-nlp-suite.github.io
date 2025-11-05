---
layout: post
title:  Introducing spaCy Turkish models
date:   2025-09-15 12:05:55 +0300
image:  /assets/images/blog/post-4.jpg
author: Duygu
tags:   spaCy Turkish models
---

**Natural Language Processing (NLP) has grown leaps and bounds in the past decade, but if you're working with Turkish, you’ve probably faced some challenges. Turkish is a morphologically rich, agglutinative language that doesn’t always play nicely with NLP tools designed for English or other Indo-European languages. That's where spaCy Turkish models come in! Whether you're analyzing text, identifying parts of speech (POS), or parsing syntax, spaCy’s tr_core_news models are your go-to tools for Turkish NLP. In this blog, we’ll introduce you to spaCy's Turkish models, walk you through their features, and show how to load and use them effectively. By the end, you'll be ready to unlock the potential of Turkish texts with spaCy!**

spaCy Turkish models are around for some quite while, we already got to know the packages in the [Medium post](https://medium.com/google-developer-experts/brand-new-spacy-turkish-models-304da649eacc). Yet in this post, we'll get to know the packages better by discussing the production process of the packages.

spaCy’s Turkish models are pretrained language models that help you analyze Turkish text. They support:

    Tokenization: Splitting text into words, punctuation, and symbols.
    Part-of-Speech (POS) Tagging: Identifying whether a word is a noun, verb, adjective, etc.
    Dependency Parsing: Understanding how words relate to each other in a sentence.
    Named Entity Recognition (NER): Detecting entities like names, dates, and locations.
    Morphological Features: Capturing Turkish-specific details like case, tense, and person.

