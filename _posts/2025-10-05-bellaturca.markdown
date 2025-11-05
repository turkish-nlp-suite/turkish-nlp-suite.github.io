---
layout: post
title:  Big, Bold and Badass / BellaTurca, the First Large-Scale, Diverse and High-Quality Turkis Corpus Collection Ever
date:   2025-10-05 10:05:55 +0300
image:  /assets/images/blog/pile.png
additional_image: /assets/images/blog/categories.png
author: Duygu
tags:   Corpus, Research
---

**We proudly introduce BellaTurca, the ultimate Turkish large corpus, bringing diversity and high quality to fight the dullness and blandness in Turkish language modeling. We're talking about 250GB of text and over 30 billion words from six different subsets, all high-quality, cleaned, and carefully crafted. And guess what? They’re all freely available on Hugging Face. Yep, all yours!**


A long, long time ago, in the early days of computing, there wasn’t much diversity in the corpus world. Sure, BookCorpus made waves back in 2015 (before getting booted due to licensing issues), but it was only 6GB — decent for "small" models back then. But when people started needing big data for big models, the first thing that came to mind was web data.

Then came OSCAR and mC4 in 2019, with their giant web-based corpora measured in terabytes. These powered a lot of models — from smaller ones like BERT, to midsize ones like GPT-2 and RoBERTa, and even bigger ones like T5. But let’s be real — almost all of these datasets were just... web data. Boring. Repetitive. Bland.

In 2020, The Pile changed everything. Not only did it introduce a whopping 825GB of data, but it also brought diversity — text from 22 different subsets, covering everything from medicine to law, GitHub code to Arxiv papers. It showed the world that diversity matters in corpora.

Now, coming to Turkish... oh boy. It was even worse. All we had were two corpora — mC4 and OSCAR, which were just Turkish splits from those bigger projects on Hugging Face. And let’s be honest, no one really knew how good or bad they were. No one studied their stats, no one checked their quality. How much of it was usable? How much was filtered out? How many duplicates were there? No answers.

Even worse, if you wanted to use those datasets, you had to clean them yourself. That meant everyone was writing the same cleaning scripts, wasting time, and duplicating effort across different projects.

To fight all this dullness and blandness, we proudly introduce BellaTurca, a large-scale, diverse, and high-quality Turkish corpus collection. The collection includes six distinct subsets, each with its own unique flavor:

- ÖzenliDerlem (CraftedCrawl): Curated web text from high-quality websites on topics like culture, literature, fairy tales, stories, and travel.
- AkademikDerlem (AcademicalCrawl): Academic papers and theses, mostly crawled from DergiPark.
- Kitaplar (Books): A books corpus collected from free sources on the internet. (Note: Excluded in the final version due to concerns about licensed material.)
- ForumSohbetleri (ForumChats): Text from online forums, preserving conversational Turkish and emoticons while normalizing the text.
- Temiz OSCAR (Clean OSCAR): A heavily cleaned version of the Turkish OSCAR dataset.
- Temiz mC4 (Clean mC4): A cleaned version of the Turkish mC4 dataset.

The final collection size is around 250GB. You can find the stats for each subset on the [Bella Turca HF repo](https://huggingface.co/datasets/turkish-nlp-suite/BellaTurca). Each subset also has its own dedicated Hugging Face repo.

There’s one subset that deserves special mention: [Havadis](https://huggingface.co/datasets/turkish-nlp-suite/Havadis). It’s the first large-scale Turkish news corpus, featuring text from major news outlets like Hürriyet, Milliyet, Star, and Posta. If you’re into playing with news data, this is the one to check out. (We’ll write more about it in another blog post soon!)


#### Cleaning BellaTurca

Cleaning the datasets was no easy task. Each subset had its own challenges, so we used different methods depending on the type of data. Here’s a quick summary:

##### Web Data (OSCAR, mC4)
- Language Filtering: Removed non-Turkish content (and random languages like Chinese or Korean). 
- Deduplication: Removed duplicate paragraphs and documents using hashing.
- Content Filtering: Cleaned low-quality text like SEO pages, ads, and adult content.

##### Academic Papers
- Extracted text from PDFs using OCR (as expected).
- Removed non-text content like tables and figures.
- Cleaned abstracts (both Turkish and English).

##### Books
- Books were the hardest. We used a ByT5 model for OCR correction. 
- First, we trained it on a small, high-quality seed set of books. 
- Then, we ran the model on the rest of the books, corrected the text, and retrained the model in a self-reinforcing loop. 
- Rinse and repeat until we got clean text.

##### ForumSohbetleri
- Kept conversational Turkish and emoticons intact while normalizing the text for better usability.

##### ÖzenliDerlem
- Used a self-reinforcing autotrain loop:
- Started with a small set of high-quality pages.
- Crawled similar pages using a sentence encoder and grouped them into clusters.
- Built an n-gram model from the high-quality pages and scored the remaining pages.
- Added the high-scoring pages to the dataset.
- Repeated the process until the dataset hit a stable point.

This iterative approach helped us build a diverse, high-quality set of web pages without needing an external quality dataset.

---

BellaTurca isn’t just another corpus. It’s designed to be a core dataset for Turkish NLP, offering a foundation for building even larger, high-quality collections. Think of it as the start of something much bigger. With BellaTurca, you don’t just get data — you get diversity, quality, and the tools to take Turkish NLP to the next level.


BellaTurca is available under a commercially permissive license. You can grab the full dataset and its subsets on the [HF repo](https://huggingface.co/datasets/turkish-nlp-suite/BellaTurca) repo or explore the [full collection](https://huggingface.co/collections/turkish-nlp-suite/large-scale-turkish-corpora)

So go ahead, download it, and start building something awesome, happy modelling!
