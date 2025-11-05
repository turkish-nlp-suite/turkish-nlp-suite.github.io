---
layout: post
title:  Big, Bold and Badass / BellaTurca, the First Large-Scale, Diverse and High-Quality Turkis Corpus Collection Ever
date:   2025-10-05 10:05:55 +0300
image:  /assets/images/blog/pile.png
additional_image: /assets/images/blog/categories.png
author: Duygu
tags:   corpus, research
---

**We proudly introduce BellaTurca, the ultimate Turkish large corpora, providing diversity and high-quality to fight dullness and blandness in Turkish language modelling. We offer around 250GB of text and 30B words, from four different subsets, all high-quality, carefully cleaned and crafted. All are yours, freely available on Hugging Face.**


Long long time ago, in the far away computing times, there were not much diversity in the corpus world. Though BookCorpus made a breakthrough in 2015 (then sent away due to licencing issues), it was only 6GB in size, which was enough for comparably "small" models. However when one searches for large amounts of data to train their models, first source comes to mind is surely Web data. Then OSCAR and mC4 projects published their huge corpora, measured in treabytes, on 2019 and gave birth to quite many models including "small models" such as BERT to midsize models such as GPT-2 and RoBERTa, then large models such as T5 and GPT-2. Up to this point, almost all models were trained 
n Web data, quite boring and bland indeed. Then came the glorious Pile in 2020, not only introducing an impressive 825GBs of data, but also provide diversity with text of different genres, 22 subsets, from medicine to law, Github code to Arxiv papers.


Coming to Turkish, it's even more boring and bland. Literally there were only 2 corpora, mC4 and OSCAR, lying around as splits of those projects in Hugging Face. Also corpus statistics of those datasets and quality is not well-researched. Though some models including a BERT and some GP models were trained on those datasets, no one ever published about quality of those datasets, how much percent is usable, how much percent is filtered and how many dupliactes in the data. Moreover when one wanna use those data, they need to the cleaning by themselves, hence lots of code replication and time loss accross different work. 

To fight all the dullness and blandness, we introduce BellaTurca, a large-scale , diverse and high-quality Turkish corpus collection. The collection has 5 distinct subsets:


* ÖzenliDerlem
* AkademikDerlem
* Kitaplar 
* ForumSohbetleri
* Temiz OSCAR
* Temiz mC4

Each subset has its own characteristics. ÖzenliDerlem (CraftedCrawl) includes web text according to curated topics such as culture, literature, fairy tales, stories, travelling and more, from curated websites. AkademikDerlem (AcademicalCrawl) includes theses and papers published, mostly crawled from DergiPark. Kitaplar (Books) is our books corpus, collected from several .zips on the internet including free books. Temiz OSCAR and Temiz mC4 are extensively cleaned versions of OSCAR and mC4 Turkish splits. We excluded Kitaplar in the final version of the collection due to suspicion of including licenced material. The final collection size is around 250GB. Subcorpus sizes can be found under [Bella Turca HF repo](https://huggingface.co/datasets/turkish-nlp-suite/BellaTurca). Each subcorpus has its own dedicated HF repo as well, can be found under the main repo link.

There's also a subcorpus deserves mentioning, [Havadis](https://huggingface.co/datasets/turkish-nlp-suite/Havadis) - first big size Turkish news corpus, part of ÖzenliDerlem. This corpus includes text from news websites, including Hürriyet, Milliyet, Star, Posta and more. If you wanna play with Havadis, check out the [blog post]().


Cleaning of BellaTurca sets took place in different ways due to the needs of the dataset nature. For the cleaning of OSCAR and mC4 - Web data can include duplicates, low quality content (such as SEO optimized websites), ads, adult content and similar. We first made language filtering to web data. We also cleaned Chinese and Korean characters as well. We also run dedup on paragraph level and document level with local hashing methods. We made several steps of cleaning to web data to ensure quality.


Coming to academic papers, we collect the text from PDFs via an OCR, hence there will be some OCR mistakes surely. Another thing with papers is one needs to parse out and normalize non-text content such as tables and figures. In this subset , we focused on paper parsing. We cut the abstracts (and placed them into their own subset), clean English abstract if exists and clean some English parts.

Books took most the effort by requiring huge OCR correction. For this task we used ByT5 model. First we hold a small but quality subset of books that is collected by hand. Then made a pass with this model on the rest of the books, grab the corrected text and merge it with the seed text. Retrain ByT5 , and repeat the process so its a self-feeding autotrain loop.

ForumSohbetleri wnt through carefully text cleaning, we wanted to reserve the daily spoken language in text together with emoticons but still wanted to normalize the text. 


ÖzenliDerlem also took some stages of cleaning, again in an auto-train loop. In this subset, we started with a small number of homepages per selected topic whose quality we knew. Then, we crawled these pages and collected more page addresses. To determine quality, we converted all the pages to vectors using a sentence encoder and grouped them. We kept the pages at the center of the groups as the core set. We built an n-gram model based on the core set, ran this model over the remaining pages, and added the pages with good scores to the high-scoring set. In this way, we ran a small autotrain loop. In each round, we built the model from the high-quality pages, assigned a score to the remaining pages, and added the high-scoring pages to the high-scoring set. With each round, the dataset grew, the model improved, and it became more confident. We continued this process until the scores reached a stable point.

When you look at The Pile and similar examples, you'll see that the reverse is done: first, the pages are collected, then a quality model and a topic model are created from another set of known quality, and the remaining pages are filtered accordingly. Since we didn't have such sets of known quality, we used this method as a self-reinforcing approach. This is one of the creation goals of Bella Turca, to be a core dataset for larger datasets that will be produced as a collection of known quality and topics.



The below image offers the cleaning process of BellaTurca at a glance:










BellaTurca is available under a commercial permissive licence under its dedicated [HF repo](https://huggingface.co/datasets/turkish-nlp-suite/BellaTurca). Subsets are also available under the [HF collection](https://huggingface.co/collections/turkish-nlp-suite/large-scale-turkish-corpora).
