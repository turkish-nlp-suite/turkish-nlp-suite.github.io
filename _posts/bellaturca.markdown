---
layout: post
title:  Big, Bold and Badass: BellaTurca, the First Large-Scale, Diverse and High-Quality Turkis Corpus Collection Ever
date:   2025-10-05 10:05:55 +0300
image:  /assets/images/blog/pile.png
author: Duygu
tags:   corpus, research
---

**We proudly introduce BellaTurca, the ultimate Turkish large corpora, providing diversity and high-quality to fight dullness and blandness in Turkish language modelling. We offer around 235GB of text and 30B words, from four different subsets, all high-quality, carefully cleaned and crafted. All are yours, freely available on Hugging Face.**


Long long time ago, in the far away computing times, there were not much diversity in the corpus world. Though BookCorpus made a breakthrough in 2015 (then sent away due to licencing issues), it was only 6GB in size, which was enough for comparably "small" models. However when one searches for large amounts of data to train their models, first source comes to mind is surely Web data. Then OSCAR and mC4 projects published their huge corpora, measured in treabytes, on 2019 and gave birth to quite many models including "small models" such as BERT to midsize models such as GPT-2 and RoBERTa, then large models such as T5 and GPT-2. Up to this point, almost all models were trained 
n Web data, quite boring and bland indeed. Then came the glorious Pile in 2020, not only introducing an impressive 825GBs of data, but also provide diversity with text of different genres, 22 subsets, from medicine to law, Github code to Arxiv papers.


Coming to Turkish, it's even more boring and bland. Literally there were only 2 corpora, mC4 and OSCAR, lying around as splits of those projects in Hugging Face. Also corpus statistics of those datasets and quality is not well-researched. Though some models including a BERT and some GP models were trained on those datasets, no one ever published about quality of those datasets, how much percent is usable, how much percent is filtered and how many dupliactes in the data. Moreover when one wanna use those data, they need to the cleaning by themselves, hence lots of code replication and time loss accross different work. 

To fight all the dullness and blandness, we introduce BellaTurca


* Havadis
* ÖzenliDerlem
* AkademikDerlem
* Temiz OSCAR
* Temiz mC4


BellaTurca is available under a commercial permissive licence under its dedicated [HF repo](https://huggingface.co/datasets/turkish-nlp-suite/BellaTurca). Subsets are also available under the [HF collection](https://huggingface.co/collections/turkish-nlp-suite/large-scale-turkish-corpora).
