## Project for Statistical Natural Language Processing (COMP0087), University College London

# Headline Generation via Adversarial Training
-----
### Collaborators:
- Daniel Stancl (ucabds7@ucl.ac.uk)
- Guoliang HE (ucabggh@ucl.ac.uk)
- Dorota Jagnesakova (ucabdj1@ucl.ac.uk)
- Zakhar Borok (zcabzbo@ucl.ac.uk)

### Abstract
Originally developed for computer vision, Generative Advarsarial Nets (GANs) have achieved great success in producing real-valued data. While these nets bypass a lot of the problems associated with more widely used models based on maximum-likelihood approaches, it remains a challenge to successfully train them for tasks involving discrete outputs, such as text summarization. In this paper, we propose an adversarial training approach for an abstractive text summarization task: generating headlines for WikiHow articles. We make use of two models participating in a contest to outperform each other — an encoder-decoder generator a discriminator. We train these with respect to two different loss functions and evaluate the resulting model's performance using ROUGE metrics. In particular, we see an improvement by 25.9 \% in ROUGE-1 and 42.6 \% in ROUGE-2 of our proposed adversarially-trained model as compared to  baseline seq2seq models.

### Data
**WikiHow dataset** - https://arxiv.org/pdf/1810.09305.pdf?fbclid=IwAR22xaM5JtRTHq-EMaBqSN30DaxhqF7dllK_8T47mOsnl8IY0ikM0VX3VKQ
