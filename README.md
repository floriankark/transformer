# Transformer

This repo contains the pytorch implementation of the famous Transformer model as it has been orginally described by Vaswani et al. in [Attention Is All You Need](https://arxiv.org/abs/1706.03762)*. Moreover, this repo is the result of my work in the course "Implementing Transformers" from the winter semester 2023/24 at the [Heinrich Heine University Düsseldorf](https://www.heicad.hhu.de/lehre/masters-programme-ai-and-data-science) lead by [Carel van Niekerk](https://carelvniekerk.github.io/). 

There are many repos on implementing the transformer model, so why is this here interesting? In short, I successfully train and validate the model on an NVIDIA A100 in fp16, which requires some tricks and special attention that I would like to share with the community here :) 

Below you can find my written report to the code and course which I highly recommend to check out as it has some nice intuitive and mathematical explanations that I found or derived myself while researching on this topic.

**Disclaimer:** The code is not very intelligible with perfectly clean code and a simple training script. It is rather thougt as an educational material.

## Schedule

| Week | Dates         | Practical                                              |
|------|---------------|--------------------------------------------------------|
| 1    | 7-11.10.2023  | Practical 1: Getting Started and Introduction to Transformers and Attention |
| 2    | 14-18.10.2023 | Practical 2: Introduction to Unit Tests and Masked Attention |
| 3    | 21-25.10.2023 | Practical 3: Tokenization                              |
| 4    | 28-31.10.2023 | Practical 4: Data Preparation and Embedding Layers     |
| 5    | 4-8.11.2023   | Practical 5: Multi-Head Attention Blocks               |
| 6    | 11-15.11.2023 | Practical 6: Transformer Encoder and Decoder Layers    |        
| 7    | 18-22.11.2023 | Practical 6: Transformer Encoder and Decoder Layers    | 
| 8    | 25-29.11.2023 | Practical 7: Complete Transformer Model                | 
| 9    | 2-6.12.2023   | Practical 8: Training and Learning rate Schedules      |                                                
| 10   | 9-13.12.2023  | Practical 9: Training the model                        |
| 11   | 16-20.12.2023 | Practical 10: Training the model                       |                                               
| 12   | 6-11.01.2024  | Practical 11: Autoregressive Generation and Evaluation |                                                
| 13   | 13-17.01.2024 | Practical 12: GPU Training (HPC)                       |                                                
| 14   | 01.03.2024    | Deadline of written report                             |  
| 14   | 09.04.2024    | Oral presentation in person                            | 

Report Guidelines:
**Word Limit:** The report should not exceed 2500 words.
**Page Limit:** The report must be a maximum of 8 pages.

Unfortunately, my code was prone to the vanishing gradients problems in fp16 training which I was eventually able to fix but as you might have read, a little to late for the written report. If you are interested, have a look at the presentation slides to get a gist of the problem and see the great results.

**Presentation Guidelines:**
Prepare a 10-minute presentation highlighting the most important aspects of your report.
The presentation should focus on key insights, challenges, and outcomes from your project.


*Yes, I implemented both, Pre-LN, as described in the paper, and Post-LN, as later updated in the official code; even tested some other constellations
