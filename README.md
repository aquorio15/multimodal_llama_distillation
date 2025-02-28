# Semi-Supervised Knowledge Distillation Framework towards Lightweight Large Language Model for Spoken Language Translation

This is the code for the paper "Semi-Supervised Knowledge Distillation Framework towards Lightweight Large Language Model for Spoken Language Translation" whisch was accepted at # ICASSP-2025

## Overview

Even though large language models (LLMs) have demonstrated remarkable performance across various natural language processing tasks, their application in speech-related tasks has largely remained underexplored. This work addresses this gap by incorporating acoustic features into an LLM which can be fine-tuned for downstream direct speech-to-text translation and automatic speech recognition tasks. To address the computational demands associated with fine-tuning LLMs, a novel self and semi-supervised knowledge distillation technique is proposed to implement a lightweight LLM having 50% lesser parameters. Validated on the MuST-C and Librispeech  datasets, this technique achieves over 92% of the performance of the larger LLM, demonstrating both robust performance and computational efficiency.

![wav2vec](https://github.com/user-attachments/assets/40a9e4ee-0099-4834-b3df-550471fc3e8b)

## Installation

#### Prerequisites
* Python>=3.8
* torch>1.6
