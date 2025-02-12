<p align="center">
  <picture>
    <img alt="Gener" src="figures/logo.png" width=50%>
  </picture>
</p>

<h1 align="center">GENERanno: A Unified Genomic Foundation Model with Specialization in Gene Annotation</h1>

## 📰 News
* 🤗 **[2025-02-11]** We are pleased to announce that our models `GENERanno-prokaryote-0.5b-base`, `GENERanno-eukaryote-0.5b-base` are now available on [Hugging Face](https://huggingface.co/GenerTeam/)!

## 🔭 Overview

In this repository, we present GENERanno, a genomic foundation model featuring a context length of 8k base pairs and 500M parameters, trained on an expansive dataset comprising 386 billion base pairs of eukaryotic DNA. Our evaluations demonstrate that the GENERanno achieves comparable performance with [GENERator](https://huggingface.co/GenerTeam/GENERator-eukaryote-1.2b-base) in benchmark evaluations, including [Genomic Benchmarks](https://huggingface.co/datasets/katielink/genomic-benchmarks/tree/main), [NT tasks](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised), and our newly proposed [Gener tasks](https://huggingface.co/GenerTeam), making them the top genomic foundation models in the field (2025-02). 

Beyond benchmark performance, the GENERanno model is meticulously designed with its specialization in gene annotation. The model efficiently and accurately identifies gene locations, predicts gene function, and annotates gene structure, highlighting its potential to revolutionize genomic research by significantly enhancing the precision and efficiency of gene annotation processes.

Please note that the GENERanno is currently in the developmental phase. We are actively refining the model and will release more technical details soon. Stay tuned for updates!

In this repository, you will find the following model checkpoints:

| Model Name                       | Parameters | Data | Category | Status |
|----------------------------------|:----------:|:----------:|:----------:|:----------:|
| `GENERanno-eukaryote-0.5b-base`  |    0.5B    | 386B | Eukaryote                   | [Available](https://huggingface.co/GenerTeam) |
| `GENERanno-prokaryote-0.5b-base` |    0.5B    | 715B | Prokaryote+Virus            | [Available](https://huggingface.co/GenerTeam) |
| `GENERanno-eukaryote-1b-base`    |     1B     | 386B | Eukaryote                   | Awaiting sponsorship |
| `GENERanno-prokaryote-1b-base`   |     1B     | 715B | Prokaryote+Virus            | Awaiting sponsorship |

## 📈 Benchmark Performance
![benchmark](figures/benchmarks.png)

## 🎯 Quick Start
working in progress...

## 📚 Datasets
coming soon...

## 📜 Citation
```
@misc{wu2025generator,
      title={GENERator: A Long-Context Generative Genomic Foundation Model}, 
      author={Wei Wu and Qiuyi Li and Mingyang Li and Kun Fu and Fuli Feng and Jieping Ye and Hui Xiong and Zheng Wang},
      year={2025},
      eprint={2502.07272},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.07272}, 
}
```
