# Log of the prompts given to the AI agent

AI tool used: **Cursor (agent mode) powered by Claude**. The whole assignment –
literature search, code, experiments and this report – was produced by the
agent. Below are the prompts (lightly abridged) that were given to it, in
order, together with a short note on what the agent did in response.

---

## Prompt 1 – the assignment itself

> 完成這份 assignment, `AIassign1.pdf` 這個是要求.
> Your AI assignment report in Word/PDF format should contain the following
> details (1) how you ask AI tools to find the algorithm, (2) algorithm
> description, (3) how AI implements the algorithm, (4) experiment settings and
> results, (5) what you have learnt from this AI assignment, and (6) a webpage
> link of your source codes.
> 放在 https://github.com/HOWENGHONG/CISC3024-AI-Assignment 這裡, 但先把
> repository 中的所有檔都清除.
> 我的 student name: Ho Weng Hong, student id: uc325107

**What the agent did:** read the PDF, inspected the machine (Windows 11,
Python 3.14, AMD Ryzen 5 7600X, NVIDIA GeForce RTX 4060 8 GB but a CPU-only
PyTorch build), cloned the GitHub repository, removed the old files and
planned the work. It tried to install the CUDA build of PyTorch, but the
internet connection only delivered ~0.1 MB/s (the 2.7 GB wheel would have
taken many hours), so it decided to run everything on the CPU and to size the
models accordingly.

## Prompt 2 (internal, agent → search tools) – finding a recent algorithm

The agent issued the following searches to locate a *recent Deep CNN
algorithm with an application in computer vision / pattern recognition*:

1. Web search: `recent deep CNN architecture computer vision CVPR 2022 2023
   2024 ConvNeXt "A ConvNet for the 2020s"`
2. arXiv API query: `id_list=2201.03545` – to retrieve the title, authors,
   abstract and publication date of the ConvNeXt paper.
3. arXiv API query: `search_query=all:"ConvNeXt" AND cat:cs.CV`, sorted by
   relevance – to see how the algorithm has been followed up (ConvNeXt V2 2023,
   InceptionNeXt 2023, E-ConvNeXt 2025, medical-imaging and diffusion
   applications up to 2026), confirming that it is a *recent and still
   influential* deep CNN.

Candidates considered: ConvNeXt (CVPR 2022), ConvNeXt V2 (CVPR 2023, needs
masked-autoencoder pre-training – too heavy for one GPU), RepViT (CVPR 2024,
already used in a previous attempt), Masked Autoencoders (CVPR 2022, a deep
autoencoder – needs long pre-training). ConvNeXt was selected because it is a
pure deep CNN, is recent and highly cited, has a clear "modernisation roadmap"
that can be turned into an ablation study, and can be trained from scratch on
CIFAR-10 in a few hours on a laptop GPU.

## Prompt 3 (internal) – implementation

> Implement ConvNeXt from scratch in PyTorch (do not use the `timm` or
> `torchvision` pre-built model), adapted to 32×32 CIFAR-10 images. Keep all
> design elements of the paper: patchify-style stem, 4 stages with ratio
> 1:1:3:1, separate down-sampling layers, depthwise 7×7 convolution, inverted
> bottleneck, LayerNorm, GELU, layer scale and stochastic depth. Add switches so
> that the block can be reverted to a 3×3 kernel or to BatchNorm+ReLU for an
> ablation study, and add a CIFAR ResNet-18 baseline.

→ `src/models.py`

> Write a training script with AdamW, cosine schedule with linear warm-up,
> label smoothing, crop/flip augmentation (with an optional RandAugment /
> random-erasing "strong" mode), bf16 mixed precision and gradient clipping.
> Save a JSON with the full training history, confusion matrix, per-class
> accuracy, parameter count, training time and inference throughput.

→ `src/train.py`, `src/data.py`, `src/run_experiments.py`

> The original CIFAR-10 server is unreachable and the connection is very slow.
> Find a mirror, download it as fast as possible, verify the file and convert
> it to arrays the training script can use.

→ `src/prepare_data.py` (Hugging Face `uoft-cs/cifar10` Parquet mirror,
parallel range-request download, SHA-256 verified against the server ETag)

> Benchmark several depth/width configurations, kernel sizes and precisions
> on the CPU and pick a configuration such that the whole experiment suite
> finishes in about two hours.

→ ConvNeXt-CIFAR with depths (1, 1, 3, 1), dims (32, 64, 128, 256), bf16 +
channels-last, 15 epochs.

> Produce figures (training curves, accuracy comparison, confusion matrix,
> per-class accuracy, architecture diagrams, sample predictions) and build the
> Word report automatically from the result files.

→ `src/plot_results.py`, `src/visualize.py`, `report/make_report.py`

## Prompt 4 (internal) – experiments

> Run the full suite (ConvNeXt-CIFAR, the 3x3-kernel and BatchNorm+ReLU
> ablations, and a parameter-matched ResNet-20x2 baseline) for the same number
> of epochs with the same recipe, on the CPU. Fix the training loop if it is
> slower than the benchmark predicts.

→ `results/*.json`, `results/log_*.txt`, `report/figures/*.png`
(the first launch was slow because every DataLoader worker opened its own
12-thread pool; fixed with `torch.set_num_threads(1)` in the workers)

## Prompt 5 (internal) – report and publishing

> Write the report with the six required sections, insert the real numbers and
> figures, export it as .docx (and .pdf when possible), commit everything and
> push to the GitHub repository.
