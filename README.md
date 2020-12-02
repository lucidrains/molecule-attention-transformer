<img src="./mat.png" width="500px"></img>

## Molecule Attention Transformer - Pytorch (wip)

Pytorch reimplementation of <a href="https://arxiv.org/abs/2002.08264">Molecule Attention Transformer</a>, which uses a slightly modified transformer to tackle the graph-like structure of molecules. The repository is also meant to be educational, to understand the limitations of transformers for processing graphs (or perhaps lack thereof).

Update: Reread the paper and results do look convincing. However, I do not like how it still takes hyperparameter sweeps of the relative contributions of the distance, adjacency, and self attention matrices to achieve good results. There must be more hands-off way

## Install

```bash
$ pip install molecule-attention-transformer
```

## Usage

```python
import torch
from molecule_attention_transformer import MAT

model = MAT(
    dim_in = 26,
    model_dim = 512,
    dim_out = 1,
    depth = 6,
    Lg = 0.5,                   # lambda (g)raph - weight for adjacency matrix
    Ld = 0.5,                   # lambda (d)istance - weight for distance matrix
    La = 1,                     # lambda (a)ttention - weight for usual self-attention
    dist_kernel_fn = 'exp'      # distance kernel fn - either 'exp' or 'softmax'
)

atoms           = torch.randn(2, 100, 26)
mask            = torch.ones(2, 100).bool()
adjacency_mat   = torch.empty(2, 100, 100).random_(2).float()
distance_mat    = torch.randn(2, 100, 100)

out = model(
    atoms,
    mask = mask,
    adjacency_mat = adjacency_mat,
    distance_mat = distance_mat
) # (2, 1)
```

## Citations

```bibtex
@misc{maziarka2020molecule,
    title={Molecule Attention Transformer}, 
    author={Łukasz Maziarka and Tomasz Danel and Sławomir Mucha and Krzysztof Rataj and Jacek Tabor and Stanisław Jastrzębski},
    year={2020},
    eprint={2002.08264},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
