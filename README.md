# GENTI

This is the original code for "GENTI: GPU-powered Walk-based Subgraph Extraction for Scalable Representation Learning on Dynamic Graphs" (VLDB 2024).

## Environment
Prepare the environment. We use Anaconda to manage packages. The following command create the conda environment to be used:
```{bash}
conda env create -f environment.yml
```
Refer to `environment.yml` for more details.

## Dataset and preprocessing
#### Option 1: Use preprocessed data
Raw data and preprocessing methods can be found at the following links:
- Reddit, Wikipedia, UCI-MSG: download from [here](https://drive.google.com/drive/folders/1umS1m1YbOM10QOyVbGwtXrsiK3uTD7xQ?usp=sharing) to `./processed`, then run the following command:

```{bash}
cd processed/
unzip data.zip
```

- SuperUser: downloadable from http://snap.stanford.edu/data/sx-superuser.html.

- Wiki-Talk: downloadable from http://snap.stanford.edu/data/wiki-talk-temporal.html.

- Tgbl-Comment: https://tgb.complexdatalab.com/

- MAG: https://github.com/amazon-science/tgl.

You may check that each dataset corresponds to three files: one `.csv` containing timestamped links, and two ``.npy`` as node & link features. Note that some datasets do not have node & link features, in which case the `.npy` files will be all zeros.

#### Option 2: Use your own data
Put your data under `processed` folder. The required input data includes `ml_${DATA_NAME}.csv`, `ml_${DATA_NAME}.npy` (option) and `ml_${DATA_NAME}_node.npy`(option). They store the edge linkages, edge features and node features respectively. 

The `.csv` file has following columns
```
u, i, ts, label, idx
```
, which represents source node index, target node index, time stamp, edge label and the edge index. 

`ml_${DATA_NAME}.npy` has shape of [#temporal edges + 1, edge features dimention]. Similarly, `ml_${DATA_NAME}_node.npy` has shape of [#nodes + 1, node features dimension].


All node index starts from `1`. The zero index is reserved for `null` during padding operations. So the maximum of node index equals to the total number of nodes. Similarly, maxinum of edge index equals to the total number of temporal edges. The padding embeddings or the null embeddings is a vector of zeros.

We also recommend discretizing the timestamps (`ts`) into integers for better indexing.
## Training Commands

#### Examples:

* The command to train **GENTI** on link predction datasets:

```bash
# Training on Wikipedia
python main.py -d wikipedia --pos_dim 108 --bs 64 --n_walks 64 --n_steps 2 --w 128 --bias 1e-5 --walk_pool sum --seed 0 --gpu 0
# Training on Reddit
python main.py -d reddit --pos_dim 108 --bs 64 --n_walks 64 --n_steps 2 --w 64 --bias 1e-5 --walk_pool sum --seed 0 --gpu 0
# Training on UCI-MSG
python main.py -d uci --pos_dim 100 --bs 64 --n_walks 32 --n_steps 2 --w 32 --bias 1e-6 --walk_pool attn --seed 123 --gpu 0
# Training on SuperUser
python main.py -d superuser --pos_dim 128 --bs 128 --n_walks 32 --n_steps 2 --w 32 --bias 1e-7 --walk_pool sum --seed 123 --gpu 0
# Training on Wiki-Talk
python main.py -d wikitalk --pos_dim 128 --bs 128 --n_walks 32 --n_steps 2 --w 32 --bias 1e-7 --walk_pool sum --seed 123 --gpu 0 
# Training on MAG
python main.py -d mag --pos_dim 128 --bs 256 --n_walks 32 --n_steps 2 --w 32 --bias 1 --walk_pool sum --seed 123 --gpu 0
# Training on Tgbl-Comment
python main.py -d comment --pos_dim 128 --bs 128 --n_walks 32 --n_steps 2 --w 64 --bias 1e-7 --walk_pool sum --seed 123 --gpu 0
```


* The command to train **GENTI** on node classification datasets:

```bash
# Training on Wikipedia
python node_classification.py -d wikipedia --pos_dim 108 --bs 64 --n_walks 64 --n_steps 2 --w 64 --bias 1e-5 --walk_pool sum --seed 0 --gpu 0
python node_classification.py -d reddit --pos_dim 108 --bs 64 --n_walks 64 --n_steps 2 --w 64 --bias 1e-5 --walk_pool sum --seed 0 --gpu 0
```

Detailed logs can be found in `log/`.
 
## Usage Summary
```txt
usage: Interface for Inductive Dynamic Representation Learning for Link Prediction on Temporal Graphs
       [-h] [-d]
       [--n_walks N_WALKS] [--n_steps N_STEPS] [--bias BIAS] [--agg {tree,walk}]
       [--pos_dim POS_DIM] [--pos_sample {multinomial,binary}] [--walk_pool {attn,sum}] [--walk_n_head WALK_N_HEAD]
       [--walk_mutual] [--walk_linear_out] [--attn_agg_method {attn,lstm,mean}] [--attn_mode {prod,map}]
       [--attn_n_head ATTN_N_HEAD] [--time {time,pos,empty}] [--n_epoch N_EPOCH] [--bs BS] [--lr LR] [--drop_out DROP_OUT]
       [--tolerance TOLERANCE] [--seed SEED] [--ngh_cache] [--gpu GPU] [--verbosity VERBOSITY]
```

## Optional arguments
```{txt}
  -h, --help show this help message and exit
  -d , --data data sources to use, try wikipedia or reddit
  --n_walks [n_walks [n_walks ...]]
                        a list of neighbor sampling numbers for different hops, when only a single element is input n_steps
                        will be activated
  --n_steps n_steps     number of network layers
  --bias BIAS           the hyperparameter alpha controlling sampling preference with time closeness, default to 0 which is
                        uniform sampling
  --pos_dim POS_DIM     dimension of the positional embedding
  --pos_sample {multinomial,binary}
                        two practically different sampling methods that are equivalent in theory
  --walk_pool {attn,sum}
                        how to pool the encoded walks, using attention or simple sum, if sum will overwrite all the other
                        walk_ arguments
  --walk_n_head WALK_N_HEAD
                        number of heads to use for walk attention
  --walk_mutual         whether to do mutual query for source and target node random walks
  --walk_linear_out     whether to linearly project each node's
  --attn_agg_method {attn,lstm,mean}
                        local aggregation method, we only use the default here
  --attn_mode {prod,map}
                        use dot product attention or mapping based, we only use the default here
  --attn_n_head ATTN_N_HEAD
                        number of heads used in tree-shaped attention layer, we only use the default here
  --time {time,pos,empty}
                        how to use time information, we only use the default here
  --n_epoch N_EPOCH     number of epochs
  --bs BS               batch_size
  --lr LR               learning rate
  --drop_out DROP_OUT   dropout probability for all dropout layers
  --tolerance TOLERANCE
                        toleratd margainal improvement for early stopper
  --seed SEED           random seed for all randomized algorithms
  --ngh_cache           (currently not suggested due to overwhelming memory consumption) cache temporal neighbors previously
                        calculated to speed up repeated lookup
  --gpu GPU             which gpu to use
  --verbosity VERBOSITY
                        verbosity of the program output
```

## Acknowledgement

We adapt the code provided in the repository [CAW](https://github.com/snap-stanford/CAW) and [TGAT](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs) as the foundation for our implementation. We appreciate the authors for sharing the code.

## Cite

```
@inproceedings{yu2024genti,
  title={GENTI: GPU-powered Walk-based Subgraph Extraction for Scalable Representation Learning on Dynamic Graphs},
  author={Yu, Zihao and Liao, Ningyi and Luo, Siqiang},
  booktitle={Proceedings of the VLDB Endowment},
  volume={17},
  year={2024},
}
```