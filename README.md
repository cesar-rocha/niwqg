# niwqg
Code for a special class of solutions of the Xie & Vanneste (2015) coupled model
 in a doubly periodic domain –– the documentation is in the works.

# Installation
This software is written in `python3` and depends on `numpy` and `h5py`. I strongly
recommend the python3 pre-packaged on [Anaconda](https://www.continuum.io/downloads).
This package come with `numpy`. To install h5py, use Anaconda's package manager:
```bash
conda install h5py
```


## Obtaining niwqg
Download or clone niwqg repository.

## Installing niwqg
Inside the root niwqg directory, install the package:

```bash
python setup.py install
```

If you plan to make changes to the code, then setup the development mode

```bash
python setup.py develop
```
# Development
The code is under rapid development by @crocha700 as part of the project
"Stimulated Loss of Balance" (SLOB) with @glwagner and @wry55.

Guidelines for contributions will soon be posted in the documentation. For now,
the only requirement is that contributions be submitted via pull-request of
a cut-off branch (not master).
