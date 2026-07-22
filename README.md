# paper-soo-empathy

Research project repo. Follows the team's [research structure](claude.md): each Linear issue gets its own folder under `experiments/`, containing a results notebook plus supporting scripts.

## Layout

```
root
|- README.md
|- experiments/       # one folder per Linear issue (e.g. AGI-244-some-experiment/results.ipynb)
|- spikes/            # quick, informal explorations
|- src/
    |- soo_empathy/    # shared package code (only factor things out here when reuse is proven)
    |- tests/
```

## Setup

```bash
pip install -e .
```

See [claude.md](claude.md) for full research standards (reporting template, review process, GitHub/Drive conventions).
