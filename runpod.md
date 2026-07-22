# RunPod Deployment Checklist — `01_paper_reproduction`

Checklist for running `experiments/01_paper_reproduction` on a RunPod GPU pod.
See that folder's `README.md` for what the experiment does; this file is just
the deployment mechanics.

## 1. Before launching a pod

- [ ] Request access to `mistralai/Mistral-7B-Instruct-v0.2` on Hugging Face
      (gated model) and wait for approval.
- [ ] Generate a Hugging Face **read** token for that account.
- [ ] If you haven't already, expand the item/room dataset **locally** (not on
      the pod — no GPU needed, and keeps your Anthropic key off a rented
      instance):
  ```bash
  cd experiments/01_paper_reproduction
  pip install anthropic python-dotenv
  cp .env.example .env   # fill in ANTHROPIC_API_KEY
  python expand_dataset.py --n-per-category 20
  ```
  Confirm `expanded_inventory.json` was created and looks reasonable.
- [ ] Have a RunPod account with billing set up.

## 2. Launch the pod

- [ ] GPU: a single **A100** (40GB is enough — 4-bit quant + LoRA on a 7B
      model). Matches the paper's own setup.
- [ ] Template: CUDA version compatible with `torch==2.4.0` (a CUDA 12.1
      PyTorch template works).
- [ ] Storage: attach a **network volume** if you want checkpoints/results to
      survive pod termination — pod-local disk is ephemeral. Otherwise plan to
      download outputs before stopping the pod (see step 5).

## 3. Get the code and data onto the pod

- [ ] Upload/clone the repo onto the pod (or at minimum
      `experiments/01_paper_reproduction/`).
- [ ] Upload the locally-generated `expanded_inventory.json` into
      `01_paper_reproduction/` on the pod (scp, RunPod's file browser, or
      pack it into your repo upload — do **not** regenerate it on the pod).
- [ ] Do **not** upload your `.env` — the pod doesn't need the Anthropic key,
      only `HF_TOKEN` (set in the next step).

## 4. Install and configure

- [ ] `cd experiments/01_paper_reproduction`
- [ ] `pip install -r requirements.txt`
- [ ] `export HF_TOKEN=...` (the token from step 1)
- [ ] Sanity-check GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"`

## 5. Run

- [ ] `bash run.sh` — runs all 5 seeds (SOO fine-tune → evaluate), then
      aggregates. Expect ~13 min/seed based on the paper's reported ~65 min
      for fine-tuning all three of its models across five seeds combined; budget
      more time for this repo's full train+eval loop per seed.
- [ ] Watch the periodic `soo_loss` printouts (`train.py`) for sane (non-NaN,
      decreasing) values.
- [ ] After it finishes, check `aggregate.py`'s printed summary against the
      paper target: deceptive response rate **73.6% → 17.27 ± 1.88%**.
      - If Latent SOO barely moves or deception doesn't drop, that's the known
        mean-pooling degeneracy — set `TRAIN.pooling="last"` in `config.py`
        and re-run (see `README.md`).

## 6. Retrieve results before the pod disappears

- [ ] Download `results/*.json` (per-seed metrics) off the pod.
- [ ] Download `checkpoints/` if you want the LoRA adapters, or discard them
      if not needed (they're gitignored, so they won't come back via git).
- [ ] Fill in `results.ipynb`'s Results/Conclusion sections locally with the
      retrieved numbers.

## 7. Clean up

- [ ] Terminate the pod (and detach/delete the network volume if you attached
      one and no longer need it) to stop billing.
- [ ] Double-check nothing sensitive (`.env`, `HF_TOKEN`) is baked into any
      image or volume snapshot you keep around.
