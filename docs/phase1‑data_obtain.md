

# Why this workflow?

* **Keep the repo small & safe:** GitHub **blocks files >100 MiB**. Large datasets should not be committed; use links or a download step instead. ([GitHub Docs][1])
* **Stay clean with `.gitignore`:** Tell Git to **ignore** your local `data/` folder so CSV/PDF files never get pushed. If you ever accidentally committed a file already, remove it from tracking with `git rm --cached` and keep it locally. ([Atlassian][2])

---

# Step-by-step (copy/paste friendly)

## 1) Create a local data folder and ignore it

**What:** keep datasets only on each person’s laptop.

```bash
# from the repo root
mkdir -p data
echo "data/" >> .gitignore   # tell Git to ignore the entire data/ folder
git add .gitignore
git commit -m "chore: ignore data/ locally"
git push
```

**Why:** Files under `data/` won’t be tracked or uploaded—exactly what we want. If a file was already committed earlier, run `git rm --cached <path>` once, then commit again. (This is the standard way to start ignoring a previously tracked file.) ([Atlassian][2])

> Tip (optional): keep an **empty placeholder** so teammates see the folder exists:

```bash
echo "!data/.gitkeep" >> .gitignore
touch data/.gitkeep
git add .gitignore data/.gitkeep
git commit -m "chore: keep empty data/ placeholder"
git push
```

(The `!pattern` rule is how you make an exception inside `.gitignore`.) ([Atlassian][2])

---

## 2) One-line download for the **Kaggle Sensor Fault Detection** dataset

**What:** each teammate downloads to their own `data/` folder via Kaggle CLI.

1. Install and set up the Kaggle API:

```bash
pip install kaggle
# Create an API token on Kaggle: Profile → Account → "Create API Token"
# Windows path for the token file:
#   C:\Users\<YourWindowsUser>\.kaggle\kaggle.json
# (You can also set KAGGLE_CONFIG_DIR; see docs.)
```

2. Download & unzip into `data/`:

```bash
kaggle datasets download -d arashnic/sensor-fault-detection-data -p data --unzip
```

(Official docs: where to put `kaggle.json`, and the standard `kaggle datasets download … --unzip` command.) ([GitHub][3])

---

## 3) Get the **UCI MetroPT-3** dataset (big file)

**What:** download from UCI and place in `data/` locally. **Do not commit** it.

* UCI page (project description/files). Download the CSV(s) via browser and drop them into `data/`.
* We avoid pushing large CSVs because GitHub rejects files **>100 MiB** in normal repos; if we ever needed to track big assets, we’d use **Git LFS** instead. ([GitHub Docs][1])

---

## 4) Commit only code & instructions (never the data)

**What:** push your code, notebooks, and docs—**not** the datasets.

```bash
git add README.md scripts/  # your code and docs
git commit -m "docs: add dataset download steps; keep data/ ignored"
git push
```

**Why:** Everyone can reproduce by running the download commands; the repo stays fast, clean, and within GitHub’s limits. ([GitHub Docs][1])

---

# Drop-in README section (ready to paste)

````markdown
## Data (local only — not committed)

We **do not commit datasets**. The `data/` folder is git-ignored.

### A) Kaggle — Sensor Fault Detection
1) Install CLI and set API token:
   ```bash
   pip install kaggle
   # Get token: Kaggle Profile → Account → "Create API Token"
   # Windows token path: C:\Users\<YOU>\.kaggle\kaggle.json
````

2. Download:

   ```bash
   kaggle datasets download -d arashnic/sensor-fault-detection-data -p data --unzip
   ```

### B) UCI — MetroPT-3 (APU)

* Download from the UCI page and place files under `data/`.

> Note: GitHub **blocks files larger than 100 MiB** in normal repos.
> Keep datasets local. If we ever needed versioned large files, use Git LFS.

```

---

## Quick FAQ for the team
- **“I don’t see the data on GitHub after I push.”**  
  Correct—`data/` is ignored on purpose. Everyone downloads locally.

- **“Can I force-add a single file under `data/`?”**  
  Please don’t add datasets. If you really must, create a tiny **sample** CSV instead.

- **“What if someone already pushed a big file?”**  
  Remove it from tracking with `git rm --cached <file>` and re-push. (If it’s >100 MiB, GitHub will block it anyway.) :contentReference[oaicite:7]{index=7}

---

### Sources
- GitHub Docs — **Repository limits / Large files & LFS** (100 MiB limit; use LFS for big assets). :contentReference[oaicite:8]{index=8}  
- Kaggle API — **Credentials path & CLI usage** (`kaggle.json`, Windows path, `kaggle datasets download`). :contentReference[oaicite:9]{index=9}  
- Atlassian Git Tutorial — **.gitignore basics, untracking with `git rm --cached`, `!pattern` exceptions**. :contentReference[oaicite:10]{index=10}

If you want, I can also drop these steps into a PR that adds the README section and a `data/.gitkeep` placeholder.
::contentReference[oaicite:11]{index=11}
```

[1]: https://docs.github.com/enterprise-cloud%40latest/repositories/working-with-files/managing-large-files/about-large-files-on-github?utm_source=chatgpt.com "About large files on GitHub - GitHub Enterprise Cloud Docs"
[2]: https://www.atlassian.com/git/tutorials/saving-changes/gitignore?utm_source=chatgpt.com ".gitignore file - ignoring files in Git | Atlassian Git Tutorial"
[3]: https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md?utm_source=chatgpt.com "kaggle-api/docs/README.md at main · Kaggle/kaggle-api · GitHub"
