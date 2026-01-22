# Goals

The intended modifications are to add more datasets to this existing repo, following the same general conventions.

# Setup

We have modifed the environment now to use uv as the package manager. 

After uv is installed, run the following from this directory:

```
uv venv -p 3.9
uv sync
source .venv bin activate
```

# Downloading Weights

From the project root, run the following to download the weights from hugging face.
The weights specified by the original repo aren't available anymore.

```
bash ./pretrained/download_pretrained.sh
```
