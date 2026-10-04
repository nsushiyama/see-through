# See-through Gradio demo (fork)

`app.py` is derived from the official Hugging Face Space
[`24yearsold/see-through-demo`](https://huggingface.co/spaces/24yearsold/see-through-demo)
(Space license: MIT; the underlying code is https://github.com/shitagaki-lab/see-through, Apache-2.0).
`app_space_original.py` is the unmodified upstream Space file kept for diffing.

## Launch (local, CUDA GPU)
```
pip install -r requirements.txt -r demo/requirements.txt gradio
python demo/app.py
```
`spaces` (ZeroGPU decorator) is a no-op outside HF Spaces.

## Deploy as a Space
Copy the repo root to a Space, set `app_file: demo/app.py` in the Space README frontmatter
(or move `demo/app.py` to the root and change `_root` back to `os.path.dirname(__file__)`),
and use `demo/requirements.txt`.
