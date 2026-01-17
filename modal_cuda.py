import modal

app = modal.App("cuda-check")

# Image with CUDA tools; this is the typical pattern.
# We keep it minimal and just run `nvidia-smi` and `nvcc --version` checks.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("pciutils")  # lightweight; optional
)

@app.function(image=image, gpu="any")
def cuda_check():
    import subprocess

    print("Running CUDA checks...")

    # nvidia-smi is usually present on GPU containers; if not, we show the error
    for cmd in [["nvidia-smi"], ["bash", "-lc", "nvcc --version"]]:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            print(f"\n$ {' '.join(cmd)}\n{out}")
        except subprocess.CalledProcessError as e:
            print(f"\n$ {' '.join(cmd)}\nCommand failed:\n{e.output}")
        except FileNotFoundError:
            print(f"\n$ {' '.join(cmd)}\nNot found in this image/container.")

if __name__ == "__main__":
    with app.run():
        cuda_check.remote()
