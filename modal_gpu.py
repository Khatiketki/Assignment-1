import modal

app = modal.App("gpu-hello")

# Small image: only installs tiny package
image = modal.Image.debian_slim(python_version="3.11").pip_install("platformdirs")


@app.function(image=image, gpu="any")  # request any available GPU
def gpu_info():
    import os
    print("Hello from a GPU container on Modal!")
    # This won't guarantee CUDA libs, but proves GPU scheduling works.
    print("Environment variables (sample):")
    for k in ["NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]:
        print(f"  {k} = {os.getenv(k)}")


if __name__ == "__main__":
    with app.run():
        gpu_info.remote()
