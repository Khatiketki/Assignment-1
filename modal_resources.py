import modal
import time

app = modal.App("resources-demo")

image = modal.Image.debian_slim(python_version="3.11")

# Configure resources: keep it small to avoid burning credits
@app.function(
    image=image,
    cpu=1,              # request 1 CPU
    memory=512,         # 512 MiB RAM
    timeout=60,         # 60 seconds max
)
def resource_demo():
    print("Resources demo running.")
    print("Doing a tiny bit of work...")
    t0 = time.time()
    s = 0
    for i in range(5_000_00):  # small loop
        s += i % 7
    print("Done. checksum =", s, "elapsed =", round(time.time() - t0, 3), "sec")

if __name__ == "__main__":
    with app.run():
        resource_demo.remote()
