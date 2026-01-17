import modal

# Define a custom Modal image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")        # install additional system packages
    .pip_install("numpy")      # example: add a pip package
    .run_commands(
        "echo 'This is running inside the Modal Image!'"
    )
)

app = modal.App("modal-images-example")

# Define a function that uses that image
@app.function(image=image)
def use_image():
    import numpy as np
    print("NumPy inside Modal:", np.__version__)
    print("Hello from Modal Image!")

if __name__ == "__main__":
    with app.run():
        use_image.remote()
