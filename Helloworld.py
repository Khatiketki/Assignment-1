import modal

app = modal.App("hello-world")

@app.function()
def hello():
    print("Hello from Modal!")

if __name__ == "__main__":
    with app.run():
        hello.remote()
