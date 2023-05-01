import psutil

def memory_usage(message):
    p = psutil.Process()
    rss = p.memory_info().rss/2**20
    print(f"[{message}] memory usage: {rss:10.5f} MB")

memory_usage("#1")

print("A")


memory_usage("#2")