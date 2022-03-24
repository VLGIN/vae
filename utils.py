import re
import os
import pickle
import numpy as np

from decouple import config
from matplotlib import pyplot as plt

def process_log():
    log_file = config('LOG_FILE')

    with open(log_file) as f:
        logs = f.read().splitlines()
    last_epoch = -1
    last_logs = 0
    evaluation = []
    for i in range(len(logs)):
        if len(re.findall("EPOCH .+: loss", logs[i])) > 0:
            evaluation.append(float(logs[i].split(" ")[-1]))
            last_epoch += 1
            last_logs = i
    
    logs = logs[:last_logs + 1]

    best_loss = min(evaluation)
    best_epoch = evaluation.index(best_loss)
    with open(log_file, "w") as f:
        for item in logs:
            f.write(item + "\n")

    return {"current_best_loss": best_loss, "number_from_improvement": last_epoch - best_epoch, "current_epoch": last_epoch}

def read_data_from_disk(path):
    files = os.listdir(path)
    data = []

    for file in files:
        file_path = os.path.join(path, file)
        with open(file_path, "rb") as f:
            data.append(pickle.load(f, encoding="bytes")[b'data'].astype(float))
    
    return np.concatenate(data, axis=0)
    
def visualize_image(data):
    num_img = data.shape[0]

    fig = plt.figure(figsize=(20,20))
    rows = 4
    columns = 4
    for i in range(num_img):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(data[i])
        plt.axis("off")
    fig.savefig("infer.png")
    