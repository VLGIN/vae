import re

from decouple import config

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

    return {"current_best_loss": best_loss, "number_from_improvement": last_epoch - best_epoch, "current_epoch": last_epoch}
    