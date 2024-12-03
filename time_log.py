import time

from collections import defaultdict

steps = defaultdict(lambda: {"time": 0, "start": None, "counter": 0})


def time_step(step):
    return
    if steps[step]["start"]:
        steps[step]["time"] += time.time() - steps[step]["start"]
        steps[step]["counter"] += 1
        steps[step]["start"] = None
        print(
            f"Step '{step}' ran {steps[step]['counter']} times. in average took {steps[step]['time']/steps[step]['counter']}s to run."
        )
    else:
        steps[step]["start"] = time.time()
