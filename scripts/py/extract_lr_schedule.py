
import json
import sys

def get_lr_sched(filename):
    schedule = {}
    first = True
    with open(filename) as f:
        for line in f:
            if "Setting learning rate" not in line:
                continue

            vals = line.split(' ')
            lr = float(vals[4])
            num_backprops  = vals[6]
            if first:
                schedule["0"] = lr
                first = False
            schedule[num_backprops] = lr
    outfile = "{}.lr".format(filename)
    with open(outfile, "w+") as f:
        json.dump(schedule, f)

if __name__ == "__main__":
    filename = sys.argv[1]
    get_lr_sched(filename)
