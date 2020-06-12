import csv

without_sm  = 'C:\\Users\\aless\\Desktop\\hamiltonian-network\\csv\\without_sm.csv'
with open(without_sm, newline='') as f:
    reader = csv.reader(f)
    no_sm = list(reader)[0]
    no_sm = [float(x) for x in no_sm]

with_sm  = 'C:\\Users\\aless\\Desktop\\hamiltonian-network\\csv\\with_sm.csv'
with open(without_sm, newline='') as f:
    reader = csv.reader(f)
    sm = list(reader)[0]
    sm = [float(x) for x in sm]