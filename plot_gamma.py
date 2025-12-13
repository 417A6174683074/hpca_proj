import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_FILE = '/tmp/gamma.csv'

df = pd.read_csv(INPUT_FILE, index_col=None)

for alpha in df.columns:
    data = df[alpha]
    plt.hist(data, bins=100, alpha=0.5, label=f'alpha = {alpha}')

plt.ylim(0, 2000)
plt.legend()
plt.show()
