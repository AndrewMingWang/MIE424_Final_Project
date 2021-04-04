import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

forgetting_events = torch.load("./forgetting_events.pt")
data = forgetting_events.cpu().numpy()

plt.hist(data, weights=np.ones(len(data)) / len(data))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()