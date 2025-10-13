import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 如果你之前有处理脚本，可以复用它加载 train/test
from NASA_dataprocessing_improved import read_fd, read_rul, add_train_rul, build_test_labels, scale_by_train