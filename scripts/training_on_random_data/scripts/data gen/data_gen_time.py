import sys
sys.path.append("C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data")
from src import subgridmodel as sgm
import time

time_start = time.time()
data = sgm.generate_dataset(size=1, box_lenght=64)
time_end = time.time()
print(f"Time to gen data: {time_end - time_start}")

"""time_start = time.time()
classification = sgm.classify_dataset(dataset=data)
time_end = time.time()
print(f"Time to classify data: {time_end - time_start}")

classified_data = (data, classification)
"""