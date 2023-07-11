import shelve
import time
import sys
sys.path.append("C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data")
from src import subgridmodel as sgm

BL = 64
data = shelve.open(f'C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/fast {BL} cubed/fast_{BL}cubed.db')

print("Generating Datasets")

time_start = time.time()

training_size = 1000
data['training'] = sgm.gen_fast_classified_data(size=training_size, box_lenght=BL)
star_forming = sgm.star_forming_ratio(data['training'])
data['training info'] = {"size": training_size, "box lenght": BL, "ratio": star_forming}
print(f"training data properties: {data['training info']}")

time_end = time.time()
print(f"Generated in {(time_end - time_start)/60} mins") 
time_start = time.time()    

validation_size = 100
data['validation'] = sgm.gen_fast_classified_data(size=validation_size, box_lenght=BL)
star_forming = sgm.star_forming_ratio(data['validation'])
data['validation info'] = {"size": validation_size,  "box lenght": BL, "ratio": star_forming}
print(f"validation data properties: {data['validation info']}")

time_end = time.time()
print(f"Generated in {(time_end - time_start)/60} mins")