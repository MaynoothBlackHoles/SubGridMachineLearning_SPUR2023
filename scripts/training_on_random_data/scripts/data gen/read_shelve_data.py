import shelve

BL = 64
data = shelve.open(f'C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/fast {BL} cubed/fast_{BL}cubed.db')
print(data['training info'])
print(data['validation info'])
data.close()