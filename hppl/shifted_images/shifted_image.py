from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import timeit
from matplotlib.animation import FuncAnimation
import tracemalloc
from matplotlib import rc
from IPython.display import Javascript
import cv2

start = timeit.default_timer()
tracemalloc.start()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

img = plt.imread('task8.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print('img shape', img.shape)
my_n = int(img.shape[0]/size)


def show_img(pic, number = rank):

  plt.figure(figsize = (7,5))
  plt.xticks([])
  plt.yticks([])
  plt.imshow(pic, cmap='Purples')
  plt.savefig(f'{number} task8.png')

def change_pos(data):
  tem = data.copy()
  tem[:-1] = data[1:].copy()
  return tem

cur_range = img[rank*my_n:(rank+1)*my_n]
cur_range_ghost = np.zeros([cur_range.shape[0]+1, cur_range.shape[1]], dtype = np.int32)
ghost_total  = np.zeros([size, cur_range.shape[1]], dtype=np.int32)

print('cur',cur_range.shape)

print('gh',cur_range_ghost.shape)

cur_range_ghost[:-1] = cur_range.copy()

def get_ghost():

  comm.Send([cur_range_ghost[0], MPI.INT], dest=(rank +size -1)%size)

  comm.Recv([cur_range_ghost[-1], MPI.INT], source=(rank+1)%size)


def get_ghost_temp():

  comm.Send([temp[0], MPI.INT], dest=(rank +size -1)%size)

  comm.Recv([temp[-1], MPI.INT], source=(rank+1)%size)

t = []
temp = cur_range_ghost.copy()  

if size != 1:
  comm.Barrier()


  get_ghost()

  print('shape temp', cur_range_ghost.shape)

  for i in range(my_n): 
    temp = change_pos(temp)
    t.append(temp[:-1])
    comm.Barrier()
    get_ghost_temp()


else:
  for i in range(my_n): 
    temp = change_pos(temp)
    t.append(temp[:-1])

img_total = comm.gather(t, root = 0)

TIME = timeit.default_timer() - start

memory_gen = comm.reduce(tracemalloc.get_traced_memory()[1], op=MPI.SUM, root = 0)
tracemalloc.stop()

TIME_gen = comm.reduce(TIME, op=MPI.SUM, root = 0)
if rank == 0:
    print('final',TIME_gen/ size)
    print('memory',memory_gen/size/(1024**3))

if rank == 0: 

  img_total = np.array(img_total)
  
  print('afte vstack',img_total.shape)

  img_total = np.vstack(img_total.copy())
  print('afte vstack',img_total.shape)

  show_img(img_total[300], number=300)
