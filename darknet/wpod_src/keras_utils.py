
import numpy as np
import cv2
import time

from os.path import splitext

from wpod_src.label import Label
from wpod_src.utils import getWH, nms
from wpod_src.projection_utils import getRectPts, find_T_matrix


class DLabel (Label):

	def __init__(self,cl,pts,prob):
		self.pts = pts
		tl = np.amin(pts,1)
		br = np.amax(pts,1)
		Label.__init__(self,cl,tl,br,prob)

def save_model(model,path,verbose=0):
	path = splitext(path)[0]
	model_json = model.to_json()
	with open('%s.json' % path,'w') as json_file:
		json_file.write(model_json)
	model.save_weights('%s.h5' % path)
	if verbose: print ('Saved to %s' % path)

def load_model(path,custom_objects={},verbose=0):
	from keras.models import model_from_json

	path = splitext(path)[0]
	with open('%s.json' % path,'r') as json_file:
		model_json = json_file.read()
	model = model_from_json(model_json, custom_objects=custom_objects)
	model.load_weights('%s.h5' % path)
	if verbose: print ('Loaded from %s' % path)
	return model

def order_points(pts):
	# initialize a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = np.sum(pts,axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def reconstruct(Iorig,I,Y,out_size,threshold=.9):
  net_stride 	= 2**4
  side 		= ((208. + 40.)/2.)/net_stride # 7.75

  Probs = Y[...,0]
  Affines = Y[...,2:]
  rx,ry = Y.shape[:2]
  ywh = Y.shape[1::-1]
  iwh = np.array(I.shape[1::-1],dtype=float).reshape((2,1))

  xx,yy = np.where(Probs>threshold)

  WH = getWH(I.shape)
  MN = WH/net_stride

  vxx = vyy = 0.5 #alpha

  base = lambda vx,vy: np.matrix([[-vx,-vy,1.],[vx,-vy,1.],[vx,vy,1.],[-vx,vy,1.]]).T
  labels = []

  for i in range(len(xx)):
    y,x = xx[i],yy[i]
    affine = Affines[y,x]
    prob = Probs[y,x]

    mn = np.array([float(x) + .5,float(y) + .5])

    A = np.reshape(affine,(2,3))
    A[0,0] = max(A[0,0],0.)
    A[1,1] = max(A[1,1],0.)

    pts = np.array(A*base(vxx,vyy)) #*alpha
    # print(pts)
    pts_MN_center_mn = pts*side
    pts_MN = pts_MN_center_mn + mn.reshape((2,1))

    pts_prop = pts_MN/MN.reshape((2,1))

    labels.append(DLabel(0,pts_prop,prob))

  final_labels = nms(labels,.1)
  # print(final_labels)
  TLps = []
  points = []
  if len(final_labels):
    final_labels.sort(key=lambda x: x.prob(), reverse=True)
    for i,label in enumerate(final_labels):
      # t_ptsh 	= getRectPts(0,0,out_size[0],out_size[1])
      
      # ptsh 	= np.concatenate((label.pts*getWH(Iorig.shape).reshape((2,1)),np.ones((1,4))))
      # H 		= find_T_matrix(ptsh,t_ptsh)
      
      pts = label.pts*getWH(Iorig.shape).reshape((2,1))
      ptss = (pts[0][0], pts[1][0]), (pts[0][1], pts[1][1]), (pts[0][2], pts[1][2]), (pts[0][3], pts[1][3])
      # print(ptss)

      rect = order_points(ptss)
      points.append(rect)
      (tl, tr, br, bl) = rect
      # compute the width of the new image, which will be the
      # maximum distance between bottom-right and bottom-left
      # x-coordiates or the top-right and top-left x-coordinates
      widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
      widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
      maxWidth = max(int(widthA), int(widthB))
      # compute the height of the new image, which will be the
      # maximum distance between the top-right and bottom-right
      # y-coordinates or the top-left and bottom-left y-coordinates
      heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
      heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
      maxHeight = max(int(heightA), int(heightB))

      dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")
      # compute the perspective transform matrix and then apply it
      M = cv2.getPerspectiveTransform(rect, dst)

      Ilp 	= cv2.warpPerspective(Iorig,M,(maxWidth, maxHeight),borderValue=.0)

      TLps.append(Ilp)

  return final_labels,TLps,points
	
def detect_lp(model,I,max_dim,net_step,out_size,threshold):

  min_dim_img = min(I.shape[:2])
  factor 		= float(max_dim)/min_dim_img

  w,h = (np.array(I.shape[1::-1],dtype=float)*factor).astype(int).tolist()
  w += (w%net_step!=0)*(net_step - w%net_step)
  h += (h%net_step!=0)*(net_step - h%net_step)
  Iresized = cv2.resize(I,(w,h))

  T = Iresized.copy()
  T = T.reshape((1,T.shape[0],T.shape[1],T.shape[2]))

  start 	= time.time()
  Yr  = model.predict(T)
  Yr = np.squeeze(Yr)
  elapsed = time.time() - start

  L,TLps,points = reconstruct(I,Iresized,Yr,out_size,threshold) 

  return L,TLps,elapsed,points