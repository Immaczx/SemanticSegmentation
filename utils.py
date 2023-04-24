import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import shutil
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_data(path,size, scale = True):
  images = os.listdir(path)
  images.sort()

  X = []
  for i, img in enumerate(images):
      photo = plt.imread(os.path.join(path,img))
      if size:
          photo = tf.image.resize(photo, (size, size))
      X.append(photo)
      
  X = np.array(X)
  if scale:
      X = X/X.max() 
  return X  
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def MovImages(File_out, File_in, List):
  for img in List:
   shutil.move(File_in+img, File_out)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def parse_labelfile(path):
    """Return a dict with the corresponding rgb mask values of the labels
        Example:
        >>> labels = parse_labelfile("file/path")
        >>> print(labels) 
        >>> {"label1": (r1, g1, b1), "label2": (r2, g2, b2)} 
    """
    with open(path, "r") as FILE:
        lines = FILE.readlines()

    labels = {x.split(":")[0]: x.split(":")[1] for x in lines[1:]}

    for key in labels:
        labels[key] = np.array(labels[key].split(",")).astype("uint8")

    return labels
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def mask2categorical(Mask: tf.Tensor, labels: dict) -> tf.Tensor:
    """Pass a certain rgb mask (3-channels) to an image of ordinal classes"""
    assert type(labels) == dict, "labels variable should be a dictionary"

    X = Mask

    if X.dtype == "float32":
        X = tf.cast(X*255, dtype="uint8")

    Y = tf.zeros(X.shape[0:2] , dtype="float32")
    for i, key in enumerate(labels):
        Y = tf.where(np.all(X == labels[key], axis=-1), i, Y)
    Y = tf.cast(Y, dtype="uint8")
    return Y
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def transform_image(Y):
    Y = tf.one_hot(Y, depth=len(np.unique(Y).shape))
    return Y
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if not title:
      if normalize:
          title = 'Normalized confusion matrix'
      else:
          title = 'Confusion matrix, without normalization'

  # Compute confusion matrix
  cm = confusion_matrix(y_true, y_pred)
  # Only use the labels that appear in the data
  #classes = classes[unique_labels(y_true, y_pred)]
  if normalize:
      cm = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  
  
  fig, ax = plt.subplots()
  im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
  ax.figure.colorbar(im, ax=ax)
  # We want to show all ticks...
  ax.set(xticks=np.arange(cm.shape[1]),
          yticks=np.arange(cm.shape[0]),
          # ... and label them with the respective list entries
          xticklabels=classes, yticklabels=classes,
          title=title,
          ylabel='True label',
          xlabel='Predicted label')

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

  # Loop over data dimensions and create text annotations.
  fmt = '.1f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
          ax.text(j, i, format(cm[i, j], fmt),
                  ha="center", va="center",
                  color="white" if cm[i, j] > thresh else "black")
  fig.tight_layout()
  return ax

def dice_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection)/(union), axis=0)
    return dice
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def iou_coef(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection) / (union), axis=0)
    return iou