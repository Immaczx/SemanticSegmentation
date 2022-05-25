from unicodedata import name
import utils
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold
from keras import backend as K
from model import get_model
from sklearn.metrics import classification_report
from tabulate import tabulate
#https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def jaccard_index(A: np.array, B: np.array, labels: dict) -> list:
    """IoU(A, B) = (A & B) / (A U B)"""
    assert A.shape == B.shape

    label = [i for i, key in enumerate(labels)]
    IoU = []

    for x in label:
        Inter = np.sum((A == x) & (B == x))
        Union = np.sum((A == x) | (B == x))
        if (np.sum(A==x) == 0 and np.sum(B==x) == 0):
            IoU.append(1)
        else:
            IoU.append(Inter/Union)
    return IoU
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def dice_index(A: np.array, B: np.array, labels: dict) -> list:
    """Dice(A, B) = 2*(A & B) / (|A| + |B|)"""

    label = [i for i, key in enumerate(labels)]
    dice = []

    for x in label:
        Inter = np.sum((A == x) & (B == x))
        Den = (np.sum(A == x) + np.sum(B == x))
        if (np.sum(A==x) == 0 and np.sum(B==x) == 0):
            dice.append(1)
        else:
            dice.append(2 * Inter / Den)
    return dice
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_model_name(k): 
  return 'model_'+str(k)+'.h5'
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def train(path_trainDataset, path_trainDatasetM, path_testDataset, path_testDatasetM, labels, image_size = 224, classes = 3, epochs = 10, nFolds = 5, modelName = "U-net", backBone = "Movilnetv2"):
    save_dir = '/saved_models/'
    fold_var = 1
    # Load train dataset
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    X = utils.load_data(path_trainDataset, size=image_size)
    mask = utils.load_data(path_trainDatasetM, size=image_size)
    Y = np.zeros((mask.shape))
    for img in range(len(Y)):
        p = utils.mask2categorical(mask[img], labels)
        Y[img] = tf.one_hot(p, depth=3)
        #Y = np.expand_dims(Y,axis=-1)
    # Load test dataset
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    X_test = utils.load_data(path_testDataset, size=image_size)
    mask = utils.load_data(path_testDatasetM, size=image_size)
    Y_test = np.zeros((mask.shape[:3]))
    for img in range(len(Y_test)):
        Y_test[img] = utils.mask2categorical(mask[img], labels)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create the results directory
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if  not 'results' in os.listdir():
        os.mkdir('results')
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    kf = KFold(n_splits = nFolds)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    for train_index, val_index in kf.split(X):

        # Load train dataset
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        training_data = np.array([X[i] for i in train_index])
        training_mask = np.array([Y[i] for i in train_index])
        # Load valid dataset
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        validation_data = np.array([X[i] for i in val_index])
        validation_mask = np.array([Y[i] for i in val_index])
        # Load model
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        model = get_model(output_channels = classes, size = image_size, name = modelName, backBone = backBone)
        model.compile(
                optimizer = tf.keras.optimizers.Adam(),
                metrics = [dice_coef, iou_coef],
                loss = tf.keras.losses.CategoricalCrossentropy()
                )
        # CREATE CALLBACKS
        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var), 
                                monitor='val_accuracy', verbose=1, 
                                save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        # Train the model
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        model_history = model.fit(
                                x = training_data,
                                y = training_mask,
                                validation_data = (validation_data,validation_mask),
                                callbacks = callbacks_list,
                                epochs = epochs
                                )

        # Create the history figure
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        plt.figure(figsize=(16,9))
        for i in model_history.history:
                plt.plot(model_history.history[i],label=i)
        plt.title('Model history')
        plt.legend()
        plt.grid()
        # Save the figure History
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        i = 0
        flag = True
        while(flag==True):
                if (f'history{i}' in os.listdir('results')):
                        i+=1
                else:
                        plt.savefig(f'results/history{i}')
                        flag=False
        plt.show()
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Y_pred = model.predict(X_test)
        Y_pred = np.argmax(Y_pred, axis = 3)
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        utils.plot_confusion_matrix(Y_test.reshape(-1), Y_pred.reshape(-1), classes=['Background','germinated','no_germinated'])
        plt.show()
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print(classification_report(Y_test.reshape(-1), Y_pred.reshape(-1), target_names=['Background','germinated','no_germinated']))
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        dices, jaccards = [], []
        for i in range(Y_test.shape[0]):
            jaccards.append(tuple(jaccard_index(Y_test[i], Y_pred[i], labels))) 
            dices.append(tuple(dice_index(Y_test[i], Y_pred[i], labels)))
            jaccards=np.array(jaccards)
            dices=np.array(dices)
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        table = {"Segmentation":['Background','germinated','no_germinated'],
                "Dice": [f"{np.round(100*np.mean(dices[:,0]),2)} ± {np.round(100*np.std(dices[:,0]),2)}", f"{np.round(100*np.mean(dices[:,1]),2)} ± {np.round(100*np.std(dices[:,1]),2)}", f"{np.round(100*np.mean(dices[:,2]),2)} ± {np.round(100*np.std(dices[:,2]),2)}"],
                "Jaccard (IoU)": [f"{np.round(100*np.mean(jaccards[:,0]),2)} ± {np.round(100*np.std(jaccards[:,0]),2)}", f"{np.round(100*np.mean(jaccards[:,1]),2)} ± {np.round(100*np.std(jaccards[:,1]),2)}", f"{np.round(100*np.mean(jaccards[:,2]),2)} ± {np.round(100*np.std(jaccards[:,2]),2)}"]}
        print(tabulate(table, headers="keys",tablefmt='fancy_grid'))
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        tf.keras.backend.clear_session()
        fold_var += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    params = subparser.add_parser('params')
    params.add_argument('--trainI', type=str, required=True,
                        help="Path to the train images")
    params.add_argument('--trainM', type=str, required=True,
                        help="Path to the train masks")
    params.add_argument('--testI', type=str, required=True,
                        help="Path to the test images")
    params.add_argument('--testM', type=str, required=True,
                        help="Path to the test masks")
    params.add_argument('--labels', type=str, required=True,
                        help="Path txt labels")
    params.add_argument('--imageSize', type=int, required=False,
                        help="Image Size")
    params.add_argument('--classes', type=int, required=False,
                        help="Number of Clases")
    params.add_argument('--epochs', type=int, required=False,
                        help="Epochs")
    params.add_argument('--folds', type=int, required=False,
                        help="Number of folds for train") 
    params.add_argument('--model', type=str, required=False,
                        help="Model to use")
    params.add_argument('--backBone', type=int, required=False,
                        help="backBone")

    arguments = parser.parse_args()

    if arguments.command == "params":

        train(path_trainDataset = arguments.trainI, path_trainDatasetM = arguments.trainI, 
        path_testDataset = arguments.testI, path_testDatasetM = arguments.TestM, labels = arguments.labels, 
        image_size = arguments.imageSize, classes = arguments.classes, epochs = arguments.epochs,
        nFolds = arguments.folds, modelName = arguments.model, backBone = arguments.backBone)