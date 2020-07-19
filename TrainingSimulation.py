print('Setting Up...')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
from sklearn.model_selection import train_test_split


#### STEP 1 Importing Data Information
path = 'dataset'
data = importDataInfo(path)

#### STEP 2 Visualizing Data to be balanced
data = balancedData(data, display = False)

### STEP 3 Preprocessing to put all steering values to one list
### and images to another
imagesPath, steerings = loadData(path,data)
#print(imagesPath[0],steering[0])

### STEP 4 Splliting of data into training (80%) and validation (20%)
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

### STEP 5 Augmenting Data

### STEP 6 Preprocessing (Batch Generation)

### STEP 7

### STEP 8 model
model =createModel()
model.summary()

### STEP 9 Training Model
history = model.fit(batchGen(xTrain,yTrain,100,1),steps_per_epoch = 300,epochs=10,
        validation_data = batchGen(xVal,yVal,100,0),validation_steps=200)

### STEP 10 Saving and Plotting Model
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
