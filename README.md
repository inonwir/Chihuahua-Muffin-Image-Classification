# Chihuahua-Muffin-Image-Classification
Machine Learning for image classification model

# How to run
1. Open chihuahua-or-muffin.py (in GoogleColab or Local Python environment)<br/>
2. Read the data from dataset folder, set the path for the dataser.<br/>
3. Run and see the result.<br/>

# How thing work?
1. Read data and preprocessing.<br/>
2. Generate more data, we flip images to have more image samples.<br/>
3. Apply VGG16 with imagenet dataset, pre-trained model. We fixed the first 15 layers and train the rest. Then, perform fine tuning to get result. Optimizer that we used is RMSprop.<br/>
4. Train the model and evaluate the result, the accuracy is 100%. <br/> 
