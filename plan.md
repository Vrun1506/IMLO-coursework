Currently, we don't have access to that many images, so thinking of modifying the image dataset via the Pillow library (it is part of the environment.yml file so potential), or if there's any potential with PyTorch itself. 

Need to plan out the layers for the neural network architecture and identifying the discernible features so we can work from there and figure out how to actually set up the neural network and the layers itself.

Plan for overfitting just in case. Have a visualisation via matplotlib mapping training loss and validation loss. 

I can't use the test data as validation because I'm gonna overfit the model otherwise. 

I need to keep the test data clean and separate so that the model has never seen said test data before and I can use it to evaluate our model performance once I'm happy with the training data testing. 

For this reason, I might have to split the trainval set into training and validation so I get an idea for how the model is performing and then only using the test data once we've established that we are happy with the model. 

In this situation, I'm definitely gonna need more images to validate the model and have a look at its accuracy. Still in the air about whether I should manipulate the images in the test dataset, or whether it's just good enough to do it for the training dataset. 

