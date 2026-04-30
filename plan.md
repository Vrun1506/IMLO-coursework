Need to plan out the layers for the neural network architecture and identifying the discernible features so we can work from there and figure out how to actually set up the neural network and the layers itself.

Plan a four layer implementation approach based off the PyTorch docs.

Start with a smaller filter like 3x3 and observe what happens with our accuracy. 

I'm going to start the convolutional layer at 32 filters and then increase it by doubling the number of filters in each convolutional layer going forwards to 64, 128, 256

Going to use ReLU as the activation function to reset negatives to 0 after each convolutional layer. 

Look into how to implement dropout and whether we need to change anything in our code to be able to implement it. 

PyTorch docs uses SGD optimiser, but I'm going to use Adam for the interest of training time, and the fact that it adapts the learning rate better than SGD. 

Guest lecture talked about batch normalisation and standardisation to minimise overfitting and accelerate model training.

Need to work out how I'm going to work out the mean and standard deviation with tensor data because it's multi-dimensional, so I need to either get them all into a single tensor and then work it out, but there's way too many images for this to be viable. 

See if there's a PyTorch function to work out the mean like there is in Pandas dataframes and iterate through the set to work out the average. 

Look through forums as probs a common thing. 



