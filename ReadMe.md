# Diagnosis of respiratory illnesses.

In the current state of healthcare many hospitals are working over their capacity, and our group wanted to help fix this issue, we used machine learning to diagnose a healthy lung X-ray from a sick one.


### Getting Started

We used the public data provided with more than 4000 chest cts to develop and form a sequential model with tensorflow. We used 64 cts as our sample 32 infected and 32 healthy, with this model after 10 Epochs we were able to hit 100% accuracy.

### Prerequisites

All the libraries used are in requirements.txt

```
pip install numpy
```

### Installing

The model program can be installed in the following google drive

```
https://drive.google.com/file/d/1r0XxZCt8s_TWgDvd6Kg_ObjuLs6C2iTJ/view
```

## Running the tests

After downloading the file above, run it.

### Results

```
57/57 [==============================] - 40s 698ms/sample - loss: 1.1120e-05 - accuracy: 1.0000 - val_loss: 0.0429 - val_accuracy: 1.0000
```
We got 1.0000 accuracy with our model, which means it is 100% accurate at detecting lungs with illness.

## Authors

* **Batuhan Aktan** - [Github](https://github.com/BatuhanAktan)
* **Jody Zhou** - [Github](https://github.com/JodyZ0203)
* **Sam Zhang** - [Github](https://github.com/Dam-Sam)

## License

This project is licensed under the MIT License.

## Acknowledgments

* Shoutout to Kevin and Alvin

