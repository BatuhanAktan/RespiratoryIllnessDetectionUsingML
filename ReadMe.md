# Diagnosis of respiratory illnesses.

In the current state of healthcare many hospitals are working over their capacity, and our group wanted to help fix this issue, we used machine learning to diagnose a healthy lung X-ray from a sick one.


### Getting Started

We used the public data provided with more than 4000 chest cts to develop and form a sequential model with tensorflow. We used 64 cts as our sample 32 infected and 32 healthy, with this model after 10 Epochs we were able to hit 100% accuracy.

### Prerequisites

All the libraries used are in requirements.txt, and they can be installed using pip command.

Example.
```
pip install numpy
```

### Installing

The machine learning model can be installed in the following google drive

```
https://drive.google.com/file/d/1r0XxZCt8s_TWgDvd6Kg_ObjuLs6C2iTJ/view
```

## Running the tests

To run these tests we used the model mentioned above and with to test various X-rays and we achieved a total of 100% accuracy in the prediction of 10 X-rays.

### Results

```
57/57 [==============================] - 40s 698ms/sample - loss: 1.1120e-05 - accuracy: 1.0000 - val_loss: 0.0429 - val_accuracy: 1.0000
```
After training our machine learning model with 64 X-ray images we were able to get 100% accuracy in the determination of respiratory illnesses and we expect this result to be close to 90% with X-rays that are out of our training set.
|![Healthy Lung Image](https://github.com/BatuhanAktan/executehacks/blob/main/TestImages/IM-0001-0001.jpeg?raw=true)|![Sick Lung Image](https://github.com/BatuhanAktan/executehacks/blob/main/TestImages/person15_virus_46.jpeg?raw=true)|
|:---:|:---:|
|Healthy Lung: IM-0001-0001.jpeg|Sick Lung: person15_virus_46.jpeg|

|![Results](https://github.com/BatuhanAktan/executehacks/blob/main/TestImages/Results/Results.png?raw=true)|

```
Image being processed: IM-0001-0001.jpeg
Expected Result: Healthy 
Result: Healthy
Image being processed: person15_virus_46.jpeg
Expected Result: Sick 
Result: Sick
```


## Authors

* **Batuhan Aktan** - [Github](https://github.com/BatuhanAktan)
* **Jody Zhou** - [Github](https://github.com/JodyZ0203)
* **Sam Zhang** - [Github](https://github.com/Dam-Sam)

## License

This project is licensed under the MIT License.

## Acknowledgments

* Shoutout to Kevin and Alvin

