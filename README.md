# Semantic-segmentation-teeth-image

** Contributor   
JongHwan Park : pjh403@naver.com   
   
   
* fcn result image
![fcn 2](https://user-images.githubusercontent.com/62584810/77441078-5dd3cf00-6e2c-11ea-90e6-5af725ff8375.png)

* deeplab result image
![deeplab 2](https://user-images.githubusercontent.com/62584810/77441085-5f04fc00-6e2c-11ea-99b8-c00866256233.png)

* U net result image
![U_net 2](https://user-images.githubusercontent.com/62584810/77441093-60362900-6e2c-11ea-985f-a4a2a9b4b7cf.png)

#### Using only 50 teeth image and 50 mask image, we can check process of training normally

#### There's three main files. Choose what use model you want

> main.py => Using U-net model

> fcn_main.py => Using fcn model

> deeplab_main.py => Using deeplab model

* each main files have 6 args parsers

> mode

> optim

> loss-function

> epochs

> tricks

> train/validation batches

### example
```
$ python main.py --epoch 30 loss-function bce
```
