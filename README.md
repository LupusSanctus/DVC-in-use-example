# DVC-in-use-example
Run inference.py script to test **resnext101_32x4d**.
Optional key: 
```
--path_img: path to the image to be classified.
            data/cat.jpg will be set by default.  
```
Run unit tests (described in utest_inference.py) to test classifier results in the following way:
```
python -m unittest test.utest_inference -v
```
