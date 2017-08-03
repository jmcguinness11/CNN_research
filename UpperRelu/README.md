In this directory, I investigated the effects of using a new nonlinearity with
the potential to speed up neural network training process.  As the nonlinearity
is essentially RELU with an upper bound, I call it **"Upper ReLU"**.

What is Upper ReLU?
-----------------------
Upper ReLU is a nonlinearity that tries to squeeze values between -1 and 1.

The basic form of Upper ReLU is simple and exactly matches the standard nonlinearity
for [Cellular Neural Networks](http://www.scholarpedia.org/article/Cellular_neural_network).
<img src="(http://www.scholarpedia.org/w/images/c/c3/CNN_output.png"  
alt="CeNN Nonlinearity" width="50">
```
(∞, -1): y = -1 + ⍺(x+1)
[-1, 1]: y = x
(1, ∞):	 y = 1 - ⍺(x-1)
```

```
(∞, -1): y = -1 + ⍺(x+1)
[-1, 1]: y = x
(1, ∞):	 y = 1 - ⍺(x-1)
```


How did I find this nonlinearity?
-------------------------------------
DESCRIPTION OF THE PROCESS THAT LED ME TO UPPER RELU.

Initial Findings
----------------
After I figured out its potential, I ran some quick initial tests on Upper ReLU.
Those results are stored in the [InitialFindings](./InitialFindings) folder. 
