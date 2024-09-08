import math


# Example log values for different confidences.
# Loss is larger the lower the confidence is.
# If confidence level = 1 then the model is 100% "sure" about its prediction. 
# Therefore the loss value will be 0.

# The loss value decreases as the model's confidence in the correct class increases.
print(math.log(1.)) # 0.0
print(math.log(0.95)) # -0.05129329438755058
print(math.log(0.9)) # -0.10536051565782628
print(math.log(0.8)) # -0.2231435513142097
print("...")
print(math.log(0.2)) # -1.6094379124341003
print(math.log(0.1)) # -2.3025850929940455
print(math.log(0.05)) # -2.995732273553991
print(math.log(0.01)) # -4.605170185988091




