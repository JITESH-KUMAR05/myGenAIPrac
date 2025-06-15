
to start with adaboost we have 
step 1  create decision tree stump and select best of them with the help of entropy and gini impurity 
step 2 sum of total erros and performance of stump
    total error = total number of wrong records (sample weights)
    performance of stumps = 1/2 * ln[(1 - total error)/total error]
    what ever performance of stumps comes we will assign it to alpha 1 for model 1 as weight and then sequenciallt connect it to the next models and update the weight 
Step 3 Update weights for correctly and incorrectly classified points 
        for correctly classified weights we will decrease the weight and more incorrectly classified weights we will reduce the weights 
        for correctly classified weights = weight * e ^ -performance of stump
        fir incorrect classified points = weight * e ^ performance of stump
step 4 Normalised weight computation and assigning Bins
        the point is the total weights sum should come as 1 when we decrease the weights then the total sum will come as less than 1 so it is important for us to normalise them so that total sum will come as 1 as in starting the total sum was 1
        so first take the total sum and then divide each weight with that weight then the new sum will come near to 1 
        then we have to do bins assignment so that the next model will mostly take the data which was wrong 
        so the wrong predicted records will be having the maximum bin size so that the probability of getting that selected becomes high
        