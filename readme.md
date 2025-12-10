Some changes:
    -deleted the init_prob because it is more efficient to create the random variables within the global functions
    -Changed a bit the Euler function so that it can take multiple sets of parameters to test
    -The euler and almost functions are called on a number of blocks equal to the number of sets of parameters that we want to test.
    -in the question 3 code which is enabled for now, we are testing the two functions and the data are registered in a csv file.
    - I changed a bit the gamma function so that it can manage values of alpha <1.