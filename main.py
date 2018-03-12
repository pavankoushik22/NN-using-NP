from numpy import exp, array, random, dot
#thats it , we got all we need

class NN():
    def __init__(self):
        random.seed(9)
        #for deterministic randomisation
        self.weights = 2*random.random((3,1)) - 1
        #a 3 by 1 matrix as out model is single layer with 3ip and 1op
        #and make ws to be in range 1 -1

    def __sigmoid(self, x):
        #squash ip value to range 0 to -1
        return 1/(1+exp(-x))
    
    def __sigmoid_derivative(self,x):
        return x*(1-x)

    def train(self, ips, ops, iterations):
        for iteration in range(iterations):
            op = self.mult(ips)
            err = ops - op
            #calc err between desired and obtained op
            adjustment = dot(ips.T, err*self.__sigmoid_derivative(op))
            #derivative gives the slope and slope tells the direction to jump so we take direction and err and add it to the present weights to reach lowest point if you can imagine the contour plots
            self.weights += adjustment

    def mult(self, ips):
        return self.__sigmoid(dot(ips, self.weights))


nn = NN()
print("initial weights are:")
print(nn.weights)
print("")
tips = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
tops = array([[0,1,1,0]]).T
nn.train(tips,tops, 10000)
print("updated weights are:")
print(nn.weights)
print("")
#testing
print(nn.mult(array([1,0,0])))

