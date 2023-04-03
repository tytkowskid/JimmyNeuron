import matplotlib.pyplot as plt
import math
import numpy as np
import networkx as nx

def comparison_plot(x, y, y_nn):
    plt.scatter(x,y)
    plt.scatter(x, y_nn)
    plt.show()


def create_wb(layers):
    """ 
    Creates dense layers of weights and biases using Xavier method.
    
    Takes list of number of neurons in each layer. 
    """

    w = []
    b = []

    for i in range(1,len(layers)):
        in_neurons = layers[i-1]
        out_neurons = layers[i]
        bound = math.sqrt(6)/math.sqrt(in_neurons + out_neurons)
        w.append(np.random.uniform(low = -bound, high = bound, size=(in_neurons,out_neurons)))
        #w.append(np.zeros(shape=(in_neurons,out_neurons)))
        b.append(np.zeros(shape=(1,out_neurons)))

    return w,b


def MSE(y, y_nn):
    return  sum((y.flatten()-y_nn.flatten())**2)/len(y.flatten())


class Network:
    def __init__(self, layers):
        w, b = create_wb(layers)
        self.w = w
        self.b = b
        self.layers = layers

    def calculate(self, x):

        no_layers = len(self.layers)

        temp = np.dot(x, self.w[0]) + self.b[0]
        temp = sigmoid(temp)

        for i in range(1,no_layers-2):
            temp = np.dot(temp, self.w[i]) + self.b[i]
            temp = sigmoid(temp)

        temp = np.dot(temp, self.w[no_layers-2]) + self.b[no_layers-2]
        return temp


    def draw(self):

        print("es")

        G = nx.Graph()

        for i in range(len(self.b[0][0])):
            G.add_edge("x", str(i), weight = self.w[0][0][i])
            G.add_edge(str(i), "z", weight = self.w[1][i][0])


        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

        pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=200)

        # edges
        nx.draw_networkx_edges(G, pos, edgelist=elarge, width=3)
        nx.draw_networkx_edges(
            G, pos, edgelist=esmall, width=3, alpha=0.5, edge_color="b", style="dashed"
        )

        # node labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

def two_dim_avg(A,B):
    #print(A.shape, B.shape)

    N = len(A)
    Sum = np.array([[0 for y in range(B.shape[1])] for x in range(A.shape[1])])

    for j in range(N):
        a = A[j]
        a.shape = (A.shape[1], 1)
        b = B[j]
        b.shape = (1, B.shape[1])
        #print(np.dot(a,b)+Sum)
        Sum = np.dot(a,b)+Sum
    
    Sum = np.array(Sum/N)
    Sum.shape = (A.shape[1], B.shape[1])
    return Sum

def softmax(x):
    s = math.e**x
    sums = np.apply_along_axis(sum, 1, s)
    sums.shape = (len(sums),1)
    return s/sums

def f1(y, y_nn, k):
    label_f1 = []

    for label in range(k):
        real = np.where(y==label, True, False)
        pred = np.where(y_nn==label, True, False)

        tp = sum(np.bitwise_and(real, pred))
        fp = sum(np.bitwise_and(np.bitwise_not(real), pred))
        fn = sum(np.bitwise_and(np.bitwise_not(pred), real))

        f1 = (2*tp)/((2*tp)+fp+fn)

        label_f1.append(f1)

    return label_f1


def sigmoid(x):
    return 1/(1+math.e**(-x))

def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


