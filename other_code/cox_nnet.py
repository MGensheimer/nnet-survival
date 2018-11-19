#Originally by Travers Ching. From: https://github.com/traversc/cox-nnet/
#converted from Python 2 to Python 3 compatibility by Michael Gensheimer

#This is a package for running the neural network extension of Cox regression.  
#Classes: 
#   CoxRegression - the Cox regression output layer
#   HiddenLayer - hidden layers between input and output, based on deeplearning.net/tutorial/code/mlp.py
#   CoxMlp - container class for output layer and hidden layers

#Functions:
#   createSharedDataset - helper function to create shared dataset in theano
#   trainCoxMlp - main function for training a cox-nnet model
#   predictNewData - function for predicting new data
#   L2CVSearch - helper function for performing cross-validation on a training set, to select optimal L2 regularization parameter
#   CVLoglikelihood - calculates the cross validation likelihood (using method from Houwelingen et al. 2005)
#   varImportance - calculates variable importance (using method from Fischer 2015)
#   saveModel - saves a model to a file: saveModel(model, file_name)
#   loadModel - loads model from file: loadModel(fileName)



import time
import numpy
import theano
import random
import theano.tensor as T
from sklearn import cross_validation
import _pickle as cPickle

theano.config.openmp=True
   
def mTest():
    M=2000
    N=2000
    K=2000
    order = "C"

    a = theano.shared(numpy.ones((M, N), dtype=theano.config.floatX))
    b = theano.shared(numpy.ones((N, K), dtype=theano.config.floatX))
    c = theano.shared(numpy.ones((M, K), dtype=theano.config.floatX))
    f = theano.function([], updates=[(c, 0.4 * c + .8 * T.dot(a, b))])

    for i in range(10000):
        f()
   
def createSharedDataset(data, borrow=True, cast_int=False):
  shared_data = theano.shared(numpy.asarray(data,dtype=theano.config.floatX),borrow=borrow)
  if cast_int:
    return T.cast(shared_data, 'int32')
  else:
    return shared_data
        

class CoxRegression(object):
    def __init__(self, input, n_in):
        self.W = theano.shared(value=numpy.zeros((n_in,1),dtype=theano.config.floatX), name='W_cox',borrow=True)
        # b_values = numpy.zeros((1,), dtype=theano.config.floatX)
        # self.b = theano.shared(value=b_values, name='b_cox', borrow=True) #intercept term is unnecessary
        
        self.input = input[0] if len(input) == 1 else T.concatenate(input, axis=1)
        #self.input = input

        self.theta = T.dot(self.input, self.W) # + self.b
        self.theta = T.reshape(self.theta, newshape=[T.shape(self.theta)[0]]) #recast theta as vector
        self.exp_theta = T.exp(self.theta)
        self.params = [self.W] #, self.b]

    def negative_log_likelihood(self, R_batch, ystatus_batch):
        return(-T.mean((self.theta - T.log(T.sum(self.exp_theta * R_batch,axis=1))) * ystatus_batch)) #exp_theta * R_batch ~ sum the exp_thetas of the patients with greater time e.g., R(t)
        #e.g., all columns of product will have same value or zero, then do a rowSum
    
    def evalNewData(self, test_data):
        return(T.dot(T.concatenate(test_data, axis=1), self.W)) # + self.b)
        
#This hidden layer class code is adapted from the multilayer perceptron HL class on deeplearning.net
class HiddenLayer(object):
    def __init__(self, rng, input, n_samples, map, label, activation=T.tanh):
        W = [0] * len(map)
        b = [0] * len(map)
        input = numpy.asarray(input)
        input_cat = [0] * len(map)
        for i in range(len(map)):
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (map[i][0] + map[i][2])),
                    high=numpy.sqrt(6. / (map[i][0] + map[i][2])),
                    size=(int(map[i][0]), int(map[i][2]))
                ),
                dtype=theano.config.floatX
            )
            W[i] = theano.shared(value=W_values, name='W_' + str(label) + '_' + str(i), borrow=True)
            b_values = numpy.zeros((map[i][2],), dtype=theano.config.floatX)
            b[i] = theano.shared(value=b_values, name='b_' + str(label) + '_' + str(i), borrow=True)
            input_cat[i] = input[map[i][1][0]] if len(map[i][1]) == 1 else T.concatenate(input[map[i][1]].tolist(), axis=1)
        
        self.W = W
        self.b = b
        self.map = map
        self.activation = activation
        self.input = input
        self.input_cat = input_cat
        
        output = [0] * len(self.map)
        for i in range(len(self.map)):
            output[i] = self.activation(T.dot(self.input_cat[i], self.W[i]) + self.b[i])
            
        self.output = output
        
        # def nesterovOutput(self, new_W, new_b):
            # output = [0] * len(self.map)
            # for i in range(len(self.map)):
                # output[i] = self.activation(T.dot(self.input_cat[i], new_W[i]) + new_b[i])
            # return(output)
        
        
        self.params = [self.W, self.b]
        
    def evalNewData(self, test_data):
        test_data = numpy.asarray(test_data)
        output = [0] * len(self.W)
        for i in range(len(self.W)):
            input_cat_i = test_data[self.map[i][1][0]] if len(self.map[i][1]) == 1 else T.concatenate(test_data[self.map[i][1]].tolist(), axis=1)
            output[i] = self.activation(T.dot(input_cat_i, self.W[i]) + self.b[i])
            
        return(output)

        

class CoxMlp(object):
    def __init__(self, x_train, rng, n_samples, node_map, input_split):

        if input_split == None:
            self.input = [createSharedDataset(x_train)]
        else:
            self.input = [0] * len(input_split)
            for i in range(len(input_split)):
                self.input[i] = createSharedDataset(x_train[:,input_split[i]])
        
        if node_map == None:
            self.node_map = [[(x_train.shape[1],[0],int(numpy.ceil(x_train.shape[1] ** 0.5)))]]
        else:
            self.node_map = node_map
            
        
        self.input_split = input_split

        self.n_samples = n_samples
        self.rng = rng
        self.x_train = x_train
        
        self.hidden_list = []
        self.W = []
        self.b = []

        for i in range(len(self.node_map)):
            hidden_layer = HiddenLayer(
            rng=self.rng,
            input=self.input if i == 0 else self.hidden_list[i-1].output,
            n_samples = self.n_samples,
            map=self.node_map[i],
            label=str(i),
            activation=T.tanh
            )
            self.hidden_list.append(hidden_layer)
            self.W.extend(hidden_layer.W)
            self.b.extend(hidden_layer.b)
    
        cox_in = 0
        for i in range(len(self.node_map[-1])):
            cox_in += self.node_map[-1][i][2]
            
        self.cox_regression = CoxRegression(
           input=self.hidden_list[-1].output, #last element in hidden_list
           n_in=cox_in
        )
        self.W.append(self.cox_regression.W)
        # self.b.append(self.cox_regression.b)

        #self.L2_sqr = T.sum(T.pow(self.W[0],2))
        self.L2_sqr = 0
        for i in range(len(self.W)):
            self.L2_sqr = self.L2_sqr + T.sum(T.pow(self.W[i],2))
            #self.L2_sqr = self.L2_sqr + pow(self.W[i], 2).sum()
        
        self.negative_log_likelihood = self.cox_regression.negative_log_likelihood
        self.input = input
        
        self.params = self.W + self.b
        
    def predictNewData(self, x_test):
        if self.input_split == None:
            test_input = [createSharedDataset(x_test)]
        else:
            test_input = [0] * len(self.input_split)
            for i in range(len(self.input_split)):
                test_input[i] = createSharedDataset(x_test[:,self.input_split[i]])

        theta = test_input
        for i in range(len(self.hidden_list)):
            theta = self.hidden_list[i].evalNewData(theta)

        theta = self.cox_regression.evalNewData(theta).eval()
        return(theta[:,0])

def simpleNetArch(x_train, n_nodes):
            self.node_map = [[(x_train.shape[1],[0],n_nodes)]]

    
def predictNewData(model, x_test):
    if model.input_split == None:
        test_input = [createSharedDataset(x_test)]
    else:
        test_input = [0] * len(model.input_split)
        for i in range(len(model.input_split)):
            test_input[i] = createSharedDataset(x_test[:,model.input_split[i]])

    theta = test_input
    for i in range(len(model.hidden_list)):
        theta = model.hidden_list[i].evalNewData(theta)

    theta = model.cox_regression.evalNewData(theta).eval()
    return(theta[:,0])
    
    
def defineModelParams(model_params):
    L2_reg = model_params['L2_reg'] if "L2_reg" in model_params else numpy.exp(-1)
    node_map = model_params['node_map'] if "node_map" in model_params else None
    input_split = model_params['input_split'] if "input_split" in model_params else None
    return(L2_reg, node_map, input_split)

def defineSearchParams(search_params):
    method = search_params['method'] if "method" in search_params else "nesterov"
    learning_rate = createSharedDataset(float(search_params['learning_rate'])) if "learning_rate" in search_params else createSharedDataset(0.01)
    momentum = createSharedDataset(float(search_params['momentum'])) if "momentum" in search_params else createSharedDataset(0.9) 
    lr_decay = search_params['lr_decay'] if "lr_decay" in search_params else 0.9
    lr_growth = search_params['lr_growth'] if "lr_growth" in search_params else 1.0
    eval_step = search_params['eval_step'] if "eval_step" in search_params else 23 
    max_iter = search_params['max_iter'] if "max_iter" in search_params else 10000 
    stop_threshold = search_params['stop_threshold'] if "stop_threshold" in search_params else 0.995 
    patience = search_params['patience'] if "patience" in search_params else 2000 
    patience_incr = search_params['patience_incr'] if "patience_incr" in search_params else 2 
    rand_seed = search_params['rand_seed'] if "rand_seed" in search_params else 123 
    return(method, learning_rate, momentum, lr_decay, lr_growth, eval_step, max_iter, stop_threshold, patience, patience_incr, rand_seed)


def defineCVParams(cv_params):
    cv_seed = cv_params['cv_seed'] if "cv_seed" in cv_params else 1
    n_folds = cv_params['n_folds'] if "n_folds" in cv_params else 10 
    cv_metric = cv_params['cv_metric'] if "cv_metric" in cv_params else "loglikelihood"
    search_iters = cv_params['search_iters'] if "search_iters" in cv_params else 3 
    L2_range = cv_params['L2_range'] if "L2_range" in cv_params else [-5,-1]
    return(cv_seed, n_folds, cv_metric, search_iters, L2_range)


    
def trainCoxMlp(x_train, ytime_train, ystatus_train, model_params = dict(), search_params = dict(), verbose=False):

    L2_reg, node_map, input_split = defineModelParams(model_params)
    method, learning_rate, momentum, lr_decay, lr_growth, eval_step, max_iter, stop_threshold, patience, patience_incr, rand_seed = defineSearchParams(search_params)
    
    rng = numpy.random.RandomState(rand_seed)
    N_train = ytime_train.shape[0] #number of training examples
    #n_in = x_train.shape[1] #number of features
    
    R_matrix_train = numpy.zeros([N_train, N_train], dtype=int)
    for i in range(N_train):
        for j in range(N_train):
            R_matrix_train[i,j] = ytime_train[j] >= ytime_train[i]

    train_R = createSharedDataset(R_matrix_train)
    train_ystatus = createSharedDataset(ystatus_train, cast_int=False)

    
    model = CoxMlp(rng = rng, x_train=x_train, n_samples = N_train, node_map = node_map, input_split = input_split)

    cost = (
        model.negative_log_likelihood(train_R, train_ystatus)
        + L2_reg * model.L2_sqr
    )

    def nesterovUpdate(cost, params, learning_rate, momentum):
        updates = []
        for param in params:
            vel = createSharedDataset(param.get_value()*0.)
            grad = T.grad(cost=cost, wrt=param)
            updates.append((vel, momentum*vel-learning_rate*grad))
            updates.append((param, param + momentum*momentum*vel - (1+momentum)*learning_rate*grad))
        return updates

        
    def momentumUpdate(cost, params, learning_rate, momentum):
        updates = []
        for param in params:
            param_update = createSharedDataset(param.get_value()*0.)
            updates.append((param, param - param_update))
            updates.append((param_update, momentum*param_update + learning_rate*T.grad(cost=cost, wrt=param)))
        return updates

    
    # def nesterovUpdate(cost, params, learning_rate, momentum): 
        # updates = []
        # for param in params:
            # vel = createSharedDataset(param.get_value()*0.)
            # grad = T.grad(cost=cost, wrt=param)
            # updates.append((vel, momentum*vel-learning_rate*grad))
            # updates.append((param, param + momentum*momentum*vel - (1+momentum)*learning_rate*grad))
        # return updates
    
    
    # Adadelta, Zeiler, 2012
    # def adaDeltaUpdate(cost, params, rho, epsilon): 
        # updates = []
        # for param in params:
            # eg2 = createSharedDataset(param.get_value()*0.)
            # edelta2 = createSharedDataset(param.get_value()*0.)
            # grad = T.grad(cost=cost, wrt=param)
            # updates.append((eg2, rho*eg2 + (1-rho)*T.sqr(grad)))
            # delta = -T.sqrt(edelta2 + epsilon) / T.sqrt(eg2 + epsilon) * grad
            # updates.append((edelta2, rho*edelta2 + (1-rho)*T.sqr(delta)))
            # updates.append((param, param + delta))
        # return updates
    
    # def adaGrad(cost, params, learning_rate, epsilon): 
        # updates = []
        # for param in params:
            # g2 = createSharedDataset(param.get_value()*0.)
            # grad = T.grad(cost=cost, wrt=param)
            # updates.append((param, param - learning_rate*grad/(T.sqrt(g2) + epsilon)))
            # updates.append((g2, g2 + T.sqr(grad)))
        # return updates
    updates = []
    if method == "momentum":
        updates = momentumUpdate(cost, model.params, learning_rate, momentum)
        print("Using momentum gradient")
    elif method == "nesterov":
        updates = nesterovUpdate(cost, model.params, learning_rate, momentum)
        print("Using nesterov accelerated gradient")
    else:
        updates = momentumUpdate(cost, model.params, learning_rate, 0)
        print("Using gradient descent")
    
    #gradiant example based on http://deeplearning.net/tutorial/code/mlp.py
    # g_W = T.grad(cost=cost, wrt=model.W)
    # g_b = T.grad(cost=cost, wrt=model.b)

    # updates = [(param, param - gparam * learning_rate) for param, gparam in zip(model.W + model.b, g_W + g_b)]
    
    index = T.lscalar()
    train_model = theano.function(
        inputs=[index],
        outputs=None,
        updates=updates,
        on_unused_input='ignore'
    )
    
    start = time.time()
    best_cost = numpy.inf
    print("training model")
    for iter in range(max_iter):
        train_model(iter)
        #print cost_iter
        # if method == "momentum" or method == "gradient":
        if iter % eval_step == 0:
            cost_iter = cost.eval()
            if cost_iter > best_cost:
                best_cost = cost_iter
                learning_rate.set_value(numpy.float32(learning_rate.get_value() * lr_decay))
                if verbose == 2:
                    print((('Decreasing learning rate: %f') % (learning_rate.get_value())))
            else:
                learning_rate.set_value(numpy.float32(learning_rate.get_value() * lr_growth))
                if verbose == 2:
                    print((('Increasing learning rate: %f') % (learning_rate.get_value())))
            
            if cost_iter < best_cost * stop_threshold:
                best_cost = cost_iter
                if verbose:
                    print(('cost: %f, iteration: %i') % (best_cost, iter))
                    
                patience = max(patience, iter * patience_incr)
            
            if iter >= patience:
                break
                
        #print cost_iter
    
    print((('running time: %f seconds') % (time.time() - start)))
    print((('total iterations: %f') % (iter)))
    return(model, cost_iter)



    
#computes partial log likelihood of validation set as PL_validation = PL_full(beta) - PL_train_cv(beta)
def CVLoglikelihood(model, x_full, ytime_full, ystatus_full, x_train, ytime_train, ystatus_train):
    N_full = ytime_full.shape[0]
    R_matrix_full = numpy.zeros([N_full, N_full], dtype=int)
    for i in range(N_full):
        for j in range(N_full):
            R_matrix_full[i,j] = ytime_full[j] >= ytime_full[i]
    
    theta = model.predictNewData(x_full)
    exp_theta = numpy.exp(theta)
    PL_full = numpy.sum((theta - numpy.log(numpy.sum(exp_theta * R_matrix_full,axis=1))) * ystatus_full)
    

    N_train = ytime_train.shape[0]
    R_matrix_train = numpy.zeros([N_train, N_train], dtype=int)
    for i in range(N_train):
        for j in range(N_train):
            R_matrix_train[i,j] = ytime_train[j] >= ytime_train[i]
    
    theta = model.predictNewData(x_train)
    exp_theta = numpy.exp(theta)
    PL_train = numpy.sum((theta - numpy.log(numpy.sum(exp_theta * R_matrix_train,axis=1))) * ystatus_train)
    
    return(PL_full - PL_train)


def CIndex(model, x_test, ytime_test, ystatus_test):
    concord = 0.
    total = 0.
    N_test = ystatus_test.shape[0]
    ystatus_test = numpy.asarray(ystatus_test, dtype=bool)
    theta = model.predictNewData(x_test)
    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                if ytime_test[j] > ytime_test[i]:
                    total = total + 1
                    if theta[j] < theta[i]: concord = concord + 1
                    elif theta[j] < theta[i]: concord = concord + 0.5

    return(concord/total)

    
def crossValidate(x_train, ytime_train, ystatus_train, model_params = dict(),search_params = dict(),cv_params = dict(), verbose=False):

    cv_seed, n_folds, cv_metric, search_iters, L2_range = defineCVParams(cv_params)

    N_train = ytime_train.shape[0]
    cv_likelihoods = numpy.zeros([n_folds], dtype=numpy.dtype("float64"))
    cv_folds=cross_validation.KFold(N_train,n_folds=n_folds, shuffle=True, random_state=cv_seed)
    k=0
    for traincv, testcv in cv_folds:
        x_train_cv = x_train[traincv]
        ytime_train_cv = ytime_train[traincv]
        ystatus_train_cv = ystatus_train[traincv]
        
        model, cost_iter = trainCoxMlp(x_train = x_train_cv, ytime_train = ytime_train_cv, ystatus_train = ystatus_train_cv, model_params = model_params, search_params = search_params, verbose=verbose)
        
        x_test_cv = x_train[testcv]
        ytime_test_cv = ytime_train[testcv]
        ystatus_test_cv = ystatus_train[testcv]
        
        if cv_metric == "loglikelihood":
            cv_likelihoods[k] = CVLoglikelihood(model, x_train, ytime_train, ystatus_train, x_train_cv, ytime_train_cv, ystatus_train_cv)
        else:
            cv_likelihoods[k] = CIndex(model, x_test_cv, ytime_test_cv, ystatus_test_cv)
        k += 1
        
        
    return(cv_likelihoods)

def L2CVSearch(x_train, ytime_train, ystatus_train, model_params = dict(),search_params = dict(),cv_params = dict(), verbose=False):
    
    cv_seed, n_folds, cv_metric, search_iters, L2_range = defineCVParams(cv_params)
    
    N_train = ytime_train.shape[0]
    step_size = float(abs(L2_range[1] - L2_range[0]) / 2)
    L2_reg = float(L2_range[0] + L2_range[1]) / 2
    cv_likelihoods = numpy.zeros([0, n_folds], dtype=float)
    L2_reg_params = numpy.zeros([0], dtype="float")
    mean_cvpl = numpy.zeros([0], dtype="float")
    best_L2s = numpy.zeros([0], dtype="float")
    
    model_params['L2_reg'] = numpy.exp(L2_reg)
    cvpl = crossValidate(x_train, ytime_train, ystatus_train, model_params, search_params, cv_params, verbose=verbose)
    cv_likelihoods = numpy.concatenate((cv_likelihoods, [cvpl]), axis=0)
    L2_reg_params = numpy.append(L2_reg_params,L2_reg)
    mean_cvpl = numpy.append(mean_cvpl,numpy.mean(cvpl))
    best_cvpl = numpy.mean(cvpl)
    best_L2 = L2_reg
    # best_L2s = numpy.append(best_L2s,best_L2)
    for i in range(search_iters):
        step_size = step_size/2
        #right
        model_params['L2_reg'] = numpy.exp(best_L2 + step_size)
        right_cvpl = crossValidate(x_train, ytime_train, ystatus_train, model_params, search_params, cv_params, verbose=verbose)
        cv_likelihoods = numpy.concatenate((cv_likelihoods, [right_cvpl]), axis=0)
        L2_reg_params = numpy.append(L2_reg_params,best_L2 + step_size)
        mean_cvpl = numpy.append(mean_cvpl,numpy.mean(right_cvpl))
        #left
        model_params['L2_reg'] = numpy.exp(best_L2 - step_size)
        left_cvpl = crossValidate(x_train, ytime_train, ystatus_train, model_params, search_params, cv_params, verbose=verbose)
        cv_likelihoods = numpy.concatenate((cv_likelihoods, [left_cvpl]), axis=0)
        L2_reg_params = numpy.append(L2_reg_params,best_L2 - step_size)
        mean_cvpl = numpy.append(mean_cvpl,numpy.mean(left_cvpl))
        
        if numpy.mean(right_cvpl) > best_cvpl or numpy.mean(left_cvpl) > best_cvpl:
            if numpy.mean(right_cvpl) > numpy.mean(left_cvpl):
                best_cvpl = numpy.mean(right_cvpl)
                best_L2 = best_L2 + step_size
            else:
                best_cvpl = numpy.mean(left_cvpl)
                best_L2 = best_L2 - step_size
                
        # best_L2s = numpy.append(best_L2s,best_L2)

        
    idx = numpy.argsort(L2_reg_params)
    return(cv_likelihoods[idx], L2_reg_params[idx], mean_cvpl[idx])
    # return(cv_likelihoods, L2_reg_params, mean_cvpl, best_L2s)
    

def L2CVProfile(x_train, ytime_train, ystatus_train, model_params = dict(),search_params = dict(),cv_params = dict(), verbose=False):
    
    cv_seed, n_folds, cv_metric, search_iters, L2_range = defineCVParams(cv_params)

    N_train = ytime_train.shape[0]
    
    cv_likelihoods = numpy.zeros([len(L2_range), n_folds], dtype=float)
    mean_cvpl = numpy.zeros(len(L2_range), dtype="float")
    
    for i in range(len(L2_range)):
        model_params['L2_reg'] = numpy.exp(L2_range[i])
        cvpl = crossValidate(x_train, ytime_train, ystatus_train, model_params, search_params, cv_params, verbose=verbose)
        
        cv_likelihoods[i] = cvpl
        mean_cvpl[i] = numpy.mean(cvpl)
        
    return(cv_likelihoods, L2_range, mean_cvpl)
    

def L2Profile(x_train, ytime_train, ystatus_train, x_validation, ytime_validation, ystatus_validation, model_params = dict(),search_params = dict(),cv_params = dict(), verbose=False):
    cv_seed, n_folds, cv_metric, search_iters, L2_range = defineCVParams(cv_params)
    N_train = ytime_train.shape[0]
    
    likelihoods = []
    for i in range(len(L2_range)):
        model_params['L2_reg'] = numpy.exp(L2_range[i])
        model, cost_iter = trainCoxMlp(x_train = x_train, ytime_train = ytime_train, ystatus_train = ystatus_train, model_params = model_params, search_params = search_params, verbose=verbose)
        
        x_full=numpy.concatenate([x_train, x_validation], axis=0)
        ytime_full=numpy.concatenate([ytime_train, ytime_validation])
        ystatus_full=numpy.concatenate([ystatus_train, ystatus_validation])

        if cv_metric == "loglikelihood":
            likelihoods.append(CVLoglikelihood(model, x_full, ytime_full, ystatus_full, x_train, ytime_train, ystatus_train))
        else:
            likelihoods.append(CIndex(model, x_validation, ytime_validation, ystatus_validation))
        
        
    return(likelihoods, L2_range)
    

    
    
def varImportance(model, x_train, ytime_train, ystatus_train):
    N_train = ytime_train.shape[0]
    R_matrix_train = numpy.zeros([N_train, N_train], dtype=int)
    for i in range(N_train):
        for j in range(N_train):
            R_matrix_train[i,j] = ytime_train[j] >= ytime_train[i]
    
    theta = model.predictNewData(x_train)
    exp_theta = numpy.exp(theta)
    PL_train = numpy.sum((theta - numpy.log(numpy.sum(exp_theta * R_matrix_train,axis=1))) * ystatus_train)
    
    PL_mod = numpy.zeros([x_train.shape[1]])
    for k in range(x_train.shape[1]):
        if (k+1) % 100 == 0:
            print(str(k+1) + "...")
            
        xk_mean = numpy.mean(x_train[:,k])
        xk_train = numpy.copy(x_train)
        xk_train[:,k] = xk_mean
    
        theta = model.predictNewData(xk_train)
        exp_theta = numpy.exp(theta)
        PL_mod[k] = numpy.sum((theta - numpy.log(numpy.sum(exp_theta * R_matrix_train,axis=1))) * ystatus_train)
        
    return(PL_train - PL_mod)
    
def saveModel(model, file_name):
    b = map(lambda tvar : tvar.eval(), model.b)
    W = map(lambda tvar : tvar.eval(), model.W)
    node_map = model.node_map
    input_split = model.input_split
    n_samples = model.n_samples
    rng = model.rng
    x_train = model.x_train
    
    cPickle.dump( (W,b, node_map, input_split, n_samples, x_train, rng), open( file_name, "wb" ))
    
def loadModel(file_name):
    f = file(file_name, 'rb')
    W,b, node_map, input_split, n_samples, x_train, rng = cPickle.load(f)
    f.close()
    model = CoxMlp(rng = rng, x_train=x_train, n_samples = n_samples, node_map = node_map, input_split = input_split)
    for i in range(len(W)):
        model.W[i].set_value(W[i])
    for i in range(len(b)):
        model.b[i].set_value(b[i])
        
    return(model)
    
