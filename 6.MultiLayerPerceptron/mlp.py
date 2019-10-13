
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np

class mlp:
    """ A Multi-Layer Perceptron"""
    
    def __init__(self,inputs,targets,nhidden,beta=1,momentum=0.9,outtype='linear'):
        """ Constructor """
        # Set up network size
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden
        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype
    
        # Initialise network
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)

    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations):
    
        valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count+=1
            print(count)
            self.mlptrain(inputs,targets,eta,niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)
            
        print ("Stopped", new_val_error,old_val_error1, old_val_error2)
        return new_val_error
    	
    def mlptrain(self,inputs,targets,eta,niterations):
        inputs = np.column_stack((inputs,-np.ones((inputs[:,1].size,1))))
        nw1 = np.zeros((np.shape(self.weights1)))
        nw2 = np.zeros((np.shape(self.weights2)))
        for n in range(niterations):
            self.outputs = self.mlpfwd(inputs)
            self.E = 1/2*np.sum((targets-self.outputs)**2) 
            dEdo=-(targets-self.outputs)
            if self.outtype == 'linear':
            	dEdv = (dEdo)/self.ndata # dE/dout*dout/dv
            elif self.outtype == 'logistic':
            	dEdv = (dEdo)*self.outputs*(1.0-self.outputs)/self.ndata # dE/dout*dout/dv
            elif self.outtype == 'softmax':
                dEdv = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.ndata # dE/dout*dout/dv
            else: 
                dEdv = (dEdo)/self.ndata #no outtype assume linear
            dh = self.hidden*(1.0-self.hidden)*(np.dot(dEdv,np.transpose(self.weights2))) # dE/hidden*dhidden/dv Logistic(sigmoidal) approach found on mlpfwd   
            nw1 = eta*(np.dot(np.transpose(inputs),dh[:,:-1])) + self.momentum*nw1  #eta*dE/dout*dout/dv*dv/dw - momentum*nw1 -- The dh variable's last column or element was ignored in the calculation of nw1 because that term is a bias which has nothing to do with the input values.
            nw2 = eta*(np.dot(np.transpose(self.hidden),dEdv)) + self.momentum*nw2    #eta*dE/dhidden*dhidden/dv*dv/dw - momentum*nw2
            self.weights1 =self.weights1-nw1 #Wforhidden=Wforhid-(computed gradient and momentum)
            self.weights2 =self.weights2-nw2 #Wforout=Wforout-(computed gradient and momentum)
            print("Error",self.E)

        
    def mlpfwd(self,inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs,self.weights1);
        self.hidden = 1.0/(1.0+np.exp(-self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

        outputs = np.dot(self.hidden,self.weights2);

        # Different types of output neurons
        if self.outtype == 'linear':
        	return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print( "error")

    def confmat(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)
        
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print ("Confusion matrix is:")
        print (cm)
        print ("Percentage Correct: ",np.trace(cm)/np.sum(cm)*100)

        
        #Reference:https://www.cse.unsw.edu.au/~cs9417ml/MLP2/BackPropagation.html , https://seat.massey.ac.nz/personal/s.r.marsland/Code/Ch4/mlp.py
