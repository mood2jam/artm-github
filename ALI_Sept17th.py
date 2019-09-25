#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 23:52:38 2019

@author: aliheydari
@email: aliheydari@ucdavis.edu
@web: https://www.ali-heydari.com

"""
###################### ADAPTIVE LEARNING INTEGRATION ##########################
################ Adaptive Loss Functions WITHOUT AdaLearn #####################
import os
os.system("pip install easydict");

import easydict
import numpy as np
import torch



version = "0.0.5"
backend = "PyTorch"



def Welcome_BEARD():
  
      print("\__________     __________/")
      print(" |         |-^-|         |")
      print(" |         |   |         |")
      print("  `._____.´     `._____.´")
      print("  \                     /")
      print("   \\\                 // ")
      print("    \\\    ////\\\\\\\   //")
      print("     \\\\\           /// ")
      print("       \\\\\\\\\\\\|////// ")
      print("         \\\\\\\\|//// ")


      print(" ")
      print("ALI {} for {} imported succsessfuly".format(version,backend))

    
Welcome_BEARD();

    
""" 
SOFTADAPT in various forms


Modified version of softmax with a little spice from beta

Beta is a hyperparameter that will either sharpen or dampen the peaks 

Default Beta is 1

Variations : 
    
******* SoftAdapt : A vanilla softmax with the value of the loss function 
        at each iteration multiplied by the exponent, i.e. 
        
        softAdapt(f,s) = f_i * e^(beta*s) / (sum f_je^(beta*s_j))
        where f is the loss value and s is the slope : 


******* PlushAdapt : The same idea as soft Adapt except the slopes and the loss
        function values are normalized at each iteration
        
        
        
******* DownyAdapt : Same as SoftAdapt except only the slopes are being normalized
        at each iteration       
        



Usage : 
    pass in a np vector n with a weight beta (if not sure what to use, then pass 1)
    returns softmax in the same dimensions as n

"""


# ################## SOFT ADAPT #############################################


def SoftAdapt(n,beta,loss_tensor,i):
     # numerator
    
#      n = -1 * n;
      
     if len(n) == 2 : 
     
         fe_x = np.zeros(2);
         fe_x[0] = loss_tensor[0].data.item() * np.exp(beta * (n[0] - np.max(n)));
         fe_x[1] = loss_tensor[1].data.item() * np.exp(beta * (n[1] - np.max(n)));
         denom = fe_x[0] + fe_x[1];


     elif len(n) == 3 :
         
         fe_x = np.zeros(3);
         fe_x[0] = loss_tensor[0].data.item() * np.exp(beta * (n[0] - np.max(n)));
         fe_x[1] = loss_tensor[1].data.item() * np.exp(beta * (n[1] - np.max(n)));
         fe_x[2] = loss_tensor[2].data.item() * np.exp(beta * (n[2] - np.max(n)));
         denom = fe_x[0] + fe_x[1] + fe_x[2];  
                                               
     else :
         print("As of now, we only support 2 or 3 losses, please check input")



                                          
     return (fe_x[i]/ denom)


################## PLUSH ADAPT #############################################
def PlushAdapt(n,beta,loss_tensor,i):

#   n = -1 * n;

  if len(n) == 2 : 
        fe_x = np.zeros(2);
         
         # Normalize the slopes!!!!
        n[0] = n[0] / (np.linalg.norm(n,1) + 1e-8);
        n[1] = n[1] / (np.linalg.norm(n,1) + 1e-8);
         
         # normalize the loss functions 
         
        denom2 = loss_tensor[0].data.item() + loss_tensor[1] 
    
        fe_x[0] = loss_tensor[0].data.item() / denom2;
        fe_x[1] = loss_tensor[1].data.item() / denom2;
        
        fe_x[0] = fe_x[0] * np.exp(beta * (n[0] - np.max(n)));
        fe_x[1] = fe_x[1] * np.exp(beta * (n[1] - np.max(n)));

    
  
        denom = fe_x[0] + fe_x[1];                                      
         
         
        return (fe_x[i]/ denom)
         


  elif len(n) == 3 :
         
        fe_x = np.zeros(3);
         
         # Normalize the slopes!!!!
        n[0] = n[0] / (np.linalg.norm(n,1) + 1e-8);
        n[1] = n[1] / (np.linalg.norm(n,1) + 1e-8);
        n[2] = n[2] / (np.linalg.norm(n,1) + 1e-8);

         
         # normalize the loss functions 
         
        denom2 = loss_tensor[0].data.item() + loss_tensor[1].data.item() + loss_tensor[3].data.item() 
    
        fe_x[0] = loss_tensor[0].data.item() / denom2;
        fe_x[1] = loss_tensor[1].data.item() / denom2;
        fe_x[2] = loss_tensor[2].data.item() / denom2;

        
        fe_x[0] = fe_x[0] * np.exp(beta * (n[0] - np.max(n)));
        fe_x[1] = fe_x[1] * np.exp(beta * (n[1] - np.max(n)));
        fe_x[2] = fe_x[2] * np.exp(beta * (n[2] - np.max(n)));


    
  
        denom = fe_x[0] + fe_x[1] + fe_x[2] ;                                      
         
         
        return (fe_x[i]/ denom)
                                               
  else :
         
         print("As of now, we only support 2 or 3 losses, please check input")




################## DOWNY SOFT ADAPT #######################################
         
def DownyAdapt(n,beta,loss_tensor,i):
    # numerator
    fe_x = np.zeros(2);
    n[0] = n[0] / (np.linalg.norm(n,1) + 1e-8);
    n[1] = n[1] / (np.linalg.norm(n,1) + 1e-8);
    
    
    denom2 = loss_tensor[0].data.item() + loss_tensor[1] 
    
    fe_x[0] = loss_tensor[0].data.item() / denom2;
    fe_x[1] = loss_tensor[1].data.item() / denom2;


    
    fe_x[0] = fe_x[0] * np.exp(beta * (n[0] - np.max(n)));
    fe_x[1] = fe_x[1] * np.exp(beta * (n[1] - np.max(n)));

    
  
    denom = fe_x[0] + fe_x[1];    

    if len(n) == 2 : 
     
         fe_x = np.zeros(2);
         
         # Normalize the slopes
         n[0] = n[0] / (np.linalg.norm(n,1) + 1e-8);
         n[1] = n[1] / (np.linalg.norm(n,1) + 1e-8);
        
         fe_x[0] = loss_tensor[0].data.item() * np.exp(beta * (n[0] - np.max(n)));
         fe_x[1] = loss_tensor[1].data.item() * np.exp(beta * (n[1] - np.max(n)));
         
         denom = fe_x[0] + fe_x[1];
         
         return (fe_x[i]/ denom)


    elif len(n) == 3 :
         
         fe_x = np.zeros(3);
         n[0] = n[0] / np.linalg.norm(n,1);
         n[1] = n[1] / np.linalg.norm(n,1);
         n[2] = n[2] / np.linalg.norm(n,1); 
         
         fe_x[0] = loss_tensor[0].data.item() * np.exp(beta * (n[0] - np.max(n)));
         fe_x[1] = loss_tensor[1].data.item() * np.exp(beta * (n[1] - np.max(n)));
         fe_x[2] = loss_tensor[2].data.item() * np.exp(beta * (n[2] - np.max(n)));
         denom = fe_x[0] + fe_x[1] + fe_x[2]; 
         return (fe_x[i]/ denom)

                                               
    else :
         print("As of now, we only support 2 or 3 losses, please check input")

                            


############ FINITE DIFFERENCE #################

"""
loss Usage:
    
#    pass in 5 points as a np array 
#    outputs a forth order accurate first derivative approximation
#    if more accurate slope approximation is needed, then more points would be 
    # required
    
"""

def FD(loss_pts,args):
    

# New technique:
# New technique:

    if args["fd_order"] == 5:
        der = ((25/12) * loss_pts[4]) - ((4) * loss_pts[3]) + ((3) * loss_pts[2]) \
        - ((4/3) * loss_pts[1]) + ((1/4) * loss_pts[0])
        
    elif args["fd_order"] == 3:
        der = (-3/2) * loss_pts[0] + 2 * loss_pts[1] + (-1/2) * loss_pts[2]
    else:
        raise NotImplementedError("A finite difference order of {} is not implemented yet.".format(args.fd_order))
    
    
    return der

  
    
    
    
    
    return der

  
"""
Alpha assignment : a function that calls one of the variations of SoftAdapt
and it will return the appropiate values for each alpha

Usage : 
    
    Argument : A vector of slopes n
               A constant value for the softmax called kappa (default1 1) 
               A tensor of loss values at each iteration loss_tensor 
                   e.g. if your loss function is MSE + l1, then 
                   loss_tensor = [MSE, l1];
               A string indicating which method you want to use
               (default should be PlushAdapt)


"""

def alpha_assign(n,kappa,loss_tensor, string):

   
    alpha = np.zeros(len(n));
  
    if string == "plush":

   
        for i in range(len(n)):
                
                alpha[i] = PlushAdapt(n,kappa,loss_tensor,i);
                
    if string == "downy":

   
        for i in range(len(n)):
                
                alpha[i] = DownyAdapt(n,kappa,loss_tensor,i);

    if string == "soft":

   
        for i in range(len(n)):
                
                alpha[i] = SoftAdapt(n,kappa,loss_tensor,i);
    
    return alpha 
    
    


"""
Set Hyper :
    sets the hyper parameters alpha0 through alpha n
    
    Usage : 
        
        takes in the vector Alpha, a dictionary of arguments (highly recommend
        a global dictionary) and number of loss functions
        outputs all the values of 
        
        
    Caution : 
        
        alpha -> the vector returned by alpha_assign() 
        alphas -> a global array that its entries are multiplied by the loss
        


"""


def set_hyper(alpha,args,loss_num):
    
    for i in range (0,loss_num) : 
   
        args["alphas"][i] = alpha[i];

        
"""
Get 5 Loss :
    it stores the last 5 losses efficently and properly 
    
    Usage : 
        
       inputs :  the most current loss value -> new_loss 
                 a string that indicates which part of the loss is being stored
                     strings should be of the format "loss1", "loss2", "loss3" etc
                 
                 
                 
                 a (preferably global ) dictionary with the loss vectors
                     -> args
                 
                 
        output :  it sets the global arrays loss1 and loss2 and 
        their corresponding counters
        


"""

def get_5loss(new_loss, index, args):

    if args["global_count"] > 4 :
    
        if index == 1:

            args["loss2"] = np.hstack( (args["loss2"] , new_loss) );

            if args["global_count"] >= args["fd_order"]:
                args["loss2"] = args["loss2"][-args["fd_order"]:];

        elif index == 0:
    #             args.loss1[args.loss1_global_count % 5] = new_loss

               # to save the arrays in an orderly fashion
            args["loss1"] = np.hstack((args["loss1"], new_loss));
            if args["global_count"] >= args["fd_order"]:
                args["loss1"] = args["loss1"][-args["fd_order"]:];


        elif index == 2:
            args["loss3"] = np.hstack((args["loss3"], new_loss));
             # to save the arrays in an orderly fashion
            args.loss3.append(new_loss);

            if args.loss3.size > 5 :
                  args["loss3"] = args["loss3"][-5:];

        else :
             print("Wrong Index");

    
    else : 
        
        print("ALI idle --- less than 5 iters");
        args["global_count"] += 1;
 
###################### ADAPTIVE LEARNING INTEGRATION ##########################
################################# AdaLearn ####################################

import numpy as np



#################### avg_calc ############################
"""
average calculator : given the most recent loss it will calculate the average with resepct
to the average in a memory efficient way (I think)

input : the most recent loss (for each individual loss)
outputs : it changes the global variable args.loss1_avg (and for all other losses) and does
not return a value



"""

def avg_calc(recent_loss1,args):
  
    if args["loss1_global_count"] > 4 :
        
        if args["loss1_avg"] == 0 : 
            
            args["loss1_avg"] = np.mean(args["loss2"]);
        
        args["loss1_avg"] = (args["loss1_avg"] * 5 + recent_loss1)/6;
    
    
    
    


################## adapt_lr ##########################
"""
adaptive learning : it checks a very simple criterion and if it is true then it will set 
the learning rate to the max, and it will change a global flag and iter whcih then are used
to decay back down 

input : the most recent losses 

output : it will update the following global variables : 
        args.lr
        args.flag
        args.adapt_iter2

"""

def adapt_lr(recent_loss1,args):
    
        
    if recent_loss1 > args["loss1_avg"] and args["flag"] == 'GO':
        args["lr"] = args["lr_max"];        
        args["flag"] = 'NO'
        args["adapt_iter2"] += 1;

#         print("lr is inc to max");
        
    if recent_loss1 > args["loss1_avg"] and args["flag"] == 'NO' and  args["adapt_iter2"] > 2:
        
        args["lr"] = args["lr_min"];        
        args["flag"] = 'NO2'
        args["adapt_iter"] += 1;

#         print("lr is dec to min");
          
          
        
   
################## lr_decay ##########################
"""
learning rate decay : it will increase or decrease the learning rate depending on the specific conditions

input : a dictionary of global variables args 

output : it will update the following global variables : 
        args.lr
        args.flag
        args.adapt_iter
"""

        
def lr_decay(args):

    if args["flag"] == 'NO2' and args["lr"] < args["lr_max"] : 
        print("OK AT LEAST IT GOES THROUGH IT!")
        args["lr"] *= 2;
        if args["lr"] > args["lr_max"] :
            
                args["lr"] = args["lr_max"];
        
        
    elif args["flag"] == 'NO' and args["lr"] > args["lr_min"] : 
  
        if args["lr"] > args["lr_min"]: 

            args["lr"] = args["lr_max"] * np.exp(-1 * args["adapt_iter"] * args["kappa"]);

        if args["lr"] < args["lr_min"] : 

            args["lr"] = args["lr_min"];

        else :

            args["flag"] = 'GO'     
    else:
        
        args["flag"] = 'GO'
    
    
    
    
    
    
      
    
# if the user does not want to have a seperate file, they can make all the-
                 # needed parameters here with this function
                 
                 
                 
                 
def make_args(*args):    
    
    args = {
        
        "alphas": np.zeros(2),
        "lr":0.001,
        "lr_max" : 0.01,
        "lr_min" : 0.0001,
        "loss1" : np.zeros(5),
        "loss2" : np.zeros(5),
        "global_count" : 0,
        "loss1_global_count" : 0,
        "loss2_global_count" : 0,
        "loss1_avg" : 0,
        "loss2_avg" : 0, 
        "flag": 'Go',
        "adapt_iter": 1,
        "adapt_iter2": 1,    
        "kappa": 1.5,
        "fd_order": 5,
        "num_epochs" : 200,
        "user_lazy": 'y'
    
     }    
    
    return args
    
    
    
    
    
#########################  THE ONE FUNCTION DO IT ALL #########################   
    
"""
Adaptive Learning Integration Function: Simply calling this function will 

Usage : 
    
    input: 
            which_SA : which variation of SoftAdapt you would want to use
                SoftAdapt ->  the original (weaker) version
                Plush SoftAdapt -> the more robust (defult) variation of SA
                Downey SoftAdapt -> A very smoothed out version of SA. Use 
                    with caution and for very specific cases
                    
            AdaLearn_bool: Whether you want to use AdaLearn or not
            
            




"""
    
def ALI(recent_loss_tensor,which_SA="plush",AdaLearn_bool="False",beta=0.1\
        ,args = make_args()):
    
    # print all the parameters once at the very begining
    if args["global_count"] == 0 : 
      print("The Parameters of ALI : ");
      print("Soft Adapt Variation: {}".format(which_SA));
      print("Use AdaLearn: {}".format(AdaLearn_bool));      
      print("softmax Beta: {}".format(beta));
      print("learning rate : {}".format(args["lr"]))
      args["global_count"] += 1 ;
    


    # if we want to use AdaLearn
    if AdaLearn_bool == True : 
        
        # calculate the running average in an efficient matter
        avg_calc(recent_loss_tensor[0],recent_loss_tensor[1],args);
        
        # Set the Adaptive learning Rate
        # Decay the weight according to the performance 
        lr_decay(args);
            
    
    if len(recent_loss_tensor) == 2 :
        #store the last 5 losses for loss 1
        get_5loss(recent_loss_tensor[0].data.item(),0,args);
        get_5loss(recent_loss_tensor[1].data.item(),1,args);
                        
   
    elif len(recent_loss_tensor) == 3 :
        get_5loss(recent_loss_tensor[0].data.item(),0,args);
        get_5loss(recent_loss_tensor[1].data.item(),1,args);
        get_5loss(recent_loss_tensor[2].data.item(),2,args);
    
    else : 
        
        print("support for this coming soon");
    
    
    if args["global_count"] > 4 :
        
         if len(recent_loss_tensor) == 2 :
             slopes = np.zeros(2);
             slopes[0] = FD(args["loss1"],args);
             slopes[1] = FD(args["loss2"],args);
             
             set_hyper(alpha_assign(slopes,beta,recent_loss_tensor,which_SA)\
                       ,args,len(recent_loss_tensor));
             
        
         elif len(recent_loss_tensor) == 3 :
             slopes = np.zeros(3);
             slopes[0] = FD(args["loss1"],args);
             slopes[1] = FD(args["loss2"],args);
             slopes[2] = FD(args["loss3"],args);
                
             set_hyper(alpha_assign(slopes,beta,recent_loss_tensor,which_SA)\
                       ,args,len(recent_loss_tensor));
        
         else:
             
             print("Implementation coming soon : ) ")
                
        
# to see if we need to return alpha or not for the user
    #if hasattr(args, 'user_lazy'):
    if 1 < 2 :   
       alpha_return = np.zeros(len(recent_loss_tensor))
      
      
       # if user did not make the global dictionary
       #if args['user_lazy'] == 'y' : 
       if 1 < 2 : 
           # return the alphas
           alpha_return = args['alphas'];

       return alpha_return
    
    
    
    
    
    
    
    
    
    
    

