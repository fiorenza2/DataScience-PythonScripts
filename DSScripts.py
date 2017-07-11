################################################
################################################
######### Phil's Data Science Scripts###########
######## To expedite Data Sciencing... #########
################################################
################################################

import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

###############################################################################
# Plots variable importances for a RF model from Scikit, modified from Vidhya #
###############################################################################

def plotImp(rndfrst, train, n_imp = 0):
    # if no limit on the number of importances to show is stipulated, or more are 
    # stipulated than exist, go with the max
    n_imp = len(train.columns) if (n_imp == 0 or n_imp > len(train.columns)) else n_imp
    importances = rndfrst.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rndfrst.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1][:n_imp]
    plt.figure(figsize=(15,7))
    plt.title("Feature importances")
    plt.bar(range(n_imp), importances[indices],
           color="crimson", yerr=std[indices], align="center")
    plt.xticks(range(n_imp), train.columns[indices], rotation = 90)
    plt.xlim([-1, n_imp])
    plt.show()
    plt.gcf().clear()

##################################
############## END ###############
##################################


###############################################################################
####### Plots scatter plots of different variables against the dependant ######
###############################################################################

# NB: 'cols' is the list of strings of the variables you are plotting,
# 'against' is the string of the dependant

def print_scatters(df_in, cols, against):   
    plt.figure(1)
    # sets the number of figure row (ie: for 10 variables, we need 5, for 9 we 
    # need 5 as well)
    rows = math.ceil(len(cols)/2)
    f, axarr = plt.subplots(rows, 2, figsize=(10, rows*3))
    # for each variable you inputted, plot it against the dependant
    for col in cols:
        ind = cols.index(col)
        i = math.floor(ind/2)
        j = 0 if ind % 2 == 0 else 1
        if col != against:
            sns.regplot(data = df_in, x=col, y=against, fit_reg=False, ax=axarr[i,j])
        else:
            sns.distplot(a = df_in[col], ax=axarr[i,j])
        axarr[i, j].set_title(col)
    f.text(-0.01, 0.5, against, va='center', rotation='vertical', fontsize = 12)
    plt.tight_layout()
    plt.show()
    plt.gcf().clear()
    
##################################
############## END ###############
##################################