#
# General purpose utility functions
# author: Luca Giancardo  
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as le_me
import scipy
from scipy import stats
from scipy.stats import mannwhitneyu
from skimage import feature, transform
import statsmodels.api as sm
import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot

import json
import itertools

def readConf(confFile):
    """
    Read configuration
    :param confFile: file name
    :return: configuration dictionary
    """

    config = None
    with open(confFile, 'r') as f:
        config = json.load(f)


    return config


def sigTestAUC(data1, data2, disp='long'):
    '''
    return a string with AUC and significance based on the Mann Whitney test
    disp= short|long|auc
    '''
    u, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    # p_value *= 2 # no longer required

    p_val_str = ''
    pValStars = ''
    if (p_value <= 0.001):
        p_val_str = '***p<0.001'
        pValStars = '***'
    elif (p_value <= 0.01):
        p_val_str = '**p<0.01'
        pValStars = '**'
    elif (p_value <= 0.05):
        p_val_str = '*p<0.05'
        pValStars = '*'
    else:
        p_val_str = 'not sig. p={:0.3f}'.format(p_value)
        pValStars = ''

    aucVal = 1 - u / (len(data1) * len(data2))

    if disp == 'short':
        strOut = '{:0.3f}{:}'.format(aucVal, pValStars)
    elif disp == 'long':
        strOut = '{:0.3f} ({:})'.format(aucVal, p_val_str)
    else:
        strOut = '{:0.3f}'.format(aucVal)

    return strOut

def classSigTests( yIn, yPredProbArrIn, classesNamesIn ):
    """

    :param yIn: ground truth y, assumes classes are zero based indexed
    :param yPredProbArrIn:
    :param classesNamesIn:
    :return:
    """
    classIdArr = np.unique(yIn)
    for classId in classIdArr:
        # get probabilities 1 vs all
        probClass = yPredProbArrIn[ yIn==classId, classId ]
        probNoClass = yPredProbArrIn[yIn != classId, classId]
        # significance test
        testStr = sigTestAUC(probNoClass, probClass, disp='long')

        print( classesNamesIn[classId], ': ', testStr )

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def bootStrapMetrics( y, yPred, dataRatio=0.8 ):
    BOOT_NUM = 1000 # number of bootstraps

    classesArr = np.unique(y)
    assert( np.max(classesArr)+1 == len(classesArr) )

    smplNum = len( y )
    bootSmplNum = int(smplNum * dataRatio)
    # create bootstraps indices with replacement
    rndIdx = np.random.randint(len(y), size=(BOOT_NUM, bootSmplNum))

    # select samples/labels
    yPredBoot = yPred[rndIdx]
    yBoot = y[rndIdx]
    #-- for each bootsrap
    resLst = []
    for bIdx in range(yBoot.shape[0]):
        yTmp = yBoot[bIdx,:]
        yPredTmp = yPredBoot[bIdx, :]

        # compute accuracy
        acc = (1.0 * np.sum(yTmp == yPredTmp)) / len(yTmp)

        # compute precision/recall/fscore
        prec, rec, fscore, _ = le_me.precision_recall_fscore_support(yTmp, yPredTmp, average='weighted')
        resLst.append( [acc, prec, rec, fscore] )

    resArr = np.array(resLst)
    # --
    # compute average with full set
    fullPrec, fullRec, fullFscore, _ = le_me.precision_recall_fscore_support(y, yPred, average='weighted')
    # compute accuracy with full set
    fullAcc = (1.0 * np.sum(y == yPred)) / len(y)

    med = np.median(resArr, axis=0)
    upConf = np.percentile(resArr, 95, axis=0)
    lowConf = np.percentile(resArr, 5, axis=0)

    print( 'Accuracy: {:.3f}, [{:.3f}-{:.3f}]'.format(fullAcc, lowConf[0], upConf[0]))
    print( 'Precision: {:.3f}, [{:.3f}-{:.3f}]'.format(fullPrec, lowConf[1], upConf[1]))
    print( 'Recall: {:.3f}, [{:.3f}-{:.3f}]'.format(fullRec, lowConf[2], upConf[2]))
    print( 'fscore: {:.3f}, [{:.3f}-{:.3f}]'.format(fullFscore, lowConf[3], upConf[3]))

    pass

def prBootstrap(negArr, posArr, bootstrapsNumIn=1000):
    """
    Compute Precision-Recall curve using bootstrap.
    See plotPrAndConf for plotting
    :param negArr: np array with negative samples
    :param posArr: np array with positive samples
    :param bootstrapsNumIn: number of bootstraps
    :return: confidence_lower, confidence_upper, recallGridVec, precisionGridMat
    """
    n_bootstraps = bootstrapsNumIn
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    bootPrLst = []
    y_true = np.append([0] * len(negArr), [1] * len(posArr))
    y_pred = np.append(negArr, posArr)
    rng = np.random.RandomState(rng_seed)
    # create recall grid for interpolation
    recallGridVec = np.linspace(0, 1, 100, endpoint=True)
    # matrix containing all precision corresponding to recallGridVec
    precisionGridMat = np.zeros((len(recallGridVec), n_bootstraps))
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = le_me.average_precision_score(y_true[indices], y_pred[indices])
        tmpPrecision, tmpRecall, _ = le_me.precision_recall_curve(y_true[indices], y_pred[indices])
#         tmpRecall = np.concatenate(([0], tmpRecall, [1]))
#         tmpPrecision = np.concatenate(([0], tmpPrecision, [1]))

        # interpolate for comparable ROCs
        fInter = scipy.interpolate.interp1d(tmpRecall, tmpPrecision, kind='nearest')
        precisionGridMat[:, i] = fInter(recallGridVec)

        bootstrapped_scores.append(score)
        bootPrLst.append([tmpRecall, tmpPrecision])
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    # confidence interval for AUCs
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    precisionMean = np.mean(precisionGridMat, axis=1)
    averagePrecision = np.mean(precisionMean)

    return (averagePrecision, confidence_lower, confidence_upper, recallGridVec, precisionGridMat)

def plotPrAndConf(recallGridVec, precisionGridMat, labelIn=''):
    """
    Plot PR curve with confidence interval (estimated with Bootstrap). See prBootstrap function
    :param recallGridVec:
    :param precisionGridMat:
    :param labelIn:
    :return:
    """
    n_bootstraps = precisionGridMat.shape[1]

    # confidence interval for ROC
    precisionGridMatS = np.sort(precisionGridMat, axis=1)
    precisionLow025 = precisionGridMatS[:, int(0.025 * n_bootstraps)]
    precisionTop975 = precisionGridMatS[:, int(0.975 * n_bootstraps)]
    precisionMean = np.mean(precisionGridMat, axis=1)

    # plt.hold(True)
    ax = plt.gca()  # kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(recallGridVec[1:-1], precisionMean[1:-1], '-', linewidth=4, label=labelIn)
    ax.fill_between(recallGridVec, precisionLow025, precisionTop975, facecolor=base_line.get_color(), alpha=0.2)

def rocBootstrap(negArr, posArr, bootstrapsNumIn=1000):
    """
    Compute ROC bootstrap.
    See plotRocAndConf for plotting
    :param negArr: np array with negative samples
    :param posArr: np array with positive samples
    :param bootstrapsNumIn: number of bootstraps
    :return: confidence_lower, confidence_upper, fprGridVec, tprGridMat
    """

    n_bootstraps = bootstrapsNumIn
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    bootRocLst = []
    y_true = np.append([0] * len(negArr), [1] * len(posArr))
    y_pred = np.append(negArr, posArr)
    rng = np.random.RandomState(rng_seed)
    # create fpr grid for interpolation
    fprGridVec = np.linspace(0, 1, 100, endpoint=True)
    # matrix containing all tpr corresponding to fprGridVec
    tprGridMat = np.zeros((len(fprGridVec), n_bootstraps))
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = le_me.roc_auc_score(y_true[indices], y_pred[indices])
        tmpFpr, tmpTpr, _ = le_me.roc_curve(y_true[indices], y_pred[indices])
        tmpFpr = np.concatenate(([0], tmpFpr, [1]))
        tmpTpr = np.concatenate(([0], tmpTpr, [1]))

        # interpolate for comparable ROCs
        fInter = scipy.interpolate.interp1d(tmpFpr, tmpTpr, kind='nearest')
        tprGridMat[:, i] = fInter(fprGridVec)

        bootstrapped_scores.append(score)
        bootRocLst.append([tmpFpr, tmpTpr])
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    # confidence interval for AUCs
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    tprMean = np.mean(tprGridMat, axis=1)
    averageTPR = np.mean(tprMean)

    return (averageTPR, confidence_lower, confidence_upper, fprGridVec, tprGridMat)
def rocBootstrap(negArr, posArr, bootstrapsNumIn=1000):
    """
    Compute ROC bootstrap.
    See plotRocAndConf for plotting
    :param negArr: np array with negative samples
    :param posArr: np array with positive samples
    :param bootstrapsNumIn: number of bootstraps
    :return: confidence_lower, confidence_upper, fprGridVec, tprGridMat
    """

    n_bootstraps = bootstrapsNumIn
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    bootRocLst = []
    y_true = np.append([0] * len(negArr), [1] * len(posArr))
    y_pred = np.append(negArr, posArr)
    rng = np.random.RandomState(rng_seed)
    # create fpr grid for interpolation
    fprGridVec = np.linspace(0, 1, 100, endpoint=True)
    # matrix containing all tpr corresponding to fprGridVec
    tprGridMat = np.zeros((len(fprGridVec), n_bootstraps))
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = le_me.roc_auc_score(y_true[indices], y_pred[indices])
        tmpFpr, tmpTpr, _ = le_me.roc_curve(y_true[indices], y_pred[indices])
        tmpFpr = np.concatenate(([0], tmpFpr, [1]))
        tmpTpr = np.concatenate(([0], tmpTpr, [1]))

        # interpolate for comparable ROCs
        fInter = scipy.interpolate.interp1d(tmpFpr, tmpTpr, kind='nearest')
        tprGridMat[:, i] = fInter(fprGridVec)

        bootstrapped_scores.append(score)
        bootRocLst.append([tmpFpr, tmpTpr])
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    # confidence interval for AUCs
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]

    return (confidence_lower, confidence_upper, fprGridVec, tprGridMat)

def plotRocAndConf(fprGridVec, tprGridMat, labelIn=''):
    """
    Plot ROC curve with confidence interval (estimated with Bootstrap). See rocBootstrap function
    :param fprGridVec:
    :param tprGridMat:
    :param labelIn:
    :return:
    """
    n_bootstraps = tprGridMat.shape[1]

    # confidence interval for ROC
    tprGridMatS = np.sort(tprGridMat, axis=1)
    tprLow025 = tprGridMatS[:, int(0.025 * n_bootstraps)]
    tprTop975 = tprGridMatS[:, int(0.975 * n_bootstraps)]
    tprMean = np.mean(tprGridMat, axis=1)

    # plt.hold(True)
    ax = plt.gca()  # kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(fprGridVec, tprMean, '-', linewidth=4, label=labelIn)
    ax.fill_between(fprGridVec, tprLow025, tprTop975, facecolor=base_line.get_color(), alpha=0.2)
    
def sensitivitySpecificityBootstrap(negArr, posArr, bootstrapsNumIn=1000):
    """
    Compute sensitivity and specificity bootstrap.
    See plotRocAndConf for plotting
    :param negArr: np array with negative samples
    :param posArr: np array with positive samples
    :param bootstrapsNumIn: number of bootstraps
    :return: confidence_lower, confidence_upper, fprGridVec, tprGridMat
    """

    n_bootstraps = bootstrapsNumIn
    rng_seed = 42  # control reproducibility
    sensList = []
    specList = []
    y_true = np.append([0] * len(negArr), [1] * len(posArr))
    y_pred = np.append(negArr, posArr)
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = le_me.roc_auc_score(y_true[indices], y_pred[indices])
        tn, fp, fn, tp = le_me.confusion_matrix(y_true[indices], y_pred[indices]).ravel()
        sens = tp/(tp + fn)
        spec = tn/(tn + fp)

        specList.append(spec)
        sensList.append(sens)

    return (np.mean(sensList), np.mean(specList))

# Taken from DeepExplain
def plot(data, xi=None, cmap='RdBu_r', axis=plt, percentile=100, dilation=3.0, alpha=0.8):
    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, data.shape[1], dx)
    yy = np.arange(0.0, data.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)
    overlay = None
    if xi is not None:
        # Compute edges (to overlay to heatmaps later)
        xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
        in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(data), percentile)
    abs_min = abs_max

    if len(data.shape) == 3:
        data = np.mean(data, 2)
    axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
    axis.axis('off')
    return axis

def pearsonr_ci(x,y,alpha=0.05):
    """ 
    calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    """

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

# Diagnostic plots for regression analysis
def graph(formula, x_range, label=None):
    """
    Helper function for plotting cook's distance lines
    """
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')


def diagnostic_plots(X, y, model_fit=None):
    """
    Function to reproduce the 4 base plots of an OLS model in R.

    ---
    Inputs:

    X: A numpy array or pandas dataframe of the features to use in building the linear regression model

    y: A numpy array or pandas series/dataframe of the target variable of the linear regression model

    model_fit [optional]: a statsmodel.api.OLS model after regressing y on X. If not provided, will be
                        generated from X, y
    """

    if not model_fit:
        model_fit = sm.OLS(y, sm.add_constant(X)).fit()

    # create dataframe from X, y for easier plot handling
    dataframe = pd.concat([X, y], axis=1)

    # model values
    model_fitted_y = model_fit.fittedvalues
    # model residuals
    model_residuals = model_fit.resid
    # normalized residuals
    model_norm_residuals = model_fit.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)
    # leverage, from statsmodels internals
    model_leverage = model_fit.get_influence().hat_matrix_diag
    # cook's distance, from statsmodels internals
    model_cooks = model_fit.get_influence().cooks_distance[0]

    plot_lm_1 = plt.figure()
    plot_lm_1.axes[0] = sns.residplot(model_fitted_y, y, data=dataframe,
                            lowess=True,
                            scatter_kws={'alpha': 0.5},
                            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    plot_lm_1.axes[0].set_title('Residuals vs Fitted')
    plot_lm_1.axes[0].set_xlabel('Fitted values')
    plot_lm_1.axes[0].set_ylabel('Residuals');

    # annotations
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]
    for i in abs_resid_top_3.index:
        plot_lm_1.axes[0].annotate(i,
                                 xy=(model_fitted_y[i],
                                     model_residuals[i]));

    QQ = ProbPlot(model_norm_residuals)
    plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
    plot_lm_2.axes[0].set_title('Normal Q-Q')
    plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
    plot_lm_2.axes[0].set_ylabel('Standardized Residuals');
    # annotations
    abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]
    for r, i in enumerate(abs_norm_resid_top_3):
        plot_lm_2.axes[0].annotate(i,
                                 xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                     model_norm_residuals[i]));

    plot_lm_3 = plt.figure()
    plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5);
    sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt,
              scatter=False,
              ci=False,
              lowess=True,
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
    plot_lm_3.axes[0].set_title('Scale-Location')
    plot_lm_3.axes[0].set_xlabel('Fitted values')
    plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

    # annotations
    abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
    abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
    for i in abs_norm_resid_top_3:
        plot_lm_3.axes[0].annotate(i,
                                 xy=(model_fitted_y[i],
                                     model_norm_residuals_abs_sqrt[i]));


    plot_lm_4 = plt.figure();
    plt.scatter(model_leverage, model_norm_residuals, alpha=0.5);
    sns.regplot(model_leverage, model_norm_residuals,
              scatter=False,
              ci=False,
              lowess=True,
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
    plot_lm_4.axes[0].set_xlim(0, max(model_leverage)+0.01)
    plot_lm_4.axes[0].set_ylim(-3, 5)
    plot_lm_4.axes[0].set_title('Residuals vs Leverage')
    plot_lm_4.axes[0].set_xlabel('Leverage')
    plot_lm_4.axes[0].set_ylabel('Standardized Residuals');

    # annotations
    leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
    for i in leverage_top_3:
        plot_lm_4.axes[0].annotate(i,
                                 xy=(model_leverage[i],
                                     model_norm_residuals[i]));

    p = len(model_fit.params) # number of model parameters
    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x),
        np.linspace(0.001, max(model_leverage), 50),
        'Cook\'s distance') # 0.5 line
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),
        np.linspace(0.001, max(model_leverage), 50)) # 1 line
    plot_lm_4.legend(loc='upper right');
    
def mannAUC( data1, data2 ):
    """
    Compute AUC using the Mann Whitney test
    """
    u, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    aucVal = 1 - u / (len(data1) * len(data2))
    
    return aucVal




def findCutoffPnt3(dataPos, dataNeg):
    """
    Find cutoff point minimizing the distance to Sens 1, spec 1 and calculate statistics (with kappa).
    format confMat:
     array([[TN, FP],
            [ FN, TP]]))
    :param dataPos:
    :param dataNeg:
    :return: acc,sens,spec,roc_auc, cutoffTh, confusionMat, kappa
    """

    dataAll = np.concatenate((dataPos, dataNeg))
    lblArr = np.zeros(len(dataAll), dtype=bool)
    lblArr[0:len(dataPos)] = True

    fpr, tpr, thresholds = le_me.roc_curve(lblArr, dataAll, pos_label=True)
    roc_auc = le_me.auc(fpr, tpr)

#     # invert comparison if (ROC<0.5) required
#     if roc_auc<0.5:
#         lblArr = ~lblArr
#         fpr, tpr, thresholds = le_me.roc_curve(lblArr, dataAll, pos_label=True)
#         roc_auc = le_me.auc(fpr, tpr)
#         print 'inverting labels'


    # calculate best cut-off based on distance to top corner of ROC curve
    distArr = np.sqrt(np.power(fpr, 2) + np.power((1 - tpr), 2))
    cutoffIdx = np.argsort(distArr)[0]
    cutoffTh = thresholds[cutoffIdx]

    lblOut = dataAll >= cutoffTh

    acc = le_me.accuracy_score(lblArr, lblOut)
    sens = tpr[cutoffIdx]
    spec = 1 - fpr[cutoffIdx]
    cfMat = le_me.confusion_matrix(lblArr, lblOut)

    kappa = le_me.cohen_kappa_score( lblOut, lblArr )

    return (acc, sens, spec, roc_auc, cutoffTh, cfMat, kappa )

def useCutOffPoint( dataPos,dataNeg, cutoffTh, dataPosGtNeg = True  ):
    """
    v. 3 fixing confusion matrix
    """
    dataAll = np.concatenate((dataPos, dataNeg))
    lblArr = np.zeros(len(dataAll), dtype=bool)
    lblArr[0:len(dataPos)] = True
                       
    fpr, tpr, _ = le_me.roc_curve(lblArr, dataAll )    
    roc_auc = le_me.auc(fpr, tpr)
    
    lblOut = (dataAll >= cutoffTh)
    if not dataPosGtNeg: # invert comparison if required
        lblOut = ~lblOut
        
    acc = le_me.accuracy_score(lblArr, lblOut)
    cMat = le_me.confusion_matrix(lblArr, lblOut)
    # decompose conf matrix
    tn = cMat[0,0]
    fn = cMat[1,0]
    tp = cMat[1,1]
    fp = cMat[0,1]

    sens = 1. * (tp) / ( tp + fn ) # sensitivity / recall
    spec = 1.  * (tn) / ( tn +  fp )
    ppv = 1. * (tp) / ( tp + fp ) # positive predictive value  / precision
    npv = 1. * (tn) / ( tn + fn ) # negative predictive value  

    # sens = 1. * (cMat[0,0]) / ( cMat[0,0] +  cMat[0,1] )
    # spec = 1.  * (cMat[1,1]) / ( cMat[1,0] +  cMat[1,1] )
    
    # invert if required
    if not dataPosGtNeg:
        tmp = sens
        sens = spec
        spec = tmp

    
    return { 'acc':acc,'sens':sens,'spec':spec, 'ppv': ppv, 'npv': npv, 'auroc': roc_auc }
    