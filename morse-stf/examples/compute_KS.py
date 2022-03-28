# -*- coding: utf-8 -*-
"""
   Ant Group Copyright (c) 2004-2020 All Rights Reserved.
"""
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# def get_KS(label, score):
#     label=label.tolist()
#     #label=np.reshape(label, len(label))
#     score=score.tolist()
#     #score=np.reshape(score, len(label))
#
#
#     z=list(zip(label, score))
#
#     z.sort(key=(lambda r: r[1]), reverse=True)
#     num_T= sum(label)+1E-4
#     num_F= len(label)-num_T+1E-4
#
#     TP=0.0
#     FP=0.0
#     TPR = TP / num_T
#     FPR = FP / num_F
#     KS=0.0
#     for r in z:
#         if r[0]==1 :
#             TP=TP+1
#             TPR=TP/num_T
#         else:
#             FP=FP+1
#             FPR=FP/num_F
#
#         KS=max(KS, TPR-FPR)
#     return KS

def compute_KS_gaode3w():
    file_path='/Users/qizhi.zqz/Documents/dataset/gaode_3w_y.csv'

    predict_path='/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/tfe/qqq/predict'


    # y = pd.read_csv(file_path+"/公开信贷样本_目标变量含X特征变量.csv", index_col=["id"])

    y = pd.read_csv(file_path, index_col=["id","ent_date"])
    

    #y_hat = pd.read_csv(predict_path+"tfe/qqq/predict", index_col=["id","ent_date"])
    y_hat = pd.read_csv(predict_path, header=None,names=["id","ent_date","predict"],index_col=["id","ent_date"])
    df = y.join(y_hat).dropna()
    #df=pd.concat([x, y], axis=1)
    


    y = df.loc[:, 'label']
    
    y_hat = df.loc[:, 'predict']
    


    y = np.array(y)
    y_hat = np.array(y_hat)




    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)


    KS = max(tpr - fpr)
    



def compute_KS_gaode20w():
    file_path='/Users/qizhi.zqz/Documents/dataset/gaode_20w_y.csv'

    predict_path='/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/tfe/qqq/predict'

    # y = pd.read_csv(file_path+"/公开信贷样本_目标变量含X特征变量.csv", index_col=["id"])

    y = pd.read_csv(file_path, index_col=["id","ent_date"], header=0)

    #y_hat = pd.read_csv(predict_path+"tfe/qqq/predict", index_col=["id","ent_date"])
    y_hat = pd.read_csv(predict_path, header=None,names=["id","ent_date","predict"], index_col=["id","ent_date"])
    df = y.join(y_hat)
    #df=pd.concat([x, y], axis=1)

    y = df.loc[:, 'label']
    
    y_hat = df.loc[:, 'predict']
    


    y = np.array(y)
    y_hat = np.array(y_hat)




    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)

    # KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)
    

    AUC=metrics.auc(fpr, tpr)
    


def compute_KS_gaode20w_DNN():
    file_path='/Users/qizhi.zqz/Documents/dataset/gaode20w_Y_test.csv'

    predict_path='/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/tfe/qqq/predict'


    # y = pd.read_csv(file_path+"/公开信贷样本_目标变量含X特征变量.csv", index_col=["id"])

    y = pd.read_csv(file_path, names=["id", "label"], index_col=["id"], header=0)
    

    #y_hat = pd.read_csv(predict_path+"tfe/qqq/predict", index_col=["id","ent_date"])
    y_hat = pd.read_csv(predict_path, header=None, names=["id","predict"], index_col=["id"])
    df = y.join(y_hat)
    #df=pd.concat([x, y], axis=1)
    


    y = df.loc[:, 'label']
    
    y_hat = df.loc[:, 'predict']
    


    y = np.array(y)
    y_hat = np.array(y_hat)



    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)

    # KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)
    

    AUC=metrics.auc(fpr, tpr)
    


def compute_KS_ym5w():
    file_path = '/Users/qizhi.zqz/Documents/dataset/embed_op_fea_5w_format_y_test.csv'

    predict_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/output/predict'
    #predict_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/output/predict'

    #predict_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stensorflow/ml/nn/networks/id_and_pred'
    #predict_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/stf/qqq/predict'

    y = pd.read_csv(file_path, names=["id", "loan_date", "label"], index_col=["id","loan_date"], header=0)
    
    
    #y = pd.read_csv(file_path + "/gaode_3w_y.csv", index_col=["id", "ent_date"])


    #y_hat = pd.read_csv(predict_path + "tfe/qqq/predict", index_col=["id", "ent_date"])
    y_hat = pd.read_csv(predict_path, header=None, names=["id","loan_date","predict"], index_col=["id","loan_date"])
    
    

    df = y.join(y_hat).dropna()

    #df=pd.concat([y, y_hat], axis=1).dropna()
    
    

    y = df.loc[:, 'label']
    
    y_hat = df.loc[:, 'predict']
    

    y = np.array(y)
    y_hat = np.array(y_hat)

    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)
    auc = metrics.roc_auc_score(y, y_hat)
    print("AUC=", auc)
    # KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)
    print("KS=", KS)

def compute_KS_ymbig():
    file_path = '/morse-stf/datasets/mpc_rta_sampleset.20210307.test.cvs'

    predict_path = './predict'
    #predict_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/output/predict'
    #predict_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stensorflow/ml/nn/networks/id_and_pred'
    #predict_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/output/predict'
    #predict_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stensorflow/ml/nn/networks/id_and_pred'
    #predict_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/tfe/qqq/predict'

    y = pd.read_csv(file_path, names=["id", "loan_date", "label"], index_col=["id"], header=0)
    
    
    #y = pd.read_csv(file_path + "/gaode_3w_y.csv", index_col=["id", "ent_date"])


    #y_hat = pd.read_csv(predict_path + "tfe/qqq/predict", index_col=["id", "ent_date"])
    y_hat = pd.read_csv(predict_path, header=None, names=["id", "predict"], index_col=["id"])
    
    

    df = y.join(y_hat).dropna()

    #df=pd.concat([y, y_hat], axis=1).dropna()
    
    

    y = df.loc[:, 'label']
    
    y_hat = df.loc[:, 'predict']
    

    y = np.array(y)
    y_hat = np.array(y_hat)

    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)
    auc = metrics.roc_auc_score(y, y_hat)
    
    # KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)
    


def compute_KS_ym10w1k5():
    file_path = '/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted/file/qqq/data'

    predict_path = '/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted/'

    y = pd.read_csv(file_path+"/10w1k5col_y.csv", index_col=["oneid", "loan_date"])

    y_hat = pd.read_csv(predict_path+"tfe/qqq/predict", header=None, names=["oneid","loan_date","predict"], index_col=["oneid","loan_date"])
    df = y.join(y_hat)
    # df=pd.concat([x, y], axis=1)
    

    y = df.loc[:, 'label']
    
    y_hat = df.loc[:, 'predict']
    

    y = np.array(y)
    y_hat = np.array(y_hat)

    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)

    # KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)
    


def compute_KS_xd():
    file_path = "../dataset/xindai_xy_test.csv"
    #file_path = '/Users/guanshun/PycharmProjects/morse-stf/stf_keeper/xindai_xy_shuffle.csv'

    predict_path = '/Users/qizhi.zqz/projects/Antchain-MPC/morse-stf/output/predict'
    #predict_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/tfe/qqq/predict'

    y = pd.read_csv(file_path, index_col=["id"])
    
    #y = pd.read_csv(file_path + "/gaode_3w_y.csv", index_col=["id", "ent_date"])


    #y_hat = pd.read_csv(predict_path + "tfe/qqq/predict", index_col=["id", "ent_date"])
    y_hat = pd.read_csv(predict_path, header=None, names=["id","predict"], index_col=["id"])
    
    #plt.hist(1/(1+np.exp(-y_hat)))
    plt.hist(y_hat)
    plt.show()

    df = y.join(y_hat).dropna(axis=0)
    # df=pd.concat([x, y], axis=1)
    

    y = df.loc[:, 'y']
    
    y_hat = df.loc[:, 'predict']
    

    y = np.array(y)
    y_hat = np.array(y_hat)

    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)

    # KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)
    

    AUC=metrics.auc(fpr, tpr)
    print("KS=", KS)
    print("AUC=", AUC)
    



def compute_KS_company():
    file_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/company_datasettest.csv'
    #file_path = '/Users/guanshun/PycharmProjects/morse-stf/stf_keeper/xindai_xy_shuffle.csv'

    predict_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/tfe/qqq/predict'

    y = pd.read_csv(file_path, index_col=["id"])
    

    #y = pd.read_csv(file_path + "/gaode_3w_y.csv", index_col=["id", "ent_date"])


    #y_hat = pd.read_csv(predict_path + "tfe/qqq/predict", index_col=["id", "ent_date"])
    y_hat = pd.read_csv(predict_path, header=None, names=["id","predict"], index_col=["id"])
    df = y.join(y_hat)
    # df=pd.concat([x, y], axis=1)
    

    y = df.loc[:, 'next_negative_income']
    
    
    y_hat = df.loc[:, 'predict']
    

    y = np.array(y)
    y_hat = np.array(y_hat)

    precision, recall, thresholds = metrics.precision_recall_curve(y, y_hat)
    #precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y, y_hat, average='binary')
    
    i=np.argmax(precision+recall)
    



    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)

    auc = metrics.roc_auc_score(y, y_hat)

    # KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)
    
    


def compute_KS_company_gbdtcode():
    file_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/company_datatest_gbdtcode.csv'
    #file_path = '/Users/guanshun/PycharmProjects/morse-stf/stf_keeper/xindai_xy_shuffle.csv'

    predict_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/tfe/qqq/predict'

    y = pd.read_csv(file_path, index_col=["row_id"])
    

    #y = pd.read_csv(file_path + "/gaode_3w_y.csv", index_col=["id", "ent_date"])


    #y_hat = pd.read_csv(predict_path + "tfe/qqq/predict", index_col=["id", "ent_date"])
    y_hat = pd.read_csv(predict_path, header=None, names=["id","predict"], index_col=["id"])
    df = y.join(y_hat)
    # df=pd.concat([x, y], axis=1)
    

    y = df.loc[:, '351']
    
    
    y_hat = df.loc[:, 'predict']
    

    y = np.array(y)
    y_hat = np.array(y_hat)



    precision, recall, thresholds = metrics.precision_recall_curve(y, y_hat)
    #precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y, y_hat, average='binary')


    F1 = 2*precision*recall/(1E-10+precision+recall)
    i=np.argmax(F1)
    


    accuracy = metrics.accuracy_score(y, (y_hat>thresholds[i]).astype("int"))

    



    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)

    auc = metrics.roc_auc_score(y, y_hat)

    # KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)
    
    

def compute_trun_KS_company_gbdtcode():
    file1_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/company_data1_gbdtcode.csv'
    file2_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/company_data2_gbdtcode.csv'
    file_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/company_data1u2_gbdtcode.csv'

    predict_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/tfe/qqq/predict'

    y = pd.read_csv(file_path, index_col=["row_id"])
    

    #y = pd.read_csv(file_path + "/gaode_3w_y.csv", index_col=["id", "ent_date"])


    #y_hat = pd.read_csv(predict_path + "tfe/qqq/predict", index_col=["id", "ent_date"])
    y_hat = pd.read_csv(predict_path, header=None, names=["id","predict"], index_col=["id"])
    df = y.join(y_hat)
    # df=pd.concat([x, y], axis=1)
    

    y = df.loc[:, '351']
    
    
    y_hat = df.loc[:, 'predict']
    

    y = np.array(y)
    y_hat = np.array(y_hat)



    precision, recall, thresholds = metrics.precision_recall_curve(y, y_hat)
    #precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y, y_hat, average='binary')


    F1 = 2*precision*recall/(1E-10+precision+recall)
    i=np.argmax(F1)
    


    accuracy = metrics.accuracy_score(y, (y_hat>thresholds[i]).astype("int"))

    



    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)

    auc = metrics.roc_auc_score(y, y_hat)

    # KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)
    
    


def compute_KS_elec_gbdtcode():
    #file_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/elec_datatest_gbdtcode.csv'
    #file_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/elec_data1_gbdtcode.csv'
    #file_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/elec_data2_gbdtcode.csv'
    file_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/elec_data1u2_gbdtcode.csv'

    predict_path = '/Users/qizhi.zqz/projects/morse-stf/morse-stf/stf_keeper/tfe/qqq/predict'

    y = pd.read_csv(file_path, index_col=["row_id"])
    

    #y = pd.read_csv(file_path + "/gaode_3w_y.csv", index_col=["id", "ent_date"])


    #y_hat = pd.read_csv(predict_path + "tfe/qqq/predict", index_col=["id", "ent_date"])
    y_hat = pd.read_csv(predict_path, header=None, names=["id","predict"], index_col=["id"])
    df = y.join(y_hat)
    # df=pd.concat([x, y], axis=1)
    

    y = df.loc[:, '273']
    
    
    y_hat = df.loc[:, 'predict']
    

    y = np.array(y)
    y_hat = np.array(y_hat)



    precision, recall, thresholds = metrics.precision_recall_curve(y, y_hat)
    #precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y, y_hat, average='binary')


    F1 = 2*precision*recall/(1E-10+precision+recall)
    i=np.argmax(F1)
    


    accuracy = metrics.accuracy_score(y, (y_hat>thresholds[i]).astype("int"))

    



    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)

    auc = metrics.roc_auc_score(y, y_hat)

    # KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)
    

def compute_precision_mnist():
    file_path = "../output/predict"
    label_pred = pd.read_csv(file_path, header=None, names=["label"]+["score{}".format(i) for i in range(10)])
    pred = np.array(label_pred.iloc[:,1:])
    label = label_pred.iloc[:,0]
    pred = np.argmax(pred, axis=1)
    print(np.mean(pred==label))


def compute_KS_epsilon():
    file_path = "../dataset/epsilon_normalized_test_y"

    predict_path = '/Users/qizhi.zqz/projects/Antchain-MPC/morse-stf/output/predict'

    y = pd.read_csv(file_path, header=None, names=["y"])

    y_hat = pd.read_csv(predict_path, header=None, names=["predict"])

    df=pd.concat([y, y_hat], axis=1).dropna(axis=0)
    print("df=", df)
    y = df.loc[:, 'y']

    y_hat = df.loc[:, 'predict']

    y = np.array(y)
    y_hat = np.array(y_hat)

    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)

    # KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)

    AUC = metrics.auc(fpr, tpr)
    print("KS=", KS)
    print("AUC=", AUC)


if __name__=='__main__':
    #compute_KS_elec_gbdtcode()
    #compute_KS_ym5w()
    #compute_KS_xd()
    compute_KS_epsilon()
    #compute_KS_gaode3w()
    #compute_KS_gaode20w()
    #compute_KS_gaode20w_DNN()
    #compute_trun_KS_company_gbdtcode()
    #compute_precision_mnist()
