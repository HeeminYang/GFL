# 경로에서 폴더 리스트를 가져온다.
# 폴더의 가장 최신 파일을 가져온다.
# 파일을 읽어서 log파일을 불러온다.
# 폴더의 이름을 통해 정보를 파악한다. 
# 예시) GFL_FMNIST_c10_m1_ps0.2_wrmp4_rd100_g20_d5_clf1_th0.25_sd1 일때, 
#      c10은 class가 10개라는 뜻이다. m1은 malicious client가 1개다.
#      즉 benign client가 9개고 1~9번 클라이언트가 benign client다.
#      10번 클라이언트가 malicious client다.
#      GFL_FMNIST_c10_m7_ps0.2_wrmp4_rd100_g20_d5_clf1_th0.25_sd1 일때,
#      c10은 class가 10개라는 뜻이다. m7은 malicious client가 7개다.
#      즉 benign client가 3개고 1~3번 클라이언트가 benign client다.
#      4~10번 클라이언트가 malicious client다.
# 해당 정보를 임시로 저장해둔다.
# log 파일 안에 "Round 100"이 있는 줄을 시작지점으로 정한다.
# 이후 두 부분으로 나눠 데이터를 수집할것이다.
# 빈 리스트를 만들어서 데이터를 수집한다.
# 데이터는 사전형태로 Key는 Accuracy, Loss, ASR1, ASR2, Recall1, Recall2와 추가로 해당 클라이언트가 malicious client여부를 알려주는 malicious이다. 해당 Key는 빈 리스트를 Value로 가진다.
# 위 사전을 2개 만든다. Aggregation test와 external test를 위한 사전이다.
# 1. Aggregation test
# 시작지점 이후부터 "Aggregation test"가 있는 줄을 찾는다.
# 이후 line부터
# Client 1, 2, 3, 4, 5, 6, 7, 8, 9, 10의 데이터를 수집한다.
# 데이터는 Accuracy, Loss, ASR1, ASR2, Recall1, Recall2이다.
# 해당 구조는 예시) 
# Aggregation test
#  Client1 Accuracy: 0.90451 Loss: 0.26783 ASR1: 0.12500 ASR2: 0.09917 Recall1: 0.63281 Recall2: 0.85124
#    Summary of misclassifications:
#    True label 0 was misclassified as: OrderedDict([(1, 2), (3, 3), (4, 1), (6, 12)])
#    True label 1 was misclassified as: OrderedDict([(3, 1), (4, 2)])
#    True label 2 was misclassified as: OrderedDict([(0, 2), (3, 1), (4, 10), (6, 5)])
#    True label 3 was misclassified as: OrderedDict([(1, 1), (2, 1), (4, 2), (6, 3)])
#    True label 4 was misclassified as: OrderedDict([(2, 3), (3, 3), (6, 1), (8, 1)])
#    True label 5 was misclassified as: OrderedDict([(7, 1)])
#    True label 6 was misclassified as: OrderedDict([(0, 16), (2, 14), (3, 4), (4, 11), (8, 1), (9, 1)])
#    True label 7 was misclassified as: OrderedDict([(5, 1), (9, 2)])
#    True label 8 was misclassified as: OrderedDict([(5, 2)])
#    True label 9 was misclassified as: OrderedDict([(7, 3)])
#  Client2 Accuracy: 0.90972 Loss: 0.25626 ASR1: 0.07258 ASR2: 0.05556 Recall1: 0.73387 Recall2: 0.88889
#    Summary of misclassifications:
#    True label 0 was misclassified as: OrderedDict([(2, 3), (3, 1), (5, 1), (6, 6), (8, 1)])
#    True label 1 was misclassified as: OrderedDict([(0, 1), (3, 2)])
#    True label 2 was misclassified as: OrderedDict([(0, 2), (3, 1), (4, 6), (6, 8)])
#    True label 3 was misclassified as: OrderedDict([(0, 3), (1, 1), (2, 2), (4, 1), (6, 2)])
#    True label 4 was misclassified as: OrderedDict([(2, 10), (3, 1), (6, 5)])
#    True label 5 was misclassified as: OrderedDict([(7, 1)])
#    True label 6 was misclassified as: OrderedDict([(0, 9), (2, 13), (3, 4), (4, 6), (8, 1)])
#    True label 7 was misclassified as: OrderedDict([(5, 3), (9, 3)])
#    True label 8 was misclassified as: OrderedDict([(0, 1), (3, 1)])
#    True label 9 was misclassified as: OrderedDict([(7, 5)])
#  Client3 Accuracy: 0.90799 Loss: 0.26293 ASR1: 0.15000 ASR2: 0.07563 Recall1: 0.70833 Recall2: 0.86555
#    Summary of misclassifications:
#    True label 0 was misclassified as: OrderedDict([(2, 1), (3, 5), (4, 1), (6, 9)])
#    True label 1 was misclassified as: OrderedDict()
#    True label 2 was misclassified as: OrderedDict([(0, 1), (3, 1), (4, 4), (6, 5)])
#    True label 3 was misclassified as: OrderedDict([(0, 1), (2, 1), (4, 3), (6, 7)])
#    True label 4 was misclassified as: OrderedDict([(2, 9), (3, 5), (6, 7)])
#    True label 5 was misclassified as: OrderedDict([(7, 2), (9, 2)])
#    True label 6 was misclassified as: OrderedDict([(0, 18), (1, 1), (2, 5), (3, 4), (4, 7)])
#    True label 7 was misclassified as: OrderedDict([(5, 1), (9, 2)])
#    True label 8 was misclassified as: OrderedDict([(5, 1), (6, 2)])
#    True label 9 was misclassified as: OrderedDict([(7, 1)])
#  Client4 Accuracy: 0.91753 Loss: 0.25200 ASR1: 0.10185 ASR2: 0.06202 Recall1: 0.72222 Recall2: 0.90698
#    Summary of misclassifications:
#    True label 0 was misclassified as: OrderedDict([(2, 1), (3, 1), (6, 8), (8, 2)])
#    True label 1 was misclassified as: OrderedDict([(3, 1), (6, 1)])
#    True label 2 was misclassified as: OrderedDict([(3, 1), (4, 4), (6, 3)])
#    True label 3 was misclassified as: OrderedDict([(2, 1), (4, 5), (6, 3)])
#    True label 4 was misclassified as: OrderedDict([(2, 10), (3, 6), (6, 5)])
#    True label 5 was misclassified as: OrderedDict([(7, 3)])
#    True label 6 was misclassified as: OrderedDict([(0, 11), (2, 6), (3, 3), (4, 10)])
#    True label 7 was misclassified as: OrderedDict([(5, 1), (9, 3)])
#    True label 8 was misclassified as: OrderedDict([(2, 1), (7, 1)])
#    True label 9 was misclassified as: OrderedDict([(5, 1), (7, 3)])
#  Client5 Accuracy: 0.91580 Loss: 0.24511 ASR1: 0.08403 ASR2: 0.08257 Recall1: 0.72269 Recall2: 0.86239
#    Summary of misclassifications:
#    True label 0 was misclassified as: OrderedDict([(2, 3), (3, 2), (6, 9), (8, 1)])
#    True label 1 was misclassified as: OrderedDict([(3, 2)])
#    True label 2 was misclassified as: OrderedDict([(3, 2), (4, 3), (6, 8)])
#    True label 3 was misclassified as: OrderedDict([(0, 4), (4, 2), (6, 1)])
#    True label 4 was misclassified as: OrderedDict([(2, 3), (3, 1), (6, 3)])
#    True label 5 was misclassified as: OrderedDict([(7, 3), (8, 2), (9, 1)])
#    True label 6 was misclassified as: OrderedDict([(0, 10), (1, 1), (2, 10), (3, 3), (4, 9)])
#    True label 7 was misclassified as: OrderedDict([(5, 2), (9, 1)])
#    True label 8 was misclassified as: OrderedDict([(3, 1), (4, 1), (5, 1), (6, 1), (9, 1)])
#    True label 9 was misclassified as: OrderedDict([(5, 2), (7, 4)])
#  Client6 Accuracy: 0.90104 Loss: 0.27555 ASR1: 0.15126 ASR2: 0.08065 Recall1: 0.63025 Recall2: 0.87903
#    Summary of misclassifications:
#    True label 0 was misclassified as: OrderedDict([(2, 3), (3, 2), (6, 10)])
#    True label 1 was misclassified as: OrderedDict([(3, 2), (4, 1)])
#    True label 2 was misclassified as: OrderedDict([(0, 2), (4, 6), (6, 3)])
#    True label 3 was misclassified as: OrderedDict([(0, 2), (1, 1), (2, 1), (4, 3), (6, 1)])
#    True label 4 was misclassified as: OrderedDict([(1, 1), (2, 7), (3, 3), (6, 9), (8, 1)])
#    True label 5 was misclassified as: OrderedDict([(7, 1), (9, 1)])
#    True label 6 was misclassified as: OrderedDict([(0, 18), (2, 9), (3, 2), (4, 15)])
#    True label 7 was misclassified as: OrderedDict([(9, 4)])
#    True label 8 was misclassified as: OrderedDict([(5, 1), (7, 1)])
#    True label 9 was misclassified as: OrderedDict([(5, 2), (7, 2)])
#  Client7 Accuracy: 0.89323 Loss: 0.29817 ASR1: 0.16102 ASR2: 0.04425 Recall1: 0.62712 Recall2: 0.91150
#    Summary of misclassifications:
#    True label 0 was misclassified as: OrderedDict([(2, 2), (3, 1), (5, 1), (6, 5), (8, 1)])
#    True label 1 was misclassified as: OrderedDict([(3, 1)])
#    True label 2 was misclassified as: OrderedDict([(0, 3), (4, 10), (6, 9)])
#    True label 3 was misclassified as: OrderedDict([(0, 1), (1, 2), (2, 1), (4, 5), (6, 1)])
#    True label 4 was misclassified as: OrderedDict([(1, 1), (2, 8), (3, 2), (6, 10), (8, 1)])
#    True label 5 was misclassified as: OrderedDict([(7, 5)])
#    True label 6 was misclassified as: OrderedDict([(0, 19), (1, 1), (2, 12), (3, 4), (4, 8)])
#    True label 7 was misclassified as: OrderedDict([(5, 2), (9, 2)])
#    True label 8 was misclassified as: OrderedDict([(3, 1), (6, 1)])
#    True label 9 was misclassified as: OrderedDict([(7, 3)])
#  Client8 Accuracy: 0.89497 Loss: 0.31026 ASR1: 0.08772 ASR2: 0.14953 Recall1: 0.86842 Recall2: 0.67290
#    Summary of misclassifications:
#    True label 0 was misclassified as: OrderedDict([(2, 8), (3, 3), (4, 6), (6, 16), (8, 2)])
#    True label 1 was misclassified as: OrderedDict([(3, 1)])
#    True label 2 was misclassified as: OrderedDict([(0, 11), (1, 1), (4, 7), (6, 3)])
#    True label 3 was misclassified as: OrderedDict([(0, 3), (1, 2), (4, 9), (6, 1), (8, 1)])
#    True label 4 was misclassified as: OrderedDict([(0, 7), (2, 8), (3, 2)])
#    True label 5 was misclassified as: OrderedDict([(7, 3), (9, 1)])
#    True label 6 was misclassified as: OrderedDict([(0, 10), (2, 1), (3, 3), (8, 1)])
#    True label 7 was misclassified as: OrderedDict([(5, 3), (9, 2)])
#    True label 8 was misclassified as: OrderedDict([(1, 1), (6, 2)])
#    True label 9 was misclassified as: OrderedDict([(5, 2), (7, 1)])
#  Client9 Accuracy: 0.89844 Loss: 0.29218 ASR1: 0.05155 ASR2: 0.15517 Recall1: 0.90722 Recall2: 0.65517
#    Summary of misclassifications:
#    True label 0 was misclassified as: OrderedDict([(2, 11), (3, 3), (4, 7), (6, 18), (8, 1)])
#    True label 1 was misclassified as: OrderedDict([(2, 1)])
#    True label 2 was misclassified as: OrderedDict([(0, 4), (3, 1), (4, 14), (6, 7)])
#    True label 3 was misclassified as: OrderedDict([(0, 1), (2, 1), (4, 2), (6, 3)])
#    True label 4 was misclassified as: OrderedDict([(0, 9), (2, 1), (3, 6), (6, 1)])
#    True label 5 was misclassified as: OrderedDict([(7, 4)])
#    True label 6 was misclassified as: OrderedDict([(0, 5), (2, 1), (3, 2), (8, 1)])
#    True label 7 was misclassified as: OrderedDict([(5, 3), (9, 2)])
#    True label 8 was misclassified as: OrderedDict([(4, 1), (5, 1)])
#    True label 9 was misclassified as: OrderedDict([(5, 2), (7, 4)])
#  Client10 Accuracy: 0.89497 Loss: 0.30374 ASR1: 0.09244 ASR2: 0.16364 Recall1: 0.88235 Recall2: 0.68182
#    Summary of misclassifications:
#    True label 0 was misclassified as: OrderedDict([(2, 6), (3, 2), (4, 9), (6, 18)])
#    True label 1 was misclassified as: OrderedDict([(0, 1), (3, 4)])
#    True label 2 was misclassified as: OrderedDict([(0, 3), (1, 1), (3, 2), (4, 12), (6, 1)])
#    True label 3 was misclassified as: OrderedDict([(0, 1), (1, 1), (2, 2), (4, 4), (6, 4)])
#    True label 4 was misclassified as: OrderedDict([(0, 9), (2, 9), (3, 2)])
#    True label 5 was misclassified as: OrderedDict([(7, 5), (8, 1)])
#    True label 6 was misclassified as: OrderedDict([(0, 11), (3, 3)])
#    True label 7 was misclassified as: OrderedDict([(9, 2)])
#    True label 8 was misclassified as: OrderedDict([(0, 1), (3, 1), (5, 1), (6, 1)])
#    True label 9 was misclassified as: OrderedDict([(7, 4)])
# 첫째로, "  Client10 Accuracy: 0.89497 Loss: 0.30374 ASR1: 0.09244 ASR2: 0.16364 Recall1: 0.88235 Recall2: 0.68182"와 같은 구조의 라인을 찾는다.
# 둘째로, 각 데이터를 각각의 리스트에 넣을건데, 위에서 구한 클라이언트의 악성 정보도 같이 넣어준다. 해당 클라이언트가 malicious client면 1, benign client면 0이다.
# 예시) "  Client10 Accuracy: 0.89497 Loss: 0.30374 ASR1: 0.09244 ASR2: 0.16364 Recall1: 0.88235 Recall2: 0.68182"의 경우에
# Aggregation test 결과를 저장하는 사전의 Accuracy 리스트에는 0.89497를 넣고, Loss 리스트에는 0.30374를 넣고, ASR1 리스트에는 0.09244를 넣고, ASR2 리스트에는 0.16364를 넣고, Recall1 리스트에는 0.88235를 넣고, Recall2 리스트에는 0.68182를 넣는다.
# Malicious 리스트에는 1을 넣는다. 왜냐하면 해당 클라이언트는 malicious client이기 때문이다. 정상이라면 0이다. 이름을 통해 저장한 정보를 참고해서 malicious인지 benign인지 알 수 있다.
# 1번이 진행된 이후부터 "external test"가 있는 줄을 찾는다. 찾은 라인이 1번의 종료다.
# 2. external test
# 시작지점 이후부터 "external test"가 있는 줄을 찾는다.
# 이후 line부터
# 1번과 같은 방식으로 데이터를 수집한다. 대신, external test 결과를 저장하는 사전에 데이터를 저장한다.
# 다음 폴더로 넘어간다.
# 이 과정을 반복한다. 새로운 사전을 만들 필요는 없다. 기존의 사전에 데이터를 추가해주면 된다.
# 모든 폴더를 다 돌았으면, 각 리스트를 DataFrame으로 만들어준다.
# DataFrame을 csv파일로 저장한다.

import os
import pandas as pd
import re
import numpy as np

# 경로에서 폴더 리스트를 가져온다.
def get_folder_list(path):
    folder_list = os.listdir(path)
    return folder_list

# 폴더의 가장 최신 파일을 가져온다.
def get_recent_file(path):
    file_list = os.listdir(path)
    file_list.sort()
    recent_file = file_list[-1]
    return recent_file

# 파일을 읽어서 log파일을 불러온다.
def read_log(path, file):
    log_path = path + "/" + file
    f = open(log_path, 'r')
    log = f.readlines()
    return log

# 폴더의 이름을 통해 정보를 파악한다.
def get_info(folder_name):
    info = folder_name.split("_")
    return info

# 해당 정보를 임시로 저장해둔다.
def save_info(info):
    client = int(info[2].split("c")[1])
    malicious = int(info[3].split("m")[1])
    benign = client - malicious
    info_dict = {}
    # if malicious is 2 than [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    # if malicious is 3 than [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    for i in range(1, client+1):
        if i <= benign:
            info_dict[i] = 0
        else:
            info_dict[i] = 1

    return info_dict

# log 파일 안에 "Round 200"이 있는 줄을 시작지점으로 정한다.
def get_start_point(log):
    for i in range(len(log)):
        if "Round 100" in log[i]:
            sp = i
            break
    return sp

def get_agg_start(start_point, log):
    for i in range(start_point, len(log)):
        if "Aggregation test" in log[i]:
            agg_start_point = i
            break
    for i in range(agg_start_point, len(log)):
        if "external test" in log[i]:
            agg_end_point = i
            break
    return agg_start_point, agg_end_point

def get_ext_start(start_point, log):
    for i in range(start_point, len(log)):
        if "external test" in log[i]:
            ext_start_point = i
            break
    for i in range(ext_start_point, len(log)):
        if "End time" in log[i]:
            ext_end_point = i
            break
    return ext_start_point, ext_end_point

# 이후 두 부분으로 나눠 데이터를 수집할것이다.
# 빈 리스트를 만들어서 데이터를 수집한다.
# 데이터는 사전형태로 Key는 Accuracy, Loss, ASR1, ASR2, Recall1, Recall2와 추가로 해당 클라이언트가 malicious client여부를 알려주는 malicious이다. 해당 Key는 빈 리스트를 Value로 가진다.
# 위 사전을 2개 만든다. Aggregation test와 external test를 위한 사전이다.
def make_dict():
    agg_dict = {}
    agg_dict["Accuracy"] = []
    agg_dict["Loss"] = []
    agg_dict["ASR1"] = []
    agg_dict["ASR2"] = []
    agg_dict["Recall1"] = []
    agg_dict["Recall2"] = []
    agg_dict["malicious"] = []
    ext_dict = {}
    ext_dict["Accuracy"] = []
    ext_dict["Loss"] = []
    ext_dict["ASR1"] = []
    ext_dict["ASR2"] = []
    ext_dict["Recall1"] = []
    ext_dict["Recall2"] = []
    ext_dict["malicious"] = []
    return agg_dict, ext_dict

# 1. Aggregation test
# 시작지점 이후부터 "Aggregation test"가 있는 줄을 찾는다.
# 이후 line부터
# Client 1, 2, 3, 4, 5, 6, 7, 8, 9, 10의 데이터를 수집한다.
# 데이터는 Accuracy, Loss, ASR1, ASR2, Recall1, Recall2이다.
# 1번이 진행된 이후부터 "external test"가 있는 줄을 찾는다. 찾은 라인이 1번의 종료다.
def agg_test(log, start_point, end_point, agg_dict, info_dict):
    agg_dict = agg_dict
    for i in range(start_point, end_point):
        if "Client" in log[i+1]:
            client = int(log[i+1].split(" ")[1].replace("Client", ""))
            agg_dict["Accuracy"].append(float(log[i+1].split(" ")[3]))
            agg_dict["Loss"].append(float(log[i+1].split(" ")[5]))
            agg_dict["ASR1"].append(float(log[i+1].split(" ")[7]))
            agg_dict["ASR2"].append(float(log[i+1].split(" ")[9]))
            agg_dict["Recall1"].append(float(log[i+1].split(" ")[11]))
            agg_dict["Recall2"].append(float(log[i+1].split(" ")[13]))
            agg_dict["malicious"].append(int(info_dict[client]))
    return agg_dict

# 2. external test
# 시작지점 이후부터 "external test"가 있는 줄을 찾는다.
# 이후 line부터
# 1번과 같은 방식으로 데이터를 수집한다. 대신, external test 결과를 저장하는 사전에 데이터를 저장한다.
def ext_test(log, start_point, end_point, ext_dict, info_dict):
    ext_dict = ext_dict
    for i in range(start_point, end_point):
        if "Client" in log[i+1]:
            client = int(log[i+1].split(" ")[1].replace("Client", ""))
            ext_dict["Accuracy"].append(float(log[i+1].split(" ")[3]))
            ext_dict["Loss"].append(float(log[i+1].split(" ")[5]))
            ext_dict["ASR1"].append(float(log[i+1].split(" ")[7]))
            ext_dict["ASR2"].append(float(log[i+1].split(" ")[9]))
            ext_dict["Recall1"].append(float(log[i+1].split(" ")[11]))
            ext_dict["Recall2"].append(float(log[i+1].split(" ")[13]))
            ext_dict["malicious"].append(int(info_dict[client]))
    return ext_dict

# 첫째로, "  Client10 Accuracy: 0.89497 Loss: 0.30374 ASR1: 0.09244 ASR2: 0.16364 Recall1: 0.88235 Recall2: 0.68182"와 같은 구조의 라인을 찾는다.
# 둘째로, 각 데이터를 각각의 리스트에 넣을건데, 위에서 구한 클라이언트의 악성 정보도 같이 넣어준다. 해당 클라이언트가 malicious client면 1, benign client면 0이다.
# 예시) "  Client10 Accuracy: 0.89497 Loss: 0.30374 ASR1: 0.09244 ASR2: 0.16364 Recall1: 0.88235 Recall2: 0.68182"의 경우에
# Aggregation test 결과를 저장하는 사전의 Accuracy 리스트에는 0.89497를 넣고, Loss 리스트에는 0.30374를 넣고, ASR1 리스트에는 0.09244를 넣고, ASR2 리스트에는 0.16364를 넣고, Recall1 리스트에는 0.88235를 넣고, Recall2 리스트에는 0.68182를 넣는다.
# Malicious 리스트에는 1을 넣는다. 왜냐하면 해당 클라이언트는 malicious client이기 때문이다. 정상이라면 0이다. 이름을 통해 저장한 정보를 참고해서 malicious인지 benign인지 알 수 있다.

# 모든 폴더를 다 돌았으면, 각 리스트를 DataFrame으로 만들어준다.
def make_df(agg_dict, ext_dict):
    agg_df = pd.DataFrame(agg_dict)
    ext_df = pd.DataFrame(ext_dict)
    return agg_df, ext_df

# DataFrame을 csv파일로 저장한다.
def save_csv(agg_df, ext_df):
    agg_df.to_csv("agg.csv", index=False)
    ext_df.to_csv("ext.csv", index=False)

# main
def main():
    path = "/home/heemin/GFL/formula_log"
    folder_list = get_folder_list(path)
    agg_dict, ext_dict = make_dict()
    for folder in folder_list:
        info = get_info(folder)
        info_dict = save_info(info)
        file = get_recent_file(path + "/" + folder)
        log = read_log(path + "/" + folder, file)
        start_point = get_start_point(log)
        agg_start_point, agg_end_point = get_agg_start(start_point, log)
        ext_start_point, ext_end_point = get_ext_start(start_point, log)
        agg_dict = agg_test(log, agg_start_point, agg_end_point, agg_dict, info_dict)
        ext_dict = ext_test(log, ext_start_point, ext_end_point, ext_dict, info_dict)
    agg_df, ext_df = make_df(agg_dict, ext_dict)
    save_csv(agg_df, ext_df)

if __name__ == "__main__":
    main()
