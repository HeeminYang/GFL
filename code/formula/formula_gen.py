import os
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
import graphviz

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
        if "Neighbor Generator test" in log[i]:
            agg_start_point = i
            break
    for i in range(agg_start_point, len(log)):
        if "Changed Adjacency Matrix: " in log[i]:
            agg_end_point = i
            break
    return agg_start_point, agg_end_point

# 이후 두 부분으로 나눠 데이터를 수집할것이다.
# 빈 리스트를 만들어서 데이터를 수집한다.
# 데이터는 사전형태로 Key는 Accuracy, Loss, ASR1, ASR2, Recall1, Recall2와 추가로 해당 클라이언트가 malicious client여부를 알려주는 malicious이다. 해당 Key는 빈 리스트를 Value로 가진다.
# 위 사전을 2개 만든다. Aggregation test와 external test를 위한 사전이다.
def make_dict():
    agg_dict = {}
    agg_dict["Accuracy"] = []
    agg_dict["Loss"] = []
    for i in range(10):
        agg_dict[f'Loss{i}'] = []
    for i in range(10):
        agg_dict[f'Recall{i}'] = []
    agg_dict[f'Recall'] = []
    agg_dict["malicious"] = []
    return agg_dict

# 1. Aggregation test
# 시작지점 이후부터 "Aggregation test"가 있는 줄을 찾는다.
# 이후 line부터
# Client 1, 2, 3, 4, 5, 6, 7, 8, 9, 10의 데이터를 수집한다.
# 데이터는 Accuracy, Loss, ASR1, ASR2, Recall1, Recall2이다.
# 1번이 진행된 이후부터 "external test"가 있는 줄을 찾는다. 찾은 라인이 1번의 종료다.
def agg_test(log, start_point, end_point, agg_dict, info_dict):
    agg_dict = agg_dict
    switch = True
    for i in range(start_point, end_point):
        if "Client" in log[i+1]:
            client = int(log[i+1].split(" ")[1].replace("Client", "").replace(":\n",""))
            if info_dict[client] == 1:
                switch = False
            else:
                switch = True
        if switch: 
            if ("Generator" in log[i+1]) and ('misclassifications' not in log[i+1]):
                recall = 0
                Gen = int(log[i+1].split(" ")[3])
                loss = 0
                agg_dict["Accuracy"].append(float(log[i+1].split(" ")[5].replace(",", "")))
                loss += float(log[i+1].split(" ")[8].replace(",", ""))
                agg_dict['Loss0'].append(float(log[i+1].split(" ")[8].replace(",", "")))
                loss += float(log[i+1].split(" ")[10].replace(",", ""))
                agg_dict['Loss1'].append(float(log[i+1].split(" ")[10].replace(",", "")))
                loss += float(log[i+1].split(" ")[12].replace(",", ""))
                agg_dict['Loss2'].append(float(log[i+1].split(" ")[12].replace(",", "")))
                loss += float(log[i+1].split(" ")[14].replace(",", ""))
                agg_dict['Loss3'].append(float(log[i+1].split(" ")[14].replace(",", "")))
                loss += float(log[i+1].split(" ")[16].replace(",", ""))
                agg_dict['Loss4'].append(float(log[i+1].split(" ")[16].replace(",", "")))
                loss += float(log[i+1].split(" ")[18].replace(",", ""))
                agg_dict['Loss5'].append(float(log[i+1].split(" ")[18].replace(",", "")))
                loss += float(log[i+1].split(" ")[20].replace(",", ""))
                agg_dict['Loss6'].append(float(log[i+1].split(" ")[20].replace(",", "")))
                loss += float(log[i+1].split(" ")[22].replace(",", ""))
                agg_dict['Loss7'].append(float(log[i+1].split(" ")[22].replace(",", "")))
                loss += float(log[i+1].split(" ")[24].replace(",", ""))
                agg_dict['Loss8'].append(float(log[i+1].split(" ")[24].replace(",", "")))
                loss += float(log[i+1].split(" ")[26].replace("}\n", ""))
                agg_dict['Loss9'].append(float(log[i+1].split(" ")[26].replace("}\n", "")))
                agg_dict["Loss"].append((loss/10))
            if "True label 0" in log[i+1]:
                recall2 = 0
                pattern = len(log[i+1].split("("))
                if not pattern == 2:
                    for j in range(2, pattern):
                        recall2 += int(log[i+1].split("(")[j].replace(")","").replace("]","").split(', ')[1].replace(",", ""))
                agg_dict["Recall0"].append((100-recall2)/100)
                recall += recall2
            if "True label 1" in log[i+1]:
                recall2 = 0
                pattern = len(log[i+1].split("("))
                if not pattern == 2:
                    for j in range(2, pattern):
                        recall2 += int(log[i+1].split("(")[j].replace(")","").replace("]","").split(', ')[1].replace(",", ""))
                agg_dict["Recall1"].append((100-recall2)/100)
                recall += recall2
            if "True label 2" in log[i+1]:
                recall2 = 0
                pattern = len(log[i+1].split("("))
                if not pattern == 2:
                    for j in range(2, pattern):
                        recall2 += int(log[i+1].split("(")[j].replace(")","").replace("]","").split(', ')[1].replace(",", ""))
                agg_dict["Recall2"].append((100-recall2)/100)
                recall += recall2
            if "True label 3" in log[i+1]:
                recall2 = 0
                pattern = len(log[i+1].split("("))
                if not pattern == 2:
                    for j in range(2, pattern):
                        recall2 += int(log[i+1].split("(")[j].replace(")","").replace("]","").split(', ')[1].replace(",", ""))
                agg_dict["Recall3"].append((100-recall2)/100)
                recall += recall2
            if "True label 4" in log[i+1]:
                recall2 = 0
                pattern = len(log[i+1].split("("))
                if not pattern == 2:
                    for j in range(2, pattern):
                        recall2 += int(log[i+1].split("(")[j].replace(")","").replace("]","").split(', ')[1].replace(",", ""))
                agg_dict["Recall4"].append((100-recall2)/100)
                recall += recall2
            if "True label 5" in log[i+1]:
                recall2 = 0
                pattern = len(log[i+1].split("("))
                if not pattern == 2:
                    for j in range(2, pattern):
                        recall2 += int(log[i+1].split("(")[j].replace(")","").replace("]","").split(', ')[1].replace(",", ""))
                agg_dict["Recall5"].append((100-recall2)/100)
                recall += recall2
            if "True label 6" in log[i+1]:
                recall2 = 0
                pattern = len(log[i+1].split("("))
                if not pattern == 2:
                    for j in range(2, pattern):
                        recall2 += int(log[i+1].split("(")[j].replace(")","").replace("]","").split(', ')[1])
                agg_dict["Recall6"].append((100-recall2)/100)
                recall += recall2
            if "True label 7" in log[i+1]:
                recall2 = 0
                pattern = len(log[i+1].split("("))
                if not pattern == 2:
                    for j in range(2, pattern):
                        recall2 += int(log[i+1].split("(")[j].replace(")","").replace("]","").split(', ')[1])
                agg_dict["Recall7"].append((100-recall2)/100)
                recall += recall2
            if "True label 8" in log[i+1]:
                recall2 = 0
                pattern = len(log[i+1].split("("))
                if not pattern == 2:
                    for j in range(2, pattern):
                        recall2 += int(log[i+1].split("(")[j].replace(")","").replace("]","").split(', ')[1])
                agg_dict["Recall8"].append((100-recall2)/100)
                recall += recall2
            if "True label 9" in log[i+1]:
                recall2 = 0
                pattern = len(log[i+1].split("("))
                if not pattern == 2:
                    for j in range(2, pattern):
                        recall2 += int(log[i+1].split("(")[j].replace(")","").replace("]","").split(', ')[1])
                agg_dict["Recall9"].append((100-recall2)/100)
                recall += recall2
                agg_dict["Recall"].append((1000-recall)/1000)
                agg_dict["malicious"].append(info_dict[Gen])
    return agg_dict

# 첫째로, "  Client10 Accuracy: 0.89497 Loss: 0.30374 ASR1: 0.09244 ASR2: 0.16364 Recall1: 0.88235 Recall2: 0.68182"와 같은 구조의 라인을 찾는다.
# 둘째로, 각 데이터를 각각의 리스트에 넣을건데, 위에서 구한 클라이언트의 악성 정보도 같이 넣어준다. 해당 클라이언트가 malicious client면 1, benign client면 0이다.
# 예시) "  Client10 Accuracy: 0.89497 Loss: 0.30374 ASR1: 0.09244 ASR2: 0.16364 Recall1: 0.88235 Recall2: 0.68182"의 경우에
# Aggregation test 결과를 저장하는 사전의 Accuracy 리스트에는 0.89497를 넣고, Loss 리스트에는 0.30374를 넣고, ASR1 리스트에는 0.09244를 넣고, ASR2 리스트에는 0.16364를 넣고, Recall1 리스트에는 0.88235를 넣고, Recall2 리스트에는 0.68182를 넣는다.
# Malicious 리스트에는 1을 넣는다. 왜냐하면 해당 클라이언트는 malicious client이기 때문이다. 정상이라면 0이다. 이름을 통해 저장한 정보를 참고해서 malicious인지 benign인지 알 수 있다.

# 모든 폴더를 다 돌았으면, 각 리스트를 DataFrame으로 만들어준다.
def make_df(agg_dict):
    agg_df = pd.DataFrame(agg_dict)
    return agg_df

# DataFrame을 csv파일로 저장한다.
def save_csv(agg_df, file_name):
    agg_df.to_csv(f"{file_name}.csv", index=False)
    # dataframe have accuracy, loss, recall, malicious
    agg_df[['Accuracy', 'Loss', 'Recall', 'malicious']].to_csv(f"{file_name}_R.csv", index=False)
    agg_df[['Loss', 'Loss0', 'Recall0', 'malicious']].to_csv(f"{file_name}_R6.csv", index=False)

def dctree(data, file_name, seed):
    # Separating features and target variable
    X = data.drop('malicious', axis=1)
    y = data['malicious']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    # Creating a Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier()

    # Training the model
    dt_classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = dt_classifier.predict(X_test)

    # Evaluating the model
    classification_report_output = classification_report(y_test, y_pred)

    print(classification_report_output)

    # Exporting the decision tree as a dot file
    dot_data = export_graphviz(dt_classifier, out_file=None, 
                            feature_names=X.columns,  
                            class_names=['Non-Malicious', 'Malicious'],
                            filled=True, rounded=True, 
                            special_characters=True)

    # Generating the graph from dot data
    graph = graphviz.Source(dot_data)

    # Saving the graph to a PNG file
    png_graph_file_path = f'/home/heemin/GFL/{file_name}'
    graph.format = 'png'
    graph.render(png_graph_file_path)

# main
def main():
    path = "/home/heemin/GFL/formula_log"
    file_name = 'data_p020'
    folder_list = get_folder_list(path)
    agg_dict = make_dict()
    for folder in folder_list:
        info = get_info(folder)
        info_dict = save_info(info)
        file = get_recent_file(path + "/" + folder)
        log = read_log(path + "/" + folder, file)
        start_point = get_start_point(log)
        agg_start_point, agg_end_point = get_agg_start(start_point, log)
        agg_dict = agg_test(log, agg_start_point, agg_end_point, agg_dict, info_dict)
    agg_df = make_df(agg_dict)
    save_csv(agg_df, file_name)
    
    # for i in range(10):
    #     dctree(agg_df, file_name+f'_{i}', i)
    # dctree(agg_df, file_name, 42)
    # dctree(agg_df[['Accuracy', 'Loss', 'Recall', 'malicious']], file_name+'_R', 42)
    # dctree(agg_df[['Accuracy', 'Loss', 'Loss0', 'Loss6', 'Recall0', 'Recall6', 'malicious']], file_name+'_R6', 42)
    dctree(agg_df[['Accuracy', 'malicious']], file_name+'_Acc', 42)

if __name__ == "__main__":
    main()
