import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from tensorflow.keras.optimizers import Adam
import json

cols="""duration,
protocol_type,
service,
flag,
src_bytes,
dst_bytes,
land,
wrong_fragment,
urgent,
hot,
num_failed_logins,
logged_in,
num_compromised,
root_shell,
su_attempted,
num_root,
num_file_creations,
num_shells,
num_access_files,
num_outbound_cmds,
is_host_login,
is_guest_login,
count,
srv_count,
serror_rate,
srv_serror_rate,
rerror_rate,
srv_rerror_rate,
same_srv_rate,
diff_srv_rate,
srv_diff_host_rate,
dst_host_count,
dst_host_srv_count,
dst_host_same_srv_rate,
dst_host_diff_srv_rate,
dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate,
dst_host_serror_rate,
dst_host_srv_serror_rate,
dst_host_rerror_rate,
dst_host_srv_rerror_rate"""

columns=[]
for c in cols.split(','):
    if(c.strip()):
       columns.append(c.strip())

columns.append('target')
print(len(columns))

attacks_types = {
    'normal': 'normal',
'back': 'dos',
'buffer_overflow': 'u2r',
'ftp_write': 'r2l',
'guess_passwd': 'r2l',
'imap': 'r2l',
'ipsweep': 'probe',
'land': 'dos',
'loadmodule': 'u2r',
'multihop': 'r2l',
'neptune': 'dos',
'nmap': 'probe',
'perl': 'u2r',
'phf': 'r2l',
'pod': 'dos',
'portsweep': 'probe',
'rootkit': 'u2r',
'satan': 'probe',
'smurf': 'dos',
'spy': 'r2l',
'teardrop': 'dos',
'warezclient': 'r2l',
'warezmaster': 'r2l',
}

df = pd.read_csv('./archive/kddcup.data_10_percent.gz', names=columns)
df['Attack Type'] = df.target.apply(lambda r:attacks_types[r[:-1]])
df.head()

def standardization(df,col):
    std_scaler = MinMaxScaler()
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
    return df

def divideDataForPerceptron(df):
    # Target variable and train set
    Y = df[['Attack Type']]
    X = df.drop(['Attack Type',], axis=1)

    # Split test and train data 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_test, X_val, Y_test, Y_val = train_test_split(X, Y, test_size=0.2)
    print(X_train.shape, X_test.shape)
    print(Y_train.shape, Y_test.shape)
    return X_train, X_test, X_val, Y_train, Y_test, Y_val

def divideDataOnNormalAndAbnormal(df):
    normal_mask = df['Attack Type']==1
    attack_mask = df['Attack Type']!=1

    df.drop('Attack Type',axis=1,inplace=True)

    df_normal = df[normal_mask]
    df_attack = df[attack_mask]
    
    df_normal

    print(f"Normal count: {len(df_normal)}")
    print(f"Attack count: {len(df_attack)}")
    x_normal = df_normal.values
    x_attack = df_attack.values
    return x_normal, x_attack, df_normal, df_attack

def prepareData(df): 
    df.isnull().sum()
    num_cols = df._get_numeric_data().columns
    
    # find categorical columns
    cate_cols = list(set(df.columns)-set(num_cols))
    cate_cols.remove('target')
    cate_cols.remove('Attack Type')
    cate_cols
    
    # find columns with high correlation and delete them
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]]
    corr = df.corr()
    plt.figure(figsize=(15,12))
    sns.heatmap(corr)
    plt.show()
    df.drop('num_root',axis = 1,inplace = True)

    df.drop('srv_serror_rate',axis = 1,inplace = True)
    df.drop('srv_rerror_rate',axis = 1, inplace=True)
    df.drop('dst_host_srv_serror_rate',axis = 1, inplace=True)
    df.drop('dst_host_serror_rate',axis = 1, inplace=True)
    df.drop('dst_host_rerror_rate',axis = 1, inplace=True)
    df.drop('dst_host_srv_rerror_rate',axis = 1, inplace=True)
    df.drop('dst_host_same_srv_rate',axis = 1, inplace=True)
    df
    
    # standirtize and normilize numeric columns
    numeric_col = df.select_dtypes(include='number').columns
    len(numeric_col)
    df = standardization(df,numeric_col)
    
    # preproccess column with Cyber Attacks labels
    target_encoder = LabelEncoder()
    df['Attack Type'] = target_encoder.fit_transform(df['Attack Type'])
    df
    
    # preproccess other categorical columns 
    df = pd.get_dummies(df,columns=['flag','protocol_type','service'],prefix="",prefix_sep="") 
    df.head()
    df
    
    
    df = df.drop(['target',], axis=1)
    return df

def plotComparisonsOfLossesAndAccuracies(history_obj):
    accuracy = history_obj.history['accuracy']
    val_accuracy = history_obj.history['val_accuracy']
    loss = history_obj.history['loss']
    val_loss = history_obj.history['val_loss']
    
    plt.subplot(2, 1, 1)
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 5])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.show()

def perceptron(input_dim, output_dim, logits_for_layer_list, activation_func_type, output_activation_func_type):
        model = Sequential()
        model.add(Input(input_dim))
        for logits_for_layer in logits_for_layer_list:
            model.add(Dense(logits_for_layer, activation=activation_func_type))
        model.add(Dense(output_dim, activation=output_activation_func_type))
        return model

def autoencoder(input_dim, logits_for_layer_list, activation_func_type, output_activation_func_type):
        model = Sequential()
        model.add(Input(input_dim))
        for logits_for_layer in logits_for_layer_list:
            model.add(Dense(logits_for_layer, activation=activation_func_type))
        model.add(Dense(input_dim, activation=output_activation_func_type))
        return model
  
def prepareTrainingProcessForAE(batch_size_=10, lr=0.01, epochs_number=32, logits_for_layer_list=[150, 15, 150], metrics_list=['accuracy'], hidden_activation_func='tanh', output_activation_func='sigmoid', df=[0, 8]):
    df_for_autoencoder = prepareData(df.copy())
    df_for_autoencoder.head()
    x_normal, x_attack, df_normal, df_attack = divideDataOnNormalAndAbnormal(df_for_autoencoder)
    x_normal_train, x_normal_test = train_test_split(x_normal, test_size=0.25)  
    
    autoencoder_model = autoencoder(x_normal.shape[1], logits_for_layer_list, hidden_activation_func,output_activation_func)
    opt = Adam(learning_rate=lr)
    autoencoder_model.compile(loss='mean_squared_error', optimizer=opt, metrics=metrics_list)
    history = autoencoder_model.fit(x_normal_train,x_normal_train,epochs=epochs_number, batch_size=batch_size_)
    plotComparisonsOfLossesAndAccuracies(history)
    
def prepareTrainingProcessForPC(batch_size_=5000, lr=0.0001, epochs_number=10, logits_for_layer_list=[64, 128, 256], metrics_list=['accuracy'], hidden_activation_func='relu', output_activation_func='softmax', output_dim=10, df=[0,8]): 
    df_for_perceptron = prepareData(df.copy())
    X_train, X_test, X_val, Y_train, Y_test, Y_val = divideDataForPerceptron(df_for_perceptron.copy())
    perceptron_model = perceptron(X_train.shape[1], output_dim, logits_for_layer_list, hidden_activation_func, output_activation_func)
    opt = Adam(learning_rate=lr)
    perceptron_model.compile(loss ='sparse_categorical_crossentropy', optimizer = opt, metrics = metrics_list)
    history = perceptron_model.fit(X_train, Y_train, epochs=epochs_number, batch_size=batch_size_, validation_data=(X_val, Y_val))
    plotComparisonsOfLossesAndAccuracies(history)

def getListFromInput():
    stringA = input()
  
    res = json.loads(stringA)
    return res

def defineModelPreparation(model_type, df):
    print('Please mention batch size')
    batch_size = int(input())
    print('Please mention learning rate size')
    lr = float(input())
    print('Please mention epochs number')
    epochs_number = int(input())
    print('Please mention logits for layer list in that format [n1, n2, n3,...]')
    logits_for_layer_list = getListFromInput()
    print('Please mention metrics to be displayed while learning in that format [metric1, metric2, metric3,...]')
    metrics_list = getListFromInput()
    print('Please mention activation function for hidden layers')
    hidden_activation_func = input()
    print('Please mention activation function for output layer')
    output_activation_func = input()
    
    if model_type == "AE":
        prepareTrainingProcessForAE(batch_size, lr, epochs_number, logits_for_layer_list, metrics_list, hidden_activation_func, output_activation_func, df)
    else:
        print('Please mention output layer dim')
        output_dim = input()
        prepareTrainingProcessForPC(batch_size, lr, epochs_number, logits_for_layer_list, metrics_list, hidden_activation_func, output_activation_func, output_dim, df)

def start():
    print('Please write AE for autoencoder neutral model or PC for perceptron neutral model')
    model_type = input()
    defineModelPreparation(model_type, df.copy())
    should_continue = input()
    if(should_continue == 'Yes'):
        start()
    else:
        return

start()


