import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn import metrics
from pickle import dump
import joblib
import altair as alt
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score
# [theme]
# base="dark"
# primaryColor="purple"

st.write(""" 
# APLIKASI CEK TINGKAT STRESS MANUSIA
Oleh | FIQRY WAHYU DIKY W | 20041110125
""")

import streamlit as st

# "with" notation
# with st.sidebar:
#     st.title("Home")    

import_data, preprocessing, modeling, evaluation, implementation = st.tabs(["Import Data", "Pre Processing", "Modeling", "Evaluation", "Implementation"])



with import_data:
    st.write("# IMPORT DATA")
    uploaded_data = st.file_uploader("Upload Data Set yang Mau Digunakan", accept_multiple_files=True)
    cek_data, keterangan  = st.tabs(["Data","Keterangan"])
    for uploaded_data in uploaded_data:
        data = pd.read_csv(uploaded_data)
    with cek_data:
        st.write("Nama Dataset:", uploaded_data.name)
        st.write(data)
    with keterangan:
        st.write("### Berikut keterangan dari data anda")
        tipe_data   = data.dtypes
        data_max    = data.max()
        data_min    = data.min()
        data_kosong = data.isnull().sum()
        # fitur       = 
        st.write("="*25," Tipe data","="*25,"\n",tipe_data)
        st.write("="*25," Nilai maksimal data","="*25,"\n",round(data_max,2))
        st.write("="*25," Nilai minimal data","="*25,"\n",data_min)
        st.write("="*25," Nilai data kosong","="*30,"\n",data_kosong)


with preprocessing:
    st.write("# PRE PROCESSING")
    st.write("Data anda sudah ternormalisasi? jika belum maka klik Normalisasi")
    # encoding = st.checkbox("Encoding (Category to Numeric)")
    data_asli = st.checkbox('Tampil Data')
    normalisasi = st.checkbox('Normalisasi data')
    if data_asli:
        st.write(data)
    if normalisasi:
        st.write("Melakukan Normalisasi pada semua fitur") 
        # normalisasi/standaritation
        scaler  = MinMaxScaler()
        scaled  = scaler.fit_transform(data[['Humidity','Temperature','Step count']])
        kolom_normalisasi = ["Humaditiy","Temperature","Step count"]
        data_normalisasi = pd.DataFrame(scaled,columns=kolom_normalisasi)

        st.write (data_normalisasi)


with modeling:
    st.write("# MODELING")
    # k_nn, naive_bayes, ds_tree = st.tabs(["K-NN", "Naive Bayes", "Decission Tree"])
    X = data_normalisasi.iloc[:,:4]
    # st.write(x)
    Y = data.iloc[:,-1]
    # st.write(Y)
    X_train, X_test, y_train, y_test    = train_test_split(X,Y, test_size=0.4, random_state=1)

    # st.write(y_test.shape)
    # st.write(X_test.shape)
    # st.write(y_train.shape)
    # st.write(X_train.shape)

    st.write("Pilih metode yang digunakan")
    # st.write("Nilai Score dari semuaa K \n",scores)
    knn_cek = st.checkbox("KNN")
    Gauss   = st.checkbox("Gaussian Naive-Bayes")
    Ds      = st.checkbox("Decission Tree")


    #=================== modeling KNN =====================
    scores = {}
    scores_list = []
    k_range = range(1,50)
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        # scores[k] = metrics.accuracy_score(y_test,y_pred_knn)
        # scores_list.append(metrics.accuracy_score(y_test,y_pred_knn))
    knn_accuracy = round(100 * accuracy_score(y_test, y_pred_knn), 2)
    # st.write(accuracy_score(y_test, y_pred_knn))
    # scoress = st.pd.dataframe(scores)

    # #===================== Naive Bayes =======================
    # gaussian    = GaussianNB()
    # gaussian.fit(X_train, y_train)
    # Y_pred_GS   = gaussian.predict(X_test)
    # Gauss_accuracy  = round(100*accuracy_score(y_test, Y_pred_GS),2)

    # #==================== Decission Tree =====================
    # DecissionT = DecisionTreeClassifier(criterion="gini")
    # DecissionT.fit(X_train,y_train)
    # y_pred_DS   = DecissionT.predict(X_test)
    # ds_accuracy = round(100*accuracy_score(y_test, y_pred_DS),2)
    

    

    if knn_cek:
        st.write(knn_accuracy)
    # if Gauss:
    #     st.write(Gauss_accuracy)
    # if Ds:
    #     st.write(ds_accuracy)

with implementation:
    st.write("# IMPLEMENTATION")
    # nama_pasien = st.text_input("Masukkan nama anda")
    humidity_mean = st.number_input("Masukkan Rata-rata Kelembaban", min_value=10, max_value=30)
    temperature_mean = st.number_input("Masukkan rata-rata Suhu", min_value=79, max_value=99)
    step_count_mean = st.number_input("Masukkan rata-rata hitungan langkah", min_value=0, max_value=200)

    # if knn_accuracy > Gauss_accuracy and knn_accuracy > ds_accuracy:
    #     hasil_max = knn_accuracy
    # elif Gauss_accuracy > knn_accuracy and Gauss_accuracy > ds_accuracy:
    #     hasil_max = Gauss_accuracy
    # else:
    #     hasil_max = ds_accuracy
    
    # # st.write(hasil_max)
    
    # st.write("Cek apakah Strees anda termasuk Rendah, Sedang, atau Tinggi")
    # cek_rumus = st.button('Cek Strees')
    # inputan = [[humidity_mean, temperature_mean, step_count_mean]]
    # # scaler  = MinMaxScaler()
    # inputan_normal = scaler.transform(inputan) #normalisasi inputan
    # # st.write(inputan)
    # # st.write(inputan_normal)
    # # FIRST_IDX = 0
    # if cek_rumus:
    #     if hasil_max == ds_accuracy:
    #         hasil_tes       = DecisionTreeClassifier(criterion="gini")
    #         hasil_tes.fit(X_train, y_train)
    #         hasil_pred      = hasil_tes.predict(inputan_normal)
    #         # st.write(hasil_pred)
    #         # hasil_accuracy  = round(100*accuracy_score(y_test, hasil_pred),2)
    #         st.write("DS")
    #         if hasil_pred == 0:
    #             st.write("Low")
    #         elif hasil_pred == 1:
    #             st.write ("Normal")
    #         else:
    #             st.write("High")

    #     elif hasil_max == knn_accuracy: 
    #         k_range = range(1,50)
    #         for k in k_range:
    #             hasil_tes = KNeighborsClassifier(n_neighbors=k)
    #             hasil_tes.fit(X_train, y_train)
    #             hasil_pred = knn.predict(inputan_normal)
    #         st.write("knn")
    #         if hasil_pred == 0:
    #             st.write("Low")
    #         elif hasil_pred == 1:
    #             st.write ("Normal")
    #         else:
    #             st.write("High")
        
    #     else:
    #         hasil_tes  = GaussianNB()
    #         hasil_tes.fit(X_train, y_train)
    #         hasil_pred  = gaussian.predict(inputan_normal)
    #         st.write("NB")
    #         if hasil_pred == 0:
    #             st.write("Low")
    #         elif hasil_pred == 1:
    #             st.write ("Normal")
    #         else:
    #             st.write("High")

            # st.write(y_test)
            # st.write(hasil_accuracy)

            




