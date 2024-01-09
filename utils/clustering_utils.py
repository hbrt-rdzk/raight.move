import matplotlib.pyplot as plt
import json
import numpy as np
import yaml
import os
from tqdm import tqdm
import pandas as pd
from utils.keypoints_tools import *
from PIL import Image
from sklearn.cluster import KMeans
from itertools import combinations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap
from ast import literal_eval
import umap
import plotly.express as px
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score
from itertools import chain, combinations
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import pandas as pd

def check_popularity(df, cluster_size, ascending_flag, info=True):
    columns_list = ['popular_z_score_median', 'popular_regression', 'popular_engagement_mean']

    # Inicjalizacja wykresu
    if info:
        fig, axes = plt.subplots(nrows=len(columns_list), ncols=1, figsize=(10, 15))
    for i, column in enumerate(columns_list):
        # Grupowanie danych wg 'labels' i liczenie True/False
        grouped_data = df.groupby('labels')[column].value_counts().unstack().fillna(0.99)
        
        # Obliczanie stosunku True do False
        grouped_data['Ratio'] = grouped_data[True] / grouped_data[False]
        predicted_column_name = f"predicted_{column}"
        df_tmp = df.merge(grouped_data[['Ratio']], left_on='labels', right_index=True, how='left')
        df[predicted_column_name] = df_tmp['Ratio'] >= 1

        if info:
            big_clusers = grouped_data[(grouped_data[True] + grouped_data[False]) >= cluster_size]
            # Sortowanie wg stosunku i wybieranie top 10
            top_clusters = big_clusers.sort_values(by='Ratio', ascending=ascending_flag).head(10)
            # Rysowanie wykresów słupkowych dla top 10 klastrów
            clusters = top_clusters.index
            true_values = top_clusters[True]
            false_values = top_clusters[False]
            ratios = top_clusters['Ratio']  # Dodane wartości Ratio
            
            bar_width = 0.35
            bar_positions_true = range(1, len(clusters) * 2, 2)
            bar_positions_false = [pos + bar_width for pos in bar_positions_true]
            
            axes[i].bar(bar_positions_true, true_values, label='Popular', width=bar_width)
            axes[i].bar(bar_positions_false, false_values, label='Unpopular', width=bar_width)
            
            # Dodanie etykiet Ratio dla każdego klastra
            for pos, ratio in zip(bar_positions_true, ratios):
                axes[i].text(pos, max(true_values.max(), false_values.max()), f'Ratio: {ratio:.2f}', ha='center', va='bottom')
            
            axes[i].set_xlabel('Cluster')
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'Column: {column}')
            axes[i].set_xticks(bar_positions_true)
            axes[i].set_xticklabels(clusters)
            axes[i].legend()
    if info:
        plt.show()
        
def show_samples(df, size, images_path, show_title=False):
    # Pobierz listę nazw plików jpg z kolumny 'jpg' w DataFrame
    image_names = df['photo_filename'].tolist()

    # Upewnij się, że istnieje ścieżka do folderu z obrazami
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"The specified images path '{images_path}' does not exist.")

    # Utwórz subploty na wykresie o rozmiarze size x size
    fig, axs = plt.subplots(size, size, figsize=(10, 10))

    # Iteruj po współrzędnych subplotów
    for i in range(size):
        for j in range(size):
            # Znajdź indeks obrazu w liście
            idx = i * size + j

            # Sprawdź, czy istnieje tyle obrazów w liście, ile subplotów
            if idx < len(image_names):
                # Utwórz pełną ścieżkę do obrazu
                image_path = os.path.join(images_path, image_names[idx])

                # Wczytaj obraz za pomocą PIL
                img = Image.open(image_path)

                # Wyłącz oznaczenia osi
                axs[i, j].axis('off')

                # Wyświetl obraz na subplotcie
                axs[i, j].imshow(img)
                if show_title:
                    axs[i, j].set_title(image_names[idx], fontsize=8)
            else:
                # Jeśli nie ma więcej obrazów, ukryj puste subploty
                axs[i, j].axis('off')

    # Wyświetl wykres
    plt.show()
    
def vectorize_columns(df, column_names, standardize=False):
    vectors_series = df.apply(lambda row: np.concatenate([np.array(row[col]) for col in column_names]), axis=1)
    vectors_np = np.vstack(vectors_series)
    
    if standardize:
        scaler = StandardScaler()
        vectors_np = scaler.fit_transform(vectors_np)
    df['vector'] = [vectors_np[i] for i in range(len(vectors_np))]
    df['vector'] = df['vector'].apply(np.array)
    return vectors_np

def get_photo_from_keypoints(filename):
    id = filename.split('-')[:2] 
    id = joined_string = '-'.join(id)
    id += '.jpg'
    return id
    
def get_photo_map_from_keypoints(filename):
    id = filename.split('-')[1] 
    id += '.jpg'
    return id
    
def podnies_do_potegi(lista, potega):
    return [element ** potega for element in lista]

def count_features(df_reprezentation, connections_2d, useless_points_2d, base_connestion_2d, n_bins, strategy, threshold_prob, pow_value, imput_type):
    lenght_prob = threshold_prob * threshold_prob
    variant_angle_prob = threshold_prob * threshold_prob
    angle_prob = threshold_prob * threshold_prob * threshold_prob
    discretizerType = KBinsDiscretizer
    model_args = {'n_bins': n_bins, 'encode': 'ordinal', 'strategy': strategy, 'subsample':None}

    relative_lenght_prob = f'relative_lengths_{lenght_prob}'
    variant_angles_prob = f'variant_angles_{variant_angle_prob}'
    angles_prob = f'angles_{angle_prob}'

    add_lenghts(df_reprezentation, connections_2d, useless_points=useless_points_2d)
    add_angles(df_reprezentation, angles_2d, three_dim=False, useless_points=useless_points_2d)
    add_variant_angle(df_reprezentation, connections_2d, three_dim=False, useless_points=useless_points_2d)
    add_relative_lenghts(df_reprezentation, base_connestion_2d)

    exclude_low_prob(df_reprezentation, 'relative_lengths', 'lengths_prob', lenght_prob)
    exclude_low_prob(df_reprezentation, 'variant_angles', 'variant_angles_prob', variant_angle_prob)
    exclude_low_prob(df_reprezentation, 'angles', 'angles_prob', angle_prob)

    bin_column(df_reprezentation, 'relative_lengths', 'relative_lengths_bin', discretizerType, model_args)
    bin_column(df_reprezentation, 'variant_angles', 'variant_angles_bin', discretizerType, model_args)
    bin_column(df_reprezentation, 'angles', 'angles_bin', discretizerType, model_args)
    
    bin_column(df_reprezentation, 'lengths_prob', 'lengths_prob_bin', discretizerType, model_args)
    bin_column(df_reprezentation, 'variant_angles_prob', 'variant_angles_prob_bin', discretizerType, model_args)
    bin_column(df_reprezentation, 'angles_prob', 'angles_prob_bin', discretizerType, model_args)
    
    df_reprezentation['relative_lengths_bin_pow'] = df_reprezentation['relative_lengths_bin'].apply(lambda x: podnies_do_potegi(x, pow_value))
    df_reprezentation['variant_angles_bin_pow'] = df_reprezentation['variant_angles_bin'].apply(lambda x: podnies_do_potegi(x, pow_value))
    df_reprezentation['angles_bin_pow'] = df_reprezentation['angles_bin'].apply(lambda x: podnies_do_potegi(x, pow_value))

    std_column(df_reprezentation, relative_lenght_prob, 'relative_lengths_std', imput_type)
    std_column(df_reprezentation, variant_angles_prob, 'variant_angles_std', imput_type)
    std_column(df_reprezentation, angles_prob, 'angles_std', imput_type)

    bin_column(df_reprezentation, 'relative_lengths_std', 'relative_lengths_std_bin', discretizerType, model_args)
    bin_column(df_reprezentation, 'variant_angles_std', 'variant_angles_std_bin', discretizerType, model_args)
    bin_column(df_reprezentation, 'angles_std', 'angles_std_bin', discretizerType, model_args)

    df_reprezentation['relative_lengths_std_bin_pow'] = df_reprezentation['relative_lengths_std_bin'].apply(lambda x: podnies_do_potegi(x, pow_value))
    df_reprezentation['variant_angles_std_bin_pow'] = df_reprezentation['variant_angles_std_bin'].apply(lambda x: podnies_do_potegi(x, pow_value))
    df_reprezentation['angles_std_bin_pow'] = df_reprezentation['angles_std_bin'].apply(lambda x: podnies_do_potegi(x, pow_value))
    
    return relative_lenght_prob, variant_angles_prob, angles_prob

def show_clusers(transform_result, labels, paint_cluster=False):
    #tsne_df_filtered = tsne_df[tsne_df['labels'] == 6]
    #tsne_df_filtered = tsne_df
    tsne_df = pd.DataFrame(data={'TSNE_1': transform_result[:, 0], 'TSNE_2': transform_result[:, 1], 'labels': labels})

    # Dodanie kolumny 'color' jako warunku dla zmiany koloru
    tsne_df['color'] = 'default'  # Ustaw kolor domyślny
    tsne_df.loc[tsne_df['labels'] == paint_cluster, 'color'] = 'red'  # Zmień kolor tylko dla labels == 6

    # Tworzenie wykresu przy użyciu Plotly Express
    if type(paint_cluster) is int:
        fig_tsne = px.scatter(tsne_df, x='TSNE_1', y='TSNE_2', color='color', title='Uproszczona reprezentacja przy pomocy t-SNE')
    else: 
        fig_tsne = px.scatter(tsne_df, x='TSNE_1', y='TSNE_2', color='labels', title='Uproszczona reprezentacja przy pomocy t-SNE')
    fig_tsne.update_layout(
        height=1000,
        width=1000, # Wysokość wykresu
        xaxis=dict(
            title_font=dict(size=21),  # Zwiększenie rozmiaru czcionki w podpisie osi x
            tickfont=dict(size=17)  # Zwiększenie rozmiaru czcionki w etykietach osi x
        ),
        yaxis=dict(
            title_font=dict(size=21),  # Zwiększenie rozmiaru czcionki w podpisie osi y
            tickfont=dict(size=17)  # Zwiększenie rozmiaru czcionki w etykietach osi y
        ),
        title=dict(
            text='Wizualizacja reprezentacji przy pomocy TSNE oraz DBCAN',  # Tytuł
            font=dict(size=29)  # Zwiększenie rozmiaru czcionki w tytule
        )
    )
    fig_tsne.show()
    
def transform_vector(df, column_list, dbscan_eps, info=True):
    vector_features = vectorize_columns(df, column_list, False)

    transform_model = TSNE()
    #transform_model = umap.UMAP()
    transform_result = transform_model.fit_transform(vector_features)

    # Stworzenie DataFrame dla wyników PCA
    tsne_df = pd.DataFrame(data={'TSNE_1': transform_result[:, 0], 'TSNE_2': transform_result[:, 1]})
    
    model = DBSCAN(eps=dbscan_eps)
    labels = model.fit_predict(transform_result)
    print(len(set(labels)))
    df['labels'] = labels
    df['TSNE_1'] = transform_result[:, 0]
    df['TSNE_2'] = transform_result[:, 1]
    if info:
        paint_cluster = False
        show_clusers(transform_result, labels, paint_cluster=paint_cluster)

def get_cluster_samples(df, cluster_label, size, seed=0):
    # Ustawienie ziarna dla reprodukowalności losowania
    np.random.seed(seed)
    
    # Wybierz podzbiór DataFrame, gdzie 'labels' jest równy cluster_label
    cluster_df = df[df['labels'] == cluster_label]

    # Jeśli liczba dostępnych próbek jest mniejsza niż size * size, wybierz wszystkie dostępne próbki
    num_samples = min(size * size, len(cluster_df))

    # Wybierz losowo size * size elementów z podzbioru
    sampled_df = cluster_df.sample(n=num_samples)

    # Zwróć wynikowy DataFrame z wybranymi próbkami
    return sampled_df

def check_ratio(df, column):

    counts = df[column].value_counts()
    true_count = counts.get(True, 0)
    false_count = counts.get(False, 0)
    
    if false_count != 0:
        ratio = true_count / false_count
        print(f"Stosunek liczby True do liczby False dla {column}: {ratio}\n")
    else:
        print(f"Brak wartości False dla kolumny {column}.\n")
        
        
def count_acc(final_df, model_type=XGBClassifier,  low_info=False, info_threshold=0.0):
    columns = ['popular_z_score_median', 'popular_regression', 'popular_engagement_mean']
    
    used_columns = ['relative_lengths_std_bin', 'variant_angles_std_bin', 'angles_std_bin']
    df_dataset = pd.DataFrame()
    for column in used_columns:
        df_dataset = pd.concat([df_dataset, pd.DataFrame(final_df[column].to_list()).add_prefix(f'{column}_')], axis=1)
    
    for column in columns:
        
        # Tworzenie kolumny predicted_{column}
        
        X = df_dataset
        y = final_df[column]

        stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        # Lista przechowująca wyniki
        f1_scores = []
        accuracy_scores = []
        
        f1_scores_dummy = []
        accuracy_scores_dummy = []

        # Pętla kroswalidacyjna
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Inicjalizacja modelu
            rf_model = model_type()
            dump_model = DummyClassifier(strategy='most_frequent')

            # Trenowanie modelu
            rf_model.fit(X_train, y_train)
            dump_model.fit(X_train, y_train)

            # Przewidywanie na zbiorze testowym
            y_pred = rf_model.predict(X_test)
            y_pred_dump = dump_model.predict(X_test)

            # Obliczanie metryk dla foldu
            accuracy_fold = accuracy_score(y_test, y_pred)
            f1_fold = f1_score(y_test, y_pred)
            f1_dump = f1_score(y_test, y_pred_dump)
            accuracy_dump = accuracy_score(y_test, y_pred_dump)

            # Zapisanie wyników
            accuracy_scores.append(accuracy_fold)
            f1_scores.append(f1_fold)
            
            f1_scores_dummy.append(f1_dump)
            accuracy_scores_dummy.append(accuracy_dump)

        # Obliczenie średnich wartości
        accuracy_svm = np.mean(accuracy_scores)
        f1_svm = np.mean(f1_scores)
        
       
        
        
        f1_dump = np.mean(f1_scores_dummy)
        accuracy_dump = np.mean(accuracy_scores_dummy)
        
        # Macierz pomyłek dla SVM
        cm_svm = confusion_matrix(y_test, y_pred)
        
        # Macierz pomyłek dla DummyClassifier
        cm_dump = confusion_matrix(y_test, y_pred_dump)

        
        labels = [0, 1]  # Załóżmy, że mamy dwie klasy
        z_text_svm = [[str(value) for value in row] for row in cm_svm]
        z_text_dump = [[str(value) for value in row] for row in cm_dump]

        # Tworzenie figury z macierzami pomyłek
        confusion_matrix_fig_svm = ff.create_annotated_heatmap(z=cm_svm,
                                                               x=labels,
                                                               y=labels,
                                                               annotation_text=z_text_svm,
                                                               colorscale=[[0, 'rgb(255, 220, 220)'], [0.66, 'rgb(255, 150, 150)'], [1.0, 'rgb(255, 50, 50)']],
                                                               showscale=False)

        confusion_matrix_fig_dump = ff.create_annotated_heatmap(z=cm_dump,
                                                                x=labels,
                                                                y=labels,
                                                                annotation_text=z_text_dump,
                                                                colorscale=[[0, 'rgb(255, 159, 97)'], [0.66, 'rgb(165, 255, 206)'], [1.0, 'rgb(162, 115, 255)']],
                                                                showscale=False)
        confusion_matrix_fig_svm.update_layout(
            height=500,  # Wysokość wykresu
            width=500,   # Szerokość wykresu
            showlegend=False,
            title_text='Dystrybucje pewności predykcji stawów na badanym zbiorze przez model RTMPose'
        )
        confusion_matrix_fig_dump.update_layout(
            height=500,  # Wysokość wykresu
            width=500,   # Szerokość wykresu
            showlegend=False,
            title_text='Dystrybucje pewności predykcji stawów na badanym zbiorze przez model RTMPose'
        )
        # Wyświetlanie macierzy pomyłek
        #confusion_matrix_fig_svm.show()
        #confusion_matrix_fig_dump.show()

        # Wyświetlanie wyników
        if f1_svm >= f1_dump or accuracy_svm >= accuracy_dump:
            print("7"*50)
            print("7"*50)
            print("7"*50)
            
        check_ratio(final_df, column)
        print(f"Wyniki dla kolumny {column}:")
        print(f"F1 Score SVM: {f1_svm}")
        print(f"F1 Score dump: {f1_dump}")
        print(f"Accuracy SVM: {accuracy_svm}")
        print(f"Accuracy dump: {accuracy_dump}")
        
    return f1_svm, f1_dump, accuracy_svm, accuracy_dump


def all_nonempty_subsets(lst):
    return list(chain.from_iterable(combinations(lst, r) for r in range(1, len(lst)+1)))


def log_test(threshold_prob, _drop_value, n_bins, strategy, imput_type, dbscan_eps, pow_value, f1, f1_dump, acc, acc_dump):
    return {
        'threshold_prob': threshold_prob,
        '_drop_value': _drop_value,
        'n_bins': n_bins,
        'strategy': strategy,
        'imput_type': imput_type,
        'dbscan_eps': dbscan_eps,
        'pow_value': pow_value,
        'f1': f1,
        'f1_dump': f1_dump,
        'acc': acc,
        'acc_dump': acc_dump
    }