import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import re
#from adjustText import adjust_text


def create_dissimilarity_matrix(raw_matrix: pd.DataFrame, unique_words: np.array, subjects_no: int=-1) -> (pd.DataFrame): #(pd.DataFrame, pd.DataFrame):
    """
    targetデータフレームから単語のdissimilarity matrixとそのstdの行列を作成する関数。
    
    Parameters:
    - distance_matrix (pd.DataFrame): 
    - unique_words (np.array):
    - subject_no: -1→全体のmeanをとる それ以外：その番号を参照
    
    Returns:
    - pd.DataFrame: 作成されたdissimilarity matrix
    - # pd.DataFrame: dissimilarity matrix std
    """
    # 'word1'と'word2'の情報を抽出
    word1 = raw_matrix.iloc[0, :]  # 最初の行に含まれる単語
    word2 = raw_matrix.iloc[1, :]  # 2番目の行に含まれる単語

    # Dissimilarity Matrixを初期化
    dissim_mtx_mean = pd.DataFrame(np.nan, index=unique_words, columns=unique_words) # indexに[::-1]とするとAmyと同一に
    #dissim_mtx_std  = pd.DataFrame(np.nan, index=unique_words, columns=unique_words)
    # 各ペアに対応する距離を距離行列に埋めていく
    for i in range(len(word1)):
        w1 = word1.iloc[i]
        w2 = word2.iloc[i]

        if subjects_no == -1:
            dist_mean = raw_matrix.iloc[2:, i].mean()
            # dist_std = distance_matrix.iloc[2:, i].std()
        else:
            dist_mean = raw_matrix.iloc[2+subjects_no, i]
        
        # 対応するセルに距離を埋める
        dissim_mtx_mean.at[w1, w2] = dist_mean
        dissim_mtx_mean.at[w2, w1] = dist_mean
        # dissim_mtx_std.at[w1, w2] = dist_std
        # dissim_mtx_std.at[w2, w1] = dist_std

    # NaNを0に置換
    dissim_mtx_mean.fillna(value=0, inplace=True)
    #dissim_mtx_std.fillna(value=0, inplace=True)

    return dissim_mtx_mean# , dissim_mtx_std

def plot_dissimilarity_heatmap(dissimilarity_matrix: pd.DataFrame, annotation: bool=False, mode_dissim: bool=True, figpath: str="/home/jovyan/work/Amy/fig/temp/dissimilarity_matrix", save: bool=False):
    """
    dissimilarity matrixをヒートマップとして表示する関数。
    
    Parameters:
    - dissimilarity_matrix (pd.DataFrame): 表示するdissimilarity matrix
    - title (str): ヒートマップのタイトル
    - annotation: numbers in each cell (or not)
    """
    fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)  # グラフのサイズを調整
    if mode_dissim == True:
        sns.heatmap(dissimilarity_matrix, annot=annotation, fmt=".2f", cmap="GnBu", cbar=True, vmin=0, vmax=7, square=True)
    else: 
        sns.heatmap(dissimilarity_matrix, annot=annotation, fmt=".2f", cmap="GnBu_r", cbar=True, square=True)
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()
    ax.set_xticklabels(xticklabels,fontsize=15)
    ax.set_yticklabels(yticklabels,fontsize=15)
    figpath = Path(figpath) # 保存パスをPathオブジェクトに変換
    # ファイル名の不正文字 (:, , 空白) を安全な文字 (_) に変換
    safe_stem = re.sub(r"[:/\s]", "_", figpath.stem)
    save_path = figpath.with_name(safe_stem).with_suffix(".png")
    title = figpath.stem # last part of figpath
    ax.set_title(title, fontsize=24)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)

    if save:
        figpath.parent.mkdir(parents=True, exist_ok=True)     # 親ディレクトリが存在しない場合は作成
        save_path = figpath.with_suffix(".png")
        plt.savefig(figpath, dpi=300)
    plt.show()



def plot_MDS(mds_res: pd.DataFrame, dissimilarity_matrix: pd.DataFrame, figpath: str='/home/jovyan/work/Amy/fig/temp/MDS Plot'):
    # Plot the MDS results
    fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
    ax.scatter(mds_res[:, 0], mds_res[:, 1]) # plt.scatter?
    
    # ラベルを追加
    texts = [plt.text(mds_res[i, 0], mds_res[i, 1], word, fontsize=15) for i, word in enumerate(dissimilarity_matrix.index)]
    
    # ラベルの重なりを自動調整
    adjust_text(texts, only_move={'points': 'xy', 'text': 'xy'}, arrowprops=dict(arrowstyle="->", color='b', lw=1.2))
    
    # 最初の10点を赤い線でつなぐ
    ax.plot(mds_res[:10, 0], mds_res[:10, 1], 'r-', lw=1, label='Colours')
    
    # 次の15点を黒い線でつなぐ
    ax.plot(mds_res[10:25, 0], mds_res[10:25, 1], 'g-', lw=1, label='Emotions')
    
    # 凡例を追加
    ax.legend(fontsize=22)
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()
    ax.set_xticklabels(xticklabels,fontsize=15)
    ax.set_yticklabels(yticklabels,fontsize=15)
    ax.set_xlabel('MDS Dimension 1', fontsize=18)
    ax.set_ylabel('MDS Dimension 2', fontsize=18)
    figpath = Path(figpath) # 保存パスをPathオブジェクトに変換
    title = figpath.stem # last part of figpath
    ax.set_title(title, fontsize=24)
    figpath.parent.mkdir(parents=True, exist_ok=True)     # 親ディレクトリが存在しない場合は作成
    save_path = figpath.with_suffix(".png")
    plt.savefig(figpath, dpi=300)
    plt.show()

def get_tri_vector(dissim_mtx: pd.DataFrame) -> np.array:
    
    if np.ndim(dissim_mtx) == 2: #2次元だった場合、iのカウント外なのでここで除外する　
        upper_tri_ind = np.triu_indices_from(dissim_mtx, k=1) # get index of upper triangle and not diag
        len_vec = int((np.shape(dissim_mtx)*np.shape(dissim_mtx) - np.shape(dissim_mtx)) / 2) # 253
        len_vec = np.zeros((np.shape(dissim_mtx), len_vec))
        dissim_vecs = np.full_like(len_vec, np.nan, dtype=np.float64) # nan行列に代入しvaluesを機能させる
        # 上三角部分の要素を抽出
        for i in range(len(dissim_mtx)):
            dissim_vecs = dissim_mtx.values[upper_tri_ind]
            
    elif np.ndim(dissim_mtx) == 3:
        for i in range(len(dissim_mtx)):
            upper_tri_ind = np.triu_indices_from(dissim_mtx[0], k=1) # get index of upper triangle and not diag
            len_vec = int((np.shape(dissim_mtx)[1]*np.shape(dissim_mtx)[1] - np.shape(dissim_mtx)[1]) / 2) # 253
            len_vec = np.zeros((np.shape(dissim_mtx)[0], len_vec))
            dissim_vecs = np.full_like(len_vec, np.nan, dtype=np.float64) # nan行列に代入しvaluesを機能させる
            # 上三角部分の要素を抽出
            for i in range(len(dissim_mtx)):
                dissim_vecs[i] = dissim_mtx[i].values[upper_tri_ind]
            
    return dissim_vecs
    
def calc_corr_dissim(vectors: np.array, groups_name: list):
    
    for i, group1 in enumerate(groups_name):
        for j, group2 in enumerate(groups_name):
            if i<j:
                vec1 = vectors[i]
                vec2 = vectors[j]
                title = f"{group1} vs {group2} peason's correlation"
                
                # ブートストラップの設定
                n_bootstraps = 1000  # ブートストラップの反復回数
                bootstrap_corrs = []
                
                # ブートストラップサンプリング
                for _ in range(n_bootstraps):
                    # ランダムにインデックスを抽出
                    indices = np.random.choice(len(vec1), len(vec1), replace=True)
                    sample_vec1 = vec1[indices]
                    sample_vec2 = vec2[indices]
                    
                    # 相関係数を計算
                    r, _ = stats.pearsonr(sample_vec1, sample_vec2)
                    bootstrap_corrs.append(r)
                
                # 信頼区間の計算
                lower_bound = np.percentile(bootstrap_corrs, 2.5)
                upper_bound = np.percentile(bootstrap_corrs, 97.5)
                
                plt.plot(bootstrap_corrs)
                plt.xlabel("Number of bootstrap trials")
                plt.ylabel("Peason's r")
                plt.title(title)
                plt.show()
                
                print(f"95% Confidence Interval for correlation coefficient: ({lower_bound}, {upper_bound})")
            else:
                pass # avoid double count

def calc_inverse_matrix(df_distance: pd.DataFrame, unique_words: np.array) -> (pd.DataFrame, np.array, bool, list):
    fullrank = True
    duplicate_rows = []
    try:
        np_distance = df_distance.to_numpy()
        np_inv = np.linalg.inv(np_distance)
        df_inv = pd.DataFrame(np_inv, index=unique_words, columns=unique_words)

    except np.linalg.LinAlgError:
        fullrank = False
        print(f"Matrix is singular. Dropping duplicated rows.")
        duplicate_rows = df_distance.index[df_distance.duplicated(keep=False)].tolist()
        drop_rows = df_distance.index[df_distance.duplicated(keep='first')].tolist()
        df_distance_nondpl = df_distance.drop(index=drop_rows, columns=drop_rows)
        unique_words_nondpl = [item for item in unique_words if item not in drop_rows]
        np_distance = df_distance_nondpl.to_numpy()
        if np.linalg.matrix_rank(np_distance) < np_distance.shape[0]:
            print(f"Matrix still singular after duplicate removal. Using pinv.")
            np_inv = np.linalg.pinv(np_distance)
        else:
            np_inv = np.linalg.inv(np_distance)
        df_inv = pd.DataFrame(np_inv, index=unique_words_nondpl, columns=unique_words_nondpl)
        
    return (df_inv, np_inv, fullrank, duplicate_rows)