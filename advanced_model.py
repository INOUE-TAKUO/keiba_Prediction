#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高度な予測モデルの実装
"""

import os
import logging
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定数
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_FILE = os.path.join(DATA_DIR, 'horse_racing.db')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# ディレクトリの作成
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class AdvancedModel:
    """
    競馬予測の高度なモデルを実装するクラス
    """
    
    def __init__(self, db_file=DB_FILE):
        """
        初期化
        
        Args:
            db_file (str): データベースファイルのパス
        """
        self.db_file = db_file
        self.conn = None
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
    
    def connect_db(self):
        """
        データベースに接続する
        
        Returns:
            bool: 接続に成功したかどうか
        """
        try:
            self.conn = sqlite3.connect(self.db_file)
            logger.info(f"データベースに接続しました: {self.db_file}")
            return True
        except Exception as e:
            logger.error(f"データベース接続中にエラーが発生しました: {e}")
            return False
    
    def close_db(self):
        """
        データベース接続を閉じる
        """
        if self.conn:
            self.conn.close()
            logger.info("データベース接続を閉じました")
    
    def get_features_data(self, min_date=None, max_date=None):
        """
        特徴量データを取得する
        
        Args:
            min_date (str, optional): 最小日付
            max_date (str, optional): 最大日付
            
        Returns:
            DataFrame: 特徴量データ
        """
        try:
            # レース情報を取得
            race_query = """
            SELECT race_id, race_date, race_name, track_type, distance
            FROM races
            """
            
            if min_date or max_date:
                conditions = []
                params = []
                
                if min_date:
                    conditions.append("race_date >= ?")
                    params.append(min_date)
                
                if max_date:
                    conditions.append("race_date <= ?")
                    params.append(max_date)
                
                race_query += " WHERE " + " AND ".join(conditions)
            
            races_df = pd.read_sql_query(race_query, self.conn, params=params if min_date or max_date else None)
            
            if races_df.empty:
                logger.warning("レース情報が見つかりませんでした")
                return pd.DataFrame()
            
            # 特徴量データを取得
            features_data = []
            
            for _, race in races_df.iterrows():
                race_id = race['race_id']
                
                # 特徴量を取得
                features_query = """
                SELECT f.race_id, f.horse_id, f.feature_name, f.feature_value
                FROM features f
                WHERE f.race_id = ?
                """
                
                features_df = pd.read_sql_query(features_query, self.conn, params=(race_id,))
                
                if features_df.empty:
                    logger.warning(f"レース {race_id} の特徴量が見つかりませんでした")
                    continue
                
                # ピボットテーブルに変換
                pivot_df = features_df.pivot_table(
                    index=['race_id', 'horse_id'],
                    columns='feature_name',
                    values='feature_value',
                    aggfunc='first'
                ).reset_index()
                
                # レース情報を追加
                pivot_df['race_date'] = race['race_date']
                pivot_df['race_name'] = race['race_name']
                pivot_df['track_type_str'] = race['track_type']
                pivot_df['distance'] = race['distance']
                
                # レース結果を取得
                results_query = """
                SELECT race_id, horse_id, order_of_finish
                FROM race_results
                WHERE race_id = ?
                """
                
                results_df = pd.read_sql_query(results_query, self.conn, params=(race_id,))
                
                if not results_df.empty:
                    # 結果をマージ
                    pivot_df = pd.merge(
                        pivot_df,
                        results_df,
                        on=['race_id', 'horse_id'],
                        how='left'
                    )
                    
                    # 複勝（3着以内）フラグを追加
                    pivot_df['is_place'] = (pivot_df['order_of_finish'] <= 3).astype(int)
                    
                    # 勝利フラグを追加
                    pivot_df['is_win'] = (pivot_df['order_of_finish'] == 1).astype(int)
                
                features_data.append(pivot_df)
            
            if not features_data:
                logger.warning("特徴量データが見つかりませんでした")
                return pd.DataFrame()
            
            # 全データを結合
            all_features_df = pd.concat(features_data, ignore_index=True)
            
            # 欠損値を0で埋める
            all_features_df = all_features_df.fillna(0)
            
            return all_features_df
        
        except Exception as e:
            logger.error(f"特徴量データの取得中にエラーが発生しました: {e}")
            return pd.DataFrame()
    
    def prepare_data_for_classification(self, features_df, target='is_place', scale=True):
        """
        分類問題用にデータを準備する
        
        Args:
            features_df (DataFrame): 特徴量データ
            target (str): ターゲット変数名（'is_win' または 'is_place'）
            scale (bool): 特徴量を標準化するかどうか
            
        Returns:
            tuple: (X, y, feature_names)
                X (ndarray): 特徴量行列
                y (ndarray): ターゲット変数
                feature_names (list): 特徴量名のリスト
        """
        if features_df.empty:
            logger.warning("特徴量データが空です")
            return None, None, None
        
        # ターゲット変数が存在するか確認
        if target not in features_df.columns:
            logger.warning(f"ターゲット変数 {target} が見つかりませんでした")
            return None, None, None
        
        # 使用しない列を除外
        exclude_cols = ['race_id', 'horse_id', 'race_date', 'race_name', 'track_type_str',
                        'order_of_finish', 'is_win', 'is_place']
        
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # 特徴量とターゲットを分離
        X = features_df[feature_cols].values
        y = features_df[target].values
        
        # 特徴量の標準化
        if scale:
            X = self.scaler.fit_transform(X)
        
        return X, y, feature_cols
    
    def prepare_data_for_regression(self, features_df, target='order_of_finish', scale=True):
        """
        回帰問題用にデータを準備する
        
        Args:
            features_df (DataFrame): 特徴量データ
            target (str): ターゲット変数名（'order_of_finish'）
            scale (bool): 特徴量を標準化するかどうか
            
        Returns:
            tuple: (X, y, feature_names)
                X (ndarray): 特徴量行列
                y (ndarray): ターゲット変数
                feature_names (list): 特徴量名のリスト
        """
        if features_df.empty:
            logger.warning("特徴量データが空です")
            return None, None, None
        
        # ターゲット変数が存在するか確認
        if target not in features_df.columns:
            logger.warning(f"ターゲット変数 {target} が見つかりませんでした")
            return None, None, None
        
        # 使用しない列を除外
        exclude_cols = ['race_id', 'horse_id', 'race_date', 'race_name', 'track_type_str',
                        'order_of_finish', 'is_win', 'is_place']
        
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # 特徴量とターゲットを分離
        X = features_df[feature_cols].values
        y = features_df[target].values
        
        # 特徴量の標準化
        if scale:
            X = self.scaler.fit_transform(X)
        
        return X, y, feature_cols
    
    def train_lightgbm_classification(self, X_train, y_train, params=None):
        """
        LightGBM分類モデルを学習する
        
        Args:
            X_train (ndarray): 学習用特徴量
            y_train (ndarray): 学習用ターゲット
            params (dict, optional): LightGBMのパラメータ
            
        Returns:
            object: 学習済みモデル
        """
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
        
        # 学習データセットの作成
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # モデルの学習
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        self.model = model
        
        return model
    
    def train_lightgbm_regression(self, X_train, y_train, params=None):
        """
        LightGBM回帰モデルを学習する
        
        Args:
            X_train (ndarray): 学習用特徴量
            y_train (ndarray): 学習用ターゲット
            params (dict, optional): LightGBMのパラメータ
            
        Returns:
            object: 学習済みモデル
        """
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
        
        # 学習データセットの作成
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # モデルの学習
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        self.model = model
        
        return model
    
    def train_random_forest_classification(self, X_train, y_train, params=None):
        """
        ランダムフォレスト分類モデルを学習する
        
        Args:
            X_train (ndarray): 学習用特徴量
            y_train (ndarray): 学習用ターゲット
            params (dict, optional): ランダムフォレストのパラメータ
            
        Returns:
            object: 学習済みモデル
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        
        # モデルの初期化
        model = RandomForestClassifier(**params)
        
        # モデルの学習
        model.fit(X_train, y_train)
        
        self.model = model
        
        return model
    
    def train_random_forest_regression(self, X_train, y_train, params=None):
        """
        ランダムフォレスト回帰モデルを学習する
        
        Args:
            X_train (ndarray): 学習用特徴量
            y_train (ndarray): 学習用ターゲット
            params (dict, optional): ランダムフォレストのパラメータ
            
        Returns:
            object: 学習済みモデル
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        
        # モデルの初期化
        model = RandomForestRegressor(**params)
        
        # モデルの学習
        model.fit(X_train, y_train)
        
        self.model = model
        
        return model
    
    def evaluate_classification_model(self, model, X_test, y_test, model_type='lightgbm'):
        """
        分類モデルを評価する
        
        Args:
            model (object): 学習済みモデル
            X_test (ndarray): テスト用特徴量
            y_test (ndarray): テスト用ターゲット
            model_type (str): モデルの種類（'lightgbm' または 'random_forest'）
            
        Returns:
            dict: 評価指標
        """
        # 予測
        if model_type == 'lightgbm':
            y_pred_proba = model.predict(X_test)
        else:  # random_forest
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 評価指標の計算
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # 混同行列
        cm = confusion_matrix(y_test, y_pred)
        
        # 結果をまとめる
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
        logger.info(f"分類モデルの評価結果: {metrics}")
        
        return metrics
    
    def evaluate_regression_model(self, model, X_test, y_test, model_type='lightgbm'):
        """
        回帰モデルを評価する
        
        Args:
            model (object): 学習済みモデル
            X_test (ndarray): テスト用特徴量
            y_test (ndarray): テスト用ターゲット
            model_type (str): モデルの種類（'lightgbm' または 'random_forest'）
            
        Returns:
            dict: 評価指標
        """
        # 予測
        if model_type == 'lightgbm':
            y_pred = model.predict(X_test)
        else:  # random_forest
            y_pred = model.predict(X_test)
        
        # 評価指標の計算
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 結果をまとめる
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        }
        
        logger.info(f"回帰モデルの評価結果: {metrics}")
        
        return metrics
    
    def plot_feature_importance(self, model, feature_names, top_n=20, save_path=None, model_type='lightgbm'):
        """
        特徴量の重要度をプロットする
        
        Args:
            model (object): 学習済みモデル
            feature_names (list): 特徴量名のリスト
            top_n (int, optional): 表示する上位の特徴量数
            save_path (str, optional): 保存先のパス
            model_type (str): モデルの種類（'lightgbm' または 'random_forest'）
            
        Returns:
            None
        """
        # 特徴量の重要度を取得
        if model_type == 'lightgbm':
            importance = model.feature_importance(importance_type='gain')
        else:  # random_forest
            importance = model.feature_importances_
        
        # 特徴量名と重要度をDataFrameに変換
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # 重要度でソート
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # 上位N個の特徴量を選択
        top_features = feature_importance.head(top_n)
        
        # プロット
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"特徴量重要度のプロットを保存しました: {save_path}")
        
        plt.close()
    
    def save_model(self, model, model_name, model_type='lightgbm'):
        """
        モデルを保存する
        
        Args:
            model (object): 保存するモデル
            model_name (str): モデル名
            model_type (str): モデルの種類（'lightgbm' または 'random_forest'）
            
        Returns:
            str: 保存先のパス
        """
        # 現在の日時を取得
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存先のパス
        if model_type == 'lightgbm':
            save_path = os.path.join(MODELS_DIR, f'{model_name}_{now}.txt')
            # モデルを保存
            model.save_model(save_path)
        else:  # random_forest
            save_path = os.path.join(MODELS_DIR, f'{model_name}_{now}.joblib')
            # モデルを保存
            joblib.dump(model, save_path)
        
        logger.info(f"モデルを保存しました: {save_path}")
        
        return save_path
    
    def load_model(self, model_path, model_type='lightgbm'):
        """
        モデルを読み込む
        
        Args:
            model_path (str): モデルファイルのパス
            model_type (str): モデルの種類（'lightgbm' または 'random_forest'）
            
        Returns:
            object: 読み込んだモデル
        """
        try:
            if model_type == 'lightgbm':
                model = lgb.Booster(model_file=model_path)
            else:  # random_forest
                model = joblib.load(model_path)
            
            self.model = model
            
            logger.info(f"モデルを読み込みました: {model_path}")
            
            return model
        except Exception as e:
            logger.error(f"モデルの読み込み中にエラーが発生しました: {e}")
            return None
    
    def tune_lightgbm_hyperparameters(self, X, y, is_classification=True, cv=5):
        """
        LightGBMのハイパーパラメータをチューニングする
        
        Args:
            X (ndarray): 特徴量
            y (ndarray): ターゲット
            is_classification (bool): 分類問題かどうか
            cv (int): 交差検証の分割数
            
        Returns:
            dict: 最適なパラメータ
        """
        # パラメータグリッド
        param_grid = {
            'num_leaves': [31, 50, 100],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 500, 1000],
            'max_depth': [-1, 5, 10],
            'min_child_samples': [20, 50, 100],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        # モデルの初期化
        if is_classification:
            model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', verbose=-1)
        else:
            model = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1)
        
        # グリッドサーチの実行
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy' if is_classification else 'neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # 最適なパラメータ
        best_params = grid_search.best_params_
        
        logger.info(f"最適なパラメータ: {best_params}")
        
        return best_params
    
    def tune_random_forest_hyperparameters(self, X, y, is_classification=True, cv=5):
        """
        ランダムフォレストのハイパーパラメータをチューニングする
        
        Args:
            X (ndarray): 特徴量
            y (ndarray): ターゲット
            is_classification (bool): 分類問題かどうか
            cv (int): 交差検証の分割数
            
        Returns:
            dict: 最適なパラメータ
        """
        # パラメータグリッド
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # モデルの初期化
        if is_classification:
            model = RandomForestClassifier(random_state=42)
        else:
            model = RandomForestRegressor(random_state=42)
        
        # グリッドサーチの実行
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy' if is_classification else 'neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # 最適なパラメータ
        best_params = grid_search.best_params_
        
        logger.info(f"最適なパラメータ: {best_params}")
        
        return best_params
    
    def predict_race(self, race_id, model=None, model_type='lightgbm', prediction_type='classification'):
        """
        レースの結果を予測する
        
        Args:
            race_id (str): レースID
            model (object, optional): 使用するモデル（Noneの場合は self.model を使用）
            model_type (str): モデルの種類（'lightgbm' または 'random_forest'）
            prediction_type (str): 予測の種類（'classification' または 'regression'）
            
        Returns:
            DataFrame: 予測結果
        """
        if model is None:
            model = self.model
        
        if model is None:
            logger.error("モデルが設定されていません")
            return None
        
        try:
            # レース情報を取得
            race_query = """
            SELECT race_id, race_date, race_name, track_type, distance
            FROM races
            WHERE race_id = ?
            """
            
            race_df = pd.read_sql_query(race_query, self.conn, params=(race_id,))
            
            if race_df.empty:
                logger.warning(f"レース情報が見つかりませんでした: {race_id}")
                return None
            
            race_info = race_df.iloc[0]
            
            # 特徴量を取得
            features_query = """
            SELECT f.race_id, f.horse_id, f.feature_name, f.feature_value
            FROM features f
            WHERE f.race_id = ?
            """
            
            features_df = pd.read_sql_query(features_query, self.conn, params=(race_id,))
            
            if features_df.empty:
                logger.warning(f"レース {race_id} の特徴量が見つかりませんでした")
                return None
            
            # ピボットテーブルに変換
            pivot_df = features_df.pivot_table(
                index=['race_id', 'horse_id'],
                columns='feature_name',
                values='feature_value',
                aggfunc='first'
            ).reset_index()
            
            # 馬情報を取得
            horses_query = """
            SELECT h.horse_id, h.horse_name
            FROM horses h
            WHERE h.horse_id IN (
                SELECT DISTINCT horse_id FROM features WHERE race_id = ?
            )
            """
            
            horses_df = pd.read_sql_query(horses_query, self.conn, params=(race_id,))
            
            # 馬情報をマージ
            pivot_df = pd.merge(
                pivot_df,
                horses_df,
                on='horse_id',
                how='left'
            )
            
            # 使用しない列を除外
            exclude_cols = ['race_id', 'horse_id', 'horse_name']
            feature_cols = [col for col in pivot_df.columns if col not in exclude_cols]
            
            # 特徴量行列を作成
            X = pivot_df[feature_cols].fillna(0).values
            
            # 特徴量の標準化
            X = self.scaler.transform(X)
            
            # 予測
            if prediction_type == 'classification':
                if model_type == 'lightgbm':
                    y_pred_proba = model.predict(X)
                else:  # random_forest
                    y_pred_proba = model.predict_proba(X)[:, 1]
                
                # 結果をDataFrameに変換
                results_df = pd.DataFrame({
                    'race_id': race_id,
                    'horse_id': pivot_df['horse_id'],
                    'horse_name': pivot_df['horse_name'],
                    'probability': y_pred_proba
                })
                
                # 確率でソート
                results_df = results_df.sort_values('probability', ascending=False)
                
                # 順位を追加
                results_df['predicted_rank'] = range(1, len(results_df) + 1)
                
            else:  # regression
                if model_type == 'lightgbm':
                    y_pred = model.predict(X)
                else:  # random_forest
                    y_pred = model.predict(X)
                
                # 結果をDataFrameに変換
                results_df = pd.DataFrame({
                    'race_id': race_id,
                    'horse_id': pivot_df['horse_id'],
                    'horse_name': pivot_df['horse_name'],
                    'predicted_order': y_pred
                })
                
                # 予測着順でソート
                results_df = results_df.sort_values('predicted_order')
                
                # 順位を追加
                results_df['predicted_rank'] = range(1, len(results_df) + 1)
            
            # 実際の結果を取得（あれば）
            results_query = """
            SELECT rr.race_id, rr.horse_id, rr.order_of_finish
            FROM race_results rr
            WHERE rr.race_id = ?
            """
            
            actual_results_df = pd.read_sql_query(results_query, self.conn, params=(race_id,))
            
            if not actual_results_df.empty:
                # 実際の結果をマージ
                results_df = pd.merge(
                    results_df,
                    actual_results_df,
                    on=['race_id', 'horse_id'],
                    how='left'
                )
            
            return results_df
            
        except Exception as e:
            logger.error(f"レース予測中にエラーが発生しました: {e}")
            return None
    
    def evaluate_race_prediction(self, prediction_df):
        """
        レース予測の評価を行う
        
        Args:
            prediction_df (DataFrame): 予測結果
            
        Returns:
            dict: 評価指標
        """
        if prediction_df is None or prediction_df.empty:
            logger.warning("予測結果が空です")
            return None
        
        # 実際の結果があるか確認
        if 'order_of_finish' not in prediction_df.columns:
            logger.warning("実際の結果が含まれていません")
            return None
        
        try:
            # 1着的中
            win_hit = prediction_df.iloc[0]['order_of_finish'] == 1
            
            # 複勝的中（3着以内に予測1位の馬が入っているか）
            place_hit = prediction_df.iloc[0]['order_of_finish'] <= 3
            
            # 3連複的中（予測上位3頭が実際のトップ3に含まれているか）
            top3_horses = set(prediction_df.head(3)['horse_id'])
            actual_top3 = set(prediction_df[prediction_df['order_of_finish'] <= 3]['horse_id'])
            trio_hit = len(top3_horses.intersection(actual_top3)) == 3
            
            # 馬連的中（予測上位2頭が実際のトップ2に含まれているか）
            top2_horses = set(prediction_df.head(2)['horse_id'])
            actual_top2 = set(prediction_df[prediction_df['order_of_finish'] <= 2]['horse_id'])
            quinella_hit = len(top2_horses.intersection(actual_top2)) == 2
            
            # 馬単的中（予測上位2頭が実際のトップ2と同じ順序で含まれているか）
            exacta_hit = (prediction_df.iloc[0]['order_of_finish'] == 1 and 
                          prediction_df.iloc[1]['order_of_finish'] == 2)
            
            # 評価指標
            metrics = {
                'win_hit': win_hit,
                'place_hit': place_hit,
                'trio_hit': trio_hit,
                'quinella_hit': quinella_hit,
                'exacta_hit': exacta_hit
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"レース予測の評価中にエラーが発生しました: {e}")
            return None
    
    def calculate_betting_returns(self, prediction_df, bet_type='win', bet_amount=100):
        """
        馬券の回収率を計算する
        
        Args:
            prediction_df (DataFrame): 予測結果
            bet_type (str): 馬券の種類（'win', 'place', 'quinella', 'exacta', 'trio'）
            bet_amount (int): 賭け金額
            
        Returns:
            float: 回収率
        """
        if prediction_df is None or prediction_df.empty:
            logger.warning("予測結果が空です")
            return 0.0
        
        # 実際の結果があるか確認
        if 'order_of_finish' not in prediction_df.columns:
            logger.warning("実際の結果が含まれていません")
            return 0.0
        
        try:
            # オッズ情報を取得
            race_id = prediction_df['race_id'].iloc[0]
            
            # レース結果を取得
            results_query = """
            SELECT rr.horse_id, rr.order_of_finish, rr.odds
            FROM race_results rr
            WHERE rr.race_id = ?
            """
            
            results_df = pd.read_sql_query(results_query, self.conn, params=(race_id,))
            
            if results_df.empty:
                logger.warning(f"レース {race_id} の結果が見つかりませんでした")
                return 0.0
            
            # 予測結果とマージ
            prediction_df = pd.merge(
                prediction_df,
                results_df[['horse_id', 'odds']],
                on='horse_id',
                how='left'
            )
            
            # 馬券の種類に応じた回収率の計算
            if bet_type == 'win':
                # 単勝
                if prediction_df.iloc[0]['order_of_finish'] == 1:
                    return prediction_df.iloc[0]['odds'] * bet_amount
                else:
                    return 0.0
                
            elif bet_type == 'place':
                # 複勝
                if prediction_df.iloc[0]['order_of_finish'] <= 3:
                    # 複勝オッズは単勝の約1/3として簡易計算
                    return prediction_df.iloc[0]['odds'] * bet_amount / 3
                else:
                    return 0.0
                
            elif bet_type == 'quinella':
                # 馬連（上位2頭の組み合わせ）
                top2_horses = set(prediction_df.head(2)['horse_id'])
                actual_top2 = set(results_df[results_df['order_of_finish'] <= 2]['horse_id'])
                
                if len(top2_horses.intersection(actual_top2)) == 2:
                    # 馬連オッズは単勝オッズの積として簡易計算
                    odds1 = prediction_df.iloc[0]['odds']
                    odds2 = prediction_df.iloc[1]['odds']
                    return odds1 * odds2 * bet_amount / 5
                else:
                    return 0.0
                
            elif bet_type == 'exacta':
                # 馬単（上位2頭の順序付き組み合わせ）
                if (prediction_df.iloc[0]['order_of_finish'] == 1 and 
                    prediction_df.iloc[1]['order_of_finish'] == 2):
                    # 馬単オッズは単勝オッズの積として簡易計算
                    odds1 = prediction_df.iloc[0]['odds']
                    odds2 = prediction_df.iloc[1]['odds']
                    return odds1 * odds2 * bet_amount / 3
                else:
                    return 0.0
                
            elif bet_type == 'trio':
                # 3連複（上位3頭の組み合わせ）
                top3_horses = set(prediction_df.head(3)['horse_id'])
                actual_top3 = set(results_df[results_df['order_of_finish'] <= 3]['horse_id'])
                
                if len(top3_horses.intersection(actual_top3)) == 3:
                    # 3連複オッズは単勝オッズの積として簡易計算
                    odds1 = prediction_df.iloc[0]['odds']
                    odds2 = prediction_df.iloc[1]['odds']
                    odds3 = prediction_df.iloc[2]['odds']
                    return odds1 * odds2 * odds3 * bet_amount / 20
                else:
                    return 0.0
            
            else:
                logger.warning(f"未対応の馬券種類です: {bet_type}")
                return 0.0
            
        except Exception as e:
            logger.error(f"馬券回収率の計算中にエラーが発生しました: {e}")
            return 0.0
    
    def run_model_comparison_experiment(self, target='is_place', test_size=0.2, random_state=42):
        """
        複数のモデルを比較する実験を実行する
        
        Args:
            target (str): ターゲット変数名
            test_size (float): テストデータの割合
            random_state (int): 乱数シード
            
        Returns:
            dict: 各モデルの評価指標
        """
        if not self.connect_db():
            return None
        
        try:
            # データの取得
            features_df = self.get_features_data()
            
            if features_df.empty:
                logger.warning("特徴量データが空です")
                return None
            
            # データの準備
            is_classification = target in ['is_win', 'is_place']
            
            if is_classification:
                X, y, feature_names = self.prepare_data_for_classification(features_df, target=target)
            else:
                X, y, feature_names = self.prepare_data_for_regression(features_df, target=target)
            
            if X is None or y is None:
                return None
            
            # 特徴量名を保存
            self.feature_names = feature_names
            
            # データの分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # モデルの定義
            models = {}
            
            if is_classification:
                # LightGBM分類モデル
                lgb_model = self.train_lightgbm_classification(X_train, y_train)
                models['lightgbm'] = {
                    'model': lgb_model,
                    'metrics': self.evaluate_classification_model(lgb_model, X_test, y_test, model_type='lightgbm')
                }
                
                # ランダムフォレスト分類モデル
                rf_model = self.train_random_forest_classification(X_train, y_train)
                models['random_forest'] = {
                    'model': rf_model,
                    'metrics': self.evaluate_classification_model(rf_model, X_test, y_test, model_type='random_forest')
                }
            else:
                # LightGBM回帰モデル
                lgb_model = self.train_lightgbm_regression(X_train, y_train)
                models['lightgbm'] = {
                    'model': lgb_model,
                    'metrics': self.evaluate_regression_model(lgb_model, X_test, y_test, model_type='lightgbm')
                }
                
                # ランダムフォレスト回帰モデル
                rf_model = self.train_random_forest_regression(X_train, y_train)
                models['random_forest'] = {
                    'model': rf_model,
                    'metrics': self.evaluate_regression_model(rf_model, X_test, y_test, model_type='random_forest')
                }
            
            # 特徴量重要度のプロット
            for model_name, model_info in models.items():
                plot_path = os.path.join(RESULTS_DIR, f'feature_importance_{model_name}_{target}.png')
                self.plot_feature_importance(
                    model_info['model'], 
                    feature_names, 
                    save_path=plot_path,
                    model_type=model_name
                )
            
            # モデルの保存
            for model_name, model_info in models.items():
                model_type = 'classification' if is_classification else 'regression'
                model_path = self.save_model(
                    model_info['model'], 
                    f'{model_type}_{model_name}_{target}',
                    model_type=model_name
                )
            
            # 結果の比較
            comparison = {}
            
            for model_name, model_info in models.items():
                comparison[model_name] = model_info['metrics']
            
            logger.info(f"モデル比較結果: {comparison}")
            
            return comparison
            
        except Exception as e:
            logger.error(f"モデル比較実験中にエラーが発生しました: {e}")
            return None
            
        finally:
            self.close_db()
    
    def run_hyperparameter_tuning_experiment(self, target='is_place', model_type='lightgbm', cv=5):
        """
        ハイパーパラメータチューニングの実験を実行する
        
        Args:
            target (str): ターゲット変数名
            model_type (str): モデルの種類（'lightgbm' または 'random_forest'）
            cv (int): 交差検証の分割数
            
        Returns:
            dict: 最適なパラメータと評価指標
        """
        if not self.connect_db():
            return None
        
        try:
            # データの取得
            features_df = self.get_features_data()
            
            if features_df.empty:
                logger.warning("特徴量データが空です")
                return None
            
            # データの準備
            is_classification = target in ['is_win', 'is_place']
            
            if is_classification:
                X, y, feature_names = self.prepare_data_for_classification(features_df, target=target)
            else:
                X, y, feature_names = self.prepare_data_for_regression(features_df, target=target)
            
            if X is None or y is None:
                return None
            
            # 特徴量名を保存
            self.feature_names = feature_names
            
            # ハイパーパラメータチューニング
            if model_type == 'lightgbm':
                best_params = self.tune_lightgbm_hyperparameters(X, y, is_classification=is_classification, cv=cv)
            else:  # random_forest
                best_params = self.tune_random_forest_hyperparameters(X, y, is_classification=is_classification, cv=cv)
            
            # 最適なパラメータでモデルを学習
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if model_type == 'lightgbm':
                if is_classification:
                    model = self.train_lightgbm_classification(X_train, y_train, params=best_params)
                    metrics = self.evaluate_classification_model(model, X_test, y_test, model_type='lightgbm')
                else:
                    model = self.train_lightgbm_regression(X_train, y_train, params=best_params)
                    metrics = self.evaluate_regression_model(model, X_test, y_test, model_type='lightgbm')
            else:  # random_forest
                if is_classification:
                    model = self.train_random_forest_classification(X_train, y_train, params=best_params)
                    metrics = self.evaluate_classification_model(model, X_test, y_test, model_type='random_forest')
                else:
                    model = self.train_random_forest_regression(X_train, y_train, params=best_params)
                    metrics = self.evaluate_regression_model(model, X_test, y_test, model_type='random_forest')
            
            # 特徴量重要度のプロット
            plot_path = os.path.join(RESULTS_DIR, f'feature_importance_tuned_{model_type}_{target}.png')
            self.plot_feature_importance(model, feature_names, save_path=plot_path, model_type=model_type)
            
            # モデルの保存
            model_name = 'classification' if is_classification else 'regression'
            model_path = self.save_model(model, f'{model_name}_tuned_{model_type}_{target}', model_type=model_type)
            
            # 結果
            result = {
                'best_params': best_params,
                'metrics': metrics
            }
            
            logger.info(f"ハイパーパラメータチューニング結果: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"ハイパーパラメータチューニング実験中にエラーが発生しました: {e}")
            return None
            
        finally:
            self.close_db()
    
    def run_betting_simulation(self, model, model_type='lightgbm', prediction_type='classification', 
                              bet_type='win', bet_amount=100, min_date=None, max_date=None):
        """
        馬券購入シミュレーションを実行する
        
        Args:
            model (object): 使用するモデル
            model_type (str): モデルの種類（'lightgbm' または 'random_forest'）
            prediction_type (str): 予測の種類（'classification' または 'regression'）
            bet_type (str): 馬券の種類（'win', 'place', 'quinella', 'exacta', 'trio'）
            bet_amount (int): 賭け金額
            min_date (str, optional): 最小日付
            max_date (str, optional): 最大日付
            
        Returns:
            dict: シミュレーション結果
        """
        if not self.connect_db():
            return None
        
        try:
            # レース情報を取得
            race_query = """
            SELECT race_id, race_date, race_name
            FROM races
            """
            
            if min_date or max_date:
                conditions = []
                params = []
                
                if min_date:
                    conditions.append("race_date >= ?")
                    params.append(min_date)
                
                if max_date:
                    conditions.append("race_date <= ?")
                    params.append(max_date)
                
                race_query += " WHERE " + " AND ".join(conditions)
            
            race_query += " ORDER BY race_date"
            
            races_df = pd.read_sql_query(race_query, self.conn, params=params if min_date or max_date else None)
            
            if races_df.empty:
                logger.warning("レース情報が見つかりませんでした")
                return None
            
            # シミュレーション結果
            results = []
            
            # 各レースで予測と馬券購入シミュレーション
            for _, race in races_df.iterrows():
                race_id = race['race_id']
                
                # レース予測
                prediction_df = self.predict_race(
                    race_id, 
                    model=model, 
                    model_type=model_type, 
                    prediction_type=prediction_type
                )
                
                if prediction_df is None or prediction_df.empty:
                    continue
                
                # 馬券回収率の計算
                returns = self.calculate_betting_returns(prediction_df, bet_type=bet_type, bet_amount=bet_amount)
                
                # 結果を記録
                result = {
                    'race_id': race_id,
                    'race_date': race['race_date'],
                    'race_name': race['race_name'],
                    'bet_amount': bet_amount,
                    'returns': returns,
                    'profit': returns - bet_amount,
                    'roi': (returns - bet_amount) / bet_amount if bet_amount > 0 else 0.0
                }
                
                results.append(result)
            
            # 結果をDataFrameに変換
            results_df = pd.DataFrame(results)
            
            # 集計
            total_races = len(results_df)
            total_bet = total_races * bet_amount
            total_returns = results_df['returns'].sum()
            total_profit = total_returns - total_bet
            roi = total_profit / total_bet if total_bet > 0 else 0.0
            hit_rate = (results_df['returns'] > 0).mean()
            
            # 結果をまとめる
            summary = {
                'total_races': total_races,
                'total_bet': total_bet,
                'total_returns': total_returns,
                'total_profit': total_profit,
                'roi': roi,
                'hit_rate': hit_rate
            }
            
            logger.info(f"馬券購入シミュレーション結果: {summary}")
            
            # 結果を保存
            results_file = os.path.join(RESULTS_DIR, f'betting_simulation_{bet_type}.csv')
            results_df.to_csv(results_file, index=False)
            
            return {
                'summary': summary,
                'results': results_df
            }
            
        except Exception as e:
            logger.error(f"馬券購入シミュレーション中にエラーが発生しました: {e}")
            return None
            
        finally:
            self.close_db()


def main():
    """メイン関数"""
    advanced = AdvancedModel()
    
    # モデル比較実験
    logger.info("モデル比較実験を開始します")
    comparison = advanced.run_model_comparison_experiment(target='is_place')
    
    if comparison:
        logger.info(f"モデル比較結果: {comparison}")
    
    # ハイパーパラメータチューニング実験
    logger.info("ハイパーパラメータチューニング実験を開始します")
    tuning_result = advanced.run_hyperparameter_tuning_experiment(target='is_place', model_type='lightgbm')
    
    if tuning_result:
        logger.info(f"チューニング結果: {tuning_result}")
    
    # 最適化されたモデルを使用した馬券購入シミュレーション
    if tuning_result:
        logger.info("馬券購入シミュレーションを開始します")
        simulation_result = advanced.run_betting_simulation(
            advanced.model,
            model_type='lightgbm',
            prediction_type='classification',
            bet_type='win'
        )
        
        if simulation_result:
            logger.info(f"シミュレーション結果: {simulation_result['summary']}")


if __name__ == '__main__':
    main()
