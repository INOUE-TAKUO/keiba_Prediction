#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ベースラインモデルの実装
"""

import os
import logging
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from datetime import datetime

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baseline_model.log'),
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

class BaselineModel:
    """
    競馬予測のベースラインモデルを実装するクラス
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
    
    def prepare_data_for_classification(self, features_df, target='is_place'):
        """
        分類問題用にデータを準備する
        
        Args:
            features_df (DataFrame): 特徴量データ
            target (str): ターゲット変数名（'is_win' または 'is_place'）
            
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
        
        return X, y, feature_cols
    
    def prepare_data_for_regression(self, features_df, target='order_of_finish'):
        """
        回帰問題用にデータを準備する
        
        Args:
            features_df (DataFrame): 特徴量データ
            target (str): ターゲット変数名（'order_of_finish'）
            
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
        
        return X, y, feature_cols
    
    def train_classification_model(self, X_train, y_train, params=None):
        """
        分類モデルを学習する
        
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
            num_boost_round=100,
            valid_sets=[train_data],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        self.model = model
        
        return model
    
    def train_regression_model(self, X_train, y_train, params=None):
        """
        回帰モデルを学習する
        
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
            num_boost_round=100,
            valid_sets=[train_data],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        self.model = model
        
        return model
    
    def evaluate_classification_model(self, model, X_test, y_test):
        """
        分類モデルを評価する
        
        Args:
            model (object): 学習済みモデル
            X_test (ndarray): テスト用特徴量
            y_test (ndarray): テスト用ターゲット
            
        Returns:
            dict: 評価指標
        """
        # 予測
        y_pred_proba = model.predict(X_test)
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
    
    def evaluate_regression_model(self, model, X_test, y_test):
        """
        回帰モデルを評価する
        
        Args:
            model (object): 学習済みモデル
            X_test (ndarray): テスト用特徴量
            y_test (ndarray): テスト用ターゲット
            
        Returns:
            dict: 評価指標
        """
        # 予測
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
    
    def plot_feature_importance(self, model, feature_names, top_n=20, save_path=None):
        """
        特徴量の重要度をプロットする
        
        Args:
            model (object): 学習済みモデル
            feature_names (list): 特徴量名のリスト
            top_n (int, optional): 表示する上位の特徴量数
            save_path (str, optional): 保存先のパス
            
        Returns:
            None
        """
        # 特徴量の重要度を取得
        importance = model.feature_importance(importance_type='gain')
        
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
    
    def save_model(self, model, model_name):
        """
        モデルを保存する
        
        Args:
            model (object): 保存するモデル
            model_name (str): モデル名
            
        Returns:
            str: 保存先のパス
        """
        # 現在の日時を取得
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存先のパス
        save_path = os.path.join(MODELS_DIR, f'{model_name}_{now}.txt')
        
        # モデルを保存
        model.save_model(save_path)
        
        logger.info(f"モデルを保存しました: {save_path}")
        
        return save_path
    
    def load_model(self, model_path):
        """
        モデルを読み込む
        
        Args:
            model_path (str): モデルファイルのパス
            
        Returns:
            object: 読み込んだモデル
        """
        try:
            model = lgb.Booster(model_file=model_path)
            self.model = model
            
            logger.info(f"モデルを読み込みました: {model_path}")
            
            return model
        except Exception as e:
            logger.error(f"モデルの読み込み中にエラーが発生しました: {e}")
            return None
    
    def predict_race(self, race_id, model=None, model_type='classification'):
        """
        レースの結果を予測する
        
        Args:
            race_id (str): レースID
            model (object, optional): 使用するモデル（Noneの場合は self.model を使用）
            model_type (str): モデルの種類（'classification' または 'regression'）
            
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
            
            # 予測
            if model_type == 'classification':
                y_pred_proba = model.predict(X)
                
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
            
            # 評価指標
            metrics = {
                'win_hit': win_hit,
                'place_hit': place_hit,
                'trio_hit': trio_hit
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"レース予測の評価中にエラーが発生しました: {e}")
            return None
    
    def run_classification_experiment(self, target='is_place', test_size=0.2, random_state=42):
        """
        分類実験を実行する
        
        Args:
            target (str): ターゲット変数名（'is_win' または 'is_place'）
            test_size (float): テストデータの割合
            random_state (int): 乱数シード
            
        Returns:
            tuple: (model, metrics, feature_names)
                model (object): 学習済みモデル
                metrics (dict): 評価指標
                feature_names (list): 特徴量名のリスト
        """
        if not self.connect_db():
            return None, None, None
        
        try:
            # データの取得
            features_df = self.get_features_data()
            
            if features_df.empty:
                logger.warning("特徴量データが空です")
                return None, None, None
            
            # データの準備
            X, y, feature_names = self.prepare_data_for_classification(features_df, target=target)
            
            if X is None or y is None:
                return None, None, None
            
            # 特徴量名を保存
            self.feature_names = feature_names
            
            # データの分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # モデルの学習
            model = self.train_classification_model(X_train, y_train)
            
            # モデルの評価
            metrics = self.evaluate_classification_model(model, X_test, y_test)
            
            # 特徴量重要度のプロット
            plot_path = os.path.join(RESULTS_DIR, f'feature_importance_{target}.png')
            self.plot_feature_importance(model, feature_names, save_path=plot_path)
            
            # モデルの保存
            model_path = self.save_model(model, f'classification_{target}')
            
            return model, metrics, feature_names
            
        except Exception as e:
            logger.error(f"分類実験中にエラーが発生しました: {e}")
            return None, None, None
            
        finally:
            self.close_db()
    
    def run_regression_experiment(self, target='order_of_finish', test_size=0.2, random_state=42):
        """
        回帰実験を実行する
        
        Args:
            target (str): ターゲット変数名（'order_of_finish'）
            test_size (float): テストデータの割合
            random_state (int): 乱数シード
            
        Returns:
            tuple: (model, metrics, feature_names)
                model (object): 学習済みモデル
                metrics (dict): 評価指標
                feature_names (list): 特徴量名のリスト
        """
        if not self.connect_db():
            return None, None, None
        
        try:
            # データの取得
            features_df = self.get_features_data()
            
            if features_df.empty:
                logger.warning("特徴量データが空です")
                return None, None, None
            
            # データの準備
            X, y, feature_names = self.prepare_data_for_regression(features_df, target=target)
            
            if X is None or y is None:
                return None, None, None
            
            # 特徴量名を保存
            self.feature_names = feature_names
            
            # データの分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # モデルの学習
            model = self.train_regression_model(X_train, y_train)
            
            # モデルの評価
            metrics = self.evaluate_regression_model(model, X_test, y_test)
            
            # 特徴量重要度のプロット
            plot_path = os.path.join(RESULTS_DIR, f'feature_importance_{target}.png')
            self.plot_feature_importance(model, feature_names, save_path=plot_path)
            
            # モデルの保存
            model_path = self.save_model(model, f'regression_{target}')
            
            return model, metrics, feature_names
            
        except Exception as e:
            logger.error(f"回帰実験中にエラーが発生しました: {e}")
            return None, None, None
            
        finally:
            self.close_db()
    
    def run_time_series_experiment(self, target='is_place', n_splits=5):
        """
        時系列分割による実験を実行する
        
        Args:
            target (str): ターゲット変数名
            n_splits (int): 分割数
            
        Returns:
            tuple: (model, metrics, feature_names)
                model (object): 学習済みモデル
                metrics (dict): 評価指標
                feature_names (list): 特徴量名のリスト
        """
        if not self.connect_db():
            return None, None, None
        
        try:
            # データの取得
            features_df = self.get_features_data()
            
            if features_df.empty:
                logger.warning("特徴量データが空です")
                return None, None, None
            
            # 日付でソート
            features_df['race_date'] = pd.to_datetime(features_df['race_date'])
            features_df = features_df.sort_values('race_date')
            
            # データの準備
            if target in ['is_win', 'is_place']:
                X, y, feature_names = self.prepare_data_for_classification(features_df, target=target)
                is_classification = True
            else:
                X, y, feature_names = self.prepare_data_for_regression(features_df, target=target)
                is_classification = False
            
            if X is None or y is None:
                return None, None, None
            
            # 特徴量名を保存
            self.feature_names = feature_names
            
            # 時系列分割
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            fold_metrics = []
            
            for fold, (train_index, test_index) in enumerate(tscv.split(X)):
                logger.info(f"時系列分割 {fold+1}/{n_splits}")
                
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # モデルの学習
                if is_classification:
                    model = self.train_classification_model(X_train, y_train)
                    metrics = self.evaluate_classification_model(model, X_test, y_test)
                else:
                    model = self.train_regression_model(X_train, y_train)
                    metrics = self.evaluate_regression_model(model, X_test, y_test)
                
                fold_metrics.append(metrics)
            
            # 最終モデルの学習（全データ）
            if is_classification:
                final_model = self.train_classification_model(X, y)
            else:
                final_model = self.train_regression_model(X, y)
            
            # 特徴量重要度のプロット
            plot_path = os.path.join(RESULTS_DIR, f'feature_importance_ts_{target}.png')
            self.plot_feature_importance(final_model, feature_names, save_path=plot_path)
            
            # モデルの保存
            model_type = 'classification' if is_classification else 'regression'
            model_path = self.save_model(final_model, f'{model_type}_ts_{target}')
            
            # 平均メトリクスの計算
            avg_metrics = {}
            
            for key in fold_metrics[0].keys():
                if key != 'confusion_matrix':
                    avg_metrics[key] = np.mean([m[key] for m in fold_metrics])
            
            logger.info(f"時系列分割の平均評価指標: {avg_metrics}")
            
            return final_model, avg_metrics, feature_names
            
        except Exception as e:
            logger.error(f"時系列実験中にエラーが発生しました: {e}")
            return None, None, None
            
        finally:
            self.close_db()


def main():
    """メイン関数"""
    baseline = BaselineModel()
    
    # 複勝圏内予測モデル（分類）
    logger.info("複勝圏内予測モデルの学習を開始します")
    place_model, place_metrics, place_features = baseline.run_classification_experiment(target='is_place')
    
    if place_model and place_metrics:
        logger.info(f"複勝圏内予測モデルの評価指標: {place_metrics}")
    
    # 勝利予測モデル（分類）
    logger.info("勝利予測モデルの学習を開始します")
    win_model, win_metrics, win_features = baseline.run_classification_experiment(target='is_win')
    
    if win_model and win_metrics:
        logger.info(f"勝利予測モデルの評価指標: {win_metrics}")
    
    # 着順予測モデル（回帰）
    logger.info("着順予測モデルの学習を開始します")
    order_model, order_metrics, order_features = baseline.run_regression_experiment(target='order_of_finish')
    
    if order_model and order_metrics:
        logger.info(f"着順予測モデルの評価指標: {order_metrics}")
    
    # 時系列実験（複勝圏内予測）
    logger.info("時系列分割による複勝圏内予測モデルの学習を開始します")
    ts_model, ts_metrics, ts_features = baseline.run_time_series_experiment(target='is_place')
    
    if ts_model and ts_metrics:
        logger.info(f"時系列分割による複勝圏内予測モデルの評価指標: {ts_metrics}")


if __name__ == '__main__':
    main()
