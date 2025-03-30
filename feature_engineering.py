#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
特徴量エンジニアリングモジュール
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定数
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_FILE = os.path.join(DATA_DIR, 'horse_racing.db')

class FeatureEngineering:
    """
    特徴量エンジニアリングを行うクラス
    """
    
    def __init__(self, db_file=DB_FILE):
        """
        初期化
        
        Args:
            db_file (str): データベースファイルのパス
        """
        self.db_file = db_file
        self.conn = None
    
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
    
    def get_race_data(self, race_id):
        """
        レースデータを取得する
        
        Args:
            race_id (str): レースID
            
        Returns:
            tuple: (race_info, race_results)
                race_info (dict): レース情報
                race_results (DataFrame): レース結果
        """
        try:
            # レース情報の取得
            race_query = "SELECT * FROM races WHERE race_id = ?"
            race_info = pd.read_sql_query(race_query, self.conn, params=(race_id,))
            
            if race_info.empty:
                logger.warning(f"レース情報が見つかりませんでした: {race_id}")
                return None, None
            
            race_info = race_info.iloc[0].to_dict()
            
            # レース結果の取得
            results_query = """
            SELECT rr.*, h.horse_name, j.jockey_name
            FROM race_results rr
            LEFT JOIN horses h ON rr.horse_id = h.horse_id
            LEFT JOIN jockeys j ON rr.jockey_id = j.jockey_id
            WHERE rr.race_id = ?
            """
            race_results = pd.read_sql_query(results_query, self.conn, params=(race_id,))
            
            if race_results.empty:
                logger.warning(f"レース結果が見つかりませんでした: {race_id}")
                return race_info, None
            
            return race_info, race_results
        
        except Exception as e:
            logger.error(f"レースデータの取得中にエラーが発生しました: {e}")
            return None, None
    
    def get_horse_history(self, horse_id, before_date=None):
        """
        馬の過去のレース履歴を取得する
        
        Args:
            horse_id (str): 馬ID
            before_date (str, optional): この日付より前のレース履歴を取得
            
        Returns:
            DataFrame: 馬のレース履歴
        """
        try:
            if before_date:
                query = """
                SELECT * FROM horse_race_history
                WHERE horse_id = ? AND race_date < ?
                ORDER BY race_date DESC
                """
                history = pd.read_sql_query(query, self.conn, params=(horse_id, before_date))
            else:
                query = """
                SELECT * FROM horse_race_history
                WHERE horse_id = ?
                ORDER BY race_date DESC
                """
                history = pd.read_sql_query(query, self.conn, params=(horse_id,))
            
            return history
        
        except Exception as e:
            logger.error(f"馬の履歴取得中にエラーが発生しました: {e}")
            return pd.DataFrame()
    
    def get_jockey_stats(self, jockey_id, before_date=None, track_type=None, distance_range=None):
        """
        騎手の統計情報を取得する
        
        Args:
            jockey_id (str): 騎手ID
            before_date (str, optional): この日付より前のレース結果を取得
            track_type (str, optional): トラックタイプ（'芝' or 'ダート'）
            distance_range (tuple, optional): 距離の範囲 (min, max)
            
        Returns:
            dict: 騎手の統計情報
        """
        try:
            conditions = ["jockey_id = ?"]
            params = [jockey_id]
            
            if before_date:
                conditions.append("race_date < ?")
                params.append(before_date)
            
            if track_type:
                conditions.append("track_type = ?")
                params.append(track_type)
            
            if distance_range:
                conditions.append("distance BETWEEN ? AND ?")
                params.extend(distance_range)
            
            where_clause = " AND ".join(conditions)
            
            query = f"""
            SELECT 
                COUNT(*) as total_races,
                SUM(CASE WHEN order_of_finish = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN order_of_finish <= 3 THEN 1 ELSE 0 END) as top3,
                AVG(CASE WHEN order_of_finish > 0 THEN order_of_finish ELSE NULL END) as avg_order
            FROM horse_race_history
            WHERE {where_clause}
            """
            
            stats = pd.read_sql_query(query, self.conn, params=params)
            
            if stats.empty:
                return {
                    'total_races': 0,
                    'wins': 0,
                    'win_rate': 0.0,
                    'top3_rate': 0.0,
                    'avg_order': 0.0
                }
            
            stats = stats.iloc[0].to_dict()
            
            # 勝率と連対率の計算
            total_races = stats['total_races']
            if total_races > 0:
                stats['win_rate'] = stats['wins'] / total_races
                stats['top3_rate'] = stats['top3'] / total_races
            else:
                stats['win_rate'] = 0.0
                stats['top3_rate'] = 0.0
            
            return stats
        
        except Exception as e:
            logger.error(f"騎手の統計情報取得中にエラーが発生しました: {e}")
            return {
                'total_races': 0,
                'wins': 0,
                'win_rate': 0.0,
                'top3_rate': 0.0,
                'avg_order': 0.0
            }
    
    def calculate_basic_features(self, race_id):
        """
        基本的な特徴量を計算する
        
        Args:
            race_id (str): レースID
            
        Returns:
            DataFrame: 特徴量のDataFrame
        """
        race_info, race_results = self.get_race_data(race_id)
        
        if race_info is None or race_results is None or race_results.empty:
            logger.warning(f"レースデータが不足しているため特徴量を計算できません: {race_id}")
            return pd.DataFrame()
        
        # レース日を取得
        race_date = race_info.get('race_date')
        if not race_date:
            logger.warning(f"レース日が不明です: {race_id}")
            race_date = '2000-01-01'  # デフォルト値
        
        # 特徴量のDataFrameを初期化
        features = pd.DataFrame()
        features['horse_id'] = race_results['horse_id']
        features['race_id'] = race_id
        
        # レース情報から特徴量を追加
        features['track_type'] = 1 if race_info.get('track_type') == '芝' else 0
        features['distance'] = race_info.get('distance', 0)
        features['weather_sunny'] = 1 if race_info.get('weather') == '晴' else 0
        features['weather_cloudy'] = 1 if race_info.get('weather') == '曇' else 0
        features['weather_rainy'] = 1 if race_info.get('weather') == '雨' else 0
        features['track_condition_good'] = 1 if race_info.get('track_condition') == '良' else 0
        features['track_condition_slightly_heavy'] = 1 if race_info.get('track_condition') == '稍重' else 0
        features['track_condition_heavy'] = 1 if race_info.get('track_condition') == '重' else 0
        features['track_condition_bad'] = 1 if race_info.get('track_condition') == '不良' else 0
        
        # レース結果から特徴量を追加
        features['post_position'] = race_results['post_position'].astype(float)
        features['horse_number'] = race_results['horse_number'].astype(float)
        features['weight'] = race_results['weight'].astype(float)
        features['weight_change'] = race_results['weight_change'].astype(float)
        
        # 馬ごとの特徴量を計算
        for i, row in race_results.iterrows():
            horse_id = row['horse_id']
            jockey_id = row['jockey_id']
            
            # 馬の過去のレース履歴を取得
            horse_history = self.get_horse_history(horse_id, before_date=race_date)
            
            # 過去のレース数
            features.loc[features['horse_id'] == horse_id, 'past_races_count'] = len(horse_history)
            
            if not horse_history.empty:
                # 直近3レースの平均着順
                recent_races = horse_history.head(3)
                avg_order = recent_races['order_of_finish'].mean()
                features.loc[features['horse_id'] == horse_id, 'avg_order_last3'] = avg_order
                
                # 勝率
                win_rate = (horse_history['order_of_finish'] == 1).mean()
                features.loc[features['horse_id'] == horse_id, 'win_rate'] = win_rate
                
                # 複勝率（3着以内に入る確率）
                place_rate = (horse_history['order_of_finish'] <= 3).mean()
                features.loc[features['horse_id'] == horse_id, 'place_rate'] = place_rate
                
                # 同じトラックタイプでの成績
                same_track = horse_history[horse_history['track_type'] == race_info.get('track_type', '')]
                if not same_track.empty:
                    same_track_win_rate = (same_track['order_of_finish'] == 1).mean()
                    features.loc[features['horse_id'] == horse_id, 'same_track_win_rate'] = same_track_win_rate
                else:
                    features.loc[features['horse_id'] == horse_id, 'same_track_win_rate'] = 0.0
                
                # 同じ距離での成績
                same_distance = horse_history[horse_history['distance'] == race_info.get('distance', 0)]
                if not same_distance.empty:
                    same_distance_win_rate = (same_distance['order_of_finish'] == 1).mean()
                    features.loc[features['horse_id'] == horse_id, 'same_distance_win_rate'] = same_distance_win_rate
                else:
                    features.loc[features['horse_id'] == horse_id, 'same_distance_win_rate'] = 0.0
                
                # 同じ馬場状態での成績
                same_condition = horse_history[horse_history['track_condition'] == race_info.get('track_condition', '')]
                if not same_condition.empty:
                    same_condition_win_rate = (same_condition['order_of_finish'] == 1).mean()
                    features.loc[features['horse_id'] == horse_id, 'same_condition_win_rate'] = same_condition_win_rate
                else:
                    features.loc[features['horse_id'] == horse_id, 'same_condition_win_rate'] = 0.0
                
                # 平均タイム
                avg_time = horse_history['time_seconds'].mean()
                features.loc[features['horse_id'] == horse_id, 'avg_time'] = avg_time
                
                # 平均上がり3F
                avg_last_3f = horse_history['last_3f_time'].mean()
                features.loc[features['horse_id'] == horse_id, 'avg_last_3f'] = avg_last_3f
            else:
                # 履歴がない場合はデフォルト値を設定
                features.loc[features['horse_id'] == horse_id, 'avg_order_last3'] = 0.0
                features.loc[features['horse_id'] == horse_id, 'win_rate'] = 0.0
                features.loc[features['horse_id'] == horse_id, 'place_rate'] = 0.0
                features.loc[features['horse_id'] == horse_id, 'same_track_win_rate'] = 0.0
                features.loc[features['horse_id'] == horse_id, 'same_distance_win_rate'] = 0.0
                features.loc[features['horse_id'] == horse_id, 'same_condition_win_rate'] = 0.0
                features.loc[features['horse_id'] == horse_id, 'avg_time'] = 0.0
                features.loc[features['horse_id'] == horse_id, 'avg_last_3f'] = 0.0
            
            # 騎手の統計情報
            jockey_stats = self.get_jockey_stats(jockey_id, before_date=race_date)
            features.loc[features['horse_id'] == horse_id, 'jockey_total_races'] = jockey_stats.get('total_races', 0)
            features.loc[features['horse_id'] == horse_id, 'jockey_win_rate'] = jockey_stats.get('win_rate', 0.0)
            features.loc[features['horse_id'] == horse_id, 'jockey_top3_rate'] = jockey_stats.get('top3_rate', 0.0)
            
            # 同じトラックタイプでの騎手の成績
            jockey_track_stats = self.get_jockey_stats(
                jockey_id, before_date=race_date, track_type=race_info.get('track_type')
            )
            features.loc[features['horse_id'] == horse_id, 'jockey_track_win_rate'] = jockey_track_stats.get('win_rate', 0.0)
            
            # 同じ距離での騎手の成績
            distance = race_info.get('distance', 0)
            distance_range = (distance - 200, distance + 200)  # 距離の範囲
            jockey_distance_stats = self.get_jockey_stats(
                jockey_id, before_date=race_date, distance_range=distance_range
            )
            features.loc[features['horse_id'] == horse_id, 'jockey_distance_win_rate'] = jockey_distance_stats.get('win_rate', 0.0)
        
        # 欠損値を0で埋める
        features = features.fillna(0)
        
        return features
    
    def calculate_advanced_features(self, race_id, basic_features):
        """
        高度な特徴量を計算する
        
        Args:
            race_id (str): レースID
            basic_features (DataFrame): 基本的な特徴量
            
        Returns:
            DataFrame: 高度な特徴量を追加したDataFrame
        """
        if basic_features.empty:
            logger.warning(f"基本特徴量が空のため高度な特徴量を計算できません: {race_id}")
            return pd.DataFrame()
        
        # 特徴量のコピーを作成
        features = basic_features.copy()
        
        # レース内での相対的な特徴量
        # 馬体重の相対値
        mean_weight = features['weight'].mean()
        std_weight = features['weight'].std()
        if std_weight > 0:
            features['relative_weight'] = (features['weight'] - mean_weight) / std_weight
        else:
            features['relative_weight'] = 0
        
        # 勝率の相対値
        mean_win_rate = features['win_rate'].mean()
        std_win_rate = features['win_rate'].std()
        if std_win_rate > 0:
            features['relative_win_rate'] = (features['win_rate'] - mean_win_rate) / std_win_rate
        else:
            features['relative_win_rate'] = 0
        
        # 騎手の勝率の相対値
        mean_jockey_win_rate = features['jockey_win_rate'].mean()
        std_jockey_win_rate = features['jockey_win_rate'].std()
        if std_jockey_win_rate > 0:
            features['relative_jockey_win_rate'] = (features['jockey_win_rate'] - mean_jockey_win_rate) / std_jockey_win_rate
        else:
            features['relative_jockey_win_rate'] = 0
        
        # 交互作用特徴量
        # 馬と騎手の相性（勝率の積）
        features['horse_jockey_synergy'] = features['win_rate'] * features['jockey_win_rate']
        
        # トラックタイプと馬の相性
        features['horse_track_synergy'] = features['same_track_win_rate'] * features['track_type']
        
        # 距離と馬の相性
        features['horse_distance_synergy'] = features['same_distance_win_rate'] * features['distance'] / 1000
        
        # 馬場状態と馬の相性
        track_condition_features = [
            'track_condition_good', 
            'track_condition_slightly_heavy', 
            'track_condition_heavy', 
            'track_condition_bad'
        ]
        
        for condition in track_condition_features:
            features[f'horse_{condition}_synergy'] = features['same_condition_win_rate'] * features[condition]
        
        # 時系列特徴量
        # 直近の成績トレンド（直近3レースの平均着順）
        # すでに基本特徴量で計算済み（avg_order_last3）
        
        return features
    
    def save_features_to_db(self, features):
        """
        特徴量をデータベースに保存する
        
        Args:
            features (DataFrame): 特徴量のDataFrame
            
        Returns:
            bool: 保存に成功したかどうか
        """
        try:
            cursor = self.conn.cursor()
            
            # 既存の特徴量を削除
            for _, row in features.iterrows():
                race_id = row['race_id']
                horse_id = row['horse_id']
                
                cursor.execute(
                    "DELETE FROM features WHERE race_id = ? AND horse_id = ?",
                    (race_id, horse_id)
                )
            
            # 新しい特徴量を挿入
            for _, row in features.iterrows():
                race_id = row['race_id']
                horse_id = row['horse_id']
                
                for column in features.columns:
                    if column not in ['race_id', 'horse_id']:
                        feature_name = column
                        feature_value = float(row[column])
                        
                        cursor.execute(
                            "INSERT INTO features (race_id, horse_id, feature_name, feature_value) VALUES (?, ?, ?, ?)",
                            (race_id, horse_id, feature_name, feature_value)
                        )
            
            self.conn.commit()
            logger.info(f"特徴量をデータベースに保存しました: {len(features)}件")
            return True
            
        except Exception as e:
            logger.error(f"特徴量の保存中にエラーが発生しました: {e}")
            return False
    
    def process_race(self, race_id):
        """
        レースの特徴量を処理する
        
        Args:
            race_id (str): レースID
            
        Returns:
            DataFrame: 特徴量のDataFrame
        """
        logger.info(f"レース {race_id} の特徴量を処理中...")
        
        # 基本的な特徴量を計算
        basic_features = self.calculate_basic_features(race_id)
        
        if basic_features.empty:
            logger.warning(f"基本特徴量の計算に失敗しました: {race_id}")
            return pd.DataFrame()
        
        # 高度な特徴量を計算
        advanced_features = self.calculate_advanced_features(race_id, basic_features)
        
        if advanced_features.empty:
            logger.warning(f"高度な特徴量の計算に失敗しました: {race_id}")
            return basic_features
        
        # 特徴量をデータベースに保存
        self.save_features_to_db(advanced_features)
        
        return advanced_features
    
    def process_races(self, race_ids=None, limit=100):
        """
        複数のレースの特徴量を処理する
        
        Args:
            race_ids (list, optional): 処理するレースIDのリスト
            limit (int, optional): 処理する最大レース数
            
        Returns:
            int: 処理したレース数
        """
        if not self.connect_db():
            return 0
        
        try:
            if race_ids is None:
                # データベースから最新のレースを取得
                query = """
                SELECT race_id FROM races
                ORDER BY race_date DESC
                LIMIT ?
                """
                races_df = pd.read_sql_query(query, self.conn, params=(limit,))
                race_ids = races_df['race_id'].tolist()
            
            processed_count = 0
            
            for race_id in race_ids:
                features = self.process_race(race_id)
                
                if not features.empty:
                    processed_count += 1
                
                logger.info(f"進捗: {processed_count}/{len(race_ids)}")
            
            logger.info(f"合計 {processed_count} レースの特徴量を処理しました")
            return processed_count
            
        except Exception as e:
            logger.error(f"レースの処理中にエラーが発生しました: {e}")
            return 0
            
        finally:
            self.close_db()


def main():
    """メイン関数"""
    feature_engineering = FeatureEngineering()
    
    # テスト用に最新の10レースを処理
    processed_count = feature_engineering.process_races(limit=10)
    
    logger.info(f"処理完了: {processed_count}レース")


if __name__ == '__main__':
    main()
