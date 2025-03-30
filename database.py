#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
データベース設計と実装
"""

import os
import sqlite3
import logging
import pandas as pd

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定数
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
DB_FILE = os.path.join(DATA_DIR, 'horse_racing.db')

# ディレクトリの作成
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

class HorseRacingDatabase:
    """
    競馬データベースを管理するクラス
    """
    
    def __init__(self, db_file=DB_FILE):
        """
        初期化
        
        Args:
            db_file (str): データベースファイルのパス
        """
        self.db_file = db_file
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """
        データベースに接続する
        
        Returns:
            bool: 接続に成功したかどうか
        """
        try:
            self.conn = sqlite3.connect(self.db_file)
            self.cursor = self.conn.cursor()
            logger.info(f"データベースに接続しました: {self.db_file}")
            return True
        except Exception as e:
            logger.error(f"データベース接続中にエラーが発生しました: {e}")
            return False
    
    def close(self):
        """
        データベース接続を閉じる
        """
        if self.conn:
            self.conn.close()
            logger.info("データベース接続を閉じました")
    
    def create_tables(self):
        """
        必要なテーブルを作成する
        
        Returns:
            bool: テーブル作成に成功したかどうか
        """
        try:
            # レーステーブル
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS races (
                race_id TEXT PRIMARY KEY,
                race_name TEXT,
                race_date TEXT,
                race_round TEXT,
                track_type TEXT,
                distance INTEGER,
                weather TEXT,
                track_condition TEXT,
                race_details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # 馬テーブル
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS horses (
                horse_id TEXT PRIMARY KEY,
                horse_name TEXT,
                birth_date TEXT,
                sex TEXT,
                color TEXT,
                trainer TEXT,
                owner TEXT,
                breeder TEXT,
                birthplace TEXT,
                father TEXT,
                mother TEXT,
                maternal_grandfather TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # 騎手テーブル
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS jockeys (
                jockey_id TEXT PRIMARY KEY,
                jockey_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # レース結果テーブル
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS race_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                horse_id TEXT,
                jockey_id TEXT,
                order_of_finish INTEGER,
                post_position INTEGER,
                horse_number INTEGER,
                odds REAL,
                popularity INTEGER,
                weight INTEGER,
                weight_change INTEGER,
                time_seconds REAL,
                margin TEXT,
                corner_position TEXT,
                last_3f_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (race_id) REFERENCES races (race_id),
                FOREIGN KEY (horse_id) REFERENCES horses (horse_id),
                FOREIGN KEY (jockey_id) REFERENCES jockeys (jockey_id)
            )
            ''')
            
            # 馬のレース履歴テーブル
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS horse_race_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                horse_id TEXT,
                race_date TEXT,
                race_name TEXT,
                race_id TEXT,
                track_type TEXT,
                distance INTEGER,
                weather TEXT,
                track_condition TEXT,
                order_of_finish INTEGER,
                jockey_id TEXT,
                weight INTEGER,
                time_seconds REAL,
                margin TEXT,
                corner_position TEXT,
                last_3f_time REAL,
                odds REAL,
                popularity INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (horse_id) REFERENCES horses (horse_id),
                FOREIGN KEY (jockey_id) REFERENCES jockeys (jockey_id)
            )
            ''')
            
            # 特徴量テーブル
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                horse_id TEXT,
                feature_name TEXT,
                feature_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (race_id) REFERENCES races (race_id),
                FOREIGN KEY (horse_id) REFERENCES horses (horse_id)
            )
            ''')
            
            # 予測結果テーブル
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                horse_id TEXT,
                model_name TEXT,
                prediction_type TEXT,
                predicted_value REAL,
                confidence REAL,
                actual_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (race_id) REFERENCES races (race_id),
                FOREIGN KEY (horse_id) REFERENCES horses (horse_id)
            )
            ''')
            
            self.conn.commit()
            logger.info("テーブルを作成しました")
            return True
        except Exception as e:
            logger.error(f"テーブル作成中にエラーが発生しました: {e}")
            return False
    
    def insert_race_info(self, race_info):
        """
        レース情報をデータベースに挿入する
        
        Args:
            race_info (dict): レース情報の辞書
            
        Returns:
            bool: 挿入に成功したかどうか
        """
        try:
            self.cursor.execute('''
            INSERT OR REPLACE INTO races (
                race_id, race_name, race_date, race_round, track_type, 
                distance, weather, track_condition, race_details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                race_info.get('race_id'),
                race_info.get('race_name'),
                race_info.get('race_date'),
                race_info.get('race_round'),
                race_info.get('track_type'),
                race_info.get('distance'),
                race_info.get('weather'),
                race_info.get('track_condition'),
                race_info.get('race_details')
            ))
            
            self.conn.commit()
            logger.info(f"レース情報を挿入しました: {race_info.get('race_id')}")
            return True
        except Exception as e:
            logger.error(f"レース情報の挿入中にエラーが発生しました: {e}")
            return False
    
    def insert_horse_info(self, horse_info):
        """
        馬情報をデータベースに挿入する
        
        Args:
            horse_info (dict): 馬情報の辞書
            
        Returns:
            bool: 挿入に成功したかどうか
        """
        try:
            self.cursor.execute('''
            INSERT OR REPLACE INTO horses (
                horse_id, horse_name, birth_date, sex, color, trainer, 
                owner, breeder, birthplace, father, mother, maternal_grandfather
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                horse_info.get('horse_id'),
                horse_info.get('horse_name'),
                horse_info.get('birth_date'),
                horse_info.get('sex'),
                horse_info.get('color'),
                horse_info.get('trainer'),
                horse_info.get('owner'),
                horse_info.get('breeder'),
                horse_info.get('birthplace'),
                horse_info.get('father'),
                horse_info.get('mother'),
                horse_info.get('maternal_grandfather')
            ))
            
            self.conn.commit()
            logger.info(f"馬情報を挿入しました: {horse_info.get('horse_id')}")
            return True
        except Exception as e:
            logger.error(f"馬情報の挿入中にエラーが発生しました: {e}")
            return False
    
    def insert_jockey(self, jockey_id, jockey_name):
        """
        騎手情報をデータベースに挿入する
        
        Args:
            jockey_id (str): 騎手ID
            jockey_name (str): 騎手名
            
        Returns:
            bool: 挿入に成功したかどうか
        """
        try:
            self.cursor.execute('''
            INSERT OR REPLACE INTO jockeys (jockey_id, jockey_name)
            VALUES (?, ?)
            ''', (jockey_id, jockey_name))
            
            self.conn.commit()
            logger.info(f"騎手情報を挿入しました: {jockey_id}")
            return True
        except Exception as e:
            logger.error(f"騎手情報の挿入中にエラーが発生しました: {e}")
            return False
    
    def insert_race_results(self, race_results_df):
        """
        レース結果をデータベースに挿入する
        
        Args:
            race_results_df (DataFrame): レース結果のDataFrame
            
        Returns:
            bool: 挿入に成功したかどうか
        """
        try:
            # DataFrameをレコードのリストに変換
            records = race_results_df.to_dict('records')
            
            for record in records:
                # 着順を数値に変換
                try:
                    order_of_finish = int(record.get('着順', 0))
                except (ValueError, TypeError):
                    order_of_finish = 0
                
                # 枠番を数値に変換
                try:
                    post_position = int(record.get('枠番', 0))
                except (ValueError, TypeError):
                    post_position = 0
                
                # 馬番を数値に変換
                try:
                    horse_number = int(record.get('馬番', 0))
                except (ValueError, TypeError):
                    horse_number = 0
                
                # オッズを数値に変換
                try:
                    odds = float(record.get('オッズ', 0))
                except (ValueError, TypeError):
                    odds = 0.0
                
                # 人気を数値に変換
                try:
                    popularity = int(record.get('人気', 0))
                except (ValueError, TypeError):
                    popularity = 0
                
                # 馬体重を数値に変換
                weight = 0
                weight_change = 0
                if '馬体重' in record and record['馬体重']:
                    weight_str = record['馬体重']
                    if '(' in weight_str:
                        weight_parts = weight_str.split('(')
                        try:
                            weight = int(weight_parts[0])
                            weight_change = int(weight_parts[1].replace(')', '').replace('+', ''))
                            if '-' in weight_parts[1]:
                                weight_change = -weight_change
                        except (ValueError, IndexError):
                            pass
                
                # タイムを秒に変換
                time_seconds = 0.0
                if 'タイム' in record and record['タイム']:
                    time_str = record['タイム']
                    try:
                        minutes, seconds = time_str.split(':')
                        time_seconds = float(minutes) * 60 + float(seconds)
                    except (ValueError, TypeError):
                        pass
                
                # 上がり3Fを数値に変換
                last_3f_time = 0.0
                if '上り' in record and record['上り']:
                    try:
                        last_3f_time = float(record['上り'])
                    except (ValueError, TypeError):
                        pass
                
                self.cursor.execute('''
                INSERT INTO race_results (
                    race_id, horse_id, jockey_id, order_of_finish, post_position, 
                    horse_number, odds, popularity, weight, weight_change, 
                    time_seconds, margin, corner_position, last_3f_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.get('race_id'),
                    record.get('horse_id'),
                    record.get('jockey_id'),
                    order_of_finish,
                    post_position,
                    horse_number,
                    odds,
                    popularity,
                    weight,
                    weight_change,
                    time_seconds,
                    record.get('着差'),
                    record.get('通過'),
                    last_3f_time
                ))
            
            self.conn.commit()
            logger.info(f"レース結果を挿入しました: {len(records)}件")
            return True
        except Exception as e:
            logger.error(f"レース結果の挿入中にエラーが発生しました: {e}")
            return False
    
    def insert_horse_race_history(self, horse_id, race_history_df):
        """
        馬のレース履歴をデータベースに挿入する
        
        Args:
            horse_id (str): 馬ID
            race_history_df (DataFrame): レース履歴のDataFrame
            
        Returns:
            bool: 挿入に成功したかどうか
        """
        try:
            # DataFrameをレコードのリストに変換
            records = race_history_df.to_dict('records')
            
            for record in records:
                # 日付を取得
                race_date = record.get('日付', '')
                
                # レース名を取得
                race_name = record.get('レース名', '')
                
                # 着順を数値に変換
                try:
                    order_of_finish = int(record.get('着順', 0))
                except (ValueError, TypeError):
                    order_of_finish = 0
                
                # 距離とコース種別を取得
                track_type = ''
                distance = 0
                if '距離' in record and record['距離']:
                    distance_str = record['距離']
                    if '芝' in distance_str:
                        track_type = '芝'
                    elif 'ダ' in distance_str:
                        track_type = 'ダート'
                    
                    try:
                        distance = int(''.join(filter(str.isdigit, distance_str)))
                    except (ValueError, TypeError):
                        pass
                
                # 馬場状態を取得
                track_condition = ''
                if '馬場' in record and record['馬場']:
                    track_condition = record['馬場']
                
                # 天候を取得
                weather = ''
                if '天候' in record and record['天候']:
                    weather = record['天候']
                
                # タイムを秒に変換
                time_seconds = 0.0
                if 'タイム' in record and record['タイム']:
                    time_str = record['タイム']
                    try:
                        minutes, seconds = time_str.split(':')
                        time_seconds = float(minutes) * 60 + float(seconds)
                    except (ValueError, TypeError):
                        pass
                
                # 騎手IDを取得（ここでは仮のIDを生成）
                jockey_name = record.get('騎手', '')
                jockey_id = f"jockey_{hash(jockey_name) % 10000:04d}"
                
                # 騎手情報を挿入
                if jockey_name:
                    self.insert_jockey(jockey_id, jockey_name)
                
                # 馬体重を数値に変換
                weight = 0
                if '馬体重' in record and record['馬体重']:
                    weight_str = record['馬体重']
                    if '(' in weight_str:
                        try:
                            weight = int(weight_str.split('(')[0])
                        except (ValueError, IndexError):
                            pass
                
                # 上がり3Fを数値に変換
                last_3f_time = 0.0
                if '上り' in record and record['上り']:
                    try:
                        last_3f_time = float(record['上り'])
                    except (ValueError, TypeError):
                        pass
                
                # オッズを数値に変換
                odds = 0.0
                if 'オッズ' in record and record['オッズ']:
                    try:
                        odds = float(record['オッズ'])
                    except (ValueError, TypeError):
                        pass
                
                # 人気を数値に変換
                popularity = 0
                if '人気' in record and record['人気']:
                    try:
                        popularity = int(record['人気'])
                    except (ValueError, TypeError):
                        pass
                
                self.cursor.execute('''
                INSERT INTO horse_race_history (
                    horse_id, race_date, race_name, track_type, distance, 
                    weather, track_condition, order_of_finish, jockey_id, 
                    weight, time_seconds, margin, corner_position, last_3f_time, 
                    odds, popularity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    horse_id,
                    race_date,
                    race_name,
                    track_type,
                    distance,
                    weather,
                    track_condition,
                    order_of_finish,
                    jockey_id,
                    weight,
                    time_seconds,
                    record.get('着差', ''),
                    record.get('通過', ''),
                    last_3f_time,
                    odds,
                    popularity
                ))
            
            self.conn.commit()
            logger.info(f"馬のレース履歴を挿入しました: {horse_id}, {len(records)}件")
            return True
        except Exception as e:
            logger.error(f"馬のレース履歴の挿入中にエラーが発生しました: {e}")
            return False
    
    def insert_feature(self, race_id, horse_id, feature_name, feature_value):
        """
        特徴量をデータベースに挿入する
        
        Args:
            race_id (str): レースID
            horse_id (str): 馬ID
            feature_name (str): 特徴量名
            feature_value (float): 特徴量値
            
        Returns:
            bool: 挿入に成功したかどうか
        """
        try:
            self.cursor.execute('''
            INSERT INTO features (race_id, horse_id, feature_name, feature_value)
            VALUES (?, ?, ?, ?)
            ''', (race_id, horse_id, feature_name, feature_value))
            
            self.conn.commit()
            logger.info(f"特徴量を挿入しました: {race_id}, {horse_id}, {feature_name}")
            return True
        except Exception as e:
            logger.error(f"特徴量の挿入中にエラーが発生しました: {e}")
            return False
    
    def insert_prediction(self, race_id, horse_id, model_name, prediction_type, 
                         predicted_value, confidence=None, actual_value=None):
        """
        予測結果をデータベースに挿入する
        
        Args:
            race_id (str): レースID
            horse_id (str): 馬ID
            model_name (str): モデル名
            prediction_type (str): 予測タイプ（例: 'win', 'place', 'show'）
            predicted_value (float): 予測値
            confidence (float, optional): 信頼度
            actual_value (float, optional): 実際の値
            
        Returns:
            bool: 挿入に成功したかどうか
        """
        try:
            self.cursor.execute('''
            INSERT INTO predictions (
                race_id, horse_id, model_name, prediction_type, 
                predicted_value, confidence, actual_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                race_id, horse_id, model_name, prediction_type, 
                predicted_value, confidence, actual_value
            ))
            
            self.conn.commit()
            logger.info(f"予測結果を挿入しました: {race_id}, {horse_id}, {model_name}")
            return True
        except Exception as e:
            logger.error(f"予測結果の挿入中にエラーが発生しました: {e}")
            return False
    
    def get_races(self, limit=100):
        """
        レース情報を取得する
        
        Args:
            limit (int): 取得する最大件数
            
        Returns:
            list: レース情報の辞書のリスト
        """
        try:
            self.cursor.execute('''
            SELECT * FROM races
            ORDER BY race_date DESC
            LIMIT ?
            ''', (limit,))
            
            columns = [column[0] for column in self.cursor.description]
            races = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            
            return races
        except Exception as e:
            logger.error(f"レース情報の取得中にエラーが発生しました: {e}")
            return []
    
    def get_race_results(self, race_id):
        """
        レース結果を取得する
        
        Args:
            race_id (str): レースID
            
        Returns:
            list: レース結果の辞書のリスト
        """
        try:
            self.cursor.execute('''
            SELECT rr.*, h.horse_name, j.jockey_name
            FROM race_results rr
            LEFT JOIN horses h ON rr.horse_id = h.horse_id
            LEFT JOIN jockeys j ON rr.jockey_id = j.jockey_id
            WHERE rr.race_id = ?
            ORDER BY rr.order_of_finish
            ''', (race_id,))
            
            columns = [column[0] for column in self.cursor.description]
            results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            
            return results
        except Exception as e:
            logger.error(f"レース結果の取得中にエラーが発生しました: {e}")
            return []
    
    def get_horse_info(self, horse_id):
        """
        馬情報を取得する
        
        Args:
            horse_id (str): 馬ID
            
        Returns:
            dict: 馬情報の辞書
        """
        try:
            self.cursor.execute('''
            SELECT * FROM horses
            WHERE horse_id = ?
            ''', (horse_id,))
            
            columns = [column[0] for column in self.cursor.description]
            row = self.cursor.fetchone()
            
            if row:
                return dict(zip(columns, row))
            else:
                return None
        except Exception as e:
            logger.error(f"馬情報の取得中にエラーが発生しました: {e}")
            return None
    
    def get_horse_race_history(self, horse_id):
        """
        馬のレース履歴を取得する
        
        Args:
            horse_id (str): 馬ID
            
        Returns:
            list: レース履歴の辞書のリスト
        """
        try:
            self.cursor.execute('''
            SELECT * FROM horse_race_history
            WHERE horse_id = ?
            ORDER BY race_date DESC
            ''', (horse_id,))
            
            columns = [column[0] for column in self.cursor.description]
            history = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            
            return history
        except Exception as e:
            logger.error(f"馬のレース履歴の取得中にエラーが発生しました: {e}")
            return []
    
    def get_features(self, race_id, horse_id=None):
        """
        特徴量を取得する
        
        Args:
            race_id (str): レースID
            horse_id (str, optional): 馬ID
            
        Returns:
            list: 特徴量の辞書のリスト
        """
        try:
            if horse_id:
                self.cursor.execute('''
                SELECT * FROM features
                WHERE race_id = ? AND horse_id = ?
                ''', (race_id, horse_id))
            else:
                self.cursor.execute('''
                SELECT * FROM features
                WHERE race_id = ?
                ''', (race_id,))
            
            columns = [column[0] for column in self.cursor.description]
            features = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            
            return features
        except Exception as e:
            logger.error(f"特徴量の取得中にエラーが発生しました: {e}")
            return []
    
    def get_predictions(self, race_id, model_name=None):
        """
        予測結果を取得する
        
        Args:
            race_id (str): レースID
            model_name (str, optional): モデル名
            
        Returns:
            list: 予測結果の辞書のリスト
        """
        try:
            if model_name:
                self.cursor.execute('''
                SELECT p.*, h.horse_name
                FROM predictions p
                LEFT JOIN horses h ON p.horse_id = h.horse_id
                WHERE p.race_id = ? AND p.model_name = ?
                ORDER BY p.predicted_value DESC
                ''', (race_id, model_name))
            else:
                self.cursor.execute('''
                SELECT p.*, h.horse_name
                FROM predictions p
                LEFT JOIN horses h ON p.horse_id = h.horse_id
                WHERE p.race_id = ?
                ORDER BY p.model_name, p.predicted_value DESC
                ''', (race_id,))
            
            columns = [column[0] for column in self.cursor.description]
            predictions = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            
            return predictions
        except Exception as e:
            logger.error(f"予測結果の取得中にエラーが発生しました: {e}")
            return []


def main():
    """メイン関数"""
    # データベースの初期化
    db = HorseRacingDatabase()
    
    if db.connect():
        # テーブルの作成
        db.create_tables()
        
        # 接続を閉じる
        db.close()
        
        logger.info("データベースの初期化が完了しました")
    else:
        logger.error("データベースの初期化に失敗しました")


if __name__ == '__main__':
    main()
