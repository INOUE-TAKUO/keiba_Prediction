#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
netkeiba.comからレース情報をスクレイピングするスクリプト
"""

import os
import time
import random
import logging
import datetime
import pandas as pd
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('netkeiba_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定数
BASE_URL = 'https://db.netkeiba.com'
RACE_SEARCH_URL = 'https://db.netkeiba.com/?pid=race_search_detail'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/raw')

# 出力ディレクトリの作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

class NetkeibaRaceScraper:
    """
    netkeiba.comからレース情報をスクレイピングするクラス
    """
    
    def __init__(self):
        """初期化"""
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
    
    def _random_sleep(self, min_seconds=1, max_seconds=3):
        """
        サーバーに負荷をかけないようにランダムな時間待機する
        
        Args:
            min_seconds (int): 最小待機時間（秒）
            max_seconds (int): 最大待機時間（秒）
        """
        time.sleep(random.uniform(min_seconds, max_seconds))
    
    def search_race_urls(self, start_year, end_year, track_types=None, save_to_file=True):
        """
        指定された年の間のレースURLを検索する
        
        Args:
            start_year (int): 開始年
            end_year (int): 終了年
            track_types (list): トラックタイプのリスト（例: ['芝', 'ダート']）
            save_to_file (bool): 結果をファイルに保存するかどうか
            
        Returns:
            list: レースURLのリスト
        """
        if track_types is None:
            track_types = ['芝', 'ダート']
            
        race_urls = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"{year}年のレースURLを検索中...")
            
            for track_type in track_types:
                logger.info(f"トラックタイプ: {track_type}")
                
                # 検索フォームのパラメータ
                params = {
                    'pid': 'race_search_detail',
                    'year': year,
                    'track_type': '芝' if track_type == '芝' else 'ダート',
                    'submit': '検索'
                }
                
                try:
                    response = self.session.post(RACE_SEARCH_URL, data=params)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    race_links = soup.select('a[href*="/race/"]')
                    
                    for link in race_links:
                        race_url = urljoin(BASE_URL, link['href'])
                        if race_url not in race_urls:
                            race_urls.append(race_url)
                    
                    logger.info(f"{year}年{track_type}のレースURL: {len(race_links)}件")
                    
                except Exception as e:
                    logger.error(f"検索中にエラーが発生しました: {e}")
                
                self._random_sleep()
        
        logger.info(f"合計レースURL: {len(race_urls)}件")
        
        if save_to_file:
            output_file = os.path.join(OUTPUT_DIR, f'race_urls_{start_year}_{end_year}.txt')
            with open(output_file, 'w', encoding='utf-8') as f:
                for url in race_urls:
                    f.write(f"{url}\n")
            logger.info(f"レースURLを {output_file} に保存しました")
        
        return race_urls
    
    def scrape_race_data(self, race_url):
        """
        レースURLからレース情報をスクレイピングする
        
        Args:
            race_url (str): レースのURL
            
        Returns:
            tuple: (race_info_dict, race_results_df)
                race_info_dict: レース情報の辞書
                race_results_df: レース結果のDataFrame
        """
        logger.info(f"レース情報をスクレイピング中: {race_url}")
        
        try:
            response = self.session.get(race_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # レース情報の抽出
            race_info = {}
            
            # レース名
            race_name_elem = soup.select_one('.data_intro h1')
            race_info['race_name'] = race_name_elem.text.strip() if race_name_elem else None
            
            # レース詳細情報
            race_details_elem = soup.select_one('.data_intro .smalltxt')
            if race_details_elem:
                race_details = race_details_elem.text.strip()
                race_info['race_details'] = race_details
                
                # 日付、開催場所、ラウンド、天気、馬場状態などを抽出
                details_parts = race_details.split()
                for part in details_parts:
                    if '年' in part and '月' in part and '日' in part:
                        race_info['race_date'] = part
                    elif '回' in part and '日' in part:
                        race_info['race_round'] = part
                    elif '天候' in part:
                        race_info['weather'] = part.replace('天候:', '')
                    elif '芝' in part or 'ダート' in part:
                        race_info['track_type'] = '芝' if '芝' in part else 'ダート'
                        distance_str = part.split(':')[-1].replace('m', '')
                        try:
                            race_info['distance'] = int(distance_str)
                        except ValueError:
                            race_info['distance'] = None
                    elif '馬場' in part:
                        race_info['track_condition'] = part.replace('馬場:', '')
            
            # レース結果テーブルの抽出
            race_table = soup.select_one('.race_table_01')
            if race_table:
                # テーブルヘッダーの取得
                headers = [th.text.strip() for th in race_table.select('thead th')]
                
                # テーブルデータの取得
                rows = []
                for tr in race_table.select('tbody tr'):
                    row = [td.text.strip() for td in tr.select('td')]
                    rows.append(row)
                
                # DataFrameの作成
                race_results_df = pd.DataFrame(rows, columns=headers)
                
                # 馬のIDを抽出
                horse_ids = []
                for tr in race_table.select('tbody tr'):
                    horse_link = tr.select_one('td:nth-child(4) a')
                    if horse_link and 'href' in horse_link.attrs:
                        href = horse_link['href']
                        horse_id = href.split('/')[-1].replace('?pid=horse_detail', '')
                        horse_ids.append(horse_id)
                    else:
                        horse_ids.append(None)
                
                race_results_df['horse_id'] = horse_ids
                
                # 騎手のIDを抽出
                jockey_ids = []
                for tr in race_table.select('tbody tr'):
                    jockey_link = tr.select_one('td:nth-child(7) a')
                    if jockey_link and 'href' in jockey_link.attrs:
                        href = jockey_link['href']
                        jockey_id = href.split('/')[-1].replace('?pid=jockey_detail', '')
                        jockey_ids.append(jockey_id)
                    else:
                        jockey_ids.append(None)
                
                race_results_df['jockey_id'] = jockey_ids
                
                # レースIDを追加
                race_id = race_url.split('/')[-1].replace('?pid=race_result', '')
                race_info['race_id'] = race_id
                race_results_df['race_id'] = race_id
                
                return race_info, race_results_df
            else:
                logger.warning(f"レース結果テーブルが見つかりませんでした: {race_url}")
                return None, None
            
        except Exception as e:
            logger.error(f"レース情報のスクレイピング中にエラーが発生しました: {e}")
            return None, None
    
    def scrape_races_from_urls(self, race_urls, output_dir=None):
        """
        レースURLのリストからレース情報をスクレイピングする
        
        Args:
            race_urls (list): レースURLのリスト
            output_dir (str): 出力ディレクトリ
            
        Returns:
            tuple: (race_info_list, race_results_df)
                race_info_list: レース情報の辞書のリスト
                race_results_df: 全レース結果を結合したDataFrame
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        race_info_list = []
        race_results_list = []
        
        for i, race_url in enumerate(race_urls):
            logger.info(f"レース {i+1}/{len(race_urls)} をスクレイピング中: {race_url}")
            
            race_info, race_results = self.scrape_race_data(race_url)
            
            if race_info and race_results is not None:
                race_info_list.append(race_info)
                race_results_list.append(race_results)
                
                # 個別のレース結果を保存
                race_id = race_info['race_id']
                race_date = race_info.get('race_date', 'unknown_date')
                
                # レース情報をJSON形式で保存
                race_info_file = os.path.join(output_dir, f'race_info_{race_id}.json')
                pd.Series(race_info).to_json(race_info_file, force_ascii=False, indent=4)
                
                # レース結果をCSV形式で保存
                race_results_file = os.path.join(output_dir, f'race_results_{race_id}.csv')
                race_results.to_csv(race_results_file, index=False, encoding='utf-8')
                
                logger.info(f"レース情報を保存しました: {race_info_file}, {race_results_file}")
            
            self._random_sleep(2, 5)  # サーバーに負荷をかけないよう長めに待機
        
        # 全レース情報を結合して保存
        if race_info_list:
            all_race_info_df = pd.DataFrame(race_info_list)
            all_race_info_file = os.path.join(output_dir, 'all_race_info.csv')
            all_race_info_df.to_csv(all_race_info_file, index=False, encoding='utf-8')
            logger.info(f"全レース情報を保存しました: {all_race_info_file}")
        
        # 全レース結果を結合して保存
        if race_results_list:
            all_race_results_df = pd.concat(race_results_list, ignore_index=True)
            all_race_results_file = os.path.join(output_dir, 'all_race_results.csv')
            all_race_results_df.to_csv(all_race_results_file, index=False, encoding='utf-8')
            logger.info(f"全レース結果を保存しました: {all_race_results_file}")
            
            return race_info_list, all_race_results_df
        
        return race_info_list, None
    
    def scrape_horse_data(self, horse_id):
        """
        馬のIDから馬の情報をスクレイピングする
        
        Args:
            horse_id (str): 馬のID
            
        Returns:
            dict: 馬の情報の辞書
        """
        horse_url = f"{BASE_URL}/horse/{horse_id}"
        logger.info(f"馬情報をスクレイピング中: {horse_url}")
        
        try:
            response = self.session.get(horse_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 馬の情報の抽出
            horse_info = {'horse_id': horse_id}
            
            # 馬名
            horse_name_elem = soup.select_one('.horse_title h1')
            horse_info['horse_name'] = horse_name_elem.text.strip() if horse_name_elem else None
            
            # 馬の基本情報
            horse_data_elem = soup.select_one('.db_prof_table')
            if horse_data_elem:
                for tr in horse_data_elem.select('tr'):
                    th = tr.select_one('th')
                    td = tr.select_one('td')
                    if th and td:
                        key = th.text.strip()
                        value = td.text.strip()
                        
                        if key == '生年月日':
                            horse_info['birth_date'] = value
                        elif key == '調教師':
                            horse_info['trainer'] = value
                        elif key == '馬主':
                            horse_info['owner'] = value
                        elif key == '生産者':
                            horse_info['breeder'] = value
                        elif key == '産地':
                            horse_info['birthplace'] = value
                        elif key == '性別':
                            horse_info['sex'] = value
                        elif key == '毛色':
                            horse_info['color'] = value
                        elif key == '父':
                            horse_info['father'] = value
                        elif key == '母':
                            horse_info['mother'] = value
                        elif key == '母父':
                            horse_info['maternal_grandfather'] = value
            
            # 競走成績
            race_history_table = soup.select_one('.db_h_race_results')
            if race_history_table:
                # テーブルヘッダーの取得
                headers = [th.text.strip() for th in race_history_table.select('thead th')]
                
                # テーブルデータの取得
                rows = []
                for tr in race_history_table.select('tbody tr'):
                    row = [td.text.strip() for td in tr.select('td')]
                    rows.append(row)
                
                # DataFrameの作成
                race_history_df = pd.DataFrame(rows, columns=headers)
                horse_info['race_history'] = race_history_df.to_dict('records')
            
            return horse_info
            
        except Exception as e:
            logger.error(f"馬情報のスクレイピング中にエラーが発生しました: {e}")
            return None
    
    def scrape_horses_from_race_results(self, race_results_df, output_dir=None):
        """
        レース結果から馬の情報をスクレイピングする
        
        Args:
            race_results_df (DataFrame): レース結果のDataFrame
            output_dir (str): 出力ディレクトリ
            
        Returns:
            list: 馬の情報の辞書のリスト
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        os.makedirs(os.path.join(output_dir, 'horses'), exist_ok=True)
        
        # 重複を除いた馬IDのリストを取得
        horse_ids = race_results_df['horse_id'].dropna().unique().tolist()
        logger.info(f"スクレイピングする馬の数: {len(horse_ids)}")
        
        horse_info_list = []
        
        for i, horse_id in enumerate(horse_ids):
            logger.info(f"馬 {i+1}/{len(horse_ids)} をスクレイピング中: {horse_id}")
            
            # 既に保存されているかチェック
            horse_file = os.path.join(output_dir, 'horses', f'horse_{horse_id}.json')
            if os.path.exists(horse_file):
                logger.info(f"馬情報は既に存在します: {horse_file}")
                try:
                    horse_info = pd.read_json(horse_file, typ='series').to_dict()
                    horse_info_list.append(horse_info)
                    continue
                except Exception as e:
                    logger.warning(f"既存の馬情報の読み込みに失敗しました: {e}")
            
            horse_info = self.scrape_horse_data(horse_id)
            
            if horse_info:
                horse_info_list.append(horse_info)
                
                # 馬情報をJSON形式で保存
                pd.Series({k: v for k, v in horse_info.items() if k != 'race_history'}).to_json(
                    horse_file, force_ascii=False, indent=4
                )
                
                # レース履歴を別途CSVで保存
                if 'race_history' in horse_info and horse_info['race_history']:
                    race_history_df = pd.DataFrame(horse_info['race_history'])
                    race_history_file = os.path.join(output_dir, 'horses', f'horse_history_{horse_id}.csv')
                    race_history_df.to_csv(race_history_file, index=False, encoding='utf-8')
                
                logger.info(f"馬情報を保存しました: {horse_file}")
            
            self._random_sleep(2, 5)  # サーバーに負荷をかけないよう長めに待機
        
        # 全馬情報を結合して保存
        if horse_info_list:
            # race_historyを除外して保存
            horse_info_for_df = [{k: v for k, v in info.items() if k != 'race_history'} 
                                for info in horse_info_list if info]
            
            all_horse_info_df = pd.DataFrame(horse_info_for_df)
            all_horse_info_file = os.path.join(output_dir, 'all_horse_info.csv')
            all_horse_info_df.to_csv(all_horse_info_file, index=False, encoding='utf-8')
            logger.info(f"全馬情報を保存しました: {all_horse_info_file}")
        
        return horse_info_list


def main():
    """メイン関数"""
    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    scraper = NetkeibaRaceScraper()
    
    # 最近の2年分のレースURLを取得
    current_year = datetime.datetime.now().year
    race_urls = scraper.search_race_urls(current_year - 2, current_year)
    
    # テスト用に最初の10件だけ処理
    test_urls = race_urls[:10]
    race_info_list, race_results_df = scraper.scrape_races_from_urls(test_urls)
    
    # レース結果から馬情報を取得
    if race_results_df is not None:
        horse_info_list = scraper.scrape_horses_from_race_results(race_results_df)
        logger.info(f"取得した馬情報: {len(horse_info_list)}件")


if __name__ == '__main__':
    main()
