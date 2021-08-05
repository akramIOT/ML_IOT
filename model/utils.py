import argparse
import os
import requests
from typing import Optional
import pandas as pd
from pandas_profiling import ProfileReport
# get dataset url
data_url = 'http://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2021-07-04/visualisations/listings.csv'

# make datapath avaliable
dirname = os.path.dirname(__file__)
datapath = os.path.join(dirname, '../data/amsterdam-airbnb-data.csv')
outpath = os.path.join(dirname, '../results/analysis.html')

def download_dataset(path: str=datapath, url: str=data_url):
    print(f"Download data from {url}")
    data = requests.get(url, allow_redirects=True)
    print(f"Save dataset into {path}")
    open(path, 'wb').write(data.content)


def create_base_analysis(datapath: str, output_path: Optional[str]=None,
                         explorative: Optional[bool]=True,
                         title: Optional[str]='Amsterdam-airbnb') -> Optional[ProfileReport]:
    df = pd.read_csv(datapath)
    profile = ProfileReport(df, title=title, explorative=explorative)
    if output_path:
        profile.to_file(output_path)
    else:
        return profile

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_data', action='store_true', help='Download default dataset')
    parser.add_argument('--data_path', type=str, default=datapath, help='Default data path')
    parser.add_argument('--create_analysis', action='store_true', help="Create Pandas profiling")
    parser.add_argument('--output_path', type=str, default=outpath, help="Path for html report from profiling")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    if opt.download_data:
        if opt.data_path != datapath:
            download_dataset(opt.data_path)
        else:
            download_dataset()
    if opt.create_analysis:
        print(outpath)
        create_base_analysis(opt.data_path, opt.output_path,
                             explorative=True)