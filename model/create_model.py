import argparse
import numpy as np
import os
import shutil
from pycaret.regression import *
import datetime
DIR = os.path.dirname(__file__)
BASE_DATA = os.path.join(DIR, '../data/amsterdam-airbnb-data.csv')
MODEL_PATH = os.path.join(DIR, '../results')


def base_processing(df: pd.DataFrame, train=True) -> pd.DataFrame:
    if train:
        df = df[(df.price > 50) & (df.price < 500)]

    df['cos_geo'] = np.cos(df.latitude) * np.cos(df.longitude)
    df['cossin_geo'] = np.cos(df.latitude) * np.sin(df.longitude)
    df['sin_geo'] = np.sin(df.latitude)

    return df


def _setup_train(df: pd.DataFrame) -> tuple:

    columns_drop = ['id', 'host_id', 'host_name', 'name', 'neighbourhood_group',
                    'last_review', 'reviews_per_month', "number_of_reviews",
                    "calculated_host_listings_count"]
    exp_1 = setup(data=df.drop(columns=columns_drop),
                  target='price', session_id=123,
                  normalize=True, transformation=True, transform_target=True,
                  combine_rare_levels=True, rare_level_threshold=0.05,
                  remove_multicollinearity=True, multicollinearity_threshold=0.8,
                  categorical_features=['room_type', 'neighbourhood'],
                  polynomial_features=True, trigonometry_features=True,
                  feature_interaction=True, feature_ratio=True,
                  feature_selection=True, feature_selection_method='boruta',
                  silent=True
                  )
    return exp_1


def _analyze_model(model):
    for plot in ["feature_all", "residuals", "error", "vc"]:
        name = plot_model(model, plot=plot, save=True)
        shutil.move(name, os.path.join(DIR, "../results", name))


def create_experiment(data_path:str, model_path: str) -> None:
    df = pd.read_csv(data_path)
    df = base_processing(df, train=True)
    # setup train
    exp = _setup_train(df)
    # compare models and select best to optimize
    best = create_model('lightgbm', fold=10)
    print('Tunning best model')
    tuned_best = tune_model(best, optimize='R2', n_iter=150)
    print('Generate analysis')
    _analyze_model(tuned_best)
    print("Save model")
    final_model = finalize_model(tuned_best)
    save_model(final_model,
               os.path.join(model_path, f'best_{datetime.datetime.now().strftime("%Y%m%d%H%M")}'))
    save_model(final_model,
               os.path.join(model_path, 'best_latest'))
    try:
        # Deploy a model
        deploy_model(final_model, model_name='best_latest',
                     platform='aws',
                     authentication={'bucket': 'pycaret-serverless'})
    except Exception as e:
        warnings.warn('Cannot deploy to S3, do it manually')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=BASE_DATA, help='Default data path')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Default model save path')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    create_experiment(opt.data_path, opt.model_path)