# Imports
import numpy as np
import pandas as pd
import haversine as hs
import random

# Imports from
from haversine import Unit
from sklearn.model_selection import GroupKFold
from fuzzywuzzy import fuzz
from xgboost import XGBClassifier


# Fix all random seeds
random.seed(42)


# Custom Func's/Class's

def jaccard(list_a: list, list_b: list) -> float:
    a = set(list_a)
    b = set(list_b)
    return len(a & b) / len(a | b)


def jaccard_score(df_sub_a: pd.DataFrame, df_sub_b: pd.DataFrame) -> float:
    
    assert len(df_sub_a) == len(df_sub_b)
    df = pd.merge(df_sub_a, df_sub_b, on='id', suffixes=['_a','_b'])
    return (df.apply(lambda x: jaccard(x.matches_a.split(), x.matches_b.split()),
                     axis=1)).mean()


def get_submission_predict(df_original: pd.DataFrame, 
                           df_pairs: pd.DataFrame, 
                           labels: np.array,
                           pairs_drop_orders_dublicates = False) -> pd.DataFrame:
    
    # task: понял поздно, уже после написания, но есть проблема, нужно ее решить,
    #       переписать в единственном варианте, либо добавить возможность возвращать
    #       оба варианта, в зависимости от перданного параметра (и проверить на
    #       реальных сабмитах, какой вариант дает больший скор);
    #       проблема: если модель предсказала, что (id1, id2) и (id2, id3) дублиаты,
    #       а (id1, id3) нет, то в текущей реализации в сабмит добавятся строки 
    #       (id1, [id1, id2]), (id2, [id2, id1, id3], (id3, [id3, id2]), хотя на
    #       самом деле они либо все дубликаты, либо где то предсказание ошибочно.
    
    # task: возможно медленно работает, протестировать с прошлой версией во времени
    
    # формируем датасет из пар, для которых match/label == 1
    df_pairs = df_pairs[['id_1', 'id_2']]
    df_pairs['match'] = labels
    df_pairs = df_pairs[df_pairs.match == 1][['id_1','id_2']]
    
    # если мы оставляли пары только в одном направлении (id_1, id_2),
    # то возвращаем что бы они были в обоих (id_1, id_2) и (id_2, id_1)
    if pairs_drop_orders_dublicates:
        df_pairs = (pd.concat([df_pairs, 
                               df_pairs.rename(columns={'id_1': 'id_2', 
                                                        'id_2': 'id_1'})])
                    .drop_duplicates()) #drop_duplicates не обязателен
    
    # добавляем сапоставление  id  самому себе, т.к. этого требует выходной
    # формат
    pairs_one_to_one = pd.DataFrame({'id_1': df_pairs.id_1.unique()})
    pairs_one_to_one['id_2'] = pairs_one_to_one.id_1
    df_pairs = pd.concat([pairs_one_to_one, df_pairs])
    
    # переводим в формат id, matches, где в matches через пробел перечислены все 
    # найденные дубликаты (в том числе сам id попадет в matches)
    df_pairs = (df_pairs.groupby('id_1').id_2.agg(' '.join).to_frame().reset_index()
                .rename(columns={'id_1': 'id', 'id_2': 'matches'}))
    
    # в df_pairs остались только id, для которых найдены дубликаты, мерджим со 
    # всеми id из исходного датасета и добавляем в matchs id самого себя, для 
    # тех id, которые не попали в df_pairs (после merge у них matches == NaN)
    df_submission = pd.merge(df_original['id'], df_pairs, on='id', how='left')
    df_submission['matches'] = df_submission.matches.fillna(df_submission.id)
    
    assert len(df_submission) == len(df_original)
    
    return df_submission


def get_submission_true(df_original: pd.DataFrame) -> pd.DataFrame:
    df_original = df_original[['id', 'point_of_interest']]
    df_poi_matches = (df_original.groupby('point_of_interest').id.agg(' '.join)
                      .to_frame().reset_index().rename(columns={'id': 'matches'}))
    return pd.merge(df_original, df_poi_matches, 
                    on='point_of_interest', how='left')[['id','matches']]

def get_pairs_metrics(df_original: pd.DataFrame, 
                      df_pairs: pd.DataFrame, 
                      labels: np.array,
                      pairs_drop_orders_dublicates = False) -> dict:
    metrics = {}
    submission_true = get_submission_true(df_original)
    submission_pairs_max_true = get_submission_predict(df_original, 
                                                       df_pairs, 
                                                       labels, 
                                                       pairs_drop_orders_dublicates)
    metrics['Jaccard (max)'] = jaccard_score(submission_true, submission_pairs_max_true)
    
    return metrics

def generate_pairs_df(df_dataset: pd.DataFrame, drop_order_dub=False, get_metrics = False, real_test = False) -> (pd.DataFrame, dict):
    # Отбираем только колонки, которые планируем использовать в дальнейшем
    # task: перенести в параметры/гиперпараметры?
    
    FIRST_COLUMN_SELECTION = ['id', 'name', 'latitude', 'longitude', 'country', 'city', 'categories', 'point_of_interest']
    
    # в реальном тесте отсутсвует 'point_of_interest', удаляем из наших колонок
    if real_test:
        FIRST_COLUMN_SELECTION.remove('point_of_interest')
    
    
    df_dataset = df_dataset[FIRST_COLUMN_SELECTION]

    # task: Изучить возможности sklearn для подобных целей, скорей всего это будет наилучший вариант (примерные ключевые слова:
    # sklearn neighbors by coordinate)
    FIRST_COORD_ROUND = 3 # сотые ~= 1 км, тысячные ~= 0.1 км

    # task: Если использовать подобный подход, нужно обязательно производить с перекрытием (придумать как)
    # task: Устранить SettingWithCopyWarning

    # Первоначальный вариант (в 'lat_lon_round' мы получим строкивые представления округленных координат):
    # df_dataset.loc['lat_lon_group'] = (df_dataset.latitude.map(lambda x: str(round(x,FIRST_COORD_ROUND))) + '_' + 
    #                                    df_dataset.longitude.map(lambda x: str(round(x,FIRST_COORD_ROUND))))
    # Альтернативный вариант (результат аналогичный, за исключенем того, что в 'lat_lon_round' мы получим номера групп):
    df_dataset['lat_lon_group'] = df_dataset.groupby([df_dataset.latitude.round(FIRST_COORD_ROUND), 
                                                      df_dataset.longitude.round(FIRST_COORD_ROUND)]).ngroup()
    
    ## Формирование пар-кандидитов
    columns_to_pairs = ['lat_lon_group'] #колонки для совоставления в пары
    df_pairs = pd.merge(df_dataset, df_dataset, on=columns_to_pairs, suffixes=['_1','_2'])

    # Оставляем пары только в одном направлении (id_1, id_2) или в обоих (id_1, id_2) и (id_2, id_1)
    if drop_order_dub:
        df_pairs = df_pairs[df_pairs.id_1 < df_pairs.id_2]
    else: #удаляем только полные дубликаты (id_1, id_1)
        df_pairs = df_pairs[df_pairs.id_1 != df_pairs.id_2]
    
    
    #Generate metrics for current candidates
    metrics = {}
    if get_metrics and not real_test:
        labels = np.array(get_match_label(df_pairs))
        metrics = get_pairs_metrics(df_dataset, 
                                    df_pairs, 
                                    labels,
                                    drop_order_dub)

    return df_pairs, metrics

def add_feauture_geo_distance(df_pairs: pd.DataFrame, normalize=False, prefix='ftr_') -> pd.DataFrame:
    # Считаем расстояние в км между точками
    # task: Возможно нет смысла считать точно через haversine (с учетом шарообразности земли), 
    # можно считать более грубо, но быстрее
    
    df_pairs[f'{prefix}geo_distance'] = df_pairs.apply(lambda x: hs.haversine((x.latitude_1,x.longitude_1), 
                                                 (x.latitude_2,x.longitude_2), 
                                                 unit=Unit.KILOMETERS), axis=1)
    return df_pairs


def add_feauture_levenshtein_distance(df_pairs: pd.DataFrame, normalize=False, prefix='ftr_') -> pd.DataFrame:
    # Levenshtein distance of names
    df_pairs[f'{prefix}name_levenshtein'] = df_pairs.apply(lambda x: fuzz.token_set_ratio(x.name_1,
                                                                                          x.name_2), axis=1)
    return df_pairs


def run_futures_pipeline(dataset: pd.DataFrame, futures_pipeline: list, prefix='ftr_') -> pd.DataFrame:
    for step in futures_pipeline:
        dataset = step['func'](dataset, prefix=prefix, **step['params'])
    future_columns = [col for col in dataset.columns if col.startswith(prefix)]
    return dataset.reset_index(drop=True), future_columns


def get_match_label(dataset: pd.DataFrame) -> pd.Series: # 1 = match, 0 = not match
    #Наша целевая переменная (label/target)
    return (dataset['point_of_interest_1'] == dataset['point_of_interest_2']).astype(int)
