import pandas as pd
import os
import numpy as np
from numerai.scoring import sharpe
import numerapi


class DataGetter:
    """ Used to get latest data from numerapi, note that it is live,
    so results might differ notebook is ran at different point in time."""

    def __init__(self):
        """NumerAPI instance"""
        self.napi = numerapi.NumerAPI(public_id=os.getenv("PUBLIC_ID"),
                                      secret_key=os.getenv("SECRET_KEY"),
                                      verbosity='info')

    def get_data(self, load_from=None, training_or_tournament='training', use_cols=None, chunksize=None):
        """ Loads data, reduces memory footprint, indexes and processes `era` column as number for analysis

        Parameters:
        -----------
        training_or_tournament: str
            Whether to get training or tournament data
            Tournament data is more than couple GB size, so only fetch it for predictions
        use_cols: list
            List of columns to read from data
            Can be used to limit data size if only relevant columns are needed for prediction

        Returns
        -------
        pandas.core.frame.DataFrame
        """
        data_path = self._get_data_path(load_from, training_or_tournament)
        print(f'Getting {training_or_tournament} data from {data_path}\n')
        self._check_if_data_downloaded(data_path)
        usecols = self._get_columns_to_use(data_path, use_cols)
        reader = pd.read_csv(data_path, usecols=usecols, chunksize=chunksize, iterator=True)
        return pd.concat(chunk.pipe(self._reduce_memory_footprint)
                         .assign(era=lambda x: x['era'].str.replace('era', ''))
                         .pipe(self._set_index)
                         for chunk in reader)

    def _get_columns_to_use(self, data_path, use_cols,
                            column_patterns=['feature', 'id', 'target', 'era']):
        if use_cols is None:
            print(f'No columns specified to select, fetching all matching pattern {column_patterns}\n')
            header = pd.read_csv(data_path, skiprows=0, nrows=0).columns.tolist()
            use_cols = [col for col in header
                        if any(pattern in col for pattern in column_patterns)]
        return use_cols

    def _check_if_data_downloaded(self, data_path):
        if not os.path.exists(data_path):
            print(f'File {data_path} does not exist, downloading data...')
            self.napi.download_current_dataset(unzip=True)

    def _get_data_path(self, load_from, training_or_tournament):
        """Downloaded data is stored with latest prediction round number suffix included in data path"""
        if load_from is not None:
            if not os.path.exists(load_from):
                print(f'{load_from} directory not found!')
        else:
            round_number = self.napi.get_competitions()[0]['number']
            load_from = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     f'numerai_dataset_{round_number}')
        return os.path.join(load_from, f'numerai_{training_or_tournament}_data.csv')

    def _reduce_memory_footprint(self, df):
        """ Since features and target all have values in [0, .25, .5, .75, 1] range,
        everything is downcasted into int8 instead of float64 to save memory."""
        print(f'Reducing memory usage by converting features and target to integers\n')
        cols = [col for col in df.columns if 'feature' in col] + ['target']
        df[cols] = df[cols].replace(self.value_map).astype('int8')
        return df

    def _set_index(self, df):
        print('Setting id and era columns as index')
        return (df.set_index(pd.MultiIndex.from_frame(df[['id', 'era']]))
                .drop(['id', 'era'], axis=1))

    @property
    def value_map(self):
        return {
            np.nan: 0,
            0: 1,
            .25: 2,
            .5: 3,
            .75: 4,
            1: 5,
        }


class PredictionSubmitter(DataGetter):

    def __init__(self, napi, model, napi_user,
                 features_to_use=None, prediction_file_name='preds.csv', chunksize=500_000):
        super().__init__(napi)
        self.model = model
        self.model_id = self.napi.get_models()[napi_user]
        self.features_to_use = (features_to_use if features_to_use is not None else
                                self._get_columns_to_use(self._get_data_path('tournament'),
                                                         features_to_use,
                                                         column_patterns=['feature']))
        self.prediction_file_name = prediction_file_name
        self.chunksize = chunksize

    def submit(self):
        use_cols = None if self.features_to_use is None else self.features_to_use + ['id', 'target', 'era']
        tournament_data = self.get_data('tournament',
                                        use_cols=use_cols,
                                        chunksize=self.chunksize)
        print('Making predictions')
        self.make_predictions(tournament_data)
        print('Uploading predictions')
        self.napi.upload_predictions(self.prediction_file_name)
        print(self.napi.submission_status(self.model_id))

    def make_predictions(self, tournament_data):
        X, y = tournament_data[self.features_to_use], tournament_data['target']
        predictions = pd.Series(self.model.predict(X), index=y.index, name='prediction')
        self.get_validation_score(y, predictions)
        self.save_predictions(predictions)

    @staticmethod
    def get_validation_score(y, predictions):
        valid_y = y[y != 0]
        valid_preds = predictions.loc[valid_y.index]
        validation_score = sharpe(valid_y, valid_preds)  # TODO: change to generic metric
        print(f'Score on validation set: {validation_score}')

    def save_predictions(self, predictions):
        print(f'Saving predictions to {self.prediction_file_name}')
        (predictions.map(self.inverse_value_map)
         .reset_index()
         .drop('era', axis=1)
         .to_csv(self.prediction_file_name, index=False))

    @property
    def inverse_value_map(self):
        return {v: k for k, v in self.value_map.items()}
