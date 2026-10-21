"""Part 2, Task D: train a model on the ETL output and save it.

Command line:
    python -m quakes.train --task regression --model linear
    python -m quakes.train --task regression --model poly --degree 3 --alpha 10
    python -m quakes.train --task classification --model logistic [--balanced]

Steps (see the lab handout):
 1. load the table from the SQLite database written by quakes.etl
 2. X = everything except 'mag' and 'big_quake'; y = 'mag' or 'big_quake'
 3. train/test split (stratified for classification, seeded with config.SEED)
 4. fit the dummy baseline AND your model; report both on the test set
 5. cross-validate on the TRAIN set only
 6. classification only: pick the threshold from out-of-fold training
    predictions (cross_val_predict, method='predict_proba'), then apply it to test
 7. save the pipeline to models/ and the report (JSON) to reports/
"""


def load_data(db_path=None):
    ...


def main() -> None:
    ...


if __name__ == "__main__":
    main()

