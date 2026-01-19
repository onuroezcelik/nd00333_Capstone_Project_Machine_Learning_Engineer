from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Dataset

def clean_data(data):

    x_df = data.to_pandas_dataframe().dropna()
    y_df = x_df.pop("DEATH_EVENT")

    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", float(args.C))
    run.log("Max iterations:", int(args.max_iter))

    subscription_id = '6971f5ac-8af1-446e-8034-05acea24681f'
    resource_group = 'aml-quickstarts-294022'
    workspace_name = 'quick-starts-ws-294022'

    workspace = Workspace(subscription_id, resource_group, workspace_name)

    ds = Dataset.get_by_name(workspace, name='heart_failure_clinical_records')
    
    x, y = clean_data(ds) 

    # TODO: Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    acc = model.score(x_test, y_test)
    run.log("Accuracy", float(acc))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model,'outputs/model.joblib')

if __name__ == '__main__':
    main()
