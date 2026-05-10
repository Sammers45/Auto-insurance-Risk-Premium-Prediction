import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_preprocessing import (
    create_data_if_missing,
    load_data,
    prepare_features,
    split_data,
)
from src.model_training import (
    train_classification_models,
    train_regression_models,
)
from src.evaluation import (
    evaluate_classification,
    evaluate_regression,
)


def save_chart(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / 'data' / 'insurance_data.csv'
    source_path = base_dir.parent / 'insurance.csv'
    output_dir = base_dir / 'outputs'
    charts_dir = output_dir / 'charts'
    charts_dir.mkdir(parents=True, exist_ok=True)

    print('Loading or generating dataset...')
    df = create_data_if_missing(source_path, data_path)
    print(f'Dataset loaded: {len(df)} rows')

    print('Preparing features...')
    X, y_class, y_reg = prepare_features(df)

    X_train_cl, X_test_cl, y_train_cl, y_test_cl = split_data(X, y_class)
    X_train_rg, X_test_rg, y_train_rg, y_test_rg = split_data(X, y_reg)

    print('Training classification models...')
    classification_models = train_classification_models(X_train_cl, y_train_cl)

    print('Training regression models...')
    regression_models = train_regression_models(X_train_rg, y_train_rg)

    print('Evaluating classification models...')
    classification_results = {}
    for name, model in classification_models.items():
        metrics, report = evaluate_classification(model, X_test_cl, y_test_cl)
        classification_results[name] = {
            'metrics': metrics,
            'report': report,
        }

    print('Evaluating regression models...')
    regression_results = {}
    for name, model in regression_models.items():
        metrics = evaluate_regression(model, X_test_rg, y_test_rg)
        regression_results[name] = metrics

    # Save reports to outputs
    report_path = output_dir / 'classification_report.txt'
    with report_path.open('w', encoding='utf-8') as report_file:
        for name, result in classification_results.items():
            report_file.write(f'Model: {name}\n')
            report_file.write('Metrics:\n')
            for key, value in result['metrics'].items():
                report_file.write(f'  {key}: {value:.4f}\n')
            report_file.write('\n')
            report_file.write('Classification Report:\n')
            report_file.write(result['report'])
            report_file.write('\n' + '=' * 60 + '\n\n')

    regression_path = output_dir / 'regression_results.txt'
    with regression_path.open('w', encoding='utf-8') as reg_file:
        for name, metrics in regression_results.items():
            reg_file.write(f'Model: {name}\n')
            reg_file.write('Metrics:\n')
            reg_file.write(f"  MAE: {metrics['mae']:.2f}\n")
            reg_file.write(f"  RMSE: {metrics['rmse']:.2f}\n")
            reg_file.write(f"  R2 score: {metrics['r2_score']:.4f}\n")
            reg_file.write('\n' + '=' * 50 + '\n\n')

    print('Creating charts...')
    # Claim risk distribution
    fig = plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='claim_risk')
    plt.title('Claim Risk Distribution')
    plt.xlabel('High Risk (1) vs Low Risk (0)')
    plt.ylabel('Count')
    save_chart(fig, charts_dir / 'claim_risk_distribution.png')

    # Premium distribution
    fig = plt.figure(figsize=(6, 4))
    sns.histplot(df['premium'], kde=True, color='steelblue')
    plt.title('Predicted Insurance Premium Distribution')
    plt.xlabel('Premium')
    save_chart(fig, charts_dir / 'premium_distribution.png')

    # Feature relationship visuals
    fig = plt.figure(figsize=(8, 5))
    sns.boxplot(x='smoker', y='premium', data=df)
    plt.title('Premium by Smoker Status')
    plt.xlabel('Smoker (0=no, 1=yes)')
    plt.ylabel('Premium')
    save_chart(fig, charts_dir / 'premium_by_smoker.png')

    fig = plt.figure(figsize=(8, 5))
    sns.scatterplot(x='vehicle_age', y='premium', hue='claim_risk', data=df, palette='Set1')
    plt.title('Premium vs Vehicle Age with Claim Risk')
    plt.xlabel('Vehicle Age')
    plt.ylabel('Premium')
    save_chart(fig, charts_dir / 'premium_vehicle_age.png')

    print('Results saved in outputs/')
    print('Classification and regression reports are ready.')

    return classification_results, regression_results


if __name__ == '__main__':
    main()
