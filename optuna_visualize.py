import optuna
import os
import plotly.io as pio

pio.renderers.default = "png"

def generate_plots():
    db_path = "sqlite:///optuna_study.db"
    study_name = "fve_attention_study"
    
    print(f"Connecting to the database and loading the study: {study_name}")
    study = optuna.load_study(study_name=study_name, storage=db_path)
    
    os.makedirs("plots", exist_ok=True)
    
    fig_history = optuna.visualization.plot_optimization_history(study)
    fig_history.write_image("plots/optimization_history.png")
    print("Optimization history graph saved to plots/optimization_history.png")
    
    fig_importance = optuna.visualization.plot_param_importances(study)
    fig_importance.write_image("plots/param_importance.png")
    print("Hyperparameter importance graph stored to plots/param_importance.png")

if __name__ == "__main__":
    generate_plots()