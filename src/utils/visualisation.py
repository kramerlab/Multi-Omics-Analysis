import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from Bio import Entrez
import os

Entrez.email = os.environ.get("MAIL")

sns.set_style("whitegrid")


def save_auroc_plots(all_aucs, path, iteration, model_transitions=None):
    best_aucs = np.maximum.accumulate(all_aucs, axis=0)

    x = list(range(1, len(all_aucs) + 1))
    file_names = ("all", "best")
    for y, name in zip((all_aucs, best_aucs), file_names):
        ax = sns.lineplot(x=x, y=y)
        ax.set(
            xlabel="Iteration",
            ylabel="Auroc",
            title="Model performance vs. # of iterations",
        )
        if model_transitions is not None and model_transitions > 0:
            ax.vlines(
                x=model_transitions, ymin=min(y), ymax=max(y), linestyles="dashed"
            )
        ax.get_figure().savefig(
            str(path / f"{name}_multi-omics_iteration_{iteration}.svg")
        )
        ax.get_figure().clf()


def save_auroc_with_variance_plots(aucs_list, path, iteration, model_transitions=None):
    mean_aucs = np.mean(aucs_list, axis=0)
    std_aucs = np.std(aucs_list, axis=0)
    y_upper = [1 if i > 1 else i for i in mean_aucs + std_aucs]
    y_lower = mean_aucs - std_aucs
    best_aucs = np.maximum.accumulate(mean_aucs)
    x = list(range(1, len(mean_aucs) + 1))

    # best
    ax = sns.lineplot(x=x, y=best_aucs)
    ax.set(
        xlabel="Iteration",
        ylabel="Auroc",
        title="Model performance vs. # of iterations",
    )
    if model_transitions is not None and model_transitions > 0:
        ax.vlines(
            x=model_transitions,
            ymin=min(best_aucs),
            ymax=max(best_aucs),
            linestyles="dashed",
        )
    ax.get_figure().savefig(str(path / f"best_multi-omics_iteration_{iteration}.svg"))
    ax.get_figure().clf()

    # all
    upper_and_lower = np.concatenate([y_upper, y_lower])
    ax = sns.lineplot(x=x, y=mean_aucs)
    ax.set(
        xlabel="Iteration",
        ylabel="Auroc",
        title="Model performance vs. # of iterations",
    )
    ax.fill_between(x, y_lower, y_upper, alpha=0.2)

    if model_transitions is not None and model_transitions > 0:
        ax.vlines(
            x=model_transitions,
            ymin=min(upper_and_lower),
            ymax=max(upper_and_lower),
            linestyles="dashed",
        )
    ax.get_figure().savefig(str(path / f"all_multi-omics_iteration_{iteration}.svg"))
    ax.get_figure().clf()


# Helper method to print importances and visualize distribution
def visualize_importances(
    feature_names,
    importances,
    targets,
    predictions,
    feature_values,
    title="Average Feature Importances",
    axis_title="Features",
    number_of_most_important_features=10,
    path="",
    file_name="",
    convert_ids=False,
    number_of_expression_features=0,
    number_of_mutation_features=0,
):

    number_of_most_important_features += 1
    mean_importances = np.mean(importances, axis=0)
    sd_importances = np.std(importances, axis=0)
    sorted_indices = mean_importances.argsort()
    highest_indices = sorted_indices[-1:-number_of_most_important_features:-1].copy()
    most_important_features = feature_names[highest_indices]
    highest_importances = mean_importances[highest_indices]
    highest_importance_sd = sd_importances[highest_indices]

    absolute_sorted_indices = (np.abs(mean_importances)).argsort()
    absolute_highest_indices = absolute_sorted_indices[
        -1:-number_of_most_important_features:-1
    ].copy()
    absolute_most_important_features = feature_names[absolute_highest_indices]
    absolute_highest_importances = mean_importances[absolute_highest_indices]
    absolute_highest_importance_sd = sd_importances[absolute_highest_indices]

    if convert_ids:
        most_important_features = convert_genez_id_to_name(most_important_features)
        negative_most_important_features = convert_genez_id_to_name(
            negative_most_important_features
        )
        absolute_most_important_features = convert_genez_id_to_name(
            absolute_most_important_features
        )

    draw_swarm_attributions(
        path,
        file_name,
        absolute_most_important_features,
        importances[:, absolute_highest_indices],
        feature_values[:, absolute_highest_indices],
    )

    sum_of_rest = np.sum(
        mean_importances[sorted_indices[:-number_of_most_important_features]]
    )
    most_important_features = np.append(most_important_features, "Remaining features")
    highest_importances = np.append(highest_importances, sum_of_rest)
    highest_importance_sd = np.append(highest_importance_sd, 0)

    draw_attributions(
        title,
        axis_title,
        path,
        file_name + "_with_rest",
        most_important_features,
        highest_importances,
        highest_importance_sd,
    )

    draw_attributions(
        title,
        axis_title,
        path,
        file_name + "_absolute",
        absolute_most_important_features,
        absolute_highest_importances,
        absolute_highest_importance_sd,
    )


    plot_omics_importance(
        np.mean(np.abs(importances), axis=0),
        number_of_expression_features,
        number_of_mutation_features,
        path,
        file_name + "_omics_importance",
    )


def draw_attributions(
    title,
    axis_title,
    path,
    file_name,
    most_important_features,
    highest_importances,
    highest_importance_sd,
):
    ax = sns.barplot(x=most_important_features, y=highest_importances, color="b")
    ax.set_xlabel(axis_title)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70)

    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    plt.errorbar(x=x_coords, y=y_coords, yerr=highest_importance_sd, fmt="none", c="k")

    fig = ax.get_figure()
    fig.savefig(str(path / f"{file_name}.pdf"), bbox_inches="tight")
    fig.clf()


def draw_swarm_attributions(
    path,
    file_name,
    most_important_features,
    highest_importances,
    features_values,
):
    # create df to make it easier
    number_of_samples = len(features_values)
    multiplied_important_features = np.tile(most_important_features, number_of_samples)
    d = {
        "Feature Name": multiplied_important_features,
        "Value": features_values.flatten(),
        "Attribution": highest_importances.flatten(),
    }
    df = pd.DataFrame(d)
    ax = sns.stripplot(
        data=df,
        x="Attribution",
        y="Feature Name",
        hue="Value",
        palette="viridis",
        size=4,
    )
    ax.set_xlabel("Attribution")

    norm = plt.Normalize(df["Value"].min(), df["Value"].max())

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    ax.figure.colorbar(sm)

    fig = ax.get_figure()
    fig.savefig(str(path / f"{file_name}_swarm.pdf"), bbox_inches="tight")
    fig.clf()


def convert_genez_id_to_name(feature_names):
    names = []
    types = ids = [feature.split(" ")[0] for feature in feature_names]
    ids = [feature.split(" ")[1] for feature in feature_names]

    # Rest call to get names for ids
    request = Entrez.epost("gene", id=",".join(ids))
    result = Entrez.read(request)
    webEnv = result["WebEnv"]
    queryKey = result["QueryKey"]
    data = Entrez.esummary(db="gene", webenv=webEnv, query_key=queryKey)
    annotations = Entrez.read(data)
    for annotation in annotations.items():
        document_summary = annotation[1]["DocumentSummary"]
        for gene_data in document_summary:
            gene_name = gene_data["Name"]
            names.append(gene_name)

    # return converted features
    return [type + f" {name}" for type, name in zip(types, names)]


def plot_omics_importance(
    importances,
    number_of_expression_features,
    number_of_mutation_features,
    path,
    file_name,
):
    expression_importance = np.sum(np.abs(importances[0:number_of_expression_features]))
    mutation_importance = np.sum(
        np.abs(
            importances[
                number_of_expression_features : number_of_expression_features
                + number_of_mutation_features
            ]
        )
    )
    cna_importance = np.sum(
        np.abs(
            importances[number_of_expression_features + number_of_mutation_features :]
        )
    )
    x = [expression_importance, mutation_importance, cna_importance]
    y = ["Expression", "Mutation", "CNA"]
    ax = sns.barplot(x=x, y=y, color="b")
    plt.xticks(rotation=45)

    ax.set_ylabel("Omics")
    ax.set_xlabel("Sum Absolute Attribution")

    fig = ax.get_figure()
    fig.savefig(str(path / f"{file_name}.pdf"), bbox_inches="tight")
    fig.clf()
