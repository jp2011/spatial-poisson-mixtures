import yaml
from pathlib import Path
import os
import pandas as pd


def pretty_string(input_string):
    """
    Retrieve a more user-friendly label for the given input_string
    """
    data = {
        "RMSE_IN": "RMSE (in-sample)",
        "RMSE_OUT": "RMSE (out-of-sample)",

        "intercept": "intercept",
        "log.household.count": "log households",
        "log.unemployment.count": "log unemployed",
        "log.unemployment.percent": "log unemployment rate",
        "unemployment.percent": "unemployment rate",
        "log.poi.retail.count": "log retail POIs",
        "log.poi.eatdrink.count": "log eating/drinking POIs",
        "log.poi.edu.health.count": "log edu/health POIs",
        "log.poi.accommodation.count": "log accommodation POIs",
        "log.poi.sport.entertainment.count": "log sport/entertainment POIs",
        "ethnic.heterogeneity": "ethnic heterogeneity measure",
        "logit.ethnic.heterogeneity": "ethnic heterogeneity measure",
        "log.median.age": "log median age",
        "median.age": "median age",
        "log.accessibility": "log accessibility measure",
        "accessibility": "accessibility measure",
        "log.people.moved.in": "log people moved in",
        "log.people.moved.out": "log people moved out",
        "log.people.moved.in.out": "log people moved in/out",
        "log.population.turnover": "log population turnover",
        "population.turnover": "population turnover",
        "log.all.houses.count": "log all houses",
        "log.all.houses": "log all houses",
        "all.houses": "all houses",
        "log.detached.houses.count": "log detached houses",
        "log.detached.houses": "log detached houses",
        "detached.houses": "detached houses",
        "log.semidetached.houses.count": "log semidetached houses",
        "log.semidetached.houses": "log semidetached houses",
        "semidetached.houses": "semidetached houses",
        "log.terraced.houses.count": "log terraced houses",
        "log.terraced.houses": "log terraced houses",
        "terraced.houses": "terraced houses",
        "log.detached.semidetached.houses.count": "log (semi-)detached houses",
        "log.detached.semidetached.houses": "log (semi-)detached houses",
        "detached.semidetached.houses": "(semi-)detached houses",
        "occupation.variation": "occupation variation measure",
        "logit.occupation.variation": "occupation variation measure",
        "log.tenure.owned": "log owned dwelling",
        "tenure.owned": "owned dwelling",
        "log.tenure.rented.social": "log socially-rented dwellings",
        "tenure.rented.social": "socially-rented dwellings",
        "log.tenure.rented.private": "log privately-rented dwellings",
        "tenure.rented.private": "privately-rented dwellings",
        "log.tenure.other": "log other-tenure dwellings",
        "tenure.other": "other-tenure dwellings",
        "log.single.parent.household.count": "log single-parent households",
        "log.single.parent.household": "log single-parent households",
        "single.parent.household": "single-parent households",
        "log.one.person.household.count": "log one-person household",
        "log.one.person.household": "log one-person household",
        "one.person.household": "one-person household",
        "log.couple.with.children.count": "log couple with children households",
        "log.couple.with.children": "log couple with children households",
        "couple.with.children": "couple with children households",
        "log.mean.hh.income": "log mean household income",
        "mean.hh.income": "mean household income",
        "log.house.price": "log house price",
        "house.price": "house price",
        "urban.proportion": "urbanisation index",
        "urban.suburban.proportion": "(sub)urbanisation index",

        "all.houses.fraction": "Houses (fraction of dwellings)",
        "detached.houses.fraction": "Detached houses (fraction of dwellings)",
        "semidetached.houses.fraction": "Semi-detached houses (fraction of dwellings)",
        "terraced.houses.fraction": "Terraced houses (fraction of dwellings)",
        "detached.semidetached.houses.fraction": "(Semi-)detached houses (fraction of dwellings)",
        "tenure.owned.fraction": "Owned dwelling (fraction of dwellings)",
        "tenure.rented.social.fraction": "Social housing (fraction of dwellings)",
        "tenure.rented.private.fraction": "Rented dwelling (fraction of dwellings)",
        "tenure.other.fraction": "Other tenure (fraction of dwellings)",
        "single.parent.household.fraction": "Single-parent households (fraction of households)",
        "one.person.household.fraction": "One-person households (fraction of households)",
        "couple.with.children.fraction": "Couple with children households (fraction of households)",
        "log.poi.all": "log POIs (all categories)",

        "log.all.houses.count": "Houses log-count",
        "log.detached.houses.count": "Detached houses log-count",
        "log.semidetached.houses.count": "Semi-detached houses log-count",
        "log.terraced.houses.count": "Terraced houses log-count",
        "log.detached.semidetached.houses.count": "(Semi-)detached houses log-count",
        "log.tenure.owned.count": "Owned dwelling log-count",
        "log.tenure.rented.social.count": "Social housing log-count",
        "log.tenure.rented.private.count": "Rented dwelling log-count",
        "log.tenure.other.count": "Other tenure log-count",
        "log.single.parent.household.count": "Single-parent households log-count",
        "log.one.person.household.count": "One-person households log-count",
        "log.couple.with.children.count": "Couple with children households log-count",

        "field": "field",

        "burglary_raw_0": "specification 0",
        "burglary_raw_1": "specification 1",
        "burglary_raw_2": "specification 2",
        "burglary_raw_3": "specification 3",
        "burglary_raw_4": "specification 4",

        "12013-122015": "01/2013-12/2015",
        "12015-122015": "01/2015-12/2015",
    }
    output_string = data[input_string]
    return output_string


def get_specification_table(selected_models=['burglary_raw_1', 'burglary_raw_2', 'burglary_raw_3', 'burglary_raw_4']):
    """
    Creates a concise summary of which covariates are included in the selected models. This table can easily be
    converted to LaTeX using built-in pandas command.
    """
    all_relevant_covariates = set()

    for model_name in selected_models:
        model_config_file_path = Path(os.getcwd()) / 'models' / 'config' / f"{model_name}.yml"
        with open(model_config_file_path, 'r') as stream:
            try:
                covariates_config = yaml.safe_load(stream)
                all_relevant_covariates = all_relevant_covariates | set(covariates_config['covariates'])
            except yaml.YAMLError as exc:
                print(exc)
    model_selection_df = pd.DataFrame("", index=all_relevant_covariates, columns=selected_models, dtype=str)

    for model_name in selected_models:
        model_config_file_path = Path(os.getcwd()) / 'models' / 'config' / f"{model_name}.yml"
        with open(model_config_file_path, 'r') as stream:
            try:
                covariates_config = yaml.safe_load(stream)
                covs_for_model = covariates_config['covariates']
                for cov_name in all_relevant_covariates:
                    if cov_name in covs_for_model:
                        model_selection_df.loc[cov_name, model_name] = "x"
            except yaml.YAMLError as exc:
                print(exc)  

    model_selection_df.index = [pretty_string(cov_name) for cov_name in model_selection_df.index]
    model_selection_df.columns = [model_name[-1:] for model_name in selected_models]
    return model_selection_df