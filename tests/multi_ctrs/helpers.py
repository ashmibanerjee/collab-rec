import sys, os
from dotenv import load_dotenv

load_dotenv()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json, ast
from collections import Counter
import shutil

data_dir = "/".join(os.getcwd().split("/")[:-2]) + "/data/conv-trs"


def load_queries():
    samples_df = pd.read_csv(f"{data_dir}/multi-agent/sample-data/llama3point2_sample.csv")
    samples_df["difficulty"] = samples_df.config_id.apply(lambda x: x.split("_")[-1])
    samples_df["popularity"] = samples_df.config_id.apply(lambda x: x.split("_")[-2])
    samples_df["filters"] = samples_df["config"].apply(lambda x: ast.literal_eval(x)["filters"])

    return samples_df


def load_kb():
    listings_df = pd.read_csv(f"{data_dir}/kg-generation/new-kg/data/merged_listing.csv")

    return listings_df


def match_city_with_filters(listings_df, city, filters):
    city_df = listings_df[listings_df['city'] == city].drop_duplicates()

    filter_matches = {}
    for key, val in filters.items():
        if "month" in key:
            filter_matches[key] = val

        elif "seasonality" in key:
            low_season_months = city_df['low_season'].notna()
            if len(low_season_months):
                filter_matches[key] = val

        else:
            try:
                if val.casefold() in (item.casefold() for item in city_df[key].values.tolist()):
                    filter_matches[key] = val
            except Exception as e:
                # print(city_df[key].notna().values.tolist())
                pass
    return filter_matches


def get_agent_relevance_score(offers, retrieved_cities):
    """
    Computes agent hit rate relevance score
    """
    tp = len([city for city in offers if city in retrieved_cities])
    if offers:
        return tp / len(offers)
    else:
        return 0


def get_city_relevance_score(response, filters, listings_df, sasi=False):
    """
    Computes proportional relevance for city w.r.t filters
    """
    if sasi:
        cities = response
    else:
        cities = response["collective"]["offers"]
    for city in cities:
        matched_filters = match_city_with_filters(city=city, filters=filters, listings_df=listings_df)
        unmatched = [key for key in filters.keys() if key not in matched_filters.keys()]
        rel_score = len(matched_filters) / len(filters)

    return rel_score


def get_offer(df, difficulty=None, popularity=None, method=None, model="gemini2flash", rounds=10):
    if method:
        dir = f'{data_dir}/multi-agent/results/{method}/{model}'
    else:
        dir = f'{data_dir}/multi-agent/results/prototype/{model}'

    if difficulty:
        df = df[df.difficulty == difficulty]
    if popularity:
        df = df[df.popularity == popularity]

    offers = []
    for sample in df.config_id.to_list():
        row = {}
        row["config_id"] = sample
        try:
            with open(f'{dir}/states/states_{sample}.json') as f:
                print(sample)
                d = json.load(f)
                row["first_offer"] = d["0"]["collective"]["offers"]
                row["final_offer"] = d[str(rounds - 1)]["collective"]["offers"]

                offers.append(row)

        except Exception as e:
            print(f"Result file for {sample} not found")

    return offers


def get_offer_proportion(df, difficulty=None, popularity=None, method=None, model="gemini2flash", rounds=10):
    if method:
        dir = f'{data_dir}/multi-agent/results/{method}/{model}'
    else:
        dir = f'{data_dir}/multi-agent/results/prototype/{model}'

    if difficulty:
        df = df[df.difficulty == difficulty]
    if popularity:
        df = df[df.popularity == popularity]

    offers = []
    for sample in df.config_id.to_list():
        try:
            with open(f'{dir}/states/states_{sample}.json') as f:
                print(sample)
                d = json.load(f)
                for key, val in d.items():
                    row = {}
                    row["config_id"] = sample
                    row["round"] = key
                    coll_offer = d[key]["collective"]["offers"]
                    for role in ["popularity", "constraint", "sustainability"]:
                        row[role] = len([item for item in d[key][role]["offers"] if item in coll_offer])
                    offers.append(row)

        except Exception as e:
            print(f"Result file for {sample} not found")
    return offers


# def get_offer_proportion_origin(df, difficulty=None, popularity=None, method=None, model="gemini2flash", rounds=10):
#     if method:
#         dir = f'{data_dir}/multi-agent/results/{method}/{model}'
#     else:
#         dir = f'{data_dir}/multi-agent/results/prototype/{model}'

#     if difficulty:
#         df = df[df.difficulty==difficulty]
#     if popularity:
#         df = df[df.popularity==popularity]

#     offers = []
#     cities = {}
#     for sample in df.config_id.to_list():
#         try:
#             with open(f'{dir}/states/states_{sample}.json') as f:
#                 print(sample)
#                 d = json.load(f)
#                 for key, val in d.items():
#                     row = {}
#                     row["config_id"] = sample
#                     row["round"] = key
#                     coll_offer = d[key]["collective"]["offers"]
#                     for offer in coll_offer:
#                         if not cities.get(offer):
#                             for role in ["popularity", "constraint", "sustainability"]:
#                             cities[offer] =
#                     for role in ["popularity", "constraint", "sustainability"]:
#                         row[role] = len([item for item in d[key][role]["offers"] if item in coll_offer])
#                     offers.append(row)

#         except Exception as e:
#             print(f"Result file for {sample} not found")
#     return offers

def get_matched_proportion(df, difficulty=None, popularity=None, method=None, model="gemini2flash", rounds=10,
                           agent="collective", filter_focus="collective"):
    source_df = load_kb()
    if method:
        dir = f'{data_dir}/multi-agent/results/{method}/{model}'
    else:
        dir = f'{data_dir}/multi-agent/results/prototype/{model}'

    if difficulty:
        df = df[df.difficulty == difficulty]
    if popularity:
        df = df[df.popularity == popularity]

    role_filter = {"popularity": ["popularity"],
                   "constraint": ["budget", "month", "interests"],
                   "sustainability": ["aqi", "walkability", "seasonality"]}
    matches = []
    for sample in df.config_id.to_list():
        filter = df[df.config_id == sample]["filters"].values[0]
        filter_focus = {k: v for k, v in filter.items() if k in role_filter[filter_focus]}
        try:
            with open(f'{dir}/states/states_{sample}.json') as f:
                print(sample)
                d = json.load(f)
                for key, val in d.items():
                    row = {}
                    row["config_id"] = sample
                    row["round"] = key
                    offers = val[agent]["offers"]
                    for role in ["popularity", "constraint", "sustainability"]:
                        agent_filter = {k: v for k, v in filter.items() if k in role_filter[role]}
                        filter_matches = [match_city_with_filters(source_df, city, agent_filter).keys() for city in
                                          coll_offers]
                        matched = [len(matches) == len(agent_filter) for matches in filter_matches]
                        row[role] = np.sum(matched)
                    matches.append(row)

        except Exception as e:
            print(f"Result file for {sample} not found")
    return matches


def get_origin_proportion(df, difficulty=None, popularity=None, method=None, model="gemini2flash", rounds=10):
    source_df = load_kb()
    if method:
        dir = f'{data_dir}/multi-agent/results/{method}/{model}'
    else:
        dir = f'{data_dir}/multi-agent/results/prototype/{model}'

    if difficulty:
        df = df[df.difficulty == difficulty]
    if popularity:
        df = df[df.popularity == popularity]
    origins = []
    for sample in df.config_id.to_list():
        filter = df[df.config_id == sample]["filters"].values[0]
        try:
            with open(f'{dir}/states/states_{sample}.json') as f:
                print(sample)
                cities = {}
                d = json.load(f)
                for key, val in d.items():
                    row = {}
                    row["config_id"] = sample
                    row["round"] = key
                    row["coll_offers"] = val["collective"]["offers"]
                    for city in row["coll_offers"]:
                        origin = []
                        if not cities.get(city):
                            for role in ["popularity", "constraint", "sustainability"]:
                                if city in val[role]["offers"]:
                                    origin.append(role)
                            cities[city] = origin
                    row["cities"] = cities.copy()
                    origins.append(row)
        except Exception as e:
            print(f"Result file for {sample} not found")
    df = pd.DataFrame(origins)
    df["filters"] = df.apply(lambda row: [row["cities"][city] for city in row["coll_offers"]], axis=1)
    expanded = df["filters"].apply(lambda x: Counter([item for sublist in x for item in sublist])).apply(pd.Series)
    df_expanded = pd.concat([df, expanded], axis=1)

    return df_expanded


def get_scores(df, difficulty=None, popularity=None, folder_name=None, model="gemini2flash", hit_rate=False,
               sustainability=False, scoring_method="prop"):
    """
    get halucination, relevance, and reliability score
    """
    source_df = load_kb()
    if folder_name:
        dir = f'{data_dir}/multi-agent/results/{folder_name}/{model}'
    else:
        dir = f'{data_dir}/multi-agent/results/prototype/{model}'

    if difficulty:
        df = df[df.difficulty == difficulty]
    if popularity:
        df = df[df.popularity == popularity]

    hal = {"popularity": [],
           "constraint": [],
           "sustainability": []}
    rely = {"popularity": [],
            "constraint": [],
            "sustainability": []}
    rev = {"popularity": [],
           "constraint": [],
           "sustainability": [],
           "collective": []}
    for sample in df.config_id.to_list():
        filter = df[df.config_id == sample]["filters"].values[0]
        if sustainability:
            filter = {"seasonality": "low", "walkability": "great", "aqi": "great"}
        try:
            with open(f'{dir}/states/states_{sample}.json') as f:
                # print(sample)
                d = json.load(f)
                for role in ["popularity", "constraint", "sustainability"]:
                    hal[role].append([val[role]["hallucination_rate"] for val in d.values()])
                    rely[role].append([val[role]["reliability_score"] for val in d.values()])
                    if (scoring_method == "prop") and hit_rate:
                        rev[role].append([get_agent_relevance_score(val[role]["offers"], ast.literal_eval(
                            df[df.config_id == sample]["city"].values[0])) for val in d.values()])

                    else:
                        rev[role].append([val[role]["relevance_score"] for val in d.values()])

                if (scoring_method == "prop") and hit_rate:
                    rev["collective"].append([get_agent_relevance_score(val["collective"]["offers"], ast.literal_eval(
                        df[df.config_id == sample]["city"].values[0])) for val in d.values()])
                else:
                    rev["collective"].append([get_city_relevance_score(val, filter, source_df) for val in d.values()])

        except Exception as e:
            print(f"Result file for {sample} not found")

    return hal, rely, rev


def get_collective_distribution(df, model, folder_name, mami_round=9, popularity_level=None):
    listings_df = load_kb()
    data_dir = "/".join(os.getcwd().split("/")[:-2]) + "/data/conv-trs/"
    results_dir = data_dir + "multi-agent/results/"
    print(folder_name)
    hal, rely, rev = get_scores(df, model=str.lower(model), folder_name=folder_name, hit_rate=False,
                                scoring_method="prop", popularity=popularity_level)

    mami = np.array(rev["collective"])[:, mami_round]
    masi = np.array(rev["collective"])[:, 0]

    if model.lower() == "gemini2point5flash":
        sasi = pd.read_json(f"{results_dir}Gemini2Point5Flash_sasi_results.json")
        sasi['filters'] = sasi.config.apply(lambda x: ast.literal_eval(x)['filters'])
        sasi_scores = sasi.apply(
            lambda row: get_city_relevance_score(row[f"response_{model}"], row["filters"], listings_df, sasi=True) if
            row[f"response_{model}"] else None, axis=1).to_list()
    else:

        sasi = pd.read_csv(f"{results_dir}sasi_results.csv")
        sasi['filters'] = sasi.config.apply(lambda x: ast.literal_eval(x)['filters'])
        sasi_scores = sasi.apply(
            lambda row: get_city_relevance_score(row[f"response_{model}"], row["filters"], listings_df, sasi=True) if
            row[f"response_{model}"] else None, axis=1).to_list()
    return sasi_scores, masi, mami


def plot_collective_distribution(sasi, masi, mami, method, rejection_strategy, model, popularity_level=None, kde=True):
    # Plot KDEs
    plt.figure(figsize=(22, 12))
    if kde:
        sns.kdeplot(sasi, label='SASI', linewidth=5, fill=True)
        sns.kdeplot(masi, label='MASI', linewidth=5, fill=True)
        sns.kdeplot(mami, label='MAMI', linewidth=5, fill=True)
    else:
        sns.histplot(sasi, label='SASI', element='step', stat='density', common_norm=False)
        sns.histplot(masi, label='MASI', element='step', stat='density', common_norm=False)
        sns.histplot(mami, label='MAMI', element='step', stat='density', common_norm=False)

    # Add legend and show
    plt.legend()
    # plt.title(f'KDE of Moderator Success for {method} {rejection_strategy} with {model}')
    plt.xlabel('Moderator Success')
    plt.ylabel('Density')
    # plt.show()
    if popularity_level is None:
        popularity_level = "all"
    file_to_save = f"{model}_{rejection_strategy}_{popularity_level}".replace(" ", "_")
    save_plots(file_name=file_to_save, subfolder="kde",
               extensions=["pdf", "png"],
               copy_to_paper=True,
               paper_location=os.getenv("PAPER_LOCATION"))


def plot_and_annotate(ax, x_vals, y_vals, color, label):
    ax.plot(x_vals, y_vals, color=color, marker='o', label=label)
    for x, y in zip(x_vals, y_vals):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 20), ha='center', fontsize=20)


def plot_result(metrics_json, metrics_name, folder_name, model, pop_level, rounds=5, collective=True):
    # Plot of Scores
    fig, ax = plt.subplots(figsize=(12, 8))

    x_vals = [i for i in range(rounds)]
    # plot_and_annotate(ax, x_vals, np.mean(metrics_json["popularity"], axis=0), 'blue', 'popularity')
    # plot_and_annotate(ax, x_vals, np.mean(metrics_json["constraint"], axis=0), 'green', 'constraint')
    # plot_and_annotate(ax, x_vals, np.mean(metrics_json["sustainability"], axis=0), 'orange', 'sustainability')

    ax.plot(x_vals, np.mean(metrics_json["popularity"], axis=0), color='blue', marker='o', label='Popularity')
    ax.plot(x_vals, np.mean(metrics_json["constraint"], axis=0), color='green', marker='o', label='Personalization')
    ax.plot(x_vals, np.mean(metrics_json["sustainability"], axis=0), color='orange', marker='o', label='Sustainability')

    if collective:
        ax.plot(x_vals, np.mean(metrics_json["collective"], axis=0), color='purple', marker='o',
                label='Collective Offer')
        # plot_and_annotate(ax, x_vals, np.mean(metrics_json["collective"], axis=0), 'purple', 'collective')

    # ax.set_title(f"Average {metrics_name} score using {method} method with {model} for {difficulty} queries")
    ax.set_xlabel("Rounds")
    ax.set_ylabel(f"{metrics_name.title()} Score")

    ax.set_xticks(x_vals)  # Ensure only integer ticks on the x-axis

    ax.legend(loc="best", title="Agents")
    plt.tight_layout()
    # plt.show()
    if pop_level is None:
        pop_level = "all"
    file_to_save = f"{model}_{metrics_name}_{folder_name}_{pop_level}".replace(" ", "_")
    save_plots(file_name=file_to_save, subfolder=metrics_name.replace(" ", "_"),
               extensions=["pdf", "png"],
               copy_to_paper=True,
               paper_location=os.getenv("PAPER_LOCATION"))


def plot_together(metrics_json1, metrics_json2, metrics_name, folder_name, pop_level, rounds=5, collective=True):
    # Plot of Scores
    fig, ax = plt.subplots(figsize=(12, 8))

    x_vals = [i for i in range(rounds)]
    # plot_and_annotate(ax, x_vals, np.mean(metrics_json["popularity"], axis=0), 'blue', 'popularity')
    # plot_and_annotate(ax, x_vals, np.mean(metrics_json["constraint"], axis=0), 'green', 'constraint')
    # plot_and_annotate(ax, x_vals, np.mean(metrics_json["sustainability"], axis=0), 'orange', 'sustainability')
    # Plot continuous lines and store their handles
    line_popularity, = ax.plot(x_vals, np.mean(metrics_json1["popularity"], axis=0),
                               color='blue', marker='o')
    line_personalization, = ax.plot(x_vals, np.mean(metrics_json1["constraint"], axis=0),
                                    color='green', marker='o')
    line_sustainability, = ax.plot(x_vals, np.mean(metrics_json1["sustainability"], axis=0),
                                   color='orange', marker='o')

    ax.plot(x_vals, np.mean(metrics_json2["popularity"], axis=0),
            color='blue', marker='o',
            linestyle='dashed', linewidth=3)
    ax.plot(x_vals, np.mean(metrics_json2["constraint"], axis=0),
            color='green', marker='o', label='_nolegend_',
            linestyle='dashed', linewidth=3)
    ax.plot(x_vals, np.mean(metrics_json2["sustainability"], axis=0),
            color='orange', marker='o',
            label='_nolegend_',
            linestyle='dashed', linewidth=3)

    if collective:
        line_collective, = ax.plot(x_vals, np.mean(metrics_json1["collective"], axis=0),
                                   color='purple', marker='o', label='Collective Offer')
        ax.plot(x_vals, np.mean(metrics_json2["collective"], axis=0),
                color='purple', marker='o',
                label='Collective Offer',
                linestyle='dashed', linewidth=3)
        # plot_and_annotate(ax, x_vals, np.mean(metrics_json["collective"], axis=0), 'purple', 'collective')

    # ax.set_title(f"Average {metrics_name} score using {method} method with {model} for {difficulty} queries")
    ax.set_xlabel("Rounds")
    ax.set_ylabel(f"{metrics_name.title()} Score")

    ax.set_xticks(x_vals)  # Ensure only integer ticks on the x-axis
    if pop_level == "medium" or pop_level is None:
        if "aggressive" in folder_name:
            handles = [line_popularity, line_personalization, line_sustainability]
            labels = ['Popularity', 'Personalization', 'Sustainability']
            if collective:
                handles.append(line_collective)
                labels.append('Collective Offer')
            ax.legend(handles, labels, loc="upper right", title="Agents", title_fontsize=24)
    plt.tight_layout()
    # plt.show()
    if pop_level is None:
        pop_level = "all"
    file_to_save = f"{metrics_name}_{folder_name}_{pop_level}".replace(" ", "_")
    save_plots(file_name=file_to_save, subfolder=metrics_name.replace(" ", "_"),
               extensions=["pdf", "png"],
               copy_to_paper=True,
               paper_location=os.getenv("PAPER_LOCATION"))


def save_plots(file_name, subfolder=None,
               extensions=None, copy_to_paper=False,
               paper_location=None):
    if extensions is None or "pdf" not in extensions:
        extensions = ['pdf', 'png']
    for extension in extensions:
        file_name_new = file_name + '.' + extension
        print("new file name: ", file_name_new)
        if subfolder is None:
            src_file_name = str(os.getcwd()) + '/../../plots/' + extension + '/' + file_name_new
        else:
            src_file_name = str(os.getcwd()) + '/../../plots/' + extension + '/' + subfolder + '/' + file_name_new
        print(src_file_name)
        plt.tight_layout()
        plt.savefig(src_file_name, bbox_inches='tight')

        if extension == "pdf" and copy_to_paper:
            print("copying to paper location: ", paper_location)
            shutil.copy(src_file_name, paper_location + '/plots/' + subfolder + '/')


def get_early_stopping_round(config_id, state, filters, listings, threshold):
    # get initial round
    for round_init in range(5):
        round_init_score = get_city_relevance_score(
            response=state[str(round_init)],
            filters=filters,
            listings_df=listings
        )
        if round_init_score > 0:
            break

    max_rounds = ast.literal_eval(list(state.keys())[-1])

    early_stopping = False

    for round in range(5, max_rounds + 1):
        round_score = get_city_relevance_score(
            response=state[str(round)],
            filters=filters,
            listings_df=listings
        )

        improvement = max(0, (round_score - round_init_score) / round_init_score)

        if improvement > threshold or round_score == 1:
            early_stopping = True
            max_improvement = {
                'config_id': config_id,
                'round_init': round_init,
                'round_stop': round,
                'collective_offer': state[str(round)]['collective']['offers'],
                'success_score': round_score,
                'improvement': improvement
            }
            break

    if early_stopping:
        return max_improvement
    else:
        return {
            'config_id': config_id,
            'round_init': round_init,
            'round_stop': max_rounds,
            'collective_offer': state[str(max_rounds)]['collective']['offers'],
            'success_score': round_score,
            'improvement': 0
        }


def set_paper_style():
    sns.set_theme(style="white")
    # plt.subplots(figsize=(22, 12))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 24
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['xtick.labelsize'] = 28
    plt.rcParams['ytick.labelsize'] = 28
    plt.rcParams['legend.fontsize'] = 26
    plt.rcParams['figure.titlesize'] = 28
    plt.rcParams['lines.linewidth'] = 5.0
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True