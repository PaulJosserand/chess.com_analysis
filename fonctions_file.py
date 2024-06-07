from chessdotcom import Client, get_player_game_archives
import pprint
import requests
import re
import pandas as pd
from pandasql import sqldf
import datetime
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import fonctions_file as f

# Update the User-Agent header for chessdotcom client
Client.request_config['headers'] = {
    "User-Agent": "My Python Application. Contact me at email@example.com"
}


username = 'deutschequalitat'

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Fonction pour récupérer le mois et l'année d'une url
def get_month_url(username, year_month):
    data = get_player_game_archives(username).json['archives']
    idx = 0
    while year_month != data[idx][-7:]:
        idx += 1
    url = data[idx]
    return url

#Fonction pour récupérer les parties jouées au cours d'un mois donné
def get_month_games(username, year_month):
    url = get_month_url(username, year_month)
    games = requests.get(url, headers=Client.request_config['headers']).json()
    return games["games"]

#Fonction pour récupérer les informations utiles d'une partie jouée
def get_game_info(game):
    game_info = {
        'game_id': re.search(r'\d+$', game['url']).group(),  # Extract the game ID from the URL
        'time_class': game['time_class'],
        'time_control': game['time_control'],
        'end_time': game['end_time'],
        'white_username': game['white']['username'],
        'white_result': game['white']['result'],
        'white_rating': game['white']['rating'],
        'black_username': game['black']['username'],
        'black_result': game['black']['result'],
        'black_rating': game['black']['rating']
    }
    return game_info

#Fonction pour récupérer les résultats de parties du mois considéré
def get_month_results(username, year_month):
    games = get_month_games(username, year_month)
    games_info = [get_game_info(game) for game in games]  # List of dictionaries
    df = pd.DataFrame(games_info)
    return df

#Fonction pour récupérer la liste de tous les mois où un joueur a joué une partie
def get_months_list(username):
    data = get_player_game_archives(username).json['archives']
    months_list = []
    for idx in range(len(data)):
        month = data[idx][-7:]
        months_list.append(month)
    return months_list

#Fonction pour récupérer toutes les parties d'un joueur dans un dataframe (données non propres)
def get_player_historic_brut(username):
    months_list = get_months_list(username)
    df = pd.DataFrame()
    for month in months_list:
        added_df = get_month_results(username, month)
        added_df['Year_Month'] = month
        df = pd.concat([df, added_df], ignore_index=True)
    return df

#Fonction pour renvoyer l'historique d'un joueur avec le bon format
def get_player_historic(username):
    # Load the player's game history
    df = get_player_historic_brut(username)
    # Clean and format the data
    clean_df = pd.DataFrame()
    clean_df['game_id'] = df['game_id']
    clean_df['year_month'] = df['Year_Month']
    clean_df['time_class'] = df['time_class']
    clean_df['end_time'] = df['end_time']
    for i in range(df.shape[0]):
        if username == df['white_username'].iloc[i]:
            clean_df.loc[i, 'player_username'] = df['white_username'].iloc[i]
            clean_df.loc[i, 'player_color'] = 'white'
            clean_df.loc[i, 'player_rating'] = df['white_rating'].iloc[i]
            clean_df.loc[i, 'opponent_username'] = df['black_username'].iloc[i]
            clean_df.loc[i, 'opponent_color'] = 'black'
            clean_df.loc[i, 'opponent_rating'] = df['black_rating'].iloc[i]
        else:
            clean_df.loc[i, 'player_username'] = df['black_username'].iloc[i]
            clean_df.loc[i, 'player_color'] = 'black'
            clean_df.loc[i, 'player_rating'] = df['black_rating'].iloc[i]
            clean_df.loc[i, 'opponent_username'] = df['white_username'].iloc[i]
            clean_df.loc[i, 'opponent_color'] = 'white'
            clean_df.loc[i, 'opponent_rating'] = df['white_rating'].iloc[i]
        if df['white_result'].iloc[i] == 'win':
            if clean_df.loc[i, 'player_color'] == 'white':
                clean_df.loc[i, 'player_result'] = 'win'
            else:
                clean_df.loc[i, 'player_result'] = 'lose'
        elif df['black_result'].iloc[i] == 'win':
            if clean_df.loc[i, 'player_color'] == 'black':
                clean_df.loc[i, 'player_result'] = 'win'
            else:
                clean_df.loc[i, 'player_result'] = 'lose'
        else:
            clean_df.loc[i, 'player_result'] = 'draw'
            
    #On modifie certaines colonnes en integer
    clean_df['game_id'] = clean_df['game_id'].astype('int64')
    clean_df['end_time'] = clean_df['end_time'].astype('int64')
    clean_df['player_rating'] = clean_df['player_rating'].astype(int)
    clean_df['opponent_rating'] = clean_df['opponent_rating'].astype(int) 
    
    #Permet de conserver uniquement les parties dans la catégories qui nous intéresse
    clean_df = clean_df[clean_df['time_class']== 'rapid']
    clean_df = clean_df.reset_index(names='full_histo_idx') #On met jour les index des lignes pour repartir de 0 à x 

    return clean_df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_df_losing_streak (df):
    # Identifier les séquences de défaites consécutives
    streak = (df['player_result'] == 'lose').astype(int)
    df['streak'] = streak.groupby((streak != streak.shift()).cumsum()).cumsum()

    # Trouver les indices des séries de défaites de plus de 3
    lose_streak_indices = df[df['streak'] > 3].index

    # Collecter tous les indices des séries de défaites consécutives
    lose_series_indices = set()
    for start in lose_streak_indices:
        streak_length = df.loc[start, 'streak']
        indices = range(start - streak_length + 1, start + 1)
        lose_series_indices.update(indices)

    # Trier les indices pour un affichage ordonné
    sorted_indices = sorted(lose_series_indices)

    # Filtrer les lignes correspondant aux indices triés
    output_df = df.loc[sorted_indices]
    output_df = output_df.drop(['streak'], axis=1)
    
    #On modifie certaines colonnes en integer
    output_df['game_id'] = output_df['game_id'].astype('int64')
    output_df['player_rating'] = output_df['player_rating'].astype(int)
    output_df['opponent_rating'] = output_df['opponent_rating'].astype(int) 
    
    output_df = output_df.reset_index(names='rapid_histo_idx') #On met à jour les index des lignes pour repartir de 0 à x 

    return output_df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_last_100_games_before_ref(game_id_ref, df):
    
    #On retrouve dans l'historique du joueur l'indice de la partie donnée en référence
    idx_game_ref = df[df['game_id']==game_id_ref].index[0]

    #On récupère les 100 parties qui ont précédées la partie de référence
    df_last_100_games = df.loc[max(0, idx_game_ref - 100):idx_game_ref-1].copy() #on utilise max() dans le cas ou un dataframe soit inférieur à 100 lignes

    #On enregistre le game_id de la partie qui nous a opposé
    df_last_100_games['game_id_ref']=game_id_ref
    
    return df_last_100_games


#Cette fonction permet de générer un dataframe qui regroupe les 100 parties de tous les adversaire d'un joueur en fonction du dataframe fourni
def get_df_opponents_last_100_games (player_username, df):
    
    #On cré un dataframe dans lequel on charge les 100 parties du premier adversaire
    opponent_username = df['opponent_username'].loc[0]
    game_id_reference = df['game_id'].loc[0]
    df_opponent_last_100_games = get_last_100_games_before_ref(game_id_reference, get_player_historic(opponent_username))
    
    #Pour la suite on fait la même chose mais on ajoute à la suite du dataframe les parties des autres adversaires
    for i in tqdm(range(1, df.shape[0]), desc="Generating KPI"):
        #On met à jour les variables
        opponent_username = df['opponent_username'].loc[i]
        game_id_reference = df['game_id'].loc[i]
        
        #On ajoute à la suite les données chargées
        added_df = get_last_100_games_before_ref(game_id_reference, get_player_historic(opponent_username))
        df_opponent_last_100_games = pd.concat([df_opponent_last_100_games, added_df], axis=0)
        
        #On enregistre une fois tous les 10 itérations
        if i%10==0 : df_opponent_last_100_games.to_csv(f'./dataframes_saved/df_opponent_last_100_games_{player_username}.csv', index=False)
        
    #On enregistre le dataframe
    df_opponent_last_100_games.to_csv(f'./dataframes_saved/df_opponent_last_100_games_{player_username}.csv', index=False)
    
    return df_last_100

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def get_last_10_games_score(df_last_100_games):
    
    #On récupère les 10 drnières parties
    df_last_10_games = df_last_100_games.iloc[-10:]
    
    #On calcule le score sur les 10 dernières parties
    score_map = {'win': 1, 'lose': -1, 'draw': 0.5}
    returned_score = df_last_10_games['player_result'].map(score_map).sum()
    
    return returned_score

def get_KPI(df_last_100):
    #On génère des KPI pour tous les adversaires rencontrés
    q = """
    SELECT
        game_id_ref,
        player_username,
        MAX(player_rating) AS best_rating,
        COUNT(game_id) AS nbr_of_games,
        SUM(CASE WHEN player_result = 'win' THEN 1 ELSE 0 END) AS win_rate,
        SUM(CASE WHEN player_result = 'draw' THEN 1 ELSE 0 END) AS draw_rate,
        SUM(CASE WHEN player_result = 'lose' THEN 1 ELSE 0 END) AS lose_rate
    FROM df_last_100
    GROUP BY game_id_ref
    """
    df_KPI_1 = sqldf(q)

    df_KPI_1['win_rate'] = df_KPI_1['win_rate']/df_KPI_1['nbr_of_games']
    df_KPI_1['draw_rate'] = df_KPI_1['draw_rate']/df_KPI_1['nbr_of_games']
    df_KPI_1['lose_rate'] = df_KPI_1['lose_rate']/df_KPI_1['nbr_of_games']

    #last_10_games_score
    #On récupère la liste des usernames
    list_game_id = df_last_100['game_id_ref'].unique()
    
    df_score_per_username = pd.DataFrame(columns=['game_id_ref','username', 'last_10_games_score'])

    #on recommence pour tous les autres usernames
    for game_id in list_game_id:
        #On récupère les 10 dernière parties jouées par le joueur sélectionné
        df_10_last_games = df_last_100[df_last_100['game_id_ref']==game_id].sort_values('end_time', ascending = False)[:10]
        username = df_10_last_games['player_username'].iloc[0]
        score = get_last_10_games_score(df_10_last_games)
        
        df_player_to_add = pd.DataFrame({'game_id_ref':[game_id], 'username': [username], 'last_10_games_score': [score]})
        df_score_per_username = pd.concat([df_score_per_username, df_player_to_add], axis=0)

    #Jointure des dataframes df_opponents_KPI et df_score_per_username en fonction du username
    q = """
    SELECT 
        k.game_id_ref,
        k.player_username,
        k.best_rating,
        k.win_rate,
        k.draw_rate,
        k.lose_rate,
        s.last_10_games_score
    FROM df_KPI_1 k
    LEFT JOIN df_score_per_username s 
    ON k.game_id_ref = s.game_id_ref;
    """
    df_KPI = sqldf(q)
    return df_KPI


#Fonction pour regrouper les KPI du joueur et de son adversaire sur la même ligne
def get_players_KPI_comparison (df1, df2):
    #On modifie le nom des colonnes
    df2 = df2.rename(columns={
        "player_username" : "opponent_username",
        "game_id_ref" : "opponent_game_id_reference",
        "username" : "opponent_username", 
        "best_rating" : "opponent_best_rating", 
        "last_10_games_score" : "opponent_last_10_games_score", 
        "win_rate" : "opponent_win_rate",
        "draw_rate" : "opponent_draw_rate", 
        "lose_rate" : "opponent_lose_rate" 
    })
    
    #On concatène les dataframes pour les avoir sur une même ligne
    df3 = pd.merge(df1, df2, left_on='game_id_ref', right_on='opponent_game_id_reference', how='left')
    df3 = df3.drop(['opponent_game_id_reference'], axis=1)
    
    return df3



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLOT
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def comparison_distribution_curve (df1, df2, attributStr):
    # Configuration du style des graphiques seaborn
    sns.set(style="darkgrid")

    # Calcul des statistiques
    mean_df1 = df1[attributStr].mean()
    median_df1 = df1[attributStr].median()
    q1_df1 = df1[attributStr].quantile(0.25)
    q3_df1 = df1[attributStr].quantile(0.75)

    mean_df2 = df2[attributStr].mean()
    median_df2 = df2[attributStr].median()
    q1_df2= df2[attributStr].quantile(0.25)
    q3_df2 = df2[attributStr].quantile(0.75)


    # Tracer la courbe de distribution avec un histogramme plus détaillé
    plt.figure(figsize=(10, 6))

    sns.histplot(df1[attributStr], kde=True, bins=20, color='blue', stat='density', kde_kws={'bw_adjust': 2}, label='df1') #kde_kws permet de lisser plus ou moins la courbe
    sns.histplot(df2[attributStr], kde=True, bins=20, color='red', stat='density', kde_kws={'bw_adjust': 2}, label='df2') #kde_kws permet de lisser plus ou moins la courbe

    # Ajouter des lignes verticales pour les statistiques
    plt.axvline(median_df1, color='cyan', linestyle='-', linewidth=2, label=f'Mediane de df1: {median_df1:.2f}')
    plt.axvline(median_df2, color='pink', linestyle='-', linewidth=2, label=f'Mediane de df2: {median_df2:.2f}')

    # Ajouter des labels et un titre
    plt.xlabel(attributStr)
    plt.ylabel('Densité')
    plt.title(f'Courbe de dictribution de {attributStr}')

    # Ajouter une légende
    plt.legend()

    # Afficher le graphique
    plt.show()
    
    print(df1[attributStr].describe(), df2[attributStr].describe())

def distribution_curve(df, attributStr):
    # Configuration du style des graphiques seaborn
    sns.set(style="darkgrid")

    # Calcul des statistiques
    mean = df[attributStr].mean()
    median = df[attributStr].median()
    q1 = df[attributStr].quantile(0.25)
    q3 = df[attributStr].quantile(0.75)

    # Tracer la courbe de distribution avec un histogramme plus détaillé
    plt.figure(figsize=(10, 6))
    
    sns.histplot(df[attributStr], kde=True, bins=20, color='blue', stat='density', kde_kws={'bw_adjust': 3}) #kde_kws permet de lisser plus ou moins la courbe

    # Ajouter des lignes verticales pour les statistiques
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='-', linewidth=2, label=f'Median: {median:.2f}')
    plt.axvline(q1, color='purple', linestyle='-.', linewidth=2, label=f'Q1: {q1:.2f}')
    plt.axvline(q3, color='orange', linestyle='-.', linewidth=2, label=f'Q3: {q3:.2f}')

    # Ajouter des labels et un titre
    plt.xlabel(attributStr)
    plt.ylabel('Density')
    plt.title(f'Distribution Curve of {attributStr}')

    # Ajouter une légende
    plt.legend()

    # Afficher le graphique
    plt.show()
    
    
# Fonction de conversion d'un timestamp en date
def convert_timestamp_to_date(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')