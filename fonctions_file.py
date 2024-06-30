from chessdotcom import Client, get_player_game_archives, get_player_current_games_to_move
from datetime import datetime, timedelta
import fonctions_file as f
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandasql import sqldf
import pprint
import requests
import re
from tqdm.notebook import tqdm
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings

# Update the User-Agent header for chessdotcom client
Client.request_config['headers'] = {
    "User-Agent": "My Python Application"
}

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------FONCTIONS PLOT
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
    plt.title(f'Courbe de distribution de {attributStr}')

    # Ajouter une légende
    plt.legend()

    # Afficher le graphique
    plt.show()

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



#------------------------------------------------------------------------------------------------------------------------------------
#--------------------IMPORTER HISTORQUE COMPLET D UN JOUEUR
#------------------------------------------------------------------------------------------------------------------------------------

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
    clean_df['moves'] = df['moves']
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

#Fonction pour récupérer toutes les parties d'un joueur dans un dataframe (données non propres)
def get_player_historic_brut(username):
    months_list = get_months_list(username)
    df = pd.DataFrame()
    for month in months_list:
        added_df = get_month_results(username, month)
        added_df['Year_Month'] = month
        df = pd.concat([df, added_df], ignore_index=True)
    return df


#Fonction pour récupérer la liste de tous les mois où un joueur a joué une partie
def get_months_list(username):
    data = get_player_game_archives(username).json['archives']
    months_list = []
    for idx in range(len(data)):
        month = data[idx][-7:]
        months_list.append(month)
    return months_list

#Fonction pour récupérer les résultats de parties du mois considéré
def get_month_results(username, year_month):
    games = get_month_games(username, year_month)
    games_info = [get_game_info(game) for game in games]  # List of dictionaries
    df = pd.DataFrame(games_info)
    return df

#Fonction pour récupérer les parties jouées au cours d'un mois donné
def get_month_games(username, year_month):
    url = get_month_url(username, year_month)
    games = requests.get(url, headers=Client.request_config['headers']).json()
    return games["games"]

#Fonction pour récupérer les informations utiles d'une partie jouée
def get_game_info(game):
    #Get moves of the game
    moves = get_moves_of_a_game(game)
    
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
        'black_rating': game['black']['rating'],
        'moves' : moves
    }
    return game_info


#Renvoyer les mouvements d'une parties
def get_moves_of_a_game(game):
    subStr = '"]\n\n'
    moves = game['pgn'][find_index_substr_in_str(game['pgn'],subStr) + len(subStr):]   
    return moves


#Fonction pour retrouver l'index d'une sous chaine dans une chaine de caractères
def find_index_substr_in_str(inputStr, subStr):
    index = inputStr.find(subStr)
    return index

#Fonction pour récupérer le mois et l'année d'une url
def get_month_url(username, year_month):
    data = get_player_game_archives(username).json['archives']
    idx = 0
    while year_month != data[idx][-7:]:
        idx += 1
    url = data[idx]
    return url





#------------------------------------------------------------------------------------------------------------------------------------
#-----------------------FILTRER LES SERIES DE DEFAITES
#------------------------------------------------------------------------------------------------------------------------------------
def get_df_losing_streak (df):
    # Identifier les séquences de défaites consécutives
    streak = (df['player_result'] == 'lose').astype(int)
    df['streak'] = streak.groupby((streak != streak.shift()).cumsum()).cumsum()

    # Trouver les indices des séries de défaites de plus de 3
    lose_streak_indices = df[df['streak'] >= 3].index

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







#------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------IMPORTER LES 100 DERNIERES PARTIES DES ADVERSAIRES
#------------------------------------------------------------------------------------------------------------------------------------
#Cette fonction permet de générer un dataframe qui regroupe les 100 parties de tous les adversaire d'un joueur en fonction du dataframe fourni
def get_df_opponents_last_100_games (player_username, df):
    
    #On cré un dataframe dans lequel on charge les 100 parties du premier adversaire
    opponent_username = df['opponent_username'].loc[0]
    game_id_reference = df['game_id'].loc[0]
    year_month = df['year_month'].loc[0]
    df_opponent_last_100_games = get_last_100_games_before_ref(game_id_reference, get_opponent_historic(opponent_username, year_month))
    
    #Pour la suite on fait la même chose mais on ajoute à la suite du dataframe les parties des autres adversaires
    for i in tqdm(range(1, df.shape[0]), desc="Generating KPI"):
        #On met à jour les variables
        opponent_username = df['opponent_username'].loc[i]
        game_id_reference = df['game_id'].loc[i]
        year_month = df['year_month'].loc[i]
        #On ajoute à la suite les données chargées
        added_df = get_last_100_games_before_ref(game_id_reference, get_opponent_historic(opponent_username, year_month))
        df_opponent_last_100_games = pd.concat([df_opponent_last_100_games, added_df], axis=0)
        
        #On enregistre une fois tous les 10 itérations
        if i%10==0 : df_opponent_last_100_games.to_csv(f'./dataframes_saved/df_opponent_last_100_games_{player_username}.csv', index=False)
        
    #On enregistre le dataframe une dernière fois
    df_opponent_last_100_games.to_csv(f'./dataframes_saved/df_opponent_last_100_games_{player_username}.csv', index=False)
    
    return df_opponent_last_100_games
        
def get_last_100_games_before_ref(game_id_ref, df):
    
    #On retrouve dans l'historique du joueur l'indice de la partie donnée en référence
    idx_game_ref = df[df['game_id']==game_id_ref].index[0]

    #On récupère les 100 parties qui ont précédées la partie de référence
    df_last_100_games = df.loc[max(0, idx_game_ref - 100):idx_game_ref-1].copy() #on utilise max() dans le cas ou un dataframe soit inférieur à 100 lignes

    #On enregistre le game_id de la partie qui nous a opposé
    df_last_100_games['game_id_ref']=game_id_ref
    
    return df_last_100_games

#Fonction pour renvoyer l'historique d'un joueur avec le bon format
def get_opponent_historic(username, year_month):
    # Load the player's game history
    df = get_opponent_historic_brut(username, year_month)
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

def get_opponent_historic_brut(username, year_month):
    # Obtenir la liste des mois pour l'utilisateur donné
    months_list = get_months_list(username)
    rev_months_list = list(reversed(months_list))
    # Trouver l'index du mois donné dans la liste
    index = rev_months_list.index(year_month)

    # Initialiser un DataFrame vide
    df = pd.DataFrame()
    
    # Ajouter des données jusqu'à ce que le DataFrame ait au moins 100 lignes
    for month in rev_months_list[max(index-1, 0):index+3]:
        added_df = get_month_results(username, month)
        added_df['Year_Month'] = month
        df = pd.concat([df, added_df], ignore_index=True)
    # Retourner uniquement les 100 premières lignes pour s'assurer de la limite
    return df







        


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------FONCTIONS ASSOCIEES AU KPI
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
# ANALYSE DES COUPS
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Regrouper tous les coups dans un même dataframe en discociant les coups joués par les blancs et ceux joués par les noirs
def get_moves_from_a_game(game, game_id):
    df_white, df_black = get_moves_for_a_game(game)

    df = pd.merge(df_white, df_black, left_on='move_nbr', right_on='move_nbr', how='left')
    df['game_id_ref'] = game_id
    return df

#Extraire les coups du json
def get_moves_for_a_game (moves) :
    # Pattern pour extraire les informations
    white_pattern = r'(\d+\.\s*\S+)\s*\{\[%clk (\d+:\d+:\d+(?:\.\d+)?)\]}'
    black_pattern = r' (\d+\...\s*\S+)\s*\{\[%clk (\d+:\d+:\d+(?:\.\d+)?)\]} '

    # Récupération des coups de chaque joueur
    df_white_moves = pd.DataFrame(re.findall(white_pattern, moves))
    df_black_moves = pd.DataFrame(re.findall(black_pattern, moves))

    df_white_moves = rename_moves_columns(df_white_moves, 'white')
    df_black_moves = rename_moves_columns(df_black_moves, 'black')
    
    df_white_moves = set_moves_for_a_game_format(df_white_moves, 'white')
    df_black_moves = set_moves_for_a_game_format(df_black_moves, 'black')
    
    return df_white_moves, df_black_moves

def rename_moves_columns(df, color):
    df = df.rename(columns={0: f'{color}_move'})
    df = df.rename(columns={1: f'{color}_remaining_time'})
    return df

#On définit le format souhaité pour les données
def set_moves_for_a_game_format(df, color):
    # Convertir les chaînes de caractères en objets datetime
    df[f'{color}_remaining_time'] = pd.to_datetime(df[f'{color}_remaining_time'])
    # Ignorer tous les avertissements
    warnings.filterwarnings('ignore')
    
    # Convertir le temps initial en objet datetime
    initial_time = pd.to_datetime('0:10:00.0')

    # Calculer le temps pris pour chaque coup en termes de timedelta
    df[f'{color}_time_to_play_the_move'] = df[f'{color}_remaining_time'].shift(1) - df[f'{color}_remaining_time']

    # Calculer le temps pris pour le premier coup
    df.loc[0, f'{color}_time_to_play_the_move'] = initial_time - df.loc[0, f'{color}_remaining_time']

    # Appliquer la fonction pour formater le timedelta
    df[f'{color}_time_to_play_the_move'] = df[f'{color}_time_to_play_the_move'].apply(extract_time_components)

    # on formate la colonne 'white_remaining_time' pour ne garder que des minutes, secondes et millisecondes
    df[f'{color}_remaining_time'] = df[f'{color}_remaining_time'].apply(lambda x: x.strftime('%M:%S.%f')[:-3])
    
    #On récupère le numéro du mouvement et 
    if color == 'white' :
        splitStr = '. '
    else :
        splitStr = '... '
    df['move_nbr'] = [move.split(splitStr)[0] for move in df[f'{color}_move']]
    df[f'{color}_move'] = [move.split(splitStr)[1] for move in df[f'{color}_move']]

    return df

# Fonction pour extraire les minutes, secondes et millisecondes d'un timedelta
def extract_time_components(td):
    total_seconds = int(td.total_seconds())
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    milliseconds = td.microseconds // 1000
    return f'{minutes:02}:{seconds:02}.{milliseconds:03}'



def convert_string_in_deciseconds (time_str):
    minutes = int(time_str[:2])
    secondes = int(time_str[3:5])
    milliseconds = int(time_str[6:9])
    return (minutes*60*1000 + secondes*1000 + milliseconds)//100
    
    
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#                       A TRIER

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_evol_curve_over_time (df, abscisseStr, ordonneeStr, titleStr) :
    # Conversion de la colonne de dates en datetime
    df[abscisseStr] = pd.to_datetime(df[abscisseStr])

    # Calculer la moyenne mobile sur une semaine
    rolling_average = df[ordonneeStr].rolling(window=7).mean()

    # Tracer la courbe des données originales
    plt.figure(figsize=(10, 6))
    plt.plot(df[abscisseStr], df[ordonneeStr], marker=',', label=abscisseStr)

    # Tracer la moyenne mobile sur une semaine
    plt.plot(df[abscisseStr], rolling_average, color='red', label='Moyenne mobile (1 semaine)')

    # Ajouter des titres et des labels
    plt.title(titleStr)
    plt.xlabel(abscisseStr)
    plt.ylabel(ordonneeStr)
    plt.legend()

    # Afficher le graphique
    plt.show()
    

    
def clean_dataframe_of_outliers(df, columnStr, threshold=2):
    #On calcule la moyenne et l'écart type
    mean_value = df[columnStr].mean()
    std_dev_value = df[columnStr].std()
    
    #On définit les seuils bas et hauts de suppression
    lower_threshold = mean_value - threshold * std_dev_value
    upper_threshold = mean_value + threshold * std_dev_value
    
    #On identifie les outliers dans le dataframe
    outliers_df = df[(df[columnStr] < lower_threshold) | (df[columnStr] > upper_threshold)]
    
    #On les retire
    cleaned_df = df[(df[columnStr] >= lower_threshold) & (df[columnStr] <= upper_threshold)]
    
    return cleaned_df


def classer_dataframe(df):
    # Créer une nouvelle colonne pour les classes
    df['Classe'] = pd.cut(df['nbr_game_per_day'], bins=[0, 5, 10, 15, 20, 25, 30, float('inf')], labels=['Q1 : <5', 'Q2 : 5<x<10', 'Q3 : 10<x<15', 'Q4 : 15<x<20', 'Q5 : 20<x<25', 'Q6 : 25<x<30', 'Q7 : 30<'], right=False)
    return df



def plot_histogramme (df, abscisseStr, ordonneeStr, title):
    
    #On trie les données dans l'ordre décroissant
    df = df.sort_values(ordonneeStr, ascending=False)
    
    # Tracer l'histogramme
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df[abscisseStr], df[ordonneeStr], color='skyblue')

    # Ajouter le nombre contenu dans chaque barre
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{round(df.iloc[idx][ordonneeStr],1)}', ha='center', va='bottom', rotation = 45)

    plt.xlabel(abscisseStr)
    plt.ylabel(ordonneeStr)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')  # Rotation des dates pour une meilleure lisibilité
    plt.grid(axis='y')  # Grille uniquement sur l'axe y
    plt.tight_layout()  # Ajustement automatique de la mise en page
    plt.show()
    
    
    
def get_all_moves_played_by_username (df_moves):
    #On télécharge les coups de la première partie
    moves = df_moves['moves'].loc[0] #On récupère les coups sous forme de STR
    game_id = df_moves['game_id'].iloc[0] #On donne le game_id pour pouvoir faire le parallele entre la partie jouée et les coups
    df_all_moves = f.get_moves_from_a_game(moves, game_id) #On enregistre dans un dataframe

    #on télécharge les coups de toutes les autres parties
    for i in range(1,df_moves.shape[0]):
        moves = df_moves['moves'].loc[i]
        game_id = df_moves['game_id'].iloc[i]
        try :
            #On ajoute à la suite les données chargées
            added_df = f.get_moves_from_a_game(moves, game_id)
            df_all_moves = pd.concat([df_all_moves, added_df], axis=0)
        except KeyError:
            print("Erreur d'import index :", i, 'moves :', moves)
            
    #On crée le dataframe de mapping de couleur
    df_color_mapping = df_moves[['game_id', 'player_color']].reset_index()

    # Initialisation du DataFrame df_moves_played_by_username
    df_moves_played_by_username = pd.DataFrame()

    # Boucle sur les lignes de df_color_mapping
    for i in range(df_color_mapping.shape[0]):
        game_id = df_color_mapping['game_id'].iloc[i]
        color = df_color_mapping['player_color'].iloc[i]

        # Filtrer les données de df_all_moves une seule fois
        filtered_moves = df_all_moves[df_all_moves['game_id_ref'] == game_id]

        # Créer le DataFrame df_moves_played_by_username avec les données filtrées
        added_df = pd.DataFrame({
            'move_nbr': filtered_moves['move_nbr'],
            'move': filtered_moves[f'{color}_move'],
            'game_id': game_id,
            'color': color
        })

        # Concaténer added_df avec df_moves_played_by_username
        df_moves_played_by_username = pd.concat([df_moves_played_by_username, added_df], axis=0)

        
    #On supprime les NaN
    df_moves_played_by_username = df_moves_played_by_username[~df_moves_played_by_username['move'].isna()]
    df_moves_played_by_username.reset_index(drop=True, inplace=True)
    
    return df_moves_played_by_username



def rate_avg_ELO_win_per_attribut (df, attributStr, plotTitleStr):
    grouped_df = df.groupby(attributStr).agg(
        times_appearance=('nbr_game_per_day', 'size'),
        avg_ELO_win=('ELO_win_per_nbr_game', 'mean')
    ).reset_index()

    #Comme certains nbr_games_played ne sont pas représentés suffisamment dans l'échantillon,
    #nous allons déterminer un seuil à partir duquel, on considère que les statistiques seront en nombre suffisant 
    #Pour cela, on calcule la moyenne proportionnelle, qui sera notre seuil minimum
    seuil = min(20, grouped_df['times_appearance'].sum() / grouped_df.shape[0])
    #On retire les données dont l'échantillon est trop faible
    grouped_df = grouped_df[grouped_df['times_appearance']>=seuil] 

    #On affiche les histogrammes
    plot_histogramme (grouped_df, attributStr, 'avg_ELO_win', plotTitleStr)

    
    
#Maintenant, on récupère la séquence des 3 premières parties
def get_sequence_first_x_games (df, date, x_first_gamesInt):
    return str(df[df['end_date']==date][:x_first_gamesInt]['player_result'].tolist())



def get_classification_first_x_games_result (df, x_first_gamesInt):
    
    df_date = df['end_date']
    q="""
    SELECT 
        end_date,
        COUNT(end_date) AS nbr_games 
    FROM df_date
    GROUP BY end_date
    """
    df_date =sqldf(q)

    #On conserve uniquement les jours où on a au moins X parties jouées
    df_date = df_date[df_date['nbr_games'] >= x_first_gamesInt]

    list_dates = df_date['end_date'].tolist()

    #Obtenir la liste des parties pour les jours où on a joué au moins x_first_gamesInt
    df_games = pd.DataFrame()
    for date in list_dates :
        added_df = df[df['end_date']== date]
        df_games = pd.concat((df_games,added_df), axis=0)


    #On compte le nombre de parties jouées par jour (sans compter les x premières parties)
    q="""
    SELECT 
        end_date,
        COUNT(game_id) AS nbr_games_played
    FROM df_games
    GROUP BY end_date
    """
    df_nbr_games_played_per_date = sqldf(q)
    df_nbr_games_played_per_date['nbr_games_played'] = df_nbr_games_played_per_date['nbr_games_played'] - x_first_gamesInt


    #A présent on veut connaître quels ont été les résultats des x premières parties
    df_sequence_first_x_games = pd.DataFrame()
    for date in list_dates :
        sequence_first_x_games = get_sequence_first_x_games(df_games, date, x_first_gamesInt)

        added_df = pd.DataFrame({
            'end_date': [date] * len(sequence_first_x_games),  # Répéter la date pour chaque résultat
            'sequence': sequence_first_x_games  # Colonnes pour les résultats des 3 premières parties
        })

        df_sequence_first_x_games = pd.concat([df_sequence_first_x_games, added_df], ignore_index=True)


    #Maintenant on joint les dataframes df_sequence_first_3_games & df_nbr_games_played_per_date
    df_sequence_first_x_games = pd.merge(df_sequence_first_x_games, df_nbr_games_played_per_date, on='end_date', how='left')
    df_sequence_first_x_games = df_sequence_first_x_games.drop_duplicates().reset_index(drop=True)
    
    #Maintenant on joint les dataframes df_sequence_first_3_games & df_nbr_games_played_per_date
    df_x_games_or_more = pd.merge(df_games, df_sequence_first_x_games, on='end_date', how='left')
    df_x_games_or_more = df_x_games_or_more.drop_duplicates().reset_index(drop=True)
    
    #On retire les x premières parties de chaque jour :
    classification_first_x_games_result = pd.DataFrame()
    for date in list_dates :
        added_df = df_x_games_or_more[(df_x_games_or_more['end_date'] == date)].iloc[x_first_gamesInt:]
        classification_first_x_games_result = pd.concat([classification_first_x_games_result, added_df], ignore_index=True)
        
    #On ajoute une colonne 'score'
    classification_first_x_games_result = get_score_for_sequence(classification_first_x_games_result)
    
    return classification_first_x_games_result



def get_ELO_win_this_day (df):
    q="""
    SELECT
        end_date,
        player_rating AS ELO_beginning_of_the_day
    FROM df
    GROUP BY end_date
    """
    df = sqldf(q)

    df['ELO_win_this_day'] = df['ELO_beginning_of_the_day'].diff().fillna(0)
    
    return df


def get_stats_per_day (df):
    # Utilisation de shift pour décaler les valeurs vers le bas
    df['ELO_end_of_day'] = 0
    df['ELO_end_of_day'] = df['ELO_beginning_of_the_day'].shift(-1)

    #On calcule le nombre de points ELO gagnés après les 3 premières parties
    df['ELO_win_after_sequence'] = df['ELO_end_of_day'] - df['ELO_after_sequence']

    #On calcule le nombre de points ELO gagnés en moyenne par partie après la séquence jouée
    df['avg_ELO_win_per_game'] = round(df['ELO_win_after_sequence']/df['nbr_games'], 2)    

    #On calcule le nombre moyen de coups par partie et par jour
    df['avg_moves_per_game'] = round(df['nbr_moves_per_day']/df['nbr_games'], 2)
    
    #On calcule le ratio de victoires/nulles/défaites
    df['win_rate_after_sequence'] = round(df['nbr_win_after_sequence']/df['nbr_games'],2)
    df['draw_rate_after_sequence'] = round(df['nbr_draw_after_sequence']/df['nbr_games'],2)
    df['lose_rate_after_sequence'] = round(df['nbr_lose_after_sequence']/df['nbr_games'],2)


    #On réorganise le dataframe
    df = df[['end_date','sequence', 'score', 'ELO_beginning_of_the_day', 'ELO_after_sequence', 'ELO_end_of_day','ELO_win_after_sequence','nbr_games','nbr_moves_per_day','avg_moves_per_game','avg_ELO_win_per_game', 'nbr_win_after_sequence', 'nbr_draw_after_sequence', 'nbr_lose_after_sequence', 'win_rate_after_sequence', 'draw_rate_after_sequence', 'lose_rate_after_sequence']]

    return df


def  filtre_seuil(df, attibutStr, seuil_minInt):
    seuil = min(seuil_minInt, df[attibutStr].sum() / df.shape[0])
    #On retire les données dont l'échantillon est trop faible
    df = df[df[attibutStr]>=seuil] 
    
    return df



def get_score_for_sequence(df):
    scores = []
    for i in range(df.shape[0]):
        sequence = df['sequence'].iloc[i]
        
        nbr_wins = sequence.count('win')
        nbr_draws = sequence.count('draw')
        nbr_loses = sequence.count('lose')
        
        score = f"W: {nbr_wins} - D: {nbr_draws} - L: {nbr_loses}"
        scores.append(score)
    
    df['score'] = scores
    return df




def plot_linear_regression (df, abscisseStr, ordonneeStr) :
    # Calculer la moyenne mobile sur une semaine
    rolling_average = df[ordonneeStr].rolling(window=14).mean()

    
    # Convertir les dates en nombres de jours depuis le début
    df[abscisseStr] = pd.to_datetime(df[abscisseStr])
    days_since_start = (df[abscisseStr] - df[abscisseStr].min()).dt.days

    # Créer un modèle de régression linéaire
    model = LinearRegression()

    # Adapter le modèle aux données existantes
    model.fit(np.array(days_since_start).reshape(-1, 1), df[ordonneeStr])

    # Générer les prédictions de la régression linéaire sur l'intervalle des données existantes
    predictions = model.predict(np.array(days_since_start).reshape(-1, 1))

    # Tracer la courbe des données originales
    plt.figure(figsize=(10, 6))
    plt.plot(df[abscisseStr], df[ordonneeStr], marker='o', label=abscisseStr)

    # Tracer la moyenne mobile sur une semaine
    plt.plot(df[abscisseStr], rolling_average, color='red', label='Moyenne mobile (2 semaines)')

    # Tracer la projection de l'évolution de la courbe
    plt.plot(df[abscisseStr], predictions, linestyle='--', color='magenta', label='Projection (régression linéaire)')

    # Ajouter des titres et des labels
    plt.title(f"Évolution {ordonneeStr} dans le temps")
    plt.xlabel(abscisseStr)
    plt.ylabel(ordonneeStr)
    plt.grid(True)
    plt.legend()

    # Afficher le graphique
    plt.show()
    
    #Calcul equation de droite
    if (predictions[-1]-predictions[0])>= 0 :
        print('Régression linéaire croissante')
    else :
          print('Régression linéaire décroissante')
            
            
            
            
def get_stats_next_game (df_grouped_by_day, df_next_game) :
    #Initialisation de la colonne
    df_grouped_by_day['next_game_result'] = 0
    
    #On récupère la liste des dates
    list_dates = df_grouped_by_day['end_date'].unique().tolist()

    for date in list_dates:
        #On filtre df_next_game avec la date actuelle
        filtered_df = df_next_game[df_next_game['end_date'] == date]
        #On réucpère l'index ou se trouve la date dans le dataframe
        index = df_grouped_by_day[df_grouped_by_day['end_date'] == date].index[0]
        
        if filtered_df['player_result'].empty:
            result = 'no game played'
        else:
            # Assumer ici que player_result est un seul résultat par date
            result = filtered_df['player_result'].iloc[0]
        
        #On renseigne la valeur dans le dataframe
        df_grouped_by_day['next_game_result'].iloc[index] = result
        
    return df_grouped_by_day
    


def get_df_group_by_day (df1, df2) :
    
    df_grouped_by_day = df2
    #on regroupe les données par date
    q="""
    SELECT
        end_date,
        sequence,
        score,
        player_rating AS ELO_after_sequence,
        SUM(CASE WHEN player_result = 'win' THEN 1 ELSE 0 END) AS nbr_win_after_sequence,
        SUM(CASE WHEN player_result = 'draw' THEN 1 ELSE 0 END) AS nbr_draw_after_sequence,
        SUM(CASE WHEN player_result = 'lose' THEN 1 ELSE 0 END) AS nbr_lose_after_sequence,
        SUM(CASE WHEN player_result = 'win' THEN 1 ELSE 0 END) / nbr_games_played AS win_rate_after_sequence,
        SUM(CASE WHEN player_result = 'draw' THEN 1 ELSE 0 END) / nbr_games_played AS draw_rate_after_sequence,
        SUM(CASE WHEN player_result = 'lose' THEN 1 ELSE 0 END) / nbr_games_played AS lose_rate_after_sequence,
        SUM(nbr_moves) AS nbr_moves_per_day,
        nbr_games_played AS nbr_games
    FROM df_grouped_by_day
    GROUP BY end_date
    """
    df_grouped_by_day = sqldf(q)

    #On récupère les statistiques sur toutes les parties
    df_ELO_win_this_day = f.get_ELO_win_this_day(df1)

    #Maintenant on joint les dataframes df_grouped_by_day & df_stats_per_day
    df_grouped_by_day = pd.merge(df_grouped_by_day, df_ELO_win_this_day, on='end_date', how='left')
    df_grouped_by_day = df_grouped_by_day.drop_duplicates().reset_index(drop=True)

    #On calcule les statistiques souhaitées
    df_grouped_by_day = f.get_stats_per_day(df_grouped_by_day)

    return df_grouped_by_day



def get_df_next_game_after_x_games (df, x_first_games):
    #On charge les parties avec ces propirétés :
        # les jours où il y a eu au moins 3 parties jouées
        # On charge la séquence des 3 premières parties
        # On n'importe pas les 3 premières parties jouées par jour
    df_grouped_by_day = f.get_classification_first_x_games_result(df, x_first_games).sort_values('end_time')
    
    list_dates = df_grouped_by_day['end_date'].unique().tolist()
    
    #Obtenir la liste des parties pour les jours où on a joué au moins x_first_gamesInt
    df_next_match = pd.DataFrame()
    for date in list_dates :
        try :
            added_df = df_grouped_by_day[df_grouped_by_day['end_date']== date].iloc[x_first_games].to_frame().T
            df_next_match = pd.concat((df_next_match,added_df), axis=0)
        except IndexError:
            # Gérer l'erreur si il n'y a pas assez de lignes dans df_grouped_by_day pour sélectionner avec iloc[x_first_games]
            shape_is_too_small=0 #instruction pour passr l'étape
    return df_next_match




def get_df_stats_after_x_games(df, x_first_games, groupbyStr):
    #On charge les parties avec ces propirétés :
        # les jours où il y a eu au moins 3 parties jouées
        # On charge la séquence des 3 premières parties
        # On n'importe pas les 3 premières parties jouées par jour
    df_grouped_by_day = f.get_classification_first_x_games_result(df, x_first_games).sort_values('end_time')
    df_next_game = get_df_next_game_after_x_games(df, x_first_games)
    
    #On regroupe les données par jour
    df_grouped_by_day = get_df_group_by_day (df, df_grouped_by_day)
    
    #On recupère le résultat du match qui suit
    df_grouped_by_day = get_stats_next_game (df_grouped_by_day, df_next_game)
    
    #On regroupe les données par séquence
    q=f"""
        SELECT
            {groupbyStr},
            COUNT(end_date) AS nbr_{groupbyStr}_occured,
            SUM(nbr_win_after_sequence) AS nbr_win_by_{groupbyStr},
            SUM(nbr_draw_after_sequence) AS nbr_draw_by_{groupbyStr},
            SUM(nbr_lose_after_sequence) AS nbr_lose_by_{groupbyStr},
            SUM(CASE WHEN next_game_result = 'win' THEN 1 ELSE 0 END) AS nbr_win_next_games,
            SUM(CASE WHEN next_game_result = 'draw' THEN 1 ELSE 0 END) AS nbr_draw_next_games,
            SUM(CASE WHEN next_game_result = 'lose' THEN 1 ELSE 0 END) AS nbr_lose_next_games,
            SUM(CASE WHEN next_game_result = 'no game played' THEN 1 ELSE 0 END) AS nbr_nogame_next_games,
            SUM(nbr_games) AS nbr_games,
            ROUND(AVG(nbr_games), 2) AS avg_games_played_after_{groupbyStr},
            AVG(avg_moves_per_game) AS avg_nbr_moves,
            AVG(avg_ELO_win_per_game) AS avg_ELO_win_per_game_after_{groupbyStr}
        FROM df_grouped_by_day
        GROUP BY {groupbyStr}
    """
    df_grouped =sqldf(q)
    
    divisor = df_grouped['nbr_win_next_games']+df_grouped['nbr_draw_next_games']+df_grouped['nbr_lose_next_games']

    #On caclule les probabilités de victoires/nulles/défaites sur la parties suivante 
    df_grouped['win_rate_next_games'] = df_grouped['nbr_win_next_games']/(divisor)*100
    df_grouped['draw_rate_next_games'] = df_grouped['nbr_draw_next_games']/(divisor)*100
    df_grouped['lose_rate_next_games'] = df_grouped['nbr_lose_next_games']/(divisor)*100
    
    #On calcule les taux de victoires/nulles/défaites sur toutes les parties suivantes
    df_grouped[f'win_rate_by_{groupbyStr}'] = df_grouped[f'nbr_win_by_{groupbyStr}']/df_grouped['nbr_games']*100
    df_grouped[f'draw_rate_by_{groupbyStr}'] = df_grouped[f'nbr_draw_by_{groupbyStr}']/df_grouped['nbr_games']*100
    df_grouped[f'lose_rate_by_{groupbyStr}'] = df_grouped[f'nbr_lose_by_{groupbyStr}']/df_grouped['nbr_games']*100

    #On ne prend pas en compte les données qui ne sont pas suffisemment représentées
    df_grouped = f.filtre_seuil(df_grouped, f'nbr_{groupbyStr}_occured', 10)
    df_grouped = round(df_grouped, 2)

    return df_grouped



def plot_win_probability (df_entry, first_x_games, groupbyStr, title):

    #On récupère les statistiques
    df = get_df_stats_after_x_games(df_entry, first_x_games, groupbyStr).sort_values('win_rate_next_games', ascending= False)

    # Récupération des données depuis le DataFrame
    sequences = df[groupbyStr].tolist()
    win_rates = df['win_rate_next_games'].tolist()
    draw_rates = df['draw_rate_next_games'].tolist()
    lose_rates = df['lose_rate_next_games'].tolist()


    # Création de l'histogramme empilé
    plt.figure(figsize=(10, 6))
    bar_width = 0.35

    # Position des barres sur l'axe des x
    r = range(len(sequences))

    # Création des barres empilées
    bars1 = plt.bar(r, win_rates, color='olivedrab', edgecolor='white', width=bar_width, label='Gagné')
    bars2 = plt.bar(r, draw_rates, bottom=win_rates, color='gold', edgecolor='white', width=bar_width, label='Nul')
    bars3 = plt.bar(r, lose_rates, bottom=[i+j for i,j in zip(win_rates, draw_rates)], color='indianred', edgecolor='white', width=bar_width, label='Perdu')

    # Ajout des pourcentages sur chaque segment de barre
    for bar, win, draw, lose in zip(r, win_rates, draw_rates, lose_rates):
        plt.text(bar, win / 2, f'{round(win,1)}%', ha='center', va='center', color='black')
        plt.text(bar, win + draw / 2, f'{round(draw,1)}%', ha='center', va='center', color='black')
        plt.text(bar, win + draw + lose / 2, f'{round(lose,1)}%', ha='center', va='center', color='black')

    # Ajout des étiquettes et titre
    plt.xlabel(f'Groupé par {groupbyStr}', fontweight='bold')
    plt.xticks(r, sequences, rotation =45)
    plt.ylabel('Probabilités associées')
    plt.title(title)
    plt.legend()

    # Affichage de l'histogramme
    plt.show()
    
    
    
    
    
def get_df_grouped_by_end_date(df):
    #On récupère les données en les triant par end_time
    df_grouped_by_day = df.sort_values('end_time')

    #On réalise une requete SQL pour les regrouper par date 
    q="""
    SELECT
        end_date,
        player_rating AS ELO_beginning_of_day,
        SUM(CASE WHEN player_result = 'win' THEN 1 ELSE 0 END) AS nbr_win_this_day,
        SUM(CASE WHEN player_result = 'draw' THEN 1 ELSE 0 END) AS nbr_draw_this_day,
        SUM(CASE WHEN player_result = 'lose' THEN 1 ELSE 0 END) AS nbr_lose_this_day,
        SUM(nbr_moves) AS nbr_moves_played_per_day,
        COUNT(game_id) AS nbr_game_per_day
    FROM df_grouped_by_day
    GROUP BY end_date
    """
    df_grouped_by_day=sqldf(q)

    #Initialisation de la colonne ELO_diff pour calculer la différence d'ELO entre le début et la fin de la journée
    df_grouped_by_day['ELO_diff']= 0
    #On calcule le ELO gagné ou perdu entre 2 jours consécutifs
    df_grouped_by_day['ELO_diff'].iloc[1:] = df_grouped_by_day['ELO_beginning_of_day'].diff().iloc[1:]

    #On calcule le ratio pour chaque jour d'ELO gagné par partie pour chaque jour
    df_grouped_by_day['ELO_win_per_nbr_game']=0
    for i in range(1, df_grouped_by_day.shape[0]):
        df_grouped_by_day.at[i, 'ELO_win_per_nbr_game'] = df_grouped_by_day.at[i, 'ELO_diff'] / df_grouped_by_day.at[i-1, 'nbr_game_per_day']


    #On calcule le nombre moyen de coups par partie et par jour
    df_grouped_by_day['moves_per_game'] = df_grouped_by_day['nbr_moves_played_per_day']/df_grouped_by_day['nbr_game_per_day']
    df_grouped_by_day['moves_per_game'] = df_grouped_by_day['moves_per_game'].round().astype(int)

    return df_grouped_by_day



#Fonction pour supprimer les outliers d'un dataframe
def remove_outliers(df, column_name, threshold=3.5):
    median = df[column_name].median()
    mad = np.median(np.abs(df[column_name] - median))
    max_threshold = median + threshold * mad
    min_threshold = median - threshold * mad
    
    # Filter the DataFrame to remove outliers
    filtered_df = df[(df[column_name] >= min_threshold) & (df[column_name] <= max_threshold)]
    
    return filtered_df


def plot_distribution_by_attribut (df, attributStr):
    #On réalise des KPI via une requete SQL
    #On affiche les statistiques importantes
    print(round(df[attributStr].describe(),2))

    q=f"""
    SELECT
        {attributStr},
        COUNT(end_date) AS frequency
    FROM df
    GROUP BY {attributStr}
    ORDER BY {attributStr}
    """
    df = sqldf(q)
    
    #On supprime les valeurs outliers
    df = remove_outliers(df, attributStr)

    f.plot_histogramme (df, attributStr, 'frequency', f'Distribution de {attributStr}')