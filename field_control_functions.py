import csv
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.metrics import auc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore


in_notebook = False
if in_notebook == True:
	from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
	init_notebook_mode(connected=True)
	pd.options.display.max_rows = 150
	%matplotlib inline


def is_score(game_df):
    """
    Rules based system to check to see if either team scored on the 
    previous play. Takes in a dataframe for a game and returns a list
    to be turned into a column for the initial dataframe. 
    """
    is_score = list() 
    for idx, row in game_df.iterrows():
        # Iterates through each row of the dataframe
        if idx == 0:
            is_score.append(None)
        else:
            prev_row = game_df.iloc[(idx - 1), :]
            if (np.abs(row.score_differential) == np.abs( \
            	prev_row.score_differential)) & \
                (row.offteam == prev_row.offteam):
                is_score.append(None) # something other than None?
            elif (row.score_differential > -prev_row.score_differential) & \
                (row.offteam != prev_row.offteam):
                is_score.append(1)
            elif (row.score_differential < -prev_row.score_differential) & \
                (row.offteam != prev_row.offteam):
                is_score.append(prev_row.offteam)
            elif (row.score_differential == -prev_row.score_differential) & \
                (row.offteam != prev_row.offteam):
                is_score.append(None)
            elif (row.score_differential != prev_row.score_differential) & \
                (row.offteam == prev_row.offteam):
                is_score.append(prev_row.defteam)
            else:
                is_score.append(500)
    
    return is_score


def force_zero_start(game):
    """
    Places the first play of the game to 0 seconds to ensure all plots start
    at the same point. 
    """
    game_copy = game.reset_index(drop = True).copy() # might be able to drop this
    game_copy.loc[0, 'time_in_reg'] = 0
    return game_copy

def force_max_end(game):
    """
    Stretches the last play of the game to the maximum time in seconds for a 
    regulation NCAA football game, which is 3600 seconds to ensure all plots
    end at the same point. 
    """
    game_copy = game.reset_index(drop = True).copy() # might be able to drop this
    last_row = len(game_copy) - 1
    game_copy.loc[last_row, 'time_in_reg'] = 3600
    return game_copy
    
def get_game(df, game_num):
    """
    Takes in inital df with all games, finds relevant game using the game number, 
    and subsets and sorts the dataframe before returning it. 
    """
    game_df = df.query("gsis_game_id == @game_num")
    game_cols = ['play_index','time_in_reg', 'offteam', 
    				'defteam', 'field_position', 'score_differential']
    final_game_df = game_df.loc[:, game_cols].sort_values('time_in_reg').\
    					reset_index(drop = True).copy()
    return final_game_df
    

def flip_fp(row, main_team):
    """
    Flips the field position column of the game dataframe so that each play 
    is going in the same direction on the field. This enables the stacked
    area plot to be properly formatted. 
    """
    if row["defteam"]==main_team:
        row["field_position"] = 100 - row["field_position"]
    return row

def field_control_score(game_df):
	"""
	Calculates the field control score using sklearn's auc function
	Note: It sorts the game_df based on field position and game_df
		needs to be resorted afterwards.
	"""
    game_dfc = game_df.copy()
    x_game = game_dfc['time_in_reg']
    y_game = game_dfc['field_position']
    total_area = 3600*100
    score = auc(x = x_game, y = y_game, reorder=True)/total_area
    return score


def plot_field_control(game_df, fc_score):
    """
    Stacked area plots showing field position on the y-axis and time in 
    regulation on the x-axis. Additionally includes vertical lines for 
    scoring plays by each team.
    :game_df: A dataframe with play observations for a single game
    :fc_score: A field control score calculated for the specified game. 
    """
    team_one = game_df.loc[0, 'offteam']
    team_two = game_df.loc[0, 'defteam']
    trace0 = go.Scatter(
        x=game_df['time_in_reg'],
        y=game_df['field_position'],
        mode='lines',
        line=dict(width=0.5,
                  color='rgb(244,165,130)'),
        fill='tonexty', 
        name = team_one,
        text = game_df['offteam'],
        hoveron = 'none'
    )
    trace1 = go.Scatter(
        x=game_df['time_in_reg'],
        y=[100]*len(game_df['time_in_reg']),
        mode='lines',
        line=dict(width=0.5,
                  color='rgb(66, 134, 244)'),
        fill='tonexty',
        name = team_two, 
        hoverinfo = 'none'
    )

    scoring_lines = list()

    team_one_scoring_lines = [{'type': 'line', 'x0': row['time_in_reg'], 
    						'y0': 0, 'x1': row['time_in_reg'], 'y1': 100, 
    						'line': {'color': 'rgb(178,24,43)', 'width':3}}
                    for idx, row in game_df.loc[(game_df.is_score == 
                    	team_one), :].iterrows()]

    team_two_scoring_lines = [{'type': 'line', 'x0': row['time_in_reg'], 
    						'y0': 0, 'x1': row['time_in_reg'], 'y1': 100, 
    						'line': {'color': 'rgb(33,102,172)', 'width':3}}
                    for idx, row in game_df.loc[(game_df.is_score == 
                    	team_two), :].iterrows()]
    
    scoring_lines.extend(team_one_scoring_lines)
    scoring_lines.extend(team_two_scoring_lines)
    scoring_lines.append({'type': 'line', 'x0': 1800, 'y0': 0, 'x1': 1800, 
    	'y1': 100, 'line': {'color': 'rgb(105, 105, 105)', 'width': 2, 
    	'dash':'dot'}})
    
    
    
    data = [trace0, trace1]

    xaxis = go.XAxis(
        title = "Time Elapsed in Regulation (Seconds)",
        showgrid = True,
        range = [0, 3600],
        tickvals = [300*x for x in range(13)],
        ticktext = ['15:00 Q1', '10:00 Q1', '05:00 Q1', '15:00 Q2',
        			'10:00 Q2', '05:00 Q2', 'Half', '10:00 Q3', 
        			'05:00 Q3', '15:00 Q4', '10:00 Q4', '05:00 Q4', 
                   	'End of Game'],
        showticklabels = True
    )

    yaxis = go.YAxis(
        title = "Field Position",
        range = [1, 100],
        type = 'linear'
    )

    layout = go.Layout(
        title = "<b>" + game_df.loc[0, 'offteam'] + " vs " + \
        	game_df.loc[0, 'defteam'] + " Field Control Plot </b><br>" + \
            team_one + " FC Score: " + str(round(fc_score, 3)) + "<br>" + team_two + " FC Score: " + \
            str(round(1-fc_score,3)),
        xaxis = xaxis,
        yaxis = yaxis, 
        shapes = scoring_lines,
        showlegend = True
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='stacked-area-plot')
    return


def field_control_pipeline(df, game_num):
    """
    Pipeline taking in the season long data and a game number and performing 
    all of the relevant transformations and operations before plotting it. 
    """
    dfc = df.copy()
    game_df = get_game(dfc, game_num)
    game_df = force_zero_start(game=game_df)
    game_df = force_max_end(game=game_df)
    main_team = game_df.loc[0, 'offteam']
    game_df = game_df.apply(flip_fp, axis = 1, args = (main_team, ))
    game_df['is_score'] = is_score(game_df)
    fc_score = field_control_score(game_df=game_df)
    game_df = game_df.sort_values(by = 'time_in_reg').reset_index(drop = True)

    plot_field_control(game_df, fc_score)
    return 


def get_fc_scores_season(df):
    """
    Calculate the field control scores for every game in a season.
    Return a dictionary of games and information relating to them, such
    as teams, score differential, and field control scores. 
    """
    dfc = df.copy()
    all_games = dfc.gsis_game_id.unique().tolist()
    all_fc_scores = dict()
    for game_num in all_games:
        game_df = get_game(dfc, game_num)
        game_df = force_zero_start(game=game_df)
        game_df = force_max_end(game=game_df)
        main_team = game_df.loc[0, 'offteam']
        other_team = game_df.loc[0, 'defteam']
        if main_team == game_df.offteam.iloc[-1]:
            main_score_diff = game_df.score_differential.iloc[-1]
            other_score_diff = -main_score_diff
        else:
            other_score_diff = game_df.score_differential.iloc[-1]
            main_score_diff = -other_score_diff
        game_df = game_df.apply(flip_fp, axis = 1, args = (main_team, ))
        fc_score = field_control_score(game_df=game_df)
        all_fc_scores[game_num] = [main_team, fc_score, main_score_diff,
                                   other_team, 1-fc_score, other_score_diff]
    return all_fc_scores


def fc_vs_score_scatter(df, game_num = None):
    """
    Plots scatter graph of all games in the dataframe.
    Dataframe coming in should have game number, team, field control score,
    and score differential.
    Isolates one game and highlights it among all games,
    if a game is specified.
    """
    
    trace_all = go.Scatter(
        x = df.fc_team_1,
        y = df.team_1_score_diff,
        mode = 'markers',
        marker = {'opacity': 0.3, 'color': 'rgb(67,147,195)'},
        text = [row.team_1 + '<br>' + str(row.game_num) 
        		for _, row in df.iterrows()]
    )
    
    if game_num is not None:
        game_df = df.loc[df.game_num == game_num].reset_index(drop = True)
        team_one = game_df.loc[0, 'team_1']
        team_two = game_df.loc[1, 'team_1']

        trace_single = go.Scatter(
            x = game_df.fc_team_1,
            y = game_df.team_1_score_diff,
            mode = 'markers',
            marker = {'color': 'rgb(178,24,43)', 'size': 12},
            hoverinfo = 'none'
        )

        data = [trace_all, trace_single]  
    else:
        data = [trace_all]

    xaxis = go.XAxis(
        title = "Field Control Score",
        showgrid = True,
        showticklabels = True
    )

    yaxis = go.YAxis(
        title = "Score Differential",
        type = 'linear'
    )
    
    if game_num is not None:
        layout = go.Layout(
            title = "<b>Field Control Score vs Score Differential</b><br>" + \
            	team_one + " vs. " + \
                team_two,
            xaxis = xaxis,
            yaxis = yaxis, 
            showlegend = False
        )
    else:
        layout = go.Layout(
            title = "<b>Field Control Score vs Score Differential</b>",
            xaxis = xaxis,
            yaxis = yaxis, 
            showlegend = False
        )
    
    
    fig = go.Figure(data=data, layout=layout)   
    iplot(fig, filename = 'fc_score_scatter')
    return

# for highlighting, try changing contrast of other dots, increasing highlighted dot size/color
# time series field control score for each team
# FC Score --> Field Control Score


def fc_score_distplot(df, game_num = None):
    """
    This function takes in the season long data for each game, plots the
    distribution of field control scores, and then highlights the scores
    for the specified game, if one is specified. 
    """
    
    sns.set_style("whitegrid")

    g = sns.distplot(df.fc_team_1, color = '#92c5de')
    g.set_title('Distribution of Field Control Scores')
    g.set_xlabel('Field Control Score')
    g.set_yticks([])
    if game_num is not None:
        game_df = df.loc[df.game_num == game_num].reset_index(drop = True)
        team_one = game_df.loc[0, 'team_1']
        team_two = game_df.loc[1, 'team_1']
        fc_one = game_df.loc[game_df.team_1 == team_one].fc_team_1.values[0]
        fc_two = game_df.loc[game_df.team_1 == team_two].fc_team_1.values[0]
        fc_one_pctile = round(percentileofscore(df.fc_team_1, fc_one), 1)
        fc_two_pctile = round(percentileofscore(df.fc_team_1, fc_two), 1)
        g.vlines(x = fc_one, ymin = 0, 
                 ymax = 10, color = '#b2182b', label = team_one)
        g.vlines(x = fc_two, ymin = 0, 
                 ymax = 10, color = '#2166ac', label = team_two)
        g.legend()
        print team_one + " FC Score %ile: " + str(fc_one_pctile)
        print team_two + " FC Score %ile: " + str(fc_two_pctile)
    return