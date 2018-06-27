import csv
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.metrics import auc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore


#in_notebook = False
#if in_notebook == True:
#	from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#	init_notebook_mode(connected=True)
#	pd.options.display.max_rows = 150
#	%matplotlib inline


def is_score(game_df):
	"""
	Rules based system to check to see if either team scored on two 
	consecutive plays. Takes in a dataframe for a game and returns a list
	to be turned into a column for the initial dataframe. 
	"""
	is_score = list()
	last_idx = game_df.index[-1]
	for idx, row in game_df.iterrows():
		# Iterates through each row of the dataframe
		
		if idx == last_idx:
			is_score.append(None)
		else:
			next_row = game_df.iloc[(idx + 1), :]
			if (np.abs(row.score_differential) == np.abs(next_row.score_differential)) & \
				(row.offteam == next_row.offteam):
				is_score.append(None) # something other than None?
			elif (row.score_differential > -next_row.score_differential) & \
				(row.offteam != next_row.offteam):
				is_score.append(1)
			elif (row.score_differential < -next_row.score_differential) & \
				(row.offteam != next_row.offteam):
				is_score.append(next_row.defteam)
			elif (row.score_differential == -next_row.score_differential) & \
				(row.offteam != next_row.offteam):
				is_score.append(None)
			elif (row.score_differential != next_row.score_differential) & \
				(row.offteam == next_row.offteam):
				is_score.append(next_row.defteam)
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
	if game_copy.loc[last_row, 'time_in_reg'] < 3600:
		game_copy.loc[last_row, 'time_in_reg'] = 3600
	return game_copy
	
def get_game(df, game_num):
	"""
	Takes in inital df with all games, finds relevant game using the game number, 
	and subsets and sorts the dataframe before returning it. 
	"""
	game_df = df.query("gsis_game_id == @game_num")
	game_cols = ['play_index', 'drive', 'drive_play', 'time_in_reg', 'offteam',
				'defteam', 'field_position', 'score_differential', 'game_state', 'date']
	game_df = game_df.loc[:, game_cols].sort_values('time_in_reg').reset_index(drop = True).copy()
	game_df = force_zero_start(game=game_df)
	game_df = force_max_end(game=game_df)
	main_team = game_df.loc[0, 'offteam']
	game_df = game_df.apply(flip_fp, axis = 1, args = (main_team, ))
	game_df['is_score'] = is_score(game_df)
	return game_df
	

def flip_fp(row, main_team):
	"""
	Flips the field position column of the game dataframe so that each play 
	is going in the same direction on the field. This enables the stacked
	area plot to be properly formatted. 
	"""
	if row["defteam"]==main_team:
		row["field_position"] = 100 - row["field_position"]
	return row

# OLD FC SCORE CALCULATOR
# def field_control_score(game_df):
# 	"""
# 	Calculates the field control score using sklearn's auc function
# 	Note: It sorts the game_df based on field position and game_df
# 		needs to be resorted afterwards.
# 	"""
# 	game_dfc = game_df.copy()
# 	x_game = game_dfc['time_in_reg']
# 	y_game = game_dfc['field_position']
# 	total_area = 3600*100
# 	score = round(auc(x = x_game, y = y_game, reorder=True)/total_area, 3)
# 	return score


def fc_score_prep(game_df):
	"""
	Prepares dataframes for including and excluding garbage time for
	calculation of field control scores on the resulting dataframes.
	"""
	game_dfc = game_df.copy()
	t1_score, t2_score = field_control_score(game_dfc)
	filtered_game = game_dfc.query('game_state != "Game Effectively Over"').copy()
	t1_score_filtered, t2_score_filtered = field_control_score(filtered_game)
	return t1_score, t2_score, t1_score_filtered, t2_score_filtered


def field_control_score(game_df):
	### change to fc_score_compute()?
	### return both with and without garbage scores or have an optional argument 
	### for garbage time inclusion and call function twice where necessary?
	game_dfc = game_df.copy()
	last_idx = game_df.index[-1]
	
	game_df['fp_delta'] = 0
	game_df['time_delta'] = 0
	game_df['t1_area'] = 0
	game_df['t2_area'] = 0

	for idx, row in game_df.iterrows():
		if idx != last_idx:
			fp_0 = row.field_position
			fp_1 = game_df.loc[idx+1, 'field_position']
			t_0 = row.time_in_reg
			t_1 = game_df.loc[idx+1, 'time_in_reg']
			game_df.loc[idx+1, 'fp_delta'] = fp_1 - fp_0
			game_df.loc[idx+1, 'time_delta'] = t_1 - t_0
			game_df.loc[idx+1, 't1_area'] = fp_0*(t_1 - t_0) + 0.5*(t_1 - t_0)*(fp_1 - fp_0)
			game_df.loc[idx+1, 't2_area'] = 100*(t_1 - t_0) - game_df.loc[idx+1, 't1_area']
			
	total_fc = game_df.t1_area.sum() + game_df.t2_area.sum()
	t1_score = round(game_df.t1_area.sum()/total_fc, 3)
	t2_score = round(game_df.t2_area.sum()/total_fc, 3)
	return t1_score, t2_score


def calc_time_of_poss(game_df):
	"""
	Calculates the time of possession for each team in a game
	"""
	offteam_top = 0
	last_idx = game_df.index[-1]
	for idx, row in game_df.iterrows():
		if row.offteam == game_df.loc[0, 'offteam']:
			if idx == last_idx:
				offteam_top += 0
			else:
				offteam_top += (game_df.loc[idx+1, 'time_in_reg'] - row.time_in_reg)
	return offteam_top

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


def field_control_pipeline(df, game_num, plot=True):
	"""
	Pipeline taking in the season long data and a game number and performing 
	all of the relevant transformations and operations before plotting it. 
	"""
	dfc = df.copy()
	game_df = get_game(dfc, game_num)
	
	fc_score = field_control_score(game_df=game_df)
	game_df = game_df.sort_values(by = 'time_in_reg').reset_index(drop = True)

	if plot:
		plot_field_control(game_df, fc_score)
	return 


def get_fc_scores_season(df):
	"""
	Calculate the field control scores for every game in a season.
	Return a dictionary of games and information relating to them, such
	as teams, score differential, time of possession,
	and field control scores. 
	"""
	dfc = df.copy()
	all_games = dfc.gsis_game_id.unique().tolist()
	all_fc_scores = dict()
	for game_num in all_games:
		game_df = get_game(dfc, game_num)
		main_team = game_df.loc[0, 'offteam']
		other_team = game_df.loc[0, 'defteam']
		if main_team == game_df.offteam.iloc[-1]:
			main_score_diff = game_df.score_differential.iloc[-1]
			other_score_diff = -main_score_diff
		else:
			other_score_diff = game_df.score_differential.iloc[-1]
			main_score_diff = -other_score_diff
		#game_df = game_df.apply(flip_fp, axis = 1, args = (main_team, ))
		fc_score = field_control_score(game_df=game_df)
		main_top = calc_time_of_poss(game_df)
		main_avg_fp, other_avg_fp = get_mean_field_position(game_df)
		all_fc_scores[game_num] = [main_team, fc_score, main_score_diff, main_top,
								   main_avg_fp, other_team, 1-fc_score, other_score_diff,
								  3600 - main_top, other_avg_fp]
		
	fc_scores_df = pd.DataFrame.from_dict(all_fc_scores, orient = 'index').reset_index()
	fc_scores_df.columns = ['game_num', 'team_1', 'fc_team_1', 'team_1_score_diff',
							'team_1_top', 'team_1_avg_start_fp', 'team_2', 'fc_team_2',
							'team_2_score_diff', 'team_2_top', 'team_2_avg_start_fp']

	season_fc_df = pd.concat([fc_scores_df[['game_num', 'team_1', 'fc_team_1', 
											'team_1_score_diff', 'team_1_top',
										   'team_1_avg_start_fp']], 
		   fc_scores_df[['game_num', 'team_2', 'fc_team_2', 
						 'team_2_score_diff', 'team_2_top',
						'team_2_avg_start_fp']].\
				   rename(columns = {'team_2': 'team_1', 'fc_team_2':'fc_team_1',
									'team_2_score_diff': 'team_1_score_diff',
									'team_2_top': 'team_1_top', 
									'team_2_avg_start_fp':'team_1_avg_start_fp'})]).\
			sort_values(by = 'game_num').reset_index(drop = True)
	
	return season_fc_df


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


def fc_score_distplot(df, game_num = None, games = 'all'):
	"""
	This function takes in the season long data for each game, plots the distribution
	of field control scores, and then highlights the scores for the specified game, if 
	one is specified. 
	:games: 'all', 'wins', or 'losses'
	"""
	
	sns.set_style("whitegrid")
	df = df.copy()
	
	if games == 'wins':
		df = df.loc[df.team_1_score_diff >= 0]
	elif games == 'losses':
		df = df.loc[df.team_1_score_diff < 0]
	else:
		df = df        

	g = sns.distplot(df.fc_team_1, color = '#92c5de')
	g.set_xlim(0.25, 0.75)
	g.set_xlabel('Field Control Score')
	g.set_yticks([])
	if games == 'wins':
		g.set_title('Distribution of Field Control Scores - Wins')
	elif games == 'losses':
		g.set_title('Distribution of Field Control Scores - Losses')
	else:
		g.set_title('Distribution of Field Control Scores')
	
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


def n_garbage_time(df):
	garbage_time = 0
	last_index = df.index[-1]
	for idx, row in df.iterrows():              
		if row.game_state == 'Game Effectively Over':
			if idx == last_index:
				garbage_time += 0
			else:
				new_time = df.loc[idx+1, 'time_in_reg'] - df.loc[idx, 'time_in_reg']
				#print new_time
				garbage_time += new_time
				
	return garbage_time


def get_garbage_df(df):
	game_cols = ['gsis_game_id', 'play_index', 'time_in_reg', 'offteam', 'defteam', 
				 'field_position', 'score_differential', 'game_state']
	dfc = df.copy().loc[:, game_cols]
	unique_games = dfc.gsis_game_id.unique()
	garbage_dict = dict()
	
	for game in unique_games:
		dft = dfc.loc[dfc.gsis_game_id == game].sort_values(by = 'time_in_reg').reset_index(drop = True)
		team_1 = dft.loc[0, 'offteam']
		team_2 = dft.loc[0, 'defteam']
		garbage_time = n_garbage_time(dft)
		garbage_dict[game] = garbage_time
	
	garbage_df = pd.DataFrame.from_dict(garbage_dict, orient = 'index').reset_index(drop = False)
	garbage_df.columns = ['game_num', 'garbage_time']
	
	return garbage_df


def fc_game_data_json(game_df):
	"""
	Records and stores the relevant data for a single game.
	"""
	game_dict = dict()
	meta = dict()
	team1 = game_df.loc[0, 'offteam']
	team2 = game_df.loc[0, 'defteam']
	meta['date'] = game_df.loc[0, 'date']
	meta['team1'] = team1
	meta['team2'] = team2
	fc_game_score = fcf.field_control_score(game_df)
	meta['t1_fc'] = fc_game_score
	meta['t2_fc'] = 1 - fc_game_score
	
	drives_grouped = game_df.groupby(['drive', 'offteam'])
	drives_list = list()
	for drive in drives_grouped:
		drive_df = drive[1]
		drive_dict = dict()

		first_idx = drive_df.index[0]
		last_idx = drive_df.index[-1]
		drive_dict['end_event'] = 0 if drive_df.loc[last_idx, 'is_score'] is None else 1
		drive_dict['team'] = drive_df.loc[first_idx, 'offteam']

		flow = list()
		for row in drive_df.itertuples():
			# gotta keep track of the correct indices for the row when adding columns to the initial df
			play_dict = dict()
			play_dict['ts'] = row[4]
			play_dict['y'] = row[7]
			flow.append(play_dict)
		drive_dict['flow'] = flow
	
		drives_list.append(drive_dict)
	
	game_dict['meta'] = meta
	game_dict['drives'] = drives_list
	
	return game_dict


def fc_all_games_json(df):
	"""
	Loops through all games in the dataframe and calls fc_game_data_json
	on each game. Returns values in a dict with game_id as the key and
	the game data as the value
	"""
	dfc = df.copy() #.loc[:, game_cols]
	all_games = dfc.gsis_game_id.unique()
	all_games_metadata = list()
	
	for game_num in all_games:
		dft = fcf.get_game(dfc, game_num)
#         print(dft.head())
#         dft = dfc.query('gsis_game_id == @game_num').sort_values(by = 'time_in_reg').reset_index(drop = True)
#         all_games_metadata[game_num] = fc_game_data_json(dft)
		all_games_metadata.append(blah(dft))
	
	return all_games_metadata

def get_mean_field_position(game_df):
#     game_dft = game_df.copy()
	team_1 = game_df.loc[0, 'offteam']
	team_2 = game_df.loc[0, 'defteam']
	team_1_fp = game_df.query('offteam == @team_1 & drive_play == 1').field_position.mean()
	team_2_fp = 100 - game_df.query('offteam == @team_2 & drive_play == 1').field_position.mean()
	
	return team_1_fp, team_2_fp

def all_games_field_position(df):
	"""
	Loops through all games in the dataframe and calls get_mean_field_position
	on each game. Returns values in a dict with game_id as the key and
	the game data as the value
	"""
	game_cols = ['gsis_game_id', 'play_index', 'drive_play', 'time_in_reg',
				 'offteam', 'defteam', 'field_position',
				 'score_differential', 'game_state']
	dfc = df.copy().loc[:, game_cols]
	all_games = dfc.gsis_game_id.unique()
	all_games_fp = dict()
	
	for game_num in all_games:
#         print(game_num)
		dft = dfc.query('gsis_game_id == @game_num').sort_values('time_in_reg').reset_index(drop = True)
		all_games_fp[game_num] = get_mean_field_position(dft)
	
	fp_df = pd.DataFrame.from_dict(all_games_fp, orient='index').reset_index()
	fp_df.columns = ['game_num', 't1_avg_start_fp', 't2_avg_start_fp']
	return fp_df