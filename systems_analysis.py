import pandas as pd
import math
from matplotlib import pyplot as plt 
import numpy as np
import os
import mysql.connector
from datetime import date
from sklearn.linear_model import BayesianRidge, LinearRegression

from user_definition import * 

# Currently only for Python 2 - need to figure out mysql connection for Py3


def execute_ncaab_query(query, user, password, host, db):
	"""
	Connects to BV's NCAA schema using the specified query.
	Currently specified 
	"""
	try:
		conn = mysql.connector.connect(user = user,
			password = password,
			host = host,
			database = db)
		df = pd.read_sql_query(query, con = conn)
		conn.close()
		return df 
	except Exception as e:
		print "Can't connect to ncaab_voodoo"


def remove_zero_scores(df):
	"""
	Removes any games in the dataframe where one team is recorded as 
	scoring zero points. Likely indicator that team is not Division 1
	"""
	dfc = df.copy()
	dfc = dfc[(dfc.t1_points > 0) & (dfc.t2_points > 0)]
	return dfc


def column_swap(df, is_sbr = False):
	"""
	Swaps columns to make 't1' team to match the 'team' column.
	"""
	dfc = df.copy()
	# name must be last since comparison depends on it
	t1_cols = list(reversed([col for col in dfc.columns if col.startswith('t1_')]))
	t2_cols = list(reversed([col for col in dfc.columns if col.startswith('t2_')]))

	if is_sbr:
		for i in range(len(t1_cols)):
			smp_array = np.where(dfc.home_name != dfc.t1_name, [dfc[t2_cols[i]], dfc[t1_cols[i]]],
				[dfc[t1_cols[i]], dfc[t2_cols[i]]])
			dfc.loc[:, t1_cols[i]] = smp_array[0]
			dfc.loc[:, t2_cols[i]] = smp_array[1]
	else: 
		for i in range(len(t1_cols)):
			smp_array = np.where(dfc.ncaa_name != dfc.t1_name, [dfc[t2_cols[i]], dfc[t1_cols[i]]],
				[dfc[t1_cols[i]], dfc[t2_cols[i]]])
			dfc.loc[:, t1_cols[i]] = smp_array[0]
			dfc.loc[:, t2_cols[i]] = smp_array[1]

	return dfc


def kenpom_query(df, date, game):
	"""
	Returns home/away, score, and KenPom efficiency information for games
	"""

	dfq = df.query('date == "' + str(date.date()) + '" and game_id == "' + str(game) + '"')
	game_dict = {}
	for index, row in dfq.iterrows():
		if row.loc['t1_side'] == 'home':
			game_dict['game_id'] = row.loc['game_id']
			game_dict['game_date'] = row.loc['game_date']
			game_dict['home_team'] = row.loc['ncaa_name']
			game_dict['kenpom_off_home'] = row.loc['offensive_efficiency']
			game_dict['kenpom_def_home'] = row.loc['defensive_efficiency']
			game_dict['home_score'] = row.loc['t1_points']
			game_dict['home_conf'] = row.loc['t1_conf']
		else:
			game_dict['game_id'] = row.loc['game_id']
			game_dict['game_date'] = row.loc['game_date']
			game_dict['away_team'] = row.loc['ncaa_name']
			game_dict['kenpom_off_away'] = row.loc['offensive_efficiency']
			game_dict['kenpom_def_away'] = row.loc['defensive_efficiency']
			game_dict['away_score'] = row.loc['t1_points']
			game_dict['away_conf'] = row.loc['t1_conf']
	return game_dict


def moore_query(df, date, game):
	"""
	Returns home/away, score, and Moore rating information for games
	"""

	dfq = df.query('date == "' + str(date.date()) + '" and game_id == "' + str(game) + '"')
	game_dict = {}
	for index, row in dfq.iterrows():
		if row.loc['t1_side'] == 'home':
			game_dict['game_id'] = row.loc['game_id']
			game_dict['game_date'] = row.loc['game_date']
			game_dict['home_team'] = row.loc['ncaa_name']
			game_dict['moore_home'] = row.loc['pr']
			game_dict['home_score'] = row.loc['t1_points']
			game_dict['home_conf'] = row.loc['t1_conf']
		else:
			game_dict['game_id'] = row.loc['game_id']
			game_dict['game_date'] = row.loc['game_date']
			game_dict['away_team'] = row.loc['ncaa_name']
			game_dict['moore_away'] = row.loc['pr']
			game_dict['away_score'] = row.loc['t1_points']
			game_dict['away_conf'] = row.loc['t1_conf']
	return game_dict


def sagarin_query(df, date, game):
	"""
	Returns home/away, score, and Sagarin rating information for games
	"""
	
	dfq = df.query('date == "' + str(date.date()) + '" and game_id == "' + str(game) + '"')
	game_dict = {}
	for index, row in dfq.iterrows():
		if row.loc['t1_side'] == 'home':
			game_dict['game_id'] = row.loc['game_id']
			game_dict['game_date'] = row.loc['game_date']
			game_dict['home_team'] = row.loc['ncaa_name']
			game_dict['sagarin_home'] = row.loc['rating']
			game_dict['home_score'] = row.loc['t1_points']
			game_dict['home_conf'] = row.loc['t1_conf']
		else:
			game_dict['game_id'] = row.loc['game_id']
			game_dict['game_date'] = row.loc['game_date']
			game_dict['away_team'] = row.loc['ncaa_name']
			game_dict['sagarin_away'] = row.loc['rating']
			game_dict['away_score'] = row.loc['t1_points']
			game_dict['away_conf'] = row.loc['t1_conf']
	return game_dict


def sbr_query(df, date, game):
	"""
	Returns home/away, score, and SBR rating information for games
	"""
	
	dfq = df.query('date == "' + str(date.date()) + '" and game_id == "' + str(game) + '"')
	game_dict = {}
	for index, row in dfq.iterrows():
		if row.loc['t1_side'] == 'home':
			game_dict['game_id'] = row.loc['game_id']
			game_dict['game_date'] = row.loc['game_date']
			game_dict['home_team'] = row.loc['home_name']
			game_dict['home_spread'] = row.loc['home_spread']
			game_dict['home_money_line'] = row.loc['home_money_line']
			game_dict['home_score'] = row.loc['t1_points']
			game_dict['home_conf'] = row.loc['t1_conf']
		#else:
			game_dict['game_id'] = row.loc['game_id']
			game_dict['game_date'] = row.loc['game_date']
			game_dict['away_team'] = row.loc['away_name']
			game_dict['away_spread'] = row.loc['away_spread']
			game_dict['away_money_line'] = row.loc['away_money_line']
			game_dict['away_score'] = row.loc['t2_points']
			game_dict['away_conf'] = row.loc['t2_conf']
	return game_dict


def create_games_df(df, query_type):
	"""
	Create dataframe with game information and corresponding Moore ratings 
	for home and away teams involved in the game
	"""
	game_dt_range = pd.date_range(df.game_date.min(), df.game_date.max())
	game_list = []
	dft = df.copy()
	for date in game_dt_range:
		dft = df[pd.to_datetime(df.date) == date] # do i need this? i think makes it faster but not sure
		for game in dft.game_id.unique():
			if query_type == 'kenpom':
				res = kenpom_query(dft, date, game)
			elif query_type == 'moore':
				res = moore_query(dft, date, game)
			elif query_type == 'sagarin':
				res = sagarin_query(dft, date, game)
			game_list.append(res)
			
	if query_type == 'kenpom':
		cols = ['game_id', 'game_date', 'home_team', 'home_conf', 
				'kenpom_off_home', 'kenpom_def_home', 'home_score',
				'away_team', 'away_conf', 'kenpom_off_away', 'kenpom_def_away',
				'away_score']
	elif query_type == 'moore':
		cols = ['game_id', 'game_date', 'home_team', 'home_conf',
				'moore_home', 'home_score', 'away_team',
			   	'away_conf', 'moore_away', 'away_score']
	elif query_type == 'sagarin':
		cols = ['game_id', 'game_date', 'home_team', 'home_conf',
				'sagarin_home', 'home_score', 'away_team',
			   	'away_conf', 'sagarin_away', 'away_score']
	elif query_type == 'sbr':
		cols = ['game_id', 'game_date', 'home_team', 'home_conf', 'home_score', 
				'home_spread', 'home_money_line', 
			   	'away_team', 'away_conf', 'away_score', 'away_spread', 'away_money_line']
	else:
		print "Not a system"
	
	df_accuracy = pd.DataFrame(game_list, columns = cols)
	return df_accuracy


def create_system_accuracy_df(df, system):
	"""
	Adds accuracy columns to dataframe for more analysis
	"""
	if system == 'sbr':
		dfc = df[np.isfinite(df['home_spread']) | np.isfinite(df['away_spread'])].copy().reset_index(drop = True)
	else:
		dfc = df.copy().dropna(axis = 0).reset_index(drop = True)

	dfc['score_diff'] = dfc.apply(lambda x: x.home_score - x.away_score, axis = 1)

	if system == 'kenpom':
		dfc['kenpom_off_diff'] = dfc.apply(lambda x: x.kenpom_off_home + 3.117 - x.kenpom_off_away, axis = 1)
		dfc['kenpom_def_diff'] = dfc.apply(lambda x: x.kenpom_def_home - x.kenpom_def_away, axis = 1)
		dfc['kenpom_diff'] = dfc.apply(lambda x: x.kenpom_off_diff - x.kenpom_def_diff, axis = 1).astype('float16')
		dfc['kenpom_correct'] = dfc.apply(lambda x: np.sign(x.kenpom_diff) == np.sign(x.score_diff),
			axis = 1)
	elif system == 'moore':
		dfc['moore_diff'] = dfc.apply(lambda x: (x.moore_home + 3.25) - x.moore_away, axis = 1)
		dfc['moore_correct'] = dfc.apply(lambda x: np.sign(x.score_diff) == np.sign(x.moore_diff), 
			axis = 1)
	elif system == 'sagarin':
		dfc['sagarin_diff'] = dfc.apply(lambda x: (x.sagarin_home + 3.17) - x.sagarin_away, axis = 1)
		dfc['sagarin_correct'] = dfc.apply(lambda x: np.sign(x.score_diff) == np.sign(x.sagarin_diff),
			axis = 1)
	elif system == 'sbr':
		dfc['sbr_correct'] = dfc.apply(lambda x: np.sign(x.score_diff) != np.sign(x.home_spread),
												   axis = 1)
	else:
		print "Not a system"

	return dfc


def create_system_graphs(df, system):
	"""
	Creates a graph of the cumulative accuracy for the system, the accuracy by day, 
	and the count of games for that day. 
	"""
	cum_correct = df.groupby('game_date')[system + '_correct'].sum().cumsum()
	cum_total = df.groupby('game_date')['game_id'].count().cumsum()
	cum_accuracy = cum_correct/cum_total

	f, (ax1, ax2, ax3) = plt.subplots(3, sharex = True, figsize = (15, 6))
	x = pd.to_datetime(df.groupby('game_date')[system + '_correct'].mean().index)
	y = df.groupby('game_date')[system + '_correct'].mean().values
	err = df.groupby('game_date')[system + "_correct"].count().apply(lambda n: 1/n**0.5).values

	ax1.plot(x, cum_accuracy.values)
	ax1.set_ylabel('Cumulative Acc')
	ax2.errorbar(x, y, yerr = err, ecolor = 'xkcd:sky blue')
	ax2.set_ylabel('Acc by Day')
	ax3.bar(x, sbr_accuracy.groupby('game_date')['game_id'].count().values)
	ax3.set_ylabel('# Games by Day')
	f.subplots_adjust(hspace=0.1)
	plt.suptitle(system.uppper() + ' Accuracy Information')

	return f 


def get_in_out_by_conf(df, system):
	"""
	Splits dataframe into games by in and out of conference and returns
	a dataframe with each conference's in/out splits
	"""
	conferences = df.home_conf.unique().tolist()
	all_confs = list()
	
	for conf in conferences:
		#conf_list = list()
		games = df[(df.home_conf == conf) | (df.away_conf == conf)]
		games_in_conf = games[games.home_conf == games.away_conf]
		games_out_conf = games[games.home_conf != games.away_conf]

		in_conf_accuracy = games_in_conf[system + '_correct'].mean()
		in_conf_count = games_in_conf[system + '_correct'].count()

		out_conf_accuracy = games_out_conf[system + '_correct'].mean()
		out_conf_count = games_out_conf[system + '_correct'].count()

		conf_list = [conf, in_conf_accuracy, in_conf_count, out_conf_accuracy, out_conf_count]
		all_confs.append(conf_list)
	conf_df = pd.DataFrame(all_confs, columns = ['Conf', 'In_Conf_Acc', 'In_Conf_Count', 
												 'Out_Conf_Acc', 'Out_Conf_Count'])
	return conf_df


def get_in_out_splits(df, system):
	"""
	Calculates and returns a system's overall home and away splits
	"""
	dfc = df.copy()

	in_accuracy = dfc[dfc.home_conf == dfc.away_conf][system + '_correct'].mean()
	out_accuracy = dfc[dfc.home_conf != dfc.away_conf][system + '_correct'].mean()

	return in_accuracy, out_accuracy


def get_home_away_splits(df, system):
	"""
	Calculates and returns a system's overall home and away splits
	"""
	dfc = df.copy()
	if system != 'sbr':
		home_acc = dfc[dfc[system + '_differential'] > 0][system + '_correct'].mean()
		away_acc = dfc[dfc[system + '_differential'] < 0][system + '_correct'].mean()
	elif system == 'sbr':
		home_acc = dfc[dfc.home_spread > 0].sbr_correct.mean()
		away_acc = dfc[dfc.home_spread < 0].sbr_correct.mean()
	else:
		print "Not a system"

	return home_acc, away_acc


def implied_probability(moneyline):
	"""
	Calculates and returns the implied win probabilities based on the moneyline
	"""
	if (moneyline > 0):
		return 100 / (100 + moneyline)
	else:
		return np.absolute(moneyline) / (100 + np.absolute(moneyline))



######################################
### Start Bayesian Model Functions ###
######################################


def learn_global(df):
    """
    Find all of the coefficients for the global model across all dates
    Need to be passed df with columns already filtered down to the linreg cols?
    """
    # save results to a dictionary
    global_models = dict()
    model_cols = ['sagarin_home', 'sagarin_away', 'kenpom_off_home', 'kenpom_def_home',
              'kenpom_off_away', 'kenpom_def_away', 'moore_home', 'moore_away', 
              'score_diff']
    
    for day in pd.date_range(df.game_date.min(), df.game_date.max()): # start at min + 1?
        # filter by day and relevant columns for our regression
        dfc = df.loc[(df.game_date < str(day)), model_cols].copy().dropna(axis = 0)
        
        # run linear regression on filtered df
        X, y = dfc.drop('score_diff', axis = 1), dfc.score_diff
        lr = LinearRegression(normalize=True)
        lr_global = lr.fit(X, y)
        
        global_models[str(day.date())] = lr_global
        
    return global_models


def learn_conf(df, global_models, conf):
    """
    Go through individual conferences and fit models
    """
    # Filter by conference
    dfc = df.loc[(df.home_conf == conf) & (df.away_conf == conf)].copy().dropna(axis = 0)
    dfc = dfc.drop(['home_conf', 'away_conf'], axis = 1)
    bayes_conf = dict()
    model_cols = ['sagarin_home', 'sagarin_away', 'kenpom_off_home', 
    			'kenpom_def_home', 'kenpom_off_away', 'kenpom_def_away',
    			'moore_home', 'moore_away', 'score_diff']
        
    for day in pd.date_range(df.game_date.min(), df.game_date.max()): 
        # start at min+1? I think that would avoid the cold-start issues encountered on days with no games
        day = str(day.date())
        df_conf = dfc.loc[(dfc.game_date == day), model_cols] 
        
        # other thought: have prev day outside of if, put isinstance statements in initial if?
        
        if (len(df_conf) == 0):
            try:
                prev_day = str((pd.to_datetime(day) - pd.Timedelta(days = 1)).date())
                if isinstance(bayes_conf[prev_day], LinearRegression):
                    # use global lr?
                    bayes_conf[day] = global_models[day]
                elif isinstance(bayes_conf[prev_day], BayesianRidge):
                    bayes_conf[day] = bayes_conf[prev_day]
            except:
                bayes_conf[day] = global_models[day] # should only happen on first day
        else:
            bayes = BayesianRidge(normalize=True)
            bayes.coef_ = global_models[day].coef_
            
            X = df_conf.drop('score_diff', axis = 1)
            y = df_conf.score_diff
            
            bayes_mod = bayes.fit(X, y)
            bayes_conf[day] = bayes_mod            
                       
    return bayes_conf


def learn_non_conf(df, global_models):
    """
    Learn a model for all non-conference games
    Could this be rolled into the learn_conf function easily?
    """
    # Filter by conference - should this happen inside or outside the for loop?
    dfc = df.loc[df.home_conf != df.away_conf].copy().dropna(axis = 0)
    dfc = dfc.drop(['home_conf', 'away_conf'], axis = 1)
    bayes_non_conf = dict()
    model_cols = ['sagarin_home', 'sagarin_away', 'kenpom_off_home', 
    			'kenpom_def_home', 'kenpom_off_away', 'kenpom_def_away',
    			'moore_home', 'moore_away', 'score_diff']
        
    for day in pd.date_range(df.game_date.min(), df.game_date.max()): 
        # start at min+1? I think that would avoid the cold-start issues encountered on days with no games
        day = str(day.date())
        df_conf = dfc.loc[(dfc.game_date == day), model_cols] # rename linreg_cols to model_cols?
        
        # other thought: have prev day outside of if, put isinstance statements in initial if?
        
        if (len(df_conf) == 0):
            try:
                prev_day = str((pd.to_datetime(day) - pd.Timedelta(days = 1)).date())
                if isinstance(bayes_non_conf[prev_day], LinearRegression):
                    # use global lr?
                    bayes_non_conf[day] = global_models[day]
                elif isinstance(bayes_non_conf[prev_day], BayesianRidge):
                    bayes_non_conf[day] = bayes_non_conf[prev_day]
            except:
                bayes_non_conf[day] = global_models[day] # should only happen on first day
        else:
            bayes = BayesianRidge(normalize=True)
            bayes.coef_ = global_models[day].coef_
            
            X = df_conf.drop('score_diff', axis = 1)
            y = df_conf.score_diff
            
            bayes_mod = bayes.fit(X, y)
            bayes_non_conf[day] = bayes_mod            
                       
    return bayes_non_conf


def learn_all_confs(df, global_models):
    """
    Learn models for each conf (including independents and not non-conf) 
    """
    all_confs = df.home_conf.unique().tolist()
    conference_models = dict()
    conference_models['all_confs'] = global_models
    conference_models['non_conf'] = learn_non_conf(df, global_models)
    
    for conf in all_confs:
        conf_mod_dict = learn_conf(df, global_models, conf)
        conference_models[conf] = conf_mod_dict
        
    return conference_models
        

def predict_all_games(df, all_models):
    """
    Identify necessary model for each game and use it to predict the game
    Format/shape of output? Append column to df?
    """
    dfc = df.copy()
    model_cols = ['sagarin_home', 'sagarin_away', 'kenpom_off_home',
    			'kenpom_def_home', 'kenpom_off_away', 'kenpom_def_away',
    			'moore_home', 'moore_away', 'score_diff']

    dfc['preds'] = 0
    preds_list = list()
    
    for idx, row in dfc.iterrows():
        if row.home_conf == row.away_conf:
            conf = row.home_conf
        else:
            conf = 'non_conf'
            
        if row.game_date == dfc.game_date.min():
            # current option
            # could also just move the rowc/pred/preds_list lines into the else and pass 
            # if first day of df
            model = all_models['all_confs'][dfc.game_date.max()]
        else:
            fit_date = str((pd.to_datetime(row.game_date) - pd.Timedelta(days = 1)).date())
            pred_date = row.game_date
            model = all_models[conf][fit_date]
        
        rowc = row[model_cols].copy().drop('score_diff')
        pred = model.predict(rowc.values.reshape(1, -1))
        # for error checking
        # if np.isnan(model.coef_).any():
        #     print conf, fit_date, model.coef_
        #     print rowc
        #     print pred
        preds_list.extend(pred)
    
    dfc.preds = pd.Series(preds_list)
    return dfc


######################################
### Start SBR Comparison Functions ###
######################################

def model_cover_pick(row):
    """
    Check if the model predicts the home team to cover or not
    """
    if row.preds > row.home_spread:
        return 'home covers'
    else:
        return 'away covers'
    

def model_pick_correct(row):
    """
    Check if the model's pick was in line with actual results
    """
    if row.score_diff > row.home_spread:
        result = 'home covers'
    else:
        result = 'away covers'
    
    if result == row.model_pick:
        return True
    else:
        return False
    

def pick_return(row):
    if row.result_pick == True:
        return 1.0
    else:
        return -1.1


def get_conference(row):
    if row.home_conf == row.away_conf:
        return row.home_conf
    else:
        return 'non_conf'


